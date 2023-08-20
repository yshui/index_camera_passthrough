//! The overlay in VR space can be seen as a "portal" to the real world. By projecting it to the
//! Index camera's clipping space, using the camera's projection matrix, we can decide which
//! portion of the camera's view can be seen through this portal.
//!
//! Overlay vertex * Overlay Model * HMD View * Camera Project -> Texture coordinates used to
//! sample the camera's view.
//!
//! Overlay vertex: calculate based on Overlay width we set
//! Overlay Model: the overlay transform matrix we set
//! HMD View: inverse of HMD pose
//! Camera Project: estimated from camera calibration.
use anyhow::{anyhow, Result};
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferError, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferUsage::OneTimeSubmit, RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::{DescriptorSetAlloc, DescriptorSetAllocator},
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    image::{
        view::{ImageView, ImageViewCreateInfo},
        AttachmentImage, ImageAccess,
    },
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocatePreference, MemoryUsage, StandardMemoryAllocator,
    },
    pipeline::{
        graphics::{
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::Vertex as VertexTrait,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    sampler::{Filter, Sampler, SamplerCreateInfo},
    sync::GpuFuture,
    Handle, VulkanObject,
};
mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/projection.vert",
        custom_derives: [Copy, Clone, Debug, Default],
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/projection.frag",
        custom_derives: [Copy, Clone, Debug],
    }
}

#[derive(PartialEq, Debug)]
pub struct ProjectionParameters {
    pub ipd: f32,
    pub overlay_width: f32,
    /// MVP matrices for the left and right eye, respectively.
    pub mvps: [Matrix4<f32>; 2],
    pub camera_calib: Option<crate::vrapi::StereoCamera>,
    pub mode: ProjectionMode,
}

struct Uniforms {
    transforms: [Subbuffer<vs::Transform>; 2],
}

impl std::fmt::Debug for Uniforms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Uniforms")
            .field("transforms", &())
            .finish_non_exhaustive()
    }
}

pub struct Projection<T> {
    source: Arc<AttachmentImage>,
    pipeline: Arc<GraphicsPipeline>,
    render_pass: Arc<RenderPass>,
    // [0: left, 1: right]
    uniforms: Uniforms,
    saved_parameters: ProjectionParameters,
    desc_sets: [Arc<PersistentDescriptorSet<T>>; 2],
}
impl<T> std::fmt::Debug for Projection<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Projection")
            .field("source", &self.source.inner().image.handle().as_raw())
            .field("pipeline", &self.pipeline.handle().as_raw())
            .field("render_pass", &self.render_pass.handle().as_raw())
            .field("uniforms", &self.uniforms)
            .field("saved_parameters", &self.saved_parameters)
            .finish_non_exhaustive()
    }
}
use crate::config::ProjectionMode;
#[derive(VertexTrait, Default, Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Vertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R32G32B32_SFLOAT)]
    in_tex_coord: [f32; 3],
}

#[allow(dead_code)]
fn format_matrix<
    A: Scalar + ToString,
    B: nalgebra::Dim,
    C: nalgebra::Dim,
    D: RawStorage<A, B, C>,
>(
    m: &nalgebra::Matrix<A, B, C, D>,
) -> String {
    use itertools::Itertools;
    format!(
        "numpy.matrix([{}])",
        m.row_iter()
            .map(|r| {
                let it = r.into_iter();
                format!("[{}]", it.map(|v| v.to_string()).join(","))
            })
            .join(",")
    )
}

use nalgebra::{matrix, Matrix4, RawStorage, Scalar};
impl<DSA: DescriptorSetAlloc + 'static> Projection<DSA> {
    /// Calculate the _physical_ camera's MVP, for each eye.
    /// camera_calib = camera calibration data.
    /// fov_left/right = adjusted fovs, in ratio (not in pixels)
    /// frame_time = how long after the first frame is the current frame taken
    /// time_origin = instant when the first frame is taken
    pub(crate) fn calculate_mvp(
        &self,
        mode: ProjectionMode,
        overlay_transform: &Matrix4<f64>,
        camera_calib: &Option<crate::vrapi::StereoCamera>,
        fov: &[[f64; 2]; 2],
        eye_to_head: &[Matrix4<f64>; 2],
        hmd_transform: &Matrix4<f64>,
    ) -> (Matrix4<f32>, Matrix4<f32>) {
        let left_extrinsics_position = camera_calib
            .map(|c| c.left.extrinsics.position)
            .unwrap_or_default();
        // Camera space to HMD space transform, based on physical measurements
        let left_cam: Matrix4<_> = matrix![
            1.0, 0.0, 0.0, -left_extrinsics_position[0];
            0.0, 1.0, 0.0, -left_extrinsics_position[1];
            0.0, 0.0, 1.0, -left_extrinsics_position[2];
            0.0, 0.0, 0.0, 1.0;
        ];
        let right_extrinsics_position = camera_calib
            .map(|c| c.right.extrinsics.position)
            .unwrap_or_default();
        let right_cam: Matrix4<_> = matrix![
            1.0, 0.0, 0.0, -right_extrinsics_position[0];
            0.0, 1.0, 0.0, -right_extrinsics_position[1];
            0.0, 0.0, 1.0, -right_extrinsics_position[2];
            0.0, 0.0, 0.0, 1.0;
        ];

        let (left_eye, right_eye) = match mode {
            ProjectionMode::FromEye => (
                hmd_transform * eye_to_head[0],
                hmd_transform * eye_to_head[1],
            ),
            ProjectionMode::FromCamera => (hmd_transform * left_cam, hmd_transform * right_cam),
        };
        let left_view = left_eye
            .try_inverse()
            .expect("HMD transform not invertable?");
        let right_view = right_eye
            .try_inverse()
            .expect("HMD transform not invertable?");

        // X gets fov / 2.0 because the source texture is a side-by-side stereo texture
        // X translation element is used to map them to left/right side of the texture,
        // respectively.
        //
        let camera_projection_left = matrix![
            fov[0][0] / 2.0, 0.0, 0.0, 0.0;
            0.0, fov[0][1], 0.0, 0.0;
            0.0, 0.0, -1.0, 0.0;
            0.0, 0.0, 0.0, 1.0;
        ];
        let camera_projection_right = matrix![
            fov[1][0] / 2.0, 0.0, 0.0, 0.0;
            0.0, fov[1][1] , 0.0, 0.0;
            0.0, 0.0, -1.0, 0.0;
            0.0, 0.0, 0.0, 1.0;
        ];
        (
            (camera_projection_left * left_view * overlay_transform).cast(),
            (camera_projection_right * right_view * overlay_transform).cast(),
        )
    }
    pub fn set_params(&mut self, params: &ProjectionParameters) -> Result<()> {
        if &self.saved_parameters == params {
            return Ok(());
        }
        let ProjectionParameters {
            mode,
            overlay_width,
            ipd,
            mvps,
            camera_calib,
        } = params;
        assert_eq!(camera_calib, &self.saved_parameters.camera_calib);
        let left_extrinsics_position = camera_calib
            .map(|c| c.left.extrinsics.position)
            .unwrap_or_default();
        let mut transforms_write = self.uniforms.transforms.each_ref().try_map(|u| u.write())?;
        if mode != &self.saved_parameters.mode {
            let eye_offset_left = if *mode == ProjectionMode::FromEye {
                [
                    -left_extrinsics_position[0] as f32 - ipd / 2.0,
                    left_extrinsics_position[1] as f32,
                ]
            } else {
                [0.0, 0.0]
            };
            let eye_offset_right = [-eye_offset_left[0], eye_offset_left[1]];
            transforms_write[0].eyeOffset = eye_offset_left;
            transforms_write[1].eyeOffset = eye_offset_right;
            self.saved_parameters.mode = *mode;
        }

        for ((mvp, saved_mvp), write) in mvps
            .iter()
            .zip(self.saved_parameters.mvps.iter_mut())
            .zip(transforms_write.iter_mut())
        {
            if mvp != saved_mvp {
                *saved_mvp = *mvp;
                write.mvp = *mvp.as_ref();
            }
        }

        if overlay_width != &self.saved_parameters.overlay_width {
            self.saved_parameters.overlay_width = *overlay_width;
            transforms_write[0].overlayWidth = (*overlay_width).into();
            transforms_write[1].overlayWidth = (*overlay_width).into();
        }
        Ok(())
    }
    fn make_uniform_buffer<T: BufferContents>(
        allocator: &impl vulkano::memory::allocator::MemoryAllocator,
        uniform: T,
    ) -> Result<Subbuffer<T>, BufferError> {
        log::debug!("uniform buffer size {}", std::mem::size_of::<T>());
        Buffer::from_data(
            allocator,
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                allocate_preference: MemoryAllocatePreference::Unknown,
                ..Default::default()
            },
            uniform,
        )
    }
    pub fn new<DSA2: DescriptorSetAllocator<Alloc = DSA>>(
        device: Arc<Device>,
        allocator: &impl vulkano::memory::allocator::MemoryAllocator,
        descriptor_set_allocator: &DSA2,
        source: &Arc<AttachmentImage>,
        camera_calib: &Option<crate::vrapi::StereoCamera>,
    ) -> Result<Self> {
        let [w, h] = source.dimensions().width_height();
        if w != h * 2 {
            return Err(anyhow!("Input not square"));
        }
        let vs = vs::load(device.clone())?;
        let fs = fs::load(device.clone())?;
        let render_pass = vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Load,
                    store: Store,
                    format: vulkano::format::Format::R8G8B8A8_UNORM,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )
        .unwrap();
        let tex_offsets = [
            fs::Info {
                texOffset: [0.0, 0.0],
            },
            fs::Info {
                texOffset: [0.5, 0.0],
            },
        ]
        .try_map(|u| Self::make_uniform_buffer(allocator, u))?;
        let transforms =
            [vs::Transform::default(); 2].try_map(|u| Self::make_uniform_buffer(allocator, u))?;
        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(Vertex::per_vertex())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .input_assembly_state(
                InputAssemblyState::new().topology(PrimitiveTopology::TriangleStrip),
            )
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())?;
        let init_params = ProjectionParameters {
            mode: ProjectionMode::FromCamera, // This means `eyeOffset` should be zero, which would
            // be what is returned by `bytemuck::Zeroable`
            ipd: f32::NAN,
            overlay_width: f32::NAN,
            camera_calib: *camera_calib,
            mvps: [Matrix4::identity(), Matrix4::identity()],
        };
        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let sampler = Sampler::new(
            device,
            SamplerCreateInfo {
                min_filter: Filter::Linear,
                mag_filter: Filter::Linear,
                ..Default::default()
            },
        )?;
        let desc_sets = [0, 1].try_map(|i| {
            Ok::<_, anyhow::Error>(PersistentDescriptorSet::new(
                descriptor_set_allocator,
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, transforms[i].clone()),
                    WriteDescriptorSet::image_view_sampler(
                        1,
                        ImageView::new(source.clone(), ImageViewCreateInfo::from_image(&source))?,
                        sampler.clone(),
                    ),
                    WriteDescriptorSet::buffer(2, tex_offsets[i].clone()),
                ],
            )?)
        })?;
        Ok(Self {
            saved_parameters: init_params,
            uniforms: Uniforms { transforms },
            desc_sets,
            render_pass,
            pipeline,
            source: source.clone(),
        })
    }
    pub fn project(
        &mut self,
        allocator: &StandardMemoryAllocator,
        cmdbuf_allocator: &StandardCommandBufferAllocator,
        after: impl GpuFuture,
        queue: &Arc<Queue>,
        output: Arc<dyn ImageAccess>,
    ) -> Result<impl GpuFuture> {
        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            vulkano::render_pass::FramebufferCreateInfo {
                attachments: vec![ImageView::new(
                    output.clone(),
                    ImageViewCreateInfo::from_image(&output),
                )?],
                ..Default::default()
            },
        )?;
        let ProjectionParameters { overlay_width, .. } = &self.saved_parameters;
        let [w, h] = self.source.dimensions().width_height();
        let mut cmdbuf = AutoCommandBufferBuilder::primary(
            cmdbuf_allocator,
            queue.queue_family_index(),
            OneTimeSubmit,
        )?;
        //cmdbuf.copy_image(CopyImageInfo::images(self.source.clone(), output.clone()))?;

        // Y is flipped from the vertex Y because texture coordinate is top-down
        let vertex_buffer = Buffer::from_iter::<Vertex, _>(
            allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                allocate_preference: MemoryAllocatePreference::Unknown,
                ..Default::default()
            },
            [
                Vertex {
                    position: [-1.0, -1.0],
                    in_tex_coord: [-overlay_width / 2.0, overlay_width / 2.0, 0.0],
                },
                Vertex {
                    position: [-1.0, 1.0],
                    in_tex_coord: [-overlay_width / 2.0, -overlay_width / 2.0, 0.0],
                },
                Vertex {
                    position: [1.0, -1.0],
                    in_tex_coord: [overlay_width / 2.0, overlay_width / 2.0, 0.0],
                },
                Vertex {
                    position: [1.0, 1.0],
                    in_tex_coord: [overlay_width / 2.0, -overlay_width / 2.0, 0.0],
                },
            ]
            .iter()
            .cloned(),
        )
        .unwrap();
        // Left

        let mut render_pass_begin_info = RenderPassBeginInfo::framebuffer(framebuffer.clone());
        render_pass_begin_info.clear_values = vec![None];
        cmdbuf
            .begin_render_pass(render_pass_begin_info, SubpassContents::Inline)?
            .set_viewport(
                0,
                [Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [(w / 2) as f32, h as f32],
                    depth_range: 0.0..1.0,
                }],
            )
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                self.desc_sets[0].clone(),
            )
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .draw(vertex_buffer.len() as u32, 1, 0, 0)?
            .end_render_pass()?;

        // Right
        let mut render_pass_begin_info = RenderPassBeginInfo::framebuffer(framebuffer);
        render_pass_begin_info.clear_values = vec![None];
        cmdbuf
            .begin_render_pass(render_pass_begin_info, SubpassContents::Inline)?
            .set_viewport(
                0,
                [Viewport {
                    origin: [(w / 2) as f32, 0.0],
                    dimensions: [(w / 2) as f32, h as f32],
                    depth_range: 0.0..1.0,
                }],
            )
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                self.desc_sets[1].clone(),
            )
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .draw(vertex_buffer.len() as u32, 1, 0, 0)?
            .end_render_pass()?;
        Ok(after.then_execute(queue.clone(), cmdbuf.build()?)?)
    }
}
