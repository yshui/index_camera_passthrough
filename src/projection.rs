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
    buffer::{BufferContents, BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferUsage::OneTimeSubmit, CopyImageInfo, RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    image::{
        view::{ImageView, ImageViewCreateInfo},
        AttachmentImage, ImageAccess,
    },
    memory::allocator::{AllocationCreationError, FastMemoryAllocator, StandardMemoryAllocator},
    pipeline::{graphics::viewport::Viewport, GraphicsPipeline, Pipeline, PipelineBindPoint},
    render_pass::{Framebuffer, RenderPass, Subpass},
    sampler::{Filter, Sampler, SamplerCreateInfo},
    sync::GpuFuture,
};
mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/projection.vert",
        types_meta: {
            #[derive(::bytemuck::Zeroable, ::bytemuck::Pod, Copy, Clone, Debug)]
        },
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/projection.frag",
        types_meta: {
            #[derive(::bytemuck::Zeroable, ::bytemuck::Pod, Copy, Clone, Debug)]
        },
    }
}

#[derive(PartialEq)]
pub struct ProjectionParameters {
    pub ipd: f32,
    pub overlay_width: f32,
    /// MVP matrices for the left and right eye, respectively.
    pub mvps: [Matrix4<f32>; 2],
    pub camera_calib: crate::steam::StereoCamera,
    pub mode: ProjectionMode,
}

struct UniformData {
    tex_offsets: [fs::ty::Info; 2],
    transforms: [vs::ty::Transform; 2],
}

struct Uniforms {
    tex_offsets: [Arc<CpuAccessibleBuffer<fs::ty::Info>>; 2],
    transforms: [Arc<CpuAccessibleBuffer<vs::ty::Transform>>; 2],
}

pub struct Projection {
    device: Arc<Device>,
    source: Arc<AttachmentImage>,
    pipeline: Arc<GraphicsPipeline>,
    render_pass: Arc<RenderPass>,
    // [0: left, 1: right]
    uniforms: Uniforms,
    saved_parameters: ProjectionParameters,
    desc_sets: [Arc<PersistentDescriptorSet>; 2],
}
use crate::config::ProjectionMode;
#[derive(Default, Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Vertex {
    position: [f32; 2],
    in_tex_coord: [f32; 3],
}
vulkano::impl_vertex!(Vertex, position, in_tex_coord);

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
                let it: MatrixIter<_, _, _, _> = r.into_iter();
                format!("[{}]", it.map(|v| v.to_string()).join(","))
            })
            .join(",")
    )
}

use nalgebra::{iter::MatrixIter, matrix, Matrix4, RawStorage, Scalar};
impl Projection {
    /// Calculate the _physical_ camera's MVP, for each eye.
    /// camera_calib = camera calibration data.
    /// fov_left/right = adjusted fovs, in ratio (not in pixels)
    /// frame_time = how long after the first frame is the current frame taken
    /// time_origin = instant when the first frame is taken
    pub fn calculate_mvp(
        &self,
        mode: ProjectionMode,
        overlay_transform: &Matrix4<f64>,
        camera_calib: &crate::steam::StereoCamera,
        (fov_left, fov_right): (&[f64; 2], &[f64; 2]),
        ivrsystem: &crate::openvr::VRSystem,
        hmd_transform: &Matrix4<f64>,
    ) -> (Matrix4<f32>, Matrix4<f32>) {
        let left_eye: Matrix4<_> = ivrsystem
            .pin_mut()
            .GetEyeToHeadTransform(openvr_sys::EVREye::Eye_Left)
            .into();
        let right_eye: Matrix4<_> = ivrsystem
            .pin_mut()
            .GetEyeToHeadTransform(openvr_sys::EVREye::Eye_Right)
            .into();

        // Camera space to HMD space transform, based on physical measurements
        let left_cam: Matrix4<_> = matrix![
            1.0, 0.0, 0.0, -camera_calib.left.extrinsics.position[0];
            0.0, 1.0, 0.0, -camera_calib.left.extrinsics.position[1];
            0.0, 0.0, 1.0, -camera_calib.left.extrinsics.position[2];
            0.0, 0.0, 0.0, 1.0;
        ];
        let right_cam: Matrix4<_> = matrix![
            1.0, 0.0, 0.0, -camera_calib.right.extrinsics.position[0];
            0.0, 1.0, 0.0, -camera_calib.right.extrinsics.position[1];
            0.0, 0.0, 1.0, -camera_calib.right.extrinsics.position[2];
            0.0, 0.0, 0.0, 1.0;
        ];

        let (left_eye, right_eye) = match mode {
            ProjectionMode::FromEye => (hmd_transform * left_eye, hmd_transform * right_eye),
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
            fov_left[0] / 2.0, 0.0, 0.0, 0.0;
            0.0, fov_left[1], 0.0, 0.0;
            0.0, 0.0, -1.0, 0.0;
            0.0, 0.0, 0.0, 1.0;
        ];
        let camera_projection_right = matrix![
            fov_right[0] / 2.0, 0.0, 0.0, 0.0;
            0.0, fov_right[1] , 0.0, 0.0;
            0.0, 0.0, -1.0, 0.0;
            0.0, 0.0, 0.0, 1.0;
        ];
        (
            (camera_projection_left * left_view * overlay_transform).cast(),
            (camera_projection_right * right_view * overlay_transform).cast(),
        )
    }
    fn update_uniforms(&mut self, params: &ProjectionParameters) -> Result<()> {
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
        let mut transforms_write = self.uniforms.transforms.each_ref().try_map(|u| u.write())?;
        if mode != &self.saved_parameters.mode {
            let eye_offset_left = if *mode == ProjectionMode::FromEye {
                [
                    -camera_calib.left.extrinsics.position[0] as f32 - ipd / 2.0,
                    camera_calib.left.extrinsics.position[1] as f32,
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
            transforms_write[0].overlayWidth = *overlay_width;
            transforms_write[1].overlayWidth = *overlay_width;
        }
        Ok(())
    }
    fn make_uniform_buffer<T: BufferContents>(
        allocator: &StandardMemoryAllocator,
        uniform: T,
    ) -> Result<Arc<CpuAccessibleBuffer<T>>, AllocationCreationError> {
        CpuAccessibleBuffer::from_data(
            allocator,
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            uniform,
        )
    }
    pub fn new(
        device: Arc<Device>,
        allocator: &StandardMemoryAllocator,
        descriptor_set_allocator: &StandardDescriptorSetAllocator,
        source: Arc<AttachmentImage>,
        camera_calib: &crate::steam::StereoCamera,
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
            fs::ty::Info {
                texOffset: [0.0, 0.0],
            },
            fs::ty::Info {
                texOffset: [0.5, 0.0],
            },
        ]
        .try_map(|u| Self::make_uniform_buffer(allocator, u))?;
        let transforms = [bytemuck::Zeroable::zeroed(); 2]
            .try_map(|u| Self::make_uniform_buffer(allocator, u))?;
        let pipeline = GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .triangle_strip()
            .viewports_dynamic_scissors_irrelevant(1)
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
            device.clone(),
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
                    WriteDescriptorSet::buffer(0, tex_offsets[i].clone()),
                    WriteDescriptorSet::image_view_sampler(
                        1,
                        ImageView::new(source.clone(), ImageViewCreateInfo::from_image(&source))?,
                        sampler.clone(),
                    ),
                    WriteDescriptorSet::buffer(2, transforms[i].clone()),
                ],
            )?)
        })?;
        Ok(Self {
            saved_parameters: init_params,
            uniforms: Uniforms {
                tex_offsets,
                transforms,
            },
            desc_sets,
            device,
            render_pass,
            pipeline,
            source,
        })
    }
    pub fn project(
        &mut self,
        allocator: &FastMemoryAllocator,
        cmdbuf_allocator: &StandardCommandBufferAllocator,
        after: impl GpuFuture,
        queue: Arc<Queue>,
        output: Arc<AttachmentImage>,
        params: &ProjectionParameters,
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
        self.update_uniforms(params)?;
        let ProjectionParameters { overlay_width, .. } = params;
        let [w, h] = self.source.dimensions().width_height();
        let mut cmdbuf = AutoCommandBufferBuilder::primary(
            cmdbuf_allocator,
            queue.queue_family_index(),
            OneTimeSubmit,
        )?;
        cmdbuf.copy_image(CopyImageInfo::images(self.source.clone(), output.clone()))?;

        // Y is flipped from the vertex Y because texture coordinate is top-down
        let vertex_buffer = CpuAccessibleBuffer::<[Vertex]>::from_iter(
            allocator,
            BufferUsage {
                vertex_buffer: true,
                ..BufferUsage::empty()
            },
            false,
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
                    depth_range: -1.0..1.0,
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
                    depth_range: -1.0..1.0,
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
        Ok(after.then_execute(queue, cmdbuf.build()?)?)
    }
}
