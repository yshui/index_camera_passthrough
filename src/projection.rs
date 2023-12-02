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
use anyhow::Result;
use std::sync::Arc;
use vulkano::{
    buffer::{
        AllocateBufferError, Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::CommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferExecError,
        CommandBufferUsage::OneTimeSubmit, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
        SubpassEndInfo,
    },
    descriptor_set::{allocator::DescriptorSetAllocator, DescriptorSet, WriteDescriptorSet},
    device::{Device, Queue},
    image::view::{ImageView, ImageViewCreateInfo},
    image::{
        sampler::{Filter, Sampler, SamplerCreateInfo},
        Image, ImageLayout,
    },
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocatePreference, MemoryAllocator, MemoryTypeFilter,
    },
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex as VertexTrait, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::{IntoPipelineLayoutCreateInfoError, PipelineDescriptorSetLayoutCreateInfo},
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    sync::{GpuFuture, HostAccessError},
    Validated, VulkanError,
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

#[derive(Debug)]
pub struct Projection {
    extent: [u32; 2],
    pipeline: Arc<GraphicsPipeline>,
    render_pass: Arc<RenderPass>,
    // [0: left, 1: right]
    uniforms: Uniforms,
    saved_parameters: ProjectionParameters,
    mode_ipd_changed: bool,
    mvps_changed: bool,
    desc_sets: [Arc<DescriptorSet>; 2],
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

#[derive(thiserror::Error, Debug)]
pub enum ProjectorError {
    #[error("input image is not square {0}x{1}")]
    NotSquare(u32, u32),
    #[error("vulkan error {0}")]
    Vulkan(#[from] Validated<VulkanError>),
    #[error("{0}")]
    CreateInfo(#[from] IntoPipelineLayoutCreateInfoError),
    #[error("buffer allocation error: {0}")]
    BufferAlloc(#[from] Validated<AllocateBufferError>),
    #[error("host access error: {0}")]
    HostAccess(#[from] HostAccessError),
    #[error("command buffer execution error: {0}")]
    CommandBuffer(#[from] CommandBufferExecError),
}

impl From<Box<vulkano::ValidationError>> for ProjectorError {
    fn from(value: Box<vulkano::ValidationError>) -> Self {
        Self::Vulkan(Validated::from(value))
    }
}

use nalgebra::{matrix, Matrix4, RawStorage, Scalar};
impl Projection {
    /// Calculate the _physical_ camera's MVP, for each eye.
    /// camera_calib = camera calibration data.
    /// fov_left/right = adjusted fovs, in ratio (not in pixels)
    /// frame_time = how long after the first frame is the current frame taken
    /// time_origin = instant when the first frame is taken
    pub(crate) fn update_mvps(
        &mut self,
        overlay_transform: &Matrix4<f64>,
        fov: &[[f64; 2]; 2],
        eye_to_head: &[Matrix4<f64>; 2],
        hmd_transform: &Matrix4<f64>,
    ) -> Result<(), ProjectorError> {
        let left_extrinsics_position = self
            .saved_parameters
            .camera_calib
            .map(|c| c.left.extrinsics.position)
            .unwrap_or_default();
        // Camera space to HMD space transform, based on physical measurements
        let left_cam: Matrix4<_> = matrix![
            1.0, 0.0, 0.0, -left_extrinsics_position[0];
            0.0, 1.0, 0.0, -left_extrinsics_position[1];
            0.0, 0.0, 1.0, -left_extrinsics_position[2];
            0.0, 0.0, 0.0, 1.0;
        ];
        let right_extrinsics_position = self
            .saved_parameters
            .camera_calib
            .map(|c| c.right.extrinsics.position)
            .unwrap_or_default();
        let right_cam: Matrix4<_> = matrix![
            1.0, 0.0, 0.0, -right_extrinsics_position[0];
            0.0, 1.0, 0.0, -right_extrinsics_position[1];
            0.0, 0.0, 1.0, -right_extrinsics_position[2];
            0.0, 0.0, 0.0, 1.0;
        ];

        let (left_eye, right_eye) = match self.saved_parameters.mode {
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
        self.set_mvps([
            (camera_projection_left * left_view * overlay_transform).cast(),
            (camera_projection_right * right_view * overlay_transform).cast(),
        ]);
        Ok(())
    }
    pub fn set_ipd(&mut self, ipd: f32) {
        if self.saved_parameters.ipd == ipd {
            return;
        }

        self.saved_parameters.ipd = ipd;
        self.mode_ipd_changed = true;
    }
    fn set_mvps(&mut self, mvps: [Matrix4<f32>; 2]) {
        if self.saved_parameters.mvps == mvps {
            return;
        }
        self.saved_parameters.mvps = mvps;
        self.mvps_changed = true;
    }
    pub fn set_mode(&mut self, mode: ProjectionMode) {
        if self.saved_parameters.mode == mode {
            return;
        }
        self.saved_parameters.mode = mode;
        self.mode_ipd_changed = true;
    }
    pub fn recalculate_uniforms(&mut self) -> Result<(), ProjectorError> {
        if !self.mode_ipd_changed && !self.mvps_changed {
            return Ok(());
        }

        let ProjectionParameters {
            mode,
            ipd,
            mvps,
            camera_calib,
            ..
        } = &self.saved_parameters;
        let mut transforms_write = self.uniforms.transforms.each_ref().try_map(|u| u.write())?;
        if self.mode_ipd_changed {
            let left_extrinsics_position = camera_calib
                .map(|c| c.left.extrinsics.position)
                .unwrap_or_default();
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
            self.mode_ipd_changed = false;
        }
        if self.mvps_changed {
            for (mvp, write) in mvps.iter().zip(transforms_write.iter_mut()) {
                write.mvp = *mvp.as_ref();
            }
            self.mvps_changed = false;
        }

        Ok(())
    }
    fn make_uniform_buffer<T: BufferContents>(
        allocator: Arc<dyn MemoryAllocator>,
        uniform: T,
    ) -> Result<Subbuffer<T>, Validated<AllocateBufferError>> {
        log::debug!("uniform buffer size {}", std::mem::size_of::<T>());
        Buffer::from_data(
            allocator,
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::PREFER_DEVICE,
                allocate_preference: MemoryAllocatePreference::Unknown,
                ..Default::default()
            },
            uniform,
        )
    }
    pub fn new(
        device: Arc<Device>,
        allocator: Arc<dyn MemoryAllocator>,
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
        source: &Arc<Image>,
        overlay_width: f32,
        camera_calib: &Option<crate::vrapi::StereoCamera>,
    ) -> Result<Self, ProjectorError> {
        let [w, h, _] = source.extent();
        if w != h * 2 {
            return Err(ProjectorError::NotSquare(w, h));
        }
        let vs = vs::load(device.clone())?;
        let fs = fs::load(device.clone())?;
        let render_pass = vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    format: vulkano::format::Format::R8G8B8A8_UNORM,
                    samples: 1,
                    load_op: Load,
                    store_op: Store,
                    final_layout: ImageLayout::TransferSrcOptimal,
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
        .try_map(|u| Self::make_uniform_buffer(allocator.clone(), u))?;
        let vs = vs.entry_point("main").unwrap();
        let fs = fs.entry_point("main").unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(vs.clone()),
            PipelineShaderStageCreateInfo::new(fs),
        ];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())?,
        )?;
        let transforms = [vs::Transform::default(); 2]
            .try_map(|u| Self::make_uniform_buffer(allocator.clone(), u))?;
        {
            let mut transform_writes = [transforms[0].write()?, transforms[1].write()?];
            transform_writes[0].overlayWidth = overlay_width.into();
            transform_writes[1].overlayWidth = overlay_width.into();
        }
        log::info!("before");
        let pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                vertex_input_state: Some(
                    Vertex::per_vertex()
                        .definition(&vs.info().input_interface)
                        .map_err(Validated::<VulkanError>::from)?,
                ),
                stages: stages.into_iter().collect(),
                input_assembly_state: Some(InputAssemblyState {
                    topology: PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                }),
                viewport_state: Some(ViewportState::default()),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(Subpass::from(render_pass.clone(), 0).unwrap().into()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    1,
                    Default::default(),
                )),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )?;
        log::info!("after");
        let init_params = ProjectionParameters {
            mode: ProjectionMode::FromCamera, // This means `eyeOffset` should be zero, which would
            // be what is returned by `bytemuck::Zeroable`
            ipd: f32::NAN,
            overlay_width,
            camera_calib: *camera_calib,
            mvps: [Matrix4::identity(), Matrix4::identity()],
        };
        let layout = pipeline.layout().set_layouts().first().unwrap();
        let sampler = Sampler::new(
            device,
            SamplerCreateInfo {
                min_filter: Filter::Linear,
                mag_filter: Filter::Linear,
                ..Default::default()
            },
        )?;
        let desc_sets = [0, 1].try_map(|i| {
            DescriptorSet::new(
                descriptor_set_allocator.clone(),
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, transforms[i].clone()),
                    WriteDescriptorSet::image_view_sampler(
                        1,
                        ImageView::new(source.clone(), ImageViewCreateInfo::from_image(source))?,
                        sampler.clone(),
                    ),
                    WriteDescriptorSet::buffer(2, tex_offsets[i].clone()),
                ],
                None,
            )
            .map_err(ProjectorError::from)
        })?;
        let source_extent = source.extent();
        Ok(Self {
            saved_parameters: init_params,
            uniforms: Uniforms { transforms },
            desc_sets,
            render_pass,
            pipeline,
            extent: [source_extent[0], source_extent[1]],
            mode_ipd_changed: true,
            mvps_changed: true,
        })
    }
    pub fn project(
        &mut self,
        allocator: Arc<dyn MemoryAllocator>,
        cmdbuf_allocator: Arc<dyn CommandBufferAllocator>,
        after: impl GpuFuture,
        queue: &Arc<Queue>,
        output: Arc<Image>,
    ) -> Result<impl GpuFuture, ProjectorError> {
        self.recalculate_uniforms()?;
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
        let [w, h] = self.extent;
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
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::PREFER_DEVICE,
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
            .begin_render_pass(
                render_pass_begin_info,
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )?
            .set_viewport(
                0,
                Some(Viewport {
                    offset: [0.0, 0.0],
                    extent: [(w / 2) as f32, h as f32],
                    depth_range: 0.0..=1.0,
                })
                .into_iter()
                .collect(),
            )?
            .bind_pipeline_graphics(self.pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                self.desc_sets[0].clone(),
            )?
            .bind_vertex_buffers(0, vertex_buffer.clone())?
            .draw(vertex_buffer.len() as u32, 1, 0, 0)?
            .end_render_pass(SubpassEndInfo::default())?;

        // Right
        let mut render_pass_begin_info = RenderPassBeginInfo::framebuffer(framebuffer);
        render_pass_begin_info.clear_values = vec![None];
        cmdbuf
            .begin_render_pass(
                render_pass_begin_info,
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )?
            .set_viewport(
                0,
                Some(Viewport {
                    offset: [(w / 2) as f32, 0.0],
                    extent: [(w / 2) as f32, h as f32],
                    depth_range: 0.0..=1.0,
                })
                .into_iter()
                .collect(),
            )?
            .bind_pipeline_graphics(self.pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                self.desc_sets[1].clone(),
            )?
            .bind_vertex_buffers(0, vertex_buffer.clone())?
            .draw(vertex_buffer.len() as u32, 1, 0, 0)?
            .end_render_pass(SubpassEndInfo::default())?;
        Ok(after.then_execute(queue.clone(), cmdbuf.build()?)?)
    }
}
