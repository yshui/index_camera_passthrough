use anyhow::{anyhow, Result};
use log::{info, trace};
use smallvec::smallvec;
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::CommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferUsage::OneTimeSubmit, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
        SubpassEndInfo,
    },
    descriptor_set::{allocator::DescriptorSetAllocator, DescriptorSet, WriteDescriptorSet},
    device::{Device, Queue},
    image::{
        sampler::{Filter, Sampler, SamplerCreateInfo},
        view::{ImageView, ImageViewCreateInfo},
        Image, ImageLayout,
    },
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocatePreference, MemoryAllocator, MemoryTypeFilter,
    },
    pipeline::{graphics::viewport::Viewport, PipelineBindPoint},
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineSubpassType,
            vertex_input::{Vertex as VertexTrait, VertexDefinition},
            viewport::ViewportState,
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::GpuFuture,
    Handle, VulkanObject,
};

#[derive(VertexTrait, Default, Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(non_snake_case)]
#[repr(C)]
struct Vertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R32G32_SFLOAT)]
    inCoord: [f32; 2],
}

/// Lens correction for a stereo side-by-side image
pub struct StereoCorrection {
    device: Arc<Device>,
    render_passes: [Arc<RenderPass>; 2],
    pipelines: [Arc<GraphicsPipeline>; 2],
    desc_sets: [Arc<DescriptorSet>; 2],
    /// field-of-view parameter, 0 = left eye, 1 = right eye
    fov: [[f32; 2]; 2],
}

impl std::fmt::Debug for StereoCorrection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StereoCorrection")
            .field("device", &self.device.handle().as_raw())
            .field(
                "render_passes",
                &self.render_passes.each_ref().map(|x| x.handle().as_raw()),
            )
            .field(
                "pipelines",
                &self.pipelines.each_ref().map(|x| x.handle().as_raw()),
            )
            .field("fov", &self.fov)
            .finish_non_exhaustive()
    }
}

impl StereoCorrection {
    pub fn fov(&self) -> [[f32; 2]; 2] {
        self.fov
    }
    /// i.e. solving Undistort(src) = dst for the smallest non-zero root.
    fn undistort_inverse(coeff: &[f64; 4], dst: f64) -> Option<f64> {
        // solving: x * (1 + k1*x^2 + k2*x^4 + k3*x^6 + k4*x^8) - dst = 0
        let f = |x: f64| {
            let x2 = x * x;
            x * (1.0 + x2 * (coeff[0] + x2 * (coeff[1] + x2 * (coeff[2] + x2 * coeff[3])))) - dst
        };
        let fp = |x: f64| {
            let x2 = x * x;
            1.0 + x2
                * (3.0 * coeff[0]
                    + x2 * (5.0 * coeff[1] + x2 * (7.0 * coeff[2] + x2 * 9.0 * coeff[3])))
        };
        const MAX_ITER: u32 = 100;
        let mut x = 0.0;
        for _ in 0..MAX_ITER {
            if fp(x) == 0.0 {
                // Give up
                info!("Divided by zero");
                return None;
            }
            trace!("{} {} {}", x, f(x), fp(x));
            if f(x).abs() < 1e-6 {
                info!("Inverse is: {}, {} {}", x, f(x), dst);
                return Some(x);
            }
            x = x - f(x) / fp(x);
        }
        // Give up
        info!("Cannot find scale");
        None
    }
    // Find a scale that maps the middle point of 4 edges of the undistorted image to
    // the edge of the field of view of the distorted image.
    //
    // Returns the scales and the adjusted fovs
    fn find_scale(coeff: &[f64; 4], center: &[f64; 2], focal: &[f64; 2]) -> [(f32, f32); 2] {
        [0, 1].map(|i| {
            let min_edge_dist = center[i].min(1.0 - center[i]) / focal[i];
            // Find the input theta angle where Undistort(theta) = min_edge_dist
            if let Some(theta) = Self::undistort_inverse(coeff, min_edge_dist) {
                if theta >= std::f64::consts::PI / 2.0 {
                    // infinity?
                    (1.0, focal[i] as f32)
                } else {
                    // Find the input coordinates that will give us that theta
                    let target_edge = theta.tan();
                    log::info!("{}", target_edge);
                    (
                        (target_edge / (0.5 / focal[i])) as f32,
                        (1.0 / min_edge_dist / 2.0) as f32,
                    )
                }
            } else {
                // Cannot find scale so just don't scale
                (1.0, focal[i] as f32)
            }
        })
    }
    /// Input size is (size * 2, size)
    /// returns also the adjusted FOV for left and right
    ///
    /// # Arguments
    ///
    /// - is_final: whether this is the final stage of the pipeline.
    ///             if true, the output image will be submitted to
    ///             the vr compositor.
    pub fn new(
        device: Arc<Device>,
        allocator: Arc<dyn MemoryAllocator>,
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
        input: Arc<Image>,
        camera_calib: &crate::vrapi::StereoCamera,
    ) -> Result<Self> {
        let [w, h, _] = input.extent();
        if w != h * 2 {
            return Err(anyhow!("Input not square"));
        }
        let size = h as f64;
        let center_left = [
            camera_calib.left.intrinsics.center_x / size,
            camera_calib.left.intrinsics.center_y / size,
        ];
        let center_right = [
            camera_calib.right.intrinsics.center_x / size,
            camera_calib.right.intrinsics.center_y / size,
        ];
        let focal_left = [
            camera_calib.left.intrinsics.focal_x / size,
            camera_calib.left.intrinsics.focal_y / size,
        ];
        let focal_right = [
            camera_calib.right.intrinsics.focal_x / size,
            camera_calib.right.intrinsics.focal_y / size,
        ];
        let coeff_left = camera_calib.left.intrinsics.distort.coeffs;
        let coeff_right = camera_calib.left.intrinsics.distort.coeffs;
        let vs = vs::load(device.clone())?;
        let fs = fs::load(device.clone())?;
        let render_passes = [
            vulkano::single_pass_renderpass!(device.clone(),
                attachments: {
                    color: {
                        format: vulkano::format::Format::R8G8B8A8_UNORM,
                        samples: 1,
                        load_op: DontCare,
                        store_op: Store,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {}
                }
            )?,
            vulkano::single_pass_renderpass!(device.clone(),
                attachments: {
                    color: {
                        format: vulkano::format::Format::R8G8B8A8_UNORM,
                        samples: 1,
                        load_op: Load,
                        store_op: Store,
                        final_layout: ImageLayout::ColorAttachmentOptimal,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {}
                }
            )?,
        ];
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
        let pipelines = [0, 1].try_map(|id| {
            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages[..].into(),
                    vertex_input_state: Some(
                        Vertex::per_vertex().definition(&vs.info().input_interface)?,
                    ),
                    input_assembly_state: Some(InputAssemblyState {
                        topology: PrimitiveTopology::TriangleStrip,
                        ..Default::default()
                    }),
                    viewport_state: Some(ViewportState {
                        viewports: smallvec![Viewport {
                            offset: [size as f32 * id as f32, 0.0],
                            extent: [size as f32, size as f32],
                            depth_range: 0.0..=1.0,
                        }],
                        ..Default::default()
                    }),
                    subpass: Some(PipelineSubpassType::BeginRenderPass(
                        Subpass::from(render_passes[id].clone(), 0).unwrap(),
                    )),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        1,
                        Default::default(),
                    )),
                    ..GraphicsPipelineCreateInfo::layout(layout.clone())
                },
            )
            .map_err(anyhow::Error::from)
        })?;
        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                ..Default::default()
            },
        )?;

        let coeffs = [
            (0, coeff_left, center_left, focal_left),
            (1, coeff_right, center_right, focal_right),
        ];
        let scale_fov = coeffs
            .each_ref()
            .map(|(_, coeff, center, focal)| Self::find_scale(coeff, center, focal));
        // Left pass
        let desc_sets = coeffs.try_map(|(id, coeff, center, focal)| {
            let uniform = fs::Parameters {
                center: center.map(|x| x as f32),
                dcoef: coeff.map(|x| x as f32),
                focal: focal.map(|x| x as f32),
                sensorSize: (size as f32).into(),
                scale: [scale_fov[id][0].0, scale_fov[id][1].0],
                texOffset: [0.5 * id as f32, 0.0],
            };
            let uniform = Buffer::from_data(
                allocator.clone(),
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
            )?;
            let desc_set_layout = pipelines[id].layout().set_layouts().first().unwrap();
            Ok::<_, anyhow::Error>(DescriptorSet::new(
                descriptor_set_allocator.clone(),
                desc_set_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, uniform),
                    WriteDescriptorSet::image_view_sampler(
                        1,
                        ImageView::new(input.clone(), ImageViewCreateInfo::from_image(&input))?,
                        sampler.clone(),
                    ),
                ],
                None,
            )?)
        })?;

        Ok(Self {
            device,
            render_passes,
            pipelines,
            desc_sets,
            fov: [
                [scale_fov[0][0].1, scale_fov[0][1].1],
                [scale_fov[1][0].1, scale_fov[1][1].1],
            ],
        })
    }
    pub fn correct(
        &self,
        cmdbuf_allocator: Arc<dyn CommandBufferAllocator>,
        allocator: Arc<dyn MemoryAllocator>,
        after: impl GpuFuture,
        queue: &Arc<Queue>,
        output: Arc<Image>,
    ) -> Result<impl GpuFuture> {
        use vulkano::device::DeviceOwned;
        if queue.device() != &self.device {
            return Err(anyhow!("Device mismatch"));
        }
        if let Some(after_queue) = after.queue() {
            if &after_queue != queue {
                return Err(anyhow!("Queue mismatch"));
            }
        }
        let mut cmdbuf = AutoCommandBufferBuilder::primary(
            cmdbuf_allocator,
            queue.queue_family_index(),
            OneTimeSubmit,
        )?;
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
                    inCoord: [-0.5, -0.5],
                },
                Vertex {
                    position: [-1.0, 1.0],
                    inCoord: [-0.5, 0.5],
                },
                Vertex {
                    position: [1.0, -1.0],
                    inCoord: [0.5, -0.5],
                },
                Vertex {
                    position: [1.0, 1.0],
                    inCoord: [0.5, 0.5],
                },
            ]
            .iter()
            .cloned(),
        )
        .unwrap();
        for id in 0..2 {
            let framebuffer = Framebuffer::new(
                self.render_passes[id].clone(),
                FramebufferCreateInfo {
                    attachments: vec![ImageView::new(
                        output.clone(),
                        ImageViewCreateInfo::from_image(&output),
                    )?],
                    ..Default::default()
                },
            )?;
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
                .bind_pipeline_graphics(self.pipelines[id].clone())?
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipelines[id].layout().clone(),
                    0,
                    self.desc_sets[id].clone(),
                )?
                .bind_vertex_buffers(0, vertex_buffer.clone())?
                .draw(vertex_buffer.len() as u32, 1, 0, 0)?
                .end_render_pass(SubpassEndInfo::default())?;
        }
        Ok(after.then_execute(queue.clone(), cmdbuf.build()?)?)
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "#version 450
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 inCoord;
layout(location = 0) out vec2 coord;

void main() {
    gl_Position = vec4(position, 0, 1);
    coord = inCoord;
}"
    }
}
mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/stereo_correction.frag",
        custom_derives: [Copy, Clone, Debug],
    }
}
