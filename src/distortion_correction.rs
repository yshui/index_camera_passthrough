use anyhow::{anyhow, Result};
use log::{info, trace};
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferUsage::OneTimeSubmit, RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    image::{
        view::{ImageView, ImageViewCreateInfo},
        AttachmentImage, ImageAccess,
    },
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocatePreference, MemoryUsage, StandardMemoryAllocator,
    },
    pipeline::{graphics::viewport::Viewport, PipelineBindPoint},
    pipeline::{
        graphics::{
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::Vertex as VertexTrait,
            viewport::ViewportState,
        },
        GraphicsPipeline, Pipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sampler::{Filter, Sampler, SamplerCreateInfo},
    sync::GpuFuture,
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
    desc_sets: [Arc<PersistentDescriptorSet>; 2],
}

impl StereoCorrection {
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
    fn find_scale(coeff: &[f64; 4], center: &[f64; 2], focal: &[f64; 2]) -> [(f64, f64); 2] {
        [0, 1].map(|i| {
            let min_edge_dist = center[i].min(1.0 - center[i]) / focal[i];
            // Find the input theta angle where Undistort(theta) = min_edge_dist
            if let Some(theta) = Self::undistort_inverse(coeff, min_edge_dist) {
                if theta >= std::f64::consts::PI / 2.0 {
                    // infinity?
                    (1.0, focal[i])
                } else {
                    // Find the input coordinates that will give us that theta
                    let target_edge = theta.tan();
                    log::info!("{}", target_edge);
                    (target_edge / (0.5 / focal[i]), 1.0 / min_edge_dist / 2.0)
                }
            } else {
                // Cannot find scale so just don't scale
                (1.0, focal[i])
            }
        })
    }
    /// Input size is (size * 2, size)
    /// returns also the adjusted FOV for left and right
    pub fn new(
        device: Arc<Device>,
        allocator: &StandardMemoryAllocator,
        descriptor_set_allocator: &StandardDescriptorSetAllocator,
        input: Arc<AttachmentImage>,
        camera_calib: &crate::steam::StereoCamera,
    ) -> Result<(Self, [f64; 2], [f64; 2])> {
        if input.dimensions().width() != input.dimensions().height() * 2 {
            return Err(anyhow!("Input not square"));
        }
        let size = input.dimensions().height() as f64;
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
                        load: DontCare,
                        store: Store,
                        format: vulkano::format::Format::R8G8B8A8_UNORM,
                        samples: 1,
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
            )?,
        ];
        let pipelines = [0, 1].try_map(|id| {
            GraphicsPipeline::start()
                .vertex_input_state(Vertex::per_vertex())
                .vertex_shader(vs.entry_point("main").unwrap(), ())
                .input_assembly_state(
                    InputAssemblyState::new().topology(PrimitiveTopology::TriangleStrip),
                )
                .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
                    Viewport {
                        origin: [size as f32 * id as f32, 0.0],
                        dimensions: [size as f32, size as f32],
                        depth_range: -1.0..1.0,
                    },
                ]))
                .fragment_shader(fs.entry_point("main").unwrap(), ())
                .render_pass(Subpass::from(render_passes[id].clone(), 0).unwrap())
                .build(device.clone())
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
                scale: [scale_fov[id][0].0 as f32, scale_fov[id][1].0 as f32],
                texOffset: [0.5 * id as f32, 0.0],
            };
            let uniform = Buffer::from_data(
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
            )?;
            let desc_set_layout = pipelines[id].layout().set_layouts().get(0).unwrap();
            Ok::<_, anyhow::Error>(PersistentDescriptorSet::new(
                descriptor_set_allocator,
                desc_set_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, uniform),
                    WriteDescriptorSet::image_view_sampler(
                        1,
                        ImageView::new(input.clone(), ImageViewCreateInfo::from_image(&input))?,
                        sampler.clone(),
                    ),
                ],
            )?)
        })?;

        Ok((
            Self {
                device,
                render_passes,
                pipelines,
                desc_sets,
            },
            [scale_fov[0][0].1, scale_fov[0][1].1],
            [scale_fov[1][0].1, scale_fov[1][1].1],
        ))
    }
    pub fn correct(
        &self,
        cmdbuf_allocator: &StandardCommandBufferAllocator,
        allocator: &StandardMemoryAllocator,
        after: impl GpuFuture,
        queue: Arc<Queue>,
        output: Arc<AttachmentImage>,
    ) -> Result<impl GpuFuture> {
        use vulkano::device::DeviceOwned;
        if queue.device() != &self.device {
            return Err(anyhow!("Device mismatch"));
        }
        if let Some(after_queue) = after.queue() {
            if after_queue != queue {
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
                usage: MemoryUsage::Upload,
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
            let image_view_create_info = ImageViewCreateInfo::from_image(&output);
            let framebuffer = Framebuffer::new(
                self.render_passes[id].clone(),
                FramebufferCreateInfo {
                    attachments: vec![ImageView::new(output.clone(), image_view_create_info)?],
                    ..Default::default()
                },
            )?;
            let mut render_pass_begin_info = RenderPassBeginInfo::framebuffer(framebuffer);
            render_pass_begin_info.clear_values = vec![None];
            cmdbuf
                .begin_render_pass(render_pass_begin_info, SubpassContents::Inline)?
                .bind_pipeline_graphics(self.pipelines[id].clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipelines[id].layout().clone(),
                    0,
                    self.desc_sets[id].clone(),
                )
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(vertex_buffer.len() as u32, 1, 0, 0)?
                .end_render_pass()?;
        }
        Ok(after.then_execute(queue, cmdbuf.build()?)?)
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
