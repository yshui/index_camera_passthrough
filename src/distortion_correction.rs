use anyhow::{anyhow, Result};
use log::{info, trace};
use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage::OneTimeSubmit, SubpassContents,
    },
    descriptor_set::PersistentDescriptorSet,
    device::{Device, Queue},
    image::{view::ImageView, AttachmentImage},
    pipeline::GraphicsPipeline,
    pipeline::{viewport::Viewport, PipelineBindPoint},
    render_pass::{Framebuffer, RenderPass, Subpass},
    sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode},
    sync::GpuFuture,
};

#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 2],
}
vulkano::impl_vertex!(Vertex, position);

/// Lens correction for a stereo side-by-side image
pub struct StereoCorrection {
    device: Arc<Device>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    desc_set: Arc<PersistentDescriptorSet>,
}

impl StereoCorrection {
    /// Return a scale that will map src to dst.
    /// i.e. solving Undistort(src * scale) = dst for the smallest non-zero root.
    fn find_scale(coeff: [f32; 3], src: f32, dst: f32) -> f32 {
        let coeff = [coeff[0] as f64, coeff[1] as f64, coeff[2] as f64];
        let dst = dst as f64;
        // solving: x * (1 + k1*x^2 + k2*x^4 + k3*x^6) - dst = 0
        let f = |x: f64| x + x * x * x * (coeff[0] + x * x * (coeff[1] + x * x * coeff[2])) - dst;
        let fp = |x: f64| {
            1.0 + x * x * (3.0 * coeff[0] + x * x * (5.0 * coeff[1] + x * x * 7.0 * coeff[2]))
        };
        const MAX_ITER: u32 = 100;
        let mut x = 0.0;
        for _ in 0..MAX_ITER {
            if fp(x) == 0.0 {
                // Give up
                info!("Divided by zero");
                return 1.0;
            }
            trace!("{} {} {}", x, f(x), fp(x));
            if f(x).abs() < 1e-6 {
                info!("Scale is: {}, {} {}", x as f32 / src, f(x), dst);
                return x as f32 / src;
            }
            x = x - f(x) / fp(x);
        }
        // Give up
        info!("Cannot find scale");
        return 1.0;
    }
    /// Input size is (size * 2, size)
    /// returns also the adjusted FOV
    pub fn new(
        device: Arc<Device>,
        input: Arc<AttachmentImage>,
        coeff: [f32; 3],
        cleft: [f32; 2],
        cright: [f32; 2],
        focal: f32,
    ) -> Result<(Self, f32)> {
        if input.dimensions()[0] != input.dimensions()[1] * 2 {
            return Err(anyhow!("Input not square"));
        }
        let size = input.dimensions()[1];
        let vs = vs::Shader::load(device.clone())?;
        let fs = fs::Shader::load(device.clone())?;
        let render_pass = Arc::new(
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
            )
            .unwrap(),
        );
        let pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_strip()
                .viewports([Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [size as f32 * 2.0, size as f32],
                    depth_range: -1.0..1.0,
                }])
                .fragment_shader(fs.main_entry_point(), ())
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(device.clone())?,
        );
        let desc_set_layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
        let mut desc_set_builder = PersistentDescriptorSet::start(desc_set_layout.clone());
        let sizef = size as f32;
        let min_off_center = [
            sizef - cleft[0],
            sizef - cleft[1],
            sizef - cright[0],
            sizef - cright[1],
            cleft[0],
            cleft[1],
            cright[0],
            cright[1],
        ]
        .into_iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
        let focal_px = focal * size as f32;
        let uniform = fs::ty::Parameters {
            cleft,
            cright,
            dcoef: coeff,
            focal,
            scale: Self::find_scale(
                coeff,
                (size / 2) as f32 / focal_px,
                min_off_center / focal_px,
            ),
            _dummy0: Default::default(),
        };
        let uniform = CpuAccessibleBuffer::from_data(
            device.clone(),
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::none()
            },
            false,
            uniform,
        )?;
        let sampler = Sampler::new(
            device.clone(),
            Filter::Linear,
            Filter::Linear,
            MipmapMode::Nearest,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            0.0,
            1.0,
            0.0,
            0.0,
        )?;

        desc_set_builder
            .add_buffer(uniform)?
            .add_sampled_image(ImageView::new(input.clone())?, sampler)?;
        let desc_set = Arc::new(desc_set_builder.build()?);
        Ok((Self {
            device,
            render_pass,
            pipeline,
            desc_set,
        }, focal * min_off_center * 2.0 / size as f32))
    }
    pub fn correct(
        &self,
        after: impl GpuFuture,
        queue: Arc<Queue>,
        output: Arc<AttachmentImage>
    ) -> Result<impl GpuFuture> {
        use vulkano::device::DeviceOwned;
        if queue.device() != &self.device {
            return Err(anyhow!("Device mismatch"));
        }
        if let Some(queue) = after.queue() {
            if !queue.is_same(&queue) {
                return Err(anyhow!("Queue mismatch"));
            }
        }
        // Submit the source image to GPU
        let mut cmdbuf =
            AutoCommandBufferBuilder::primary(self.device.clone(), queue.family(), OneTimeSubmit)?;
        // Build a pipeline to do yuyv -> rgb
        let vertex_buffer = CpuAccessibleBuffer::<[Vertex]>::from_iter(
            self.device.clone(),
            BufferUsage::vertex_buffer(),
            false,
            [
                Vertex {
                    position: [-1.0, -1.0],
                },
                Vertex {
                    position: [-1.0, 1.0],
                },
                Vertex {
                    position: [1.0, -1.0],
                },
                Vertex {
                    position: [1.0, 1.0],
                },
            ]
            .iter()
            .cloned(),
        )
        .unwrap();
        let framebuffer = Arc::new(
            Framebuffer::start(self.render_pass.clone())
                .add(ImageView::new(output.clone())?)?
                .build()?,
        );
        cmdbuf
            .begin_render_pass(
                framebuffer.clone(),
                SubpassContents::Inline,
                [vulkano::format::ClearValue::None],
            )?
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                self.desc_set.clone(),
            )
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .draw(vertex_buffer.len() as u32, 1, 0, 0)?
            .end_render_pass()?;
        Ok(after.then_execute(queue, cmdbuf.build()?)?)
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/trivial.vert",
    }
}
mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/stereo_correction.frag",
    }
}
