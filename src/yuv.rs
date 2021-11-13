use anyhow::{anyhow, Result};
use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage::OneTimeSubmit},
    command_buffer::{CopyBufferImageError, SubpassContents},
    descriptor_set::{persistent::PersistentDescriptorSet, DescriptorSetError},
    device::{Device, Queue},
    format::Format::R8G8B8A8_UNORM,
    image::view::ImageView,
    image::{view::ImageViewCreationError, AttachmentImage, ImageCreationError, ImageUsage},
    pipeline::{
        viewport::Viewport, GraphicsPipeline, GraphicsPipelineCreationError, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreationError, RenderPass, Subpass},
    sync::GpuFuture,
    OomError,
};

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/trivial.vert",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/yuyv2rgb.frag",
    }
}

#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 2],
}
vulkano::impl_vertex!(Vertex, position);

#[derive(thiserror::Error, Debug)]
pub enum ConverterError {
    #[error("something went wrong: {0}")]
    Anyhow(#[from] anyhow::Error),
    #[error("{0}")]
    VkOom(#[from] OomError),
    #[error("{0}")]
    GraphicsPipelineCreationError(#[from] GraphicsPipelineCreationError),
    #[error("{0}")]
    ImageCreationError(#[from] ImageCreationError),
    #[error("{0}")]
    ImageViewCreationError(#[from] ImageViewCreationError),
    #[error("{0}")]
    DescriptorSetError(#[from] DescriptorSetError),
    #[error("{0}")]
    CopyBufferImageError(#[from] CopyBufferImageError),
    #[error("{0}")]
    FramebufferCreationError(#[from] FramebufferCreationError),
}

pub struct GpuYuyvConverter {
    device: Arc<Device>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    src: Arc<AttachmentImage>,
    desc_set: Arc<PersistentDescriptorSet>,
}

/// XXX: We can use VK_KHR_sampler_ycbcr_conversion for this, but I don't
/// know if it's widely supported. And the image format we need (G8B8G8R8_422_UNORM)
/// seems to have even less support than the extension itself.
impl GpuYuyvConverter {
    pub fn new(device: Arc<Device>, w: u32, h: u32) -> Result<Self> {
        if w % 2 != 0 {
            return Err(anyhow!("Width can't be odd"));
        }
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
                    dimensions: [w as f32, h as f32],
                    depth_range: -1.0..1.0,
                }])
                .fragment_shader(fs.main_entry_point(), ())
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(device.clone())?,
        );
        let src = AttachmentImage::with_usage(
            device.clone(),
            [w / 2, h], // 1 pixel of YUYV = 2 pixels of RGB
            R8G8B8A8_UNORM,
            ImageUsage {
                transfer_source: false,
                transfer_destination: true,
                sampled: true,
                storage: false,
                color_attachment: true,
                depth_stencil_attachment: false,
                transient_attachment: false,
                input_attachment: false,
            },
        )?;
        let desc_set_layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
        let mut desc_set_builder = PersistentDescriptorSet::start(desc_set_layout.clone());
        use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
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
        desc_set_builder.add_sampled_image(ImageView::new(src.clone())?, sampler)?;
        let desc_set = Arc::new(desc_set_builder.build()?);
        Ok(Self {
            src,
            render_pass,
            pipeline,
            device,
            desc_set,
        })
    }
    /// receives a buffer containing a YUYV image, upload it to GPU,
    /// and convert it to RGBA8.
    ///
    /// Returns a GPU future representing the operation, and an image.
    /// You must make sure the previous conversion is completed before
    /// calling this function again.
    pub fn yuyv_buffer_to_vulkan_image(
        &self,
        buf: &[u8],
        after: impl GpuFuture,
        queue: Arc<Queue>,
        buffer: &CpuBufferPool<u8>,
        output: Arc<AttachmentImage>,
    ) -> Result<impl GpuFuture> {
        use vulkano::device::DeviceOwned;
        if queue.device() != &self.device || buffer.device() != &self.device {
            return Err(anyhow!("Device mismatch"));
        }
        if let Some(queue) = after.queue() {
            if !queue.is_same(&queue) {
                return Err(anyhow!("Queue mismatch"));
            }
        }
        // Submit the source image to GPU
        let subbuffer = buffer
            .chunk(buf.iter().map(|x| *x))
            .map_err(|e| ConverterError::Anyhow(e.into()))?;
        let mut cmdbuf =
            AutoCommandBufferBuilder::primary(self.device.clone(), queue.family(), OneTimeSubmit)?;
        cmdbuf.copy_buffer_to_image(subbuffer, self.src.clone())?;
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
            )
            .map_err(|e| ConverterError::Anyhow(e.into()))?
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                self.desc_set.clone(),
            )
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .map_err(|e| ConverterError::Anyhow(e.into()))?
            .end_render_pass()
            .map_err(|e| ConverterError::Anyhow(e.into()))?;
        Ok(after.then_execute(
            queue,
            cmdbuf
                .build()
                .map_err(|e| ConverterError::Anyhow(e.into()))?,
        )?)
    }
}
