use std::sync::Arc;
use anyhow::anyhow;
use vulkano::{
    buffer::TypedBufferAccess,
    command_buffer::PrimaryCommandBuffer,
    command_buffer::SubpassContents,
    device::{Device, Queue},
    image::view::ImageView,
    image::AttachmentImage,
    pipeline::PipelineBindPoint,
    render_pass::Framebuffer,
    sync::GpuFuture,
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
    VkOom(#[from] vulkano::OomError),
    #[error("{0}")]
    GraphicsPipelineCreationError(#[from] vulkano::pipeline::GraphicsPipelineCreationError),
    #[error("{0}")]
    ImageCreationError(#[from] vulkano::image::ImageCreationError),
    #[error("{0}")]
    ImageViewCreationError(#[from] vulkano::image::view::ImageViewCreationError),
    #[error("{0}")]
    DescriptorSetError(#[from] vulkano::descriptor_set::DescriptorSetError),
    #[error("{0}")]
    CopyBufferImageError(#[from] vulkano::command_buffer::CopyBufferImageError),
    #[error("{0}")]
    FramebufferCreationError(#[from] vulkano::render_pass::FramebufferCreationError),
}

pub struct GpuYuyvConverter {
    device: Arc<Device>,
    render_pass: Arc<vulkano::render_pass::RenderPass>,
    pipeline: Arc<vulkano::pipeline::GraphicsPipeline>,
    src: Arc<AttachmentImage>,
    desc_set: Arc<vulkano::descriptor_set::persistent::PersistentDescriptorSet>,
    w: u32,
    h: u32,
}

impl GpuYuyvConverter {
    pub fn new(device: Arc<Device>, w: u32, h: u32) -> Result<Self, ConverterError> {
        if w % 2 != 0 {
            return Err(ConverterError::Anyhow(anyhow!("Width can't be odd")));
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
            vulkano::pipeline::GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_strip()
                .viewports([vulkano::pipeline::viewport::Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [w as f32, h as f32],
                    depth_range: -1.0..1.0,
                }])
                .fragment_shader(fs.main_entry_point(), ())
                .render_pass(vulkano::render_pass::Subpass::from(render_pass.clone(), 0).unwrap())
                .build(device.clone())?,
        );
        let src = AttachmentImage::with_usage(
            device.clone(),
            [w / 2, h], // 1 pixel of YUYV = 2 pixels of RGB
            vulkano::format::Format::R8G8B8A8_UNORM,
            vulkano::image::ImageUsage {
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
        let mut desc_set_builder =
            vulkano::descriptor_set::persistent::PersistentDescriptorSet::start(
                desc_set_layout.clone(),
            );
        use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
        let sampler = Sampler::new(
            device.clone(),
            Filter::Linear,
            Filter::Linear,
            MipmapMode::Nearest,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            0.0,
            1.0,
            0.0,
            0.0,
        )
        .unwrap();
        desc_set_builder
            .add_sampled_image(vulkano::image::view::ImageView::new(src.clone())?, sampler)?;
        let desc_set = Arc::new(desc_set_builder.build()?);
        Ok(Self {
            src,
            render_pass,
            pipeline,
            device,
            w,
            h,
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
        queue: Arc<Queue>,
        buffer: &vulkano::buffer::CpuBufferPool<u8>,
    ) -> Result<(impl GpuFuture, Arc<AttachmentImage>), ConverterError> {
        use vulkano::device::DeviceOwned;
        if queue.device() != &self.device || buffer.device() != &self.device {
            return Err(ConverterError::Anyhow(anyhow!("Device mismatch")));
        }
        // Submit the source image to GPU
        let subbuffer = buffer
            .chunk(buf.iter().map(|x| *x))
            .map_err(|e| ConverterError::Anyhow(e.into()))?;
        let mut cmdbuf = vulkano::command_buffer::AutoCommandBufferBuilder::primary(
            self.device.clone(),
            queue.family(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )?;
        cmdbuf.copy_buffer_to_image(subbuffer, self.src.clone())?;
        // Build a pipeline to do yuyv -> rgb
        let dst = AttachmentImage::with_usage(
            self.device.clone(),
            [self.w, self.h],
            vulkano::format::Format::R8G8B8A8_UNORM,
            vulkano::image::ImageUsage {
                transfer_source: true,
                transfer_destination: false,
                sampled: true,
                storage: false,
                color_attachment: true,
                depth_stencil_attachment: false,
                transient_attachment: false,
                input_attachment: false,
            },
        )?;
        let vertex_buffer = vulkano::buffer::CpuAccessibleBuffer::<[Vertex]>::from_iter(
            self.device.clone(),
            vulkano::buffer::BufferUsage::vertex_buffer(),
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
                .add(ImageView::new(dst.clone())?)?
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
        Ok((
            cmdbuf
                .build()
                .map_err(|e| ConverterError::Anyhow(e.into()))?
                .execute(queue.clone())
                .map_err(|e| ConverterError::Anyhow(e.into()))?,
            dst,
        ))
    }
}
