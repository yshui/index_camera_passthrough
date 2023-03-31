use anyhow::{anyhow, Result};
use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::SubpassContents,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferUsage::OneTimeSubmit, CopyBufferToImageInfo, RenderPassBeginInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, persistent::PersistentDescriptorSet,
        WriteDescriptorSet,
    },
    device::{Device, DeviceOwned, Queue},
    format::Format::R8G8B8A8_UNORM,
    image::view::{ImageView, ImageViewCreateInfo},
    image::{view::ImageViewCreationError, AttachmentImage, ImageUsage},
    memory::allocator::{FastMemoryAllocator, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreationError,
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{
        Framebuffer, FramebufferCreateInfo, FramebufferCreationError, RenderPass, Subpass,
    },
    sampler::{Filter, Sampler, SamplerCreateInfo},
    sync::GpuFuture,
    OomError,
};

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "#version 450
layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position, 0, 1);
}"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/yuyv2rgb.frag",
    }
}

#[derive(Default, Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
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
    //#[error("{0}")]
    //ImageCreationError(#[from] ImageCreationError),
    #[error("{0}")]
    ImageViewCreationError(#[from] ImageViewCreationError),
    //#[error("{0}")]
    //DescriptorSetError(#[from] DescriptorSetError),
    //#[error("{0}")]
    //CopyBufferImageError(#[from] CopyBufferImageError),
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
    pub fn new(
        device: Arc<Device>,
        allocator: &StandardMemoryAllocator,
        descriptor_set_allocator: &StandardDescriptorSetAllocator,
        w: u32,
        h: u32,
    ) -> Result<Self> {
        if w % 2 != 0 {
            return Err(anyhow!("Width can't be odd"));
        }
        let vs = vs::load(device.clone())?;
        let fs = fs::load(device.clone())?;
        let render_pass = vulkano::single_pass_renderpass!(device.clone(),
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
        )?;
        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .input_assembly_state(
                InputAssemblyState::new().topology(PrimitiveTopology::TriangleStrip),
            )
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
                Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [w as f32, h as f32],
                    depth_range: -1.0..1.0,
                },
            ]))
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())?;
        let src = AttachmentImage::with_usage(
            allocator,
            [w / 2, h], // 1 pixel of YUYV = 2 pixels of RGB
            R8G8B8A8_UNORM,
            ImageUsage {
                transfer_dst: true,
                sampled: true,
                color_attachment: true,
                ..ImageUsage::empty()
            },
        )?;
        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                min_filter: Filter::Linear,
                mag_filter: Filter::Linear,
                ..Default::default()
            },
        )?;
        let desc_set_layout = pipeline.layout().set_layouts().get(0).unwrap();
        let desc_set = PersistentDescriptorSet::new(
            descriptor_set_allocator,
            desc_set_layout.clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                ImageView::new(src.clone(), ImageViewCreateInfo::from_image(&src))?,
                sampler,
            )],
        )?;
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
        allocator: &FastMemoryAllocator,
        cmdbuf_allocator: &StandardCommandBufferAllocator,
        buf: &[u8],
        after: impl GpuFuture,
        queue: Arc<Queue>,
        output: Arc<AttachmentImage>,
    ) -> Result<impl GpuFuture> {
        if queue.device() != &self.device
            || allocator.device() != &self.device
            || cmdbuf_allocator.device() != &self.device
        {
            return Err(anyhow!("Device mismatch"));
        }
        if let Some(after_queue) = after.queue() {
            if queue != after_queue {
                return Err(anyhow!("Queue mismatch"));
            }
        }
        // Submit the source image to GPU
        let subbuffer = unsafe {
            CpuAccessibleBuffer::uninitialized_array(
                allocator,
                buf.len() as u64,
                BufferUsage {
                    transfer_src: true,
                    ..BufferUsage::empty()
                },
                false,
            )
        }?;
        subbuffer.write()?.copy_from_slice(buf);
        let mut cmdbuf = AutoCommandBufferBuilder::primary(
            cmdbuf_allocator,
            queue.queue_family_index(),
            OneTimeSubmit,
        )?;
        cmdbuf.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            subbuffer,
            self.src.clone(),
        ))?;
        // Build a pipeline to do yuyv -> rgb
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
        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
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
            .begin_render_pass(render_pass_begin_info, SubpassContents::Inline)
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
