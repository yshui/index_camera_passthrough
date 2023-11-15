use anyhow::{anyhow, Result};
use smallvec::smallvec;
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferUsage::OneTimeSubmit, RenderPassBeginInfo, SubpassEndInfo,
    },
    command_buffer::{SubpassBeginInfo, SubpassContents},
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, persistent::PersistentDescriptorSet,
        WriteDescriptorSet,
    },
    device::{Device, DeviceOwned, Queue},
    image::sampler::{Filter, Sampler, SamplerCreateInfo},
    image::view::{ImageView, ImageViewCreateInfo},
    image::Image,
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
        layout::PipelineDescriptorSetLayoutCreateInfo,
        GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::GpuFuture,
    Handle, VulkanObject,
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

#[derive(VertexTrait, Default, Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Vertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

#[derive(thiserror::Error, Debug)]
pub enum ConverterError {
    #[error("something went wrong: {0}")]
    Anyhow(#[from] anyhow::Error),
}

pub struct GpuYuyvConverter {
    device: Arc<Device>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    desc_set: Arc<PersistentDescriptorSet>,
}

impl std::fmt::Debug for GpuYuyvConverter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuYuyvConverter")
            .field("device", &self.device.handle().as_raw())
            .field("render_pass", &self.render_pass.handle().as_raw())
            .field("pipeline", &self.pipeline.handle().as_raw())
            .finish_non_exhaustive()
    }
}

/// XXX: We can use VK_KHR_sampler_ycbcr_conversion for this, but I don't
/// know if it's widely supported. And the image format we need (G8B8G8R8_422_UNORM)
/// seems to have even less support than the extension itself.
impl GpuYuyvConverter {
    /// Create a new YUYV to RGBA8 converter.
    /// Note the input image's width has to be `w/2`, and `w` has to be even.
    pub fn new(
        device: Arc<Device>,
        descriptor_set_allocator: &StandardDescriptorSetAllocator,
        w: u32,
        h: u32,
        input: &Arc<Image>,
    ) -> Result<Self> {
        if w % 2 != 0 {
            return Err(anyhow!("Width can't be odd"));
        }
        let vs = vs::load(device.clone())?;
        let fs = fs::load(device.clone())?;
        let render_pass = vulkano::single_pass_renderpass!(device.clone(),
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
        )?;
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
        let pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                vertex_input_state: Some(
                    Vertex::per_vertex().definition(&vs.info().input_interface)?,
                ),
                stages: stages.into_iter().collect(),
                input_assembly_state: Some(InputAssemblyState {
                    topology: PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                }),
                viewport_state: Some(ViewportState {
                    viewports: smallvec![Viewport {
                        offset: [0.0, 0.0],
                        extent: [w as f32, h as f32],
                        depth_range: 0.0..=1.0,
                    }],
                    ..Default::default()
                }),
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
                ImageView::new(input.clone(), ImageViewCreateInfo::from_image(input))?,
                sampler,
            )],
            None,
        )?;
        Ok(Self {
            render_pass,
            pipeline,
            device,
            desc_set,
        })
    }
    /// Receives a buffer containing a YUYV image, convert it to RGBA8. Note the
    /// output image's width would be double that of the input image.
    ///
    /// Returns a GPU future representing the operation, and an image. You must
    /// make sure the previous conversion is completed before calling this
    /// function again.
    pub fn yuyv_buffer_to_vulkan_image(
        &self,
        allocator: Arc<dyn MemoryAllocator>,
        cmdbuf_allocator: &StandardCommandBufferAllocator,
        after: impl GpuFuture,
        queue: &Arc<Queue>,
        output: Arc<Image>,
    ) -> Result<impl GpuFuture> {
        if queue.device() != &self.device
            || allocator.device() != &self.device
            || cmdbuf_allocator.device() != &self.device
        {
            return Err(anyhow!("Device mismatch"));
        }
        if let Some(after_queue) = after.queue() {
            if queue != &after_queue {
                return Err(anyhow!("Queue mismatch"));
            }
        }
        let mut cmdbuf = AutoCommandBufferBuilder::primary(
            cmdbuf_allocator,
            queue.queue_family_index(),
            OneTimeSubmit,
        )?;
        // Build a pipeline to do yuyv -> rgb
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
            .begin_render_pass(
                render_pass_begin_info,
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .map_err(|e| ConverterError::Anyhow(e.into()))?
            .bind_pipeline_graphics(self.pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                self.desc_set.clone(),
            )?
            .bind_vertex_buffers(0, vertex_buffer.clone())?
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .map_err(|e| ConverterError::Anyhow(e.into()))?
            .end_render_pass(SubpassEndInfo::default())
            .map_err(|e| ConverterError::Anyhow(e.into()))?;
        Ok(after.then_execute(
            queue.clone(),
            cmdbuf
                .build()
                .map_err(|e| ConverterError::Anyhow(e.into()))?,
        )?)
    }
}
