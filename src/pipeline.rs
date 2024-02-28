use std::sync::Arc;

use crate::utils::DeviceExt as _;
use anyhow::Result;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::CommandBufferAllocator, CommandBufferBeginInfo, CommandBufferLevel,
        CommandBufferUsage, CopyBufferToImageInfo, RecordingCommandBuffer,
    },
    descriptor_set::allocator::DescriptorSetAllocator,
    device::{Device, DeviceOwned},
    format::Format,
    image::{Image as VkImage, ImageCreateInfo, ImageUsage},
    memory::allocator::{MemoryAllocator, MemoryTypeFilter},
    sync::GpuFuture,
    Handle, VulkanObject,
};

pub(crate) struct Pipeline {
    yuv: Option<crate::yuv::GpuYuyvConverter>,
    correction: Option<crate::distortion_correction::StereoCorrection>,
    capture: bool,
    render_doc: Option<renderdoc::RenderDoc<renderdoc::V100>>,
    cpu_image_buffer: Arc<Buffer>,
    yuv_texture: Arc<VkImage>,
    textures: [Arc<VkImage>; 2],
    camera_config: Option<crate::vrapi::StereoCamera>,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("yuv", &self.yuv)
            .field("correction", &self.correction)
            .field("capture", &self.capture)
            .field("render_doc", &self.render_doc)
            .field("yuv_texture", &self.yuv_texture.handle().as_raw())
            .field(
                "textures",
                &self.textures.each_ref().map(|t| t.handle().as_raw()),
            )
            .field("camera_config", &self.camera_config)
            .finish_non_exhaustive()
    }
}

use crate::CAMERA_SIZE;

enum EitherGpuFuture<L, R> {
    Left(L),
    Right(R),
}

unsafe impl<L: DeviceOwned, R: DeviceOwned> DeviceOwned for EitherGpuFuture<L, R> {
    fn device(&self) -> &Arc<Device> {
        match self {
            EitherGpuFuture::Left(l) => l.device(),
            EitherGpuFuture::Right(r) => r.device(),
        }
    }
}

unsafe impl<L: GpuFuture, R: GpuFuture> GpuFuture for EitherGpuFuture<L, R> {
    #[inline]
    unsafe fn build_submission(
        &self,
    ) -> std::result::Result<
        vulkano::sync::future::SubmitAnyBuilder,
        vulkano::Validated<vulkano::VulkanError>,
    > {
        match self {
            EitherGpuFuture::Left(l) => l.build_submission(),
            EitherGpuFuture::Right(r) => r.build_submission(),
        }
    }

    #[inline]
    fn check_buffer_access(
        &self,
        buffer: &Buffer,
        range: std::ops::Range<vulkano::DeviceSize>,
        exclusive: bool,
        queue: &vulkano::device::Queue,
    ) -> std::result::Result<(), vulkano::sync::future::AccessCheckError> {
        match self {
            EitherGpuFuture::Left(l) => l.check_buffer_access(buffer, range, exclusive, queue),
            EitherGpuFuture::Right(r) => r.check_buffer_access(buffer, range, exclusive, queue),
        }
    }

    #[inline]
    fn cleanup_finished(&mut self) {
        match self {
            EitherGpuFuture::Left(l) => l.cleanup_finished(),
            EitherGpuFuture::Right(r) => r.cleanup_finished(),
        }
    }

    #[inline]
    fn flush(&self) -> std::result::Result<(), vulkano::Validated<vulkano::VulkanError>> {
        match self {
            EitherGpuFuture::Left(l) => l.flush(),
            EitherGpuFuture::Right(r) => r.flush(),
        }
    }

    #[inline]
    unsafe fn signal_finished(&self) {
        match self {
            EitherGpuFuture::Left(l) => l.signal_finished(),
            EitherGpuFuture::Right(r) => r.signal_finished(),
        }
    }

    #[inline]
    fn queue(&self) -> Option<Arc<vulkano::device::Queue>> {
        match self {
            EitherGpuFuture::Left(l) => l.queue(),
            EitherGpuFuture::Right(r) => r.queue(),
        }
    }

    #[inline]
    fn queue_change_allowed(&self) -> bool {
        match self {
            EitherGpuFuture::Left(l) => l.queue_change_allowed(),
            EitherGpuFuture::Right(r) => r.queue_change_allowed(),
        }
    }

    #[inline]
    fn check_image_access(
        &self,
        image: &vulkano::image::Image,
        range: std::ops::Range<vulkano::DeviceSize>,
        exclusive: bool,
        expected_layout: vulkano::image::ImageLayout,
        queue: &vulkano::device::Queue,
    ) -> std::result::Result<(), vulkano::sync::future::AccessCheckError> {
        match self {
            EitherGpuFuture::Left(l) => {
                l.check_image_access(image, range, exclusive, expected_layout, queue)
            }
            EitherGpuFuture::Right(r) => {
                r.check_image_access(image, range, exclusive, expected_layout, queue)
            }
        }
    }

    #[inline]
    fn check_swapchain_image_acquired(
        &self,
        swapchain: &vulkano::swapchain::Swapchain,
        image_index: u32,
        before: bool,
    ) -> std::result::Result<(), vulkano::sync::future::AccessCheckError> {
        match self {
            EitherGpuFuture::Left(l) => {
                l.check_swapchain_image_acquired(swapchain, image_index, before)
            }
            EitherGpuFuture::Right(r) => {
                r.check_swapchain_image_acquired(swapchain, image_index, before)
            }
        }
    }
}

impl Pipeline {
    pub(crate) fn submit_cpu_image(
        &self,
        img: &[u8],
        cmdbuf_allocator: Arc<dyn CommandBufferAllocator>,
        queue: &Arc<vulkano::device::Queue>,
        output: Arc<VkImage>,
    ) -> Result<impl GpuFuture> {
        let buffer = Subbuffer::new(self.cpu_image_buffer.clone()).slice(0..img.len() as u64);
        buffer.write()?.copy_from_slice(img);
        let mut cmdbuf = RecordingCommandBuffer::new(
            cmdbuf_allocator,
            queue.queue_family_index(),
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                ..Default::default()
            },
        )?;
        cmdbuf.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(buffer, output))?;
        Ok(cmdbuf.end()?.execute(queue.clone())?)
    }

    /// Create post-processing stages
    ///
    /// Camera data -> upload -> internal texture
    /// internal texture -> YUYV conversion -> textures[0]
    /// textures[0] -> Lens correction -> textures[1]
    /// textures[1] -> projection -> Final output
    pub(crate) fn new(
        device: Arc<Device>,
        allocator: Arc<dyn MemoryAllocator>,
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
        source_is_yuv: bool,
        camera_config: Option<crate::vrapi::StereoCamera>,
    ) -> Result<Self> {
        let render_doc = renderdoc::RenderDoc::new().ok();
        if render_doc.is_some() {
            log::info!("RenderDoc loaded");
        }
        // Allocate intermediate textures
        let yuv_texture = device.clone().new_image(
            ImageCreateInfo {
                extent: [CAMERA_SIZE, CAMERA_SIZE, 1],
                format: Format::R8G8B8A8_UNORM,
                usage: ImageUsage::TRANSFER_DST
                    | ImageUsage::TRANSFER_SRC
                    | ImageUsage::SAMPLED
                    | ImageUsage::COLOR_ATTACHMENT,
                ..Default::default()
            },
            MemoryTypeFilter::HOST_SEQUENTIAL_WRITE | MemoryTypeFilter::PREFER_DEVICE,
        )?;
        device.set_debug_utils_object_name(&yuv_texture, Some("yuv_texture"))?;
        let textures = [0, 1].try_map(|id| {
            let tex = device.clone().new_image(
                ImageCreateInfo {
                    extent: [CAMERA_SIZE * 2, CAMERA_SIZE, 1],
                    format: Format::R8G8B8A8_UNORM,
                    usage: ImageUsage::SAMPLED | ImageUsage::COLOR_ATTACHMENT,
                    ..Default::default()
                },
                MemoryTypeFilter::PREFER_DEVICE,
            )?;
            device.set_debug_utils_object_name(&tex, Some(&format!("texture{id}")))?;
            anyhow::Ok(tex)
        })?;
        // if source is YUV: upload -> yuv_texture -> converter -> output_a
        // if source is RGB: upload -> output_a
        let converter = source_is_yuv
            .then(|| {
                crate::yuv::GpuYuyvConverter::new(
                    device.clone(),
                    descriptor_set_allocator.clone(),
                    CAMERA_SIZE * 2,
                    CAMERA_SIZE,
                    &yuv_texture,
                )
            })
            .transpose()?;
        // if correction is enabled: output_a -> correction -> textures[1]
        // otherwise: output_a
        let correction = camera_config
            .map(|cfg| {
                crate::distortion_correction::StereoCorrection::new(
                    device.clone(),
                    allocator,
                    descriptor_set_allocator,
                    textures[0].clone(),
                    &cfg,
                )
            })
            .transpose()?;
        let cpu_buffer = device.clone().new_buffer(
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                // This should be more than enough. Camera sources are YUV subsampled,
                // so it won't be 4 bytes per pixel.
                size: CAMERA_SIZE as u64 * CAMERA_SIZE as u64 * 2 * 4,
                ..Default::default()
            },
            MemoryTypeFilter::HOST_SEQUENTIAL_WRITE | MemoryTypeFilter::PREFER_DEVICE,
        )?;
        log::debug!("correction fov: {:?}", correction.as_ref().map(|x| x.fov()));
        let fov = correction
            .as_ref()
            .map(|c| c.fov())
            .unwrap_or([[1.19; 2]; 2]); // default to roughly 100 degrees fov, hopefully this is sensible
        log::info!("Adjusted FOV: {:?}", fov);
        Ok(Self {
            correction,
            yuv: converter,
            capture: false,
            render_doc,
            textures,
            yuv_texture,
            camera_config,
            cpu_image_buffer: cpu_buffer,
        })
    }
    pub fn fov(&self) -> [[f32; 2]; 2] {
        self.correction
            .as_ref()
            .map(|c| c.fov())
            .unwrap_or([[1.19; 2]; 2])
    }
    /// Run the pipeline
    ///
    /// # Arguments
    ///
    /// - time: Time offset into the past when the camera frame is captured
    pub(crate) fn run(
        &mut self,
        queue: &Arc<vulkano::device::Queue>,
        allocator: Arc<dyn MemoryAllocator>,
        cmdbuf_allocator: Arc<dyn CommandBufferAllocator>,
        input: &[u8],
        output: Arc<vulkano::image::Image>,
    ) -> Result<impl GpuFuture> {
        if self.capture {
            if let Some(rd) = self.render_doc.as_mut() {
                log::info!("Start Capture");
                rd.start_frame_capture(std::ptr::null(), std::ptr::null());
            }
        }

        // 1. submit image to GPU
        // 2. convert YUYV to RGB
        let texture = if self.correction.is_some() {
            self.textures[0].clone()
        } else {
            output.clone()
        };
        let future = if let Some(converter) = &self.yuv {
            let future = self.submit_cpu_image(
                input,
                cmdbuf_allocator.clone(),
                queue,
                self.yuv_texture.clone(),
            )?;
            let future = converter.yuyv_buffer_to_vulkan_image(
                allocator.clone(),
                cmdbuf_allocator.clone(),
                future,
                queue,
                texture.clone(),
            )?;
            EitherGpuFuture::Left(future)
        } else {
            let future =
                self.submit_cpu_image(input, cmdbuf_allocator.clone(), queue, texture.clone())?;
            EitherGpuFuture::Right(future)
        };
        future.flush()?;
        // 3. lens correction
        let future = if let Some(correction) = &self.correction {
            let mut future = correction.correct(
                cmdbuf_allocator,
                allocator.clone(),
                future,
                queue,
                output.clone(),
            )?;
            future.flush()?;
            future.cleanup_finished();
            EitherGpuFuture::Left(future)
        } else {
            EitherGpuFuture::Right(future)
        };
        // TODO combine correction and projection

        if self.capture {
            if let Some(rd) = self.render_doc.as_mut() {
                log::info!("End Capture");
                rd.end_frame_capture(std::ptr::null(), std::ptr::null());
            }
            self.capture = false;
        }
        Ok(future)
    }
    pub(crate) fn capture_next_frame(&mut self) {
        self.capture = true;
    }
}
