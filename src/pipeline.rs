use std::sync::Arc;

use anyhow::Result;
use nalgebra::Matrix4;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::{CommandBufferAllocator, StandardCommandBufferAllocator},
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryCommandBufferAbstract,
    },
    descriptor_set::allocator::{StandardDescriptorSetAlloc, StandardDescriptorSetAllocator},
    device::{Device, DeviceOwned},
    format::Format,
    image::{AttachmentImage, ImageAccess, ImageUsage},
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocator, MemoryUsage, StandardMemoryAllocator,
    },
    sync::GpuFuture,
    Handle, VulkanObject,
};

pub(crate) struct Pipeline {
    yuv: Option<crate::yuv::GpuYuyvConverter>,
    correction: Option<crate::distortion_correction::StereoCorrection>,
    projection: Option<crate::projection::Projection<StandardDescriptorSetAlloc>>,
    projection_params: Option<crate::projection::ProjectionParameters>,
    capture: bool,
    render_doc: Option<renderdoc::RenderDoc<renderdoc::V100>>,
    cmdbuf_allocator: StandardCommandBufferAllocator,
    memory_allocator: StandardMemoryAllocator,
    yuv_texture: Arc<AttachmentImage>,
    textures: [Arc<AttachmentImage>; 2],
    ipd: f32,
    camera_config: Option<crate::vrapi::StereoCamera>,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("yuv", &self.yuv)
            .field("correction", &self.correction)
            .field("projection", &self.projection)
            .field("projection_params", &self.projection_params)
            .field("capture", &self.capture)
            .field("render_doc", &self.render_doc)
            .field(
                "yuv_texture",
                &self.yuv_texture.inner().image.handle().as_raw(),
            )
            .field(
                "textures",
                &self
                    .textures
                    .each_ref()
                    .map(|t| t.inner().image.handle().as_raw()),
            )
            .field("ipd", &self.ipd)
            .field("camera_config", &self.camera_config)
            .finish_non_exhaustive()
    }
}

use crate::{config::DisplayMode, CAMERA_SIZE};

pub(crate) fn submit_cpu_image(
    img: &[u8],
    cmdbuf_allocator: &impl CommandBufferAllocator,
    allocator: &impl MemoryAllocator,
    queue: &Arc<vulkano::device::Queue>,
    output: &Arc<AttachmentImage>,
) -> Result<impl GpuFuture> {
    let buffer = Buffer::new_slice::<u8>(
        allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            allocate_preference: vulkano::memory::allocator::MemoryAllocatePreference::Unknown,
            ..Default::default()
        },
        img.len() as u64,
    )?;
    buffer.write()?.copy_from_slice(&img);
    let mut cmdbuf = AutoCommandBufferBuilder::primary(
        cmdbuf_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;
    cmdbuf.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(buffer, output.clone()))?;
    Ok(cmdbuf.build()?.execute(queue.clone())?)
}

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
    ) -> std::result::Result<vulkano::sync::future::SubmitAnyBuilder, vulkano::sync::FlushError>
    {
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
    fn flush(&self) -> std::result::Result<(), vulkano::sync::FlushError> {
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
        image: &vulkano::image::sys::Image,
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
    /// Create post-processing stages
    ///
    /// Camera data -> upload -> internal texture
    /// internal texture -> YUYV conversion -> textures[0]
    /// textures[0] -> Lens correction -> textures[1]
    /// textures[1] -> projection -> Final output
    pub(crate) fn new(
        device: Arc<Device>,
        source_is_yuv: bool,
        display_mode: DisplayMode,
        ipd: f32,
        camera_config: Option<crate::vrapi::StereoCamera>,
    ) -> Result<Self> {
        log::info!("IPD: {}", ipd);
        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
        let allocator = StandardMemoryAllocator::new_default(device.clone());
        // Allocate intermediate textures
        let yuv_texture = AttachmentImage::with_usage(
            &allocator,
            [CAMERA_SIZE, CAMERA_SIZE],
            Format::R8G8B8A8_UNORM,
            ImageUsage::TRANSFER_DST
                | ImageUsage::TRANSFER_SRC
                | ImageUsage::SAMPLED
                | ImageUsage::COLOR_ATTACHMENT,
        )?;
        let textures = [0, 1].try_map(|_| {
            AttachmentImage::with_usage(
                &allocator,
                [CAMERA_SIZE * 2, CAMERA_SIZE],
                Format::R8G8B8A8_UNORM,
                ImageUsage::SAMPLED | ImageUsage::COLOR_ATTACHMENT,
            )
        })?;
        // if source is YUV: upload -> yuv_texture -> converter -> output_a
        // if source is RGB: upload -> output_a
        let converter = source_is_yuv
            .then(|| {
                crate::yuv::GpuYuyvConverter::new(
                    device.clone(),
                    &descriptor_set_allocator,
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
                    &allocator,
                    &descriptor_set_allocator,
                    textures[0].clone(),
                    &cfg,
                )
            })
            .transpose()?;
        let (projector, projection_mode) =
            if let DisplayMode::Stereo { projection_mode } = display_mode {
                let texture = if correction.is_some() {
                    &textures[1]
                } else {
                    &textures[0]
                };
                (
                    Some(crate::projection::Projection::new(
                        device.clone(),
                        &allocator,
                        &descriptor_set_allocator,
                        texture,
                        &camera_config,
                    )?),
                    Some(projection_mode),
                )
            } else {
                (None, None)
            };
        let projection_params =
            projection_mode.map(|mode| crate::projection::ProjectionParameters {
                ipd,
                overlay_width: 1.0,
                mvps: [Matrix4::identity(), Matrix4::identity()],
                camera_calib: camera_config,
                mode,
            });
        let fov = correction
            .as_ref()
            .map(|c| c.fov())
            .unwrap_or([[1.19; 2]; 2]); // default to roughly 100 degrees fov, hopefully this is sensible
        log::info!("Adjusted FOV: {:?}", fov);
        let render_doc = renderdoc::RenderDoc::new().ok();
        if render_doc.is_some() {
            log::info!("RenderDoc loaded");
        }
        Ok(Self {
            //projection: projector,
            //projection_params,
            //correction,
            projection: None,
            projection_params: None,
            correction: None,
            yuv: converter,
            capture: false,
            render_doc,
            cmdbuf_allocator: StandardCommandBufferAllocator::new(
                device.clone(),
                Default::default(),
            ),
            memory_allocator: allocator,
            textures,
            yuv_texture,
            ipd,
            camera_config,
        })
    }
    /// Run the pipeline
    ///
    /// # Arguments
    ///
    /// - time: Time offset into the past when the camera frame is captured
    pub(crate) fn run(
        &mut self,
        eye_to_head: &[Matrix4<f64>; 2],
        hmd_transform: &Matrix4<f64>,
        overlay_transform: &Matrix4<f64>,
        queue: &Arc<vulkano::device::Queue>,
        input: &[u8],
        output: &Arc<AttachmentImage>,
    ) -> Result<impl GpuFuture> {
        if self.capture {
            if let Some(rd) = self.render_doc.as_mut() {
                log::info!("Start Capture");
                rd.start_frame_capture(std::ptr::null(), std::ptr::null());
            }
        }

        // 1. submit image to GPU
        // 2. convert YUYV to RGB
        let texture = if self.projection.is_some() || self.correction.is_some() {
            &self.textures[0]
        } else {
            &output
        };
        let future = if let Some(converter) = &self.yuv {
            let future = submit_cpu_image(
                input,
                &self.cmdbuf_allocator,
                &self.memory_allocator,
                queue,
                &self.yuv_texture,
            )?;
            let future = converter.yuyv_buffer_to_vulkan_image(
                &self.memory_allocator,
                &self.cmdbuf_allocator,
                future,
                queue,
                texture,
            )?;
            EitherGpuFuture::Left(future)
        } else {
            let future = submit_cpu_image(
                input,
                &self.cmdbuf_allocator,
                &self.memory_allocator,
                queue,
                texture,
            )?;
            EitherGpuFuture::Right(future)
        };
        // TODO combine correction and projection
        // 3. lens correction
        let future = if let Some(correction) = &self.correction {
            let future = correction.correct(
                &self.cmdbuf_allocator,
                &self.memory_allocator,
                future,
                queue,
                if self.projection.is_some() {
                    &self.textures[1]
                } else {
                    output
                },
            )?;
            EitherGpuFuture::Left(future)
        } else {
            EitherGpuFuture::Right(future)
        };
        // 4. projection
        let future = if let Some(projector) = self.projection.as_mut() {
            let projection_params = self.projection_params.as_mut().unwrap();
            let fov = self
                .correction
                .as_ref()
                .map(|c| c.fov())
                .unwrap_or([[1.19; 2]; 2]);
            // Finally apply projection
            // Calculate each eye's Model View Project matrix at the moment the current frame is taken
            let (l, r) = projector.calculate_mvp(
                projection_params.mode,
                &overlay_transform,
                &self.camera_config,
                &fov,
                eye_to_head,
                hmd_transform,
            );
            projection_params.mvps = [l, r];
            projection_params.ipd = self.ipd;
            projector.set_params(projection_params)?;
            EitherGpuFuture::Left(projector.project(
                &self.memory_allocator,
                &self.cmdbuf_allocator,
                future,
                queue,
                output,
            )?)
        } else {
            EitherGpuFuture::Right(future)
        };

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

    pub(crate) fn set_ipd(&mut self, ipd: f32) {
        self.ipd = ipd;
    }
}
