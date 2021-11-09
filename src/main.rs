#![feature(untagged_unions, try_trait_v2)]

use anyhow::{anyhow, Context, Result};
use ash::vk::Handle;
use image::Pixel;
use std::sync::Arc;
use std::{marker::PhantomData, pin::Pin};
use vulkano::{
    command_buffer::PrimaryCommandBuffer,
    image::ImageAccess,
    instance::{Instance, InstanceExtensions, Version},
    sync::GpuFuture,
    SynchronizedVulkanObject, VulkanObject,
};

#[allow(unused_imports)]
use log::info;

static APP_KEY: &str = "index_camera_passthrough_rs\0";
static APP_NAME: &str = "Camera\0";

pub struct VRSystem(*mut openvr_sys::IVRSystem);

pub struct VRCompositor<'a>(
    *mut openvr_sys::IVRCompositor,
    PhantomData<&'a openvr_sys::IVRSystem>,
);

impl<'a> VRCompositor<'a> {
    pub fn pin_mut(&self) -> Pin<&mut openvr_sys::IVRCompositor> {
        unsafe { Pin::new_unchecked(&mut *self.0) }
    }
    pub fn required_extensions<'b>(
        &self,
        pdev: vulkano::device::physical::PhysicalDevice,
        buf: &'b mut Vec<u8>,
    ) -> impl Iterator<Item = &'b std::ffi::CStr> {
        let bytes_needed = unsafe {
            self.pin_mut().GetVulkanDeviceExtensionsRequired(
                std::mem::transmute(pdev.internal_object().as_raw()),
                std::ptr::null_mut(),
                0,
            )
        };
        buf.reserve(bytes_needed as usize);
        unsafe {
            self.pin_mut().GetVulkanDeviceExtensionsRequired(
                std::mem::transmute(pdev.internal_object().as_raw()),
                buf.as_mut_ptr() as *mut _,
                bytes_needed,
            );
            buf.set_len(bytes_needed as usize);
        };
        let () = buf
            .iter_mut()
            .map(|item| {
                if *item == b' ' {
                    *item = b'\0';
                }
            })
            .collect();
        buf.as_slice()
            .split_inclusive(|ch| *ch == b'\0')
            .map(|slice| unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(slice) })
    }
}

impl VRSystem {
    pub fn init() -> Result<Self> {
        let mut error = openvr_sys::EVRInitError::VRInitError_None;
        let isystem_raw = unsafe {
            openvr_sys::VR_Init(
                &mut error,
                openvr_sys::EVRApplicationType::VRApplication_Overlay,
                std::ptr::null(),
            )
        };
        error.into_result()?;
        Ok(Self(isystem_raw))
    }
    pub fn overlay<'a>(&'a self) -> VROverlay<'a> {
        VROverlay(openvr_sys::VROverlay(), PhantomData)
    }
    pub fn compositor<'a>(&'a self) -> VRCompositor<'a> {
        VRCompositor(openvr_sys::VRCompositor(), PhantomData)
    }
    pub fn pin_mut(&self) -> Pin<&mut openvr_sys::IVRSystem> {
        unsafe { Pin::new_unchecked(&mut *self.0) }
    }
}

pub struct VROverlay<'a>(
    *mut openvr_sys::IVROverlay,
    PhantomData<&'a openvr_sys::IVRSystem>,
);

impl<'a> VROverlay<'a> {
    pub fn pin_mut(&self) -> Pin<&mut openvr_sys::IVROverlay> {
        unsafe { Pin::new_unchecked(&mut *self.0) }
    }
    pub fn create_overlay(&'a self, key: &'a str, name: &'a str) -> Result<VROverlayHandle<'a>> {
        if !key.contains('\0') || !name.contains('\0') {
            return Err(anyhow!("key and name must both contain a NUL byte"));
        }
        let mut overlayhandle = std::mem::MaybeUninit::<openvr_sys::VROverlayHandle_t>::uninit();
        unsafe {
            self.pin_mut().CreateOverlay(
                key.as_bytes().as_ptr() as *const _,
                name.as_bytes().as_ptr() as *const _,
                overlayhandle.as_mut_ptr(),
            )
        }
        .into_result()?;
        Ok(VROverlayHandle {
            raw: unsafe { overlayhandle.assume_init() },
            ivr_overlay: self,
            texture: None,
        })
    }
    /// Safety: could destroy an overlay that is still owned by a VROverlayHandle.
    unsafe fn destroy_overlay_raw(&self, overlay: openvr_sys::VROverlayHandle_t) -> Result<()> {
        let error = self.pin_mut().DestroyOverlay(overlay);
        if error != openvr_sys::EVROverlayError::VROverlayError_None {
            Err(anyhow!("Failed to destroy overlay {:?}", error))
        } else {
            Ok(())
        }
    }
}

struct TextureState {
    _image: Arc<dyn vulkano::image::ImageAccess>,
    _device: Arc<vulkano::device::Device>,
    _queue: Arc<vulkano::device::Queue>,
    _instance: Arc<vulkano::instance::Instance>,
}
pub struct VROverlayHandle<'a> {
    raw: openvr_sys::VROverlayHandle_t,
    ivr_overlay: &'a VROverlay<'a>,

    /// Used to hold references to vulkan objects so they don't die.
    texture: Option<TextureState>,
}

impl<'a> VROverlayHandle<'a> {
    pub fn as_raw(&self) -> openvr_sys::VROverlayHandle_t {
        self.raw
    }
    pub fn set_texture(
        &mut self,
        w: u32,
        h: u32,
        image: Arc<impl vulkano::image::ImageAccess + 'static>,
        dev: Arc<vulkano::device::Device>,
        queue: Arc<vulkano::device::Queue>,
        instance: Arc<vulkano::instance::Instance>,
    ) -> Result<(), openvr_sys::EVROverlayError> {
        let texture = TextureState {
            _image: image.clone() as Arc<_>,
            _device: dev.clone(),
            _queue: queue.clone(),
            _instance: instance.clone(),
        };
        self.texture.replace(texture);
        let mut vrimage = openvr_sys::VRVulkanTextureData_t {
            m_nWidth: w,
            m_nHeight: h,
            m_nFormat: image.format() as u32,
            m_nSampleCount: image.samples() as u32,
            m_nImage: image.inner().image.internal_object().as_raw(),
            m_pPhysicalDevice: unsafe {
                std::mem::transmute(dev.physical_device().internal_object().as_raw())
            },
            m_pDevice: unsafe { std::mem::transmute(dev.internal_object().as_raw()) },
            m_pQueue: unsafe { std::mem::transmute(queue.internal_object_guard().as_raw()) },
            m_pInstance: unsafe { std::mem::transmute(instance.internal_object().as_raw()) },
            m_nQueueFamilyIndex: queue.family().id(),
        };
        let vrtexture = openvr_sys::Texture_t {
            handle: &mut vrimage as *mut _ as *mut std::ffi::c_void,
            eType: openvr_sys::ETextureType::TextureType_Vulkan,
            eColorSpace: openvr_sys::EColorSpace::ColorSpace_Auto,
        };
        unsafe {
            self.ivr_overlay
                .pin_mut()
                .SetOverlayTexture(self.as_raw(), &vrtexture)
                .into_result()
        }
    }
}

impl<'a> Drop for VROverlayHandle<'a> {
    fn drop(&mut self) {
        if let Err(e) = unsafe { self.ivr_overlay.destroy_overlay_raw(self.raw) } {
            eprintln!("{}", e.to_string());
        }
    }
}

impl Drop for VRSystem {
    fn drop(&mut self) {
        openvr_sys::VR_Shutdown();
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        r.store(false, std::sync::atomic::Ordering::Relaxed);
    })
    .expect("Error setting Ctrl-C handler");

    let backing_image = image::io::Reader::open("image.png")?
        .with_guessed_format()?
        .decode()?
        .to_rgba8();
    println!("Image loaded");
    let (w, h) = backing_image.dimensions();

    // Create vulkan instance, and setup openvr.
    // Then create a vulkan device based on openvr's requirements
    let instance = Instance::new(
        None,
        Version::V1_1,
        &InstanceExtensions::supported_by_core()?,
        None,
    )?;
    let vrsys = VRSystem::init()?;
    println!("{}", openvr_sys::VR_IsHmdPresent());
    let mut target_device = 0u64;
    unsafe {
        vrsys.pin_mut().GetOutputDevice(
            &mut target_device,
            openvr_sys::ETextureType::TextureType_Vulkan,
            std::mem::transmute(instance.internal_object().as_raw()),
        )
    };

    let target_device = ash::vk::PhysicalDevice::from_raw(target_device);
    let device = vulkano::device::physical::PhysicalDevice::enumerate(&instance)
        .find(|physical_device| {
            if physical_device.internal_object() == target_device {
                println!(
                    "Found matching device: {}",
                    physical_device.properties().device_name
                );
                true
            } else {
                false
            }
        })
        .with_context(|| anyhow!("Cannot find the device openvr asked for"))?;
    let queue_family = device
        .queue_families()
        .find(|qf| {
            qf.supports_graphics() && qf.supports_stage(vulkano::sync::PipelineStage::AllGraphics)
        })
        .with_context(|| anyhow!("Cannot create a suitable queue"))?;
    let (device, mut queues) = {
        let mut buf = Vec::new();
        let extensions = vulkano::device::DeviceExtensions::from(
            vrsys.compositor().required_extensions(device, &mut buf),
        )
        .union(&device.required_extensions());
        vulkano::device::Device::new(
            device,
            &vulkano::device::Features::none(),
            &extensions,
            [(queue_family, 1.0)],
        )?
    };
    let queue = queues.next().unwrap();
    let image = {
        // Submit the backing_image to GPU, and get a vulkan image back
        let image = vulkano::image::AttachmentImage::with_usage(
            device.clone(),
            [w, h],
            vulkano::format::Format::R8G8B8A8_UNORM,
            vulkano::image::ImageUsage {
                transfer_source: true,
                transfer_destination: true,
                sampled: true,
                storage: true,
                color_attachment: true,
                depth_stencil_attachment: false,
                transient_attachment: false,
                input_attachment: false,
            },
        )?;
        let buffer = vulkano::buffer::CpuBufferPool::upload(device.clone());
        let subbuffer = buffer.chunk(backing_image.pixels().map(|p| {
            let (a, b, c, d) = p.channels4();
            [a, b, c, d]
        }))?;
        let mut cmdbuf = vulkano::command_buffer::AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            vulkano::command_buffer::CommandBufferUsage::MultipleSubmit,
        )?;
        cmdbuf.copy_buffer_to_image(subbuffer, image.clone())?;
        cmdbuf
            .build()?
            .execute(queue.clone())?
            .then_signal_fence_and_flush()?
            .wait(None)?;
        image
    };

    // Create a VROverlay and submit our image as its texture
    let vroverlay = vrsys.overlay();
    let mut overlay = vroverlay.create_overlay(APP_KEY, APP_NAME)?;
    vroverlay
        .pin_mut()
        .SetOverlayFlag(
            overlay.as_raw(),
            openvr_sys::VROverlayFlags::VROverlayFlags_SideBySide_Parallel,
            true,
        )
        .into_result()?;
    overlay.set_texture(w, h, image, device.clone(), queue.clone(), instance.clone())?;
    let transformation = openvr_sys::HmdMatrix34_t {
        m: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, -1.0],
        ],
    };
    unsafe {
        vroverlay.pin_mut().SetOverlayTransformAbsolute(
            overlay.as_raw(),
            openvr_sys::ETrackingUniverseOrigin::TrackingUniverseStanding,
            &transformation,
        )
    };

    // Display the overlay
    vroverlay
        .pin_mut()
        .ShowOverlay(overlay.as_raw())
        .into_result()?;

    let mut event = std::mem::MaybeUninit::<openvr_sys::VREvent_t>::uninit();
    'main_loop: loop {
        while unsafe {
            vrsys.pin_mut().PollNextEvent(
                event.as_mut_ptr() as *mut _,
                std::mem::size_of::<openvr_sys::VREvent_t>() as u32,
            )
        } {
            let event = unsafe { event.assume_init_ref() };
            println!("{:?}", unsafe {
                std::mem::transmute::<_, openvr_sys::EVREventType>(event.eventType)
            });
            if event.eventType == openvr_sys::EVREventType::VREvent_ButtonPress as u32 {
                println!("{:?}", unsafe { event.data.controller.button });
                if unsafe { event.data.controller.button == 33 } {
                    break 'main_loop;
                }
            } else if event.eventType == openvr_sys::EVREventType::VREvent_Quit as u32 {
                vrsys.pin_mut().AcknowledgeQuit_Exiting();
                break 'main_loop;
            }
        }
        if !running.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    Ok(())
}
