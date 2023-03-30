use anyhow::{anyhow, Result};
use std::cell::Cell;
use std::sync::Arc;
/// Incomplete set of OpenVR wrappeprs.
use std::{marker::PhantomData, pin::Pin};
use vulkano::{
    device::{physical::PhysicalDevice, Device, Queue},
    image::ImageAccess,
    instance::Instance,
    Handle, VulkanObject,
};

pub struct VRSystem(*mut openvr_sys::IVRSystem, Cell<Option<Arc<Device>>>);
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
        pdev: &PhysicalDevice,
        buf: &'b mut Vec<u8>,
    ) -> impl Iterator<Item = &'b std::ffi::CStr> {
        let bytes_needed = unsafe {
            self.pin_mut().GetVulkanDeviceExtensionsRequired(
                std::mem::transmute(pdev.handle().as_raw()),
                std::ptr::null_mut(),
                0,
            )
        };
        buf.reserve(bytes_needed as usize);
        unsafe {
            self.pin_mut().GetVulkanDeviceExtensionsRequired(
                std::mem::transmute(pdev.handle().as_raw()),
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
        Ok(Self(isystem_raw, Cell::new(None)))
    }
    pub fn overlay(&self) -> VROverlay<'_> {
        VROverlay(openvr_sys::VROverlay(), self)
    }
    pub fn compositor(&self) -> VRCompositor<'_> {
        VRCompositor(openvr_sys::VRCompositor(), PhantomData)
    }
    fn hold_vulkan_device(&self, device: Arc<Device>) {
        self.1.replace(Some(device));
    }
    #[allow(dead_code)]
    pub fn find_hmd(&self) -> Option<u32> {
        for i in 0..64 {
            if self.pin_mut().GetTrackedDeviceClass(i)
                == openvr_sys::ETrackedDeviceClass::TrackedDeviceClass_HMD
            {
                return Some(i);
            }
        }
        None
    }
    pub fn hmd_transform(&self, time_offset: f32) -> nalgebra::Matrix4<f64> {
        let mut hmd_transform = std::mem::MaybeUninit::<openvr_sys::TrackedDevicePose_t>::uninit();
        unsafe {
            self.pin_mut().GetDeviceToAbsoluteTrackingPose(
                openvr_sys::ETrackingUniverseOrigin::TrackingUniverseStanding,
                time_offset,
                hmd_transform.as_mut_ptr(),
                1,
            );
            (&hmd_transform.assume_init().mDeviceToAbsoluteTracking).into()
        }
    }
    pub fn pin_mut(&self) -> Pin<&mut openvr_sys::IVRSystem> {
        unsafe { Pin::new_unchecked(&mut *self.0) }
    }
}

pub struct VROverlay<'a>(*mut openvr_sys::IVROverlay, &'a VRSystem);

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
    _device: Arc<Device>,
    _queue: Arc<Queue>,
    _instance: Arc<Instance>,
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
        dev: Arc<Device>,
        queue: Arc<Queue>,
        instance: Arc<Instance>,
    ) -> Result<(), openvr_sys::EVROverlayError> {
        let texture = TextureState {
            _image: image.clone() as Arc<_>,
            _device: dev.clone(),
            _queue: queue.clone(),
            _instance: instance.clone(),
        };
        self.texture.replace(texture);
        // Once we set a texture, the VRSystem starts to depend on Vulkan
        // instance being alive.
        self.ivr_overlay.1.hold_vulkan_device(dev.clone());
        let mut vrimage = openvr_sys::VRVulkanTextureData_t {
            m_nWidth: w,
            m_nHeight: h,
            m_nFormat: image.format() as u32,
            m_nSampleCount: image.samples() as u32,
            m_nImage: image.inner().image.handle().as_raw(),
            m_pPhysicalDevice: unsafe {
                std::mem::transmute(dev.physical_device().handle().as_raw())
            },
            m_pDevice: unsafe { std::mem::transmute(dev.handle().as_raw()) },
            m_pQueue: unsafe { std::mem::transmute(queue.handle().as_raw()) },
            m_pInstance: unsafe { std::mem::transmute(instance.handle().as_raw()) },
            m_nQueueFamilyIndex: queue.queue_family_index(),
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
        log::info!("Dropping overlay handle");
        if let Err(e) = unsafe { self.ivr_overlay.destroy_overlay_raw(self.raw) } {
            eprintln!("{}", e);
        }
    }
}

impl Drop for VRSystem {
    fn drop(&mut self) {
        log::info!("Shutdown OpenVR");
        openvr_sys::VR_Shutdown();
    }
}
