use anyhow::{anyhow, Result};
use openvr_sys2::{EVRInitError, EVROverlayError};
use std::cell::Cell;
use std::sync::Arc;
/// Incomplete set of OpenVR wrappeprs.
use std::{marker::PhantomData, pin::Pin};
use vulkano::{
    device::{physical::PhysicalDevice, Device},
    Handle, VulkanObject,
};

pub struct VRSystem(*mut openvr_sys2::IVRSystem, Cell<Option<Arc<Device>>>);
pub struct VRCompositor<'a>(
    *mut openvr_sys2::IVRCompositor,
    PhantomData<&'a openvr_sys2::IVRSystem>,
);

impl<'a> VRCompositor<'a> {
    pub fn pin_mut(&self) -> Pin<&mut openvr_sys2::IVRCompositor> {
        unsafe { Pin::new_unchecked(&mut *self.0) }
    }
    pub fn required_extensions<'b>(
        &self,
        pdev: &PhysicalDevice,
        buf: &'b mut Vec<u8>,
    ) -> impl Iterator<Item = &'b std::ffi::CStr> {
        let bytes_needed = unsafe {
            self.pin_mut().GetVulkanDeviceExtensionsRequired(
                pdev.handle().as_raw() as *mut _,
                std::ptr::null_mut(),
                0,
            )
        };
        buf.reserve(bytes_needed as usize);
        unsafe {
            self.pin_mut().GetVulkanDeviceExtensionsRequired(
                pdev.handle().as_raw() as *mut _,
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
    pub fn init() -> Result<Self, EVRInitError> {
        let mut error = openvr_sys2::EVRInitError::VRInitError_None;
        let isystem_raw = unsafe {
            openvr_sys2::VR_Init(
                &mut error,
                openvr_sys2::EVRApplicationType::VRApplication_Overlay,
                std::ptr::null(),
            )
        };
        error.into_result()?;
        Ok(Self(isystem_raw, Cell::new(None)))
    }
    pub fn overlay(&self) -> VROverlay<'_> {
        VROverlay(openvr_sys2::VROverlay(), self)
    }
    pub fn compositor(&self) -> VRCompositor<'_> {
        VRCompositor(openvr_sys2::VRCompositor(), PhantomData)
    }
    pub fn hold_vulkan_device(&self, device: Arc<Device>) {
        self.1.replace(Some(device));
    }
    pub fn find_hmd(&self) -> Option<u32> {
        (0..64).find(|&i| {
            self.pin_mut().GetTrackedDeviceClass(i)
                == openvr_sys2::ETrackedDeviceClass::TrackedDeviceClass_HMD
        })
    }
    pub fn hmd_transform(&self, time_offset: f32) -> nalgebra::Matrix4<f64> {
        let mut hmd_transform = std::mem::MaybeUninit::<openvr_sys2::TrackedDevicePose_t>::uninit();
        unsafe {
            self.pin_mut().GetDeviceToAbsoluteTrackingPose(
                openvr_sys2::ETrackingUniverseOrigin::TrackingUniverseStanding,
                time_offset,
                hmd_transform.as_mut_ptr(),
                1,
            );
            (&hmd_transform.assume_init().mDeviceToAbsoluteTracking).into()
        }
    }
    pub fn pin_mut(&self) -> Pin<&mut openvr_sys2::IVRSystem> {
        unsafe { Pin::new_unchecked(&mut *self.0) }
    }
}

pub struct VROverlay<'a>(*mut openvr_sys2::IVROverlay, &'a VRSystem);

impl<'a> VROverlay<'a> {
    pub fn pin_mut(&self) -> Pin<&mut openvr_sys2::IVROverlay> {
        unsafe { Pin::new_unchecked(&mut *self.0) }
    }
    pub fn create_overlay(
        &'a self,
        key: &'a str,
        name: &'a str,
    ) -> Result<openvr_sys2::VROverlayHandle_t, EVROverlayError> {
        if !key.contains('\0') || !name.contains('\0') {
            return Err(EVROverlayError::VROverlayError_InvalidParameter);
        }
        let mut overlayhandle = std::mem::MaybeUninit::<openvr_sys2::VROverlayHandle_t>::uninit();
        unsafe {
            self.pin_mut().CreateOverlay(
                key.as_bytes().as_ptr() as *const _,
                name.as_bytes().as_ptr() as *const _,
                overlayhandle.as_mut_ptr(),
            )
        }
        .into_result()?;
        Ok(unsafe { overlayhandle.assume_init() })
    }
    /// Safety: could destroy an overlay that is still owned by a VROverlayHandle.
    pub unsafe fn destroy_overlay_raw(
        &self,
        overlay: openvr_sys2::VROverlayHandle_t,
    ) -> Result<()> {
        let error = self.pin_mut().DestroyOverlay(overlay);
        if error != openvr_sys2::EVROverlayError::VROverlayError_None {
            Err(anyhow!("Failed to destroy overlay {:?}", error))
        } else {
            Ok(())
        }
    }
}

impl Drop for VRSystem {
    fn drop(&mut self) {
        log::info!("Shutdown OpenVR");
        openvr_sys2::VR_Shutdown();
    }
}
