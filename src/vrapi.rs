use nalgebra::Matrix4;
use openvr_sys2::{ETrackedPropertyError, EVRInitError, EVRInputError, EVROverlayError};
use std::{ffi::CString, mem::MaybeUninit, pin::Pin, sync::Arc};
use vulkano::{
    device::{physical::PhysicalDevice, Device, Queue},
    instance::Instance,
    Handle, VulkanObject,
};

use serde::{Deserialize, Serialize};

use crate::{APP_KEY, APP_NAME};
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub struct Extrinsics {
    /// Offset of the camera from Hmd
    pub position: [f64; 3],
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub struct Distort {
    pub coeffs: [f64; 4],
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub struct Intrinsics {
    /// Optical center X
    pub center_x: f64,
    /// Optical center Y
    pub center_y: f64,
    /// X focal length in device pixels
    pub focal_x: f64,
    /// Y focal length in device pixels
    pub focal_y: f64,
    /// Height of the camera output in pixels
    pub height: f64,
    /// Width of the camera output in pixels
    pub width: f64,
    pub distort: Distort,
}
#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Copy, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Camera {
    Left,
    Right,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub struct TrackedCamera {
    pub extrinsics: Extrinsics,
    pub intrinsics: Intrinsics,
    pub name: Camera,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub struct StereoCamera {
    pub left: TrackedCamera,
    pub right: TrackedCamera,
}
pub struct Bounds {
    pub umin: f32,
    pub vmin: f32,
    pub umax: f32,
    pub vmax: f32,
}

pub enum Event {
    /// The VR API asks us to exit
    RequestExit,
    /// IPD has changed
    IpdChanged(f32),
}

pub enum Action {
    Button1 = 0,
    Button2 = 1,
    Debug = 2,
}

pub(crate) trait Vr {
    type Error: std::error::Error + Send + Sync + 'static;
    fn hmd_transform(&self, time_offset: f32) -> nalgebra::Matrix4<f64>;
    fn required_extensions(&self, pdev: &PhysicalDevice) -> vulkano::device::DeviceExtensions;
    fn target_device(&self, instance: &Arc<Instance>) -> Result<Arc<PhysicalDevice>, Self::Error>;
    fn load_camera_paramter(&self) -> Option<StereoCamera>;
    fn ipd(&self) -> Result<f32, Self::Error>;
    fn eye_to_head(&self) -> [Matrix4<f64>; 2];
    fn set_overlay_texture(
        &mut self,
        w: u32,
        h: u32,
        image: Arc<impl vulkano::image::ImageAccess + 'static>,
        dev: Arc<Device>,
        queue: Arc<Queue>,
        instance: Arc<Instance>,
    ) -> Result<(), Self::Error>;
    fn set_overlay_texture_bounds(&mut self, bounds: Bounds) -> Result<(), Self::Error>;
    fn set_overlay_stereo(&mut self, stereo: bool) -> Result<(), Self::Error>;
    fn show_overlay(&mut self) -> Result<(), Self::Error>;
    fn hide_overlay(&mut self) -> Result<(), Self::Error>;
    fn set_overlay_transformation(&mut self, transform: Matrix4<f64>) -> Result<(), Self::Error>;
    fn acknowledge_quit(&mut self);
    fn poll_next_event(&mut self) -> Option<Event>;
    fn update_action_state(&mut self) -> Result<(), Self::Error>;
    fn get_action_state(&self, action: Action) -> Result<bool, Self::Error>;
}

struct TextureState {
    _image: Arc<dyn vulkano::image::ImageAccess>,
    _device: Arc<Device>,
    _queue: Arc<Queue>,
    _instance: Arc<Instance>,
}

pub(crate) struct OpenVr {
    sys: crate::openvr::VRSystem,
    handle: openvr_sys2::VROverlayHandle_t,
    buttons: [openvr_sys2::VRActionHandle_t; 3],
    action_set: openvr_sys2::VRActionSetHandle_t,
    texture: Option<TextureState>,
}
impl OpenVr {
    pub fn new(xdg: &xdg::BaseDirectories) -> Result<Self, OpenVrError> {
        let sys = crate::openvr::VRSystem::init()?;
        let vroverlay = sys.overlay().create_overlay(APP_KEY, APP_NAME)?;
        let mut input = unsafe { Pin::new_unchecked(&mut *openvr_sys2::VRInput()) };
        let action_manifest = xdg.find_config_file("actions.json").unwrap();
        let action_manifest = std::ffi::CString::new(action_manifest.to_str().unwrap()).unwrap();
        unsafe {
            input
                .as_mut()
                .SetActionManifestPath(action_manifest.as_ptr())
        }
        .into_result()?;
        let mut button = [const { MaybeUninit::uninit() }; 3];
        for i in 0..2 {
            let name = CString::new(format!("actions/main/in/button{}", i + 1)).unwrap();
            unsafe {
                input
                    .as_mut()
                    .GetActionHandle(
                        name.as_ptr(),
                        MaybeUninit::slice_as_mut_ptr(&mut button[i..]),
                    )
                    .into_result()?;
            };
        }
        unsafe {
            let name = CString::new("actions/main/in/debug").unwrap();
            input
                .as_mut()
                .GetActionHandle(
                    name.as_ptr(),
                    MaybeUninit::slice_as_mut_ptr(&mut button[2..]),
                )
                .into_result()?;
        }
        let button = unsafe { MaybeUninit::array_assume_init(button) };

        let action_set = unsafe {
            let mut action_set = MaybeUninit::uninit();
            let action_set_name = CString::new("actions/main").unwrap();
            input
                .GetActionSetHandle(action_set_name.as_ptr(), action_set.as_mut_ptr())
                .into_result()?;
            action_set.assume_init()
        };
        Ok(Self {
            sys,
            handle: vroverlay,
            action_set,
            buttons: button,
            texture: None,
        })
    }
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum OpenVrError {
    #[error("vulkan device not found")]
    VulkanDeviceNotFound,
    #[error("openvr init error: {0}")]
    InitError(#[from] EVRInitError),
    #[error("overlay error: {0}")]
    OverlayError(#[from] EVROverlayError),
    #[error("tracked property error: {0}")]
    TrackedPropertyError(#[from] ETrackedPropertyError),
    #[error("input error: {0}")]
    InputError(#[from] EVRInputError),
}

impl Drop for OpenVr {
    fn drop(&mut self) {
        log::info!("Dropping overlay handle");
        let vroverlay = self.sys.overlay();
        if let Err(e) = unsafe { vroverlay.destroy_overlay_raw(self.handle) } {
            eprintln!("{}", e);
        }
    }
}

impl Vr for OpenVr {
    type Error = OpenVrError;
    fn hmd_transform(&self, time_offset: f32) -> nalgebra::Matrix4<f64> {
        self.sys.hmd_transform(time_offset)
    }
    fn target_device(&self, instance: &Arc<Instance>) -> Result<Arc<PhysicalDevice>, Self::Error> {
        let mut target_device = 0u64;
        unsafe {
            self.sys.pin_mut().GetOutputDevice(
                &mut target_device,
                openvr_sys2::ETextureType::TextureType_Vulkan,
                instance.handle().as_raw() as *mut _,
            )
        };
        let target_device = ash::vk::PhysicalDevice::from_raw(target_device);
        instance
            .enumerate_physical_devices()
            .unwrap()
            .find(|physical_device| {
                if physical_device.handle() == target_device {
                    println!(
                        "Found matching device: {}",
                        physical_device.properties().device_name
                    );
                    true
                } else {
                    false
                }
            })
            .ok_or(OpenVrError::VulkanDeviceNotFound)
    }
    fn required_extensions(&self, pdev: &PhysicalDevice) -> vulkano::device::DeviceExtensions {
        let compositor = self.sys.compositor();
        let mut buf = Vec::new();
        compositor
            .required_extensions(pdev, &mut buf)
            .map(|cstr| cstr.to_str().unwrap())
            .collect()
    }
    fn load_camera_paramter(&self) -> Option<StereoCamera> {
        // Load steam calibration data
        let hmd_id = self.sys.find_hmd()?;
        let mut serial_number = [0u8; 32];
        let mut error = ETrackedPropertyError::TrackedProp_Success;
        let serial_number_len = unsafe {
            self.sys.pin_mut().GetStringTrackedDeviceProperty(
                hmd_id,
                openvr_sys2::ETrackedDeviceProperty::Prop_SerialNumber_String,
                serial_number.as_mut_ptr() as *mut _,
                32,
                &mut error,
            )
        };
        if error != ETrackedPropertyError::TrackedProp_Success {
            return None;
        }
        let lhcfg = crate::steam::load_steam_config(
            std::str::from_utf8(&serial_number[..serial_number_len as usize - 1]).ok()?,
        )
        .ok()?;
        log::info!(
            "{}",
            serde_json::to_string(&lhcfg).unwrap_or("invalid json".to_owned())
        );
        Some(lhcfg)
    }
    fn set_overlay_texture(
        &mut self,
        w: u32,
        h: u32,
        image: Arc<impl vulkano::image::ImageAccess + 'static>,
        dev: Arc<Device>,
        queue: Arc<Queue>,
        instance: Arc<Instance>,
    ) -> Result<(), Self::Error> {
        let texture = TextureState {
            _image: image.clone() as Arc<_>,
            _device: dev.clone(),
            _queue: queue.clone(),
            _instance: instance.clone(),
        };
        self.texture.replace(texture);
        let vroverlay = self.sys.overlay();
        // Once we set a texture, the VRSystem starts to depend on Vulkan
        // instance being alive.
        self.sys.hold_vulkan_device(dev.clone());
        let mut vrimage = openvr_sys2::VRVulkanTextureData_t {
            m_nWidth: w,
            m_nHeight: h,
            m_nFormat: image.format() as u32,
            m_nSampleCount: image.samples() as u32,
            m_nImage: image.inner().image.handle().as_raw(),
            m_pPhysicalDevice: dev.physical_device().handle().as_raw() as *mut _,
            m_pDevice: dev.handle().as_raw() as *mut _,
            m_pQueue: queue.handle().as_raw() as *mut _,
            m_pInstance: instance.handle().as_raw() as *mut _,
            m_nQueueFamilyIndex: queue.queue_family_index(),
        };
        let vrtexture = openvr_sys2::Texture_t {
            handle: &mut vrimage as *mut _ as *mut std::ffi::c_void,
            eType: openvr_sys2::ETextureType::TextureType_Vulkan,
            eColorSpace: openvr_sys2::EColorSpace::ColorSpace_Auto,
        };
        unsafe {
            vroverlay
                .pin_mut()
                .SetOverlayTexture(self.handle, &vrtexture)
                .into_result()
                .map_err(Into::into)
        }
    }
    fn set_overlay_stereo(&mut self, stereo: bool) -> Result<(), Self::Error> {
        self.sys
            .overlay()
            .pin_mut()
            .SetOverlayFlag(
                self.handle,
                openvr_sys2::VROverlayFlags::VROverlayFlags_SideBySide_Parallel,
                stereo,
            )
            .into_result()
            .map_err(Into::into)
    }
    fn set_overlay_texture_bounds(&mut self, bounds: Bounds) -> Result<(), Self::Error> {
        let bounds = openvr_sys2::VRTextureBounds_t {
            uMin: bounds.umin,
            vMin: bounds.vmin,
            uMax: bounds.umax,
            vMax: bounds.vmax,
        };
        unsafe {
            self.sys
                .overlay()
                .pin_mut()
                .SetOverlayTextureBounds(self.handle, &bounds)
                .into_result()
                .map_err(Into::into)
        }
    }
    fn show_overlay(&mut self) -> Result<(), Self::Error> {
        self.sys
            .overlay()
            .pin_mut()
            .ShowOverlay(self.handle)
            .into_result()
            .map_err(Into::into)
    }
    fn hide_overlay(&mut self) -> Result<(), Self::Error> {
        self.sys
            .overlay()
            .pin_mut()
            .HideOverlay(self.handle)
            .into_result()
            .map_err(Into::into)
    }
    fn ipd(&self) -> Result<f32, Self::Error> {
        let mut error = MaybeUninit::<_>::uninit();
        unsafe {
            let ipd = self.sys.pin_mut().GetFloatTrackedDeviceProperty(
                0,
                openvr_sys2::ETrackedDeviceProperty::Prop_UserIpdMeters_Float,
                error.as_mut_ptr(),
            );
            error.assume_init().into_result()?;
            Ok(ipd)
        }
    }
    fn eye_to_head(&self) -> [Matrix4<f64>; 2] {
        let left_eye: Matrix4<_> = self
            .sys
            .pin_mut()
            .GetEyeToHeadTransform(openvr_sys2::EVREye::Eye_Left)
            .into();
        let right_eye: Matrix4<_> = self
            .sys
            .pin_mut()
            .GetEyeToHeadTransform(openvr_sys2::EVREye::Eye_Right)
            .into();
        [left_eye, right_eye]
    }
    fn set_overlay_transformation(&mut self, transform: Matrix4<f64>) -> Result<(), Self::Error> {
        let vroverlay = self.sys.overlay();
        unsafe {
            vroverlay.pin_mut().SetOverlayTransformAbsolute(
                self.handle,
                openvr_sys2::ETrackingUniverseOrigin::TrackingUniverseStanding,
                &(&transform).into(),
            )
        }
        .into_result()
        .map_err(Into::into)
    }
    fn acknowledge_quit(&mut self) {
        self.sys.pin_mut().AcknowledgeQuit_Exiting();
    }
    fn poll_next_event(&mut self) -> Option<Event> {
        use openvr_sys2::VREvent_t;
        let openvr_event: VREvent_t = unsafe {
            let mut event = MaybeUninit::uninit();
            let has_event = self.sys.pin_mut().PollNextEvent(
                event.as_mut_ptr() as *mut _,
                std::mem::size_of::<openvr_sys2::VREvent_t>() as u32,
            );
            has_event.then(|| event.assume_init())
        }?;
        match openvr_sys2::EVREventType::try_from(openvr_event.eventType).unwrap() {
            openvr_sys2::EVREventType::VREvent_IpdChanged => {
                let ipd = unsafe { openvr_event.data.ipd.ipdMeters };
                Some(Event::IpdChanged(ipd))
            }
            openvr_sys2::EVREventType::VREvent_Quit => Some(Event::RequestExit),
            _ => None,
        }
    }
    fn update_action_state(&mut self) -> Result<(), Self::Error> {
        let vrinput = unsafe { Pin::new_unchecked(&mut *openvr_sys2::VRInput()) };
        let mut active_action_set = openvr_sys2::VRActiveActionSet_t {
            ulActionSet: self.action_set,
            ulSecondaryActionSet: 0,
            unPadding: 0,
            ulRestrictedToDevice: openvr_sys2::vr::k_ulInvalidInputValueHandle,
            nPriority: 0,
        };
        unsafe {
            vrinput
                .UpdateActionState(
                    &mut active_action_set,
                    std::mem::size_of::<openvr_sys2::VRActiveActionSet_t>() as u32,
                    1,
                )
                .into_result()
                .map_err(Into::into)
        }
    }
    fn get_action_state(&self, action: Action) -> Result<bool, Self::Error> {
        let action_handle = self.buttons[action as usize];
        let vrinput = unsafe { Pin::new_unchecked(&mut *openvr_sys2::VRInput()) };
        let action_data = unsafe {
            let mut action_data = MaybeUninit::uninit();
            vrinput
                .GetDigitalActionData(
                    action_handle,
                    action_data.as_mut_ptr(),
                    std::mem::size_of::<openvr_sys2::VRInputValueHandle_t>() as u32,
                    openvr_sys2::vr::k_ulInvalidInputValueHandle,
                )
                .into_result()?;
            action_data.assume_init()
        };
        Ok(action_data.bState)
    }
}
