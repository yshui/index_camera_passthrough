#![feature(untagged_unions)]
autocxx::include_cpp! {
    #include "/usr/include/openvr/openvr.h"
    generate!("vr::IVRSystem")
    generate!("vr::IVROverlay")
    generate!("vr::IVRCompositor")
    generate_pod!("vr::VREvent_Reserved_t")
    generate_pod!("vr::VREvent_Controller_t")
    generate_pod!("vr::VREvent_Mouse_t")
    generate_pod!("vr::VREvent_Scroll_t")
    generate_pod!("vr::VREvent_Process_t")
    generate_pod!("vr::VREvent_Notification_t")
    generate_pod!("vr::VREvent_Overlay_t")
    generate_pod!("vr::VREvent_Status_t")
    generate_pod!("vr::VREvent_Keyboard_t")
    generate_pod!("vr::VREvent_Ipd_t")
    generate_pod!("vr::VREvent_Chaperone_t")
    generate_pod!("vr::VREvent_PerformanceTest_t")
    generate_pod!("vr::VREvent_TouchPadMove_t")
    generate_pod!("vr::VREvent_SeatedZeroPoseReset_t")
    generate_pod!("vr::VREvent_Screenshot_t")
    generate_pod!("vr::VREvent_ScreenshotProgress_t")
    generate_pod!("vr::VREvent_ApplicationLaunch_t")
    generate_pod!("vr::VREvent_EditingCameraSurface_t")
    generate_pod!("vr::VREvent_MessageOverlay_t")
    generate_pod!("vr::VREvent_Property_t")
    generate_pod!("vr::VREvent_HapticVibration_t")
    generate_pod!("vr::VREvent_WebConsole_t")
    generate_pod!("vr::VREvent_InputBindingLoad_t")
    generate_pod!("vr::VREvent_InputActionManifestLoad_t")
    generate_pod!("vr::VREvent_SpatialAnchor_t")
    generate_pod!("vr::VREvent_ProgressUpdate_t")
    generate_pod!("vr::VREvent_ShowUI_t")
    generate_pod!("vr::VREvent_ShowDevTools_t")
    generate_pod!("vr::VREvent_HDCPError_t")
    generate_pod!("vr::TrackedDeviceIndex_t")
    generate_pod!("vr::TrackedDevicePose_t")
    generate_pod!("vr::VRTextureBounds_t")
    generate_pod!("vr::VRVulkanTextureData_t")
    generate_pod!("vr::Texture_t")
    generate_pod!("vr::ETextureType")
    generate_pod!("vr::HmdMatrix34_t")
    generate!("vr::VR_Init")
    generate!("vr::VR_Shutdown")
    generate!("vr::VR_IsHmdPresent")
    generate!("vr::VROverlay")
    generate!("vr::VRCompositor")
    safety!(unsafe)
}

#[allow(non_camel_case_types, non_snake_case)]
#[repr(C)]
pub union VREvent_Data_t {
    pub reserved: ffi::vr::VREvent_Reserved_t,
    pub controller: ffi::vr::VREvent_Controller_t,
    pub mouse: ffi::vr::VREvent_Mouse_t,
    pub scroll: ffi::vr::VREvent_Scroll_t,
    pub process: ffi::vr::VREvent_Process_t,
    pub notification: ffi::vr::VREvent_Notification_t,
    pub overlay: ffi::vr::VREvent_Overlay_t,
    pub status: ffi::vr::VREvent_Status_t,
    pub keyboard: ffi::vr::VREvent_Keyboard_t,
    pub ipd: ffi::vr::VREvent_Ipd_t,
    pub chaperone: ffi::vr::VREvent_Chaperone_t,
    pub performanceTest: ffi::vr::VREvent_PerformanceTest_t,
    pub touchPadMove: ffi::vr::VREvent_TouchPadMove_t,
    pub seatedZeroPoseReset: ffi::vr::VREvent_SeatedZeroPoseReset_t,
    pub screenshot: ffi::vr::VREvent_Screenshot_t,
    pub screenshotProgress: ffi::vr::VREvent_ScreenshotProgress_t,
    pub applicationLaunch: ffi::vr::VREvent_ApplicationLaunch_t,
    pub cameraSurface: ffi::vr::VREvent_EditingCameraSurface_t,
    pub messageOverlay: ffi::vr::VREvent_MessageOverlay_t,
    pub property: ffi::vr::VREvent_Property_t,
    pub hapticVibration: ffi::vr::VREvent_HapticVibration_t,
    pub webConsole: ffi::vr::VREvent_WebConsole_t,
    pub inputBinding: ffi::vr::VREvent_InputBindingLoad_t,
    pub actionManifest: ffi::vr::VREvent_InputActionManifestLoad_t,
    pub spatialAnchor: ffi::vr::VREvent_SpatialAnchor_t,
    pub progressUpdate: ffi::vr::VREvent_ProgressUpdate_t,
    pub showUi: ffi::vr::VREvent_ShowUI_t,
    pub showDevTools: ffi::vr::VREvent_ShowDevTools_t,
    pub hdcpError: ffi::vr::VREvent_HDCPError_t,
}
#[allow(non_camel_case_types, non_snake_case)]
#[repr(C, packed(4))]
pub struct VREvent_t {
    pub eventType: u32,
    pub trackedDeviceIndex: ffi::vr::TrackedDeviceIndex_t,
    pub eventAgeSeconds: f32,
    pub data: VREvent_Data_t,
}

pub use ffi::vr::*;
pub use ffi::*;

impl EVROverlayError {
    pub fn into_result(self) -> Result<(), Self> {
        if self == EVROverlayError::VROverlayError_None {
            Ok(())
        } else {
            Err(self)
        }
    }
}

impl std::fmt::Display for EVROverlayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as std::fmt::Debug>::fmt(self, f)
    }
}

impl std::error::Error for EVROverlayError {

}

impl EVRInitError {
    pub fn into_result(self) -> Result<(), Self> {
        if self == EVRInitError::VRInitError_None {
            Ok(())
        } else {
            Err(self)
        }
    }
}

impl std::fmt::Display for EVRInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as std::fmt::Debug>::fmt(self, f)
    }
}

impl std::error::Error for EVRInitError {

}

impl Into<nalgebra::Matrix3x4<f32>> for HmdMatrix34_t {
    fn into(self) -> nalgebra::Matrix3x4<f32> {
        use nalgebra::matrix;
        matrix![
            self.m[0][0], self.m[0][1], self.m[0][2], self.m[0][3];
            self.m[1][0], self.m[1][1], self.m[1][2], self.m[1][3];
            self.m[2][0], self.m[2][1], self.m[2][2], self.m[2][3];
        ]
    }
}

impl Into<nalgebra::Matrix4<f32>> for HmdMatrix34_t {
    fn into(self) -> nalgebra::Matrix4<f32> {
        use nalgebra::matrix;
        matrix![
            self.m[0][0], self.m[0][1], self.m[0][2], self.m[0][3];
            self.m[1][0], self.m[1][1], self.m[1][2], self.m[1][3];
            self.m[2][0], self.m[2][1], self.m[2][2], self.m[2][3];
                     0.0,          0.0,          0.0,          1.0;
        ]
    }
}
