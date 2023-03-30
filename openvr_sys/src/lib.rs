use std::mem::ManuallyDrop;
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
    pub reserved: ManuallyDrop<ffi::vr::VREvent_Reserved_t>,
    pub controller: ManuallyDrop<ffi::vr::VREvent_Controller_t>,
    pub mouse: ManuallyDrop<ffi::vr::VREvent_Mouse_t>,
    pub scroll: ManuallyDrop<ffi::vr::VREvent_Scroll_t>,
    pub process: ManuallyDrop<ffi::vr::VREvent_Process_t>,
    pub notification: ManuallyDrop<ffi::vr::VREvent_Notification_t>,
    pub overlay: ManuallyDrop<ffi::vr::VREvent_Overlay_t>,
    pub status: ManuallyDrop<ffi::vr::VREvent_Status_t>,
    pub keyboard: ManuallyDrop<ffi::vr::VREvent_Keyboard_t>,
    pub ipd: ManuallyDrop<ffi::vr::VREvent_Ipd_t>,
    pub chaperone: ManuallyDrop<ffi::vr::VREvent_Chaperone_t>,
    pub performanceTest: ManuallyDrop<ffi::vr::VREvent_PerformanceTest_t>,
    pub touchPadMove: ManuallyDrop<ffi::vr::VREvent_TouchPadMove_t>,
    pub seatedZeroPoseReset: ManuallyDrop<ffi::vr::VREvent_SeatedZeroPoseReset_t>,
    pub screenshot: ManuallyDrop<ffi::vr::VREvent_Screenshot_t>,
    pub screenshotProgress: ManuallyDrop<ffi::vr::VREvent_ScreenshotProgress_t>,
    pub applicationLaunch: ManuallyDrop<ffi::vr::VREvent_ApplicationLaunch_t>,
    pub cameraSurface: ManuallyDrop<ffi::vr::VREvent_EditingCameraSurface_t>,
    pub messageOverlay: ManuallyDrop<ffi::vr::VREvent_MessageOverlay_t>,
    pub property: ManuallyDrop<ffi::vr::VREvent_Property_t>,
    pub hapticVibration: ManuallyDrop<ffi::vr::VREvent_HapticVibration_t>,
    pub webConsole: ManuallyDrop<ffi::vr::VREvent_WebConsole_t>,
    pub inputBinding: ManuallyDrop<ffi::vr::VREvent_InputBindingLoad_t>,
    pub actionManifest: ManuallyDrop<ffi::vr::VREvent_InputActionManifestLoad_t>,
    pub spatialAnchor: ManuallyDrop<ffi::vr::VREvent_SpatialAnchor_t>,
    pub progressUpdate: ManuallyDrop<ffi::vr::VREvent_ProgressUpdate_t>,
    pub showUi: ManuallyDrop<ffi::vr::VREvent_ShowUI_t>,
    pub showDevTools: ManuallyDrop<ffi::vr::VREvent_ShowDevTools_t>,
    pub hdcpError: ManuallyDrop<ffi::vr::VREvent_HDCPError_t>,
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

impl std::fmt::Debug for EVROverlayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VROverlayError_None => write!(f, "VROverlayError_None"),
            Self::VROverlayError_UnknownOverlay => write!(f, "VROverlayError_UnknownOverlay"),
            Self::VROverlayError_InvalidHandle => write!(f, "VROverlayError_InvalidHandle"),
            Self::VROverlayError_PermissionDenied => write!(f, "VROverlayError_PermissionDenied"),
            Self::VROverlayError_OverlayLimitExceeded => write!(f, "VROverlayError_OverlayLimitExceeded"),
            Self::VROverlayError_WrongVisibilityType => write!(f, "VROverlayError_WrongVisibilityType"),
            Self::VROverlayError_KeyTooLong => write!(f, "VROverlayError_KeyTooLong"),
            Self::VROverlayError_NameTooLong => write!(f, "VROverlayError_NameTooLong"),
            Self::VROverlayError_KeyInUse => write!(f, "VROverlayError_KeyInUse"),
            Self::VROverlayError_WrongTransformType => write!(f, "VROverlayError_WrongTransformType"),
            Self::VROverlayError_InvalidTrackedDevice => write!(f, "VROverlayError_InvalidTrackedDevice"),
            Self::VROverlayError_InvalidParameter => write!(f, "VROverlayError_InvalidParameter"),
            Self::VROverlayError_ThumbnailCantBeDestroyed => write!(f, "VROverlayError_ThumbnailCantBeDestroyed"),
            Self::VROverlayError_ArrayTooSmall => write!(f, "VROverlayError_ArrayTooSmall"),
            Self::VROverlayError_RequestFailed => write!(f, "VROverlayError_RequestFailed"),
            Self::VROverlayError_InvalidTexture => write!(f, "VROverlayError_InvalidTexture"),
            Self::VROverlayError_UnableToLoadFile => write!(f, "VROverlayError_UnableToLoadFile"),
            Self::VROverlayError_KeyboardAlreadyInUse => write!(f, "VROverlayError_KeyboardAlreadyInUse"),
            Self::VROverlayError_NoNeighbor => write!(f, "VROverlayError_NoNeighbor"),
            Self::VROverlayError_TooManyMaskPrimitives => write!(f, "VROverlayError_TooManyMaskPrimitives"),
            Self::VROverlayError_BadMaskPrimitive => write!(f, "VROverlayError_BadMaskPrimitive"),
            Self::VROverlayError_TextureAlreadyLocked => write!(f, "VROverlayError_TextureAlreadyLocked"),
            Self::VROverlayError_TextureLockCapacityReached => write!(f, "VROverlayError_TextureLockCapacityReached"),
            Self::VROverlayError_TextureNotLocked => write!(f, "VROverlayError_TextureNotLocked"),
            Self::VROverlayError_TimedOut => write!(f, "VROverlayError_TimedOut"),
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

impl std::fmt::Debug for ETrackedPropertyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ETrackedPropertyError::*;
        match self {
            TrackedProp_Success => write!(f, "TrackedProp_Success"),
            TrackedProp_WrongDataType => write!(f, "TrackedProp_WrongDataType"),
            TrackedProp_WrongDeviceClass => write!(f, "TrackedProp_WrongDeviceClass"),
            TrackedProp_BufferTooSmall => write!(f, "TrackedProp_BufferTooSmall"),
            TrackedProp_UnknownProperty => write!(f, "TrackedProp_UnknownProperty"),
            TrackedProp_InvalidDevice => write!(f, "TrackedProp_InvalidDevice"),
            TrackedProp_CouldNotContactServer => write!(f, "TrackedProp_CouldNotContactServer"),
            TrackedProp_ValueNotProvidedByDevice => write!(f, "TrackedProp_ValueNotProvidedByDevice"),
            TrackedProp_StringExceedsMaximumLength => write!(f, "TrackedProp_StringExceedsMaximumLength"),
            TrackedProp_NotYetAvailable => write!(f, "TrackedProp_NotYetAvailable"),
            TrackedProp_PermissionDenied => write!(f, "TrackedProp_PermissionDenied"),
            TrackedProp_InvalidOperation => write!(f, "TrackedProp_InvalidOperation"),
            TrackedProp_CannotWriteToWildcards => write!(f, "TrackedProp_CannotWriteToWildcards"),
            TrackedProp_IPCReadFailure => write!(f, "TrackedProp_IPCReadFailure"),
            TrackedProp_OutOfMemory => write!(f, "TrackedProp_OutOfMemory"),
            TrackedProp_InvalidContainer => write!(f, "TrackedProp_InvalidContainer"),
        }
    }
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

impl std::fmt::Debug for EVRInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            _ => write!(f, "VRInitError_Unknown")
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

impl<T: num::NumCast + num::Float + num::Zero + num::One + nalgebra::Scalar> Into<nalgebra::Matrix4<T>> for &'_ HmdMatrix34_t {
    fn into(self) -> nalgebra::Matrix4<T> {
        // Note: [[float; 4]; 4] -> Matrix is column major
        let mut tmp = [[T::nan(); 4]; 4];
        for i in 0..3 {
            for j in 0..4 {
                tmp[j][i] = T::from(self.m[i][j]).unwrap();
            }
        }
        for i in 0..3 {
            tmp[i][3] = T::zero();
        }
        tmp[3][3] = T::one();
        tmp.into()
    }
}

impl<T: num::NumCast + num::Float + num::Zero + num::One + nalgebra::Scalar> Into<nalgebra::Matrix4<T>> for HmdMatrix34_t {
    fn into(self) -> nalgebra::Matrix4<T> {
        (&self).into()
    }
}

impl<T: num::ToPrimitive> From<&'_ nalgebra::Matrix4<T>> for HmdMatrix34_t {
    fn from(m: &nalgebra::Matrix4<T>) -> Self {
        let mut ret = unsafe { std::mem::MaybeUninit::<Self>::zeroed().assume_init() };
        for i in 0..3 {
            for j in 0..4 {
                ret.m[i][j] = m[(i,j)].to_f32().unwrap();
            }
        }
        ret
    }
}
