use itertools::Itertools;
use nalgebra::{Affine3, Matrix3, Matrix4, Translation3, UnitQuaternion, Vector3};
use openvr_sys2::{ETrackedPropertyError, EVRInitError, EVRInputError, EVROverlayError};
use openxr::{
    ApplicationInfo, EnvironmentBlendMode, EventDataBuffer, Extent2Df, Extent2Di, EyeVisibility,
    Offset2Di, OverlaySessionCreateFlagsEXTX, Rect2Di, ReferenceSpaceType, SwapchainSubImage,
    ViewConfigurationType, ViewStateFlags,
};
use std::{
    ffi::CString,
    mem::MaybeUninit,
    pin::Pin,
    sync::{Arc, OnceLock},
    time::Duration,
};
use vulkano::{
    command_buffer::{
        allocator::{CommandBufferAllocator, StandardCommandBufferAllocator},
        sys::{CommandBufferBeginInfo, RawRecordingCommandBuffer},
        CommandBufferLevel, CommandBufferUsage,
    },
    descriptor_set::allocator::{
        DescriptorSetAllocator, StandardDescriptorSetAllocator,
        StandardDescriptorSetAllocatorCreateInfo,
    },
    device::{physical::PhysicalDevice, Device, Queue, QueueCreateInfo, QueueFlags},
    image::{Image, ImageAspects, ImageCreateInfo, ImageLayout, ImageSubresourceRange, ImageUsage},
    instance::Instance,
    memory::allocator::{MemoryAllocator, StandardMemoryAllocator},
    sync::{AccessFlags, DependencyInfo, GpuFuture, ImageMemoryBarrier, PipelineStages},
    Handle, VulkanObject,
};

use serde::{Deserialize, Serialize};

use crate::{
    config::{DisplayMode, Eye, PositionMode},
    utils::DeviceExt,
    APP_KEY, APP_NAME, CAMERA_SIZE,
};
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
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Action {
    Button1 = 0,
    Button2 = 1,
    Debug = 2,
    Reposition = 3,
}

pub(crate) trait VkContext {
    fn vk_device(&self, instance: &Arc<Instance>) -> (Arc<Device>, Arc<Queue>);
    fn vk_instance(&self) -> Arc<Instance>;
    fn vk_allocator(&self) -> Arc<dyn MemoryAllocator>;
    fn vk_descriptor_set_allocator(&self) -> Arc<dyn DescriptorSetAllocator>;
    fn vk_command_buffer_allocator(&self) -> Arc<dyn CommandBufferAllocator>;
}

pub(crate) trait Vr: VkContext {
    type Error: Send + Sync + 'static;
    fn load_camera_paramter(&mut self) -> Option<StereoCamera>;
    fn set_fallback_camera_config(&mut self, cfg: StereoCamera);
    /// Submit the render texture to overlay.
    ///
    /// Must have called `render_texture` before calling this function. The render texture must have
    /// the ColorAttachmentOptimal layout when this function is called.
    ///
    /// # Arguments
    ///
    /// - `elapsed`: duration since the image was captured.
    fn submit_texture(&mut self, elapsed: Duration, fov: &[[f32; 2]; 2])
        -> Result<(), Self::Error>;
    /// Refresh the overlay using the latest submitted camera texture.
    fn refresh(&mut self) -> Result<(), Self::Error>;
    /// Whether our render loop is synchronized with the VR runtime.
    ///
    /// If this is true, `get_render_texture`, `submit_texture` and `refresh` should be synchronized with
    /// the VR runtime.
    fn is_synchronized(&self) -> bool;
    fn set_position_mode(&mut self, mode: PositionMode) -> Result<(), Self::Error>;
    /// Change the display mode of the overlay.
    ///
    /// This invalidates previously returned render texture.
    fn set_display_mode(&mut self, mode: DisplayMode) -> Result<(), Self::Error>;
    fn show_overlay(&mut self) -> Result<(), Self::Error>;
    fn hide_overlay(&mut self) -> Result<(), Self::Error>;
    fn acknowledge_quit(&mut self);
    /// Acquire a render texture that can be submitted to the overlay.
    ///
    /// Once this is called, the render texture is considered acquired until it's released by calling `submit_texture`.
    /// Caller should upload camera frames to the render texture, then call `submit_texture`.
    fn get_render_texture(&mut self) -> Result<Option<Arc<Image>>, Self::Error>;
    fn poll_next_event(&mut self) -> Result<Option<Event>, Self::Error>;
    fn update_action_state(&mut self) -> Result<(), Self::Error>;
    fn get_action_state(&self, action: Action) -> Result<bool, Self::Error>;
    fn wait_for_ready(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

struct VrMapError<T, F>(T, F);

impl<T: VkContext, F> VkContext for VrMapError<T, F> {
    fn vk_allocator(&self) -> Arc<dyn MemoryAllocator> {
        self.0.vk_allocator()
    }
    fn vk_descriptor_set_allocator(&self) -> Arc<dyn DescriptorSetAllocator> {
        self.0.vk_descriptor_set_allocator()
    }
    fn vk_device(&self, instance: &Arc<Instance>) -> (Arc<Device>, Arc<Queue>) {
        self.0.vk_device(instance)
    }
    fn vk_instance(&self) -> Arc<Instance> {
        self.0.vk_instance()
    }
    fn vk_command_buffer_allocator(&self) -> Arc<dyn CommandBufferAllocator> {
        self.0.vk_command_buffer_allocator()
    }
}

impl<T: Vr, E: Send + Sync + 'static, F: Fn(<T as Vr>::Error) -> E> Vr for VrMapError<T, F> {
    type Error = E;
    fn acknowledge_quit(&mut self) {
        self.0.acknowledge_quit()
    }
    fn load_camera_paramter(&mut self) -> Option<StereoCamera> {
        self.0.load_camera_paramter()
    }
    fn set_fallback_camera_config(&mut self, cfg: StereoCamera) {
        self.0.set_fallback_camera_config(cfg)
    }
    fn submit_texture(
        &mut self,
        elapsed: Duration,
        fov: &[[f32; 2]; 2],
    ) -> Result<(), Self::Error> {
        self.0.submit_texture(elapsed, fov).map_err(&self.1)
    }
    fn refresh(&mut self) -> Result<(), Self::Error> {
        self.0.refresh().map_err(&self.1)
    }
    fn is_synchronized(&self) -> bool {
        self.0.is_synchronized()
    }
    fn show_overlay(&mut self) -> Result<(), Self::Error> {
        self.0.show_overlay().map_err(&self.1)
    }
    fn set_display_mode(&mut self, mode: DisplayMode) -> Result<(), Self::Error> {
        self.0.set_display_mode(mode).map_err(&self.1)
    }
    fn set_position_mode(&mut self, mode: PositionMode) -> Result<(), Self::Error> {
        self.0.set_position_mode(mode).map_err(&self.1)
    }
    fn get_render_texture(&mut self) -> Result<Option<Arc<Image>>, Self::Error> {
        self.0.get_render_texture().map_err(&self.1)
    }
    fn hide_overlay(&mut self) -> Result<(), Self::Error> {
        self.0.hide_overlay().map_err(&self.1)
    }
    fn poll_next_event(&mut self) -> Result<Option<Event>, Self::Error> {
        self.0.poll_next_event().map_err(&self.1)
    }
    fn get_action_state(&self, action: Action) -> Result<bool, Self::Error> {
        self.0.get_action_state(action).map_err(&self.1)
    }
    fn update_action_state(&mut self) -> Result<(), Self::Error> {
        self.0.update_action_state().map_err(&self.1)
    }
    fn wait_for_ready(&mut self) -> Result<(), Self::Error> {
        self.0.wait_for_ready().map_err(&self.1)
    }
}
pub(crate) trait VrExt: Sized {
    fn boxed(self) -> Box<dyn Vr<Error = anyhow::Error>>;
}

impl<T: Vr + 'static> VrExt for T
where
    T::Error: std::error::Error,
{
    fn boxed(self) -> Box<dyn Vr<Error = anyhow::Error>> {
        Box::new(VrMapError(self, anyhow::Error::from)) as _
    }
}

struct TextureState {
    _image: Arc<vulkano::image::Image>,
    _device: Arc<Device>,
    _queue: Arc<Queue>,
    _instance: Arc<Instance>,
}

pub(crate) struct OpenVr {
    sys: crate::openvr::VRSystem,
    handle: openvr_sys2::VROverlayHandle_t,
    buttons: [openvr_sys2::VRActionHandle_t; 4],
    action_set: openvr_sys2::VRActionSetHandle_t,
    texture: Option<TextureState>,
    camera_config: Option<StereoCamera>,
    position_mode: PositionMode,
    reposition: bool,
    display_mode: DisplayMode,
    overlay_transform: Matrix4<f32>,
    projector: Option<crate::projection::Projection>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    instance: Arc<Instance>,
    allocator: Arc<StandardMemoryAllocator>,
    cmdbuf_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    render_texture: Option<Arc<vulkano::image::Image>>,
    double_buffer: [Arc<vulkano::image::Image>; 2],
    texture_in_use: u64,
    ipd: Option<f32>,
}
impl OpenVr {
    fn create_vk_device(
        sys: &crate::openvr::VRSystem,
        instance: &Arc<Instance>,
    ) -> Result<(Arc<Device>, Arc<Queue>), OpenVrError> {
        let mut target_device = 0u64;
        unsafe {
            sys.pin_mut().GetOutputDevice(
                &mut target_device,
                openvr_sys2::ETextureType::TextureType_Vulkan,
                instance.handle().as_raw() as *mut _,
            )
        };
        let target_device = ash::vk::PhysicalDevice::from_raw(target_device);
        let physical_device = instance
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
            .ok_or(OpenVrError::VulkanDeviceNotFound)?;

        let queue_family = physical_device
            .queue_family_properties()
            .iter()
            .position(|qf| qf.queue_flags.contains(QueueFlags::GRAPHICS))
            .ok_or(OpenVrError::NoGraphicsQueue)?;
        let (device, mut queues) = {
            let extensions = Self::required_extensions(sys, &physical_device);
            vulkano::device::Device::new(
                physical_device,
                vulkano::device::DeviceCreateInfo {
                    enabled_features: vulkano::device::Features::empty(),
                    enabled_extensions: extensions,
                    queue_create_infos: vec![vulkano::device::QueueCreateInfo {
                        queue_family_index: queue_family as u32,
                        queues: vec![1.0],
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            )?
        };
        Ok((device, queues.next().unwrap()))
    }
    fn create_vk_instance() -> Result<Arc<Instance>, OpenVrError> {
        let library = get_vulkan_library().clone();
        // Create vulkan instance, and setup openvr.
        // Then create a vulkan device based on openvr's requirements
        let extensions = *library.supported_extensions();
        Ok(Instance::new(
            library,
            vulkano::instance::InstanceCreateInfo {
                max_api_version: Some(vulkano::Version::V1_6),
                enabled_extensions: extensions,
                // enabled_layers: vec!["VK_LAYER_KHRONOS_validation".to_owned()],
                ..Default::default()
            },
        )?)
    }
    pub fn new(xdg: &xdg::BaseDirectories) -> Result<Self, OpenVrError> {
        let sys = crate::openvr::VRSystem::init()?;
        let vroverlay = sys.overlay().create_overlay(APP_KEY, APP_NAME)?;
        sys.overlay()
            .pin_mut()
            .SetOverlayTextureColorSpace(vroverlay, openvr_sys2::EColorSpace::ColorSpace_Linear)
            .into_result()?;
        let mut input = unsafe { Pin::new_unchecked(&mut *openvr_sys2::VRInput()) };
        let action_manifest = xdg.find_data_file("actions.json").unwrap();
        let action_manifest = std::ffi::CString::new(action_manifest.to_str().unwrap()).unwrap();
        unsafe {
            input
                .as_mut()
                .SetActionManifestPath(action_manifest.as_ptr())
        }
        .into_result()?;
        let mut button = [const { MaybeUninit::uninit() }; 4];
        for i in 0..2 {
            let name = CString::new(format!("/actions/main/in/button{}", i + 1)).unwrap();
            unsafe {
                input
                    .as_mut()
                    .GetActionHandle(name.as_ptr(), button[i].as_mut_ptr())
                    .into_result()?;
            };
        }
        unsafe {
            let name = CString::new("/actions/main/in/debug").unwrap();
            input
                .as_mut()
                .GetActionHandle(name.as_ptr(), button[2].as_mut_ptr())
                .into_result()?;
        }
        unsafe {
            let name = CString::new("/actions/main/in/reposition").unwrap();
            input
                .as_mut()
                .GetActionHandle(name.as_ptr(), button[3].as_mut_ptr())
                .into_result()?;
        };
        let button = unsafe { MaybeUninit::array_assume_init(button) };

        log::debug!("buttons: {:?}", button);
        let action_set = unsafe {
            let mut action_set = MaybeUninit::uninit();
            let action_set_name = CString::new("/actions/main").unwrap();
            input
                .GetActionSetHandle(action_set_name.as_ptr(), action_set.as_mut_ptr())
                .into_result()?;
            action_set.assume_init()
        };
        log::debug!("action_set: {:?}", action_set);
        let instance = Self::create_vk_instance()?;
        let (device, queue) = Self::create_vk_device(&sys, &instance)?;
        let allocator = Arc::new(device.clone().host_to_device_allocator());
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));
        let cmdbuf_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        Ok(Self {
            sys,
            handle: vroverlay,
            action_set,
            buttons: button,
            texture: None,
            display_mode: DisplayMode::default(),
            position_mode: PositionMode::default(),
            reposition: false,
            projector: None,
            overlay_transform: Matrix4::identity(),
            camera_config: None,
            instance,
            allocator,
            descriptor_set_allocator,
            queue,
            cmdbuf_allocator,
            double_buffer: [0, 1].map(|_| {
                crate::create_submittable_image(device.clone()).expect("create_submittable_image")
            }),
            texture_in_use: 1,
            device,
            render_texture: None,
            ipd: None,
        })
    }
    fn ipd(&mut self) -> Result<f32, OpenVrError> {
        if let Some(ipd) = self.ipd {
            return Ok(ipd);
        }
        let mut error = MaybeUninit::<_>::uninit();
        unsafe {
            let ipd = self.sys.pin_mut().GetFloatTrackedDeviceProperty(
                0,
                openvr_sys2::ETrackedDeviceProperty::Prop_UserIpdMeters_Float,
                error.as_mut_ptr(),
            );
            error.assume_init().into_result()?;
            self.ipd = Some(ipd);
            Ok(ipd)
        }
    }
    fn required_extensions(
        sys: &crate::openvr::VRSystem,
        pdev: &PhysicalDevice,
    ) -> vulkano::device::DeviceExtensions {
        let compositor = sys.compositor();
        let mut buf = Vec::new();
        compositor
            .required_extensions(pdev, &mut buf)
            .map(|cstr| cstr.to_str().unwrap())
            .collect()
    }
    fn set_overlay_texture_bounds_internal(&mut self, bounds: Bounds) -> Result<(), OpenVrError> {
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
    fn set_overlay_transformation(&mut self, transform: Matrix4<f32>) -> Result<(), OpenVrError> {
        self.overlay_transform = transform;
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
    fn eye_to_head(&self) -> [Matrix4<f32>; 2] {
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
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum OpenVrError {
    #[error("vulkan device not found")]
    VulkanDeviceNotFound,
    #[error("openvr init error: {0}")]
    Init(#[from] EVRInitError),
    #[error("overlay error: {0}")]
    Overlay(#[from] EVROverlayError),
    #[error("tracked property error: {0}")]
    TrackedProperty(#[from] ETrackedPropertyError),
    #[error("input error: {0}")]
    Input(#[from] EVRInputError),
    #[error("vulkan error: {0}")]
    Vulkan(#[from] vulkano::VulkanError),
    #[error("vulkan validation error: {0}")]
    VulkanValidation(#[from] Box<vulkano::ValidationError>),
    #[error("vulkan error: {0}")]
    VulkanValidated(#[from] vulkano::Validated<vulkano::VulkanError>),
    #[error("vulkan allocation error: {0}")]
    VulkanImageAllocation(#[from] vulkano::Validated<vulkano::image::AllocateImageError>),
    #[error("vulkan error: {0}")]
    RawVulkan(#[from] ash::vk::Result),
    #[error("vulkan loading error: {0}")]
    VulkanLoading(#[from] vulkano::LoadingError),
    #[error("no suitable vulkan queue for graphics")]
    NoGraphicsQueue,
    #[error("projector error: {0}")]
    Projector(#[from] crate::projection::ProjectorError),
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
static VULKAN_LIBRARY: OnceLock<Arc<vulkano::VulkanLibrary>> = OnceLock::new();

fn get_vulkan_library() -> &'static Arc<vulkano::VulkanLibrary> {
    VULKAN_LIBRARY.get_or_init(|| vulkano::VulkanLibrary::new().unwrap())
}

impl VkContext for OpenVr {
    fn vk_device(&self, _instance: &Arc<Instance>) -> (Arc<Device>, Arc<Queue>) {
        (self.device.clone(), self.queue.clone())
    }

    fn vk_instance(&self) -> Arc<vulkano::instance::Instance> {
        self.instance.clone()
    }

    fn vk_allocator(&self) -> Arc<dyn MemoryAllocator> {
        self.allocator.clone()
    }
    fn vk_descriptor_set_allocator(&self) -> Arc<dyn DescriptorSetAllocator> {
        self.descriptor_set_allocator.clone()
    }
    fn vk_command_buffer_allocator(&self) -> Arc<dyn CommandBufferAllocator> {
        self.cmdbuf_allocator.clone()
    }
}

fn transition_layout(
    input_layout: ImageLayout,
    image: &Arc<vulkano::image::Image>,
    queue: &Arc<Queue>,
    cmdbuf_allocator: Arc<dyn CommandBufferAllocator>,
) -> Result<vulkano::sync::fence::Fence, vulkano::Validated<vulkano::VulkanError>> {
    let cmdbuf = unsafe {
        let mut builder = RawRecordingCommandBuffer::new(
            cmdbuf_allocator,
            queue.queue_family_index(),
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                inheritance_info: None,
                ..Default::default()
            },
        )?;
        builder.pipeline_barrier(&DependencyInfo {
            image_memory_barriers: Some(ImageMemoryBarrier {
                src_stages: PipelineStages::ALL_TRANSFER,
                src_access: AccessFlags::TRANSFER_WRITE,
                dst_access: AccessFlags::TRANSFER_READ,
                dst_stages: PipelineStages::ALL_TRANSFER,
                old_layout: input_layout,
                new_layout: ImageLayout::TransferSrcOptimal,
                subresource_range: ImageSubresourceRange {
                    array_layers: 0..1,
                    mip_levels: 0..1,
                    aspects: ImageAspects::COLOR,
                },
                ..ImageMemoryBarrier::image(image.clone())
            })
            .into_iter()
            .collect(),
            ..Default::default()
        })?;
        builder.end()?
    };
    let fence = vulkano::sync::fence::Fence::new(
        queue.device().clone(),
        vulkano::sync::fence::FenceCreateInfo::default(),
    )?;
    let fns = queue.device().fns();
    unsafe {
        (fns.v1_0.queue_submit)(
            queue.handle(),
            1,
            [ash::vk::SubmitInfo::builder()
                .command_buffers(&[cmdbuf.handle()])
                .build()]
            .as_ptr(),
            fence.handle(),
        )
    }
    .result()
    .map_err(vulkano::VulkanError::from)?;
    Ok(fence)
}

impl Vr for OpenVr {
    type Error = OpenVrError;
    fn load_camera_paramter(&mut self) -> Option<StereoCamera> {
        if let Some(cfg) = self.camera_config.as_ref() {
            Some(*cfg)
        } else {
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
            self.camera_config = Some(lhcfg);
            Some(lhcfg)
        }
    }
    fn set_fallback_camera_config(&mut self, cfg: StereoCamera) {
        log::warn!("Using fallback camera config");
        self.camera_config = Some(cfg);
    }
    fn get_render_texture(&mut self) -> Result<Option<Arc<Image>>, Self::Error> {
        // log::debug!("get_render_texture");
        if self.display_mode.projection_mode().is_none() {
            assert!(self.render_texture.is_none());
            self.render_texture =
                Some(self.double_buffer[(self.texture_in_use ^ 1) as usize].clone());
        }
        assert!(self.render_texture.is_some());
        Ok(self.render_texture.clone())
    }
    fn submit_texture(
        &mut self,
        elapsed: Duration,
        fov: &[[f32; 2]; 2],
    ) -> Result<(), Self::Error> {
        let hmd_transform = self.sys.hmd_transform(-elapsed.as_secs_f32()).cast::<f32>();
        if self.reposition {
            self.reposition = false;
            self.position_mode.reposition(hmd_transform);

            let transform: Matrix4<f32> = self.position_mode.transform(hmd_transform).into();
            self.set_overlay_transformation(transform)?;
        } else if matches!(self.position_mode, PositionMode::Hmd { .. }) {
            let transform: Matrix4<f32> = self.position_mode.transform(hmd_transform).into();
            self.set_overlay_transformation(transform)?;
        }
        let output = if self.display_mode.projection_mode().is_some() {
            let new_texture = self.double_buffer[(self.texture_in_use ^ 1) as usize].clone();
            let eye_to_head = self.eye_to_head();
            let view_transforms = eye_to_head.map(|m| hmd_transform * m);
            let ipd = self.ipd()?;
            let projector = self.projector.as_mut().unwrap();
            projector.update_mvps(
                &self.overlay_transform,
                fov,
                &view_transforms,
                &hmd_transform,
            )?;
            projector.set_ipd(ipd);
            let future = projector.project(
                self.allocator.clone(),
                self.cmdbuf_allocator.clone(),
                vulkano::sync::future::now(self.device.clone()),
                &self.queue,
                new_texture.clone(),
            )?;
            future.flush()?;
            future.then_signal_fence().wait(None)?;
            new_texture
        } else {
            let output = self.render_texture.take().unwrap();
            transition_layout(
                ImageLayout::ColorAttachmentOptimal,
                &output,
                &self.queue,
                self.cmdbuf_allocator.clone(),
            )?
            .wait(None)?;
            output
        };
        let texture = TextureState {
            _image: output.clone() as Arc<_>,
            _device: self.device.clone(),
            _queue: self.queue.clone(),
            _instance: self.instance.clone(),
        };
        self.texture.replace(texture);
        let vroverlay = self.sys.overlay();
        // Once we set a texture, the VRSystem starts to depend on Vulkan
        // instance being alive.
        self.sys.hold_vulkan_device(self.device.clone());
        let mut vrimage = openvr_sys2::VRVulkanTextureData_t {
            m_nWidth: crate::CAMERA_SIZE * 2,
            m_nHeight: crate::CAMERA_SIZE,
            m_nFormat: output.format() as u32,
            m_nSampleCount: output.samples() as u32,
            m_nImage: output.handle().as_raw(),
            m_pPhysicalDevice: self.device.physical_device().handle().as_raw() as *mut _,
            m_pDevice: self.device.handle().as_raw() as *mut _,
            m_pQueue: self.queue.handle().as_raw() as *mut _,
            m_pInstance: self.instance.handle().as_raw() as *mut _,
            m_nQueueFamilyIndex: self.queue.queue_family_index(),
        };
        let vrtexture = openvr_sys2::Texture_t {
            handle: &mut vrimage as *mut _ as *mut std::ffi::c_void,
            eType: openvr_sys2::ETextureType::TextureType_Vulkan,
            eColorSpace: openvr_sys2::EColorSpace::ColorSpace_Auto,
        };
        let ret = unsafe {
            vroverlay
                .pin_mut()
                .SetOverlayTexture(self.handle, &vrtexture)
                .into_result()
                .map_err(Into::into)
        };
        self.texture_in_use ^= 1;
        ret
    }
    fn is_synchronized(&self) -> bool {
        false
    }
    fn refresh(&mut self) -> Result<(), Self::Error> {
        // We can only reach here if the overlay is not visible
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(())
    }
    fn set_display_mode(&mut self, mode: DisplayMode) -> Result<(), Self::Error> {
        if self.display_mode == mode {
            return Ok(());
        }
        self.display_mode = mode;
        if let Some(projection_mode) = self.display_mode.projection_mode() {
            let camera_calib = self.load_camera_paramter();
            if self.projector.is_none() {
                self.render_texture = Some(crate::create_submittable_image(self.device.clone())?);
                let mut projector = crate::projection::Projection::new(
                    self.device.clone(),
                    self.allocator.clone(),
                    self.descriptor_set_allocator.clone(),
                    self.render_texture.as_ref().unwrap(),
                    1.0,
                    &camera_calib,
                    ImageLayout::TransferSrcOptimal,
                )?;
                projector.set_mode(projection_mode);
                self.projector = Some(projector);
            }
        } else {
            self.render_texture = None;
            self.projector = None;
        }
        self.sys
            .overlay()
            .pin_mut()
            .SetOverlayFlag(
                self.handle,
                openvr_sys2::VROverlayFlags::VROverlayFlags_SideBySide_Parallel,
                self.display_mode.is_stereo(),
            )
            .into_result()?;
        let bounds = match mode {
            DisplayMode::Flat { eye: Eye::Left } => crate::vrapi::Bounds {
                umin: 0.0,
                umax: 0.5,
                vmin: 0.0,
                vmax: 1.0,
            },
            DisplayMode::Flat { eye: Eye::Right } => crate::vrapi::Bounds {
                umin: 0.5,
                umax: 1.0,
                vmin: 0.0,
                vmax: 1.0,
            },
            DisplayMode::Stereo { .. } | DisplayMode::Direct => crate::vrapi::Bounds {
                umin: 0.0,
                umax: 1.0,
                vmin: 0.0,
                vmax: 1.0,
            },
        };
        self.set_overlay_texture_bounds_internal(bounds)
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
    fn set_position_mode(&mut self, mode: PositionMode) -> Result<(), Self::Error> {
        self.position_mode = mode;
        match mode {
            PositionMode::Absolute { transform } => {
                let transform: Matrix4<f32> = transform.into();
                self.set_overlay_transformation(transform.cast())?;
            }
            PositionMode::Sticky { .. } => {
                self.reposition = true;
            }
            _ => (),
        }
        Ok(())
    }
    fn acknowledge_quit(&mut self) {
        self.sys.pin_mut().AcknowledgeQuit_Exiting();
    }
    fn poll_next_event(&mut self) -> Result<Option<Event>, Self::Error> {
        use openvr_sys2::VREvent_t;
        loop {
            let Some(openvr_event): Option<VREvent_t> = (unsafe {
                let mut event = MaybeUninit::uninit();
                let has_event = self.sys.pin_mut().PollNextEvent(
                    event.as_mut_ptr() as *mut _,
                    std::mem::size_of::<openvr_sys2::VREvent_t>() as u32,
                );
                has_event.then(|| event.assume_init())
            }) else {
                return Ok(None);
            };
            match openvr_sys2::EVREventType::try_from(openvr_event.eventType) {
                Ok(openvr_sys2::EVREventType::VREvent_IpdChanged) => {
                    let ipd = unsafe { openvr_event.data.ipd.ipdMeters };
                    self.ipd = Some(ipd);
                }
                Ok(openvr_sys2::EVREventType::VREvent_Quit) => return Ok(Some(Event::RequestExit)),
                _ => (),
            }
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
        // log::debug!("getting action {action:?}");
        let vrinput = unsafe { Pin::new_unchecked(&mut *openvr_sys2::VRInput()) };
        let action_data = unsafe {
            let mut action_data = MaybeUninit::uninit();
            let result = vrinput
                .GetDigitalActionData(
                    action_handle,
                    action_data.as_mut_ptr(),
                    std::mem::size_of::<openvr_sys2::VRInputValueHandle_t>() as u32,
                    openvr_sys2::vr::k_ulInvalidInputValueHandle,
                )
                .into_result();
            if result.is_err() {
                log::error!("GetDigitalActionData failed: {:?}", result);
                return Ok(false);
            }
            action_data.assume_init()
        };
        // log::debug!("action_data: {}", action_data.bState);
        Ok(action_data.bState)
    }
}

pub(crate) struct OpenXr {
    instance: openxr::Instance,
    overlay_visible: bool,
    position_mode: PositionMode,
    reposition: bool,
    display_mode: DisplayMode,
    allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    cmdbuf_allocator: Arc<StandardCommandBufferAllocator>,
    action_set: openxr::ActionSet,
    action_button1: openxr::Action<bool>,
    action_button2: openxr::Action<bool>,
    action_debug: openxr::Action<bool>,
    action_reposition: openxr::Action<bool>,
    camera_config: Option<StereoCamera>,

    session_state: openxr::SessionState,
    session: openxr::Session<openxr::Vulkan>,
    frame_waiter: openxr::FrameWaiter,
    frame_stream: openxr::FrameStream<openxr::Vulkan>,
    swapchain: openxr::Swapchain<openxr::Vulkan>,
    swapchain_images: Vec<Arc<Image>>,
    frame_state: Option<openxr::FrameState>,
    space: openxr::Space,
    saved_poses: [(UnitQuaternion<f32>, Vector3<f32>); 2],
    saved_overlay_pose: Option<openxr::Posef>,

    device: Arc<Device>,
    queue: Arc<Queue>,
    vk_instance: Arc<Instance>,

    projector: Option<crate::projection::Projection>,
    render_texture: Option<Arc<Image>>,
}
fn affine_to_posef(t: Affine3<f32>) -> openxr::Posef {
    let m = t.to_homogeneous();
    let r: Matrix3<f32> = m.fixed_columns::<3>(0).fixed_rows::<3>(0).into();
    let rotation = nalgebra::geometry::Rotation3::from_matrix(&r);
    let quaternion = UnitQuaternion::from_rotation_matrix(&rotation);
    let quaternion = &quaternion.as_ref().coords;
    let translation: nalgebra::Vector3<f32> =
        [m.data.0[3][0], m.data.0[3][1], m.data.0[3][2]].into();
    openxr::Posef {
        orientation: openxr::Quaternionf {
            x: quaternion.x,
            y: quaternion.y,
            z: quaternion.z,
            w: quaternion.w,
        },
        position: openxr::Vector3f {
            x: translation.x,
            y: translation.y,
            z: translation.z,
        },
    }
}

fn posef_to_nalgebra(posef: openxr::Posef) -> (UnitQuaternion<f32>, nalgebra::Vector3<f32>) {
    let quaternion = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
        posef.orientation.w,
        posef.orientation.x,
        posef.orientation.y,
        posef.orientation.z,
    ));
    let translation: nalgebra::Vector3<f32> =
        [posef.position.x, posef.position.y, posef.position.z].into();
    (quaternion, translation)
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum OpenXrError {
    #[error("cannot load openxr loader: {0}")]
    XrLoad(#[from] openxr::LoadError),
    #[error("cannot load vulkan library: {0}")]
    VkLoad(#[from] vulkano::LoadingError),
    #[error("vulkan error: {0}")]
    Vk(#[from] vulkano::VulkanError),
    #[error("vulkan error: {0}")]
    ValidatedVk(#[from] vulkano::Validated<vulkano::VulkanError>),
    #[error("cannot allocate image: {0}")]
    AlocateImage(#[from] vulkano::Validated<vulkano::image::AllocateImageError>),
    #[error("xr: {0}")]
    Xr(#[from] openxr::sys::Result),
    #[error("vulkan error: {0}")]
    RawVk(#[from] ash::vk::Result),
    #[error("no graphics queue found")]
    NoGraphicsQueue,
    #[error("no usable texture format")]
    NoFormat,
    #[error("{0}")]
    Projection(#[from] crate::projection::ProjectorError),
    #[error("cannot allocate device memory: {0}")]
    Allocator(#[from] vulkano::memory::allocator::MemoryAllocatorError),
    #[error("vulkan version doesn't meet requirements: {0}")]
    VersionNotSupported(vulkano::Version),
    #[error("no supported blend mode")]
    NoSupportedBlendMode,
}

impl OpenXr {
    fn create_vk_device(
        xr_instance: &openxr::Instance,
        xr_system: openxr::SystemId,
        instance: &Arc<Instance>,
    ) -> Result<(Arc<Device>, Arc<Queue>), OpenXrError> {
        let vk_requirements = xr_instance.graphics_requirements::<openxr::Vulkan>(xr_system)?;
        let physical_device = unsafe {
            let physical_device =
                xr_instance.vulkan_graphics_device(xr_system, instance.handle().as_raw() as _)?;
            vulkano::device::physical::PhysicalDevice::from_handle(
                instance.clone(),
                ash::vk::PhysicalDevice::from_raw(physical_device as _),
            )
        }?;
        let min_version = vulkano::Version::major_minor(
            vk_requirements.min_api_version_supported.major() as u32,
            vk_requirements.min_api_version_supported.minor() as u32,
        );
        if physical_device.api_version() < min_version {
            return Err(OpenXrError::VersionNotSupported(
                physical_device.api_version(),
            ));
        }
        let queue_family = physical_device
            .queue_family_properties()
            .iter()
            .position(|qf| qf.queue_flags.contains(QueueFlags::GRAPHICS))
            .ok_or(OpenXrError::NoGraphicsQueue)?;
        log::debug!("queue family: {queue_family}");
        let queue_create_info = ash::vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family as u32)
            .queue_priorities(std::slice::from_ref(&1.0))
            .build();
        let create_info = ash::vk::DeviceCreateInfo::builder()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .build();
        let vulkano_create_info = vulkano::device::DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index: queue_family as u32,
                queues: vec![1.0],
                ..Default::default()
            }],
            physical_devices: [physical_device.clone()].into_iter().collect(),
            ..Default::default()
        };
        let (device, mut queues) = unsafe {
            vulkano::device::Device::from_handle(
                physical_device.clone(),
                ash::vk::Device::from_raw(
                    xr_instance
                        .create_vulkan_device(
                            xr_system,
                            get_instance_proc_addr,
                            physical_device.handle().as_raw() as _,
                            (&create_info) as *const _ as _,
                        )?
                        .map_err(ash::vk::Result::from_raw)? as _,
                ),
                vulkano_create_info,
            )
        };
        Ok((device, queues.next().unwrap()))
    }

    fn create_vk_instance(
        xr_instance: &openxr::Instance,
        xr_system: openxr::SystemId,
    ) -> Result<Arc<Instance>, OpenXrError> {
        let vk_requirements = xr_instance.graphics_requirements::<openxr::Vulkan>(xr_system)?;
        let extensions = *get_vulkan_library().supported_extensions();
        let max_version = vulkano::Version::major_minor(
            vk_requirements.max_api_version_supported.major() as u32,
            vk_requirements.max_api_version_supported.minor() as u32,
        );
        let vulkano_create_info = vulkano::instance::InstanceCreateInfo {
            max_api_version: Some(max_version),
            enabled_extensions: extensions,
            // enabled_layers: vec!["VK_LAYER_KHRONOS_validation".to_owned()],
            ..Default::default()
        };
        let extensions = extensions
            .into_iter()
            .filter(|(_, enabled)| *enabled)
            .map(|(ext, _)| std::ffi::CString::new(ext).unwrap())
            .collect::<Vec<_>>();
        let extensions = extensions
            .iter()
            .map(|s| s.as_c_str().as_ptr())
            .collect::<Vec<_>>();
        let application_info = ash::vk::ApplicationInfo::builder()
            .api_version(vulkano::Version::V1_6.try_into().unwrap());
        let create_info = ash::vk::InstanceCreateInfo::builder()
            .enabled_extension_names(&extensions)
            .application_info(&application_info)
            //.enabled_layer_names(&[b"VK_LAYER_KHRONOS_validation\0".as_ptr() as _])
            .build();
        let instance = unsafe {
            xr_instance.create_vulkan_instance(
                xr_system,
                get_instance_proc_addr,
                (&create_info) as *const _ as _,
            )?
        }
        .map_err(ash::vk::Result::from_raw)?;
        let instance = ash::vk::Instance::from_raw(instance as _);
        Ok(unsafe {
            Instance::from_handle(get_vulkan_library().clone(), instance, vulkano_create_info)
        })
    }

    fn composition_layers<'a>(
        saved_overlay_pose: &'a Option<openxr::Posef>,
        swapchain: &'a openxr::Swapchain<openxr::Vulkan>,
        space: &'a openxr::Space,
        is_stereo: bool,
    ) -> Option<(
        openxr::CompositionLayerQuad<'a, openxr::Vulkan>,
        openxr::CompositionLayerQuad<'a, openxr::Vulkan>,
    )> {
        saved_overlay_pose.map(|overlay_posef| {
            let left = openxr::CompositionLayerQuad::<openxr::Vulkan>::new()
                .eye_visibility(EyeVisibility::LEFT)
                .pose(overlay_posef)
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(swapchain)
                        .image_rect(Rect2Di {
                            offset: Offset2Di { x: 0, y: 0 },
                            extent: Extent2Di {
                                width: crate::CAMERA_SIZE as i32,
                                height: crate::CAMERA_SIZE as i32,
                            },
                        }),
                )
                .space(space)
                .size(Extent2Df {
                    width: 1.0,
                    height: 1.0,
                });
            let right = openxr::CompositionLayerQuad::<openxr::Vulkan>::new()
                .eye_visibility(EyeVisibility::RIGHT)
                .pose(overlay_posef)
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(swapchain)
                        .image_rect(Rect2Di {
                            offset: Offset2Di {
                                x: if is_stereo {
                                    crate::CAMERA_SIZE as i32
                                } else {
                                    0
                                },
                                y: 0,
                            },
                            extent: Extent2Di {
                                width: crate::CAMERA_SIZE as i32,
                                height: crate::CAMERA_SIZE as i32,
                            },
                        }),
                )
                .space(space)
                .size(Extent2Df {
                    width: 1.0,
                    height: 1.0,
                });
            (left, right)
        })
    }

    pub(crate) fn new(placement: u32) -> Result<Self, OpenXrError> {
        let entry = unsafe { openxr::Entry::load()? };
        let mut extension = openxr::ExtensionSet::default();
        extension.extx_overlay = true;
        extension.khr_vulkan_enable2 = true;
        extension.khr_convert_timespec_time = true;
        let instance = entry.create_instance(
            &ApplicationInfo {
                application_name: crate::APP_NAME,
                application_version: crate::APP_VERSION,
                engine_name: "engine",
                engine_version: 0,
            },
            &extension,
            &[],
        )?;
        let system = instance.system(openxr::FormFactor::HEAD_MOUNTED_DISPLAY)?;
        let blend_modes = instance.enumerate_environment_blend_modes(
            system,
            openxr::ViewConfigurationType::PRIMARY_STEREO,
        )?;
        if !blend_modes.contains(&openxr::EnvironmentBlendMode::OPAQUE) {
            return Err(OpenXrError::NoSupportedBlendMode);
        }
        let vk_instance = Self::create_vk_instance(&instance, system)?;
        let (device, queue) = Self::create_vk_device(&instance, system, &vk_instance)?;
        let allocator = Arc::new(device.clone().host_to_device_allocator());
        let cmdbuf_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        let action_set = instance.create_action_set("main", "main", 0)?;
        let overlay = openxr::sys::SessionCreateInfoOverlayEXTX {
            ty: openxr::sys::SessionCreateInfoOverlayEXTX::TYPE,
            next: std::ptr::null(),
            create_flags: OverlaySessionCreateFlagsEXTX::EMPTY,
            session_layers_placement: placement,
        };
        let binding = openxr::sys::GraphicsBindingVulkanKHR {
            ty: openxr::sys::GraphicsBindingVulkanKHR::TYPE,
            next: &overlay as *const _ as *const _,
            instance: vk_instance.handle().as_raw() as _,
            physical_device: device.physical_device().handle().as_raw() as _,
            device: device.handle().as_raw() as _,
            queue_family_index: queue.queue_family_index(),
            queue_index: queue.id_within_family(),
        };
        let info = openxr::sys::SessionCreateInfo {
            ty: openxr::sys::SessionCreateInfo::TYPE,
            next: &binding as *const _ as *const _,
            create_flags: Default::default(),
            system_id: system,
        };
        let mut out = openxr::sys::Session::NULL;
        let ret = unsafe { (instance.fp().create_session)(instance.as_raw(), &info, &mut out) };
        if ret.into_raw() < 0 {
            return Err(ret.into());
        }
        let (session, frame_waiter, frame_stream) = unsafe {
            openxr::Session::<openxr::Vulkan>::from_raw(instance.clone(), out, Box::new(()))
        };
        let formats = session.enumerate_swapchain_formats()?;
        if !formats
            .iter()
            .contains(&(vulkano::format::Format::R8G8B8A8_UNORM as u32))
        {
            return Err(OpenXrError::NoFormat);
        }
        let swapchain = session.create_swapchain(&openxr::SwapchainCreateInfo {
            array_size: 1,
            face_count: 1,
            create_flags: Default::default(),
            usage_flags: openxr::SwapchainUsageFlags::COLOR_ATTACHMENT
                | openxr::SwapchainUsageFlags::TRANSFER_DST,
            format: vulkano::format::Format::R8G8B8A8_UNORM as u32,
            sample_count: 1,
            width: crate::CAMERA_SIZE * 2,
            height: crate::CAMERA_SIZE,
            mip_count: 1,
        })?;
        log::debug!("created swapchain");
        let swapchain_images = swapchain
            .enumerate_images()?
            .into_iter()
            .map(|handle| {
                let handle = ash::vk::Image::from_raw(handle);
                let raw_image = unsafe {
                    vulkano::image::sys::RawImage::from_handle_borrowed(
                        device.clone(),
                        handle,
                        ImageCreateInfo {
                            format: vulkano::format::Format::R8G8B8A8_UNORM,
                            extent: [CAMERA_SIZE * 2, CAMERA_SIZE, 1],
                            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
                            ..Default::default()
                        },
                    )?
                };
                // SAFETY: OpenXR guarantees that the image is a swapchain image, thus has memory backing it.
                let image = unsafe { raw_image.assume_bound() };
                Ok::<_, OpenXrError>(Arc::new(image))
            })
            .try_collect()?;
        log::debug!("got swapchain images");
        let action_button1 = action_set.create_action("button1", "Button1", &[])?;
        let action_button2 = action_set.create_action("button2", "Button2", &[])?;
        let action_debug = action_set.create_action("debug", "Debug", &[])?;
        let action_reposition = action_set.create_action("reposition", "Reposition", &[])?;
        instance.suggest_interaction_profile_bindings(
            instance.string_to_path("/interaction_profiles/htc/vive_controller")?,
            &[
                openxr::Binding::new(
                    &action_button1,
                    instance.string_to_path("/user/hand/left/input/menu/click")?,
                ),
                openxr::Binding::new(
                    &action_button2,
                    instance.string_to_path("/user/hand/right/input/menu/click")?,
                ),
                openxr::Binding::new(
                    &action_debug,
                    instance.string_to_path("/user/hand/left/input/trigger/click")?,
                ),
                openxr::Binding::new(
                    &action_reposition,
                    instance.string_to_path("/user/hand/right/input/trigger/click")?,
                ),
            ],
        )?;
        instance.suggest_interaction_profile_bindings(
            instance.string_to_path("/interaction_profiles/valve/index_controller")?,
            &[
                openxr::Binding::new(
                    &action_button1,
                    instance.string_to_path("/user/hand/left/input/b/click")?,
                ),
                openxr::Binding::new(
                    &action_button2,
                    instance.string_to_path("/user/hand/right/input/b/click")?,
                ),
                openxr::Binding::new(
                    &action_debug,
                    instance.string_to_path("/user/hand/left/input/trigger/click")?,
                ),
                openxr::Binding::new(
                    &action_reposition,
                    instance.string_to_path("/user/hand/right/input/a/click")?,
                ),
            ],
        )?;
        instance.suggest_interaction_profile_bindings(
            instance.string_to_path("/interaction_profiles/khr/simple_controller")?,
            &[
                openxr::Binding::new(
                    &action_button1,
                    instance.string_to_path("/user/hand/left/input/menu/click")?,
                ),
                openxr::Binding::new(
                    &action_button2,
                    instance.string_to_path("/user/hand/right/input/menu/click")?,
                ),
                openxr::Binding::new(
                    &action_debug,
                    instance.string_to_path("/user/hand/left/input/select/click")?,
                ),
                openxr::Binding::new(
                    &action_reposition,
                    instance.string_to_path("/user/hand/right/input/select/click")?,
                ),
            ],
        )?;
        let space =
            session.create_reference_space(ReferenceSpaceType::STAGE, openxr::Posef::IDENTITY)?;
        log::debug!("created actions");
        session.attach_action_sets(&[&action_set])?;

        Ok(Self {
            instance,
            camera_config: None,
            overlay_visible: false,
            position_mode: PositionMode::default(),
            reposition: false,
            display_mode: DisplayMode::default(),
            action_button1,
            action_button2,
            action_debug,
            action_reposition,

            action_set,

            session_state: openxr::SessionState::IDLE,
            session,
            frame_waiter,
            frame_stream,
            swapchain,
            swapchain_images,
            frame_state: None,
            space,
            saved_poses: [Default::default(); 2],
            saved_overlay_pose: None,

            allocator,
            descriptor_set_allocator,
            cmdbuf_allocator,
            vk_instance,
            device,
            queue,

            projector: None,
            render_texture: None,
        })
    }
}

unsafe extern "system" fn get_instance_proc_addr(
    instance: openxr::sys::platform::VkInstance,
    name: *const std::ffi::c_char,
) -> Option<unsafe extern "system" fn()> {
    let instance = ash::vk::Instance::from_raw(instance as _);
    let library = get_vulkan_library();
    library.get_instance_proc_addr(instance, name)
}

impl VkContext for OpenXr {
    fn vk_device(&self, _instance: &Arc<Instance>) -> (Arc<Device>, Arc<Queue>) {
        (self.device.clone(), self.queue.clone())
    }
    fn vk_instance(&self) -> Arc<Instance> {
        self.vk_instance.clone()
    }
    fn vk_allocator(&self) -> Arc<dyn MemoryAllocator> {
        self.allocator.clone()
    }
    fn vk_descriptor_set_allocator(&self) -> Arc<dyn DescriptorSetAllocator> {
        self.descriptor_set_allocator.clone()
    }
    fn vk_command_buffer_allocator(&self) -> Arc<dyn CommandBufferAllocator> {
        self.cmdbuf_allocator.clone()
    }
}

impl Vr for OpenXr {
    fn acknowledge_quit(&mut self) {
        // intentionally left blank
    }

    type Error = OpenXrError;

    fn load_camera_paramter(&mut self) -> Option<StereoCamera> {
        self.camera_config
    }

    fn set_fallback_camera_config(&mut self, cfg: StereoCamera) {
        self.camera_config = Some(cfg);
    }

    fn submit_texture(
        &mut self,
        elapsed: Duration,
        fov: &[[f32; 2]; 2],
    ) -> Result<(), Self::Error> {
        log::trace!("submit texture");
        let frame_state = self.frame_state.as_ref().unwrap();
        let now = self.instance.now()?;
        let time_at_capture =
            openxr::Time::from_nanos((now.as_nanos() as u128 - elapsed.as_nanos()) as i64);
        let (view_state_flags, views) = self.session.locate_views(
            ViewConfigurationType::PRIMARY_STEREO,
            time_at_capture,
            &self.space,
        )?;
        let view_poses = if !view_state_flags.contains(ViewStateFlags::ORIENTATION_VALID)
            || !view_state_flags.contains(ViewStateFlags::POSITION_VALID)
        {
            log::trace!("view_state_flags: {:?}", view_state_flags);
            self.saved_poses
        } else {
            log::trace!("update pose");
            let poses = [0, 1].map(|id| posef_to_nalgebra(views[id].pose));
            self.saved_poses = poses;
            poses
        };
        let rotation_center = UnitQuaternion::from_quaternion(
            (view_poses[0].0.as_ref() + view_poses[1].0.as_ref()) / 2.0,
        );
        let center: Translation3<f32> = ((view_poses[0].1 + view_poses[1].1) / 2.0).into();
        let hmd_transform = center.to_homogeneous() * rotation_center.to_homogeneous();
        if self.reposition {
            self.position_mode.reposition(hmd_transform);
            self.reposition = false;
        }
        let transform = self.position_mode.transform(hmd_transform);
        let overlay_posef = affine_to_posef(transform);
        self.saved_overlay_pose = Some(overlay_posef);
        if self.display_mode.projection_mode().is_some() {
            // Apply projection
            let image = self.swapchain.acquire_image()? as usize;
            let view_transforms = [
                Translation3::from(view_poses[0].1).to_homogeneous()
                    * view_poses[0].0.to_homogeneous(),
                Translation3::from(view_poses[1].1).to_homogeneous()
                    * view_poses[1].0.to_homogeneous(),
            ];
            let ipd = view_poses[1].1.x - view_poses[0].1.x;
            self.swapchain.wait_image(openxr::Duration::INFINITE)?;
            let output = self.swapchain_images[image].clone();
            let projector = self.projector.as_mut().unwrap();
            projector.update_mvps(transform.matrix(), fov, &view_transforms, &hmd_transform)?;
            projector.set_ipd(ipd);
            let future = projector.project(
                self.allocator.clone(),
                self.cmdbuf_allocator.clone(),
                vulkano::sync::future::now(self.device.clone()),
                &self.queue,
                output,
            )?;
            future.flush()?;
            future.then_signal_fence().wait(None)?;
        } else {
            self.render_texture.take();
        }
        self.swapchain.release_image()?;
        let (left, right) = Self::composition_layers(
            &self.saved_overlay_pose,
            &self.swapchain,
            &self.space,
            self.display_mode.is_stereo(),
        )
        .unwrap();
        self.frame_stream.end(
            frame_state.predicted_display_time,
            EnvironmentBlendMode::OPAQUE,
            &[&left, &right],
        )?;
        Ok(())
    }

    fn is_synchronized(&self) -> bool {
        true
    }

    fn refresh(&mut self) -> Result<(), Self::Error> {
        log::trace!("refresh");
        if !self.overlay_visible {
            std::thread::sleep(std::time::Duration::from_millis(100));
            return Ok(());
        }
        let frame_state = self.frame_state.insert(self.frame_waiter.wait()?);
        self.frame_stream.begin()?;
        if let Some((left, right)) = Self::composition_layers(
            &self.saved_overlay_pose,
            &self.swapchain,
            &self.space,
            self.display_mode.is_stereo(),
        ) {
            log::trace!("reuse last image {:?}", frame_state.predicted_display_time);
            self.frame_stream.end(
                frame_state.predicted_display_time,
                EnvironmentBlendMode::OPAQUE,
                &[&left, &right],
            )?;
        } else {
            log::trace!("no saved overlay pose");
            self.frame_stream.end(
                frame_state.predicted_display_time,
                EnvironmentBlendMode::OPAQUE,
                &[],
            )?;
        }
        Ok(())
    }

    fn get_render_texture(&mut self) -> Result<Option<Arc<Image>>, Self::Error> {
        if (self.session_state != openxr::SessionState::FOCUSED
            && self.session_state != openxr::SessionState::VISIBLE
            && self.session_state != openxr::SessionState::SYNCHRONIZED
            && self.session_state != openxr::SessionState::READY)
            || !self.overlay_visible
        {
            log::debug!("VR runtime not ready");
            return Ok(None);
        }
        let frame_state = self.frame_state.insert(self.frame_waiter.wait()?);
        self.frame_stream.begin()?;
        if !frame_state.should_render {
            self.frame_stream.end(
                frame_state.predicted_display_time,
                EnvironmentBlendMode::OPAQUE,
                &[],
            )?;
            return Ok(None);
        }
        if self.display_mode.projection_mode().is_some() {
            log::trace!("render to intermediate texture");
            assert!(self.render_texture.is_some());
            return Ok(self.render_texture.clone());
        }
        log::trace!("render to swapchain image");
        let image = self.swapchain.acquire_image()? as usize;
        self.render_texture = Some(self.swapchain_images[image].clone());
        self.swapchain.wait_image(openxr::Duration::INFINITE)?;
        Ok(self.render_texture.clone())
    }

    fn set_display_mode(&mut self, mode: DisplayMode) -> Result<(), Self::Error> {
        self.display_mode = mode;
        if let Some(projection_mode) = self.display_mode.projection_mode() {
            let camera_calib = self.load_camera_paramter();
            if self.projector.is_none() {
                self.render_texture = Some(crate::create_submittable_image(self.device.clone())?);
                let mut projector = crate::projection::Projection::new(
                    self.device.clone(),
                    self.allocator.clone(),
                    self.descriptor_set_allocator.clone(),
                    self.render_texture.as_ref().unwrap(),
                    1.0,
                    &camera_calib,
                    ImageLayout::ColorAttachmentOptimal,
                )?;
                projector.set_mode(projection_mode);
                self.projector = Some(projector);
            }
        } else {
            self.render_texture = None;
            self.projector = None;
        }
        Ok(())
    }

    fn show_overlay(&mut self) -> Result<(), Self::Error> {
        if !self.overlay_visible {
            log::debug!("show overlay, {:?}", self.session_state);
            self.overlay_visible = true;
        }
        Ok(())
    }

    fn hide_overlay(&mut self) -> Result<(), Self::Error> {
        if self.overlay_visible {
            // HACK!: show a zero sized quad to hide the overlay. It's a bit ugly we
            // blocks the mainloop here to wait for a frame
            let frame_state = self.frame_waiter.wait()?;
            self.frame_stream.begin()?;
            let empty = openxr::CompositionLayerQuad::<openxr::Vulkan>::new()
                .eye_visibility(EyeVisibility::BOTH)
                .pose(openxr::Posef::IDENTITY)
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(&self.swapchain)
                        .image_rect(Rect2Di {
                            offset: Offset2Di { x: 0, y: 0 },
                            extent: Extent2Di {
                                width: 1,
                                height: 1,
                            },
                        }),
                )
                .space(&self.space)
                .size(Extent2Df {
                    width: 0.0,
                    height: 0.0,
                });
            self.frame_stream.end(
                frame_state.predicted_display_time,
                EnvironmentBlendMode::OPAQUE,
                &[&empty],
            )?;
            self.overlay_visible = false;
        }
        Ok(())
    }

    fn set_position_mode(&mut self, mode: PositionMode) -> Result<(), Self::Error> {
        self.position_mode = mode;
        if matches!(mode, PositionMode::Sticky { .. }) {
            self.reposition = true;
        }
        Ok(())
    }

    fn poll_next_event(&mut self) -> Result<Option<Event>, Self::Error> {
        let mut event = EventDataBuffer::default();
        let ret = loop {
            let event = self.instance.poll_event(&mut event)?;
            let Some(event) = event else { break None };
            use openxr::Event as XrEvent;
            match event {
                XrEvent::InstanceLossPending(_) => break Some(Event::RequestExit),
                XrEvent::SessionStateChanged(ssc) => {
                    use openxr::SessionState;
                    log::debug!(
                        "session state changed: {:?}, visible: {}",
                        ssc.state(),
                        self.overlay_visible
                    );
                    self.session_state = ssc.state();
                    match self.session_state {
                        SessionState::EXITING | SessionState::LOSS_PENDING => {
                            self.session.end()?;
                            break Some(Event::RequestExit);
                        }
                        SessionState::READY => {
                            log::debug!("begin session");
                            self.session
                                .begin(openxr::ViewConfigurationType::PRIMARY_STEREO)?;
                        }
                        SessionState::STOPPING => {
                            if self.overlay_visible {
                                self.session.end()?;
                            }
                        }
                        _ => (),
                    }
                }
                XrEvent::EventsLost(_) => (), // ? should we do something?
                _ => (),
            }
        };
        Ok(ret)
    }

    fn update_action_state(&mut self) -> Result<(), Self::Error> {
        self.session
            .sync_actions(&[openxr::ActiveActionSet::new(&self.action_set)])?;
        Ok(())
    }

    fn get_action_state(&self, action: Action) -> Result<bool, Self::Error> {
        Ok(match action {
            Action::Button1 => self
                .action_button1
                .state(&self.session, openxr::Path::NULL)?,
            Action::Button2 => self
                .action_button2
                .state(&self.session, openxr::Path::NULL)?,
            Action::Debug => self.action_debug.state(&self.session, openxr::Path::NULL)?,
            Action::Reposition => self
                .action_reposition
                .state(&self.session, openxr::Path::NULL)?,
        }
        .current_state)
    }

    fn wait_for_ready(&mut self) -> Result<(), Self::Error> {
        log::debug!("current state {:?}", self.session_state);
        while self.session_state != openxr::SessionState::FOCUSED
            && self.session_state != openxr::SessionState::VISIBLE
            && self.session_state != openxr::SessionState::SYNCHRONIZED
            && self.session_state != openxr::SessionState::READY
        {
            log::debug!("VR runtime not ready");
            self.poll_next_event()?;
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        while self.session_state != openxr::SessionState::SYNCHRONIZED
            && self.session_state != openxr::SessionState::FOCUSED
            && self.session_state != openxr::SessionState::VISIBLE
        {
            let frame_state = self.frame_state.insert(self.frame_waiter.wait()?);
            self.frame_stream.begin()?;
            self.frame_stream.end(
                frame_state.predicted_display_time,
                EnvironmentBlendMode::OPAQUE,
                &[],
            )?;
            self.poll_next_event()?;
        }
        Ok(())
    }
}
