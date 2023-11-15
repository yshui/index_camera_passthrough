use nalgebra::{matrix, Matrix4};
use openvr_sys2::{ETrackedPropertyError, EVRInitError, EVRInputError, EVROverlayError};
use openxr::ApplicationInfo;
use std::{
    ffi::CString,
    mem::MaybeUninit,
    pin::Pin,
    sync::{Arc, OnceLock},
    time::Duration,
};
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator,
        sys::{CommandBufferBeginInfo, UnsafeCommandBufferBuilder},
        CommandBufferLevel, CommandBufferUsage,
    },
    descriptor_set::allocator::{
        StandardDescriptorSetAlloc, StandardDescriptorSetAllocator,
        StandardDescriptorSetAllocatorCreateInfo,
    },
    device::{physical::PhysicalDevice, Device, Queue, QueueCreateInfo, QueueFlags},
    image::{ImageAspects, ImageLayout, ImageSubresourceRange},
    instance::Instance,
    memory::allocator::{MemoryAllocator, StandardMemoryAllocator},
    sync::{AccessFlags, DependencyInfo, GpuFuture, ImageMemoryBarrier, PipelineStages},
    Handle, VulkanObject,
};

use serde::{Deserialize, Serialize};

use crate::{
    config::{DisplayMode, Eye, PositionMode},
    APP_KEY, APP_NAME,
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
    /// IPD has changed
    IpdChanged(f32),
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Action {
    Button1 = 0,
    Button2 = 1,
    Debug = 2,
}

pub(crate) trait VkContext {
    fn vk_device(&self, instance: &Arc<Instance>) -> (Arc<Device>, Arc<Queue>);
    fn vk_instance(&self) -> Arc<Instance>;
    fn vk_allocator(&self) -> Arc<dyn MemoryAllocator>;
    fn vk_descriptor_set_allocator(&self) -> &StandardDescriptorSetAllocator;
    fn vk_command_buffer_allocator(&self) -> &StandardCommandBufferAllocator;
}

pub(crate) trait Vr: VkContext {
    type Error: Send + Sync + 'static;
    fn load_camera_paramter(&mut self) -> Option<StereoCamera>;
    fn ipd(&self) -> Result<f32, Self::Error>;
    fn eye_to_head(&self) -> [Matrix4<f64>; 2];
    /// Submit the render texture to overlay.
    ///
    /// Must have called `render_texture` before calling this function.
    ///
    /// # Arguments
    ///
    /// - `elapsed`: duration since the image was captured.
    fn submit_texture(
        &mut self,
        layout: ImageLayout,
        elapsed: Duration,
        fov: &[[f64; 2]; 2],
    ) -> Result<(), Self::Error>;
    fn set_position_mode(&mut self, mode: PositionMode) -> Result<(), Self::Error>;
    /// Change the display mode of the overlay.
    ///
    /// This invalidates previously returned render texture.
    fn set_display_mode(&mut self, mode: DisplayMode) -> Result<(), Self::Error>;
    fn show_overlay(&mut self) -> Result<(), Self::Error>;
    fn hide_overlay(&mut self) -> Result<(), Self::Error>;
    fn acknowledge_quit(&mut self);
    fn get_render_texture(&mut self) -> Result<Arc<vulkano::image::Image>, Self::Error>;
    fn poll_next_event(&mut self) -> Option<Event>;
    fn update_action_state(&mut self) -> Result<(), Self::Error>;
    fn get_action_state(&self, action: Action) -> Result<bool, Self::Error>;
}

struct VrMapError<T, F>(T, F);

impl<T: VkContext, F> VkContext for VrMapError<T, F> {
    fn vk_allocator(&self) -> Arc<dyn MemoryAllocator> {
        self.0.vk_allocator()
    }
    fn vk_descriptor_set_allocator(&self) -> &StandardDescriptorSetAllocator {
        self.0.vk_descriptor_set_allocator()
    }
    fn vk_device(&self, instance: &Arc<Instance>) -> (Arc<Device>, Arc<Queue>) {
        self.0.vk_device(instance)
    }
    fn vk_instance(&self) -> Arc<Instance> {
        self.0.vk_instance()
    }
    fn vk_command_buffer_allocator(&self) -> &StandardCommandBufferAllocator {
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
    fn ipd(&self) -> Result<f32, Self::Error> {
        self.0.ipd().map_err(&self.1)
    }
    fn eye_to_head(&self) -> [Matrix4<f64>; 2] {
        self.0.eye_to_head()
    }
    fn submit_texture(
        &mut self,
        layout: ImageLayout,
        elapsed: Duration,
        fov: &[[f64; 2]; 2],
    ) -> Result<(), Self::Error> {
        self.0.submit_texture(layout, elapsed, fov).map_err(&self.1)
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
    fn get_render_texture(&mut self) -> Result<Arc<vulkano::image::Image>, Self::Error> {
        self.0.get_render_texture().map_err(&self.1)
    }
    fn hide_overlay(&mut self) -> Result<(), Self::Error> {
        self.0.hide_overlay().map_err(&self.1)
    }
    fn poll_next_event(&mut self) -> Option<Event> {
        self.0.poll_next_event()
    }
    fn get_action_state(&self, action: Action) -> Result<bool, Self::Error> {
        self.0.get_action_state(action).map_err(&self.1)
    }
    fn update_action_state(&mut self) -> Result<(), Self::Error> {
        self.0.update_action_state().map_err(&self.1)
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
    buttons: [openvr_sys2::VRActionHandle_t; 3],
    action_set: openvr_sys2::VRActionSetHandle_t,
    texture: Option<TextureState>,
    camera_config: Option<StereoCamera>,
    position_mode: PositionMode,
    display_mode: DisplayMode,
    overlay_transform: Matrix4<f64>,
    projector: Option<crate::projection::Projection<StandardDescriptorSetAlloc>>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    instance: Arc<Instance>,
    allocator: Arc<StandardMemoryAllocator>,
    cmdbuf_allocator: StandardCommandBufferAllocator,
    descriptor_set_allocator: StandardDescriptorSetAllocator,
    render_texture: Option<Arc<vulkano::image::Image>>,
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
        let mut input = unsafe { Pin::new_unchecked(&mut *openvr_sys2::VRInput()) };
        let action_manifest = xdg.find_data_file("actions.json").unwrap();
        let action_manifest = std::ffi::CString::new(action_manifest.to_str().unwrap()).unwrap();
        unsafe {
            input
                .as_mut()
                .SetActionManifestPath(action_manifest.as_ptr())
        }
        .into_result()?;
        let mut button = [const { MaybeUninit::uninit() }; 3];
        for i in 0..2 {
            let name = CString::new(format!("/actions/main/in/button{}", i + 1)).unwrap();
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
            let name = CString::new("/actions/main/in/debug").unwrap();
            input
                .as_mut()
                .GetActionHandle(
                    name.as_ptr(),
                    MaybeUninit::slice_as_mut_ptr(&mut button[2..]),
                )
                .into_result()?;
        }
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
        let allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        );
        let cmdbuf_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());
        Ok(Self {
            sys,
            handle: vroverlay,
            action_set,
            buttons: button,
            texture: None,
            display_mode: DisplayMode::default(),
            position_mode: PositionMode::default(),
            projector: None,
            overlay_transform: Matrix4::identity(),
            camera_config: None,
            instance,
            device,
            allocator,
            descriptor_set_allocator,
            queue,
            cmdbuf_allocator,
            render_texture: None,
        })
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
    fn set_overlay_transformation(&mut self, transform: Matrix4<f64>) -> Result<(), OpenVrError> {
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
    fn vk_descriptor_set_allocator(&self) -> &StandardDescriptorSetAllocator {
        &self.descriptor_set_allocator
    }
    fn vk_command_buffer_allocator(&self) -> &StandardCommandBufferAllocator {
        &self.cmdbuf_allocator
    }
}

fn transition_layout(
    input_layout: ImageLayout,
    image: &Arc<vulkano::image::Image>,
    queue: &Arc<Queue>,
    cmdbuf_allocator: &StandardCommandBufferAllocator,
) -> Result<vulkano::sync::fence::Fence, vulkano::Validated<vulkano::VulkanError>> {
    let cmdbuf = unsafe {
        let mut builder = UnsafeCommandBufferBuilder::new(
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
        builder.build()?
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
    fn get_render_texture(&mut self) -> Result<Arc<vulkano::image::Image>, Self::Error> {
        if self.display_mode.projection_mode().is_none() {
            assert!(self.render_texture.is_none());
            self.render_texture = Some(crate::create_submittable_image(self.allocator.clone())?);
        }
        Ok(self.render_texture.clone().unwrap())
    }
    fn submit_texture(
        &mut self,
        layout: ImageLayout,
        elapsed: Duration,
        fov: &[[f64; 2]; 2],
    ) -> Result<(), Self::Error> {
        let hmd_transform = self.sys.hmd_transform(-elapsed.as_secs_f32());
        if let PositionMode::Hmd { distance } = self.position_mode {
            let overlay_transform = hmd_transform
                * matrix![
                    1.0, 0.0, 0.0, 0.0;
                    0.0, 1.0, 0.0, 0.0;
                    0.0, 0.0, 1.0, -distance as f64;
                    0.0, 0.0, 0.0, 1.0;
                ];
            self.set_overlay_transformation(overlay_transform)?;
        }
        let output = if self.display_mode.projection_mode().is_some() {
            let new_texture = crate::create_submittable_image(self.allocator.clone())?;
            let eye_to_head = self.eye_to_head();
            let ipd = self.ipd()?;
            let projector = self.projector.as_mut().unwrap();
            projector.update_mvps(&self.overlay_transform, fov, &eye_to_head, &hmd_transform)?;
            projector.set_ipd(ipd);
            let future = projector.project(
                self.allocator.clone(),
                &self.cmdbuf_allocator,
                vulkano::sync::future::now(self.device.clone()),
                &self.queue,
                new_texture.clone(),
            )?;
            future.flush()?;
            future.then_signal_fence().wait(None)?;
            new_texture
        } else {
            let output = self.render_texture.take().unwrap();
            if layout != ImageLayout::TransferSrcOptimal {
                transition_layout(layout, &output, &self.queue, &self.cmdbuf_allocator)?
                    .wait(None)?;
            }
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
        unsafe {
            vroverlay
                .pin_mut()
                .SetOverlayTexture(self.handle, &vrtexture)
                .into_result()
                .map_err(Into::into)
        }
    }
    fn set_display_mode(&mut self, mode: DisplayMode) -> Result<(), Self::Error> {
        if self.display_mode == mode {
            return Ok(());
        }
        self.display_mode = mode;
        let is_stereo = mode.projection_mode();
        if let Some(projection_mode) = is_stereo {
            let camera_calib = self.load_camera_paramter();
            if self.projector.is_none() {
                self.render_texture =
                    Some(crate::create_submittable_image(self.allocator.clone())?);
                let mut projector = crate::projection::Projection::new(
                    self.device.clone(),
                    self.allocator.clone(),
                    &self.descriptor_set_allocator,
                    self.render_texture.as_ref().unwrap(),
                    1.0,
                    &camera_calib,
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
                is_stereo.is_some(),
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
    fn set_position_mode(&mut self, mode: PositionMode) -> Result<(), Self::Error> {
        self.position_mode = mode;
        if let PositionMode::Absolute { transform } = mode {
            let transform: Matrix4<f32> = transform.into();
            self.set_overlay_transformation(transform.cast())?;
        }
        Ok(())
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
        match openvr_sys2::EVREventType::try_from(openvr_event.eventType) {
            Ok(openvr_sys2::EVREventType::VREvent_IpdChanged) => {
                let ipd = unsafe { openvr_event.data.ipd.ipdMeters };
                Some(Event::IpdChanged(ipd))
            }
            Ok(openvr_sys2::EVREventType::VREvent_Quit) => Some(Event::RequestExit),
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
        //log::debug!("action_handle: {action_handle:x}");
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
        Ok(action_data.bState)
    }
}

pub(crate) struct OpenXr {
    entry: openxr::Entry,
    instance: openxr::Instance,
    system: openxr::SystemId,
    overlay_visible: bool,
    overlay_transform: Matrix4<f64>,
    position_mode: PositionMode,
    display_mode: DisplayMode,
    allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: StandardDescriptorSetAllocator,
    cmdbuf_allocator: StandardCommandBufferAllocator,

    device: Arc<Device>,
    queue: Arc<Queue>,
    vk_instance: Arc<Instance>,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum OpenXrError {
    #[error("cannot load openxr loader: {0}")]
    XrLoad(#[from] openxr::LoadError),
    #[error("cannot load vulkan library: {0}")]
    VkLoad(#[from] vulkano::LoadingError),
    #[error("vulkan error: {0}")]
    Vk(#[from] vulkano::VulkanError),
    #[error("xr: {0}")]
    Xr(#[from] openxr::sys::Result),
    #[error("vulkan error: {0}")]
    RawVk(#[from] ash::vk::Result),
    #[error("no graphics queue found")]
    NoGraphicsQueue,
}

impl OpenXr {
    fn create_vk_device(
        xr_instance: &openxr::Instance,
        xr_system: openxr::SystemId,
        instance: &Arc<Instance>,
    ) -> Result<(Arc<Device>, Arc<Queue>), OpenXrError> {
        let physical_device = unsafe {
            let physical_device =
                xr_instance.vulkan_graphics_device(xr_system, instance.handle().as_raw() as _)?;
            vulkano::device::physical::PhysicalDevice::from_handle(
                instance.clone(),
                ash::vk::PhysicalDevice::from_raw(physical_device as _),
            )
        }?;
        let queue_family = physical_device
            .queue_family_properties()
            .iter()
            .position(|qf| qf.queue_flags.contains(QueueFlags::GRAPHICS))
            .ok_or(OpenXrError::NoGraphicsQueue)?;
        let create_info = ash::vk::DeviceCreateInfo::builder()
            .queue_create_infos(&[ash::vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family as u32)
                .build()])
            .build();
        let vulkano_create_info = vulkano::device::DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index: queue_family as u32,
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
        let extensions = *get_vulkan_library().supported_extensions();
        let vulkano_create_info = vulkano::instance::InstanceCreateInfo {
            max_api_version: Some(vulkano::Version::V1_6),
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
    pub(crate) fn new() -> Result<Self, OpenXrError> {
        let entry = unsafe { openxr::Entry::load()? };
        let mut extension = openxr::ExtensionSet::default();
        extension.extx_overlay = true;
        extension.khr_vulkan_enable2 = true;
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
        let vk_instance = Self::create_vk_instance(&instance, system)?;
        let (device, queue) = Self::create_vk_device(&instance, system, &vk_instance)?;
        let allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let cmdbuf_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());
        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        );
        Ok(Self {
            entry,
            instance,
            system,
            overlay_visible: false,
            overlay_transform: Matrix4::identity(),
            position_mode: PositionMode::default(),
            display_mode: DisplayMode::default(),

            allocator,
            descriptor_set_allocator,
            cmdbuf_allocator,
            vk_instance,
            device,
            queue,
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
    fn vk_descriptor_set_allocator(&self) -> &StandardDescriptorSetAllocator {
        &self.descriptor_set_allocator
    }
    fn vk_command_buffer_allocator(&self) -> &StandardCommandBufferAllocator {
        &self.cmdbuf_allocator
    }
}

impl Vr for OpenXr {
    fn acknowledge_quit(&mut self) {
        // intentionally left blank
    }

    type Error = OpenXrError;

    fn load_camera_paramter(&mut self) -> Option<StereoCamera> {
        None
    }

    fn ipd(&self) -> Result<f32, Self::Error> {
        todo!()
    }

    fn eye_to_head(&self) -> [Matrix4<f64>; 2] {
        todo!()
    }

    fn submit_texture(
        &mut self,
        layout: ImageLayout,
        elapsed: Duration,
        fov: &[[f64; 2]; 2],
    ) -> Result<(), Self::Error> {
        todo!()
    }

    fn get_render_texture(&mut self) -> Result<Arc<vulkano::image::Image>, Self::Error> {
        todo!()
    }

    fn set_display_mode(&mut self, mode: DisplayMode) -> Result<(), Self::Error> {
        self.display_mode = mode;
        Ok(())
    }

    fn show_overlay(&mut self) -> Result<(), Self::Error> {
        self.overlay_visible = true;
        Ok(())
    }

    fn hide_overlay(&mut self) -> Result<(), Self::Error> {
        self.overlay_visible = false;
        Ok(())
    }

    fn set_position_mode(&mut self, mode: PositionMode) -> Result<(), Self::Error> {
        self.position_mode = mode;
        Ok(())
    }

    fn poll_next_event(&mut self) -> Option<Event> {
        todo!()
    }

    fn update_action_state(&mut self) -> Result<(), Self::Error> {
        todo!()
    }

    fn get_action_state(&self, action: Action) -> Result<bool, Self::Error> {
        todo!()
    }
}
