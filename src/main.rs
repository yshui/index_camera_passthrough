#![feature(untagged_unions, try_trait_v2, destructuring_assignment)]
mod config;
mod distortion_correction;
mod events;
mod openvr;
mod projection;
mod steam;
mod yuv;

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use ash::vk::Handle;
use nalgebra::{matrix, Matrix4};
use openvr::*;
use v4l::video::Capture;
use vulkano::{
    buffer::CpuBufferPool,
    device::{self, physical::PhysicalDevice},
    image::ImageUsage,
    instance::{Instance, InstanceExtensions, Version},
    sync::GpuFuture,
    VulkanObject,
};
use yuv::GpuYuyvConverter;
/// Camera image will be (size * 2, size)
const CAMERA_SIZE: u32 = 960;
#[allow(unused_imports)]
use log::info;

static APP_KEY: &str = "index_camera_passthrough_rs\0";
static APP_NAME: &str = "Camera\0";

fn find_index_camera() -> Result<std::path::PathBuf> {
    let mut it = udev::Enumerator::new()?;
    it.match_subsystem("video4linux")?;
    it.match_property("ID_VENDOR_ID", "28de")?;
    it.match_property("ID_MODEL_ID", "2400")?;

    let dev = it
        .scan_devices()?
        .next()
        .with_context(|| anyhow!("Index camera not found"))?;
    let devnode = dev
        .devnode()
        .with_context(|| anyhow!("Index camera cannot be accessed"))?;
    Ok(devnode.to_owned())
}

static SPLASH_IMAGE: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/splash.png"));
fn load_splash(
    device: Arc<vulkano::device::Device>,
    queue: Arc<vulkano::device::Queue>,
    buffer: &CpuBufferPool<u8>,
) -> Result<Arc<vulkano::image::AttachmentImage>> {
    use vulkano::{
        command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage::*, PrimaryCommandBuffer},
        format::Format::*,
        image::AttachmentImage,
    };
    let img =
        image::load_from_memory_with_format(SPLASH_IMAGE, image::ImageFormat::Png)?.to_rgba8();
    let subbuffer = buffer.chunk(img.as_raw().iter().copied())?;
    let mut cmdbuf =
        AutoCommandBufferBuilder::primary(device.clone(), queue.family(), OneTimeSubmit)?;
    let output = AttachmentImage::with_usage(
        device,
        [CAMERA_SIZE * 2, CAMERA_SIZE],
        R8G8B8A8_UNORM,
        ImageUsage {
            transfer_source: false,
            transfer_destination: true,
            sampled: true,
            storage: false,
            color_attachment: true,
            depth_stencil_attachment: false,
            transient_attachment: false,
            input_attachment: false,
        },
    )?;
    cmdbuf.copy_buffer_to_image(subbuffer, output.clone())?;
    cmdbuf
        .build()?
        .execute(queue)?
        .then_signal_fence()
        .wait(None)?;
    Ok(output)
}

fn main() -> Result<()> {
    let cfg = config::load_config()?;
    let mut rd = None;
    if cfg.debug {
        rd = Some(renderdoc::RenderDoc::<renderdoc::V100>::new()?);
        std::env::set_var("RUST_LOG", "debug");
    }
    env_logger::init();
    let camera = v4l::Device::with_path(if cfg.camera_device.is_empty() {
        find_index_camera()?
    } else {
        std::path::Path::new(&cfg.camera_device).to_owned()
    })?;
    if !camera
        .query_caps()?
        .capabilities
        .contains(v4l::capability::Flags::VIDEO_CAPTURE)
    {
        return Err(anyhow!("Cannot capture from index camera"));
    }
    let format = camera.set_format(&v4l::Format::new(
        CAMERA_SIZE * 2,
        CAMERA_SIZE,
        v4l::FourCC::new(b"YUYV"),
    ))?;
    log::info!("{}", format);
    camera.set_params(&v4l::video::capture::Parameters::with_fps(54))?;
    // We want to make the latency as low as possible, so only set a single buffer.
    let mut video_stream =
        v4l::prelude::MmapStream::with_buffers(&camera, v4l::buffer::Type::VideoCapture, 1)?;

    let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        r.store(false, std::sync::atomic::Ordering::Relaxed);
    })
    .expect("Error setting Ctrl-C handler");

    // Create vulkan instance, and setup openvr.
    // Then create a vulkan device based on openvr's requirements
    let instance = Instance::new(
        None,
        Version::V1_1,
        &InstanceExtensions::supported_by_core()?,
        None,
    )?;
    let vrsys = VRSystem::init()?;
    let mut target_device = 0u64;
    unsafe {
        vrsys.pin_mut().GetOutputDevice(
            &mut target_device,
            openvr_sys::ETextureType::TextureType_Vulkan,
            std::mem::transmute(instance.internal_object().as_raw()),
        )
    };

    let target_device = ash::vk::PhysicalDevice::from_raw(target_device);
    let device = PhysicalDevice::enumerate(&instance)
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
        let extensions = device::DeviceExtensions::from(
            vrsys.compositor().required_extensions(device, &mut buf),
        )
        .union(device.required_extensions());
        device::Device::new(
            device,
            &device::Features::none(),
            &extensions,
            [(queue_family, 1.0)],
        )?
    };
    let queue = queues.next().unwrap();
    let buffer = CpuBufferPool::upload(device.clone());

    let splash = load_splash(device.clone(), queue.clone(), &buffer)?;

    // Load steam calibration data
    let hmd_id = vrsys.find_hmd().with_context(|| anyhow!("HMD not found"))?;
    let mut serial_number = [0u8; 32];
    let mut error = openvr_sys::ETrackedPropertyError::TrackedProp_Success;
    let serial_number_len = unsafe {
        vrsys.pin_mut().GetStringTrackedDeviceProperty(
            hmd_id,
            openvr_sys::ETrackedDeviceProperty::Prop_SerialNumber_String,
            serial_number.as_mut_ptr() as *mut _,
            32,
            &mut error,
        )
    };
    if error != openvr_sys::ETrackedPropertyError::TrackedProp_Success {
        return Err(anyhow!("Cannot get HMD's serial number"));
    }
    let lhcfg = steam::load_steam_config(std::str::from_utf8(
        &serial_number[..serial_number_len as usize - 1],
    )?)
    .unwrap_or_else(|e| {
        log::warn!(
            "Cannot find camera calibration data, using default ones {}",
            e.to_string()
        );
        steam::default_lighthouse_config()
    });
    log::info!("{}", serde_json::to_string(&lhcfg)?);

    // Create a VROverlay
    let vroverlay = vrsys.overlay();
    let mut overlay = vroverlay.create_overlay(APP_KEY, APP_NAME)?;
    if let config::DisplayMode::Stereo { .. } = cfg.display_mode {
        vroverlay
            .pin_mut()
            .SetOverlayFlag(
                overlay.as_raw(),
                openvr_sys::VROverlayFlags::VROverlayFlags_SideBySide_Parallel,
                true,
            )
            .into_result()?;
    }

    // Allocate intermediate textures
    let textures: Result<Vec<_>> = (0..2)
        .map(|_| {
            vulkano::image::AttachmentImage::with_usage(
                device.clone(),
                [CAMERA_SIZE * 2, CAMERA_SIZE],
                vulkano::format::Format::R8G8B8A8_UNORM,
                ImageUsage {
                    transfer_source: true,
                    transfer_destination: true,
                    sampled: true,
                    color_attachment: true,
                    ..ImageUsage::none()
                },
            )
            .map_err(|e| e.into())
        })
        .collect();
    let textures = textures?;

    let projector = match cfg.display_mode {
        config::DisplayMode::Stereo { projection_mode } => Some(projection::Projection::new(
            device.clone(),
            textures[1].clone(),
            projection_mode,
        )?),
        config::DisplayMode::Flat {
            eye: config::Eye::Left,
        } => {
            let bound = openvr_sys::VRTextureBounds_t {
                uMin: 0.0,
                uMax: 0.5,
                vMin: 0.0,
                vMax: 1.0,
            };
            unsafe {
                vroverlay
                    .pin_mut()
                    .SetOverlayTextureBounds(overlay.as_raw(), &bound)
            };
            None
        }
        config::DisplayMode::Flat {
            eye: config::Eye::Right,
        } => {
            let bound = openvr_sys::VRTextureBounds_t {
                uMin: 0.5,
                uMax: 1.0,
                vMin: 0.0,
                vMax: 1.0,
            };
            unsafe {
                vroverlay
                    .pin_mut()
                    .SetOverlayTextureBounds(overlay.as_raw(), &bound)
            };
            None
        }
    };

    let hmd_transform = vrsys.hmd_transform(0.0);
    let mut overlay_transform: Matrix4<f64> = match cfg.overlay.position {
        config::PositionMode::Absolute { transform } => {
            let mut transformation = openvr_sys::HmdMatrix34_t { m: [[0.0; 4]; 3] };
            transformation.m[..].copy_from_slice(&transform[..3]);
            unsafe {
                vroverlay.pin_mut().SetOverlayTransformAbsolute(
                    overlay.as_raw(),
                    openvr_sys::ETrackingUniverseOrigin::TrackingUniverseStanding,
                    &transformation,
                )
            };
            let transform: Matrix4<f32> = transform.into();
            transform.cast()
        }
        config::PositionMode::Hmd { distance } => {
            hmd_transform
                * matrix![
                    1.0, 0.0, 0.0, 0.0;
                    0.0, 1.0, 0.0, 0.0;
                    0.0, 0.0, 1.0, -distance as f64;
                    0.0, 0.0, 0.0, 1.0;
                ]
        }
    };

    // Set initial position for overlay
    unsafe {
        vroverlay.pin_mut().SetOverlayTransformAbsolute(
            overlay.as_raw(),
            openvr_sys::ETrackingUniverseOrigin::TrackingUniverseStanding,
            &(&overlay_transform).into(),
        )
    };

    // Show splash screen
    overlay.set_texture(
        CAMERA_SIZE * 2,
        CAMERA_SIZE,
        splash,
        device.clone(),
        queue.clone(),
        instance.clone(),
    )?;

    // Show overlay
    vroverlay
        .pin_mut()
        .ShowOverlay(overlay.as_raw())
        .into_result()?;

    // Create post-processing stages
    //
    // Camera data -> upload -> internal texture
    // internal texture -> YUYV conversion -> textures[0]
    // textures[0] -> Lens correction -> textures[1]
    // textures[1] -> projection -> Final output
    let converter = GpuYuyvConverter::new(device.clone(), CAMERA_SIZE * 2, CAMERA_SIZE)?;
    let (correction, fov_left, fov_right) =
        distortion_correction::StereoCorrection::new(device.clone(), textures[0].clone(), &lhcfg)?;
    log::info!("Adjusted FOV: {:?} {:?}", fov_left, fov_right);

    // Fetch the first camera frame
    let (mut frame, mut metadata) = v4l::io::traits::CaptureStream::next(&mut video_stream)?;
    let first_frame_timestamp: std::time::Duration = metadata.timestamp.into();
    let first_frame_instant = std::time::Instant::now();
    let mut capture = false;
    let mut error = std::mem::MaybeUninit::<_>::uninit();
    let mut ipd = unsafe {
        vrsys.pin_mut().GetFloatTrackedDeviceProperty(
            0,
            openvr_sys::ETrackedDeviceProperty::Prop_UserIpdMeters_Float,
            error.as_mut_ptr(),
        )
    };
    if unsafe { error.assume_init() } != openvr_sys::ETrackedPropertyError::TrackedProp_Success {
        return Err(anyhow!("Cannot get device IPD {:?}", error));
    }
    log::info!("IPD: {}", ipd);

    let mut state = events::State::new(cfg.toggle_button, cfg.open_delay);
    'main_loop: loop {
        if state.visible() {
            // We try to get the pose at the time when the camera frame is captured. GetDeviceToAbsoluteTrackingPose
            // doesn't specifically say if a negative time offset will work...
            // also, do this as early as possible.
            let frame_time =
                Into::<std::time::Duration>::into(metadata.timestamp) - first_frame_timestamp;
            log::trace!(
                "{:?} {:?}",
                std::time::Instant::now() - first_frame_instant,
                frame_time
            );
            let mut elapsed = std::time::Instant::now() - first_frame_instant;
            if elapsed > frame_time {
                elapsed -= frame_time;
            } else {
                elapsed = std::time::Duration::ZERO;
            }
            let hmd_transform = vrsys.hmd_transform(-elapsed.as_secs_f32());
            // Allocate final image
            let output = vulkano::image::AttachmentImage::with_usage(
                device.clone(),
                [CAMERA_SIZE * 2, CAMERA_SIZE],
                vulkano::format::Format::R8G8B8A8_UNORM,
                vulkano::image::ImageUsage {
                    transfer_source: true,
                    transfer_destination: true,
                    sampled: true,
                    storage: false,
                    color_attachment: true,
                    depth_stencil_attachment: false,
                    transient_attachment: false,
                    input_attachment: false,
                },
            )?;

            if capture {
                if let Some(rd) = rd.as_mut() {
                    log::info!("Start Capture");
                    rd.start_frame_capture(std::ptr::null(), std::ptr::null());
                }
            }
            // First convert YUYV to RGB
            let future = converter.yuyv_buffer_to_vulkan_image(
                frame,
                vulkano::sync::now(device.clone()),
                queue.clone(),
                &buffer,
                textures[0].clone(),
            )?;

            match cfg.overlay.position {
                config::PositionMode::Absolute { .. } => (),
                config::PositionMode::Hmd { distance } => {
                    // We only move the overlay when we get new camera frame
                    // this way the overlay should be reprojected correctly in-between
                    overlay_transform = hmd_transform
                        * matrix![
                            1.0, 0.0, 0.0, 0.0;
                            0.0, 1.0, 0.0, 0.0;
                            0.0, 0.0, 1.0, -distance as f64;
                            0.0, 0.0, 0.0, 1.0;
                        ];
                    unsafe {
                        vroverlay.pin_mut().SetOverlayTransformAbsolute(
                            overlay.as_raw(),
                            openvr_sys::ETrackingUniverseOrigin::TrackingUniverseStanding,
                            &(&overlay_transform).into(),
                        )
                    };
                }
            };

            // TODO combine correction and projection
            if let Some(projector) = projector.as_ref() {
                // Then do lens correction
                let future = correction.correct(future, queue.clone(), textures[1].clone())?;
                // Finally apply projection
                // Calculate each eye's Model View Project matrix at the moment the current frame is taken
                let (l, r) = projector.calculate_mvp(
                    &overlay_transform,
                    &lhcfg,
                    (&fov_left, &fov_right),
                    &vrsys,
                    &hmd_transform,
                );
                let future = projector.project(
                    future,
                    queue.clone(),
                    output.clone(),
                    1.0,
                    ipd,
                    &lhcfg,
                    (&l, &r),
                )?;

                // Wait for work to complete
                future.then_signal_fence().wait(None)?;
            } else {
                // Lens correction
                let future = correction.correct(future, queue.clone(), output.clone())?;
                future.then_signal_fence().wait(None)?;
            }

            if capture {
                if let Some(rd) = rd.as_mut() {
                    log::info!("End Capture");
                    rd.end_frame_capture(std::ptr::null(), std::ptr::null());
                }
                capture = false;
            }

            // Submit the texture
            overlay.set_texture(
                CAMERA_SIZE * 2,
                CAMERA_SIZE,
                output,
                device.clone(),
                queue.clone(),
                instance.clone(),
            )?;
        }

        let mut event = std::mem::MaybeUninit::<openvr_sys::VREvent_t>::uninit();
        // Handle OpenVR events
        while unsafe {
            vrsys.pin_mut().PollNextEvent(
                event.as_mut_ptr() as *mut _,
                std::mem::size_of::<openvr_sys::VREvent_t>() as u32,
            )
        } {
            let event = unsafe { event.assume_init_ref() };
            log::debug!("{:?}", unsafe {
                std::mem::transmute::<_, openvr_sys::EVREventType>(event.eventType)
            });
            if event.eventType == openvr_sys::EVREventType::VREvent_ButtonPress as u32 {
                log::debug!("{:?}", unsafe { event.data.controller.button });
                if unsafe { event.data.controller.button == 33 } {
                    capture = true;
                }
            } else if event.eventType == openvr_sys::EVREventType::VREvent_Quit as u32 {
                vrsys.pin_mut().AcknowledgeQuit_Exiting();
                break 'main_loop;
            } else if event.eventType == openvr_sys::EVREventType::VREvent_IpdChanged as u32 {
                ipd = unsafe { event.data.ipd.ipdMeters };
                log::info!("ipd: {}", ipd);
            }
            state.handle(event);
        }

        // Handle Ctrl-C
        if !running.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }

        match state.turn() {
            events::Action::ShowOverlay => vroverlay
                .pin_mut()
                .ShowOverlay(overlay.as_raw())
                .into_result()?,
            events::Action::HideOverlay => vroverlay
                .pin_mut()
                .HideOverlay(overlay.as_raw())
                .into_result()?,
            _ => (),
        }

        std::thread::sleep(state.interval());

        if state.visible() {
            // Fetch next frame only when overlay is visible
            (frame, metadata) = v4l::io::traits::CaptureStream::next(&mut video_stream)?;
        }
    }

    Ok(())
}
