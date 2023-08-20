#![feature(
    try_trait_v2,
    array_try_map,
    array_methods,
    inline_const,
    maybe_uninit_slice,
    maybe_uninit_array_assume_init
)]
#![deny(rust_2018_idioms)]
mod config;
mod distortion_correction;
mod events;
mod openvr;
mod pipeline;
mod projection;
mod steam;
mod vrapi;
mod yuv;

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};

use nalgebra::{matrix, Matrix4};

use v4l::video::Capture;
use vulkano::{
    command_buffer::allocator::{CommandBufferAllocator, StandardCommandBufferAllocator},
    device::{self, DeviceCreateInfo, QueueCreateInfo, QueueFlags},
    image::{
        immutable::ImmutableImageInitialization, ImageAccess, ImageCreateFlags, ImageDimensions,
        ImageLayout, ImageUsage, ImmutableImage, MipmapsCount,
    },
    instance::{Instance, InstanceCreateInfo, Version},
    memory::allocator::{MemoryAllocator, StandardMemoryAllocator},
    sync::GpuFuture,
    VulkanLibrary,
};
use xdg::BaseDirectories;
/// Camera image will be (size * 2, size)
const CAMERA_SIZE: u32 = 960;
#[allow(unused_imports)]
use log::info;

use crate::vrapi::Vr;

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

fn first_run(xdg: &BaseDirectories) -> Result<()> {
    const STEAMVR_ACTION_MANIFEST: &str =
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/actions.json"));
    const DEFAULT_CONFIG: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/index_camera_passthrough.toml"
    ));
    let config = xdg.place_config_file("index_camera_passthrough.toml")?;
    if !config.exists() {
        std::fs::write(&config, DEFAULT_CONFIG)?;
    }
    let action_manifest = xdg.place_data_file("actions.json")?;
    if !action_manifest.exists() {
        std::fs::write(&action_manifest, STEAMVR_ACTION_MANIFEST)?;
    }
    Ok(())
}

fn create_submittable_image(
    allocator: &impl MemoryAllocator,
    queue: &vulkano::device::Queue,
) -> Result<(Arc<ImmutableImage>, Arc<ImmutableImageInitialization>)> {
    Ok(ImmutableImage::uninitialized(
        allocator,
        ImageDimensions::Dim2d {
            width: CAMERA_SIZE * 2,
            height: CAMERA_SIZE,
            array_layers: 1,
        },
        vulkano::format::Format::R8G8B8A8_UNORM,
        MipmapsCount::One,
        ImageUsage::TRANSFER_DST
            | ImageUsage::SAMPLED
            | ImageUsage::COLOR_ATTACHMENT
            | ImageUsage::TRANSFER_SRC,
        ImageCreateFlags::empty(),
        ImageLayout::TransferSrcOptimal,
        [queue.queue_family_index()],
    )?)
}

fn load_splash(
    cmdbuf_allocator: &impl CommandBufferAllocator,
    allocator: &impl MemoryAllocator,
    queue: Arc<vulkano::device::Queue>,
) -> Result<Arc<ImmutableImage>> {
    log::debug!("loading splash");
    let img = image::load_from_memory_with_format(SPLASH_IMAGE, image::ImageFormat::Png)?
        .into_rgba8()
        .into_raw();
    let (output, output_init) = create_submittable_image(allocator, &queue)?;
    queue
        .device()
        .set_debug_utils_object_name(&output.inner().image, Some("splash"))?;
    pipeline::submit_cpu_image(&img, cmdbuf_allocator, allocator, &queue, output_init)?
        .then_signal_fence()
        .wait(None)?;
    log::debug!("splash loaded");
    Ok(output)
}

fn main() -> Result<()> {
    let xdg = xdg::BaseDirectories::with_prefix("index_camera_passthrough")?;
    first_run(&xdg)?;

    let cfg = config::load_config(&xdg)?;
    if cfg.debug {
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

    let library = VulkanLibrary::new()?;
    // Create vulkan instance, and setup openvr.
    // Then create a vulkan device based on openvr's requirements
    let extensions = *library.supported_extensions();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            max_api_version: Some(Version::V1_6),
            enabled_extensions: extensions,
            // enabled_layers: vec!["VK_LAYER_KHRONOS_validation".to_owned()],
            ..Default::default()
        },
    )?;
    let mut vrsys = crate::vrapi::OpenVr::new(&xdg)?;
    let device = vrsys.target_device(&instance)?;
    let queue_family = device
        .queue_family_properties()
        .iter()
        .position(|qf| qf.queue_flags.contains(QueueFlags::GRAPHICS))
        .with_context(|| anyhow!("Cannot create a suitable queue"))?;
    let (device, mut queues) = {
        let extensions: device::DeviceExtensions = vrsys.required_extensions(&device);
        device::Device::new(
            device,
            DeviceCreateInfo {
                enabled_features: device::Features::empty(),
                enabled_extensions: extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: queue_family as u32,
                    queues: vec![1.0],
                    ..Default::default()
                }],
                ..Default::default()
            },
        )?
    };
    let queue = queues.next().unwrap();

    let cmdbuf_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        vulkano::command_buffer::allocator::StandardCommandBufferAllocatorCreateInfo::default(),
    );
    let allocator = StandardMemoryAllocator::new_default(device.clone());

    let splash = load_splash(&cmdbuf_allocator, &allocator, queue.clone())?;

    // Create a VROverlay
    match &cfg.display_mode {
        config::DisplayMode::Stereo { .. } => {
            vrsys.set_overlay_stereo(true)?;
        }
        &config::DisplayMode::Flat { eye } => {
            let bound = if eye == crate::config::Eye::Left {
                crate::vrapi::Bounds {
                    umin: 0.0,
                    umax: 0.5,
                    vmin: 0.0,
                    vmax: 1.0,
                }
            } else {
                crate::vrapi::Bounds {
                    umin: 0.5,
                    umax: 1.0,
                    vmin: 0.0,
                    vmax: 1.0,
                }
            };
            vrsys.set_overlay_texture_bounds(bound)?;
        }
    }
    // load camera config
    let camera_config = vrsys
        .load_camera_paramter()
        .or_else(|| steam::find_steam_config()); // if the backend doesn't give us the parameters, we try
                                                 // to search in the steam config for whatever that looks
                                                 // like a camera parameter file

    let hmd_transform = vrsys.hmd_transform(0.0);
    let mut overlay_transform: Matrix4<f64> = match cfg.overlay.position {
        config::PositionMode::Absolute { transform } => {
            let mut transformation = openvr_sys2::HmdMatrix34_t { m: [[0.0; 4]; 3] };
            transformation.m[..].copy_from_slice(&transform[..3]);
            vrsys.set_overlay_transformation(transformation.into())?;
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
    vrsys.set_overlay_transformation(overlay_transform)?;

    // Show splash screen
    log::debug!("showing splash");
    vrsys.set_overlay_texture(
        CAMERA_SIZE * 2,
        CAMERA_SIZE,
        splash,
        device.clone(),
        queue.clone(),
        instance.clone(),
    )?;

    // Show overlay
    log::debug!("showing overlay");
    vrsys.show_overlay()?;

    // TODO: don't hardcode this
    struct AppConfig {
        need_yuv_conversion: bool,
    }
    let config = AppConfig {
        need_yuv_conversion: true,
    };

    let mut pipeline = pipeline::Pipeline::new(
        device.clone(),
        config.need_yuv_conversion,
        cfg.display_mode,
        vrsys.ipd()?,
        camera_config,
    )?;

    log::debug!("pipeline: {pipeline:?}");

    // Fetch the first camera frame
    let (mut frame, mut metadata) = v4l::io::traits::CaptureStream::next(&mut video_stream)?;
    let first_camera_frame_timestamp: std::time::Duration = metadata.timestamp.into();
    let render_start_instant = std::time::Instant::now();

    let mut state = events::State::new(cfg.open_delay);
    let mut debug_pressed = false;
    'main_loop: loop {
        if state.visible() {
            // We try to get the pose at the time when the camera frame is captured. GetDeviceToAbsoluteTrackingPose
            // doesn't specifically say if a negative time offset will work...
            // also, do this as early as possible.
            let frame_time = Into::<std::time::Duration>::into(metadata.timestamp)
                - first_camera_frame_timestamp;
            log::trace!(
                "{:?} {:?}",
                std::time::Instant::now() - render_start_instant,
                frame_time
            );
            let mut elapsed = std::time::Instant::now() - render_start_instant;
            if elapsed > frame_time {
                elapsed -= frame_time;
            } else {
                elapsed = std::time::Duration::ZERO;
            }
            // Allocate final image
            let (output, output_init) = create_submittable_image(&allocator, &queue)?;

            let hmd_transform = vrsys.hmd_transform(-elapsed.as_secs_f32());
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
                    vrsys.set_overlay_transformation(overlay_transform)?;
                }
            };

            let future = pipeline.run(
                &vrsys.eye_to_head(),
                &hmd_transform,
                &overlay_transform,
                &queue,
                frame,
                output_init,
            )?;
            //println!("submission: {:?}", submission);
            future.flush()?; // can't use then_signal_fence_and_flush() because of a vulkano bug
            future.then_signal_fence().wait(None)?;

            // Submit the texture
            vrsys.set_overlay_texture(
                CAMERA_SIZE * 2,
                CAMERA_SIZE,
                output,
                device.clone(),
                queue.clone(),
                instance.clone(),
            )?;
        }

        // Handle Ctrl-C
        if !running.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }

        // Handle OpenVR events
        while let Some(event) = vrsys.poll_next_event() {
            //log::debug!("{:?}", unsafe {
            //    std::mem::transmute::<_, openvr_sys2::EVREventType>(event.eventType)
            //});
            match event {
                vrapi::Event::RequestExit => {
                    vrsys.acknowledge_quit();
                    break 'main_loop;
                }
                vrapi::Event::IpdChanged(new_ipd) => {
                    pipeline.set_ipd(new_ipd);
                }
            }
        }

        // Handle user inputs
        vrsys.update_action_state()?;
        if vrsys.get_action_state(vrapi::Action::Debug)? {
            if !debug_pressed {
                log::debug!("Capture next frame");
                pipeline.capture_next_frame();
                debug_pressed = true;
            }
        } else {
            debug_pressed = false;
        }
        state.handle(&vrsys)?;
        match state.turn() {
            events::Action::ShowOverlay => vrsys.show_overlay()?,
            events::Action::HideOverlay => vrsys.hide_overlay()?,
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
