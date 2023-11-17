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

use v4l::video::Capture;
use vulkano::{
    command_buffer::allocator::{CommandBufferAllocator, StandardCommandBufferAllocator},
    image::{AllocateImageError, Image, ImageCreateInfo, ImageLayout, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    sync::GpuFuture,
    Validated,
};
use xdg::BaseDirectories;
/// Camera image will be (size * 2, size)
const CAMERA_SIZE: u32 = 960;
#[allow(unused_imports)]
use log::info;

use crate::{config::Backend, vrapi::VrExt};

static APP_KEY: &str = "index_camera_passthrough_rs\0";
static APP_NAME: &str = "Camera\0";
static APP_VERSION: u32 = 0;

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
    allocator: Arc<dyn MemoryAllocator>,
) -> Result<Arc<Image>, Validated<AllocateImageError>> {
    Image::new(
        allocator,
        ImageCreateInfo {
            extent: [CAMERA_SIZE * 2, CAMERA_SIZE, 1],
            format: vulkano::format::Format::R8G8B8A8_UNORM,
            usage: ImageUsage::TRANSFER_DST
                | ImageUsage::SAMPLED
                | ImageUsage::COLOR_ATTACHMENT
                | ImageUsage::TRANSFER_SRC,
            mip_levels: 1,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
}

fn load_splash(
    output: Arc<Image>,
    cmdbuf_allocator: &(impl CommandBufferAllocator + 'static),
    allocator: Arc<dyn MemoryAllocator>,
    queue: Arc<vulkano::device::Queue>,
) -> Result<()> {
    log::debug!("loading splash");
    let img = image::load_from_memory_with_format(SPLASH_IMAGE, image::ImageFormat::Png)?
        .into_rgba8()
        .into_raw();
    queue
        .device()
        .set_debug_utils_object_name(&output, Some("splash"))?;
    let future =
        pipeline::submit_cpu_image(&img, cmdbuf_allocator, allocator, &queue, output.clone())?;
    future.flush()?;
    future.then_signal_fence().wait(None)?;

    log::debug!("splash loaded");
    Ok(())
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
    log::info!("{:?}", cfg.backend);
    let mut vrsys = match cfg.backend {
        Backend::OpenVR => crate::vrapi::OpenVr::new(&xdg)?.boxed(),
        Backend::OpenXR => crate::vrapi::OpenXr::new()?.boxed(),
    };
    let instance = vrsys.vk_instance();
    let (device, queue) = vrsys.vk_device(&instance);

    let cmdbuf_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        vulkano::command_buffer::allocator::StandardCommandBufferAllocatorCreateInfo::default(),
    );

    load_splash(
        vrsys.get_render_texture()?,
        &cmdbuf_allocator,
        vrsys.vk_allocator(),
        queue.clone(),
    )?;

    // Create a VROverlay
    vrsys.set_display_mode(config::DisplayMode::Direct)?;
    // load camera config
    let camera_config = vrsys
        .load_camera_paramter()
        .or_else(steam::find_steam_config); // if the backend doesn't give us the parameters, we try
                                            // to search in the steam config for whatever that looks
                                            // like a camera parameter file

    vrsys.set_position_mode(cfg.overlay.position)?;

    // Show splash screen
    log::debug!("showing splash");
    vrsys.submit_texture(
        vulkano::image::ImageLayout::ColorAttachmentOptimal,
        std::time::Duration::ZERO,
        &[[0., 0.], [0., 0.]],
    )?;

    // Show overlay
    log::debug!("showing overlay");
    vrsys.show_overlay()?;

    vrsys.set_display_mode(cfg.display_mode)?;

    // TODO: don't hardcode this
    struct AppConfig {
        need_yuv_conversion: bool,
    }
    let config = AppConfig {
        need_yuv_conversion: true,
    };

    let mut pipeline = pipeline::Pipeline::new(
        device.clone(),
        vrsys.vk_allocator(),
        vrsys.vk_descriptor_set_allocator(),
        config.need_yuv_conversion,
        cfg.display_mode,
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
            let output = vrsys.get_render_texture()?;

            let future = pipeline.run(
                &queue,
                vrsys.vk_allocator(),
                vrsys.vk_command_buffer_allocator(),
                frame,
                output.clone(),
            )?;
            //println!("submission: {:?}", submission);
            future.flush()?; // can't use then_signal_fence_and_flush() because of a vulkano bug
            future.then_signal_fence().wait(None)?;

            // Submit the texture
            vrsys.submit_texture(
                ImageLayout::ColorAttachmentOptimal,
                elapsed,
                &pipeline.fov(),
            )?;
        }

        // Handle Ctrl-C
        if !running.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }

        // Handle OpenVR events
        #[allow(clippy::never_loop)]
        while let Some(event) = vrsys.poll_next_event() {
            match event {
                vrapi::Event::RequestExit => {
                    log::info!("RequestExit");
                    vrsys.acknowledge_quit();
                    break 'main_loop;
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
        state.handle(&*vrsys)?;
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
