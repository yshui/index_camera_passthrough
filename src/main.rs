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

use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};

use v4l::video::Capture;
use vulkano::{
    image::{AllocateImageError, Image, ImageCreateInfo, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    sync::GpuFuture,
    Validated,
};
use xdg::BaseDirectories;
/// Camera image will be (size * 2, size)
const CAMERA_SIZE: u32 = 960;
#[allow(unused_imports)]
use log::info;

use crate::{config::Backend, pipeline::submit_cpu_image, vrapi::VrExt};

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
    const ACTIONS_JSON : &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/actions.json"));
    const DATA_FILES: &[(&str, &str)] = &[
        (
            "vive_controller_bindings.json",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/vive_controller_bindings.json"
            )),
        ),
        (
            "knuckles_bindings.json",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/knuckles_bindings.json"
            )),
        ),
        (
            "generic_bindings.json",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/generic_bindings.json"
            )),
        ),
    ];
    const DEFAULT_CONFIG: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/index_camera_passthrough.toml"
    ));
    let config = xdg.place_config_file("index_camera_passthrough.toml")?;
    if !config.exists() {
        std::fs::write(&config, DEFAULT_CONFIG)?;
    }
    let actions: serde_json::Value = serde_json::from_str(ACTIONS_JSON)?;
    std::fs::write(xdg.place_data_file("actions.json")?, actions.to_string())?;
    for (name, data) in DATA_FILES {
        let path = xdg.place_data_file(name)?;
        if !path.exists() {
            let obj :serde_json::Value = serde_json::from_str(data)?;
            std::fs::write(&path, obj.to_string())?;
        }
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

struct FrameInfo {
    frame: Vec<u8>,
    frame_time: Option<std::time::Instant>,
    bypass_pipeline: bool,
}

fn load_splash() -> Result<Vec<u8>> {
    log::debug!("loading splash");
    let img = image::load_from_memory_with_format(SPLASH_IMAGE, image::ImageFormat::Png)?
        .into_rgba8()
        .into_raw();

    log::debug!("splash loaded");
    Ok(img)
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum State {
    Running,
    Capturing,
    Stopping,
}

struct AppState {
    state: Mutex<State>,
    notify: std::sync::Condvar,
}

impl AppState {
    fn new() -> Self {
        Self {
            state: Mutex::new(State::Running),
            notify: std::sync::Condvar::new(),
        }
    }
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap()
    }
    fn wait_while<'a, F>(
        &'a self,
        guard: std::sync::MutexGuard<'a, State>,
        f: F,
    ) -> std::sync::MutexGuard<'a, State>
    where
        F: FnMut(&mut State) -> bool,
    {
        self.notify.wait_while(guard, f).unwrap()
    }
    fn stop(&self) {
        *self.state.lock().unwrap() = State::Stopping;
        self.notify.notify_all();
    }
    fn start_capture(&self) {
        let mut state = self.state.lock().unwrap();
        if *state != State::Stopping {
            *state = State::Capturing;
        }
        self.notify.notify_all();
    }
    fn stop_capture(&self) {
        let mut state = self.state.lock().unwrap();
        if *state != State::Stopping {
            *state = State::Running;
        }
    }
}

struct CameraThread {
    notify_new_frame: Arc<std::sync::Condvar>,
    state: Arc<AppState>,
    frame: Arc<Mutex<Option<FrameInfo>>>,
    camera: v4l::Device,
}

impl CameraThread {
    fn run(self) -> Result<()> {
        let Self {
            notify_new_frame,
            state,
            frame,
            camera,
        } = self;

        let mut first_frame_time = None;
        // We want to make the latency as low as possible, so only set a single buffer.
        let mut video_stream =
            v4l::prelude::MmapStream::with_buffers(&camera, v4l::buffer::Type::VideoCapture, 1)
                .context("cannot open camera mmap stream")?;
        loop {
            {
                let guard = state.lock();
                log::trace!("state: {:?}", *guard);
                if *state.wait_while(guard, |state| *state == State::Running) == State::Stopping {
                    break;
                }
            }
            log::trace!("getting camera frame");
            let (frame_data, metadata) = v4l::io::traits::CaptureStream::next(&mut video_stream)?;
            let frame_time = if let Some((camera_reference, reference)) = first_frame_time {
                let camera_elapsed =
                    std::time::Duration::from(metadata.timestamp) - camera_reference;
                reference + camera_elapsed
            } else {
                let now = std::time::Instant::now();
                first_frame_time = Some((metadata.timestamp.into(), now));
                now
            };
            log::trace!("got camera frame {:?}", frame_time);
            let mut frame = frame.lock().unwrap();
            if let Some(frame) = &mut *frame {
                frame.frame.resize(frame_data.len(), 0);
                frame.frame.copy_from_slice(frame_data);
                frame.frame_time = Some(frame_time);
                frame.bypass_pipeline = false;
            } else {
                *frame = Some(FrameInfo {
                    frame: frame_data.to_vec(),
                    frame_time: Some(frame_time),
                    bypass_pipeline: false,
                });
            }
            // log::debug!("got camera frame {}", frame_data.len());
            notify_new_frame.notify_all();
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let xdg = xdg::BaseDirectories::with_prefix("index_camera_passthrough")?;
    first_run(&xdg)?;

    let cfg = config::load_config(&xdg)?;
    let env =
        env_logger::Env::default().default_filter_or(if cfg.debug { "debug" } else { "info" });
    env_logger::init_from_env(env);
    let camera = v4l::Device::with_path(if cfg.camera_device.is_empty() {
        find_index_camera()?
    } else {
        std::path::Path::new(&cfg.camera_device).to_owned()
    })
    .context("cannot open camera device")?;
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
    let splash = load_splash()?;
    let frame = Arc::new(Mutex::new(Some(FrameInfo {
        frame: splash.clone(),
        frame_time: None,
        bypass_pipeline: true,
    })));

    let app_state = Arc::new(AppState::new());
    let state2 = app_state.clone();

    ctrlc::set_handler(move || {
        state2.stop();
    })
    .expect("Error setting Ctrl-C handler");

    let notify_new_frame = Arc::new(std::sync::Condvar::new());
    let camera_thread = CameraThread {
        notify_new_frame: notify_new_frame.clone(),
        frame: frame.clone(),
        state: app_state.clone(),
        camera,
    };
    let camera_thread = std::thread::spawn(move || camera_thread.run());

    log::info!("{:?}", cfg.backend);
    let mut vrsys = match cfg.backend {
        Backend::OpenVR => crate::vrapi::OpenVr::new(&xdg)?.boxed(),
        Backend::OpenXR => crate::vrapi::OpenXr::new(cfg.z_order)?.boxed(),
    };
    let instance = vrsys.vk_instance();
    let (device, queue) = vrsys.vk_device(&instance);

    // Create a VROverlay
    vrsys.set_display_mode(config::DisplayMode::Direct)?;
    // load camera config
    let camera_config = if let Some(cfg) = vrsys.load_camera_paramter() {
        Some(cfg)
    } else if let Some(cfg) = steam::find_steam_config() {
        // if the backend doesn't give us the parameters, we try
        // to search in the steam config for whatever that looks
        // like a camera parameter file
        vrsys.set_fallback_camera_config(cfg);
        Some(cfg)
    } else {
        log::warn!("No camera parameters found");
        None
    };

    vrsys.set_position_mode(cfg.overlay.position)?;

    // Show overlay
    log::debug!("showing overlay");
    vrsys.show_overlay()?;
    app_state.start_capture();

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
        camera_config,
    )?;

    log::debug!("pipeline: {pipeline:?}");

    let mut ui_state = events::State::new(cfg.open_delay);
    let mut debug_pressed = false;
    let mut maybe_current_frame: Option<FrameInfo> = None;
    let mut frame_changed = false;
    let is_synchronized = vrsys.is_synchronized();
    loop {
        // If synchronized, get the new frame if available, and refresh with existing frame if not;
        // otherwise wait for the next frame.
        if ui_state.is_visible() {
            let mut other_frame = frame.lock().unwrap();
            while !frame_changed {
                let new_frame_elapsed = other_frame.as_ref().map(|new_frame| new_frame.frame_time);
                let current_frame_elapsed = maybe_current_frame
                    .as_ref()
                    .map(|current_frame| current_frame.frame_time);
                frame_changed = new_frame_elapsed > current_frame_elapsed;
                if frame_changed {
                    std::mem::swap(&mut maybe_current_frame, &mut *other_frame);
                    log::trace!("frame changed");
                } else if is_synchronized {
                    // Don't wait if we are obliged to synchronize with VR runtime
                    break;
                } else {
                    other_frame = notify_new_frame.wait(other_frame).unwrap();
                }
            }
            drop(other_frame);
            let Some(current_frame) = maybe_current_frame.as_ref() else {
                // Should never happen, because we should at least have the splash screen
                log::error!("No frame");
                continue;
            };
            // log::debug!("frame bypass pipeline: {}", current_frame.bypass_pipeline);
            if current_frame.bypass_pipeline {
                vrsys.set_display_mode(config::DisplayMode::Direct)?;
            } else {
                vrsys.set_display_mode(cfg.display_mode)?;
            }

            if frame_changed {
                // We try to get the pose at the time when the camera frame is captured. GetDeviceToAbsoluteTrackingPose
                // doesn't specifically say if a negative time offset will work...
                // also, do this as early as possible.
                let elapsed = current_frame
                    .frame_time
                    .map(|frame_time| std::time::Instant::now() - frame_time)
                    .unwrap_or_default();
                log::trace!("elapsed: {elapsed:?}");
                // Allocate final image
                if let Some(output) = vrsys.get_render_texture()? {
                    if current_frame.bypass_pipeline {
                        let future = submit_cpu_image(
                            &current_frame.frame,
                            vrsys.vk_command_buffer_allocator(),
                            vrsys.vk_allocator(),
                            &queue,
                            output,
                        )?;
                        future.flush()?;
                        future.then_signal_fence().wait(None)?;
                    } else {
                        let future = pipeline.run(
                            &queue,
                            vrsys.vk_allocator(),
                            vrsys.vk_command_buffer_allocator(),
                            &current_frame.frame,
                            output.clone(),
                        )?;
                        //println!("submission: {:?}", submission);
                        future.flush()?; // can't use then_signal_fence_and_flush() because of a vulkano bug
                        future.then_signal_fence().wait(None)?;
                    }

                    // Submit the texture
                    vrsys.submit_texture(elapsed, &pipeline.fov())?;
                    frame_changed = false;
                }
            } else {
                vrsys.refresh()?;
            }
        } else if !is_synchronized {
            // Not synchronized, and we didn't wait for new camera frame because we are not visible.
            // We have to throttle our main loop.
            std::thread::sleep(std::time::Duration::from_millis(100));
        } else {
            // Synchronize with VR Runtime.
            vrsys.refresh()?;
        }

        // Handle OpenVR events
        #[allow(clippy::never_loop)]
        while let Some(event) = vrsys.poll_next_event()? {
            match event {
                vrapi::Event::RequestExit => {
                    log::info!("RequestExit");
                    app_state.stop();
                    vrsys.acknowledge_quit();
                }
            }
        }

        if *app_state.lock() == State::Stopping {
            // Ctrl-C or request exit
            break;
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
        ui_state.handle(&*vrsys)?;
        match ui_state.turn() {
            events::Action::ShowOverlay => {
                log::debug!("showing overlay");
                vrsys.show_overlay()?;
                {
                    let mut other_frame = frame.lock().unwrap();
                    *other_frame = Some(FrameInfo {
                        frame: splash.clone(),
                        frame_time: None,
                        bypass_pipeline: true,
                    });
                }
                app_state.start_capture();
            }
            events::Action::HideOverlay => {
                log::debug!("hiding overlay");
                vrsys.hide_overlay()?;
                maybe_current_frame = None;
                app_state.stop_capture();
            }
            _ => (),
        }
        if vrsys.get_action_state(vrapi::Action::Reposition)? {
            vrsys.set_position_mode(cfg.overlay.position)?;
        }
    }
    camera_thread.join().unwrap()?;
    Ok(())
}
