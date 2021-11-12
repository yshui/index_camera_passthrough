#![feature(untagged_unions, try_trait_v2)]
mod distortion_correction;
mod openvr;
mod yuv;

use anyhow::{anyhow, Context, Result};
use ash::vk::Handle;
use openvr::*;
use v4l::video::Capture;
use vulkano::{
    buffer::CpuBufferPool,
    device::{self, physical::PhysicalDevice},
    instance::{Instance, InstanceExtensions, Version},
    sync::GpuFuture,
    VulkanObject,
};
use yuv::GpuYuyvConverter;

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

fn main() -> Result<()> {
    env_logger::init();
    let camera = v4l::Device::with_path(find_index_camera()?)?;
    if !camera
        .query_caps()?
        .capabilities
        .contains(v4l::capability::Flags::VIDEO_CAPTURE)
    {
        return Err(anyhow!("Cannot capture from index camera"));
    }
    let format = camera.set_format(&v4l::Format::new(1920, 960, v4l::FourCC::new(b"YUYV")))?;
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
        .union(&device.required_extensions());
        device::Device::new(
            device,
            &device::Features::none(),
            &extensions,
            [(queue_family, 1.0)],
        )?
    };
    let queue = queues.next().unwrap();
    let buffer = CpuBufferPool::upload(device.clone());

    // Create a VROverlay
    let vroverlay = vrsys.overlay();
    let mut overlay = vroverlay.create_overlay(APP_KEY, APP_NAME)?;
    vroverlay
        .pin_mut()
        .SetOverlayFlag(
            overlay.as_raw(),
            openvr_sys::VROverlayFlags::VROverlayFlags_SideBySide_Parallel,
            true,
        )
        .into_result()?;
    let transformation = openvr_sys::HmdMatrix34_t {
        m: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, -1.0],
        ],
    };
    unsafe {
        vroverlay.pin_mut().SetOverlayTransformAbsolute(
            overlay.as_raw(),
            openvr_sys::ETrackingUniverseOrigin::TrackingUniverseStanding,
            &transformation,
        )
    };

    // Set up post-processing passes
    let converted = vulkano::image::AttachmentImage::with_usage(
        device.clone(),
        [1920, 960],
        vulkano::format::Format::R8G8B8A8_UNORM,
        vulkano::image::ImageUsage {
            transfer_source: true,
            transfer_destination: false,
            sampled: true,
            storage: false,
            color_attachment: true,
            depth_stencil_attachment: false,
            transient_attachment: false,
            input_attachment: false,
        },
    )?;
    let converter = GpuYuyvConverter::new(device.clone(), converted.clone(), 1920, 960)?;
    let correction = distortion_correction::StereoCorrection::new(
        device.clone(),
        converted,
        [-0.17, 0.021, -0.001],
        [483.4, 453.6],
        [489.2, 473.7],
        0.428,
    )?;

    let mut event = std::mem::MaybeUninit::<openvr_sys::VREvent_t>::uninit();
    'main_loop: loop {
        let (frame, _metadata) = v4l::io::traits::CaptureStream::next(&mut video_stream)?;
        let future = converter.yuyv_buffer_to_vulkan_image(
            frame,
            vulkano::sync::now(device.clone()),
            queue.clone(),
            &buffer,
        )?;
        let (future, image) = correction.correct(future, queue.clone())?;
        future.then_signal_fence().wait(None)?;

        overlay.set_texture(
            1920,
            960,
            image,
            device.clone(),
            queue.clone(),
            instance.clone(),
        )?;
        if !vroverlay.pin_mut().IsOverlayVisible(overlay.as_raw()) {
            // Display the overlay
            vroverlay
                .pin_mut()
                .ShowOverlay(overlay.as_raw())
                .into_result()?;
        }
        while unsafe {
            vrsys.pin_mut().PollNextEvent(
                event.as_mut_ptr() as *mut _,
                std::mem::size_of::<openvr_sys::VREvent_t>() as u32,
            )
        } {
            let event = unsafe { event.assume_init_ref() };
            println!("{:?}", unsafe {
                std::mem::transmute::<_, openvr_sys::EVREventType>(event.eventType)
            });
            if event.eventType == openvr_sys::EVREventType::VREvent_ButtonPress as u32 {
                println!("{:?}", unsafe { event.data.controller.button });
                if unsafe { event.data.controller.button == 33 } {
                    break 'main_loop;
                }
            } else if event.eventType == openvr_sys::EVREventType::VREvent_Quit as u32 {
                vrsys.pin_mut().AcknowledgeQuit_Exiting();
                break 'main_loop;
            }
        }
        if !running.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
    }

    Ok(())
}
