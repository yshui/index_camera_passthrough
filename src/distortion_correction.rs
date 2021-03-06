use anyhow::{anyhow, Result};
use log::{info, trace};
use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage::OneTimeSubmit, SubpassContents,
    },
    descriptor_set::PersistentDescriptorSet,
    device::{Device, Queue},
    image::{view::ImageView, AttachmentImage},
    pipeline::GraphicsPipeline,
    pipeline::{viewport::Viewport, PipelineBindPoint},
    render_pass::{Framebuffer, RenderPass, Subpass},
    sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode},
    sync::GpuFuture,
};

#[derive(Default, Debug, Clone)]
#[allow(non_snake_case)]
struct Vertex {
    position: [f32; 2],
    inCoord: [f32; 2],
}
vulkano::impl_vertex!(Vertex, position, inCoord);

/// Lens correction for a stereo side-by-side image
pub struct StereoCorrection {
    device: Arc<Device>,
    // Left
    render_pass1: Arc<RenderPass>,
    pipeline1: Arc<GraphicsPipeline>,
    desc_set1: Arc<PersistentDescriptorSet>,
    // Right
    render_pass2: Arc<RenderPass>,
    pipeline2: Arc<GraphicsPipeline>,
    desc_set2: Arc<PersistentDescriptorSet>,
}

impl StereoCorrection {
    /// i.e. solving Undistort(src) = dst for the smallest non-zero root.
    fn undistort_inverse(coeff: &[f64; 4], dst: f64) -> Option<f64> {
        let dst = dst as f64;
        // solving: x * (1 + k1*x^2 + k2*x^4 + k3*x^6 + k4*x^8) - dst = 0
        let f = |x: f64| {
            let x2 = x * x;
            x * (1.0 + x2 * (coeff[0] + x2 * (coeff[1] + x2 * (coeff[2] + x2 * coeff[3])))) - dst
        };
        let fp = |x: f64| {
            let x2 = x * x;
            1.0 + x2
                * (3.0 * coeff[0]
                    + x2 * (5.0 * coeff[1] + x2 * (7.0 * coeff[2] + x2 * 9.0 * coeff[3])))
        };
        const MAX_ITER: u32 = 100;
        let mut x = 0.0;
        for _ in 0..MAX_ITER {
            if fp(x) == 0.0 {
                // Give up
                info!("Divided by zero");
                return None;
            }
            trace!("{} {} {}", x, f(x), fp(x));
            if f(x).abs() < 1e-6 {
                info!("Inverse is: {}, {} {}", x, f(x), dst);
                return Some(x);
            }
            x = x - f(x) / fp(x);
        }
        // Give up
        info!("Cannot find scale");
        None
    }
    // Find a scale that maps the middle point of 4 edges of the undistorted image to
    // the edge of the field of view of the distorted image.
    //
    // Returns the scales and the adjusted fovs
    fn find_scale(
        coeff: &[f64; 4],
        center: &[f64; 2],
        focal: &[f64; 2],
    ) -> [(f64, f64); 2] {
        [0, 1].map(|i| {
            let min_edge_dist = center[i].min(1.0 - center[i]) / focal[i];
            // Find the input theta angle where Undistort(theta) = min_edge_dist
            if let Some(theta) = Self::undistort_inverse(coeff, min_edge_dist) {
                if theta >= std::f64::consts::PI / 2.0 {
                    // infinity?
                    (1.0, focal[i])
                } else {
                    // Find the input coordinates that will give us that theta
                    let target_edge = theta.tan();
                    log::info!("{}", target_edge);
                    (
                        target_edge / (0.5 / focal[i]),
                        1.0 / min_edge_dist / 2.0,
                    )
                }
            } else {
                // Cannot find scale so just don't scale
                (1.0, focal[i])
            }
        })
    }
    /// Input size is (size * 2, size)
    /// returns also the adjusted FOV for left and right
    pub fn new(
        device: Arc<Device>,
        input: Arc<AttachmentImage>,
        camera_calib: &crate::steam::StereoCamera,
    ) -> Result<(Self, [f64; 2], [f64; 2])> {
        if input.dimensions()[0] != input.dimensions()[1] * 2 {
            return Err(anyhow!("Input not square"));
        }
        let size = input.dimensions()[1] as f64;
        let center_left = [
            camera_calib.left.intrinsics.center_x / size,
            camera_calib.left.intrinsics.center_y / size,
        ];
        let center_right = [
            camera_calib.right.intrinsics.center_x / size,
            camera_calib.right.intrinsics.center_y / size,
        ];
        let focal_left = [
            camera_calib.left.intrinsics.focal_x / size,
            camera_calib.left.intrinsics.focal_y / size,
        ];
        let focal_right = [
            camera_calib.right.intrinsics.focal_x / size,
            camera_calib.right.intrinsics.focal_y / size,
        ];
        let coeff_left = camera_calib.left.intrinsics.distort.coeffs;
        let coeff_right = camera_calib.left.intrinsics.distort.coeffs;
        let vs = vs::Shader::load(device.clone())?;
        let fs = fs::Shader::load(device.clone())?;
        let render_pass1 = Arc::new(
            vulkano::single_pass_renderpass!(device.clone(),
                attachments: {
                    color: {
                        load: DontCare,
                        store: Store,
                        format: vulkano::format::Format::R8G8B8A8_UNORM,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {}
                }
            )
            .unwrap(),
        );
        let render_pass2 = Arc::new(
            vulkano::single_pass_renderpass!(device.clone(),
                attachments: {
                    color: {
                        load: Load,
                        store: Store,
                        format: vulkano::format::Format::R8G8B8A8_UNORM,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {}
                }
            )
            .unwrap(),
        );
        let pipeline1 = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_strip()
                .viewports([Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [size as f32, size as f32],
                    depth_range: -1.0..1.0,
                }])
                .fragment_shader(fs.main_entry_point(), ())
                .render_pass(Subpass::from(render_pass1.clone(), 0).unwrap())
                .build(device.clone())?,
        );
        let pipeline2 = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_strip()
                .viewports([Viewport {
                    origin: [size as f32, 0.0],
                    dimensions: [size as f32, size as f32],
                    depth_range: -1.0..1.0,
                }])
                .fragment_shader(fs.main_entry_point(), ())
                .render_pass(Subpass::from(render_pass2.clone(), 0).unwrap())
                .build(device.clone())?,
        );
        let sampler = Sampler::new(
            device.clone(),
            Filter::Linear,
            Filter::Linear,
            MipmapMode::Nearest,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            0.0,
            1.0,
            0.0,
            0.0,
        )?;

        // Left pass
        let desc_set_layout = pipeline1.layout().descriptor_set_layouts().get(0).unwrap();
        let mut desc_set_builder = PersistentDescriptorSet::start(desc_set_layout.clone());
        let scale_fov_left = Self::find_scale(&coeff_left, &center_left, &focal_left);
        let uniform = fs::ty::Parameters {
            center: center_left.map(|x| x as f32),
            dcoef: coeff_left.map(|x| x as f32),
            focal: focal_left.map(|x| x as f32),
            sensorSize: size as f32,
            scale: [scale_fov_left[0].0 as f32, scale_fov_left[1].0 as f32],
            texOffset: [0.0, 0.0],
            _dummy0: Default::default(),
        };
        let uniform = CpuAccessibleBuffer::from_data(
            device.clone(),
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::none()
            },
            false,
            uniform,
        )?;

        desc_set_builder
            .add_buffer(uniform)?
            .add_sampled_image(ImageView::new(input.clone())?, sampler.clone())?;
        let desc_set1 = Arc::new(desc_set_builder.build()?);

        // Right pass
        let desc_set_layout = pipeline2.layout().descriptor_set_layouts().get(0).unwrap();
        let mut desc_set_builder = PersistentDescriptorSet::start(desc_set_layout.clone());
        let scale_fov_right = Self::find_scale(&coeff_right, &center_right, &focal_right);
        let uniform = fs::ty::Parameters {
            center: center_right.map(|x| x as f32),
            dcoef: coeff_right.map(|x| x as f32),
            focal: focal_right.map(|x| x as f32),
            sensorSize: size as f32,
            scale: [scale_fov_right[0].0 as f32, scale_fov_right[1].0 as f32],
            texOffset: [0.5, 0.0], // right side of the texture
            _dummy0: Default::default(),
        };
        let uniform = CpuAccessibleBuffer::from_data(
            device.clone(),
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::none()
            },
            false,
            uniform,
        )?;

        desc_set_builder
            .add_buffer(uniform)?
            .add_sampled_image(ImageView::new(input)?, sampler)?;
        let desc_set2 = Arc::new(desc_set_builder.build()?);

        Ok((
            Self {
                device,
                render_pass1,
                pipeline1,
                desc_set1,
                render_pass2,
                pipeline2,
                desc_set2,
            },
            [scale_fov_left[0].1, scale_fov_left[1].1],
            [scale_fov_right[0].1, scale_fov_right[1].1],
        ))
    }
    pub fn correct(
        &self,
        after: impl GpuFuture,
        queue: Arc<Queue>,
        output: Arc<AttachmentImage>,
    ) -> Result<impl GpuFuture> {
        use vulkano::device::DeviceOwned;
        if queue.device() != &self.device {
            return Err(anyhow!("Device mismatch"));
        }
        if let Some(queue) = after.queue() {
            if !queue.is_same(&queue) {
                return Err(anyhow!("Queue mismatch"));
            }
        }
        let mut cmdbuf =
            AutoCommandBufferBuilder::primary(self.device.clone(), queue.family(), OneTimeSubmit)?;
        let vertex_buffer = CpuAccessibleBuffer::<[Vertex]>::from_iter(
            self.device.clone(),
            BufferUsage::vertex_buffer(),
            false,
            [
                Vertex {
                    position: [-1.0, -1.0],
                    inCoord: [-0.5, -0.5],
                },
                Vertex {
                    position: [-1.0, 1.0],
                    inCoord: [-0.5, 0.5],
                },
                Vertex {
                    position: [1.0, -1.0],
                    inCoord: [0.5, -0.5],
                },
                Vertex {
                    position: [1.0, 1.0],
                    inCoord: [0.5, 0.5],
                },
            ]
            .iter()
            .cloned(),
        )
        .unwrap();
        // Left side
        let framebuffer = Arc::new(
            Framebuffer::start(self.render_pass1.clone())
                .add(ImageView::new(output.clone())?)?
                .build()?,
        );
        cmdbuf
            .begin_render_pass(
                framebuffer,
                SubpassContents::Inline,
                [vulkano::format::ClearValue::None],
            )?
            .bind_pipeline_graphics(self.pipeline1.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline1.layout().clone(),
                0,
                self.desc_set1.clone(),
            )
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .draw(vertex_buffer.len() as u32, 1, 0, 0)?
            .end_render_pass()?;
        // Right side
        let framebuffer = Arc::new(
            Framebuffer::start(self.render_pass2.clone())
                .add(ImageView::new(output)?)?
                .build()?,
        );
        cmdbuf
            .begin_render_pass(
                framebuffer,
                SubpassContents::Inline,
                [vulkano::format::ClearValue::None],
            )?
            .bind_pipeline_graphics(self.pipeline2.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline2.layout().clone(),
                0,
                self.desc_set2.clone(),
            )
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .draw(vertex_buffer.len() as u32, 1, 0, 0)?
            .end_render_pass()?;
        Ok(after.then_execute(queue, cmdbuf.build()?)?)
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "#version 450
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 inCoord;
layout(location = 0) out vec2 coord;

void main() {
    gl_Position = vec4(position, 0, 1);
    coord = inCoord;
}"
    }
}
mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/stereo_correction.frag",
    }
}
