use serde::{Deserialize, Serialize};

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

/// Extract relevant bits of information from steam config files
#[derive(Serialize, Deserialize)]
pub struct LighthouseConfig {
    pub tracked_cameras: Vec<TrackedCamera>,
}
use anyhow::{anyhow, Context, Result};
pub fn load_steam_config(hmd_serial: &str) -> Result<StereoCamera> {
    let xdg = xdg::BaseDirectories::new()?;
    let steam = xdg
        .find_data_file("steam")
        .with_context(|| anyhow!("Cannot find steam directory"))?;
    let lhconfig = std::fs::read_to_string(
        steam
            .join("config")
            .join("lighthouse")
            .join(hmd_serial.to_lowercase())
            .join("config.json"),
    )?;
    let lhconfig: LighthouseConfig = serde_json::from_str(&lhconfig)?;
    let left = *lhconfig
        .tracked_cameras
        .iter()
        .find(|p| p.name == Camera::Left)
        .with_context(|| anyhow!("No left camera found"))?;
    let right = *lhconfig
        .tracked_cameras
        .iter()
        .find(|p| p.name == Camera::Right)
        .with_context(|| anyhow!("No right camera found"))?;

    Ok(StereoCamera { left, right })
}

pub fn default_lighthouse_config() -> StereoCamera {
    StereoCamera {
        left: TrackedCamera {
            name: Camera::Left,
            extrinsics: Extrinsics {
                position: [-0.067, -0.039, -0.07],
            },
            intrinsics: Intrinsics {
                center_x: 483.4,
                center_y: 453.6,
                focal_x: 411.4,
                focal_y: 411.4,
                distort: Distort {
                    coeffs: [0.17, 0.07, -0.24, 0.1],
                },
            },
        },
        right: TrackedCamera {
            name: Camera::Right,
            extrinsics: Extrinsics {
                position: [0.067, -0.039, -0.07],
            },
            intrinsics: Intrinsics {
                center_x: 489.2,
                center_y: 473.7,
                focal_x: 411.4,
                focal_y: 411.4,
                distort: Distort {
                    coeffs: [0.17, 0.07, -0.24, 0.1],
                },
            },
        },
    }
}
