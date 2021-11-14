use serde::{Serialize, Deserialize};

/// Because your eye and the camera is at different physical locations, it is impossible
/// to project camera view into VR space perfectly. There are trade offs approximating
/// this projection. (viewing range means you must be within this distance from the real world
/// objects you are looking at).
#[derive(Eq, PartialEq, Debug, Serialize, Deserialize)]
pub enum ProjectionMode {
    /// in this mode, we assume your eyes are at the cameras' physical location. this mode
    /// has larger viewing range (~2m), but everything will _seem_ smaller to you.
    FromCamera,
    /// in this mode, we assume your cameras are at your eyes' physical location. everything will
    /// have the right scale in this mode, but the viewing range (~1m) is smaller.
    FromEye,
}

impl Default for ProjectionMode {
    fn default() -> Self {
        Self::FromCamera
    }
}
pub const fn default_overlay_distance() -> f32 { 1.0 }
#[derive(PartialEq, Debug, Serialize, Deserialize)]
#[serde(tag = "mode")]
pub enum PositionMode {
    /// the overlay is shown right in front of your HMD
    Hmd {
        /// how far away should the overlay be
        #[serde(default = "default_overlay_distance")]
        distance: f32,
    },
    /// the overlay is at a fixed location in space
    Absolute {
        /// transformation matrix for the overlay
        transform: [[f32; 4]; 4]
    }
}

impl Default for PositionMode {
    fn default() -> Self {
        Self::Hmd { distance: 1.0 }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Eye {
    Left, 
    Right,
}

pub const fn default_display_eye() -> Eye { Eye::Left }

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "mode")]
pub enum DisplayMode {
    /// display a stereo image on the overlay. conceptually the overlay becomes a portal from VR
    /// space to real world. you will be able to see more of the real world if the overlay occupys
    /// more of your field of view.
    Stereo {
        /// how is the camera's image projected onto the overlay
        #[serde(default)]
        projection_mode: ProjectionMode,
    },
    /// display one of the camera's image on the overlay
    Flat {
        /// which camera's image to display
        #[serde(default = "default_display_eye")]
        eye: Eye
    }
}

impl Default for DisplayMode {
    fn default() -> Self {
        Self::Flat { eye: Eye::Left }
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct OverlayConfig {
    /// how is the overlay positioned
    #[serde(default)]
    pub position: PositionMode,

}

/// Index camera passthrough
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Config {
    /// camera device to use. auto detect if not set
    #[serde(default)]
    pub camera_device: String,
    /// overlay related configuration
    #[serde(default)]
    pub overlay: OverlayConfig,
    /// how is the camera view displayed on the overlay
    #[serde(default)]
    pub display_mode: DisplayMode,
    /// enable debug option, including:
    ///   - use trigger button to do renderdoc capture
    #[serde(default)]
    pub debug: bool,
}

use anyhow::Result;
pub fn load_config() -> Result<Config> {
    let xdgdir = xdg::BaseDirectories::new()?;
    if let Some(f) = xdgdir.find_config_file("index_camera_passthrough.toml") {
        let cfg = std::fs::read_to_string(f)?;
        Ok(toml::from_str(&cfg)?)
    } else {
        Ok(Default::default())
    }
}
