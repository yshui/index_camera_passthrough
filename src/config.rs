use serde::{Serialize, Deserialize};

/// Because your eye and the camera is at different physical locations, it is impossible
/// to project camera view into VR space perfectly. There are trade offs approximating
/// this projection. (viewing range means things too close to you will give you double vision).
#[derive(Eq, PartialEq, Debug, Serialize, Deserialize)]
pub enum ProjectionMode {
    /// in this mode, we assume your eyes are at the cameras' physical location. this mode
    /// has larger viewing range, but everything will smaller to you.
    FromCamera,
    /// in this mode, we assume your cameras are at your eyes' physical location. everything will
    /// have the right scale in this mode, but the viewing range is smaller.
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

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum Button {
    Menu,
    Grip,
    Trigger,
    A,
    B,
}

impl From<Button> for openvr_sys::EVRButtonId {
    fn from(b: Button) -> Self {
        use openvr_sys::EVRButtonId;
        match b {
            Button::Menu => EVRButtonId::k_EButton_ApplicationMenu,
            Button::Grip => EVRButtonId::k_EButton_Grip,
            Button::Trigger => EVRButtonId::k_EButton_Axis1,
            Button::A => EVRButtonId::k_EButton_Grip,
            Button::B => EVRButtonId::k_EButton_ApplicationMenu,
        }
    }
}

pub const fn default_toggle_button() -> Button {
    Button::Menu
}

pub const fn default_open_delay() -> std::time::Duration {
    std::time::Duration::ZERO
}

/// Index camera passthrough
#[derive(Debug, Serialize, Deserialize)]
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
    /// which button should toggle the overlay visibility. press things
    /// button on both controllers to toggle the overlay.
    #[serde(default = "default_toggle_button")]
    pub toggle_button: Button,
    /// how long does the button need to be held before the overlay open,
    /// closing the overlay is always instantaneous
    #[serde(default = "default_open_delay", with = "humantime_serde")]
    pub open_delay: std::time::Duration,
    /// enable debug option, including:
    ///   - use trigger button to do renderdoc capture
    #[serde(default)]
    pub debug: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            camera_device: "".to_owned(),
            overlay: Default::default(),
            display_mode: Default::default(),
            toggle_button: default_toggle_button(),
            open_delay: std::time::Duration::ZERO,
            debug: false,
        }
    }
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
