use serde::{Deserialize, Serialize};

use crate::vrapi::{Camera, StereoCamera, TrackedCamera};

/// Extract relevant bits of information from steam config files
#[derive(Serialize, Deserialize)]
pub struct LighthouseConfig {
    pub tracked_cameras: Vec<TrackedCamera>,
}
use anyhow::{anyhow, Context, Result};
/// Try to find the config file for index
pub fn find_steam_config() -> Option<StereoCamera> {
    let xdg = xdg::BaseDirectories::new().ok()?;
    let steam = xdg.find_data_file("steam")?;
    let steam_config = steam.join("config").join("ligthouse");
    steam_config
        .read_dir()
        .ok()?
        .filter_map(|dir| {
            let dir = dir.ok()?;
            let config = dir.path().join("config.json");
            let json = std::fs::read_to_string(config).ok()?;
            let lhconfig: LighthouseConfig = serde_json::from_str(&json).ok()?;
            let left = lhconfig
                .tracked_cameras
                .iter()
                .copied()
                .find(|p| p.name == Camera::Left)?;
            let right = lhconfig
                .tracked_cameras
                .iter()
                .copied()
                .find(|p| p.name == Camera::Right)?;
            Some(StereoCamera { left, right })
        })
        .next()
}
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
