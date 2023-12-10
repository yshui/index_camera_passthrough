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
    log::debug!("Base directories: {:?}", xdg);
    let steam = xdg.find_data_file("steam")?;
    log::debug!("Steam directory: {:?}", steam);
    let steam_config = steam.join("config").join("lighthouse");
    log::debug!("Enumerating steam config dir {:?}", steam_config);
    let mut files = steam_config.read_dir().ok()?;
    files.find_map(|dir| {
        log::debug!("Trying to find config in {:?}", dir);
        let dir = dir.ok()?;
        let config = dir.path().join("config.json");
        log::debug!("Trying to config from {:?}", config);
        let json = std::fs::read_to_string(config).ok()?;
        log::debug!("Trying to parse config");
        let lhconfig: LighthouseConfig = serde_json::from_str(&json).ok()?;
        log::debug!("Trying to find left camera");
        let left = lhconfig
            .tracked_cameras
            .iter()
            .copied()
            .find(|p| p.name == Camera::Left)?;
        log::debug!("Trying to find right camera");
        let right = lhconfig
            .tracked_cameras
            .iter()
            .copied()
            .find(|p| p.name == Camera::Right)?;
        Some(StereoCamera { left, right })
    })
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
