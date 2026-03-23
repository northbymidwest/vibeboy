use vibeboy::emulator::Emulator;
use vibeboy::joypad;
use vibeboy::model::GbModel;
use std::fs;
use std::path::{Path, PathBuf};

/// Game Boy frame buffer width in pixels.
pub const GB_FB_WIDTH: usize = 160;
/// Game Boy frame buffer height in pixels.
pub const GB_FB_HEIGHT: usize = 144;

pub fn make_emu(rom: Vec<u8>, boot_rom: Option<Vec<u8>>, model: GbModel) -> Emulator {
    let mut emu = Emulator::new(rom, boot_rom, model, None, vibeboy::clock::default_clock());
    emu.set_headless(true);
    emu
}

pub fn collect_roms(dir: &Path, out: &mut Vec<PathBuf>) {
    if dir.is_file() {
        let ext = dir.extension().and_then(|e| e.to_str());
        if ext == Some("gb") || ext == Some("gbc") {
            out.push(dir.to_path_buf());
        }
        return;
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            collect_roms(&entry.path(), out);
        }
    }
}

pub fn parse_keys(keys: &str) -> Vec<(u32, u8, bool)> {
    let mut events = Vec::new();
    if keys.is_empty() {
        return events;
    }
    for part in keys.split(',') {
        let parts: Vec<&str> = part.trim().split(':').collect();
        if parts.len() != 2 {
            continue;
        }
        let frame: u32 = parts[0].parse().unwrap_or(0);
        let btn = match parts[1].to_lowercase().as_str() {
            "a" => joypad::BTN_A,
            "b" => joypad::BTN_B,
            "start" => joypad::BTN_START,
            "select" => joypad::BTN_SELECT,
            "up" => joypad::BTN_UP,
            "down" => joypad::BTN_DOWN,
            "left" => joypad::BTN_LEFT,
            "right" => joypad::BTN_RIGHT,
            _ => continue,
        };
        events.push((frame, btn, true));
        events.push((frame + 4, btn, false));
    }
    events.sort_by_key(|e| e.0);
    events
}
