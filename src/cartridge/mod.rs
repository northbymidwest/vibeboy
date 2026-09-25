mod huc1;
mod huc3;
mod mbc1;
mod mbc2;
mod mbc3;
mod mbc5;
mod mbc6;
mod mbc7;
mod mmm01;
mod pocket_camera;
/// Cartridge abstraction — all known Game Boy mappers.
mod rom_only;
mod rom_ram;
mod tama5;

use huc1::HuC1;
use huc3::HuC3;
use mbc1::Mbc1;
use mbc2::Mbc2;
use mbc3::Mbc3;
use mbc5::Mbc5;
use mbc6::Mbc6;
use mbc7::Mbc7;
use mmm01::Mmm01;
use pocket_camera::PocketCamera;
use rom_only::RomOnly;
use rom_ram::RomRam;
use tama5::Tama5;

use std::sync::Arc;

use crate::clock::Clock;

pub trait Cartridge: Send {
    fn read_rom(&self, addr: u16) -> u8;
    fn write_rom(&mut self, addr: u16, val: u8);
    fn read_ram(&self, addr: u16) -> u8;
    fn write_ram(&mut self, addr: u16, val: u8);
    fn has_battery(&self) -> bool {
        false
    }
    fn ram_data(&self) -> &[u8] {
        &[]
    }
    /// Returns save data (may include extra metadata like RTC state).
    fn save_data(&self) -> Vec<u8> {
        self.ram_data().to_vec()
    }
    fn load_ram(&mut self, _data: &[u8]) {}
    /// Returns true if this cartridge has a camera sensor (Pocket Camera).
    fn has_camera(&self) -> bool {
        false
    }
    /// Feed a 128×112 grayscale image from a webcam into the camera sensor.
    fn set_camera_image(&mut self, _grayscale: &[u8; 128 * 112]) {}
    /// Returns true if this cartridge has a rumble motor (MBC5+Rumble).
    fn has_rumble(&self) -> bool {
        false
    }
    /// Returns true if the rumble motor is currently on.
    fn rumble_active(&self) -> bool {
        false
    }
    /// Returns true if rumble was active at any point since the last call, then clears
    /// the latch. Call once per frame to check for rumble pulses.
    fn drain_rumble(&mut self) -> bool {
        false
    }
    /// Returns true if this cartridge has an accelerometer (MBC7).
    fn has_accelerometer(&self) -> bool {
        false
    }
    /// Feed accelerometer values in MBC7 u16 format (center = 0x81D0).
    fn set_accelerometer(&mut self, _x: u16, _y: u16) {}
    /// Snapshot mutable cartridge state (registers + RAM, not ROM) for save states / rewind.
    fn snapshot_state(&self) -> CartState;
    /// Check that `state` was taken from this mapper type with the same RAM
    /// size, and that its registers hold values the mapper can produce.
    fn validate_state(&self, state: &CartState) -> Result<(), &'static str>;
    /// Restore mutable cartridge state from a snapshot that passed
    /// `validate_state`. Panics if `state` is for a different mapper type.
    fn restore_state(&mut self, state: &CartState);
}

/// Mutable state of each mapper, for save states and rewind. The ROM, the
/// wiring detected from the header (battery, rumble, MBC30, multicart) and
/// external sensor input are not part of it. RTC mappers also leave out their
/// wall-clock base, which restarts from the current time on restore.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub enum CartState {
    RomOnly,
    RomRam(rom_ram::RomRamState),
    Mbc1(mbc1::Mbc1State),
    Mbc2(mbc2::Mbc2State),
    Mbc3(mbc3::Mbc3State),
    Mbc5(mbc5::Mbc5State),
    Mbc6(mbc6::Mbc6State),
    Mbc7(mbc7::Mbc7State),
    Mmm01(mmm01::Mmm01State),
    PocketCamera(pocket_camera::PocketCameraState),
    Tama5(tama5::Tama5State),
    HuC1(huc1::HuC1State),
    HuC3(huc3::HuC3State),
}

const WRONG_MAPPER: &str = "save state is for a different cartridge mapper";

/// Fail with `msg` unless `ok`.
fn ensure(ok: bool, msg: &'static str) -> Result<(), &'static str> {
    if ok { Ok(()) } else { Err(msg) }
}

/// Fail unless a saved RAM (or flash) image has the size this cartridge uses.
fn ensure_ram_len(saved: &[u8], own: &[u8]) -> Result<(), &'static str> {
    ensure(
        saved.len() == own.len(),
        "save state cartridge RAM size does not match",
    )
}

/// Error for registers outside the range the mapper can produce.
const BAD_REGISTERS: &str = "save state cartridge registers out of range";

/// Construct the appropriate cartridge from a ROM image.
pub fn make_cartridge(rom: Arc<[u8]>, clock: Arc<dyn Clock>) -> Box<dyn Cartridge> {
    let cart_type = rom.get(0x0147).copied().unwrap_or(0);
    let ram_size: usize = match rom.get(0x0149).copied().unwrap_or(0) {
        0x01 => 0x0800,
        0x02 => 0x2000,
        0x03 => 0x8000,
        0x04 => 0x20000,
        0x05 => 0x10000,
        _ => 0,
    };

    log::info!(
        "Cart type={:#04X} title={} ram_size={:#X}",
        cart_type,
        rom.get(0x0134..0x0143)
            .and_then(|s| std::str::from_utf8(s).ok())
            .unwrap_or("?")
            .trim_matches('\0'),
        ram_size
    );

    match cart_type {
        0x00 => Box::new(RomOnly::new(rom)),
        0x01..=0x03 => {
            let battery = cart_type == 0x03;
            Box::new(Mbc1::new(rom, ram_size, battery))
        }
        0x05 | 0x06 => {
            let battery = cart_type == 0x06;
            Box::new(Mbc2::new(rom, battery))
        }
        0x08 | 0x09 => {
            let battery = cart_type == 0x09;
            Box::new(RomRam::new(rom, battery))
        }
        0x0B..=0x0D => {
            let battery = cart_type == 0x0D;
            Box::new(Mmm01::new(rom, ram_size, battery))
        }
        0x0F..=0x13 => {
            let battery = matches!(cart_type, 0x0F | 0x10 | 0x13);
            let has_rtc = matches!(cart_type, 0x0F | 0x10);
            Box::new(Mbc3::new(rom, ram_size, battery, has_rtc, clock.clone()))
        }
        0x19..=0x1E => {
            let battery = matches!(cart_type, 0x1B | 0x1E);
            let has_rumble = matches!(cart_type, 0x1C..=0x1E);
            Box::new(Mbc5::new(rom, ram_size, battery, has_rumble))
        }
        0x20 => Box::new(Mbc6::new(rom, ram_size)),
        0x22 => Box::new(Mbc7::new(rom)),
        0xFC => Box::new(PocketCamera::new(rom)),
        0xFD => Box::new(Tama5::new(rom, clock.clone())),
        0xFE => Box::new(HuC3::new(rom, ram_size, clock.clone())),
        0xFF => Box::new(HuC1::new(rom, ram_size)),
        other => {
            log::warn!("Unsupported cart type {:#04X}, using ROM-only", other);
            Box::new(RomOnly::new(rom))
        }
    }
}
