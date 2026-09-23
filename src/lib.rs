pub mod apu;
pub mod bootrom;
pub mod bus;
pub mod cartridge;
pub mod clock;
pub mod cpu;
pub mod emulator;
pub mod joypad;
pub mod model;
pub mod ppu;
pub mod printer;
pub mod rewind;
pub mod savestate;
pub mod serial;
pub mod sgb;
pub mod snapshot;
pub mod snes;
pub mod timer;

pub mod scaling;

pub mod ui_util;
pub mod util;

#[cfg(target_os = "macos")]
pub mod macos_accel;

#[cfg(feature = "web")]
#[path = "frontends/web/mod.rs"]
pub mod web;

#[cfg(feature = "libretro")]
#[path = "frontends/libretro/mod.rs"]
pub mod libretro;
