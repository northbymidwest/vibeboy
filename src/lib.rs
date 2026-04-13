pub mod clock;
pub mod apu;
pub mod bootrom;
pub mod bus;
pub mod cartridge;
pub mod cpu;
pub mod emulator;
pub mod joypad;
pub mod model;
pub mod ppu;
pub mod serial;
pub mod sgb;
pub mod rewind;
pub mod savestate;
pub mod snapshot;
pub mod snes;
pub mod timer;
pub mod printer;

pub mod scaling;

pub mod util;
pub mod ui_util;

#[cfg(target_os = "macos")]
pub mod macos_accel;

#[cfg(feature = "web")]
#[path = "frontends/web/mod.rs"]
pub mod web;

#[cfg(feature = "libretro")]
#[path = "frontends/libretro/mod.rs"]
pub mod libretro;
