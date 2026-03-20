pub mod apu;
pub mod bus;
pub mod cartridge;
pub mod cpu;
pub mod emulator;
pub mod joypad;
pub mod model;
pub mod ppu;
pub mod serial;
pub mod sgb;
pub mod snapshot;
pub mod snes;
pub mod timer;

#[cfg(not(target_arch = "wasm32"))]
pub mod printer;

#[cfg(not(target_arch = "wasm32"))]
pub mod scaling;

#[cfg(not(target_arch = "wasm32"))]
pub mod vectorize;

// wgpu vectorize pipeline — available for both native and wasm
#[cfg(feature = "web")]
#[path = "scaling/wgpu_vectorize.rs"]
pub mod wgpu_vectorize;

#[cfg(not(target_arch = "wasm32"))]
pub mod ui_util;

#[cfg(feature = "web")]
pub mod web_printer;

#[cfg(feature = "web")]
pub mod web;
