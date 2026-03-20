#[path = "../../apu.rs"]
mod apu;
#[path = "../../bus.rs"]
mod bus;
#[path = "../../cartridge/mod.rs"]
mod cartridge;
#[path = "../../cpu/mod.rs"]
mod cpu;
#[path = "../../emulator.rs"]
mod emulator;
#[path = "../../joypad.rs"]
mod joypad;
#[path = "../../model.rs"]
mod model;
#[path = "../../ppu/mod.rs"]
mod ppu;
#[path = "../../printer.rs"]
mod printer;
#[path = "../../serial.rs"]
mod serial;
#[path = "../../sgb.rs"]
mod sgb;
#[path = "../../savestate.rs"]
mod savestate;
#[path = "../../snapshot.rs"]
mod snapshot;
#[path = "../../snes/mod.rs"]
mod snes;
#[path = "../../timer.rs"]
mod timer;
#[path = "../../scaling/mod.rs"]
mod scaling;
#[path = "../../vectorize/mod.rs"]
mod vectorize;
#[path = "../../ui_util.rs"]
mod ui_util;

mod gpu;
mod audio;
mod camera;
mod menu;
mod app;

use clap::Parser;
use model::GbModel;
use std::path::PathBuf;
use winit::event_loop::EventLoop;

pub(crate) const SCALE: u32 = 3;
pub(crate) const GB_W: u32 = 160;
pub(crate) const GB_H: u32 = 144;
pub(crate) const SGB_W: u32 = 256;
pub(crate) const SGB_H: u32 = 224;
pub(crate) const AUDIO_SAMPLE_RATE: u32 = 96_000;

#[derive(Parser)]
#[command(name = "vibeboy", about = "Game Boy / Game Boy Color emulator (winit frontend)")]
pub(crate) struct Cli {
    /// Path to ROM file (.gb / .gbc). If omitted, a file dialog will open.
    pub rom: Option<PathBuf>,

    /// Path to boot ROM file (auto-detected if not specified)
    #[arg(long)]
    pub bootrom: Option<PathBuf>,

    /// Hardware model (auto-detected from cart header if not specified)
    #[arg(long, value_parser = |s: &str| s.parse::<GbModel>())]
    pub model: Option<GbModel>,

    /// Skip boot ROM
    #[arg(long)]
    pub no_boot: bool,
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();
    let mut app = app::App::new(cli);

    let event_loop = EventLoop::new().unwrap();
    event_loop.run_app(&mut app).unwrap();
}
