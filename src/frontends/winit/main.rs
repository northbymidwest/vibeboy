use vibeboy::*;

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

    /// Connect a Game Boy Printer (saves PNGs to prints/ directory)
    #[arg(long)]
    pub printer: bool,
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();
    let mut app = app::App::new(cli);

    let event_loop = EventLoop::new().unwrap();
    event_loop.run_app(&mut app).unwrap();
}
