#[path = "../apu.rs"]
mod apu;
#[path = "../bus.rs"]
mod bus;
#[path = "../cartridge/mod.rs"]
mod cartridge;
#[path = "../cpu/mod.rs"]
mod cpu;
#[path = "../emulator.rs"]
mod emulator;
#[path = "../joypad.rs"]
mod joypad;
#[path = "../model.rs"]
mod model;
#[path = "../ppu/mod.rs"]
mod ppu;
#[path = "../printer.rs"]
mod printer;
#[path = "../serial.rs"]
mod serial;
#[path = "../sgb.rs"]
mod sgb;
#[path = "../snapshot.rs"]
mod snapshot;
#[path = "../snes/mod.rs"]
mod snes;
#[path = "../timer.rs"]
mod timer;
#[path = "../vectorize/mod.rs"]
mod vectorize;

#[path = "."]
mod test_runner {
    pub mod commands;
    pub mod harness;
    pub mod harnesses;
    #[path = "model.rs"]
    pub mod test_model;
    pub mod util;
}

use clap::{Parser, Subcommand};
use model::GbModel;
use std::path::PathBuf;

use test_runner::commands;
use test_runner::harness::run_tests;
use test_runner::harnesses::blargg::BlarggHarness;
use test_runner::harnesses::gambatte::GambatteHarness;
use test_runner::harnesses::gbmicrotest::GbMicrotestHarness;
use test_runner::harnesses::mooneye::MooneyeHarness;
use test_runner::harnesses::tearoom::TearoomHarness;
use test_runner::test_model::resolve_boot_rom;

#[derive(Parser)]
#[command(name = "test_runner", about = "Game Boy emulator test runner")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run test ROMs with a specific harness
    Test {
        #[command(subcommand)]
        subcommand: TestCommand,
    },
    /// Take a screenshot after N frames
    Screenshot {
        /// Path to ROM file
        path: PathBuf,
        /// Number of frames to run before capturing
        #[arg(long, default_value = "300")]
        frames: u32,
        /// Output file path
        #[arg(long, default_value = "screenshot.png")]
        out: String,
        /// Output format: png, svg, or raster (vectorized then rasterized)
        #[arg(long, default_value = "png")]
        format: String,
        /// Scale factor for raster format (default 4)
        #[arg(long, default_value = "4")]
        scale: usize,
        /// Button presses: "frame:button,frame:button" e.g. "100:a,200:start"
        #[arg(long, default_value = "")]
        keys: String,
        /// Hardware model override
        #[arg(long, value_parser = parse_model)]
        model: Option<GbModel>,
        /// Use boot ROM (auto-detected by model)
        #[arg(long)]
        boot: bool,
        /// Path to boot ROM file (implies --boot)
        #[arg(long)]
        bootrom: Option<PathBuf>,
    },
    /// Vectorize an input PNG image to SVG using Kopf-Lischinski
    Vectorize {
        /// Path to input image
        path: PathBuf,
        /// Output SVG file path
        #[arg(long, default_value = "output.svg")]
        out: String,
    },
    /// Analyze frame buffer (debug tool)
    Analyze {
        /// Path to ROM file
        path: PathBuf,
        /// Number of frames to run
        #[arg(long, default_value = "300")]
        frames: u32,
        /// Hardware model override
        #[arg(long, value_parser = parse_model)]
        model: Option<GbModel>,
        /// Use boot ROM (auto-detected by model)
        #[arg(long)]
        boot: bool,
        /// Path to boot ROM file (implies --boot)
        #[arg(long)]
        bootrom: Option<PathBuf>,
    },
    /// Trace timer state around boot ROM handoff (debug tool)
    TraceTimer {
        /// Path to ROM file
        path: PathBuf,
        /// Hardware model override
        #[arg(long, value_parser = parse_model)]
        model: Option<GbModel>,
        /// Use boot ROM (auto-detected by model)
        #[arg(long)]
        boot: bool,
        /// Path to boot ROM file (implies --boot)
        #[arg(long)]
        bootrom: Option<PathBuf>,
    },
    /// Dump PPU/timer state at PC=$0100 for all models with boot ROMs
    Calibrate {
        /// Path to ROM file
        path: PathBuf,
    },
}

#[derive(clap::Args)]
struct TestArgs {
    /// Path to ROM file or directory
    path: PathBuf,
    /// Hardware model override
    #[arg(long, value_parser = parse_model)]
    model: Option<GbModel>,
    /// Use boot ROM (auto-detected by model)
    #[arg(long)]
    boot: bool,
    /// Path to boot ROM file (implies --boot)
    #[arg(long)]
    bootrom: Option<PathBuf>,
    /// Print extra diagnostics per test
    #[arg(long)]
    verbose: bool,
    /// Only print summary
    #[arg(long)]
    quiet: bool,
}

#[derive(Subcommand)]
enum TestCommand {
    /// Mooneye tests (breakpoint + Fibonacci register check)
    Mooneye(TestArgs),
    /// Blargg tests (serial output detection)
    Blargg(TestArgs),
    /// Gambatte tests (hex output comparison after 15 frames)
    Gambatte(TestArgs),
    /// GBMicrotest (HRAM result check after 2 frames)
    Gbmicrotest(TestArgs),
    /// Mealybug Tearoom tests (screenshot comparison after LD B,B breakpoint)
    Tearoom(TestArgs),
}

fn parse_model(s: &str) -> Result<GbModel, String> {
    s.parse::<GbModel>()
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Command::Test { subcommand } => {
            let args = match &subcommand {
                TestCommand::Mooneye(a) | TestCommand::Blargg(a) | TestCommand::Gambatte(a)
                | TestCommand::Gbmicrotest(a) | TestCommand::Tearoom(a) => a,
            };
            let model = args.model;
            let verbose = args.verbose;
            let quiet = args.quiet;
            let path = args.path.clone();
            let harness: Box<dyn test_runner::harness::TestHarness> = match subcommand {
                TestCommand::Mooneye(args) => {
                    let br = resolve_boot_rom(
                        args.boot,
                        args.bootrom.as_deref(),
                        model.unwrap_or(GbModel::Dmg),
                    );
                    Box::new(MooneyeHarness { force_model: model, boot_rom: br })
                }
                TestCommand::Blargg(_) => Box::new(BlarggHarness { force_model: model }),
                TestCommand::Gambatte(_) => Box::new(GambatteHarness { force_model: model }),
                TestCommand::Gbmicrotest(_) => Box::new(GbMicrotestHarness { force_model: model }),
                TestCommand::Tearoom(_) => Box::new(TearoomHarness { force_model: model }),
            };
            run_tests(&path, harness.as_ref(), verbose, quiet);
        }
        Command::Screenshot {
            path,
            frames,
            out,
            format,
            scale,
            keys,
            model,
            boot,
            bootrom,
        } => {
            commands::cmd_screenshot(
                &path,
                model,
                boot,
                bootrom.as_deref(),
                frames,
                &out,
                &format,
                scale,
                &keys,
            );
        }
        Command::Vectorize { path, out } => {
            commands::cmd_vectorize(&path, &out);
        }
        Command::Analyze {
            path,
            frames,
            model,
            boot,
            bootrom,
        } => {
            commands::cmd_analyze(&path, model, boot, bootrom.as_deref(), frames);
        }
        Command::TraceTimer {
            path,
            model,
            boot,
            bootrom,
        } => {
            commands::cmd_trace_timer(&path, model, boot, bootrom.as_deref());
        }
        Command::Calibrate { path } => {
            commands::cmd_calibrate(&path);
        }
    }
}
