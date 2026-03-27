mod commands;
mod debug_commands;
mod harness;
mod harnesses;
#[path = "model.rs"]
mod test_model;
mod svg;
mod util;

use clap::{Parser, Subcommand};
use vibeboy::model::GbModel;
use std::path::PathBuf;

use harness::run_tests;
use harnesses::blargg::BlarggHarness;
use harnesses::gambatte::GambatteHarness;
use harnesses::gbmicrotest::GbMicrotestHarness;
use harnesses::mooneye::MooneyeHarness;
use harnesses::tearoom::TearoomHarness;
use test_model::resolve_boot_rom;

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
        /// Scaling filter to apply (e.g. xbrz2x, xbr2x, super-xbr, hq2x, epx)
        #[arg(long)]
        filter: Option<String>,
        /// Use GPU shader for the filter instead of CPU implementation
        #[arg(long)]
        gpu: bool,
    },
    /// Vectorize an input PNG image to SVG or scaled PNG
    Vectorize {
        /// Path to input image
        path: PathBuf,
        /// Output file path (.svg for vector, .png for raster)
        #[arg(long, default_value = "output.svg")]
        out: String,
        /// Vectorize filter: vectorize (default), vectorize-diffusion, vectorize-spline-diffusion,
        /// vectorize-gpu (uses CPU fallback), vectorize-legacy, edge, gpu-full
        #[arg(long, default_value = "vectorize")]
        filter: String,
        /// Scale factor for raster output (default 4)
        #[arg(long, default_value = "4")]
        scale: usize,
        /// Use GPU shader for rasterization (where available)
        #[arg(long)]
        gpu: bool,
        /// Force CPU-only (no GPU shaders)
        #[arg(long)]
        cpu_filter: bool,
    },
    /// Generate a built-in boot ROM
    GenBootrom {
        /// Output file path
        #[arg(long, default_value = "bootroms/vibeboy_cgb_boot.bin")]
        out: String,
        /// Boot ROM type
        #[arg(long, default_value = "cgb")]
        model: String,
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

use vibeboy::ui_util::parse_model;

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
            let harness: Box<dyn harness::TestHarness> = match subcommand {
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
        Command::GenBootrom { out, model } => {
            let rom: &[u8] = match model.as_str() {
                "cgb" => vibeboy::bootrom::CGB,
                "agb" | "gba" => vibeboy::bootrom::AGB,
                "dmg" => vibeboy::bootrom::DMG,
                other => {
                    eprintln!("Unknown boot ROM model: {other}. Available: dmg, cgb, agb");
                    std::process::exit(1);
                }
            };
            std::fs::write(&out, rom).unwrap();
            eprintln!("Wrote {model} boot ROM: {out} ({} bytes)", rom.len());
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
            filter,
            gpu,
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
                filter.as_deref(),
                gpu,
            );
        }
        Command::Vectorize { path, out, filter, scale, gpu, cpu_filter } => {
            commands::cmd_vectorize(&path, &out, &filter, scale, gpu && !cpu_filter);
        }
        Command::Analyze {
            path,
            frames,
            model,
            boot,
            bootrom,
        } => {
            debug_commands::cmd_analyze(&path, model, boot, bootrom.as_deref(), frames);
        }
        Command::TraceTimer {
            path,
            model,
            boot,
            bootrom,
        } => {
            debug_commands::cmd_trace_timer(&path, model, boot, bootrom.as_deref());
        }
        Command::Calibrate { path } => {
            debug_commands::cmd_calibrate(&path);
        }
    }
}
