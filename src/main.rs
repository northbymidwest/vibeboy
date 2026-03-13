mod apu;
mod bus;
mod cartridge;
mod cpu;
mod emulator;
mod joypad;
mod scaling;
mod model;
mod ppu;
mod printer;
mod serial;
mod sgb;
mod snapshot;
mod snes;
mod timer;
mod vectorize;
#[cfg(target_os = "macos")]
mod macos_accel;

use clap::Parser;
use emulator::Emulator;
use model::GbModel;
use sdl3::audio::{AudioFormat, AudioSpec};
use sdl3::dialog::{self, DialogFileFilter};
use sdl3::event::Event;
use sdl3::keyboard::{Keycode, Scancode};
use sdl3::pixels::PixelFormat;
use sdl3::render::ScaleMode;
use sdl3::sys::camera::{
    SDL_AcquireCameraFrame, SDL_Camera, SDL_CameraSpec, SDL_CloseCamera, SDL_GetCameras,
    SDL_OpenCamera, SDL_ReleaseCameraFrame,
};
use sdl3::sys::pixels::{SDL_Colorspace, SDL_PixelFormat as SysPixelFormat};
use sdl3::sensor::{SensorData, SensorType};
use sdl3::sys::stdinc::SDL_free;
use sdl3::sys::surface::SDL_Surface;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Which accelerometer source is active.
enum AccelSource {
    None,
    #[cfg(target_os = "macos")]
    MacosNative,
    Sdl(sdl3::sensor::Sensor),
}

const SCALE: u32 = 3;
/// Target frame time: 70224 T-cycles / cpu_clock_rate.
/// Standard: ~16.74ms (~59.73 fps). SGB1: ~16.35ms (~61.17 fps).
fn frame_duration(model: GbModel) -> Duration {
    let nanos = 70_224u64 * 1_000_000_000 / model.cpu_clock_rate() as u64;
    Duration::from_nanos(nanos)
}

const AUDIO_SAMPLE_RATE: u32 = 96_000;

#[derive(Parser)]
#[command(name = "vibeboy", about = "Game Boy / Game Boy Color emulator")]
struct Cli {
    /// Path to ROM file (.gb / .gbc). If omitted, a file dialog will open.
    rom: Option<PathBuf>,

    /// Path to boot ROM file (auto-detected if not specified)
    #[arg(long)]
    bootrom: Option<PathBuf>,

    /// Hardware model (auto-detected from cart header if not specified)
    #[arg(long, value_parser = parse_model)]
    model: Option<GbModel>,

    /// Path to SNES program ROM for SGB LLE (auto-detected if not specified)
    #[arg(long)]
    snes_rom: Option<PathBuf>,

    /// Enable SGB Low-Level Emulation (run SNES CPU with BIOS ROM)
    #[arg(long)]
    lle: bool,

    /// Skip boot ROM (start at PC=0x100 with post-boot state)
    #[arg(long)]
    no_boot: bool,

    /// Connect a Game Boy Printer (saves PNGs to prints/ directory)
    #[arg(long)]
    printer: bool,

    /// Scaling filter
    #[arg(long, default_value = "nearest", value_parser = parse_filter)]
    filter: String,
}

fn parse_model(s: &str) -> Result<GbModel, String> {
    s.parse::<GbModel>()
}

fn parse_filter(s: &str) -> Result<String, String> {
    let valid = [
        "nearest", "none", "bilinear", "bicubic", "epx", "scale2x", "scale3x", "scale4x", "eagle",
        "hq2x", "hq3x", "hq4x", "xbr2x", "xbr3x", "xbr4x",
        "xbrz2x", "xbrz3x", "xbrz4x", "xbrz5x", "xbrz6x",
        "xbr-hybrid", "super-xbr", "nedi", "dcci", "edi",
        "omniscale", "omniscale-legacy",
        "aa-nearest", "vectorize", "vectorize-adaptive",
    ];
    let lower = s.to_lowercase();
    if valid.contains(&lower.as_str()) {
        Ok(lower)
    } else {
        Err(format!("unknown filter '{}'\n  [possible values: nearest, bilinear, bicubic, epx, scale2x, scale3x, scale4x, eagle, hq2x-4x, xbr2x-4x, xbrz2x-6x, xbr-hybrid, super-xbr, nedi, dcci, edi, omniscale, omniscale-legacy, aa-nearest, vectorize, vectorize-adaptive]", s))
    }
}

/// Show an SDL3 file dialog to pick a ROM file. Exits if the user cancels.
fn pick_rom_file() -> PathBuf {
    let sdl = sdl3::init().unwrap();
    let _video = sdl.video().unwrap();
    let mut event_pump = sdl.event_pump().unwrap();

    let (tx, rx) = mpsc::channel::<Option<PathBuf>>();

    let filters = [
        DialogFileFilter {
            name: "Game Boy ROMs",
            pattern: "gb;gbc",
        },
        DialogFileFilter {
            name: "All files",
            pattern: "*",
        },
    ];

    dialog::show_open_file_dialog(
        &filters,
        None::<&str>,
        false,
        None::<&sdl3::video::Window>,
        Box::new(move |result, _filter| {
            let path = match result {
                Ok(files) if !files.is_empty() => Some(files[0].clone()),
                _ => None,
            };
            let _ = tx.send(path);
        }),
    )
    .unwrap_or_else(|e| {
        eprintln!("Failed to open file dialog: {}", e);
        std::process::exit(1);
    });

    // Pump events until the dialog callback fires
    loop {
        event_pump.pump_events();
        match rx.try_recv() {
            Ok(Some(path)) => return path,
            Ok(None) => {
                // User cancelled
                std::process::exit(0);
            }
            Err(mpsc::TryRecvError::Empty) => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                eprintln!("File dialog failed unexpectedly");
                std::process::exit(1);
            }
        }
    }
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();

    // Resolve ROM path: use CLI argument or show a file dialog
    let rom_path: PathBuf = if let Some(ref p) = cli.rom {
        p.clone()
    } else {
        pick_rom_file()
    };

    let rom = fs::read(&rom_path).unwrap_or_else(|e| {
        eprintln!("Failed to read ROM '{}': {}", rom_path.display(), e);
        std::process::exit(1);
    });

    // Resolve hardware model
    let model = cli.model.unwrap_or_else(|| {
        let cgb_flag = rom.get(0x0143).copied().unwrap_or(0);
        if cgb_flag == 0x80 || cgb_flag == 0xC0 {
            GbModel::Cgb
        } else {
            GbModel::Dmg
        }
    });

    let frame_dur = frame_duration(model);

    // Resolve boot ROM: explicit path, or auto-detect by model
    let boot_rom: Option<Vec<u8>> = if cli.no_boot {
        None
    } else if let Some(ref p) = cli.bootrom {
        Some(fs::read(p).unwrap_or_else(|e| {
            eprintln!("Failed to read boot ROM '{}': {}", p.display(), e);
            std::process::exit(1);
        }))
    } else {
        let path = match model {
            GbModel::Dmg0 => "bootroms/dmg0_boot.bin",
            GbModel::Dmg => "bootroms/dmg_boot.bin",
            GbModel::Mgb => "bootroms/mgb_boot.bin",
            GbModel::Sgb => "bootroms/sgb_boot.bin",
            GbModel::Sgb2 => "bootroms/sgb2_boot.bin",
            GbModel::Cgb0 => "bootroms/cgb0_boot.bin",
            GbModel::Cgb => "bootroms/cgb_boot.bin",
            GbModel::Agb => "bootroms/cgb_agb_boot.bin",
        };
        fs::read(path).ok()
    };

    if boot_rom.is_some() {
        eprintln!("Boot ROM loaded — executing boot sequence.");
    }

    // Load SNES program ROM for SGB LLE (only when --lle flag is set)
    let snes_rom: Option<Vec<u8>> = if model.is_sgb() && cli.lle {
        if let Some(ref p) = cli.snes_rom {
            Some(fs::read(p).unwrap_or_else(|e| {
                eprintln!("Failed to read SNES ROM '{}': {}", p.display(), e);
                std::process::exit(1);
            }))
        } else {
            // Auto-detect: try sgb1.program.rom, sgb2.program.rom, sgb.sfc, sgb2.sfc
            let candidates = match model {
                GbModel::Sgb2 => vec!["sgb2.program.rom", "sgb2.sfc"],
                GbModel::Sgb => vec!["sgb1.program.rom", "sgb.sfc"],
                _ => vec![],
            };
            candidates.iter().find_map(|name| fs::read(name).ok())
        }
    } else {
        None
    };

    if snes_rom.is_some() {
        eprintln!("SNES program ROM loaded — SGB LLE mode active.");
    }

    // Parse scaling filter
    // Filter string is already validated and lowercased by parse_filter
    let scale_filter: scaling::ScaleFilter = match cli.filter.as_str() {
        "nearest" | "none" => scaling::ScaleFilter::Nearest,
        "bilinear" => scaling::ScaleFilter::Bilinear,
        "bicubic" => scaling::ScaleFilter::Bicubic,
        "epx" => scaling::ScaleFilter::Epx,
        "scale2x" => scaling::ScaleFilter::Scale2x,
        "scale3x" => scaling::ScaleFilter::Scale3x,
        "scale4x" => scaling::ScaleFilter::Scale4x,
        "eagle" => scaling::ScaleFilter::Eagle,
        "hq2x" => scaling::ScaleFilter::Hqx(scaling::HqxScale::Hq2x),
        "hq3x" => scaling::ScaleFilter::Hqx(scaling::HqxScale::Hq3x),
        "hq4x" => scaling::ScaleFilter::Hqx(scaling::HqxScale::Hq4x),
        "xbr2x" => scaling::ScaleFilter::Xbr(scaling::XbrScale::Xbr2x),
        "xbr3x" => scaling::ScaleFilter::Xbr(scaling::XbrScale::Xbr3x),
        "xbr4x" => scaling::ScaleFilter::Xbr(scaling::XbrScale::Xbr4x),
        "xbrz2x" => scaling::ScaleFilter::Xbrz(scaling::XbrzScale::Xbrz2x),
        "xbrz3x" => scaling::ScaleFilter::Xbrz(scaling::XbrzScale::Xbrz3x),
        "xbrz4x" => scaling::ScaleFilter::Xbrz(scaling::XbrzScale::Xbrz4x),
        "xbrz5x" => scaling::ScaleFilter::Xbrz(scaling::XbrzScale::Xbrz5x),
        "xbrz6x" => scaling::ScaleFilter::Xbrz(scaling::XbrzScale::Xbrz6x),
        "xbr-hybrid" => scaling::ScaleFilter::XbrHybrid,
        "super-xbr" => scaling::ScaleFilter::SuperXbr,
        "nedi" => scaling::ScaleFilter::Nedi,
        "dcci" => scaling::ScaleFilter::Dcci,
        "edi" => scaling::ScaleFilter::Edi,
        "omniscale" => scaling::ScaleFilter::OmniScale,
        "omniscale-legacy" => scaling::ScaleFilter::OmniScaleLegacy,
        "aa-nearest" => scaling::ScaleFilter::AaNearestNeighbor,
        "vectorize" => scaling::ScaleFilter::Vectorize,
        "vectorize-adaptive" => scaling::ScaleFilter::VectorizeAdaptive,
        _ => unreachable!("filter validated by parse_filter"),
    };

    eprintln!("\nControls:");
    eprintln!("  Arrow keys  — D-pad");
    eprintln!("  Z / X       — A / B");
    eprintln!("  Enter       — Start");
    eprintln!("  Right Shift — Select");
    eprintln!("  Backspace   — Rewind");
    eprintln!("  Tab         — Fast forward (4x)");
    eprintln!("  F5 / F7     — Save / Load state");
    eprintln!("  1-9         — Select state slot");
    eprintln!("  Escape      — Quit");
    if scale_filter != scaling::ScaleFilter::Nearest {
        eprintln!("  Filter: {:?}", scale_filter);
    }
    eprintln!();

    let mut emu = Emulator::new(rom, boot_rom, Some(rom_path.as_path()), model, snes_rom);

    if cli.printer {
        let output_dir = std::path::Path::new("prints");
        emu.bus.serial.device = Box::new(printer::Printer::new(output_dir, model.cpu_clock_rate()));
        eprintln!("Game Boy Printer connected — images will be saved to prints/");
    }

    let is_sgb = emu.is_sgb();
    let (src_w, src_h): (u32, u32) = if is_sgb { (256, 224) } else { (160, 144) };
    let is_resizable = scale_filter.is_resizable();
    let scales_to_display = scale_filter.scales_to_display();
    let is_vectorize = matches!(scale_filter, scaling::ScaleFilter::Vectorize | scaling::ScaleFilter::VectorizeAdaptive);
    let mut vec_cache = match scale_filter {
        scaling::ScaleFilter::Vectorize => Some(crate::vectorize::VectorizeCache::new(false)),
        scaling::ScaleFilter::VectorizeAdaptive => Some(crate::vectorize::VectorizeCache::new(true)),
        _ => None,
    };
    let filter_factor = scale_filter.factor();
    let tex_w = src_w * filter_factor;
    let tex_h = src_h * filter_factor;
    // Resizable filters start at SCALE * src size (factor=1); fixed-factor
    // filters compute window size from their native output dimensions.
    let win_scale = if filter_factor > 1 { SCALE / filter_factor.max(1) } else { SCALE };
    let win_scale = win_scale.max(1);
    let win_w = tex_w * win_scale;
    let win_h = tex_h * win_scale;

    // ── SDL3 init ─────────────────────────────────────────────────────────────
    let sdl = sdl3::init().unwrap();
    let video = sdl.video().unwrap();
    let audio = sdl.audio().unwrap();

    // ── Video ─────────────────────────────────────────────────────────────────
    let mut window_builder = video.window("GBC Emulator", win_w, win_h);
    window_builder.position_centered();
    if is_resizable {
        window_builder.resizable();
    }
    let window = window_builder.build().unwrap();

    let mut canvas = window.into_canvas();

    let texture_creator = canvas.texture_creator();

    let mut tex_cur_w = tex_w;
    let mut tex_cur_h = tex_h;
    let mut texture = texture_creator
        .create_texture_streaming(PixelFormat::ARGB8888, tex_cur_w, tex_cur_h)
        .unwrap();
    texture.set_scale_mode(ScaleMode::Nearest);

    let mut event_pump = sdl.event_pump().unwrap();

    // ── Audio ─────────────────────────────────────────────────────────────────
    sdl3::hint::set("SDL_AUDIO_DEVICE_SAMPLE_FRAMES", "512");
    let spec = AudioSpec {
        freq:     Some(AUDIO_SAMPLE_RATE as i32),
        channels: Some(2),
        format:   Some(AudioFormat::F32LE),
    };
    let audio_device = audio.open_playback_device(&spec).unwrap();
    let audio_stream = audio.new_playback_stream(&spec, None).unwrap();
    audio_device.bind_stream(&audio_stream).unwrap();
    audio_device.resume();
    audio_stream.clear().unwrap();

    // ── Camera (webcam for Pocket Camera mapper, only if cart has camera) ──
    let camera_thread = if emu.bus.cart.has_camera() {
        CameraThread::start(&sdl)
    } else {
        None
    };
    let mut camera_buf = [0u8; 128 * 112];

    // ── Accelerometer (MBC7 / Kirby Tilt 'n' Tumble) ──
    let accel_source = if emu.bus.cart.has_accelerometer() {
        init_accel(&sdl)
    } else {
        AccelSource::None
    };

    let mut current_slot: usize = 0; // save state slot (0-indexed, shown as 1-9)

    let mut frame_start = Instant::now();
    let mut fps_timer = Instant::now();
    let mut fps_count = 0u32;
    let mut fps_emu_total = Duration::ZERO;

    'running: loop {
        // ── Events ────────────────────────────────────────────────────────────
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => break 'running,
                Event::KeyDown { keycode: Some(Keycode::F5), .. } => {
                    emu.save_state(current_slot);
                    eprintln!("State saved to slot {}", current_slot + 1);
                }
                Event::KeyDown { keycode: Some(Keycode::F7), .. } => {
                    if emu.load_state(current_slot) {
                        eprintln!("State loaded from slot {}", current_slot + 1);
                    } else {
                        eprintln!("Slot {} is empty", current_slot + 1);
                    }
                }
                Event::KeyDown { keycode: Some(k), .. } => {
                    let slot = match k {
                        Keycode::_1 => Some(0),
                        Keycode::_2 => Some(1),
                        Keycode::_3 => Some(2),
                        Keycode::_4 => Some(3),
                        Keycode::_5 => Some(4),
                        Keycode::_6 => Some(5),
                        Keycode::_7 => Some(6),
                        Keycode::_8 => Some(7),
                        Keycode::_9 => Some(8),
                        _ => None,
                    };
                    if let Some(s) = slot {
                        current_slot = s;
                        eprintln!("Slot {} selected", current_slot + 1);
                    }
                }
                _ => {}
            }
        }

        // ── Input ─────────────────────────────────────────────────────────────
        handle_input(&mut emu, &event_pump.keyboard_state());

        // ── Webcam → Pocket Camera ────────────────────────────────────────────
        if let Some(ref ct) = camera_thread {
            if ct.read_frame(&mut camera_buf) {
                emu.bus.cart.set_camera_image(&camera_buf);
            }
        }

        // ── Accelerometer → MBC7 ─────────────────────────────────────────────
        {
            const CENTER: f32 = 0x81D0 as u16 as f32; // 33232.0
            const RANGE: f32 = 0x70 as u16 as f32;   // 112.0
            let mut got = false;
            let mut gx: f32 = 0.0;
            let mut gy: f32 = 0.0;

            match accel_source {
                #[cfg(target_os = "macos")]
                AccelSource::MacosNative => {
                    if let Some((x, y, _z)) = macos_accel::poll() {
                        gx = -x; // MacBook X axis is opposite to MBC7 convention
                        gy = y;
                        got = true;
                    }
                }
                AccelSource::Sdl(ref sensor) => {
                    if let Ok(SensorData::Accel([raw_x, raw_y, _])) = sensor.get_data() {
                        gx = raw_x / 9.81_f32;
                        gy = raw_y / 9.81_f32;
                        got = true;
                    }
                }
                AccelSource::None => {}
            }

            if got {
                let mbc7_x = (CENTER + gx * RANGE).clamp(0.0, 65535.0) as u16;
                let mbc7_y = (CENTER + gy * RANGE).clamp(0.0, 65535.0) as u16;
                emu.bus.cart.set_accelerometer(mbc7_x, mbc7_y);
            }
        }

        // ── Rewind / Fast-forward ─────────────────────────────────────────────
        let ks = event_pump.keyboard_state();
        let backspace_held = ks.is_scancode_pressed(Scancode::Backspace);
        let fast_forward = ks.is_scancode_pressed(Scancode::Tab);
        drop(ks);
        emu.rewinding = backspace_held;

        if backspace_held {
            emu.rewind_one_frame();
            emu.bus.apu.drain_samples();
        } else if fast_forward {
            // Run 4 frames, discard audio from the first 3
            for _ in 0..3 {
                emu.step_frame();
                emu.bus.apu.drain_samples();
            }
            emu.step_frame();
        } else {
            // ── Emulate one frame ─────────────────────────────────────────────
            emu.step_frame();
        }

        // ── Audio ─────────────────────────────────────────────────────────────
        let samples = emu.bus.apu.drain_samples();
        if !samples.is_empty() && !fast_forward {
            // Cap audio per frame to ~1 frame worth (stereo f32 at 96000/60 ≈ 3200 floats).
            // The first frame can generate excess audio during LCD-off init; discard excess.
            let max_samples = 3200 * 2; // Allow up to ~2 frames of audio
            if samples.len() <= max_samples {
                let _ = audio_stream.put_data_f32(&samples);
            } else {
                // Push only the tail (most recent audio) to stay in sync
                let _ = audio_stream.put_data_f32(&samples[samples.len() - max_samples..]);
            }
        }

        // ── Render (skip when window is fully occluded to avoid compositor throttle) ──
        let occluded = canvas.window().window_flags()
            & sdl3::sys::video::SDL_WINDOW_OCCLUDED != sdl3::sys::video::SDL_WindowFlags(0);
        if !occluded {
            let raw_src: &[u32] = if is_sgb {
                emu.sgb_composited_frame()
            } else {
                emu.frame_buffer()
            };
            let sw = src_w as usize;
            let sh = src_h as usize;

            // For resizable windows, compute aspect-correct display area
            let (disp_w, disp_h) = if is_resizable {
                let (ww, wh) = canvas.window().size();
                let src_aspect = src_w as f64 / src_h as f64;
                let win_aspect = ww as f64 / wh as f64;
                if win_aspect > src_aspect {
                    ((wh as f64 * src_aspect) as usize, wh as usize)
                } else {
                    (ww as usize, (ww as f64 / src_aspect) as usize)
                }
            } else {
                (0, 0)
            };

            let scaled;
            let mut vec_out: (usize, usize) = (0, 0);
            let final_src: &[u32] = match scale_filter {
                scaling::ScaleFilter::Hqx(mode) => {
                    scaled = scaling::hqx::scale(raw_src, sw, sh, mode);
                    &scaled
                }
                scaling::ScaleFilter::Epx | scaling::ScaleFilter::Scale2x => {
                    scaled = scaling::epx::scale(raw_src, sw, sh);
                    &scaled
                }
                scaling::ScaleFilter::Scale3x => {
                    scaled = scaling::scale3x::scale(raw_src, sw, sh);
                    &scaled
                }
                scaling::ScaleFilter::Scale4x => {
                    scaled = scaling::epx::scale4x(raw_src, sw, sh);
                    &scaled
                }
                scaling::ScaleFilter::Eagle => {
                    scaled = scaling::eagle::scale(raw_src, sw, sh);
                    &scaled
                }
                scaling::ScaleFilter::Bilinear => {
                    scaled = scaling::bilinear::scale_to(raw_src, sw, sh, disp_w, disp_h);
                    &scaled
                }
                scaling::ScaleFilter::Bicubic => {
                    scaled = scaling::bicubic::scale_to(raw_src, sw, sh, disp_w, disp_h);
                    &scaled
                }
                scaling::ScaleFilter::Xbr(mode) => {
                    scaled = scaling::xbr::scale(raw_src, sw, sh, mode);
                    &scaled
                }
                scaling::ScaleFilter::Xbrz(mode) => {
                    scaled = scaling::xbrz::scale(raw_src, sw, sh, mode);
                    &scaled
                }
                scaling::ScaleFilter::XbrHybrid => {
                    scaled = scaling::xbr_hybrid::scale(raw_src, sw, sh);
                    &scaled
                }
                scaling::ScaleFilter::SuperXbr => {
                    scaled = scaling::super_xbr::scale(raw_src, sw, sh);
                    &scaled
                }
                scaling::ScaleFilter::Nedi => {
                    scaled = scaling::nedi::scale(raw_src, sw, sh);
                    &scaled
                }
                scaling::ScaleFilter::Dcci => {
                    scaled = scaling::dcci::scale(raw_src, sw, sh);
                    &scaled
                }
                scaling::ScaleFilter::Edi => {
                    scaled = scaling::edi::scale(raw_src, sw, sh);
                    &scaled
                }
                scaling::ScaleFilter::OmniScale => {
                    scaled = scaling::omniscale::scale_to(raw_src, sw, sh, disp_w, disp_h);
                    &scaled
                }
                scaling::ScaleFilter::OmniScaleLegacy => {
                    scaled = scaling::omniscale_legacy::scale_to(raw_src, sw, sh, disp_w, disp_h);
                    &scaled
                }
                scaling::ScaleFilter::AaNearestNeighbor => {
                    scaled = scaling::aa_nearest::scale(raw_src, sw, sh, disp_w, disp_h);
                    &scaled
                }
                scaling::ScaleFilter::Vectorize | scaling::ScaleFilter::VectorizeAdaptive => {
                    let scale = (disp_w as f64 / sw as f64).min(disp_h as f64 / sh as f64);
                    let cache = vec_cache.as_mut().unwrap();
                    let (raster, w, h) = cache.rasterize(raw_src, sw, sh, scale);
                    vec_out = (w, h);
                    raster
                }
                scaling::ScaleFilter::Nearest => raw_src,
            };

            // Output frame dimensions and display rect
            let is_vec = matches!(scale_filter, scaling::ScaleFilter::Vectorize | scaling::ScaleFilter::VectorizeAdaptive);
            let (fw, fh) = if is_vec {
                vec_out
            } else if scales_to_display {
                (disp_w, disp_h)
            } else {
                (tex_w as usize, tex_h as usize)
            };

            // Aspect-correct destination rect (letterbox/pillarbox)
            let dst_rect: Option<sdl3::render::FRect> = if is_resizable {
                let (ww, wh) = canvas.window().size();
                // For vectorize, center the actual rasterized output
                let (dw, dh) = if is_vec { vec_out } else { (disp_w, disp_h) };
                let dx = (ww as usize).saturating_sub(dw) / 2;
                let dy = (wh as usize).saturating_sub(dh) / 2;
                Some(sdl3::render::FRect::new(dx as f32, dy as f32, dw as f32, dh as f32))
            } else {
                None
            };

            // Recreate texture if dimensions changed (vectorize resize)
            if fw as u32 != tex_cur_w || fh as u32 != tex_cur_h {
                tex_cur_w = fw as u32;
                tex_cur_h = fh as u32;
                texture = texture_creator
                    .create_texture_streaming(PixelFormat::ARGB8888, tex_cur_w, tex_cur_h)
                    .unwrap();
                texture.set_scale_mode(ScaleMode::Nearest);
            }

            texture
                .with_lock(None, |pixels: &mut [u8], pitch: usize| {
                    for y in 0..fh {
                        for x in 0..fw {
                            let argb = final_src[y * fw + x];
                            let off = y * pitch + x * 4;
                            pixels[off]     =  argb        as u8; // B
                            pixels[off + 1] = (argb >>  8) as u8; // G
                            pixels[off + 2] = (argb >> 16) as u8; // R
                            pixels[off + 3] = 0xFF;                // A
                        }
                    }
                })
                .unwrap();

            canvas.clear();
            canvas.copy(&texture, None, dst_rect).unwrap();
            canvas.present();
        }

        // ── FPS counter ───────────────────────────────────────────────────────
        let emu_time = frame_start.elapsed();
        fps_count += 1;
        fps_emu_total += emu_time;
        let fps_elapsed = fps_timer.elapsed();
        if fps_elapsed >= Duration::from_secs(1) {
            let fps = fps_count as f64 / fps_elapsed.as_secs_f64();
            let avg_emu_ms = fps_emu_total.as_secs_f64() * 1000.0 / fps_count as f64;
            eprintln!("FPS: {:.1}  emu: {:.2}ms/frame", fps, avg_emu_ms);
            fps_count = 0;
            fps_emu_total = Duration::ZERO;
            fps_timer = Instant::now();
        }

        // ── Frame rate cap ────────────────────────────────────────────────────
        // Normal: cap to ~59.73 fps. Fast-forward: same wall-clock cap but we
        // ran 4 emulated frames, so effective speed is 4×.
        let remaining = frame_dur.saturating_sub(frame_start.elapsed());
        if remaining > Duration::from_millis(2) {
            std::thread::sleep(remaining - Duration::from_millis(2));
        }
        while frame_start.elapsed() < frame_dur {
            std::hint::spin_loop();
        }
        frame_start = Instant::now();
    }

    // Cleanup accelerometer
    #[cfg(target_os = "macos")]
    if matches!(accel_source, AccelSource::MacosNative) {
        macos_accel::close();
    }

    // Camera thread shuts down automatically via Drop
    drop(camera_thread);

    emu.save();
}

fn handle_input(emu: &mut Emulator, ks: &sdl3::keyboard::KeyboardState) {
    let map: &[(Scancode, u8)] = &[
        (Scancode::Z,      Emulator::BTN_B),
        (Scancode::X,      Emulator::BTN_A),
        (Scancode::Return, Emulator::BTN_START),
        (Scancode::RShift, Emulator::BTN_SELECT),
        (Scancode::Right,  Emulator::BTN_RIGHT),
        (Scancode::Left,   Emulator::BTN_LEFT),
        (Scancode::Up,     Emulator::BTN_UP),
        (Scancode::Down,   Emulator::BTN_DOWN),
    ];
    for (sc, btn) in map {
        emu.set_button(*btn, ks.is_scancode_pressed(*sc));
    }
}

// ── Webcam helpers ───────────────────────────────────────────────────────────

/// Handle to a background camera capture thread.
struct CameraThread {
    buffer: Arc<Mutex<[u8; 128 * 112]>>,
    has_new_frame: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
    _camera_subsystem: sdl3::CameraSubsystem,
}

impl CameraThread {
    /// Enumerate cameras on the main thread, spawn a background thread to capture frames.
    /// Returns None if no camera is available (cart falls back to noise generator).
    fn start(sdl: &sdl3::Sdl) -> Option<Self> {
        let cam_sys = match sdl.camera() {
            Ok(cs) => cs,
            Err(e) => {
                log::warn!("SDL camera subsystem init failed: {} — webcam disabled", e);
                return None;
            }
        };

        let device_id = unsafe {
            let mut count: std::ffi::c_int = 0;
            let ids = SDL_GetCameras(&mut count);
            if ids.is_null() || count <= 0 {
                log::info!("No cameras found — using noise generator for Pocket Camera");
                if !ids.is_null() {
                    SDL_free(ids as *mut _);
                }
                return None;
            }
            let first_id = *ids;
            SDL_free(ids as *mut _);
            first_id
        };

        let buffer: Arc<Mutex<[u8; 128 * 112]>> = Arc::new(Mutex::new([0u8; 128 * 112]));
        let has_new_frame = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));

        let buf_clone = Arc::clone(&buffer);
        let new_frame_clone = Arc::clone(&has_new_frame);
        let stop_clone = Arc::clone(&stop);

        let handle = std::thread::Builder::new()
            .name("camera".into())
            .spawn(move || {
                camera_thread_main(device_id.0, buf_clone, new_frame_clone, stop_clone);
            })
            .expect("failed to spawn camera thread");

        log::info!("Camera capture thread started");
        Some(CameraThread {
            buffer,
            has_new_frame,
            stop,
            handle: Some(handle),
            _camera_subsystem: cam_sys,
        })
    }

    /// Read the latest frame into `buf`. Returns true if a new frame was available.
    fn read_frame(&self, buf: &mut [u8; 128 * 112]) -> bool {
        if self.has_new_frame.swap(false, Ordering::Acquire) {
            let lock = self.buffer.lock().unwrap();
            buf.copy_from_slice(&*lock);
            true
        } else {
            false
        }
    }
}

impl Drop for CameraThread {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Background thread: opens the camera, captures and processes frames in a loop.
fn camera_thread_main(
    device_id: u32,
    buffer: Arc<Mutex<[u8; 128 * 112]>>,
    has_new_frame: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
) {
    use sdl3::sys::camera::SDL_CameraID;

    // Open the camera on this thread (SDL_Camera* is not Send)
    let camera = unsafe {
        let spec = SDL_CameraSpec {
            format: SysPixelFormat::RGBA32,
            colorspace: SDL_Colorspace::SRGB,
            width: 640,
            height: 480,
            framerate_numerator: 30,
            framerate_denominator: 1,
        };
        let cam = SDL_OpenCamera(SDL_CameraID(device_id), &spec);
        if cam.is_null() {
            log::warn!("Camera thread: failed to open camera");
            return;
        }
        cam
    };

    log::info!("Camera thread: webcam opened (640x480 requested)");

    while !stop.load(Ordering::Acquire) {
        let got_frame = unsafe {
            let mut ts: u64 = 0;
            let surface = SDL_AcquireCameraFrame(camera, &mut ts);
            if surface.is_null() || ts == 0 {
                false
            } else {
                process_camera_frame(surface, &buffer, &has_new_frame);
                SDL_ReleaseCameraFrame(camera, surface);
                true
            }
        };

        if !got_frame {
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    unsafe {
        SDL_CloseCamera(camera);
    }
    log::info!("Camera thread: shut down");
}

/// Process a raw SDL surface into 128×112 grayscale and write to shared buffer.
unsafe fn process_camera_frame(
    surface: *mut SDL_Surface,
    buffer: &Arc<Mutex<[u8; 128 * 112]>>,
    has_new_frame: &Arc<AtomicBool>,
) {
    let surf: &SDL_Surface = &*surface;
    let src_w = surf.w as usize;
    let src_h = surf.h as usize;
    let pitch = surf.pitch as usize;
    let pixels = surf.pixels as *const u8;

    if pixels.is_null() || src_w == 0 || src_h == 0 {
        return;
    }

    // Build RGBA image from SDL surface (may have row padding)
    let mut rgba = vec![0u8; src_w * src_h * 4];
    for y in 0..src_h {
        let src_row = pixels.add(y * pitch);
        let dst_off = y * src_w * 4;
        std::ptr::copy_nonoverlapping(src_row, rgba.as_mut_ptr().add(dst_off), src_w * 4);
    }

    let img = image::RgbaImage::from_raw(src_w as u32, src_h as u32, rgba)
        .expect("camera frame size mismatch");

    // Crop to 8:7 aspect ratio (128:112) before resizing
    let target_ratio = 128.0 / 112.0;
    let src_ratio = src_w as f64 / src_h as f64;
    let (crop_w, crop_h) = if src_ratio > target_ratio {
        let cw = (src_h as f64 * target_ratio) as u32;
        (cw, src_h as u32)
    } else {
        let ch = (src_w as f64 / target_ratio) as u32;
        (src_w as u32, ch)
    };
    let crop_x = (src_w as u32 - crop_w) / 2;
    let crop_y = (src_h as u32 - crop_h) / 2;
    let cropped =
        image::imageops::crop_imm(&img, crop_x, crop_y, crop_w, crop_h).to_image();

    // Convert to grayscale and resize with Lanczos3
    let gray = image::imageops::grayscale(&cropped);
    let resized =
        image::imageops::resize(&gray, 128, 112, image::imageops::FilterType::Lanczos3);

    // Write to shared buffer (lock held only for memcpy)
    {
        let mut lock = buffer.lock().unwrap();
        lock.copy_from_slice(resized.as_raw());
    }
    has_new_frame.store(true, Ordering::Release);
}

// ── Accelerometer helpers ─────────────────────────────────────────────────────

/// Try to open an accelerometer: prefer macOS native IOKit on Apple Silicon,
/// fall back to SDL3 sensor API.
fn init_accel(sdl: &sdl3::Sdl) -> AccelSource {
    // Try macOS native accelerometer first
    #[cfg(target_os = "macos")]
    {
        if macos_accel::init() {
            eprintln!("Accelerometer: macOS native (Apple Silicon)");
            return AccelSource::MacosNative;
        }
        log::info!("macOS native accelerometer not available, trying SDL3");
    }

    // Fall back to SDL3 sensor
    let sensor_sys = match sdl.sensor() {
        Ok(s) => s,
        Err(e) => {
            log::warn!("SDL sensor subsystem init failed: {} — accelerometer disabled", e);
            return AccelSource::None;
        }
    };
    let ids = match sensor_sys.num_sensors() {
        Ok(ids) => ids,
        Err(e) => {
            log::info!("No sensors found: {} — accelerometer disabled", e);
            return AccelSource::None;
        }
    };
    for id in ids {
        if let Ok(sensor) = sensor_sys.open(id) {
            if sensor.sensor_type() == SensorType::Accelerometer {
                eprintln!("Accelerometer: SDL3 ({})", sensor.name());
                return AccelSource::Sdl(sensor);
            }
        }
    }
    log::info!("No accelerometer found — MBC7 will use center values");
    AccelSource::None
}
