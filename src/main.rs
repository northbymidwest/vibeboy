mod apu;
mod bus;
mod cartridge;
mod cpu;
mod emulator;
mod joypad;
mod model;
mod ppu;
mod sgb;
mod snes;
mod timer;

use clap::Parser;
use emulator::Emulator;
use model::GbModel;
use sdl3::audio::{AudioFormat, AudioSpec};
use sdl3::event::Event;
use sdl3::keyboard::{Keycode, Scancode};
use sdl3::pixels::PixelFormat;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const SCALE: u32 = 3;
/// Target frame time for ~59.73 fps.
const FRAME_DURATION: Duration = Duration::from_nanos(16_742_706);

const AUDIO_SAMPLE_RATE: u32 = 44_100;

#[derive(Parser)]
#[command(name = "gbcemu", about = "Game Boy / Game Boy Color emulator")]
struct Cli {
    /// Path to ROM file (.gb / .gbc)
    rom: PathBuf,

    /// Path to boot ROM file (auto-detected if not specified)
    #[arg(long)]
    bootrom: Option<PathBuf>,

    /// Hardware model: auto, dmg0, dmg, mgb, sgb, sgb2, cgb/gbc
    #[arg(long, default_value = "auto")]
    model: String,

    /// Path to SNES program ROM for SGB LLE (auto-detected if not specified)
    #[arg(long)]
    snes_rom: Option<PathBuf>,

    /// Enable SGB Low-Level Emulation (run SNES CPU with BIOS ROM)
    #[arg(long)]
    lle: bool,

    /// Skip boot ROM (start at PC=0x100 with post-boot state)
    #[arg(long)]
    no_bootrom: bool,
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();

    let rom = fs::read(&cli.rom).unwrap_or_else(|e| {
        eprintln!("Failed to read ROM '{}': {}", cli.rom.display(), e);
        std::process::exit(1);
    });

    // Resolve hardware model
    let model = if cli.model == "auto" {
        GbModel::Cgb
    } else {
        cli.model.parse::<GbModel>().unwrap_or_else(|e| {
            eprintln!("{}", e);
            std::process::exit(1);
        })
    };

    // Resolve boot ROM: explicit path, or auto-detect by model
    let boot_rom: Option<Vec<u8>> = if cli.no_bootrom {
        None
    } else if let Some(ref p) = cli.bootrom {
        Some(fs::read(p).unwrap_or_else(|e| {
            eprintln!("Failed to read boot ROM '{}': {}", p.display(), e);
            std::process::exit(1);
        }))
    } else {
        let candidates: &[&str] = match model {
            GbModel::Dmg0 => &["dmg0_boot.bin", "bootroms/dmg0_boot.bin", "gb_bios.bin"],
            GbModel::Dmg => &["dmg_boot.bin", "bootroms/dmg_boot.bin", "gb_bios.bin"],
            GbModel::Mgb => &["mgb_boot.bin", "bootroms/mgb_boot.bin", "gb_bios.bin"],
            GbModel::Sgb => &["sgb_boot.bin", "bootroms/sgb_boot.bin", "sgb_bios.bin"],
            GbModel::Sgb2 => &["sgb2_boot.bin", "bootroms/sgb2_boot.bin", "sgb2_bios.bin"],
            GbModel::Cgb => &["cgb_boot.bin", "bootroms/cgb_boot.bin", "gbc_bios.bin"],
        };
        candidates.iter().find_map(|name| fs::read(name).ok())
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
                _ => vec!["sgb1.program.rom", "sgb.sfc"],
            };
            candidates.iter().find_map(|name| fs::read(name).ok())
        }
    } else {
        None
    };

    if snes_rom.is_some() {
        eprintln!("SNES program ROM loaded — SGB LLE mode active.");
    }

    let mut emu = Emulator::new(rom, boot_rom, Some(cli.rom.as_path()), model, snes_rom);

    let is_sgb = emu.is_sgb();
    let (tex_w, tex_h): (u32, u32) = if is_sgb { (256, 224) } else { (160, 144) };
    let win_w = tex_w * SCALE;
    let win_h = tex_h * SCALE;

    // ── SDL3 init ─────────────────────────────────────────────────────────────
    let sdl = sdl3::init().unwrap();
    let video = sdl.video().unwrap();
    let audio = sdl.audio().unwrap();

    // ── Video ─────────────────────────────────────────────────────────────────
    let window = video
        .window("GBC Emulator", win_w, win_h)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas();
    let texture_creator = canvas.texture_creator();

    let mut texture = texture_creator
        .create_texture_streaming(PixelFormat::ARGB8888, tex_w, tex_h)
        .unwrap();

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
                _ => {}
            }
        }

        // ── Input ─────────────────────────────────────────────────────────────
        handle_input(&mut emu, &event_pump.keyboard_state());

        // ── Emulate one frame ─────────────────────────────────────────────────
        emu.step_frame();

        // ── Audio ─────────────────────────────────────────────────────────────
        let samples = emu.bus.apu.drain_samples();
        if !samples.is_empty() {
            // Cap audio per frame to ~1 frame worth (stereo f32 at 44100/60 ≈ 1478 floats).
            // The first frame can generate 0.78s of audio during LCD-off init; discard excess.
            let max_samples = 1478 * 2; // Allow up to ~2 frames of audio
            if samples.len() <= max_samples {
                let _ = audio_stream.put_data_f32(&samples);
            } else {
                // Push only the tail (most recent audio) to stay in sync
                let _ = audio_stream.put_data_f32(&samples[samples.len() - max_samples..]);
            }
        }

        // ── Render ────────────────────────────────────────────────────────────
        if is_sgb {
            let src = emu.sgb_composited_frame();
            texture
                .with_lock(None, |pixels: &mut [u8], pitch: usize| {
                    for y in 0..224usize {
                        for x in 0..256usize {
                            let argb = src[y * 256 + x];
                            let off = y * pitch + x * 4;
                            pixels[off]     =  argb        as u8; // B
                            pixels[off + 1] = (argb >>  8) as u8; // G
                            pixels[off + 2] = (argb >> 16) as u8; // R
                            pixels[off + 3] = 0xFF;                // A
                        }
                    }
                })
                .unwrap();
        } else {
            let src = emu.frame_buffer();
            texture
                .with_lock(None, |pixels: &mut [u8], pitch: usize| {
                    for y in 0..144usize {
                        for x in 0..160usize {
                            let argb = src[y * 160 + x];
                            let off = y * pitch + x * 4;
                            pixels[off]     =  argb        as u8; // B
                            pixels[off + 1] = (argb >>  8) as u8; // G
                            pixels[off + 2] = (argb >> 16) as u8; // R
                            pixels[off + 3] = 0xFF;                // A
                        }
                    }
                })
                .unwrap();
        }

        canvas.clear();
        canvas.copy(&texture, None, None).unwrap();
        canvas.present();

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
        // Sleep for the bulk, then spin-wait for precision (thread::sleep overshoots on macOS)
        let remaining = FRAME_DURATION.saturating_sub(frame_start.elapsed());
        if remaining > Duration::from_millis(2) {
            std::thread::sleep(remaining - Duration::from_millis(2));
        }
        while frame_start.elapsed() < FRAME_DURATION {
            std::hint::spin_loop();
        }
        frame_start = Instant::now();
    }

    emu.save();
}

fn handle_input(emu: &mut Emulator, ks: &sdl3::keyboard::KeyboardState) {
    let map: &[(Scancode, u8)] = &[
        (Scancode::Z,      Emulator::BTN_A),
        (Scancode::X,      Emulator::BTN_B),
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
