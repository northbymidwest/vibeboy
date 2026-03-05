mod apu;
mod bus;
mod cartridge;
mod cpu;
mod emulator;
mod joypad;
mod model;
mod ppu;
mod timer;

use emulator::Emulator;
use model::GbModel;
use sdl3::audio::{AudioFormat, AudioSpec};
use sdl3::event::Event;
use sdl3::keyboard::{Keycode, Scancode};
use sdl3::pixels::PixelFormat;
use std::env;
use std::fs;
use std::time::{Duration, Instant};

const SCALE: u32 = 3;
const WIDTH: u32 = 160 * SCALE;
const HEIGHT: u32 = 144 * SCALE;
/// Target frame time for ~59.73 fps.
const FRAME_DURATION: Duration = Duration::from_nanos(16_742_706);

const AUDIO_SAMPLE_RATE: u32 = 44_100;
/// Max queued audio bytes before we stop pushing (~100 ms of stereo f32).
const MAX_QUEUED_BYTES: u32 = (AUDIO_SAMPLE_RATE / 10) * 2 * 4;

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <rom.gb> [--bootrom <bootrom.bin>]", args[0]);
        std::process::exit(1);
    }

    let rom = fs::read(&args[1]).unwrap_or_else(|e| {
        eprintln!("Failed to read ROM '{}': {}", args[1], e);
        std::process::exit(1);
    });

    // Optional boot ROM: --bootrom <path>, or auto-detect gbc_bios.bin
    let boot_rom: Option<Vec<u8>> = {
        let mut path = None;
        let mut i = 2;
        while i < args.len() {
            if args[i] == "--bootrom" && i + 1 < args.len() {
                path = Some(args[i + 1].clone());
                i += 2;
            } else {
                i += 1;
            }
        }
        if let Some(p) = path {
            Some(fs::read(&p).unwrap_or_else(|e| {
                eprintln!("Failed to read boot ROM '{}': {}", p, e);
                std::process::exit(1);
            }))
        } else {
            // Auto-detect boot ROM in crate root
            fs::read("gbc_bios.bin").ok()
        }
    };

    if boot_rom.is_some() {
        eprintln!("Boot ROM loaded — executing boot sequence.");
    }

    // We emulate CGB hardware — DMG games run in CGB compatibility mode
    let rom_path = std::path::Path::new(&args[1]);
    let mut emu = Emulator::new(rom, boot_rom, Some(rom_path), GbModel::Cgb);

    // ── SDL3 init ─────────────────────────────────────────────────────────────
    let sdl = sdl3::init().unwrap();
    let video = sdl.video().unwrap();
    let audio = sdl.audio().unwrap();

    // ── Video ─────────────────────────────────────────────────────────────────
    let window = video
        .window("GBC Emulator", WIDTH, HEIGHT)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas();
    let texture_creator = canvas.texture_creator();

    let mut texture = texture_creator
        .create_texture_streaming(PixelFormat::ARGB8888, 160, 144)
        .unwrap();

    let mut event_pump = sdl.event_pump().unwrap();

    // ── Audio ─────────────────────────────────────────────────────────────────
    let spec = AudioSpec {
        freq:     Some(AUDIO_SAMPLE_RATE as i32),
        channels: Some(2),
        format:   Some(AudioFormat::F32LE),
    };
    let audio_device = audio.open_playback_device(&spec).unwrap();
    let audio_stream = audio_device.open_device_stream(Some(&spec)).unwrap();
    audio_stream.resume().unwrap();

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
            if audio_stream.queued_bytes().unwrap_or(0) < MAX_QUEUED_BYTES as i32 {
                let _ = audio_stream.put_data_f32(&samples);
            }
        }

        // ── Render ────────────────────────────────────────────────────────────
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
