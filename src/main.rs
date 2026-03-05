mod apu;
mod bus;
mod cartridge;
mod cpu;
mod emulator;
mod joypad;
mod ppu;
mod timer;

use emulator::Emulator;
use sdl2::audio::AudioSpecDesired;
use sdl2::event::Event;
use sdl2::keyboard::{Keycode, Scancode};
use sdl2::pixels::PixelFormatEnum;
use std::env;
use std::fs;
use std::time::{Duration, Instant};

/// Display scale factor (160×144 → 480×432).
const SCALE: u32 = 3;
const WIDTH: u32 = 160 * SCALE;
const HEIGHT: u32 = 144 * SCALE;
/// Target frame time for ~59.73 fps.
const FRAME_DURATION: Duration = Duration::from_nanos(16_742_706);

const AUDIO_SAMPLE_RATE: i32 = 44_100;
/// Max queued audio bytes before we stop pushing (~100 ms of stereo f32).
const MAX_QUEUED_BYTES: u32 = (AUDIO_SAMPLE_RATE as u32 / 10) * 2 * 4;

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <rom.gb>", args[0]);
        std::process::exit(1);
    }

    let rom = fs::read(&args[1]).unwrap_or_else(|e| {
        eprintln!("Failed to read ROM '{}': {}", args[1], e);
        std::process::exit(1);
    });

    let mut emu = Emulator::new(rom);

    // ── SDL2 setup ────────────────────────────────────────────────────────────
    let sdl = sdl2::init().unwrap();
    let video = sdl.video().unwrap();
    let audio = sdl.audio().unwrap();

    // ── Video ─────────────────────────────────────────────────────────────────
    let window = video
        .window("GBC Emulator", WIDTH, HEIGHT)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().accelerated().build().unwrap();
    let texture_creator = canvas.texture_creator();

    let mut texture = texture_creator
        .create_texture_streaming(PixelFormatEnum::ARGB8888, 160, 144)
        .unwrap();

    let mut event_pump = sdl.event_pump().unwrap();

    // ── Audio ─────────────────────────────────────────────────────────────────
    let audio_spec = AudioSpecDesired {
        freq:     Some(AUDIO_SAMPLE_RATE),
        channels: Some(2),    // stereo
        samples:  Some(1024), // device buffer size (SDL internal)
    };
    let audio_queue = audio
        .open_queue::<f32, _>(None, &audio_spec)
        .unwrap();
    audio_queue.resume();

    let mut frame_start = Instant::now();

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

        // ── Audio: push samples if queue isn't too far ahead ─────────────────
        if audio_queue.size() < MAX_QUEUED_BYTES {
            let samples = emu.bus.apu.drain_samples();
            if !samples.is_empty() {
                let _ = audio_queue.queue_audio(&samples);
            }
        } else {
            // Discard this frame's samples to prevent latency buildup
            let _ = emu.bus.apu.drain_samples();
        }

        // ── Upload frame buffer to texture ────────────────────────────────────
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

        // ── Frame rate cap ────────────────────────────────────────────────────
        let elapsed = frame_start.elapsed();
        if elapsed < FRAME_DURATION {
            std::thread::sleep(FRAME_DURATION - elapsed);
        }
        frame_start = Instant::now();
    }
}

fn handle_input(emu: &mut Emulator, ks: &sdl2::keyboard::KeyboardState) {
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
