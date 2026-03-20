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
mod ui_util;
#[cfg(target_os = "macos")]
mod macos_accel;

use clap::Parser;
use emulator::Emulator;
use model::GbModel;
use sdl3::audio::{AudioFormat, AudioSpec};
use sdl3::dialog::{self, DialogFileFilter};
use sdl3::event::Event;
use sdl3::keyboard::{Keycode, Scancode};
use sdl3::sys::camera::{
    SDL_AcquireCameraFrame, SDL_CameraSpec, SDL_CloseCamera, SDL_GetCameras,
    SDL_OpenCamera, SDL_ReleaseCameraFrame,
};
use sdl3::sys::pixels::{SDL_Colorspace, SDL_PixelFormat as SysPixelFormat};
use sdl3::gamepad::{Axis as GpAxis, Button as GpButton};
use sdl3::sensor::{SensorData, SensorType};
use sdl3::sys::joystick::SDL_JoystickID;
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
use ui_util::frame_duration;

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

    /// Force CPU-only scaling (disable GPU shader pipeline even if available)
    #[arg(long)]
    cpu_filter: bool,

    /// Use YUV similarity threshold for visible edges in vectorize filters
    /// (Paper Section 3.2). Merges near-similar colors into the same region.
    /// Off by default; can produce artifacts in games with dithering/gradients.
    #[arg(long)]
    yuv_edges: bool,
}

use ui_util::{parse_model, parse_filter};

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

    if cli.yuv_edges || cli.filter.starts_with("vectorize-spline-diffusion") {
        vectorize::contour::YUV_VISIBLE_EDGES.store(true, std::sync::atomic::Ordering::Relaxed);
    }

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
    let model = cli.model.unwrap_or_else(|| ui_util::auto_detect_model(&rom));

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

    // Parse scaling filter (name already validated and lowercased by parse_filter)
    let scale_filter = scaling::ScaleFilter::from_name(&cli.filter)
        .expect("filter validated by parse_filter");

    eprintln!("\nControls:");
    eprintln!("  Arrow keys  — D-pad         Gamepad D-pad / Left stick");
    eprintln!("  Z / X       — B / A         Gamepad South / East");
    eprintln!("  Enter       — Start         Gamepad Start");
    eprintln!("  Right Shift — Select        Gamepad Back");
    eprintln!("  Backspace   — Rewind        Gamepad L Shoulder");
    eprintln!("  Tab         — Fast fwd (4x) Gamepad R Shoulder");
    eprintln!("  Minus       — Slow motion   (hold for half speed)");
    eprintln!("  Space       — Pause         (toggle)");
    eprintln!("  Period      — Frame advance  (step one frame while paused)");
    eprintln!("  F5 / F7     — Save / Load state");
    eprintln!("  F9          — Screenshot (raw + scaled)");
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
    let _scales_to_display = scale_filter.scales_to_display();
    let mut legacy_cache = match scale_filter {
        scaling::ScaleFilter::VectorizeLegacy
        | scaling::ScaleFilter::VectorizeSplineDiffusion => Some(crate::vectorize::VectorizeCache::new_legacy(false)),
        scaling::ScaleFilter::VectorizeLegacyAdaptive
        | scaling::ScaleFilter::VectorizeSplineDiffusionAdaptive => Some(crate::vectorize::VectorizeCache::new_legacy(true)),
        _ => None,
    };
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

    // ── Video + GPU ──────────────────────────────────────────────────────────
    let mut window_builder = video.window("GBC Emulator", win_w, win_h);
    window_builder.position_centered();
    if is_resizable {
        window_builder.resizable();
    }
    let window = window_builder.build().unwrap();

    // ── Renderer init ────────────────────────────────────────────────────────
    // With sdl3-gpu-shaders: use SDL3 GPU API with shader pipelines.
    // Without: use SDL 2D canvas renderer with CPU-only scaling.

    #[cfg(feature = "sdl3-gpu-shaders")]
    let mut gpu = scaling::gpu_pipelines::GpuPipelines::new(&window, src_w, src_h);

    #[cfg(not(feature = "sdl3-gpu-shaders"))]
    let mut canvas = window.into_canvas();
    #[cfg(not(feature = "sdl3-gpu-shaders"))]
    let texture_creator = canvas.texture_creator();
    #[cfg(not(feature = "sdl3-gpu-shaders"))]
    let mut sdl_texture = {
        let mut tex = texture_creator.create_texture_streaming(
            sdl3::pixels::PixelFormat::ARGB8888, tex_w, tex_h,
        ).unwrap();
        tex.set_scale_mode(sdl3::render::ScaleMode::Nearest);
        tex
    };
    #[cfg(not(feature = "sdl3-gpu-shaders"))]
    let (mut tex_cur_w, mut tex_cur_h) = (tex_w, tex_h);

    let mut event_pump = sdl.event_pump().unwrap();

    // ── Gamepad ───────────────────────────────────────────────────────────────
    let gamepad_sys = sdl.gamepad().unwrap();
    let mut gamepad = {
        let mut found = None;
        if let Ok(ids) = gamepad_sys.gamepads() {
            for id in ids {
                if let Ok(gp) = gamepad_sys.open(id) {
                    eprintln!("Gamepad connected: {}", gp.name().unwrap_or_default());
                    enable_gamepad_sensors(&gp);
                    found = Some(gp);
                    break;
                }
            }
        }
        found
    };

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
    let mut paused = false;
    let mut step_one_frame = false;

    let mut frame_start = Instant::now();
    let mut emu_time_debt = Duration::ZERO; // accumulated emulation time for vsync decoupling
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
                Event::KeyDown { keycode: Some(Keycode::F9), .. } => {
                    // Screenshot: save raw PPU output and scaled GPU output
                    let raw: &[u32] = if is_sgb { emu.sgb_composited_frame() } else { emu.frame_buffer() };
                    let sw = src_w as usize;
                    let sh = src_h as usize;

                    // Save raw frame
                    let raw_path = "screenshot_raw.png";
                    let mut rgb = vec![0u8; sw * sh * 3];
                    for (i, &c) in raw.iter().take(sw * sh).enumerate() {
                        rgb[i*3]   = ((c >> 16) & 0xff) as u8;
                        rgb[i*3+1] = ((c >> 8) & 0xff) as u8;
                        rgb[i*3+2] = (c & 0xff) as u8;
                    }
                    if image::save_buffer(raw_path, &rgb, sw as u32, sh as u32, image::ColorType::Rgb8).is_ok() {
                        eprintln!("Raw screenshot saved to {}", raw_path);
                    }

                    // Save scaled output by downloading the live GPU texture
                    #[cfg(feature = "sdl3-gpu-shaders")]
                    {
                        let ow = gpu.tex_w;
                        let oh = gpu.tex_h;
                        if ow > 0 && oh > 0 {
                            let dl_size = ow * oh * 4;
                            if let Ok(dl_buf) = gpu.device.create_transfer_buffer()
                                .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::DOWNLOAD)
                                .with_size(dl_size).build()
                            {
                                if let Ok(cmd) = gpu.device.acquire_command_buffer() {
                                    if let Ok(cp) = gpu.device.begin_copy_pass(&cmd) {
                                        unsafe {
                                            let mut src = sdl3::sys::gpu::SDL_GPUTextureRegion::default();
                                            src.texture = gpu.tex.raw();
                                            src.w = ow;
                                            src.h = oh;
                                            src.d = 1;
                                            let mut dst = sdl3::sys::gpu::SDL_GPUTextureTransferInfo::default();
                                            dst.transfer_buffer = dl_buf.raw();
                                            sdl3::sys::gpu::SDL_DownloadFromGPUTexture(cp.raw(), &src, &dst);
                                        }
                                        gpu.device.end_copy_pass(cp);
                                        if let Ok(f) = cmd.submit_and_acquire_fence(&gpu.device) {
                                            let _ = gpu.device.wait_fences(true, &[f]);
                                            let map = dl_buf.map::<u32>(&gpu.device, false);
                                            let px = map.mem();
                                            let mut rgb2 = vec![0u8; (ow * oh) as usize * 3];
                                            for (i, &c) in px.iter().take((ow * oh) as usize).enumerate() {
                                                rgb2[i*3]   = ((c >> 16) & 0xff) as u8;
                                                rgb2[i*3+1] = ((c >> 8) & 0xff) as u8;
                                                rgb2[i*3+2] = (c & 0xff) as u8;
                                            }
                                            drop(map);
                                            let scaled_path = "screenshot_scaled.png";
                                            if image::save_buffer(scaled_path, &rgb2, ow, oh, image::ColorType::Rgb8).is_ok() {
                                                eprintln!("Scaled screenshot saved to {} ({}x{})", scaled_path, ow, oh);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Event::KeyDown { keycode: Some(Keycode::F7), .. } => {
                    if emu.load_state(current_slot) {
                        eprintln!("State loaded from slot {}", current_slot + 1);
                    } else {
                        eprintln!("Slot {} is empty", current_slot + 1);
                    }
                }
                Event::KeyDown { keycode: Some(Keycode::Space), .. } => {
                    paused = !paused;
                    if paused {
                        eprintln!("Paused");
                    } else {
                        eprintln!("Resumed");
                        emu_time_debt = Duration::ZERO;
                    }
                }
                Event::KeyDown { keycode: Some(Keycode::Period), .. } => {
                    if paused {
                        step_one_frame = true;
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
                Event::ControllerDeviceAdded { which, .. } => {
                    if gamepad.is_none() {
                        if let Ok(gp) = gamepad_sys.open(SDL_JoystickID(which)) {
                            eprintln!("Gamepad connected: {}", gp.name().unwrap_or_default());
                            enable_gamepad_sensors(&gp);
                            gamepad = Some(gp);
                        }
                    }
                }
                Event::ControllerDeviceRemoved { which, .. } => {
                    if gamepad.as_ref().is_some_and(|g| g.id().ok() == Some(SDL_JoystickID(which))) {
                        eprintln!("Gamepad disconnected");
                        gamepad = None;
                        // Try to pick up another connected gamepad
                        if let Ok(ids) = gamepad_sys.gamepads() {
                            for id in ids {
                                if let Ok(gp) = gamepad_sys.open(id) {
                                    eprintln!("Gamepad connected: {}", gp.name().unwrap_or_default());
                                    enable_gamepad_sensors(&gp);
                                    gamepad = Some(gp);
                                    break;
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // ── Input ─────────────────────────────────────────────────────────────
        handle_input(&mut emu, &event_pump.keyboard_state(), gamepad.as_ref());

        // ── Webcam → Pocket Camera ────────────────────────────────────────────
        if let Some(ref ct) = camera_thread {
            if ct.read_frame(&mut camera_buf) {
                emu.bus.cart.set_camera_image(&camera_buf);
            }
        }

        // ── Accelerometer → MBC7 ─────────────────────────────────────────────
        {
            const CENTER: f32 = 0x81D0 as f32;
            const RANGE: f32 = 0x70 as f32;
            let mut got = false;
            let mut gx: f32 = 0.0;
            let mut gy: f32 = 0.0;

            // Gamepad accelerometer takes priority (e.g. DualSense, Switch Pro)
            if let Some(ref gp) = gamepad {
                let mut data = [0.0f32; 3];
                if gp.sensor_get_data(SensorType::Accelerometer, &mut data).is_ok() {
                    gx = data[0] / 9.81_f32;
                    gy = data[1] / 9.81_f32;
                    got = true;
                }
            }

            // Fall back to device accelerometer (MacBook, SDL sensor)
            if !got {
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
            }

            if got {
                let mbc7_x = (CENTER + gx * RANGE).clamp(0.0, 65535.0) as u16;
                let mbc7_y = (CENTER + gy * RANGE).clamp(0.0, 65535.0) as u16;
                emu.bus.cart.set_accelerometer(mbc7_x, mbc7_y);
            }
        }

        // ── Rewind / Fast-forward / Slow-motion ─────────────────────────────
        let ks = event_pump.keyboard_state();
        let mut backspace_held = ks.is_scancode_pressed(Scancode::Backspace);
        let mut fast_forward = ks.is_scancode_pressed(Scancode::Tab);
        let slow_motion = ks.is_scancode_pressed(Scancode::Minus);
        drop(ks);
        // Left shoulder = rewind, right shoulder = fast forward
        if let Some(ref gp) = gamepad {
            if gp.button(GpButton::LeftShoulder) {
                backspace_held = true;
            }
            if gp.button(GpButton::RightShoulder) {
                fast_forward = true;
            }
        }
        emu.rewinding = backspace_held;

        // Accumulate elapsed wall time and run enough emulation frames to keep up.
        // This decouples emulation speed from the display refresh rate — on a 30Hz
        // monitor with vsync, we run 2 frames per loop iteration.
        let elapsed = frame_start.elapsed();
        frame_start = Instant::now();
        emu_time_debt += elapsed;
        // Cap catchup to avoid spiral of death (e.g. window was being dragged)
        let max_debt = frame_dur * 4;
        if emu_time_debt > max_debt { emu_time_debt = max_debt; }

        let emu_start = Instant::now();
        let mut frames_stepped: u32 = 0;
        if paused && !step_one_frame {
            // Consume time but don't step emulation
            emu_time_debt = Duration::ZERO;
        } else if step_one_frame {
            emu.step_frame();
            frames_stepped = 1;
            step_one_frame = false;
            emu_time_debt = Duration::ZERO;
        } else if backspace_held {
            emu.rewind_one_frame();
            emu.bus.apu.drain_samples();
            frames_stepped = 1;
            emu_time_debt = Duration::ZERO;
        } else if fast_forward {
            // Run 4 frames per wall-clock frame
            for _ in 0..3 {
                emu.step_frame();
                emu.bus.apu.drain_samples();
            }
            emu.step_frame();
            frames_stepped = 4;
            emu_time_debt = Duration::ZERO;
        } else if slow_motion {
            // Half speed: double the effective frame duration
            let slow_dur = frame_dur * 2;
            while emu_time_debt >= slow_dur {
                emu.step_frame();
                frames_stepped += 1;
                emu_time_debt -= slow_dur;
            }
        } else {
            while emu_time_debt >= frame_dur {
                emu.step_frame();
                frames_stepped += 1;
                emu_time_debt -= frame_dur;
            }
        }
        let emu_elapsed = emu_start.elapsed();

        // ── Audio ─────────────────────────────────────────────────────────────
        let samples = emu.bus.apu.drain_samples();
        if !samples.is_empty() && !fast_forward {
            // If too much audio is queued, clear it to reduce latency.
            // Target ~2 frames of buffer (stereo f32 at 96kHz/60fps ≈ 12800 bytes).
            let max_queued_bytes: i32 = 3200 * 2 * 4 * 2; // ~2 frames * stereo * sizeof(f32)
            if let Ok(queued) = audio_stream.queued_bytes() {
                if queued > max_queued_bytes {
                    let _ = audio_stream.clear();
                }
            }

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

        // ── Render ────────────────────────────────────────────────────────────
        #[cfg(feature = "sdl3-gpu-shaders")]
        let occluded = window.window_flags()
            & sdl3::sys::video::SDL_WINDOW_OCCLUDED != sdl3::sys::video::SDL_WindowFlags(0);
        #[cfg(not(feature = "sdl3-gpu-shaders"))]
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

            #[cfg(feature = "sdl3-gpu-shaders")]
            {
                use scaling::gpu_pipelines::GpuRenderMode;
                let mode = gpu.ensure_pipeline(scale_filter, &window, cli.cpu_filter);

                match mode {
                    GpuRenderMode::SplineDiffusion => {
                        let (disp_w, disp_h) = display_size(&window, src_w, src_h);
                        let scale = compute_integer_scale(disp_w, disp_h, sw, sh);
                        let (paths, bg_color) = legacy_cache.as_mut().unwrap().get_paths(raw_src, sw, sh);
                        let (gpu_edges, row_ranges, edge_indices, out_w, out_h) =
                            vectorize::rasterize::prepare_gpu_edges_v2(paths, bg_color, scale as f64, sw, sh);
                        if out_w > 0 && out_h > 0 && !gpu_edges.is_empty() {
                            gpu.render_spline_diffusion_to_window(
                                &window, &gpu_edges, &row_ranges, &edge_indices,
                                raw_src, out_w, out_h, sw as u32, sh as u32,
                                bg_color, scale as u32,
                            );
                        }
                    }
                    GpuRenderMode::Diffusion => {
                        let (disp_w, disp_h) = display_size(&window, src_w, src_h);
                        let scale = compute_integer_scale(disp_w, disp_h, sw, sh);
                        let (src_pixels, src_regions, diag_states, out_w, out_h) =
                            scaling::gpu::prepare_diffusion_data(raw_src, sw, sh, scale);
                        if out_w > 0 && out_h > 0 {
                            gpu.render_diffusion_to_window(
                                &window, &src_pixels, &src_regions, &diag_states,
                                sw as u32, sh as u32, out_w, out_h, scale as f32,
                            );
                        }
                    }
                    GpuRenderMode::Vectorize => {
                        let (disp_w, disp_h) = display_size(&window, src_w, src_h);
                        let scale = (disp_w as f64 / sw as f64).min(disp_h as f64 / sh as f64);
                        let (paths, bg_color) = legacy_cache.as_mut().unwrap().get_paths(raw_src, sw, sh);
                        let (gpu_edges, row_ranges, edge_indices, out_w, out_h) =
                            vectorize::rasterize::prepare_gpu_edges_v2(paths, bg_color, scale, sw, sh);
                        if out_w > 0 && out_h > 0 && !gpu_edges.is_empty() {
                            gpu.render_vectorize_to_window(
                                &window, &gpu_edges, &row_ranges, &edge_indices,
                                out_w, out_h, bg_color,
                            );
                        }
                    }
                    GpuRenderMode::VectorizeSharedChain => {
                        let (disp_w, disp_h) = display_size(&window, src_w, src_h);
                        let scale = (disp_w as f64 / sw as f64).min(disp_h as f64 / sh as f64);
                        let cache = vec_cache.as_mut().unwrap();
                        let (paths, bg_color) = cache.get_paths(raw_src, sw, sh);
                        let (gpu_edges, row_ranges, edge_indices, out_w, out_h) =
                            vectorize::rasterize::prepare_gpu_edges_v2(paths, bg_color, scale, sw, sh);
                        if out_w > 0 && out_h > 0 && !gpu_edges.is_empty() {
                            gpu.render_vectorize_to_window(
                                &window, &gpu_edges, &row_ranges, &edge_indices,
                                out_w, out_h, bg_color,
                            );
                        }
                    }
                    GpuRenderMode::FullGpuVectorize => {
                        let (disp_w, disp_h) = display_size(&window, src_w, src_h);
                        let scale = (disp_w as f64 / sw as f64).min(disp_h as f64 / sh as f64);
                        let out_w = (sw as f64 * scale).round() as u32;
                        let out_h = (sh as f64 * scale).round() as u32;
                        gpu.render_full_vectorize_to_window(
                            &window, raw_src, sw as u32, sh as u32,
                            out_w, out_h, scale as f32,
                        );
                    }
                    GpuRenderMode::ScaleCompute => {
                        let (disp_w, disp_h) = display_size(&window, src_w, src_h);
                        gpu.render_scale_compute(
                            scale_filter, &window, raw_src,
                            src_w, src_h, disp_w as u32, disp_h as u32,
                        );
                    }
                    GpuRenderMode::Native => {
                        gpu.render_blit(raw_src, src_w, src_h, &window, sdl3::gpu::Filter::Nearest);
                    }
                    GpuRenderMode::Cpu => {
                        let (disp_w, disp_h) = display_size(&window, src_w, src_h);
                        let (scaled, fw, fh) = cpu_scale_frame(
                            &scale_filter, raw_src, sw, sh, disp_w, disp_h, &mut legacy_cache, &mut vec_cache,
                        );
                        gpu.upload_and_blit(&scaled, fw, fh, &window);
                    }
                }
            }

            #[cfg(not(feature = "sdl3-gpu-shaders"))]
            {
                // CPU-only path: scale everything on CPU, render via SDL 2D canvas
                let (ww, wh) = canvas.window().size();
                let src_aspect = src_w as f64 / src_h as f64;
                let win_aspect = ww as f64 / wh as f64;
                let (disp_w, disp_h) = if is_resizable {
                    if win_aspect > src_aspect {
                        ((wh as f64 * src_aspect) as usize, wh as usize)
                    } else {
                        (ww as usize, (ww as f64 / src_aspect) as usize)
                    }
                } else { (0, 0) };

                let (scaled, fw, fh) = cpu_scale_frame(
                    &scale_filter, raw_src, sw, sh, disp_w, disp_h, &mut legacy_cache, &mut vec_cache,
                );
                let (fw, fh) = (fw as usize, fh as usize);
                let final_src = if fw == sw && fh == sh { raw_src } else { &scaled };

                if fw as u32 != tex_cur_w || fh as u32 != tex_cur_h {
                    tex_cur_w = fw as u32; tex_cur_h = fh as u32;
                    sdl_texture = texture_creator.create_texture_streaming(
                        sdl3::pixels::PixelFormat::ARGB8888, tex_cur_w, tex_cur_h,
                    ).unwrap();
                    sdl_texture.set_scale_mode(sdl3::render::ScaleMode::Nearest);
                }
                sdl_texture.with_lock(None, |pixels: &mut [u8], pitch: usize| {
                    for y in 0..fh {
                        for x in 0..fw {
                            let argb = final_src[y * fw + x];
                            let off = y * pitch + x * 4;
                            pixels[off]     =  argb        as u8;
                            pixels[off + 1] = (argb >>  8) as u8;
                            pixels[off + 2] = (argb >> 16) as u8;
                            pixels[off + 3] = 0xFF;
                        }
                    }
                }).unwrap();

                let dst_rect: Option<sdl3::render::FRect> = if is_resizable {
                    let dx = (ww as usize).saturating_sub(fw) / 2;
                    let dy = (wh as usize).saturating_sub(fh) / 2;
                    Some(sdl3::render::FRect::new(dx as f32, dy as f32, fw as f32, fh as f32))
                } else { None };
                canvas.clear();
                canvas.copy(&sdl_texture, None, dst_rect).unwrap();
                canvas.present();
            }
        }

        // ── FPS counter ───────────────────────────────────────────────────────
        fps_count += frames_stepped;
        if frames_stepped > 0 {
            fps_emu_total += emu_elapsed;
        }
        let fps_elapsed = fps_timer.elapsed();
        if fps_elapsed >= Duration::from_secs(1) {
            let fps = fps_count as f64 / fps_elapsed.as_secs_f64();
            let avg_emu_ms = fps_emu_total.as_secs_f64() * 1000.0 / fps_count as f64;
            eprintln!("FPS: {:.1}  emu: {:.2}ms/frame", fps, avg_emu_ms);
            fps_count = 0;
            fps_emu_total = Duration::ZERO;
            fps_timer = Instant::now();
        }

        // No manual frame cap — vsync handles pacing, and the time accumulator
        // above ensures emulation runs at the correct speed regardless of
        // display refresh rate.
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

fn handle_input(emu: &mut Emulator, ks: &sdl3::keyboard::KeyboardState, gp: Option<&sdl3::gamepad::Gamepad>) {
    const STICK_DEADZONE: i16 = 8000;

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

    if let Some(gp) = gp {
        // Gamepad buttons (OR'd with keyboard — either source can press)
        let gp_map: &[(GpButton, u8)] = &[
            (GpButton::East,      Emulator::BTN_A),
            (GpButton::South,     Emulator::BTN_B),
            (GpButton::Start,     Emulator::BTN_START),
            (GpButton::Back,      Emulator::BTN_SELECT),
            (GpButton::DPadRight, Emulator::BTN_RIGHT),
            (GpButton::DPadLeft,  Emulator::BTN_LEFT),
            (GpButton::DPadUp,    Emulator::BTN_UP),
            (GpButton::DPadDown,  Emulator::BTN_DOWN),
        ];

        // Left analog stick → d-pad
        let lx = gp.axis(GpAxis::LeftX);
        let ly = gp.axis(GpAxis::LeftY);

        for (sc, btn) in map {
            let kb = ks.is_scancode_pressed(*sc);
            let gp_btn = gp_map.iter().find(|(_, b)| b == btn).map_or(false, |(gb, _)| gp.button(*gb));
            let stick = match *btn {
                b if b == Emulator::BTN_RIGHT => lx > STICK_DEADZONE,
                b if b == Emulator::BTN_LEFT  => lx < -STICK_DEADZONE,
                b if b == Emulator::BTN_DOWN  => ly > STICK_DEADZONE,
                b if b == Emulator::BTN_UP    => ly < -STICK_DEADZONE,
                _ => false,
            };
            emu.set_button(*btn, kb || gp_btn || stick);
        }
    } else {
        for (sc, btn) in map {
            emu.set_button(*btn, ks.is_scancode_pressed(*sc));
        }
    }
}

// ── Display geometry helpers ─────────────────────────────────────────────────

/// Compute aspect-correct display dimensions for the current window.
fn display_size(window: &sdl3::video::Window, src_w: u32, src_h: u32) -> (usize, usize) {
    let (ww, wh) = window.size();
    let src_aspect = src_w as f64 / src_h as f64;
    let win_aspect = ww as f64 / wh as f64;
    if win_aspect > src_aspect {
        ((wh as f64 * src_aspect) as usize, wh as usize)
    } else {
        (ww as usize, (ww as f64 / src_aspect) as usize)
    }
}

/// Compute rounded integer scale factor from display size and source size.
fn compute_integer_scale(disp_w: usize, disp_h: usize, sw: usize, sh: usize) -> usize {
    let scale_f = (disp_w as f64 / sw as f64).min(disp_h as f64 / sh as f64);
    scale_f.round().max(1.0) as usize
}

// ── CPU scaling helper ───────────────────────────────────────────────────────

/// Run a CPU scaling filter and return (pixels, width, height).
fn cpu_scale_frame(
    filter: &scaling::ScaleFilter,
    src: &[u32], sw: usize, sh: usize,
    disp_w: usize, disp_h: usize,
    legacy_cache: &mut Option<crate::vectorize::VectorizeCache>,
    vec_cache: &mut Option<crate::vectorize::VectorizeCache>,
) -> (Vec<u32>, u32, u32) {
    // Legacy vectorize uses its own cache-based path
    if matches!(filter, scaling::ScaleFilter::VectorizeLegacy | scaling::ScaleFilter::VectorizeLegacyAdaptive) {
        let scale = (disp_w as f64 / sw as f64).min(disp_h as f64 / sh as f64);
        let cache = legacy_cache.as_mut().unwrap();
        let (raster, w, h) = cache.rasterize(src, sw, sh, scale);
        return (raster.to_vec(), w as u32, h as u32);
    }
    // Diffusion rasterizer works directly from pixels (no vector paths)
    if matches!(filter, scaling::ScaleFilter::VectorizeDiffusion) {
        let scale_f = (disp_w as f64 / sw as f64).min(disp_h as f64 / sh as f64);
        let scale = scale_f.round().max(1.0) as usize;
        let (raster, w, h) = crate::vectorize::rasterize::rasterize_diffusion(src, sw, sh, scale);
        return (raster, w as u32, h as u32);
    }
    // Shared-chain vectorization: gap-free rendering using shared boundary chains
    if matches!(filter, scaling::ScaleFilter::Vectorize | scaling::ScaleFilter::VectorizeAdaptive) {
        let scale = (disp_w as f64 / sw as f64).min(disp_h as f64 / sh as f64);
        let cache = vec_cache.as_mut().unwrap();
        let (raster, w, h) = cache.rasterize(src, sw, sh, scale);
        return (raster.to_vec(), w as u32, h as u32);
    }
    // Spline-diffusion: vectorize for paths, then Gaussian diffusion with spline boundaries
    if matches!(filter, scaling::ScaleFilter::VectorizeSplineDiffusion | scaling::ScaleFilter::VectorizeSplineDiffusionAdaptive) {
        let scale_f = (disp_w as f64 / sw as f64).min(disp_h as f64 / sh as f64);
        let scale = scale_f.round().max(1.0) as usize;
        let cache = legacy_cache.as_mut().unwrap();
        let (paths, bg_color) = cache.get_paths(src, sw, sh);
        let (raster, w, h) = crate::vectorize::rasterize::rasterize_spline_diffusion(
            paths, src, sw, sh, bg_color, scale,
        );
        return (raster, w as u32, h as u32);
    }
    // All other CPU filters use the shared dispatcher
    scaling::cpu_scale(*filter, src, sw, sh, disp_w, disp_h)
        .unwrap_or_else(|| (src.to_vec(), sw as u32, sh as u32))
}


// ── Gamepad helpers ──────────────────────────────────────────────────────────

/// Open a gamepad and enable its accelerometer if present. Logs the result.
fn enable_gamepad_sensors(gp: &sdl3::gamepad::Gamepad) {
    if unsafe { gp.has_sensor(SensorType::Accelerometer) } {
        if gp.sensor_set_enabled(SensorType::Accelerometer, true).is_ok() {
            eprintln!("  Accelerometer enabled");
        }
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
