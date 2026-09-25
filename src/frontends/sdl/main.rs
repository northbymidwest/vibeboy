use vibeboy_core::*;

mod accel;
mod camera;
mod input;
mod render;

use clap::Parser;
use emulator::Emulator;
use model::GbModel;
use sdl3::audio::{AudioFormat, AudioSpec};
use sdl3::dialog::{self, DialogFileFilter};
use sdl3::event::Event;
use sdl3::gamepad::{Axis as GpAxis, Button as GpButton};
use sdl3::keyboard::{Keycode, Scancode};
use sdl3::sensor::{SensorData, SensorType};
use sdl3::sys::camera::{
    SDL_AcquireCameraFrame, SDL_CameraSpec, SDL_CloseCamera, SDL_GetCameras, SDL_OpenCamera,
    SDL_ReleaseCameraFrame,
};
use sdl3::sys::pixels::{SDL_Colorspace, SDL_PixelFormat as SysPixelFormat};
use sdl3::sys::stdinc::SDL_free;
use sdl3::sys::surface::SDL_Surface;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use accel::{enable_gamepad_sensors, init_accel};
use camera::CameraThread;
use input::handle_input;
use render::{cpu_scale_frame, display_size};

use ui_util::{HoldInputs, Session, SessionConfig};
use util::parse_model;

/// Which accelerometer source is active.
enum AccelSource {
    None,
    #[cfg(target_os = "macos")]
    MacosNative,
    Sdl(sdl3::sensor::Sensor),
}

const SCALE: u32 = 3;

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
    #[arg(long, default_value = "nearest", value_parser = scaling::ScaleFilter::all_names())]
    filter: String,

    /// Force CPU-only scaling (disable GPU shader pipeline even if available)
    #[arg(long)]
    cpu_filter: bool,

    /// Run-ahead frames to reduce input latency. Speculatively runs N frames
    /// ahead and displays the result, so input is reflected sooner.
    /// Costs ~N× CPU per frame. Typical values: 1-2.
    #[arg(long, default_missing_value = "1", num_args = 0..=1)]
    runahead: Option<u32>,

    /// Generate shell completions and exit (bash, zsh, fish, elvish, powershell)
    #[arg(long, value_name = "SHELL", hide = true)]
    completions: Option<clap_complete::Shell>,
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

/// Open the first gamepad that SDL can open.
fn open_first_gamepad(sys: &sdl3::GamepadSubsystem) -> Option<sdl3::gamepad::Gamepad> {
    let gp = sys
        .gamepads()
        .ok()?
        .into_iter()
        .find_map(|id| sys.open(id).ok())?;
    eprintln!("Gamepad connected: {}", gp.name().unwrap_or_default());
    enable_gamepad_sensors(&gp);
    Some(gp)
}

/// Open the default playback device with a stream at the APU rate. Returns
/// `None`, and the emulator runs silent, when there is no usable device.
fn open_audio(
    sdl: &sdl3::Sdl,
) -> Option<(sdl3::audio::AudioDevice, sdl3::audio::AudioStreamOwner)> {
    let open = || {
        let audio = sdl.audio()?;
        sdl3::hint::set("SDL_AUDIO_DEVICE_SAMPLE_FRAMES", "2048");
        let spec = AudioSpec {
            freq: Some(AUDIO_SAMPLE_RATE as i32),
            channels: Some(2),
            format: Some(AudioFormat::F32LE),
        };
        let device = audio.open_playback_device(&spec)?;
        let stream = audio.new_playback_stream(&spec, None)?;
        device.bind_stream(&stream)?;
        device.resume();
        Ok::<_, sdl3::Error>((device, stream))
    };
    open()
        .map_err(|e| eprintln!("Audio unavailable, running without sound: {}", e))
        .ok()
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();

    if let Some(shell) = cli.completions {
        let mut cmd = <Cli as clap::CommandFactory>::command();
        clap_complete::generate(shell, &mut cmd, "vibeboy", &mut std::io::stdout());
        std::process::exit(0);
    }

    // Resolve ROM path: use CLI argument or show a file dialog
    let rom_path: PathBuf = if let Some(ref p) = cli.rom {
        p.clone()
    } else {
        pick_rom_file()
    };

    // Parse scaling filter (name already validated and lowercased by parse_filter)
    let scale_filter =
        scaling::ScaleFilter::from_name(&cli.filter).expect("filter validated by parse_filter");

    ui_util::print_controls();
    if scale_filter != scaling::ScaleFilter::Nearest {
        eprintln!("  Filter: {:?}", scale_filter);
    }
    eprintln!();

    let runahead = cli.runahead.unwrap_or(0);
    let mut session = Session::new(
        &rom_path,
        SessionConfig {
            model: cli.model,
            bootrom: cli.bootrom.clone(),
            no_boot: cli.no_boot,
            lle: cli.lle,
            snes_rom: cli.snes_rom.clone(),
            printer: cli.printer,
            runahead,
            sample_rate: AUDIO_SAMPLE_RATE,
        },
    )
    .unwrap_or_else(|e| {
        eprintln!("Failed to load '{}': {}", rom_path.display(), e);
        std::process::exit(1);
    });
    let frame_dur = session.frame_duration();

    let is_sgb = session.emu.is_sgb();
    let (src_w, src_h): (u32, u32) = if is_sgb { (256, 224) } else { (160, 144) };
    let is_resizable = scale_filter.is_resizable();
    let filter_factor = scale_filter.factor().max(1); // 0 = adaptive, treat as 1× for initial sizing
    let tex_w = src_w * filter_factor;
    let tex_h = src_h * filter_factor;
    // Resizable filters start at SCALE * src size (factor=1); fixed-factor
    // filters compute window size from their native output dimensions.
    let win_scale = if filter_factor > 1 {
        SCALE / filter_factor
    } else {
        SCALE
    };
    let win_scale = win_scale.max(1);
    let win_w = tex_w * win_scale;
    let win_h = tex_h * win_scale;

    // ── SDL3 init ─────────────────────────────────────────────────────────────
    let sdl = sdl3::init().unwrap();
    let video = sdl.video().unwrap();

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
    let mut gpu = scaling::sdl::pipelines::GpuPipelines::new(&window, src_w, src_h);

    #[cfg(not(feature = "sdl3-gpu-shaders"))]
    let mut canvas = window.into_canvas();
    #[cfg(not(feature = "sdl3-gpu-shaders"))]
    let texture_creator = canvas.texture_creator();
    #[cfg(not(feature = "sdl3-gpu-shaders"))]
    let mut sdl_texture = {
        let mut tex = texture_creator
            .create_texture_streaming(sdl3::pixels::PixelFormat::ARGB8888, tex_w, tex_h)
            .unwrap();
        tex.set_scale_mode(sdl3::render::ScaleMode::Nearest);
        tex
    };
    #[cfg(not(feature = "sdl3-gpu-shaders"))]
    let (mut tex_cur_w, mut tex_cur_h) = (tex_w, tex_h);

    let mut event_pump = sdl.event_pump().unwrap();

    // ── Gamepad ───────────────────────────────────────────────────────────────
    // Gamepads are optional: without the subsystem, play on the keyboard.
    let gamepad_sys = sdl
        .gamepad()
        .map_err(|e| eprintln!("Gamepad support unavailable: {}", e))
        .ok();
    let mut gamepad = gamepad_sys.as_ref().and_then(open_first_gamepad);

    // ── Audio ─────────────────────────────────────────────────────────────────
    let audio = open_audio(&sdl);

    // ── Camera (webcam for Pocket Camera mapper, only if cart has camera) ──
    let camera_thread = if session.emu.has_camera() {
        CameraThread::start(&sdl)
    } else {
        None
    };
    let mut camera_buf = [0u8; 128 * 112];

    // ── Accelerometer (MBC7 / Kirby Tilt 'n' Tumble) ──
    let accel_source = if session.emu.has_accelerometer() {
        init_accel(&sdl)
    } else {
        AccelSource::None
    };

    let has_rumble = session.emu.has_rumble();
    let mut rumble_was_on = false;
    if runahead > 0 {
        eprintln!(
            "  Run-ahead: {} frame{}",
            runahead,
            if runahead > 1 { "s" } else { "" }
        );
    }

    let mut fps = ui_util::FpsCounter::new();

    'running: loop {
        let loop_start = Instant::now();

        // ── Events ────────────────────────────────────────────────────────────
        // Hotkeys ignore key auto-repeat, except frame advance, which steps
        // repeatedly while Period is held.
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                Event::KeyDown {
                    keycode: Some(Keycode::F5),
                    repeat: false,
                    ..
                } => {
                    session.save_state(session.slot());
                }
                Event::KeyDown {
                    keycode: Some(Keycode::F9),
                    repeat: false,
                    ..
                } => {
                    // Screenshot: save raw PPU output and scaled GPU output
                    let raw: &[u32] = if is_sgb {
                        session.emu.sgb_composited_frame()
                    } else {
                        session.emu.frame_buffer()
                    };
                    let sw = src_w as usize;
                    let sh = src_h as usize;

                    // Save raw frame
                    let raw_path = "screenshot_raw.png";
                    let mut rgb = vec![0u8; sw * sh * 3];
                    for (i, &c) in raw.iter().take(sw * sh).enumerate() {
                        rgb[i * 3] = ((c >> 16) & 0xff) as u8;
                        rgb[i * 3 + 1] = ((c >> 8) & 0xff) as u8;
                        rgb[i * 3 + 2] = (c & 0xff) as u8;
                    }
                    if image::save_buffer(
                        raw_path,
                        &rgb,
                        sw as u32,
                        sh as u32,
                        image::ColorType::Rgb8,
                    )
                    .is_ok()
                    {
                        eprintln!("Raw screenshot saved to {}", raw_path);
                    }

                    // Save scaled output by downloading the live GPU texture
                    #[cfg(feature = "sdl3-gpu-shaders")]
                    {
                        let ow = gpu.tex_w;
                        let oh = gpu.tex_h;
                        if ow > 0 && oh > 0 {
                            let dl_size = ow * oh * 4;
                            if let Ok(dl_buf) = gpu
                                .device
                                .create_transfer_buffer()
                                .with_usage(sdl3::sys::gpu::SDL_GPUTransferBufferUsage::DOWNLOAD)
                                .with_size(dl_size)
                                .build()
                                && let Ok(cmd) = gpu.device.acquire_command_buffer()
                                && let Ok(cp) = gpu.device.begin_copy_pass(&cmd)
                            {
                                unsafe {
                                    let src = sdl3::sys::gpu::SDL_GPUTextureRegion {
                                        texture: gpu.tex.raw(),
                                        w: ow,
                                        h: oh,
                                        d: 1,
                                        ..Default::default()
                                    };
                                    let dst = sdl3::sys::gpu::SDL_GPUTextureTransferInfo {
                                        transfer_buffer: dl_buf.raw(),
                                        ..Default::default()
                                    };
                                    sdl3::sys::gpu::SDL_DownloadFromGPUTexture(
                                        cp.raw(),
                                        &src,
                                        &dst,
                                    );
                                }
                                gpu.device.end_copy_pass(cp);
                                if let Ok(f) = cmd.submit_and_acquire_fence(&gpu.device) {
                                    let _ = gpu.device.wait_fences(true, &[f]);
                                    let map = dl_buf.map::<u32>(&gpu.device, false);
                                    let px = map.mem();
                                    let mut rgb2 = vec![0u8; (ow * oh) as usize * 3];
                                    for (i, &c) in px.iter().take((ow * oh) as usize).enumerate() {
                                        rgb2[i * 3] = ((c >> 16) & 0xff) as u8;
                                        rgb2[i * 3 + 1] = ((c >> 8) & 0xff) as u8;
                                        rgb2[i * 3 + 2] = (c & 0xff) as u8;
                                    }
                                    map.unmap();
                                    let scaled_path = "screenshot_scaled.png";
                                    if image::save_buffer(
                                        scaled_path,
                                        &rgb2,
                                        ow,
                                        oh,
                                        image::ColorType::Rgb8,
                                    )
                                    .is_ok()
                                    {
                                        eprintln!(
                                            "Scaled screenshot saved to {} ({}x{})",
                                            scaled_path, ow, oh
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                Event::KeyDown {
                    keycode: Some(Keycode::F7),
                    repeat: false,
                    ..
                } => {
                    session.load_state(session.slot());
                }
                Event::KeyDown {
                    keycode: Some(Keycode::Space),
                    repeat: false,
                    ..
                } => {
                    session.toggle_pause();
                }
                Event::KeyDown {
                    keycode: Some(Keycode::Period),
                    ..
                } => {
                    session.request_frame_advance();
                }
                Event::KeyDown {
                    keycode: Some(k),
                    repeat: false,
                    ..
                } => {
                    let slot = match k {
                        Keycode::_0 => Some(0),
                        Keycode::_1 => Some(1),
                        Keycode::_2 => Some(2),
                        Keycode::_3 => Some(3),
                        Keycode::_4 => Some(4),
                        Keycode::_5 => Some(5),
                        Keycode::_6 => Some(6),
                        Keycode::_7 => Some(7),
                        Keycode::_8 => Some(8),
                        Keycode::_9 => Some(9),
                        _ => None,
                    };
                    if let Some(s) = slot {
                        session.select_slot(s);
                    }
                }
                Event::GamepadAdded { which, .. } => {
                    if gamepad.is_none()
                        && let Some(ref sys) = gamepad_sys
                        && let Ok(gp) = sys.open(which)
                    {
                        eprintln!("Gamepad connected: {}", gp.name().unwrap_or_default());
                        enable_gamepad_sensors(&gp);
                        gamepad = Some(gp);
                    }
                }
                Event::GamepadRemoved { which, .. }
                    if gamepad.as_ref().is_some_and(|g| g.id().ok() == Some(which)) =>
                {
                    eprintln!("Gamepad disconnected");
                    // Try to pick up another connected gamepad
                    gamepad = gamepad_sys.as_ref().and_then(open_first_gamepad);
                }
                _ => {}
            }
        }

        // ── Input ─────────────────────────────────────────────────────────────
        handle_input(
            &mut session.emu,
            &event_pump.keyboard_state(),
            gamepad.as_ref(),
        );

        // ── Webcam → Pocket Camera ────────────────────────────────────────────
        if let Some(ref ct) = camera_thread
            && ct.read_frame(&mut camera_buf)
        {
            session.emu.set_camera_image(&camera_buf);
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
                if gp
                    .sensor_get_data(SensorType::Accelerometer, &mut data)
                    .is_ok()
                {
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
                session.emu.set_accelerometer(mbc7_x, mbc7_y);
            }
        }

        // ── Emulation ─────────────────────────────────────────────────────────
        let ks = event_pump.keyboard_state();
        // Left shoulder = rewind, right shoulder = fast forward
        let shoulder = |b| gamepad.as_ref().is_some_and(|gp| gp.button(b));
        let hold = HoldInputs {
            rewind: ks.is_scancode_pressed(Scancode::Backspace) || shoulder(GpButton::LeftShoulder),
            fast_forward: ks.is_scancode_pressed(Scancode::Tab)
                || shoulder(GpButton::RightShoulder),
            slow_motion: ks.is_scancode_pressed(Scancode::Minus),
        };
        // Audio-driven pacing: the session steps by how much audio SDL has
        // queued (bytes of interleaved stereo f32), or by wall-clock time
        // without an audio device.
        let audio_queued = audio
            .as_ref()
            .and_then(|(_, stream)| stream.queued_bytes().ok())
            .map(|bytes| bytes.max(0) as usize / (2 * size_of::<f32>()));
        let out = session.tick(&hold, audio_queued);
        if let Some((_, stream)) = &audio
            && !out.audio.is_empty()
        {
            let _ = stream.put_data_f32(&out.audio);
        }

        // ── Rumble ────────────────────────────────────────────────────────────
        if has_rumble {
            let rumble_on = session.emu.drain_rumble();
            if rumble_on != rumble_was_on {
                if let Some(ref mut gp) = gamepad {
                    if rumble_on {
                        let _ = gp.set_rumble(0xAAAA, 0xAAAA, 100);
                    } else {
                        let _ = gp.set_rumble(0, 0, 0);
                    }
                }
                rumble_was_on = rumble_on;
            }
        }

        // ── Render ────────────────────────────────────────────────────────────
        // Nothing is presented (so vsync does not pace the loop) while the
        // window is occluded, minimized or hidden.
        #[cfg(feature = "sdl3-gpu-shaders")]
        let window_flags = window.window_flags();
        #[cfg(not(feature = "sdl3-gpu-shaders"))]
        let window_flags = canvas.window().window_flags();
        let occluded = window_flags
            & (sdl3::sys::video::SDL_WINDOW_OCCLUDED
                | sdl3::sys::video::SDL_WINDOW_MINIMIZED
                | sdl3::sys::video::SDL_WINDOW_HIDDEN)
            != sdl3::sys::video::SDL_WindowFlags(0);
        if !occluded {
            let raw_src: &[u32] = if is_sgb {
                session.emu.sgb_composited_frame()
            } else {
                session.emu.frame_buffer()
            };
            let sw = src_w as usize;
            let sh = src_h as usize;

            #[cfg(feature = "sdl3-gpu-shaders")]
            {
                use scaling::sdl::pipelines::GpuRenderMode;
                let mode = gpu.ensure_pipeline(scale_filter, cli.cpu_filter);

                match mode {
                    GpuRenderMode::FullGpuVectorize => {
                        let (disp_w, disp_h) = display_size(&window, src_w, src_h);
                        let scale = (disp_w as f64 / sw as f64).min(disp_h as f64 / sh as f64);
                        let out_w = (sw as f64 * scale).round() as u32;
                        let out_h = (sh as f64 * scale).round() as u32;
                        gpu.render_full_vectorize_to_window(
                            &window,
                            raw_src,
                            sw as u32,
                            sh as u32,
                            out_w,
                            out_h,
                            scale as f32,
                        );
                    }
                    GpuRenderMode::ScaleCompute => {
                        let (disp_w, disp_h) = display_size(&window, src_w, src_h);
                        gpu.render_scale_compute(
                            scale_filter,
                            &window,
                            raw_src,
                            src_w,
                            src_h,
                            disp_w as u32,
                            disp_h as u32,
                        );
                    }
                    GpuRenderMode::Cpu => {
                        let (disp_w, disp_h) = display_size(&window, src_w, src_h);
                        let (scaled, fw, fh) =
                            cpu_scale_frame(&scale_filter, raw_src, sw, sh, disp_w, disp_h);
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
                } else {
                    (0, 0)
                };

                let (scaled, fw, fh) =
                    cpu_scale_frame(&scale_filter, raw_src, sw, sh, disp_w, disp_h);
                let (fw, fh) = (fw as usize, fh as usize);
                let final_src = if fw == sw && fh == sh {
                    raw_src
                } else {
                    &scaled
                };

                if fw as u32 != tex_cur_w || fh as u32 != tex_cur_h {
                    tex_cur_w = fw as u32;
                    tex_cur_h = fh as u32;
                    sdl_texture = texture_creator
                        .create_texture_streaming(
                            sdl3::pixels::PixelFormat::ARGB8888,
                            tex_cur_w,
                            tex_cur_h,
                        )
                        .unwrap();
                    sdl_texture.set_scale_mode(sdl3::render::ScaleMode::Nearest);
                }
                sdl_texture
                    .with_lock(None, |pixels: &mut [u8], pitch: usize| {
                        for y in 0..fh {
                            for x in 0..fw {
                                let argb = final_src[y * fw + x];
                                let off = y * pitch + x * 4;
                                pixels[off] = argb as u8;
                                pixels[off + 1] = (argb >> 8) as u8;
                                pixels[off + 2] = (argb >> 16) as u8;
                                pixels[off + 3] = 0xFF;
                            }
                        }
                    })
                    .unwrap();

                let dst_rect: Option<sdl3::render::FRect> = if is_resizable {
                    let dx = (ww as usize).saturating_sub(fw) / 2;
                    let dy = (wh as usize).saturating_sub(fh) / 2;
                    Some(sdl3::render::FRect::new(
                        dx as f32, dy as f32, fw as f32, fh as f32,
                    ))
                } else {
                    None
                };
                canvas.clear();
                canvas.copy(&sdl_texture, None, dst_rect).unwrap();
                canvas.present();
            }
        }

        // ── FPS counter ───────────────────────────────────────────────────────
        fps.update(out.frames_emulated(), out.emu_time);

        // While presenting, vsync paces the loop and the session's audio
        // pacing keeps emulation at the right speed at any refresh rate.
        // Without a present, sleep out the rest of the frame instead of
        // spinning a core.
        if occluded {
            std::thread::sleep(frame_dur.saturating_sub(loop_start.elapsed()));
        }
    }

    // Cleanup accelerometer
    #[cfg(target_os = "macos")]
    if matches!(accel_source, AccelSource::MacosNative) {
        macos_accel::close();
    }

    // Camera thread shuts down automatically via Drop
    drop(camera_thread);

    session.flush_save();
}
