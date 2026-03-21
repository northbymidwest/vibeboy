use muda::{CheckMenuItem, Menu, MenuEvent};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};
use gilrs::{Gilrs, GamepadId, Button as GilButton, Axis as GilAxis};
use gilrs::ff::{EffectBuilder, BaseEffect, BaseEffectType, Replay, Repeat, Ticks};

use super::{Cli, SCALE, GB_W, GB_H, SGB_W, SGB_H, AUDIO_SAMPLE_RATE};
use super::emulator::Emulator;
use super::model::GbModel;
use super::scaling;
use super::vectorize;
use super::ui_util::frame_duration;
use super::gpu::GpuRenderer;
use super::audio::{AudioRing, start_audio};
use super::camera::CameraThread;
use super::menu::{
    ID_OPEN, ID_QUIT, ID_PAUSE, ID_RESET,
    slot_save_id, slot_load_id, filter_id_to_filter, build_menu,
};

pub(super) struct App {
    rom_path: Option<PathBuf>,
    cli: Cli,
    emu: Option<Emulator>,
    model: GbModel,
    window: Option<Arc<Window>>,
    gpu: Option<GpuRenderer>,
    _menu: Option<Menu>,
    filter_items: Vec<(CheckMenuItem, scaling::ScaleFilter)>,
    audio_ring: Arc<Mutex<AudioRing>>,
    _audio_stream: Option<cpal::Stream>,
    camera_thread: Option<CameraThread>,
    camera_buf: [u8; 128 * 112],
    scale_filter: scaling::ScaleFilter,
    vec_cache: Option<vectorize::VectorizeCache>,
    wgpu_vectorize: Option<scaling::wgpu_vectorize::WgpuVectorizePipeline>,
    frame_start: Instant,
    frame_dur: Duration,
    paused: bool,
    current_slot: usize,
    src_w: u32,
    src_h: u32,
    fps_timer: Instant,
    fps_count: u32,
    fps_emu_total: Duration,
    gilrs: Option<Gilrs>,
    active_gamepad: Option<GamepadId>,
    kb_buttons: u8,  // bitmask of keyboard-pressed buttons
    gp_buttons: u8,  // bitmask of gamepad-pressed buttons
    rumble_effect: Option<gilrs::ff::Effect>,
    rumble_gamepad: Option<GamepadId>, // which gamepad owns the effect
    rumble_on: bool,
}

impl App {
    pub fn new(cli: Cli) -> Self {
        let model = cli.model.unwrap_or(GbModel::Cgb);

        let audio_ring = Arc::new(Mutex::new(AudioRing::new(AUDIO_SAMPLE_RATE as usize / 60 * 4 * 2, AUDIO_SAMPLE_RATE))); // ~4 frames stereo
        let (stream, actual_rate) = match start_audio(Arc::clone(&audio_ring)) {
            Some((s, r)) => (Some(s), r),
            None => (None, AUDIO_SAMPLE_RATE),
        };
        audio_ring.lock().unwrap().downsample_ratio = (AUDIO_SAMPLE_RATE / actual_rate).max(1) as usize;

        App {
            rom_path: cli.rom.clone(),
            cli,
            emu: None,
            model,
            window: None,
            gpu: None,
            _menu: None,
            filter_items: Vec::new(),
            audio_ring,
            _audio_stream: stream,
            camera_thread: None,
            camera_buf: [0u8; 128 * 112],
            scale_filter: scaling::ScaleFilter::Nearest,
            vec_cache: None,
            wgpu_vectorize: None,
            frame_start: Instant::now(),
            frame_dur: frame_duration(model),
            paused: false,
            current_slot: 0,
            src_w: GB_W,
            src_h: GB_H,
            fps_timer: Instant::now(),
            fps_count: 0,
            fps_emu_total: Duration::ZERO,
            gilrs: Gilrs::new().ok(),
            active_gamepad: None,
            kb_buttons: 0,
            gp_buttons: 0,
            rumble_effect: None,
            rumble_gamepad: None,
            rumble_on: false,
        }
    }

    fn load_rom(&mut self, path: &PathBuf) {
        let rom = fs::read(path).unwrap_or_else(|e| {
            eprintln!("Failed to read ROM '{}': {}", path.display(), e);
            Vec::new()
        });
        if rom.is_empty() {
            return;
        }

        let boot_rom: Option<Vec<u8>> = if self.cli.no_boot {
            None
        } else if let Some(ref p) = self.cli.bootrom {
            fs::read(p).ok()
        } else {
            let path = match self.model {
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

        let emu = Emulator::new(rom, boot_rom, Some(path.as_path()), self.model, None);
        let is_sgb = emu.is_sgb();
        self.src_w = if is_sgb { SGB_W } else { GB_W };
        self.src_h = if is_sgb { SGB_H } else { GB_H };

        // Start camera thread if cart has camera (Pocket Camera)
        if emu.has_camera() && self.camera_thread.is_none() {
            self.camera_thread = CameraThread::start();
        }

        self.emu = Some(emu);
        self.rom_path = Some(path.clone());
        self.frame_dur = frame_duration(self.model);

        if let Some(ref window) = self.window {
            let size = LogicalSize::new(self.src_w * SCALE, self.src_h * SCALE);
            let _ = window.request_inner_size(size);
            window.set_title(&format!(
                "VibeBoy \u{2014} {}",
                path.file_name().unwrap_or_default().to_string_lossy()
            ));
        }
    }

    fn update_filter_checkmarks(&self) {
        for (item, filter) in &self.filter_items {
            item.set_checked(*filter == self.scale_filter);
        }
    }

    fn ensure_rumble_effect(&mut self, gp_id: GamepadId) {
        if self.rumble_gamepad == Some(gp_id) && self.rumble_effect.is_some() {
            return;
        }
        // Drop old effect
        self.rumble_effect = None;
        self.rumble_gamepad = None;
        self.rumble_on = false;

        let gilrs = match self.gilrs.as_mut() {
            Some(g) => g,
            None => return,
        };

        if !gilrs.gamepad(gp_id).is_ff_supported() {
            return;
        }

        // Long continuous rumble effect — we start/stop it manually
        let effect = EffectBuilder::new()
            .add_effect(BaseEffect {
                kind: BaseEffectType::Strong { magnitude: 40_000 },
                scheduling: Replay {
                    play_for: Ticks::from_ms(u32::MAX),
                    ..Default::default()
                },
                envelope: Default::default(),
            })
            .repeat(Repeat::Infinitely)
            .gamepads(&[gp_id])
            .finish(gilrs);

        match effect {
            Ok(e) => {
                self.rumble_effect = Some(e);
                self.rumble_gamepad = Some(gp_id);
            }
            Err(e) => {
                log::warn!("Failed to create rumble effect: {}", e);
            }
        }
    }

    fn update_rumble(&mut self, on: bool) {
        if on == self.rumble_on {
            return;
        }
        self.rumble_on = on;
        if let Some(ref effect) = self.rumble_effect {
            if on {
                let _ = effect.play();
            } else {
                let _ = effect.stop();
            }
        }
    }

    fn handle_menu_event(&mut self, id: &str) {
        match id {
            ID_OPEN => {
                let file = rfd::FileDialog::new()
                    .add_filter("Game Boy ROMs", &["gb", "gbc"])
                    .add_filter("All files", &["*"])
                    .pick_file();
                if let Some(path) = file {
                    self.load_rom(&path);
                }
            }
            ID_QUIT => {
                if let Some(ref emu) = self.emu {
                    emu.save();
                }
                std::process::exit(0);
            }
            ID_PAUSE => {
                self.paused = !self.paused;
                eprintln!("{}", if self.paused { "Paused" } else { "Resumed" });
            }
            ID_RESET => {
                if let Some(path) = self.rom_path.clone() {
                    self.load_rom(&path);
                }
            }
            other => {
                // Check filter menu items
                if let Some(filter) = filter_id_to_filter(other) {
                    self.scale_filter = filter;
                    self.update_filter_checkmarks();
                    match filter {
                        scaling::ScaleFilter::VectorizeLegacy => {
                            self.vec_cache = Some(vectorize::VectorizeCache::new_legacy(false));
                        }
                        scaling::ScaleFilter::VectorizeLegacyAdaptive => {
                            self.vec_cache = Some(vectorize::VectorizeCache::new_legacy(true));
                        }
                        _ => {}
                    }
                    eprintln!("Filter: {:?}", filter);
                    return;
                }

                for i in 1..=9 {
                    if other == slot_save_id(i) {
                        if let Some(ref mut emu) = self.emu {
                            emu.save_state(i - 1);
                            if let (Some(rp), Some(data)) = (&self.rom_path, emu.save_state_to_bytes(i - 1)) {
                                let path = rp.with_extension(format!("{}.ss", i));
                                match std::fs::write(&path, &data) {
                                    Ok(_) => eprintln!("State saved to slot {} ({})", i, path.display()),
                                    Err(e) => eprintln!("State saved to slot {} (disk write failed: {})", i, e),
                                }
                            } else {
                                eprintln!("State saved to slot {}", i);
                            }
                        }
                        return;
                    }
                    if other == slot_load_id(i) {
                        if let Some(ref mut emu) = self.emu {
                            if emu.load_state(i - 1) {
                                eprintln!("State loaded from slot {}", i);
                            } else if let Some(ref rp) = self.rom_path {
                                let path = rp.with_extension(format!("{}.ss", i));
                                if let Ok(data) = std::fs::read(&path) {
                                    if emu.load_state_from_bytes(i - 1, &data) {
                                        eprintln!("State loaded from disk: {}", path.display());
                                    } else {
                                        eprintln!("Failed to load state from {}", path.display());
                                    }
                                } else {
                                    eprintln!("Slot {} is empty", i);
                                }
                            } else {
                                eprintln!("Slot {} is empty", i);
                            }
                        }
                        return;
                    }
                }
            }
        }
    }

    fn step_and_render(&mut self) {
        if self.paused {
            return;
        }

        let emu = match self.emu.as_mut() {
            Some(e) => e,
            None => return,
        };

        // Feed webcam frames to Pocket Camera
        if let Some(ref ct) = self.camera_thread {
            if ct.read_frame(&mut self.camera_buf) {
                emu.set_camera_image(&self.camera_buf);
            }
        }

        emu.step_frame();

        // Push audio samples directly to ring buffer (96kHz stereo, matching APU output)
        let samples = emu.drain_audio_samples();
        if !samples.is_empty() {
            let mut ring = self.audio_ring.lock().unwrap();
            ring.push(&samples);
        }

        // Render via wgpu
        let gpu = match self.gpu.as_mut() {
            Some(g) => g,
            None => return,
        };
        let window = match self.window.as_ref() {
            Some(w) => w,
            None => return,
        };

        let is_sgb = emu.is_sgb();
        let fb: &[u32] = if is_sgb {
            emu.sgb_composited_frame()
        } else {
            emu.frame_buffer()
        };
        let sw = self.src_w as usize;
        let sh = self.src_h as usize;

        let phys = window.inner_size();
        let win_w = phys.width as usize;
        let win_h = phys.height as usize;
        if win_w == 0 || win_h == 0 {
            return;
        }

        gpu.resize(win_w as u32, win_h as u32);

        // Apply scaling filter
        let disp_w = win_w;
        let disp_h = win_h;

        let scaled;
        let (frame_pixels, frame_w, frame_h): (&[u32], usize, usize) =
            if matches!(self.scale_filter, scaling::ScaleFilter::Nearest) {
                (fb, sw, sh)
            } else if matches!(self.scale_filter, scaling::ScaleFilter::VectorizeLegacy | scaling::ScaleFilter::VectorizeLegacyAdaptive) {
                let scale = (disp_w as f64 / sw as f64).min(disp_h as f64 / sh as f64);
                let adaptive = matches!(self.scale_filter, scaling::ScaleFilter::VectorizeLegacyAdaptive);
                let cache = self.vec_cache.get_or_insert_with(|| vectorize::VectorizeCache::new_legacy(adaptive));
                let (raster, vw, vh) = cache.rasterize(fb, sw, sh, scale);
                (raster, vw, vh)
            } else if matches!(self.scale_filter, scaling::ScaleFilter::VectorizeDiffusion) {
                let s = (disp_w as f64 / sw as f64).min(disp_h as f64 / sh as f64);
                let sc = s.round().max(1.0) as usize;
                let (buf, dw, dh) = vectorize::rasterize::rasterize_diffusion(fb, sw, sh, sc);
                scaled = buf;
                (&scaled, dw, dh)
            } else if matches!(self.scale_filter, scaling::ScaleFilter::VectorizeSplineDiffusion | scaling::ScaleFilter::VectorizeSplineDiffusionAdaptive) {
                let s = (disp_w as f64 / sw as f64).min(disp_h as f64 / sh as f64);
                let sc = s.round().max(1.0) as usize;
                let cache = self.vec_cache.get_or_insert_with(|| vectorize::VectorizeCache::new_legacy(false));
                let (paths, bg) = cache.get_paths(fb, sw, sh);
                let (buf, dw, dh) = vectorize::rasterize::rasterize_spline_diffusion(
                    paths, fb, sw, sh, bg, sc,
                );
                scaled = buf;
                (&scaled, dw, dh)
            } else if matches!(self.scale_filter, scaling::ScaleFilter::VectorizeGpu) {
                // Use logical pixels for vectorize output, not Retina physical pixels.
                // The blit sampler upscales to physical resolution.
                let scale_factor = window.scale_factor();
                let logical_w = disp_w as f64 / scale_factor;
                let logical_h = disp_h as f64 / scale_factor;
                let s = (logical_w / sw as f64).min(logical_h / sh as f64);
                let ow = (sw as f64 * s).round() as u32;
                let oh = (sh as f64 * s).round() as u32;
                if self.wgpu_vectorize.is_none() {
                    self.wgpu_vectorize = Some(
                        scaling::wgpu_vectorize::WgpuVectorizePipeline::new(&gpu.device)
                    );
                }
                let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vectorize+blit"),
                });
                let pipeline = self.wgpu_vectorize.as_mut().unwrap();
                let out_tex = pipeline.encode(&gpu.device, &gpu.queue, &mut encoder, fb, sw as u32, sh as u32, ow, oh, s as f32);
                let frame = match gpu.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(_) => return,
                };
                let fb_view = frame.texture.create_view(&Default::default());
                gpu.encode_blit(&mut encoder, out_tex, &fb_view, self.src_w, self.src_h);
                gpu.queue.submit(std::iter::once(encoder.finish()));
                frame.present();
                return; // skip normal render path
            } else if let Some((s, w, h)) = scaling::cpu_scale(self.scale_filter, fb, sw, sh, disp_w, disp_h) {
                scaled = s;
                (&scaled, w as usize, h as usize)
            } else {
                (fb, sw, sh)
            };

        gpu.render(frame_pixels, frame_w as u32, frame_h as u32, self.src_w, self.src_h);

        // FPS counter
        let emu_time = self.frame_start.elapsed();
        self.fps_count += 1;
        self.fps_emu_total += emu_time;
        let elapsed = self.fps_timer.elapsed();
        if elapsed >= Duration::from_secs(1) {
            let fps = self.fps_count as f64 / elapsed.as_secs_f64();
            let avg_emu_ms = self.fps_emu_total.as_secs_f64() * 1000.0 / self.fps_count as f64;
            eprintln!("FPS: {:.1}  emu: {:.2}ms/frame", fps, avg_emu_ms);
            self.fps_count = 0;
            self.fps_emu_total = Duration::ZERO;
            self.fps_timer = Instant::now();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let win_w = self.src_w * SCALE;
        let win_h = self.src_h * SCALE;

        let attrs = Window::default_attributes()
            .with_title("VibeBoy")
            .with_inner_size(LogicalSize::new(win_w, win_h))
            .with_min_inner_size(LogicalSize::new(GB_W, GB_H))
            .with_resizable(true);

        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        // Set up menu bar
        let (menu, filter_items) = build_menu();
        #[cfg(target_os = "macos")]
        {
            menu.init_for_nsapp();
        }
        #[cfg(target_os = "windows")]
        {
            use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
            if let Ok(handle) = window.window_handle() {
                if let RawWindowHandle::Win32(h) = handle.as_raw() {
                    let _ = unsafe { menu.init_for_hwnd(h.hwnd.get() as _) };
                }
            }
        }
        #[cfg(target_os = "linux")]
        {
            // On Linux with GTK, muda needs gtk initialization
            // For now, skip -- muda will use gtk_application_set_menubar if available
        }

        let gpu = GpuRenderer::new(window.clone());

        self._menu = Some(menu);
        self.filter_items = filter_items;
        self.window = Some(window);
        self.gpu = Some(gpu);

        // Load ROM if provided on command line, otherwise show file dialog
        if let Some(path) = self.rom_path.clone() {
            self.load_rom(&path);
        } else {
            let file = rfd::FileDialog::new()
                .add_filter("Game Boy ROMs", &["gb", "gbc"])
                .add_filter("All files", &["*"])
                .pick_file();
            if let Some(path) = file {
                self.load_rom(&path);
            }
        }

        event_loop.set_control_flow(ControlFlow::Poll);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                if let Some(ref emu) = self.emu {
                    emu.save();
                }
                event_loop.exit();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    let pressed = event.state == ElementState::Pressed;

                    {
                        let btn = match key {
                            KeyCode::KeyZ => Some(Emulator::BTN_B),
                            KeyCode::KeyX => Some(Emulator::BTN_A),
                            KeyCode::Enter => Some(Emulator::BTN_START),
                            KeyCode::ShiftRight => Some(Emulator::BTN_SELECT),
                            KeyCode::ArrowRight => Some(Emulator::BTN_RIGHT),
                            KeyCode::ArrowLeft => Some(Emulator::BTN_LEFT),
                            KeyCode::ArrowUp => Some(Emulator::BTN_UP),
                            KeyCode::ArrowDown => Some(Emulator::BTN_DOWN),
                            _ => None,
                        };
                        if let Some(b) = btn {
                            if pressed {
                                self.kb_buttons |= b;
                            } else {
                                self.kb_buttons &= !b;
                            }
                        }
                    }

                    if let Some(ref mut emu) = self.emu {
                        // Apply combined keyboard + gamepad state
                        let combined = self.kb_buttons | self.gp_buttons;
                        let all_btns: &[u8] = &[
                            Emulator::BTN_RIGHT, Emulator::BTN_LEFT, Emulator::BTN_UP, Emulator::BTN_DOWN,
                            Emulator::BTN_A, Emulator::BTN_B, Emulator::BTN_SELECT, Emulator::BTN_START,
                        ];
                        for &b in all_btns {
                            emu.set_button(b, combined & b != 0);
                        }

                        if key == KeyCode::Backspace {
                            emu.set_rewinding(pressed);
                        }
                    }

                    if pressed {
                        match key {
                            KeyCode::Escape => {
                                if let Some(ref emu) = self.emu {
                                    emu.save();
                                }
                                event_loop.exit();
                            }
                            KeyCode::F5 => {
                                if let Some(ref mut emu) = self.emu {
                                    emu.save_state(self.current_slot);
                                    eprintln!(
                                        "State saved to slot {}",
                                        self.current_slot + 1
                                    );
                                }
                            }
                            KeyCode::F7 => {
                                if let Some(ref mut emu) = self.emu {
                                    if emu.load_state(self.current_slot) {
                                        eprintln!(
                                            "State loaded from slot {}",
                                            self.current_slot + 1
                                        );
                                    } else {
                                        eprintln!("Slot {} is empty", self.current_slot + 1);
                                    }
                                }
                            }
                            KeyCode::Digit1 => self.current_slot = 0,
                            KeyCode::Digit2 => self.current_slot = 1,
                            KeyCode::Digit3 => self.current_slot = 2,
                            KeyCode::Digit4 => self.current_slot = 3,
                            KeyCode::Digit5 => self.current_slot = 4,
                            KeyCode::Digit6 => self.current_slot = 5,
                            KeyCode::Digit7 => self.current_slot = 6,
                            KeyCode::Digit8 => self.current_slot = 7,
                            KeyCode::Digit9 => self.current_slot = 8,
                            _ => {}
                        }
                    }
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Process menu events
        while let Ok(event) = MenuEvent::receiver().try_recv() {
            self.handle_menu_event(event.id().0.as_str());
        }

        // -- Gamepad polling --
        if let Some(ref mut gilrs) = self.gilrs {
            while let Some(ev) = gilrs.next_event() {
                match ev.event {
                    gilrs::EventType::Connected => {
                        if self.active_gamepad.is_none() {
                            self.active_gamepad = Some(ev.id);
                            let gp = gilrs.gamepad(ev.id);
                            eprintln!("Gamepad connected: {}", gp.name());
                        }
                    }
                    gilrs::EventType::Disconnected => {
                        if self.active_gamepad == Some(ev.id) {
                            eprintln!("Gamepad disconnected");
                            self.active_gamepad = None;
                            self.rumble_effect = None;
                            self.rumble_gamepad = None;
                            self.rumble_on = false;
                        }
                    }
                    _ => {}
                }
            }

            // Find first connected gamepad if none active
            if self.active_gamepad.is_none() {
                for (id, gp) in gilrs.gamepads() {
                    if gp.is_connected() {
                        self.active_gamepad = Some(id);
                        eprintln!("Gamepad connected: {}", gp.name());
                        break;
                    }
                }
            }

            if let Some(gp_id) = self.active_gamepad {
                let gp = gilrs.gamepad(gp_id);
                const DEADZONE: f32 = 0.3;

                let gp_map: &[(GilButton, u8)] = &[
                    (GilButton::East,      Emulator::BTN_A),
                    (GilButton::South,     Emulator::BTN_B),
                    (GilButton::Start,     Emulator::BTN_START),
                    (GilButton::Select,    Emulator::BTN_SELECT),
                    (GilButton::DPadUp,    Emulator::BTN_UP),
                    (GilButton::DPadDown,  Emulator::BTN_DOWN),
                    (GilButton::DPadLeft,  Emulator::BTN_LEFT),
                    (GilButton::DPadRight, Emulator::BTN_RIGHT),
                ];

                // Left stick
                let lx = gp.axis_data(GilAxis::LeftStickX).map_or(0.0, |a| a.value());
                let ly = gp.axis_data(GilAxis::LeftStickY).map_or(0.0, |a| a.value());

                let mut gp_bits: u8 = 0;
                for &(gb, btn) in gp_map {
                    let pressed = gp.is_pressed(gb);
                    let stick = match btn {
                        b if b == Emulator::BTN_RIGHT => lx > DEADZONE,
                        b if b == Emulator::BTN_LEFT  => lx < -DEADZONE,
                        b if b == Emulator::BTN_UP    => ly > DEADZONE,
                        b if b == Emulator::BTN_DOWN  => ly < -DEADZONE,
                        _ => false,
                    };
                    if pressed || stick {
                        gp_bits |= btn;
                    }
                }
                self.gp_buttons = gp_bits;

                // Apply combined state
                if let Some(ref mut emu) = self.emu {
                    let combined = self.kb_buttons | self.gp_buttons;
                    let all_btns: &[u8] = &[
                        Emulator::BTN_RIGHT, Emulator::BTN_LEFT, Emulator::BTN_UP, Emulator::BTN_DOWN,
                        Emulator::BTN_A, Emulator::BTN_B, Emulator::BTN_SELECT, Emulator::BTN_START,
                    ];
                    for &b in all_btns {
                        emu.set_button(b, combined & b != 0);
                    }

                    // Shoulders for rewind
                    if gp.is_pressed(GilButton::LeftTrigger) {
                        emu.set_rewinding(true);
                    }
                }
            } else {
                self.gp_buttons = 0;
            }

            // Set up rumble effect for the active gamepad if cart supports it
            if let Some(gp_id) = self.active_gamepad {
                if self.emu.as_ref().is_some_and(|e| e.has_rumble()) {
                    self.ensure_rumble_effect(gp_id);
                }
            }
        }

        // Handle rewind (3x speed)
        if let Some(ref mut emu) = self.emu {
            if emu.is_rewinding() {
                for _ in 0..3 {
                    emu.rewind_one_frame();
                }
                emu.drain_audio_samples();
            }
        }

        // Step emulation
        self.step_and_render();

        // Update rumble after emulation step
        if let Some(ref emu) = self.emu {
            if emu.has_rumble() {
                let on = emu.rumble_active();
                self.update_rumble(on);
            }
        }

        // Frame rate cap
        let remaining = self.frame_dur.saturating_sub(self.frame_start.elapsed());
        if remaining > Duration::from_millis(2) {
            std::thread::sleep(remaining - Duration::from_millis(2));
        }
        while self.frame_start.elapsed() < self.frame_dur {
            std::hint::spin_loop();
        }
        self.frame_start = Instant::now();

        if let Some(ref window) = self.window {
            window.request_redraw();
        }
    }
}
