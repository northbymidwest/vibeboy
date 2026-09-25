use muda::{CheckMenuItem, Menu, MenuEvent};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use super::camera::CameraThread;
use super::cpal_audio::CpalAudio;
use super::emulator::Emulator;
use super::gpu::GpuRenderer;
use super::menu::{
    ID_FORCE_CPU, ID_OPEN, ID_PAUSE, ID_PRINTER, ID_QUIT, ID_RESET, build_menu,
    filter_id_to_filter, model_id_to_model, slot_load_id, slot_save_id,
};
use super::model::GbModel;
use super::scaling;
use super::ui_util::{self, HoldInputs, Session, SessionConfig};
use super::util::frame_duration;
use super::{AUDIO_SAMPLE_RATE, Cli, GB_H, GB_W, SCALE, SGB_H, SGB_W};

pub(super) struct App {
    quit_requested: bool,
    cli: Cli,
    /// The loaded game; `None` until a ROM has been opened.
    session: Option<Session>,
    window: Option<Arc<Window>>,
    gpu: Option<GpuRenderer>,
    _menu: Option<Menu>,
    filter_items: Vec<(CheckMenuItem, scaling::ScaleFilter)>,
    printer_item: Option<CheckMenuItem>,
    model_items: Vec<CheckMenuItem>,
    slot_items: Vec<CheckMenuItem>,
    force_cpu_item: Option<CheckMenuItem>,
    audio: Option<CpalAudio>,
    camera_thread: Option<CameraThread>,
    camera_buf: [u8; 128 * 112],
    scale_filter: scaling::ScaleFilter,
    wgpu_vectorize: Option<scaling::wgpu_vectorize::WgpuVectorizePipeline>,
    wgpu_scale: Option<scaling::wgpu_scale::WgpuScalePipeline>,
    frame_start: Instant,
    force_cpu: bool,
    src_w: u32,
    src_h: u32,
    fps: ui_util::FpsCounter,
    gamepad: Option<ui_util::GamepadPoller>,
    kb_buttons: u8, // bitmask of keyboard-pressed buttons
    gp_buttons: u8, // bitmask of gamepad-pressed buttons
    /// Hold hotkeys from the keyboard and from the gamepad; either holds.
    kb_hold: HoldInputs,
    gp_hold: HoldInputs,
}

impl App {
    pub fn new(cli: Cli) -> Self {
        let audio = CpalAudio::start(AUDIO_SAMPLE_RATE);

        App {
            quit_requested: false,
            cli,
            session: None,
            window: None,
            gpu: None,
            _menu: None,
            filter_items: Vec::new(),
            printer_item: None,
            model_items: Vec::new(),
            slot_items: Vec::new(),
            force_cpu_item: None,
            audio,
            camera_thread: None,
            camera_buf: [0u8; 128 * 112],
            scale_filter: scaling::ScaleFilter::Nearest,
            wgpu_vectorize: None,
            wgpu_scale: None,
            frame_start: Instant::now(),
            force_cpu: false,
            src_w: GB_W,
            src_h: GB_H,
            fps: ui_util::FpsCounter::new(),
            gamepad: ui_util::GamepadPoller::new(),
            kb_buttons: 0,
            gp_buttons: 0,
            kb_hold: HoldInputs::default(),
            gp_hold: HoldInputs::default(),
        }
    }

    /// Open the ROM at `path`, replacing the current game (whose battery
    /// save is written first).
    fn load_rom(&mut self, path: &Path) {
        let result = match self.session {
            Some(ref mut session) => session.load_rom(path),
            None => Session::new(
                path,
                SessionConfig {
                    model: self.cli.model,
                    bootrom: self.cli.bootrom.clone(),
                    no_boot: self.cli.no_boot,
                    printer: self.cli.printer,
                    sample_rate: AUDIO_SAMPLE_RATE,
                    ..Default::default()
                },
            )
            .map(|session| self.session = Some(session)),
        };
        if let Err(e) = result {
            eprintln!("Failed to load ROM '{}': {}", path.display(), e);
            return;
        }
        self.on_new_emulator();

        if let Some(ref window) = self.window {
            let size = LogicalSize::new(self.src_w * SCALE, self.src_h * SCALE);
            let _ = window.request_inner_size(size);
            window.set_title(&format!(
                "VibeBoy \u{2014} {}",
                path.file_name().unwrap_or_default().to_string_lossy()
            ));
        }
    }

    /// Sync frontend state with a freshly built emulator (ROM switch,
    /// reset, model change).
    fn on_new_emulator(&mut self) {
        let Some(ref session) = self.session else {
            return;
        };
        let is_sgb = session.emu.is_sgb();
        self.src_w = if is_sgb { SGB_W } else { GB_W };
        self.src_h = if is_sgb { SGB_H } else { GB_H };

        // Start camera thread if cart has camera (Pocket Camera)
        if session.emu.has_camera() && self.camera_thread.is_none() {
            self.camera_thread = CameraThread::start();
        }
    }

    fn frame_duration(&self) -> Duration {
        self.session
            .as_ref()
            .map_or_else(|| frame_duration(GbModel::Cgb), Session::frame_duration)
    }

    fn update_filter_checkmarks(&self) {
        for (item, filter) in &self.filter_items {
            item.set_checked(*filter == self.scale_filter);
        }
    }

    fn select_slot(&mut self, slot: usize) {
        if let Some(ref mut session) = self.session {
            session.select_slot(slot);
        }
        for (i, item) in self.slot_items.iter().enumerate() {
            item.set_checked(i == slot);
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
                self.quit_requested = true;
            }
            ID_PAUSE => {
                if let Some(ref mut session) = self.session {
                    session.toggle_pause();
                }
            }
            ID_RESET => {
                if let Some(ref mut session) = self.session {
                    session.reset();
                }
                self.on_new_emulator();
            }
            ID_FORCE_CPU => {
                if let Some(ref item) = self.force_cpu_item {
                    self.force_cpu = item.is_checked();
                    eprintln!("Force CPU: {}", if self.force_cpu { "on" } else { "off" });
                }
            }
            ID_PRINTER => {
                if let Some(ref item) = self.printer_item {
                    let on = item.is_checked();
                    match self.session {
                        Some(ref mut session) => session.set_printer(on),
                        // Applies to the first ROM opened.
                        None => self.cli.printer = on,
                    }
                }
            }
            other => {
                // Check model menu items
                if let Some(new_model) = model_id_to_model(other) {
                    match self.session {
                        Some(ref mut session) => session.set_model(new_model),
                        // Applies to the first ROM opened.
                        None => self.cli.model = new_model,
                    }
                    self.on_new_emulator();
                    // Update checkmarks
                    for item in &self.model_items {
                        item.set_checked(item.id().0 == other);
                    }
                    return;
                }

                // Check filter menu items
                if let Some(filter) = filter_id_to_filter(other) {
                    self.scale_filter = filter;
                    self.update_filter_checkmarks();
                    eprintln!("Filter: {:?}", filter);
                    return;
                }

                for i in 0..=9 {
                    if other == slot_save_id(i) {
                        if let Some(ref mut session) = self.session {
                            session.save_state(i);
                        }
                        return;
                    }
                    if other == slot_load_id(i) {
                        if let Some(ref mut session) = self.session {
                            session.load_state(i);
                        }
                        return;
                    }
                }

                // Slot selection checkmarks
                if other.starts_with("select_slot_")
                    && let Ok(n) = other["select_slot_".len()..].parse::<usize>()
                {
                    self.select_slot(n);
                }
            }
        }
    }

    /// Run one session tick and queue its audio.
    fn step(&mut self) {
        let Some(ref mut session) = self.session else {
            return;
        };

        // Feed webcam frames to Pocket Camera
        if let Some(ref ct) = self.camera_thread
            && ct.read_frame(&mut self.camera_buf)
        {
            session.emu.set_camera_image(&self.camera_buf);
        }

        let hold = HoldInputs {
            rewind: self.kb_hold.rewind || self.gp_hold.rewind,
            fast_forward: self.kb_hold.fast_forward || self.gp_hold.fast_forward,
            slow_motion: self.kb_hold.slow_motion,
        };
        let queued = self.audio.as_ref().map(CpalAudio::queued_frames);
        let out = session.tick(&hold, queued);
        if let Some(ref mut audio) = self.audio {
            audio.push(&out.audio);
        }

        // Rumble
        if let Some(ref mut gp) = self.gamepad
            && session.emu.has_rumble()
        {
            gp.ensure_rumble();
            gp.set_rumble(session.emu.drain_rumble());
        }

        self.fps.update(out.frames_emulated(), out.emu_time);
    }

    /// Draw the emulator's current frame.
    fn render(&mut self) {
        let Some(ref mut session) = self.session else {
            return;
        };
        let emu = &mut session.emu;

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
        let (frame_pixels, frame_w, frame_h): (&[u32], usize, usize) = if self.force_cpu {
            // Force CPU: skip all GPU paths
            if let Some((s, w, h)) =
                scaling::cpu_scale(self.scale_filter, fb, sw, sh, disp_w, disp_h)
            {
                scaled = s;
                (&scaled, w as usize, h as usize)
            } else {
                (fb, sw, sh)
            }
        } else if matches!(self.scale_filter, scaling::ScaleFilter::Nearest) {
            (fb, sw, sh)
        } else if matches!(self.scale_filter, scaling::ScaleFilter::Vectorize) {
            // Use logical pixels for vectorize output, not Retina physical pixels.
            // The blit sampler upscales to physical resolution.
            let scale_factor = window.scale_factor();
            let logical_w = disp_w as f64 / scale_factor;
            let logical_h = disp_h as f64 / scale_factor;
            let s = (logical_w / sw as f64).min(logical_h / sh as f64);
            let ow = (sw as f64 * s).round() as u32;
            let oh = (sh as f64 * s).round() as u32;
            if self.wgpu_vectorize.is_none() {
                self.wgpu_vectorize = Some(scaling::wgpu_vectorize::WgpuVectorizePipeline::new(
                    &gpu.device,
                ));
            }
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vectorize+blit"),
                });
            let pipeline = self.wgpu_vectorize.as_mut().unwrap();
            let out_tex = pipeline.encode(
                &gpu.device,
                &gpu.queue,
                &mut encoder,
                fb,
                sw as u32,
                sh as u32,
                ow,
                oh,
                s as f32,
            );
            let frame = match gpu.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(f)
                | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                _ => return,
            };
            let fb_view = frame.texture.create_view(&Default::default());
            gpu.encode_blit(&mut encoder, out_tex, &fb_view, self.src_w, self.src_h);
            gpu.queue.submit(std::iter::once(encoder.finish()));
            gpu.queue.present(frame);
            return; // skip normal render path
        } else if let Some(wgpu_filter) = map_scale_filter(self.scale_filter) {
            // GPU compute scaling filter
            if self.wgpu_scale.is_none() {
                self.wgpu_scale = Some(scaling::wgpu_scale::WgpuScalePipeline::new(&gpu.device));
            }
            let factor = self.scale_filter.factor();
            let (ow, oh) = if factor > 0 {
                (sw as u32 * factor, sh as u32 * factor)
            } else {
                let scale_factor = window.scale_factor();
                let lw = disp_w as f64 / scale_factor;
                let lh = disp_h as f64 / scale_factor;
                let s = (lw / sw as f64).min(lh / sh as f64);
                (
                    (sw as f64 * s).round() as u32,
                    (sh as f64 * s).round() as u32,
                )
            };
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("scale+blit"),
                });
            let pipeline = self.wgpu_scale.as_mut().unwrap();
            let out_tex = pipeline.encode(
                &gpu.device,
                &gpu.queue,
                &mut encoder,
                wgpu_filter,
                fb,
                sw as u32,
                sh as u32,
                ow,
                oh,
            );
            let frame = match gpu.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(f)
                | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                _ => return,
            };
            let fb_view = frame.texture.create_view(&Default::default());
            gpu.encode_blit(&mut encoder, out_tex, &fb_view, self.src_w, self.src_h);
            gpu.queue.submit(std::iter::once(encoder.finish()));
            gpu.queue.present(frame);
            return;
        } else if let Some((s, w, h)) =
            scaling::cpu_scale(self.scale_filter, fb, sw, sh, disp_w, disp_h)
        {
            scaled = s;
            (&scaled, w as usize, h as usize)
        } else {
            (fb, sw, sh)
        };

        gpu.render(
            frame_pixels,
            frame_w as u32,
            frame_h as u32,
            self.src_w,
            self.src_h,
        );
    }

    /// Apply the combined keyboard and gamepad button state.
    fn apply_buttons(&mut self) {
        if let Some(ref mut session) = self.session {
            let combined = self.kb_buttons | self.gp_buttons;
            for bit in 0..8u8 {
                let mask = 1 << bit;
                session.emu.set_button(mask, combined & mask != 0);
            }
        }
    }

    fn quit(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(ref mut session) = self.session {
            session.flush_save();
        }
        event_loop.exit();
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
        let (menu, filter_items, printer_item, model_items, slot_items, force_cpu_item) =
            build_menu(self.cli.printer);
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
        self.printer_item = Some(printer_item);
        self.model_items = model_items;
        self.slot_items = slot_items;
        self.force_cpu_item = Some(force_cpu_item);
        self.window = Some(window);
        self.gpu = Some(gpu);

        // Load ROM if provided on command line, otherwise show file dialog
        if let Some(path) = self.cli.rom.clone() {
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

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.quit(event_loop),

            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(key) = event.physical_key else {
                    return;
                };
                let pressed = event.state == ElementState::Pressed;

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
                    self.apply_buttons();
                }

                match key {
                    KeyCode::Backspace => self.kb_hold.rewind = pressed,
                    KeyCode::Tab => self.kb_hold.fast_forward = pressed,
                    KeyCode::Minus => self.kb_hold.slow_motion = pressed,
                    _ => {}
                }

                if !pressed {
                    return;
                }
                if key == KeyCode::Escape {
                    self.quit(event_loop);
                    return;
                }
                let Some(ref mut session) = self.session else {
                    return;
                };
                // Frame advance steps repeatedly while Period is held; the
                // other hotkeys ignore key auto-repeat.
                if key == KeyCode::Period {
                    session.request_frame_advance();
                    return;
                }
                if event.repeat {
                    return;
                }
                match key {
                    KeyCode::Space => {
                        session.toggle_pause();
                    }
                    KeyCode::F5 => session.save_state(session.slot()),
                    KeyCode::F7 => session.load_state(session.slot()),
                    KeyCode::Digit0 => self.select_slot(0),
                    KeyCode::Digit1 => self.select_slot(1),
                    KeyCode::Digit2 => self.select_slot(2),
                    KeyCode::Digit3 => self.select_slot(3),
                    KeyCode::Digit4 => self.select_slot(4),
                    KeyCode::Digit5 => self.select_slot(5),
                    KeyCode::Digit6 => self.select_slot(6),
                    KeyCode::Digit7 => self.select_slot(7),
                    KeyCode::Digit8 => self.select_slot(8),
                    KeyCode::Digit9 => self.select_slot(9),
                    _ => {}
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Process menu events
        while let Ok(event) = MenuEvent::receiver().try_recv() {
            self.handle_menu_event(event.id().0.as_str());
        }

        if self.quit_requested {
            self.quit(event_loop);
            return;
        }

        // -- Gamepad polling --
        match self.gamepad {
            Some(ref mut gp) => {
                let gs = gp.poll();
                self.gp_buttons = gs.buttons;
                self.gp_hold.rewind = gs.rewind;
                self.gp_hold.fast_forward = gs.fast_forward;
            }
            None => self.gp_buttons = 0,
        }
        self.apply_buttons();

        self.step();
        self.render();

        // Frame rate cap
        let frame_dur = self.frame_duration();
        let remaining = frame_dur.saturating_sub(self.frame_start.elapsed());
        if remaining > Duration::from_millis(2) {
            std::thread::sleep(remaining - Duration::from_millis(2));
        }
        while self.frame_start.elapsed() < frame_dur {
            std::hint::spin_loop();
        }
        self.frame_start = Instant::now();

        if let Some(ref window) = self.window {
            window.request_redraw();
        }
    }
}

fn map_scale_filter(filter: scaling::ScaleFilter) -> Option<scaling::wgpu_scale::WgpuScaleFilter> {
    scaling::wgpu_scale::WgpuScaleFilter::from_scale_filter(filter)
}
