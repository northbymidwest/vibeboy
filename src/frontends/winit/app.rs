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

use super::{Cli, SCALE, GB_W, GB_H, SGB_W, SGB_H, AUDIO_SAMPLE_RATE};
use super::emulator::Emulator;
use super::model::GbModel;
use super::scaling;
use super::vectorize;
use super::ui_util::{self, frame_duration};
use super::printer;
use super::serial;
use super::gpu::GpuRenderer;
use super::audio::{AudioRing, start_audio};
use super::camera::CameraThread;
use super::menu::{
    ID_OPEN, ID_QUIT, ID_PAUSE, ID_RESET, ID_PRINTER,
    slot_save_id, slot_load_id, filter_id_to_filter, model_id_to_model, build_menu, MODEL_IDS,
};

pub(super) struct App {
    rom_path: Option<PathBuf>,
    cli: Cli,
    emu: Option<Emulator>,
    model: GbModel,
    forced_model: Option<GbModel>,
    window: Option<Arc<Window>>,
    gpu: Option<GpuRenderer>,
    _menu: Option<Menu>,
    filter_items: Vec<(CheckMenuItem, scaling::ScaleFilter)>,
    printer_item: Option<CheckMenuItem>,
    model_items: Vec<CheckMenuItem>,
    audio_ring: Arc<Mutex<AudioRing>>,
    _audio_stream: Option<cpal::Stream>,
    camera_thread: Option<CameraThread>,
    camera_buf: [u8; 128 * 112],
    scale_filter: scaling::ScaleFilter,
    vec_cache: Option<vectorize::VectorizeCache>,
    wgpu_vectorize: Option<scaling::wgpu_vectorize::WgpuVectorizePipeline>,
    wgpu_shared_chain: Option<scaling::wgpu_vectorize::WgpuSharedChainRasterizer>,
    wgpu_scale: Option<scaling::wgpu_scale::WgpuScalePipeline>,
    frame_start: Instant,
    frame_dur: Duration,
    paused: bool,
    current_slot: usize,
    src_w: u32,
    src_h: u32,
    fps: ui_util::FpsCounter,
    gamepad: Option<ui_util::GamepadPoller>,
    kb_buttons: u8,  // bitmask of keyboard-pressed buttons
    gp_buttons: u8,  // bitmask of gamepad-pressed buttons
    fast_forward: bool,
    sav_flusher: Option<ui_util::SavFlusher>,
}

impl App {
    pub fn new(cli: Cli) -> Self {
        let model = cli.model.unwrap_or(GbModel::Cgb);
        let forced_model = cli.model;

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
            forced_model,
            window: None,
            gpu: None,
            _menu: None,
            filter_items: Vec::new(),
            printer_item: None,
            model_items: Vec::new(),
            audio_ring,
            _audio_stream: stream,
            camera_thread: None,
            camera_buf: [0u8; 128 * 112],
            scale_filter: scaling::ScaleFilter::Nearest,
            vec_cache: None,
            wgpu_vectorize: None,
            wgpu_shared_chain: None,
            wgpu_scale: None,
            frame_start: Instant::now(),
            frame_dur: frame_duration(model),
            paused: false,
            current_slot: 0,
            src_w: GB_W,
            src_h: GB_H,
            fps: ui_util::FpsCounter::new(),
            gamepad: ui_util::GamepadPoller::new(),
            kb_buttons: 0,
            gp_buttons: 0,
            fast_forward: false,
            sav_flusher: None,
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

        self.model = self.forced_model.unwrap_or_else(|| ui_util::auto_detect_model(&rom));
        let boot_rom = ui_util::load_boot_rom(self.model, self.cli.bootrom.as_deref(), self.cli.no_boot);

        let mut emu = Emulator::new(rom, boot_rom, self.model, None);
        ui_util::load_sav(&mut emu, path);
        let is_sgb = emu.is_sgb();
        self.src_w = if is_sgb { SGB_W } else { GB_W };
        self.src_h = if is_sgb { SGB_H } else { GB_H };

        // Start camera thread if cart has camera (Pocket Camera)
        if emu.has_camera() && self.camera_thread.is_none() {
            self.camera_thread = CameraThread::start();
        }

        // Attach printer if enabled
        if self.printer_item.as_ref().is_some_and(|p| p.is_checked()) {
            let output_dir = std::path::Path::new("prints");
            emu.attach_serial_device(
                Box::new(printer::Printer::new(output_dir, self.model.cpu_clock_rate()))
            );
        }

        self.sav_flusher = Some(ui_util::SavFlusher::new(&emu, path));
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
                if let (Some(flusher), Some(emu)) = (&mut self.sav_flusher, &self.emu) {
                    flusher.flush(emu);
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
            ID_PRINTER => {
                if let Some(ref item) = self.printer_item {
                    let now_on = item.is_checked();
                    if let Some(ref mut emu) = self.emu {
                        if now_on {
                            let output_dir = std::path::Path::new("prints");
                            emu.attach_serial_device(
                                Box::new(printer::Printer::new(output_dir, self.model.cpu_clock_rate()))
                            );
                            eprintln!("Game Boy Printer connected");
                        } else {
                            emu.attach_serial_device(Box::new(serial::Disconnected));
                            eprintln!("Game Boy Printer disconnected");
                        }
                    }
                }
            }
            other => {
                // Check model menu items
                if let Some(new_model) = model_id_to_model(other) {
                    self.forced_model = new_model;
                    // Reload ROM with new model
                    if let Some(path) = self.rom_path.clone() {
                        self.load_rom(&path);
                    }
                    // Update checkmarks
                    for item in &self.model_items {
                        item.set_checked(item.id().0 == other);
                    }
                    let name = new_model.map(|m| format!("{:?}", m)).unwrap_or("Auto".into());
                    eprintln!("Hardware model: {}", name);
                    return;
                }

                // Check filter menu items
                if let Some(filter) = filter_id_to_filter(other) {
                    self.scale_filter = filter;
                    self.update_filter_checkmarks();
                    self.vec_cache = match filter {
                        scaling::ScaleFilter::VectorizeLegacy => Some(vectorize::VectorizeCache::new_legacy(false)),
                        scaling::ScaleFilter::VectorizeLegacyAdaptive => Some(vectorize::VectorizeCache::new_legacy(true)),
                        scaling::ScaleFilter::Vectorize => Some(vectorize::VectorizeCache::new(false)),
                        scaling::ScaleFilter::VectorizeAdaptive => Some(vectorize::VectorizeCache::new(true)),
                        _ => None,
                    };
                    eprintln!("Filter: {:?}", filter);
                    return;
                }

                for i in 1..=9 {
                    if other == slot_save_id(i) {
                        if let (Some(emu), Some(rp)) = (&mut self.emu, &self.rom_path) {
                            ui_util::save_state_to_slot(emu, rp, i - 1);
                        }
                        return;
                    }
                    if other == slot_load_id(i) {
                        if let (Some(emu), Some(rp)) = (&mut self.emu, &self.rom_path) {
                            ui_util::load_state_from_slot(emu, rp, i - 1);
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

        if self.fast_forward {
            // Run 4 frames, downsample audio to fit 1 frame
            for _ in 0..4 {
                emu.step_frame();
            }
            let samples = emu.drain_audio_samples();
            if !samples.is_empty() {
                let resampled = ui_util::downsample_audio(&samples, 4);
                let mut ring = self.audio_ring.lock().unwrap();
                ring.push(&resampled);
            }
        } else {
            emu.step_frame();
            let samples = emu.drain_audio_samples();
            if !samples.is_empty() {
                let mut ring = self.audio_ring.lock().unwrap();
                ring.push(&samples);
            }
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
            } else if matches!(self.scale_filter,
                scaling::ScaleFilter::VectorizeLegacy | scaling::ScaleFilter::VectorizeLegacyAdaptive
                | scaling::ScaleFilter::Vectorize | scaling::ScaleFilter::VectorizeAdaptive)
            {
                // All vectorize variants: extract paths on CPU, rasterize on GPU
                let scale_factor = window.scale_factor();
                let logical_w = disp_w as f64 / scale_factor;
                let logical_h = disp_h as f64 / scale_factor;
                let scale = (logical_w / sw as f64).min(logical_h / sh as f64);
                let adaptive = matches!(self.scale_filter, scaling::ScaleFilter::VectorizeAdaptive);
                let cache = self.vec_cache.get_or_insert_with(|| vectorize::VectorizeCache::new(adaptive));
                let (paths, bg) = cache.get_paths(fb, sw, sh);
                let (edges, row_ranges, edge_indices, ow, oh) =
                    vectorize::rasterize::prepare_gpu_edges_v2(paths, bg, scale, sw, sh);
                if ow > 0 && oh > 0 && !edges.is_empty() {
                    if self.wgpu_shared_chain.is_none() {
                        self.wgpu_shared_chain = Some(
                            scaling::wgpu_vectorize::WgpuSharedChainRasterizer::new(&gpu.device)
                        );
                    }
                    let rasterizer = self.wgpu_shared_chain.as_ref().unwrap();
                    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("vectorize_shared_chain"),
                    });
                    let out_tex = rasterizer.encode(
                        &gpu.device, &gpu.queue, &mut encoder,
                        &edges, &row_ranges, &edge_indices, ow, oh, bg,
                    );
                    let frame = match gpu.surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                        _ => return,
                    };
                    let surface_view = frame.texture.create_view(&Default::default());
                    gpu.encode_blit(&mut encoder, &out_tex, &surface_view, sw as u32, sh as u32);
                    gpu.queue.submit(std::iter::once(encoder.finish()));
                    frame.present();
                    self.frame_start = Instant::now();
                    return;
                }
                // Fallback if edge prep failed
                (fb, sw, sh)
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
                    wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                    _ => return,
                };
                let fb_view = frame.texture.create_view(&Default::default());
                gpu.encode_blit(&mut encoder, out_tex, &fb_view, self.src_w, self.src_h);
                gpu.queue.submit(std::iter::once(encoder.finish()));
                frame.present();
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
                    ((sw as f64 * s).round() as u32, (sh as f64 * s).round() as u32)
                };
                let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("scale+blit"),
                });
                let pipeline = self.wgpu_scale.as_mut().unwrap();
                let out_tex = pipeline.encode(
                    &gpu.device, &gpu.queue, &mut encoder,
                    wgpu_filter, fb, sw as u32, sh as u32, ow, oh,
                );
                let frame = match gpu.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                    _ => return,
                };
                let fb_view = frame.texture.create_view(&Default::default());
                gpu.encode_blit(&mut encoder, out_tex, &fb_view, self.src_w, self.src_h);
                gpu.queue.submit(std::iter::once(encoder.finish()));
                frame.present();
                return;
            } else if let Some((s, w, h)) = scaling::cpu_scale(self.scale_filter, fb, sw, sh, disp_w, disp_h) {
                scaled = s;
                (&scaled, w as usize, h as usize)
            } else {
                (fb, sw, sh)
            };

        gpu.render(frame_pixels, frame_w as u32, frame_h as u32, self.src_w, self.src_h);

        // FPS counter
        let emu_time = self.frame_start.elapsed();
        self.fps.update(1, emu_time);

        // Periodic save RAM flush
        if let (Some(flusher), Some(emu)) = (&mut self.sav_flusher, &self.emu) {
            flusher.poll(emu);
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
        let (menu, filter_items, printer_item, model_items) = build_menu(self.cli.printer);
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
                if let (Some(flusher), Some(emu)) = (&mut self.sav_flusher, &self.emu) {
                    flusher.flush(emu);
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
                        if key == KeyCode::Tab {
                            self.fast_forward = pressed;
                        }
                    }

                    if pressed {
                        match key {
                            KeyCode::Escape => {
                                if let (Some(emu), Some(path)) = (&self.emu, &self.rom_path) {
                                    ui_util::flush_sav(emu, path);
                                }
                                event_loop.exit();
                            }
                            KeyCode::F5 => {
                                if let (Some(emu), Some(rp)) = (&mut self.emu, &self.rom_path) {
                                    ui_util::save_state_to_slot(emu, rp, self.current_slot);
                                }
                            }
                            KeyCode::F7 => {
                                if let (Some(emu), Some(rp)) = (&mut self.emu, &self.rom_path) {
                                    ui_util::load_state_from_slot(emu, rp, self.current_slot);
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
        if let Some(ref mut gp) = self.gamepad {
            let gs = gp.poll();
            self.gp_buttons = gs.buttons;

            // Apply combined state (keyboard | gamepad)
            if let Some(ref mut emu) = self.emu {
                let combined = self.kb_buttons | self.gp_buttons;
                for bit in 0..8u8 {
                    let mask = 1 << bit;
                    emu.set_button(mask, combined & mask != 0);
                }
                if gs.rewind { emu.set_rewinding(true); }
                self.fast_forward = self.fast_forward || gs.fast_forward;

                // Rumble
                if emu.has_rumble() {
                    gp.ensure_rumble();
                    gp.set_rumble(emu.drain_rumble());
                }
            }
        } else {
            self.gp_buttons = 0;
        }

        // Handle rewind (3x speed) with reverse audio
        if let Some(ref mut emu) = self.emu {
            if emu.is_rewinding() {
                let mut all_audio = Vec::new();
                for _ in 0..3 {
                    emu.rewind_one_frame();
                    all_audio.extend_from_slice(&emu.drain_audio_samples());
                }
                ui_util::reverse_audio(&mut all_audio);
                let resampled = ui_util::downsample_audio(&all_audio, 3);
                if !resampled.is_empty() {
                    let mut ring = self.audio_ring.lock().unwrap();
                    ring.push(&resampled);
                }
            }
        }

        // Step emulation
        self.step_and_render();

        // Update rumble after emulation step

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

fn map_scale_filter(filter: scaling::ScaleFilter) -> Option<scaling::wgpu_scale::WgpuScaleFilter> {
    use scaling::ScaleFilter as SF;
    use scaling::wgpu_scale::WgpuScaleFilter as WF;
    match filter {
        SF::Nearest => Some(WF::Nearest),
        SF::Bilinear => Some(WF::Bilinear),
        SF::Bicubic => Some(WF::Bicubic),
        SF::Epx | SF::Scale2x | SF::Scale4x => Some(WF::Epx),
        SF::Scale3x => Some(WF::Scale3x),
        SF::Eagle => Some(WF::Eagle),
        SF::Hqx(_) => Some(WF::Hqx),
        SF::Xbr(_) | SF::SuperXbr => Some(WF::Xbr),
        SF::Xbrz(_) => Some(WF::Xbrz),
        SF::AaNearestNeighbor => Some(WF::AaNearest),
        SF::OmniScale => Some(WF::OmniScale),
        SF::OmniScaleLegacy => Some(WF::OmniScaleLegacy),
        SF::Edi => Some(WF::Edi),
        SF::Nedi => Some(WF::Nedi),
        SF::Dcci => Some(WF::Dcci),
        SF::Mmpx => Some(WF::Mmpx),
        SF::LcdGrid => Some(WF::LcdGrid),
        _ => None,
    }
}
