use vibeboy::*;

mod audio;
#[cfg(target_os = "linux")]
mod compute;
mod gpu;

use clap::Parser;
use gtk4::prelude::*;
use gtk4::glib;
use model::GbModel;
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

pub(crate) const SCALE: u32 = 3;
pub(crate) const GB_W: u32 = 160;
pub(crate) const GB_H: u32 = 144;
pub(crate) const SGB_W: u32 = 256;
pub(crate) const SGB_H: u32 = 224;
pub(crate) const AUDIO_SAMPLE_RATE: u32 = 96_000;

#[derive(Parser)]
#[command(name = "vibeboy", about = "Game Boy / Game Boy Color emulator (GTK4 frontend)")]
pub(crate) struct Cli {
    /// Path to ROM file (.gb / .gbc). If omitted, a file dialog will open.
    pub rom: Option<PathBuf>,

    /// Path to boot ROM file (auto-detected if not specified)
    #[arg(long)]
    pub bootrom: Option<PathBuf>,

    /// Hardware model (auto-detected from cart header if not specified)
    #[arg(long, value_parser = |s: &str| s.parse::<GbModel>())]
    pub model: Option<GbModel>,

    /// Skip boot ROM
    #[arg(long)]
    pub no_boot: bool,

    /// Scaling filter
    #[arg(long, value_parser = ui_util::parse_filter)]
    pub filter: Option<String>,

    /// Connect a Game Boy Printer (saves PNGs to prints/ directory)
    #[arg(long)]
    pub printer: bool,
}

struct EmuState {
    emu: emulator::Emulator,
    model: GbModel,
    src_w: u32,
    src_h: u32,
    paused: bool,
    fast_forward: bool,
    slow_motion: bool,
    step_one_frame: bool,
    current_slot: usize,
    force_cpu: bool,
    scale_filter: scaling::ScaleFilter,
    kb_buttons: u8,
    gp_buttons: u8,
    rgba_buf: Vec<u8>,
    scaled_buf: Vec<u32>,
    rom_path: PathBuf,
    rom_data: std::sync::Arc<[u8]>,
    audio_ring: std::sync::Arc<std::sync::Mutex<audio::AudioRing>>,
    _audio_stream: Option<cpal::Stream>,
    vec_cache: Option<vectorize::VectorizeCache>,
    frame_timer: Option<glib::SourceId>,
    fps: ui_util::FpsCounter,
    gamepad: Option<ui_util::GamepadPoller>,
    model_override: Option<GbModel>,
    slow_tick: u32,
    sav_flusher: ui_util::SavFlusher,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    let app = gtk4::Application::builder()
        .application_id("com.vibeboy.gtk")
        .build();

    let cli = RefCell::new(Some(cli));

    app.connect_activate(move |app| {
        let cli = cli.borrow_mut().take().unwrap_or_else(|| Cli::parse());
        build_ui(app, cli);
    });

    app.run_with_args::<String>(&[]);
}

fn load_boot_rom(model: GbModel, cli: &Cli) -> Option<Vec<u8>> {
    // Always auto-detect boot ROM by model; explicit --bootrom path may be
    // for a different model than the one being loaded.
    ui_util::load_boot_rom(model, None, cli.no_boot)
}

fn create_emu_state(
    rom: std::sync::Arc<[u8]>,
    rom_path: PathBuf,
    cli: &Cli,
    initial_filter: scaling::ScaleFilter,
    model_override: Option<GbModel>,
) -> EmuState {
    let model = model_override
        .or(cli.model)
        .unwrap_or_else(|| ui_util::auto_detect_model(&rom));
    let boot_rom = load_boot_rom(model, cli);

    ui_util::print_controls();
    if initial_filter != scaling::ScaleFilter::Nearest {
        eprintln!("  Filter: {:?}", initial_filter);
    }
    eprintln!();

    let mut emu = emulator::Emulator::new(rom.clone(), boot_rom, model, None, clock::default_clock(), AUDIO_SAMPLE_RATE);
    ui_util::load_sav(&mut emu, &rom_path);

    if cli.printer {
        emu.attach_serial_device(Box::new(printer::Printer::new(model.cpu_clock_rate())));
        eprintln!("Game Boy Printer connected — images will be saved to prints/");
    }

    let is_sgb = emu.is_sgb();
    let src_w = if is_sgb { SGB_W } else { GB_W };
    let src_h = if is_sgb { SGB_H } else { GB_H };

    let audio_ring = std::sync::Arc::new(std::sync::Mutex::new(
        audio::AudioRing::new(AUDIO_SAMPLE_RATE as usize / 60 * 4 * 2, AUDIO_SAMPLE_RATE),
    ));
    let (_audio_stream, actual_rate) = match audio::start_audio(std::sync::Arc::clone(&audio_ring)) {
        Some((s, r)) => (Some(s), r),
        None => (None, AUDIO_SAMPLE_RATE),
    };
    audio_ring.lock().unwrap().downsample_ratio = (AUDIO_SAMPLE_RATE / actual_rate).max(1) as usize;

    let sav_flusher = ui_util::SavFlusher::new(&emu, &rom_path);

    EmuState {
        emu,
        model,
        src_w,
        src_h,
        paused: false,
        fast_forward: false,
        slow_motion: false,
        step_one_frame: false,
        current_slot: 0,
        force_cpu: false,
        scale_filter: initial_filter,
        kb_buttons: 0,
        gp_buttons: 0,
        rgba_buf: vec![0u8; (src_w * src_h * 4) as usize],
        scaled_buf: Vec::new(),
        rom_path,
        rom_data: rom,
        audio_ring,
        _audio_stream,
        vec_cache: None,
        frame_timer: None,
        fps: ui_util::FpsCounter::new(),
        gamepad: ui_util::GamepadPoller::new(),
        model_override,
        slow_tick: 0,
        sav_flusher,
    }
}

fn build_ui(app: &gtk4::Application, cli: Cli) {
    let initial_filter = cli.filter.as_ref().and_then(|name| {
        scaling::ScaleFilter::from_name(name)
    }).unwrap_or(scaling::ScaleFilter::Nearest);

    // Create window
    let window = gtk4::ApplicationWindow::builder()
        .application(app)
        .title("VibeBoy")
        .default_width((GB_W * SCALE) as i32)
        .default_height((GB_H * SCALE) as i32)
        .build();

    // Drawing area for rendering frames
    let drawing_area = gtk4::DrawingArea::new();
    drawing_area.set_hexpand(true);
    drawing_area.set_vexpand(true);

    // Menu bar
    let menu = gtk4::gio::Menu::new();
    let file_menu = gtk4::gio::Menu::new();
    file_menu.append(Some("Open ROM..."), Some("app.open"));
    file_menu.append(Some("Quit"), Some("app.quit"));
    menu.append_submenu(Some("File"), &file_menu);

    let emu_menu = gtk4::gio::Menu::new();
    emu_menu.append(Some("Pause"), Some("app.pause"));
    emu_menu.append(Some("Reset"), Some("app.reset"));

    let peripheral_section = gtk4::gio::Menu::new();
    peripheral_section.append(Some("Game Boy Printer"), Some("app.toggle-printer"));
    emu_menu.append_section(None, &peripheral_section);

    // Save State submenu with slots
    let save_submenu = gtk4::gio::Menu::new();
    for &slot in &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] {
        save_submenu.append(
            Some(&format!("Slot {}", slot)),
            Some(&format!("app.save-slot::{}", slot)),
        );
    }
    // Load State submenu with slots
    let load_submenu = gtk4::gio::Menu::new();
    for &slot in &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] {
        load_submenu.append(
            Some(&format!("Slot {}", slot)),
            Some(&format!("app.load-slot::{}", slot)),
        );
    }
    let state_section = gtk4::gio::Menu::new();
    state_section.append_submenu(Some("Save State"), &save_submenu);
    state_section.append_submenu(Some("Load State"), &load_submenu);
    emu_menu.append_section(None, &state_section);

    // Slot selection radio items
    let slot_section = gtk4::gio::Menu::new();
    for &slot in &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] {
        slot_section.append(
            Some(&format!("Slot {}", slot)),
            Some(&format!("app.select-slot::{}", slot)),
        );
    }
    emu_menu.append_section(None, &slot_section);

    // Hardware model submenu
    let model_submenu = gtk4::gio::Menu::new();
    model_submenu.append(Some("Auto"), Some("app.model::auto"));
    for (name, id) in [
        ("DMG0", "dmg0"), ("DMG", "dmg"), ("MGB", "mgb"),
        ("SGB", "sgb"), ("SGB2", "sgb2"),
        ("CGB", "cgb"), ("AGB", "agb"),
    ] {
        model_submenu.append(Some(name), Some(&format!("app.model::{}", id)));
    }
    let model_section = gtk4::gio::Menu::new();
    model_section.append_submenu(Some("Hardware"), &model_submenu);
    emu_menu.append_section(None, &model_section);

    menu.append_submenu(Some("Emulation"), &emu_menu);

    // Filter submenu (grouped like Cocoa/Winit: HQx, xBR, xBRZ, Edge submenus)
    let filter_menu = gtk4::gio::Menu::new();

    // Force CPU toggle at top of filter menu
    let force_cpu_section = gtk4::gio::Menu::new();
    force_cpu_section.append(Some("Force CPU"), Some("app.force-cpu"));
    filter_menu.append_section(None, &force_cpu_section);

    let mut sub_menus = std::collections::BTreeMap::new();

    for (display_name, filter) in scaling::ScaleFilter::menu_entries() {
        let action_name = format!("app.filter::{}", filter.cli_name());
        let group = filter.menu_group();
        if group == scaling::FilterMenuGroup::Main {
            filter_menu.append(Some(display_name), Some(&action_name));
        } else {
            sub_menus.entry(group.label())
                .or_insert_with(gtk4::gio::Menu::new)
                .append(Some(display_name), Some(&action_name));
        }
    }
    let sub_section = gtk4::gio::Menu::new();
    for (label, submenu) in &sub_menus {
        sub_section.append_submenu(Some(label), submenu);
    }
    filter_menu.append_section(None, &sub_section);
    menu.append_submenu(Some("Filter"), &filter_menu);

    app.set_menubar(Some(&menu));
    window.set_show_menubar(true);

    // Keyboard accelerators
    app.set_accels_for_action("app.open", &["<Control>o"]);
    app.set_accels_for_action("app.quit", &["<Control>q"]);
    app.set_accels_for_action("app.pause", &["F6"]);

    // GL area for GPU-accelerated rendering
    let gl_area = gtk4::GLArea::new();
    gl_area.set_required_version(3, 3);
    gl_area.set_auto_render(false);
    gl_area.set_hexpand(true);
    gl_area.set_vexpand(true);

    // Stack: GL area (preferred) with Cairo DrawingArea fallback
    let stack = gtk4::Stack::new();
    stack.add_named(&drawing_area, Some("cairo"));
    stack.add_named(&gl_area, Some("gl"));
    stack.set_visible_child_name("cairo");

    // Layout
    let vbox = gtk4::Box::new(gtk4::Orientation::Vertical, 0);
    vbox.append(&stack);
    window.set_child(Some(&vbox));

    // Shared emulator state (None until a ROM is loaded)
    let state: Rc<RefCell<Option<EmuState>>> = Rc::new(RefCell::new(None));

    // GL renderer state + GPU compute + pending frame for GLArea render signal
    let gl_renderer: Rc<RefCell<Option<gpu::GlRenderer>>> = Rc::new(RefCell::new(None));
    #[cfg(target_os = "linux")]
    let gpu_compute: Rc<RefCell<Option<compute::GpuCompute>>> = Rc::new(RefCell::new(None));
    #[cfg(not(target_os = "linux"))]
    let gpu_compute: Rc<RefCell<Option<()>>> = Rc::new(RefCell::new(None));
    let pending_frame: Rc<RefCell<gpu::PendingFrame>> = Rc::new(RefCell::new(gpu::PendingFrame::default()));

    // GLArea realize: init GL resources + wgpu compute (same GL context)
    gl_area.connect_realize({
        let gl_renderer = Rc::clone(&gl_renderer);
        let gpu_compute = Rc::clone(&gpu_compute);
        let stack = stack.clone();
        move |area| {
            area.make_current();
            if area.error().is_some() {
                eprintln!("GLArea error, falling back to Cairo");
                return;
            }
            match gpu::GlRenderer::new() {
                Some(r) => {
                    *gl_renderer.borrow_mut() = Some(r);
                    stack.set_visible_child_name("gl");
                    // Init wgpu compute using the same GL context (zero-copy)
                    #[cfg(target_os = "linux")]
                    {
                        match compute::GpuCompute::new(|s| gpu::gl_proc_address(s)) {
                            Some(c) => *gpu_compute.borrow_mut() = Some(c),
                            None => eprintln!("GPU compute init failed, will use CPU scaling"),
                        }
                    }
                }
                None => eprintln!("GL renderer init failed, falling back to Cairo"),
            }
        }
    });

    // GLArea unrealize: drop GL resources while context is current
    gl_area.connect_unrealize({
        let gl_renderer = Rc::clone(&gl_renderer);
        let gpu_compute = Rc::clone(&gpu_compute);
        move |area| {
            area.make_current();
            gpu_compute.borrow_mut().take();
            gl_renderer.borrow_mut().take();
        }
    });

    // GLArea render signal: GPU compute + blit, or CPU pixel upload + blit
    gl_area.connect_render({
        let gl_renderer = Rc::clone(&gl_renderer);
        let gpu_compute = Rc::clone(&gpu_compute);
        let pending = Rc::clone(&pending_frame);
        move |area, _ctx| {
            let mut r = gl_renderer.borrow_mut();
            let f = pending.borrow();
            if let Some(ref mut renderer) = *r {
                let has_frame = !f.pixels.is_empty() || f.gl_texture.is_some();
                if has_frame {
                    let scale = area.scale_factor();
                    let vp_w = area.width() * scale;
                    let vp_h = area.height() * scale;

                    // Pre-rendered GL texture (shared-chain GPU rasterizer)
                    if let Some(gl_tex) = f.gl_texture {
                        renderer.render_gl_texture(gl_tex, vp_w, vp_h, f.src_w, f.src_h);
                        return glib::Propagation::Stop;
                    }

                    // Try GPU compute path (zero-copy: compute → GL texture → blit)
                    #[cfg(target_os = "linux")]
                    if let Some(wgpu_filter) = f.gpu_filter {
                        let mut gc = gpu_compute.borrow_mut();
                        if let Some(ref mut compute) = *gc {
                            if let Some((gl_tex, ow, oh)) = compute.scale(
                                wgpu_filter,
                                &f.pixels,
                                f.frame_w,
                                f.frame_h,
                                f.fit_w,
                                f.fit_h,
                                f.factor,
                            ) {
                                renderer.render_gl_texture(gl_tex, vp_w, vp_h, f.src_w, f.src_h);
                                return glib::Propagation::Stop;
                            }
                        }
                    }

                    // CPU pixel upload path
                    renderer.render(
                        &f.pixels,
                        f.frame_w,
                        f.frame_h,
                        vp_w,
                        vp_h,
                        f.src_w,
                        f.src_h,
                    );
                }
            }
            glib::Propagation::Stop
        }
    });

    // Set up draw function
    let state_draw = Rc::clone(&state);
    drawing_area.set_draw_func(move |_da, cr, width, height| {
        let st = state_draw.borrow();
        let st = match st.as_ref() {
            Some(s) => s,
            None => {
                // No ROM loaded — draw black
                cr.set_source_rgb(0.0, 0.0, 0.0);
                let _ = cr.paint();
                return;
            }
        };
        let rgba = &st.rgba_buf;
        let fw = st.src_w as i32;
        let fh = st.src_h as i32;

        if rgba.len() < (fw * fh * 4) as usize {
            return;
        }

        let stride = cairo::Format::ARgb32.stride_for_width(fw as u32).unwrap();
        let surface = unsafe {
            cairo::ImageSurface::create_for_data_unsafe(
                rgba.as_ptr() as *mut u8,
                cairo::Format::ARgb32,
                fw,
                fh,
                stride,
            )
        };

        if let Ok(surface) = surface {
            let scale_x = width as f64 / fw as f64;
            let scale_y = height as f64 / fh as f64;
            let scale = scale_x.min(scale_y);

            let offset_x = (width as f64 - fw as f64 * scale) / 2.0;
            let offset_y = (height as f64 - fh as f64 * scale) / 2.0;

            cr.set_source_rgb(0.0, 0.0, 0.0);
            let _ = cr.paint();

            cr.translate(offset_x, offset_y);
            cr.scale(scale, scale);

            cr.set_source_surface(&surface, 0.0, 0.0).unwrap();
            cr.source().set_filter(cairo::Filter::Nearest);
            let _ = cr.paint();
        }
    });

    // Helper: start the frame timer for emulation
    let da_for_timer = drawing_area.clone();
    let gl_area_for_timer = gl_area.clone();
    let start_frame_timer = {
        let state = Rc::clone(&state);
        let gl_renderer = Rc::clone(&gl_renderer);
        let gpu_compute = Rc::clone(&gpu_compute);
        let pending_frame = Rc::clone(&pending_frame);
        move || {
            let state_tick = Rc::clone(&state);
            let da = da_for_timer.clone();
            let gl_area = gl_area_for_timer.clone();
            let gl_renderer = Rc::clone(&gl_renderer);
            let gpu_compute = Rc::clone(&gpu_compute);
            let pending_frame = Rc::clone(&pending_frame);

            // Get frame duration from the loaded emulator's model
            let interval_ms = {
                let st = state_tick.borrow();
                let st = st.as_ref().unwrap();
                let frame_dur = ui_util::frame_duration(st.model);
                frame_dur.as_millis().max(1) as u64
            };

            let source_id = glib::timeout_add_local(
                std::time::Duration::from_millis(interval_ms),
                move || {
                    {
                        let mut st = state_tick.borrow_mut();
                        let st = match st.as_mut() {
                            Some(s) => s,
                            None => return glib::ControlFlow::Break,
                        };

                        // Poll gamepad
                        if let Some(ref mut gp) = st.gamepad {
                            let gs = gp.poll();
                            // Merge gamepad buttons with keyboard
                            let old = st.gp_buttons;
                            st.gp_buttons = gs.buttons;
                            let pressed = gs.buttons & !old;
                            let released = old & !gs.buttons;
                            for bit in 0..8u8 {
                                let mask = 1 << bit;
                                if pressed & mask != 0 { st.emu.set_button(mask, true); }
                                if released & mask != 0 && st.kb_buttons & mask == 0 {
                                    st.emu.set_button(mask, false);
                                }
                            }
                            if gs.rewind { st.emu.set_rewinding(true); }
                            st.fast_forward = st.fast_forward || gs.fast_forward;
                            // Rumble
                            if st.emu.has_rumble() {
                                gp.ensure_rumble();
                                gp.set_rumble(st.emu.drain_rumble());
                            }
                        }

                        if st.emu.is_rewinding() {
                            let mut all_audio = Vec::with_capacity(19200);
                            for _ in 0..3 {
                                st.emu.rewind_one_frame();
                                all_audio.extend_from_slice(&st.emu.drain_audio_samples());
                            }
                            ui_util::reverse_audio(&mut all_audio);
                            let resampled = ui_util::downsample_audio(&all_audio, 3);
                            if !resampled.is_empty() {
                                let mut ring = st.audio_ring.lock().unwrap();
                                ring.push(&resampled);
                            }
                        }

                        if st.paused && !st.step_one_frame {
                            // Don't step emulation
                        } else if st.step_one_frame {
                            st.emu.step_frame();
                            st.step_one_frame = false;
                            let samples = st.emu.drain_audio_samples();
                            if !samples.is_empty() {
                                let mut ring = st.audio_ring.lock().unwrap();
                                ring.push(&samples);
                            }
                        } else if !st.paused {
                            let emu_start = Instant::now();
                            let frames_stepped;
                            if st.fast_forward {
                                for _ in 0..4 {
                                    st.emu.step_frame();
                                }
                                frames_stepped = 4u32;
                                let samples = st.emu.drain_audio_samples();
                                if !samples.is_empty() {
                                    let resampled = ui_util::downsample_audio(&samples, 4);
                                    let mut ring = st.audio_ring.lock().unwrap();
                                    ring.push(&resampled);
                                }
                            } else if st.slow_motion {
                                // Half speed: step every other frame
                                st.slow_tick += 1;
                                if st.slow_tick % 2 == 0 {
                                    st.emu.step_frame();
                                    frames_stepped = 1;
                                    let samples = st.emu.drain_audio_samples();
                                    if !samples.is_empty() {
                                        let mut ring = st.audio_ring.lock().unwrap();
                                        ring.push(&samples);
                                    }
                                } else {
                                    frames_stepped = 0;
                                }
                            } else {
                                st.emu.step_frame();
                                frames_stepped = 1;
                                let samples = st.emu.drain_audio_samples();
                                if !samples.is_empty() {
                                    let mut ring = st.audio_ring.lock().unwrap();
                                    ring.push(&samples);
                                }
                            }
                            let emu_elapsed = emu_start.elapsed();

                            // FPS counter
                            st.fps.update(frames_stepped, emu_elapsed);

                            // Periodic save RAM flush
                            st.sav_flusher.poll(&st.emu);

                            // Printer
                            ui_util::check_and_save_prints(&mut st.emu);

                            // Get frame buffer and dimensions
                            let is_sgb = st.emu.is_sgb();
                            let base_w = if is_sgb { SGB_W as usize } else { GB_W as usize };
                            let base_h = if is_sgb { SGB_H as usize } else { GB_H as usize };
                            let fb: &[u32] = if is_sgb {
                                st.emu.sgb_composited_frame()
                            } else {
                                st.emu.frame_buffer()
                            };
                            let da_w = da.width().max(1) as usize;
                            let da_h = da.height().max(1) as usize;

                            // Compute aspect-ratio-correct dimensions for filters
                            let scale_fit = (da_w as f64 / base_w as f64)
                                .min(da_h as f64 / base_h as f64)
                                .max(1.0);
                            let fit_w = (base_w as f64 * scale_fit).round() as usize;
                            let fit_h = (base_h as f64 * scale_fit).round() as usize;

                            // Check if we can use GPU compute for this filter
                            #[cfg(target_os = "linux")]
                            let wgpu_filter = if st.force_cpu { None } else { compute::to_wgpu_filter(st.scale_filter) };
                            #[cfg(not(target_os = "linux"))]
                            let wgpu_filter: Option<scaling::wgpu_scale::WgpuScaleFilter> = None;
                            let use_gpu = !st.force_cpu && wgpu_filter.is_some() && gpu_compute.borrow().is_some();

                            // Apply scaling filter: GPU deferred to render callback, else CPU
                            let (pixels, pw, ph, gpu_filter_for_render): (&[u32], usize, usize, Option<scaling::wgpu_scale::WgpuScaleFilter>) =
                                if use_gpu {
                                    // Send raw pixels to render callback for zero-copy GPU compute
                                    (fb, base_w, base_h, wgpu_filter)
                                } else if st.scale_filter == scaling::ScaleFilter::Nearest {
                                    (fb, base_w, base_h, None)
                                } else if !st.force_cpu && st.scale_filter == scaling::ScaleFilter::VectorizeGpu {
                                    // Full 6-stage GPU vectorize pipeline
                                    #[cfg(target_os = "linux")]
                                    {
                                        let s = scale_fit as f32;
                                        let ow = (base_w as f32 * s).round() as u32;
                                        let oh = (base_h as f32 * s).round() as u32;
                                        let mut gc = gpu_compute.borrow_mut();
                                        if let Some(ref mut compute) = *gc {
                                            if let Some((gl_tex, gw, gh)) =
                                                compute.vectorize_gpu(fb, base_w as u32, base_h as u32, ow, oh, s)
                                            {
                                                let mut pf = pending_frame.borrow_mut();
                                                pf.pixels.clear();
                                                pf.frame_w = gw;
                                                pf.frame_h = gh;
                                                pf.src_w = base_w as u32;
                                                pf.src_h = base_h as u32;
                                                pf.gpu_filter = None;
                                                pf.gl_texture = Some(gl_tex);
                                                pf.fit_w = fit_w as u32;
                                                pf.fit_h = fit_h as u32;
                                                pf.factor = 0;
                                                drop(gc);
                                                drop(pf);
                                                gl_area.queue_render();
                                                return glib::ControlFlow::Continue;
                                            }
                                        }
                                    }
                                    // CPU fallback
                                    if let Some((scaled, w, h)) = scaling::cpu_scale(
                                        st.scale_filter, fb, base_w, base_h, fit_w, fit_h,
                                    ) {
                                        st.scaled_buf = scaled;
                                        (&st.scaled_buf, w as usize, h as usize, None)
                                    } else {
                                        (fb, base_w, base_h, None)
                                    }
                                } else if matches!(
                                    st.scale_filter,
                                    scaling::ScaleFilter::VectorizeLegacy
                                        | scaling::ScaleFilter::VectorizeLegacyAdaptive
                                        | scaling::ScaleFilter::Vectorize
                                        | scaling::ScaleFilter::VectorizeAdaptive
                                ) {
                                    // All vectorize variants: GPU rasterizer on Linux, CPU fallback on macOS
                                    let adaptive = matches!(
                                        st.scale_filter,
                                        scaling::ScaleFilter::VectorizeAdaptive
                                            | scaling::ScaleFilter::VectorizeLegacyAdaptive
                                    );
                                    let is_legacy = matches!(
                                        st.scale_filter,
                                        scaling::ScaleFilter::VectorizeLegacy
                                            | scaling::ScaleFilter::VectorizeLegacyAdaptive
                                    );
                                    let cache = st.vec_cache.get_or_insert_with(|| {
                                        if is_legacy { vectorize::VectorizeCache::new_legacy(adaptive) }
                                        else { vectorize::VectorizeCache::new(adaptive) }
                                    });
                                    #[cfg(target_os = "linux")]
                                    if !st.force_cpu {
                                        let (paths, bg) = cache.get_paths(fb, base_w, base_h);
                                        let (edges, row_ranges, edge_indices, ow, oh) =
                                            vectorize::rasterize::prepare_gpu_edges_v2(
                                                paths, bg, scale_fit, base_w, base_h,
                                            );
                                        let mut gc = gpu_compute.borrow_mut();
                                        if ow > 0 && oh > 0 && !edges.is_empty() {
                                            if let Some(ref mut compute) = *gc {
                                                if let Some((gl_tex, gw, gh)) =
                                                    compute.rasterize_shared_chain(
                                                        &edges, &row_ranges, &edge_indices,
                                                        ow, oh, bg,
                                                    )
                                                {
                                                    let mut pf = pending_frame.borrow_mut();
                                                    pf.pixels.clear(); // signal: use gl_tex instead
                                                    pf.frame_w = gw;
                                                    pf.frame_h = gh;
                                                    pf.src_w = base_w as u32;
                                                    pf.src_h = base_h as u32;
                                                    pf.gpu_filter = None;
                                                    pf.gl_texture = Some(gl_tex);
                                                    pf.fit_w = fit_w as u32;
                                                    pf.fit_h = fit_h as u32;
                                                    drop(gc);
                                                    drop(pf);
                                                    gl_area.queue_render();
                                                    return glib::ControlFlow::Continue;
                                                }
                                            }
                                        }
                                    }
                                    // CPU fallback (macOS or GPU unavailable): use legacy rasterizer
                                    let cache = st.vec_cache.get_or_insert_with(|| {
                                        vectorize::VectorizeCache::new_legacy(adaptive)
                                    });
                                    let (raster, vw, vh) =
                                        cache.rasterize(fb, base_w, base_h, scale_fit);
                                    (raster, vw, vh, None)
                                } else if st.scale_filter == scaling::ScaleFilter::VectorizeDiffusion {
                                    let sc = scale_fit.round().max(1.0) as u32;
                                    let ow = base_w as u32 * sc;
                                    let oh = base_h as u32 * sc;
                                    #[cfg(target_os = "linux")]
                                    if !st.force_cpu {
                                        let mut gc = gpu_compute.borrow_mut();
                                        if let Some(ref mut compute) = *gc {
                                            if let Some((gl_tex, gw, gh)) =
                                                compute.diffusion_rasterize(
                                                    fb, base_w as u32, base_h as u32, ow, oh, sc,
                                                )
                                            {
                                                let mut pf = pending_frame.borrow_mut();
                                                pf.pixels.clear();
                                                pf.frame_w = gw;
                                                pf.frame_h = gh;
                                                pf.src_w = base_w as u32;
                                                pf.src_h = base_h as u32;
                                                pf.gpu_filter = None;
                                                pf.gl_texture = Some(gl_tex);
                                                pf.fit_w = fit_w as u32;
                                                pf.fit_h = fit_h as u32;
                                                drop(gc);
                                                drop(pf);
                                                gl_area.queue_render();
                                                return glib::ControlFlow::Continue;
                                            }
                                        }
                                    }
                                    // CPU fallback
                                    let (buf, dw, dh) = vectorize::rasterize::rasterize_diffusion(
                                        fb, base_w, base_h, sc as usize,
                                    );
                                    st.scaled_buf = buf;
                                    (&st.scaled_buf, dw, dh, None)
                                } else if matches!(
                                    st.scale_filter,
                                    scaling::ScaleFilter::VectorizeSplineDiffusion
                                        | scaling::ScaleFilter::VectorizeSplineDiffusionAdaptive
                                ) {
                                    let sc = scale_fit.round().max(1.0) as u32;
                                    let adaptive = st.scale_filter == scaling::ScaleFilter::VectorizeSplineDiffusionAdaptive;
                                    let cache = st.vec_cache.get_or_insert_with(|| {
                                        vectorize::VectorizeCache::new(adaptive)
                                    });
                                    let (paths, bg) = cache.get_paths(fb, base_w, base_h);
                                    let (edges, row_ranges, edge_indices, ow, oh) =
                                        vectorize::rasterize::prepare_gpu_edges_v2(
                                            paths, bg, scale_fit, base_w, base_h,
                                        );
                                    #[cfg(target_os = "linux")]
                                    if !st.force_cpu {
                                        if ow > 0 && oh > 0 && !edges.is_empty() {
                                            let mut gc = gpu_compute.borrow_mut();
                                            if let Some(ref mut compute) = *gc {
                                                if let Some((gl_tex, gw, gh)) =
                                                    compute.spline_diffusion(
                                                        &edges, &row_ranges, &edge_indices,
                                                        fb, base_w as u32, base_h as u32,
                                                        ow, oh, bg, sc,
                                                    )
                                                {
                                                    let mut pf = pending_frame.borrow_mut();
                                                    pf.pixels.clear();
                                                    pf.frame_w = gw;
                                                    pf.frame_h = gh;
                                                    pf.src_w = base_w as u32;
                                                    pf.src_h = base_h as u32;
                                                    pf.gpu_filter = None;
                                                    pf.gl_texture = Some(gl_tex);
                                                    pf.fit_w = fit_w as u32;
                                                    pf.fit_h = fit_h as u32;
                                                    drop(gc);
                                                    drop(pf);
                                                    gl_area.queue_render();
                                                    return glib::ControlFlow::Continue;
                                                }
                                            }
                                        }
                                    }
                                    // CPU fallback
                                    let cache = st.vec_cache.get_or_insert_with(|| {
                                        vectorize::VectorizeCache::new(adaptive)
                                    });
                                    let (paths, bg) = cache.get_paths(fb, base_w, base_h);
                                    let (buf, dw, dh) = vectorize::rasterize::rasterize_spline_diffusion(
                                        paths, fb, base_w, base_h, bg, sc as usize,
                                    );
                                    st.scaled_buf = buf;
                                    (&st.scaled_buf, dw, dh, None)
                                } else if let Some((scaled, w, h)) = scaling::cpu_scale(
                                    st.scale_filter,
                                    fb,
                                    base_w,
                                    base_h,
                                    fit_w,
                                    fit_h,
                                ) {
                                    st.scaled_buf = scaled;
                                    (&st.scaled_buf, w as usize, h as usize, None)
                                } else {
                                    (fb, base_w, base_h, None)
                                };

                            // Render: GL path or Cairo fallback
                            if gl_renderer.borrow().is_some() {
                                let mut pf = pending_frame.borrow_mut();
                                pf.pixels.clear();
                                pf.pixels.extend_from_slice(pixels);
                                pf.frame_w = pw as u32;
                                pf.frame_h = ph as u32;
                                pf.src_w = base_w as u32;
                                pf.src_h = base_h as u32;
                                pf.gpu_filter = gpu_filter_for_render;
                                pf.fit_w = fit_w as u32;
                                pf.fit_h = fit_h as u32;
                                pf.factor = st.scale_filter.factor();
                                drop(pf);
                                gl_area.queue_render();
                            } else {
                                // Cairo fallback
                                st.src_w = pw as u32;
                                st.src_h = ph as u32;
                                let needed = pw * ph * 4;
                                if st.rgba_buf.len() < needed {
                                    st.rgba_buf.resize(needed, 0);
                                }
                                for i in 0..pw * ph {
                                    let c = pixels[i];
                                    let r = (c >> 16) & 0xFF;
                                    let g = (c >> 8) & 0xFF;
                                    let b = c & 0xFF;
                                    let offset = i * 4;
                                    st.rgba_buf[offset] = b as u8;
                                    st.rgba_buf[offset + 1] = g as u8;
                                    st.rgba_buf[offset + 2] = r as u8;
                                    st.rgba_buf[offset + 3] = 0xFF;
                                }
                                da.queue_draw();
                            }
                        }
                    }
                    glib::ControlFlow::Continue
                },
            );

            // Store the source ID so we can cancel it later if needed
            let mut st = state.borrow_mut();
            if let Some(s) = st.as_mut() {
                s.frame_timer = Some(source_id);
            }
        }
    };

    // Helper: load a ROM from path
    let cli_rc = Rc::new(cli);
    let load_rom = {
        let state = Rc::clone(&state);
        let window = window.clone();
        let start_frame_timer = start_frame_timer.clone();
        let cli = Rc::clone(&cli_rc);
        move |path: PathBuf, filter: scaling::ScaleFilter| {
            let rom: std::sync::Arc<[u8]> = match std::fs::read(&path) {
                Ok(r) => r.into(),
                Err(e) => {
                    eprintln!("Failed to read ROM '{}': {}", path.display(), e);
                    return;
                }
            };

            // Cancel existing timer
            {
                let mut st = state.borrow_mut();
                if let Some(s) = st.as_mut() {
                    if let Some(id) = s.frame_timer.take() {
                        id.remove();
                    }
                }
            }

            let emu_state = create_emu_state(rom, path.clone(), &cli, filter, None);
            let src_w = emu_state.src_w;
            let src_h = emu_state.src_h;

            *state.borrow_mut() = Some(emu_state);

            window.set_title(Some(&format!(
                "VibeBoy \u{2014} {}",
                path.file_name().unwrap_or_default().to_string_lossy()
            )));
            window.set_default_size((src_w * SCALE) as i32, (src_h * SCALE) as i32);

            start_frame_timer();
        }
    };

    // Keyboard input
    let state_key = Rc::clone(&state);
    let key_controller = gtk4::EventControllerKey::new();
    key_controller.connect_key_pressed(glib::clone!(
        #[strong] state_key,
        move |_, keyval, _keycode, _modifier| {
            let mut st = state_key.borrow_mut();
            let st = match st.as_mut() {
                Some(s) => s,
                None => return glib::Propagation::Proceed,
            };

            let btn = key_to_button(keyval);
            if let Some(b) = btn {
                st.kb_buttons |= b;
                st.emu.set_button(b, true);
            }

            match keyval {
                gtk4::gdk::Key::Escape => {
                    st.sav_flusher.flush(&st.emu);
                    std::process::exit(0);
                }
                gtk4::gdk::Key::BackSpace => { st.emu.set_rewinding(true); }
                gtk4::gdk::Key::Tab => { st.fast_forward = true; }
                gtk4::gdk::Key::minus => { st.slow_motion = true; }
                gtk4::gdk::Key::period => {
                    if st.paused { st.step_one_frame = true; }
                }
                gtk4::gdk::Key::F5 => {
                    let slot = st.current_slot;
                    ui_util::save_state_to_slot(&mut st.emu, &st.rom_path, slot);
                }
                gtk4::gdk::Key::F7 => {
                    let slot = st.current_slot;
                    ui_util::load_state_from_slot(&mut st.emu, &st.rom_path, slot);
                }
                gtk4::gdk::Key::space => {
                    st.paused = !st.paused;
                    eprintln!("{}", if st.paused { "Paused" } else { "Resumed" });
                }
                gtk4::gdk::Key::_0 => st.current_slot = 0,
                gtk4::gdk::Key::_1 => st.current_slot = 1,
                gtk4::gdk::Key::_2 => st.current_slot = 2,
                gtk4::gdk::Key::_3 => st.current_slot = 3,
                gtk4::gdk::Key::_4 => st.current_slot = 4,
                gtk4::gdk::Key::_5 => st.current_slot = 5,
                gtk4::gdk::Key::_6 => st.current_slot = 6,
                gtk4::gdk::Key::_7 => st.current_slot = 7,
                gtk4::gdk::Key::_8 => st.current_slot = 8,
                gtk4::gdk::Key::_9 => st.current_slot = 9,
                _ => {}
            }
            glib::Propagation::Stop
        }
    ));
    key_controller.connect_key_released(glib::clone!(
        #[strong] state_key,
        move |_, keyval, _keycode, _modifier| {
            let mut st = state_key.borrow_mut();
            let st = match st.as_mut() {
                Some(s) => s,
                None => return,
            };
            let btn = key_to_button(keyval);
            if let Some(b) = btn {
                st.kb_buttons &= !b;
                st.emu.set_button(b, false);
            }
            match keyval {
                gtk4::gdk::Key::BackSpace => { st.emu.set_rewinding(false); }
                gtk4::gdk::Key::Tab => { st.fast_forward = false; }
                gtk4::gdk::Key::minus => { st.slow_motion = false; }
                _ => {}
            }
        }
    ));
    window.add_controller(key_controller);

    // GLib actions
    let state_quit = Rc::clone(&state);
    let action_quit = gtk4::gio::SimpleAction::new("quit", None);
    action_quit.connect_activate(move |_, _| {
        let mut st = state_quit.borrow_mut();
        if let Some(s) = st.as_mut() {
            s.sav_flusher.flush(&s.emu);
        }
        std::process::exit(0);
    });
    app.add_action(&action_quit);

    // Open ROM action — show file dialog
    let load_rom_for_open = load_rom.clone();
    let window_for_open = window.clone();
    let action_open = gtk4::gio::SimpleAction::new("open", None);
    action_open.connect_activate(move |_, _| {
        let dialog = gtk4::FileDialog::new();
        dialog.set_title("Open ROM");
        let filter = gtk4::FileFilter::new();
        filter.add_pattern("*.gb");
        filter.add_pattern("*.gbc");
        filter.set_name(Some("Game Boy ROMs"));
        let filters = gtk4::gio::ListStore::new::<gtk4::FileFilter>();
        filters.append(&filter);
        let all_filter = gtk4::FileFilter::new();
        all_filter.add_pattern("*");
        all_filter.set_name(Some("All files"));
        filters.append(&all_filter);
        dialog.set_filters(Some(&filters));

        let load = load_rom_for_open.clone();
        dialog.open(Some(&window_for_open), gtk4::gio::Cancellable::NONE, move |result| {
            if let Ok(file) = result {
                if let Some(path) = file.path() {
                    load(path, scaling::ScaleFilter::Nearest);
                }
            }
        });
    });
    app.add_action(&action_open);

    let state_pause = Rc::clone(&state);
    let action_pause = gtk4::gio::SimpleAction::new("pause", None);
    action_pause.connect_activate(move |_, _| {
        let mut st = state_pause.borrow_mut();
        if let Some(s) = st.as_mut() {
            s.paused = !s.paused;
            eprintln!("{}", if s.paused { "Paused" } else { "Resumed" });
        }
    });
    app.add_action(&action_pause);

    let state_reset = Rc::clone(&state);
    let cli_for_reset = Rc::clone(&cli_rc);
    let action_reset = gtk4::gio::SimpleAction::new("reset", None);
    action_reset.connect_activate(move |_, _| {
        let mut st = state_reset.borrow_mut();
        if let Some(s) = st.as_mut() {
            let model = s.model_override
                .unwrap_or_else(|| ui_util::auto_detect_model(&s.rom_data));
            let boot_rom = load_boot_rom(model, &cli_for_reset);
            let path = s.rom_path.clone();
            s.emu = emulator::Emulator::new(
                s.rom_data.clone(), boot_rom, model, None, clock::default_clock(), AUDIO_SAMPLE_RATE,
            );
            ui_util::load_sav(&mut s.emu, &path);
            s.model = model;
            let is_sgb = s.emu.is_sgb();
            s.src_w = if is_sgb { 256 } else { 160 };
            s.src_h = if is_sgb { 224 } else { 144 };
            s.sav_flusher = ui_util::SavFlusher::new(&s.emu, &path);
            s.paused = false;
            s.step_one_frame = false;
            eprintln!("Reset");
        }
    });
    app.add_action(&action_reset);

    // Save state slot action (parameter: slot number as string "1"-"9")
    let state_save = Rc::clone(&state);
    let action_save_slot = gtk4::gio::SimpleAction::new(
        "save-slot", Some(&glib::VariantTy::STRING),
    );
    action_save_slot.connect_activate(move |_, param| {
        if let Some(param) = param {
            if let Some(s) = param.str() {
                if let Ok(slot_num) = s.parse::<usize>() {
                    let mut st = state_save.borrow_mut();
                    if let Some(st) = st.as_mut() {
                        let slot = slot_num;
                        ui_util::save_state_to_slot(&mut st.emu, &st.rom_path, slot);
                    }
                }
            }
        }
    });
    app.add_action(&action_save_slot);

    // Load state slot action
    let state_load = Rc::clone(&state);
    let action_load_slot = gtk4::gio::SimpleAction::new(
        "load-slot", Some(&glib::VariantTy::STRING),
    );
    action_load_slot.connect_activate(move |_, param| {
        if let Some(param) = param {
            if let Some(s) = param.str() {
                if let Ok(slot_num) = s.parse::<usize>() {
                    let mut st = state_load.borrow_mut();
                    if let Some(st) = st.as_mut() {
                        let slot = slot_num;
                        ui_util::load_state_from_slot(&mut st.emu, &st.rom_path, slot);
                    }
                }
            }
        }
    });
    app.add_action(&action_load_slot);

    // Model selection action
    let state_model = Rc::clone(&state);
    let cli_for_model = Rc::clone(&cli_rc);
    let window_for_model = window.clone();
    let start_timer_for_model = start_frame_timer.clone();
    let action_model = gtk4::gio::SimpleAction::new(
        "model", Some(&glib::VariantTy::STRING),
    );
    action_model.connect_activate(move |_, param| {
        if let Some(param) = param {
            if let Some(name) = param.str() {
                let model_override = match name {
                    "auto" => None,
                    "dmg0" => Some(GbModel::Dmg0),
                    "dmg" => Some(GbModel::Dmg),
                    "mgb" => Some(GbModel::Mgb),
                    "sgb" => Some(GbModel::Sgb),
                    "sgb2" => Some(GbModel::Sgb2),
                    "cgb" => Some(GbModel::Cgb),
                    "agb" => Some(GbModel::Agb),
                    _ => return,
                };

                let mut st = state_model.borrow_mut();
                if let Some(s) = st.as_mut() {
                    // Cancel existing timer
                    if let Some(id) = s.frame_timer.take() {
                        id.remove();
                    }

                    let rom = s.rom_data.clone();
                    let rom_path = s.rom_path.clone();
                    let filter = s.scale_filter;

                    let new_state = create_emu_state(
                        rom, rom_path.clone(), &cli_for_model, filter, model_override,
                    );
                    let src_w = new_state.src_w;
                    let src_h = new_state.src_h;
                    *s = new_state;

                    window_for_model.set_title(Some(&format!(
                        "VibeBoy \u{2014} {}",
                        rom_path.file_name().unwrap_or_default().to_string_lossy()
                    )));
                    window_for_model.set_default_size(
                        (src_w * SCALE) as i32, (src_h * SCALE) as i32,
                    );
                }
                drop(st);
                start_timer_for_model();
            }
        }
    });
    app.add_action(&action_model);

    // Filter action: stateful string action — GIO shows checkmark on the matching item
    let state_filter = Rc::clone(&state);
    let action_filter = gtk4::gio::SimpleAction::new_stateful(
        "filter",
        Some(&glib::VariantTy::STRING),
        &initial_filter.cli_name().to_variant(),
    );
    action_filter.connect_activate(move |action, param| {
        if let Some(param) = param {
            if let Some(name) = param.str() {
                if let Some(filter) = scaling::ScaleFilter::from_name(name) {
                    let mut st = state_filter.borrow_mut();
                    if let Some(s) = st.as_mut() {
                        s.scale_filter = filter;
                        s.vec_cache = filter.new_vectorize_cache();
                        action.set_state(&name.to_variant());
                        eprintln!("Filter: {:?}", filter);
                    }
                }
            }
        }
    });
    app.add_action(&action_filter);

    // Printer toggle action
    let state_printer = Rc::clone(&state);
    let action_printer = gtk4::gio::SimpleAction::new_stateful(
        "toggle-printer",
        None,
        &cli_rc.printer.to_variant(),
    );
    action_printer.connect_activate(move |action, _| {
        let mut st = state_printer.borrow_mut();
        if let Some(s) = st.as_mut() {
            let currently_on = action.state().and_then(|v| v.get::<bool>()).unwrap_or(false);
            let new_state = !currently_on;
            action.set_state(&new_state.to_variant());
            if new_state {
                s.emu.attach_serial_device(
                    Box::new(printer::Printer::new(s.model.cpu_clock_rate()))
                );
                eprintln!("Game Boy Printer connected");
            } else {
                s.emu.attach_serial_device(Box::new(serial::Disconnected));
                eprintln!("Game Boy Printer disconnected");
            }
        }
    });
    app.add_action(&action_printer);

    // Force CPU toggle action
    let state_force_cpu = Rc::clone(&state);
    let action_force_cpu = gtk4::gio::SimpleAction::new_stateful(
        "force-cpu",
        None,
        &false.to_variant(),
    );
    action_force_cpu.connect_activate(move |action, _| {
        let mut st = state_force_cpu.borrow_mut();
        if let Some(s) = st.as_mut() {
            let currently_on = action.state().and_then(|v| v.get::<bool>()).unwrap_or(false);
            let new_state = !currently_on;
            action.set_state(&new_state.to_variant());
            s.force_cpu = new_state;
            eprintln!("Force CPU: {}", if new_state { "on" } else { "off" });
        }
    });
    app.add_action(&action_force_cpu);

    // Select slot action (stateful string for radio checkmarks)
    let state_select_slot = Rc::clone(&state);
    let action_select_slot = gtk4::gio::SimpleAction::new_stateful(
        "select-slot",
        Some(&glib::VariantTy::STRING),
        &"0".to_variant(),
    );
    action_select_slot.connect_activate(move |action, param| {
        if let Some(param) = param {
            if let Some(name) = param.str() {
                action.set_state(&name.to_variant());
                let mut st = state_select_slot.borrow_mut();
                if let Some(s) = st.as_mut() {
                    if let Ok(n) = name.parse::<usize>() {
                        s.current_slot = n;
                        eprintln!("Slot {} selected", n);
                    }
                }
            }
        }
    });
    app.add_action(&action_select_slot);

    window.present();

    // If ROM was provided on command line, load it now
    if let Some(ref rom_path) = cli_rc.rom {
        load_rom(rom_path.clone(), initial_filter);
    } else {
        // Show file dialog immediately
        let dialog = gtk4::FileDialog::new();
        dialog.set_title("Open ROM");
        let filter = gtk4::FileFilter::new();
        filter.add_pattern("*.gb");
        filter.add_pattern("*.gbc");
        filter.set_name(Some("Game Boy ROMs"));
        let filters = gtk4::gio::ListStore::new::<gtk4::FileFilter>();
        filters.append(&filter);
        let all_filter = gtk4::FileFilter::new();
        all_filter.add_pattern("*");
        all_filter.set_name(Some("All files"));
        filters.append(&all_filter);
        dialog.set_filters(Some(&filters));

        dialog.open(Some(&window), gtk4::gio::Cancellable::NONE, move |result| {
            if let Ok(file) = result {
                if let Some(path) = file.path() {
                    load_rom(path, scaling::ScaleFilter::Nearest);
                }
            }
        });
    }
}

fn key_to_button(keyval: gtk4::gdk::Key) -> Option<u8> {
    match keyval {
        gtk4::gdk::Key::z | gtk4::gdk::Key::Z => Some(emulator::Emulator::BTN_B),
        gtk4::gdk::Key::x | gtk4::gdk::Key::X => Some(emulator::Emulator::BTN_A),
        gtk4::gdk::Key::Return => Some(emulator::Emulator::BTN_START),
        gtk4::gdk::Key::Shift_R => Some(emulator::Emulator::BTN_SELECT),
        gtk4::gdk::Key::Right => Some(emulator::Emulator::BTN_RIGHT),
        gtk4::gdk::Key::Left => Some(emulator::Emulator::BTN_LEFT),
        gtk4::gdk::Key::Up => Some(emulator::Emulator::BTN_UP),
        gtk4::gdk::Key::Down => Some(emulator::Emulator::BTN_DOWN),
        _ => None,
    }
}

