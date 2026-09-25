use vibeboy_core::*;

#[cfg(target_os = "linux")]
mod compute;
mod gpu;

use clap::Parser;
use gtk4::glib;
use gtk4::prelude::*;
use model::GbModel;
use std::cell::RefCell;
use std::collections::HashSet;
use std::path::PathBuf;
use std::rc::Rc;
use ui_util::{HoldInputs, Session, SessionConfig};

pub(crate) const SCALE: u32 = 3;
pub(crate) const GB_W: u32 = 160;
pub(crate) const GB_H: u32 = 144;
pub(crate) const SGB_W: u32 = 256;
pub(crate) const SGB_H: u32 = 224;
pub(crate) const AUDIO_SAMPLE_RATE: u32 = 96_000;

#[derive(Parser)]
#[command(
    name = "vibeboy",
    about = "Game Boy / Game Boy Color emulator (GTK4 frontend)"
)]
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

/// Emulator window state, created when the first ROM is opened and kept
/// across ROM switches.
struct EmuState {
    session: Session,
    src_w: u32,
    src_h: u32,
    /// Hold hotkeys from the keyboard and from the gamepad; either holds.
    kb_hold: HoldInputs,
    gp_hold: HoldInputs,
    /// Hardware keycodes currently down, to tell key auto-repeat apart
    /// from fresh presses (GTK reports both as key-pressed).
    keys_down: HashSet<u32>,
    force_cpu: bool,
    scale_filter: scaling::ScaleFilter,
    kb_buttons: u8,
    gp_buttons: u8,
    rgba_buf: Vec<u8>,
    scaled_buf: Vec<u32>,
    audio: Option<cpal_audio::CpalAudio>,
    frame_timer: Option<glib::SourceId>,
    fps: ui_util::FpsCounter,
    gamepad: Option<ui_util::GamepadPoller>,
}

impl EmuState {
    fn new(session: Session, scale_filter: scaling::ScaleFilter) -> Self {
        let mut state = EmuState {
            session,
            src_w: GB_W,
            src_h: GB_H,
            kb_hold: HoldInputs::default(),
            gp_hold: HoldInputs::default(),
            keys_down: HashSet::new(),
            force_cpu: false,
            scale_filter,
            kb_buttons: 0,
            gp_buttons: 0,
            rgba_buf: Vec::new(),
            scaled_buf: Vec::new(),
            audio: cpal_audio::CpalAudio::start(AUDIO_SAMPLE_RATE),
            frame_timer: None,
            fps: ui_util::FpsCounter::new(),
            gamepad: ui_util::GamepadPoller::new(),
        };
        state.on_new_emulator();
        state
    }

    /// Sync with a freshly built emulator (ROM switch, reset, model change).
    fn on_new_emulator(&mut self) {
        let is_sgb = self.session.emu.is_sgb();
        self.src_w = if is_sgb { SGB_W } else { GB_W };
        self.src_h = if is_sgb { SGB_H } else { GB_H };
        self.rgba_buf
            .resize((self.src_w * self.src_h * 4) as usize, 0);
    }

    /// Stop the frame timer, e.g. before restarting it at a new rate.
    fn stop_frame_timer(&mut self) {
        if let Some(id) = self.frame_timer.take() {
            id.remove();
        }
    }
}

fn session_config(cli: &Cli) -> SessionConfig {
    SessionConfig {
        model: cli.model,
        bootrom: cli.bootrom.clone(),
        no_boot: cli.no_boot,
        printer: cli.printer,
        sample_rate: AUDIO_SAMPLE_RATE,
        ..Default::default()
    }
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

fn build_ui(app: &gtk4::Application, cli: Cli) {
    let initial_filter = cli
        .filter
        .as_ref()
        .and_then(|name| scaling::ScaleFilter::from_name(name))
        .unwrap_or(scaling::ScaleFilter::Nearest);

    ui_util::print_controls();
    if initial_filter != scaling::ScaleFilter::Nearest {
        eprintln!("  Filter: {:?}", initial_filter);
    }
    eprintln!();

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
        ("DMG0", "dmg0"),
        ("DMG", "dmg"),
        ("MGB", "mgb"),
        ("SGB", "sgb"),
        ("SGB2", "sgb2"),
        ("CGB", "cgb"),
        ("AGB", "agb"),
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
            sub_menus
                .entry(group.label())
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

    // Stack: GL area (preferred) with Cairo DrawingArea fallback. The GL page
    // starts visible because a Stack only realizes its visible child, and the
    // realize handler below is what sets up GL or switches to Cairo.
    let stack = gtk4::Stack::new();
    stack.add_named(&drawing_area, Some("cairo"));
    stack.add_named(&gl_area, Some("gl"));
    stack.set_visible_child_name("gl");

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
    let pending_frame: Rc<RefCell<gpu::PendingFrame>> =
        Rc::new(RefCell::new(gpu::PendingFrame::default()));

    // GLArea realize: init GL resources + wgpu compute (same GL context)
    gl_area.connect_realize({
        let gl_renderer = Rc::clone(&gl_renderer);
        let gpu_compute = Rc::clone(&gpu_compute);
        let stack = stack.clone();
        move |area| {
            area.make_current();
            if area.error().is_some() {
                eprintln!("GLArea error, falling back to Cairo");
                stack.set_visible_child_name("cairo");
                return;
            }
            match gpu::GlRenderer::new() {
                Some(r) => {
                    let gl_name = r.renderer_name();
                    *gl_renderer.borrow_mut() = Some(r);
                    // Init wgpu compute using the same GL context (zero-copy)
                    #[cfg(target_os = "linux")]
                    if gl_name.contains("SVGA3D") {
                        // VMware's SVGA3D driver drops compute-shader writes
                        // once the context has drawn to GTK's framebuffer, so
                        // every filtered frame comes out black.
                        eprintln!("GPU compute disabled on {gl_name}, will use CPU scaling");
                    } else {
                        match compute::GpuCompute::new(|s| gpu::gl_proc_address(s)) {
                            Some(c) => *gpu_compute.borrow_mut() = Some(c),
                            None => eprintln!("GPU compute init failed, will use CPU scaling"),
                        }
                    }
                }
                None => {
                    eprintln!("GL renderer init failed, falling back to Cairo");
                    stack.set_visible_child_name("cairo");
                }
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
                renderer.begin_frame();
                let has_frame = !f.pixels.is_empty() || f.gl_texture.is_some();
                let scale = area.scale_factor();
                let vp_w = area.width() * scale;
                let vp_h = area.height() * scale;
                if !has_frame {
                    // No ROM loaded yet: black, like the Cairo fallback
                    renderer.clear(vp_w, vp_h);
                } else {
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
                        &f.pixels, f.frame_w, f.frame_h, vp_w, vp_h, f.src_w, f.src_h,
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
                st.session.frame_duration().as_millis().max(1) as u64
            };

            let source_id =
                glib::timeout_add_local(std::time::Duration::from_millis(interval_ms), move || {
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
                                if pressed & mask != 0 {
                                    st.session.emu.set_button(mask, true);
                                }
                                if released & mask != 0 && st.kb_buttons & mask == 0 {
                                    st.session.emu.set_button(mask, false);
                                }
                            }
                            st.gp_hold.rewind = gs.rewind;
                            st.gp_hold.fast_forward = gs.fast_forward;
                        }

                        let hold = HoldInputs {
                            rewind: st.kb_hold.rewind || st.gp_hold.rewind,
                            fast_forward: st.kb_hold.fast_forward || st.gp_hold.fast_forward,
                            slow_motion: st.kb_hold.slow_motion,
                        };
                        let queued = st.audio.as_ref().map(cpal_audio::CpalAudio::queued_frames);
                        let out = st.session.tick(&hold, queued);
                        if let Some(ref mut audio) = st.audio {
                            audio.push(&out.audio);
                        }

                        // Rumble
                        if let Some(ref mut gp) = st.gamepad
                            && st.session.emu.has_rumble()
                        {
                            gp.ensure_rumble();
                            gp.set_rumble(st.session.emu.drain_rumble());
                        }

                        st.fps.update(out.frames_emulated(), out.emu_time);

                        // Get frame buffer and dimensions
                        let is_sgb = st.session.emu.is_sgb();
                        let base_w = if is_sgb {
                            SGB_W as usize
                        } else {
                            GB_W as usize
                        };
                        let base_h = if is_sgb {
                            SGB_H as usize
                        } else {
                            GB_H as usize
                        };
                        let fb: &[u32] = if is_sgb {
                            st.session.emu.sgb_composited_frame()
                        } else {
                            st.session.emu.frame_buffer()
                        };
                        // Size of whichever view the Stack is showing; the
                        // hidden one is unallocated and reports 0x0.
                        let (view_w, view_h) = if gl_renderer.borrow().is_some() {
                            (gl_area.width(), gl_area.height())
                        } else {
                            (da.width(), da.height())
                        };
                        let da_w = view_w.max(1) as usize;
                        let da_h = view_h.max(1) as usize;

                        // Compute aspect-ratio-correct dimensions for filters
                        let scale_fit = (da_w as f64 / base_w as f64)
                            .min(da_h as f64 / base_h as f64)
                            .max(1.0);
                        let fit_w = (base_w as f64 * scale_fit).round() as usize;
                        let fit_h = (base_h as f64 * scale_fit).round() as usize;

                        // Check if we can use GPU compute for this filter
                        #[cfg(target_os = "linux")]
                        let wgpu_filter = if st.force_cpu {
                            None
                        } else {
                            compute::to_wgpu_filter(st.scale_filter)
                        };
                        #[cfg(not(target_os = "linux"))]
                        let wgpu_filter: Option<
                            scaling::wgpu_scale::WgpuScaleFilter,
                        > = None;
                        let use_gpu = !st.force_cpu
                            && wgpu_filter.is_some()
                            && gpu_compute.borrow().is_some();

                        // Apply scaling filter: GPU deferred to render callback, else CPU
                        let (pixels, pw, ph, gpu_filter_for_render): (
                            &[u32],
                            usize,
                            usize,
                            Option<scaling::wgpu_scale::WgpuScaleFilter>,
                        ) = if use_gpu {
                            // Send raw pixels to render callback for zero-copy GPU compute
                            (fb, base_w, base_h, wgpu_filter)
                        } else if st.scale_filter == scaling::ScaleFilter::Nearest {
                            (fb, base_w, base_h, None)
                        } else if !st.force_cpu
                            && st.scale_filter == scaling::ScaleFilter::Vectorize
                        {
                            // Full 6-stage GPU vectorize pipeline
                            #[cfg(target_os = "linux")]
                            {
                                let s = scale_fit as f32;
                                let ow = (base_w as f32 * s).round() as u32;
                                let oh = (base_h as f32 * s).round() as u32;
                                let mut gc = gpu_compute.borrow_mut();
                                if let Some(ref mut compute) = *gc {
                                    if let Some((gl_tex, gw, gh)) = compute.vectorize(
                                        fb,
                                        base_w as u32,
                                        base_h as u32,
                                        ow,
                                        oh,
                                        s,
                                    ) {
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
                            }
                        } else if let Some((scaled, w, h)) =
                            scaling::cpu_scale(st.scale_filter, fb, base_w, base_h, fit_w, fit_h)
                        {
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
                    glib::ControlFlow::Continue
                });

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
        move |path: PathBuf| {
            let mut st = state.borrow_mut();
            // Switching ROMs writes the outgoing battery save first.
            let loaded = match st.as_mut() {
                Some(s) => s.session.load_rom(&path).map(|()| {
                    s.stop_frame_timer();
                    s.on_new_emulator();
                }),
                None => Session::new(&path, session_config(&cli))
                    .map(|session| *st = Some(EmuState::new(session, initial_filter))),
            };
            if let Err(e) = loaded {
                eprintln!("Failed to load ROM '{}': {}", path.display(), e);
                return;
            }
            let s = st.as_ref().unwrap();
            let (src_w, src_h) = (s.src_w, s.src_h);
            drop(st);

            window.set_title(Some(&format!(
                "VibeBoy \u{2014} {}",
                path.file_name().unwrap_or_default().to_string_lossy()
            )));
            window.set_default_size((src_w * SCALE) as i32, (src_h * SCALE) as i32);

            // The frame rate depends on the model.
            start_frame_timer();
        }
    };

    // Keyboard input
    let state_key = Rc::clone(&state);
    let app_key = app.clone();
    let key_controller = gtk4::EventControllerKey::new();
    key_controller.connect_key_pressed(glib::clone!(
        #[strong]
        state_key,
        #[strong]
        app_key,
        move |_, keyval, keycode, _modifier| {
            let mut st = state_key.borrow_mut();
            let st = match st.as_mut() {
                Some(s) => s,
                None => return glib::Propagation::Proceed,
            };
            let repeat = !st.keys_down.insert(keycode);

            let btn = key_to_button(keyval);
            if let Some(b) = btn {
                st.kb_buttons |= b;
                st.session.emu.set_button(b, true);
            }

            match keyval {
                gtk4::gdk::Key::Escape => {
                    st.session.flush_save();
                    std::process::exit(0);
                }
                gtk4::gdk::Key::BackSpace => st.kb_hold.rewind = true,
                gtk4::gdk::Key::Tab => st.kb_hold.fast_forward = true,
                gtk4::gdk::Key::minus => st.kb_hold.slow_motion = true,
                // Frame advance steps repeatedly while Period is held.
                gtk4::gdk::Key::period => st.session.request_frame_advance(),
                // The other hotkeys ignore key auto-repeat.
                _ if repeat => {}
                gtk4::gdk::Key::F5 => st.session.save_state(st.session.slot()),
                gtk4::gdk::Key::F7 => st.session.load_state(st.session.slot()),
                gtk4::gdk::Key::space => {
                    st.session.toggle_pause();
                }
                _ => {
                    if let Some(slot) = key_to_slot(keyval) {
                        st.session.select_slot(slot);
                        if let Some(action) = app_key.lookup_action("select-slot") {
                            action.change_state(&slot.to_string().to_variant());
                        }
                    }
                }
            }
            glib::Propagation::Stop
        }
    ));
    key_controller.connect_key_released(glib::clone!(
        #[strong]
        state_key,
        move |_, keyval, keycode, _modifier| {
            let mut st = state_key.borrow_mut();
            let st = match st.as_mut() {
                Some(s) => s,
                None => return,
            };
            st.keys_down.remove(&keycode);
            let btn = key_to_button(keyval);
            if let Some(b) = btn {
                st.kb_buttons &= !b;
                st.session.emu.set_button(b, false);
            }
            match keyval {
                gtk4::gdk::Key::BackSpace => st.kb_hold.rewind = false,
                gtk4::gdk::Key::Tab => st.kb_hold.fast_forward = false,
                gtk4::gdk::Key::minus => st.kb_hold.slow_motion = false,
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
            s.session.flush_save();
        }
        std::process::exit(0);
    });
    app.add_action(&action_quit);

    // Window close button: flush before the window (and the app) goes away
    let state_close = Rc::clone(&state);
    window.connect_close_request(move |_| {
        if let Some(s) = state_close.borrow_mut().as_mut() {
            s.session.flush_save();
        }
        glib::Propagation::Proceed
    });

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
        dialog.open(
            Some(&window_for_open),
            gtk4::gio::Cancellable::NONE,
            move |result| {
                if let Ok(file) = result {
                    if let Some(path) = file.path() {
                        load(path);
                    }
                }
            },
        );
    });
    app.add_action(&action_open);

    let state_pause = Rc::clone(&state);
    let action_pause = gtk4::gio::SimpleAction::new("pause", None);
    action_pause.connect_activate(move |_, _| {
        if let Some(s) = state_pause.borrow_mut().as_mut() {
            s.session.toggle_pause();
        }
    });
    app.add_action(&action_pause);

    let state_reset = Rc::clone(&state);
    let action_reset = gtk4::gio::SimpleAction::new("reset", None);
    action_reset.connect_activate(move |_, _| {
        if let Some(s) = state_reset.borrow_mut().as_mut() {
            s.session.reset();
            s.on_new_emulator();
        }
    });
    app.add_action(&action_reset);

    // Save state slot action (parameter: slot number as string "0"-"9")
    let state_save = Rc::clone(&state);
    let action_save_slot =
        gtk4::gio::SimpleAction::new("save-slot", Some(&glib::VariantTy::STRING));
    action_save_slot.connect_activate(move |_, param| {
        if let Some(slot) = param.and_then(|p| p.str()?.parse::<usize>().ok())
            && let Some(s) = state_save.borrow_mut().as_mut()
        {
            s.session.save_state(slot);
        }
    });
    app.add_action(&action_save_slot);

    // Load state slot action
    let state_load = Rc::clone(&state);
    let action_load_slot =
        gtk4::gio::SimpleAction::new("load-slot", Some(&glib::VariantTy::STRING));
    action_load_slot.connect_activate(move |_, param| {
        if let Some(slot) = param.and_then(|p| p.str()?.parse::<usize>().ok())
            && let Some(s) = state_load.borrow_mut().as_mut()
        {
            s.session.load_state(slot);
        }
    });
    app.add_action(&action_load_slot);

    // Model selection action
    let state_model = Rc::clone(&state);
    let window_for_model = window.clone();
    let start_timer_for_model = start_frame_timer.clone();
    let action_model = gtk4::gio::SimpleAction::new("model", Some(&glib::VariantTy::STRING));
    action_model.connect_activate(move |_, param| {
        let Some(name) = param.and_then(|p| p.str()) else {
            return;
        };
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
        let Some(s) = st.as_mut() else {
            return;
        };
        // Writes the outgoing battery save before rebuilding.
        s.session.set_model(model_override);
        s.stop_frame_timer();
        s.on_new_emulator();
        window_for_model.set_default_size((s.src_w * SCALE) as i32, (s.src_h * SCALE) as i32);
        drop(st);
        // The frame rate depends on the model.
        start_timer_for_model();
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
    let action_printer =
        gtk4::gio::SimpleAction::new_stateful("toggle-printer", None, &cli_rc.printer.to_variant());
    action_printer.connect_activate(move |action, _| {
        let mut st = state_printer.borrow_mut();
        if let Some(s) = st.as_mut() {
            let on = !s.session.printer_attached();
            s.session.set_printer(on);
            action.set_state(&on.to_variant());
        }
    });
    app.add_action(&action_printer);

    // Force CPU toggle action
    let state_force_cpu = Rc::clone(&state);
    let action_force_cpu =
        gtk4::gio::SimpleAction::new_stateful("force-cpu", None, &false.to_variant());
    action_force_cpu.connect_activate(move |action, _| {
        let mut st = state_force_cpu.borrow_mut();
        if let Some(s) = st.as_mut() {
            let currently_on = action
                .state()
                .and_then(|v| v.get::<bool>())
                .unwrap_or(false);
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
        let Some(name) = param.and_then(|p| p.str()) else {
            return;
        };
        action.set_state(&name.to_variant());
        if let Ok(n) = name.parse::<usize>()
            && let Some(s) = state_select_slot.borrow_mut().as_mut()
        {
            s.session.select_slot(n);
        }
    });
    app.add_action(&action_select_slot);

    window.present();

    // If ROM was provided on command line, load it now
    if let Some(ref rom_path) = cli_rc.rom {
        load_rom(rom_path.clone());
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
                    load_rom(path);
                }
            }
        });
    }
}

fn key_to_slot(keyval: gtk4::gdk::Key) -> Option<usize> {
    use gtk4::gdk::Key;
    [
        Key::_0,
        Key::_1,
        Key::_2,
        Key::_3,
        Key::_4,
        Key::_5,
        Key::_6,
        Key::_7,
        Key::_8,
        Key::_9,
    ]
    .iter()
    .position(|&k| k == keyval)
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
