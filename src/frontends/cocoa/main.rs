use vibeboy_core::*;

mod accel;
mod audio;
mod camera;
mod controls;
mod font;
mod gamepad;
mod menu;
mod metal_renderer;
mod persistence;
mod vectorize_metal;

use clap::Parser;
use model::GbModel;
use std::cell::RefCell;
use std::collections::HashSet;
use std::ffi::c_void;
use std::path::PathBuf;
use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::{AnyClass, AnyObject, Bool, ClassBuilder, Sel};
use objc2::{MainThreadOnly, msg_send, sel};
use objc2_app_kit::{
    NSApplication, NSApplicationActivationPolicy, NSBackingStoreType, NSControlStateValueOff,
    NSControlStateValueOn, NSEventMask, NSEventType, NSWindow, NSWindowStyleMask,
};
use objc2_foundation::{MainThreadMarker, NSDefaultRunLoopMode, NSPoint, NSRect, NSSize, NSString};
use objc2_metal::*;

use ui_util::parse_filter;
use ui_util::{HoldInputs, Session, SessionConfig, TickOutput};

use accel::{close_accel, init_accel, poll_accel};
use audio::AudioOutput;
use camera::CameraCapture;
use controls::{open_rom_dialog, show_controls_panel};
use font::tiny_font;
use gamepad::GamepadState;
use menu::*;
use metal_renderer::MetalRenderer;
use persistence::*;
use vectorize_metal::MetalVectorizePipeline;

pub(crate) const SCALE: u32 = 3;
pub(crate) const AUDIO_SAMPLE_RATE: u32 = 96_000;

// ── Accelerometer source tracking ─────────────────────────────────────────────

pub(crate) enum AccelSource {
    None,
    /// IOKit HID (Apple Silicon built-in accelerometer)
    IoKit,
    /// CoreMotion CMMotionManager fallback
    CoreMotion(Retained<objc2_core_motion::CMMotionManager>),
}

pub(crate) const K_ESCAPE: u16 = 53;
pub(crate) const K_F5: u16 = 96;
pub(crate) const K_F7: u16 = 98;
pub(crate) const K_TAB: u16 = 48;
pub(crate) const K_DELETE: u16 = 51;
const K_SPACE: u16 = 49;
const K_PERIOD: u16 = 47;
const K_MINUS: u16 = 27;

/// For a `FlagsChanged` event, whether the modifier key `keycode` is now
/// down. Modifier keys never produce KeyDown/KeyUp; the event only carries
/// the new modifier state. The device-dependent bits in the low word
/// (IOKit's NX_DEVICE*KEYMASK) tell left and right keys apart.
fn modifier_key_down(keycode: u16, flags: usize) -> Option<bool> {
    let mask = match keycode {
        59 => 0x0001, // left control
        56 => 0x0002, // left shift
        60 => 0x0004, // right shift
        55 => 0x0008, // left command
        54 => 0x0010, // right command
        58 => 0x0020, // left option
        61 => 0x0040, // right option
        62 => 0x2000, // right control
        _ => return None,
    };
    Some(flags & mask != 0)
}

fn keycode_to_slot(keycode: u16) -> Option<usize> {
    match keycode {
        29 => Some(0), // 0
        18 => Some(1), // 1
        19 => Some(2), // 2
        20 => Some(3), // 3
        21 => Some(4), // 4
        23 => Some(5), // 5
        22 => Some(6), // 6
        26 => Some(7), // 7
        28 => Some(8), // 8
        25 => Some(9), // 9
        _ => None,
    }
}

fn string_to_filter(s: &str) -> scaling::ScaleFilter {
    scaling::ScaleFilter::from_name(s).unwrap_or(scaling::ScaleFilter::Nearest)
}

#[derive(Parser)]
#[command(
    name = "vibeboy_cocoa",
    about = "Game Boy / Game Boy Color emulator (macOS native)"
)]
struct Cli {
    rom: Option<PathBuf>,
    #[arg(long)]
    bootrom: Option<PathBuf>,
    #[arg(long, value_parser = util::parse_model)]
    model: Option<GbModel>,
    #[arg(long)]
    snes_rom: Option<PathBuf>,
    #[arg(long)]
    lle: bool,
    #[arg(long)]
    no_boot: bool,
    #[arg(long)]
    printer: bool,
    /// Scaling filter
    #[arg(long, default_value = "nearest", value_parser = parse_filter)]
    filter: String,
}

// ── CFRunLoopTimer (for emulation during menu tracking) ─────────────────────

#[allow(non_snake_case)]
unsafe extern "C" {
    fn CFRunLoopGetMain() -> *mut c_void;
    fn CFRunLoopAddTimer(rl: *mut c_void, timer: *mut c_void, mode: *const c_void);
    fn CFRunLoopTimerCreate(
        allocator: *const c_void,
        fireDate: f64,
        interval: f64,
        flags: u64,
        order: i64,
        callout: unsafe extern "C" fn(timer: *mut c_void, info: *mut c_void),
        context: *mut CFRunLoopTimerContext,
    ) -> *mut c_void;
    fn CFRunLoopTimerInvalidate(timer: *mut c_void);
    fn CFAbsoluteTimeGetCurrent() -> f64;
    fn CFRelease(cf: *mut c_void);
    static kCFRunLoopCommonModes: *const c_void;
    static kCFAllocatorDefault: *const c_void;
}

#[repr(C)]
struct CFRunLoopTimerContext {
    version: isize,
    info: *mut c_void,
    retain: Option<unsafe extern "C" fn(*const c_void) -> *const c_void>,
    release: Option<unsafe extern "C" fn(*const c_void)>,
    copy_description: Option<unsafe extern "C" fn(*const c_void) -> *const c_void>,
}

/// Context passed to the frame timer callback. The callback only ever takes a
/// shared reference to it.
struct FrameTimerInfo {
    /// Shared with the main loop. The timer also fires inside nested run
    /// loops (menu tracking, the Open dialog, the controls panel), so the
    /// main loop must never hold a borrow across anything that spins the
    /// run loop unless it is fine for the timer to skip those ticks.
    state: Rc<RefCell<AppState>>,
    window: Retained<NSWindow>,
}

/// Called by CFRunLoopTimer at frame rate. Steps emulation, renders and
/// updates FPS. Fires on kCFRunLoopCommonModes so it continues during menu
/// tracking, keeping audio-driven emulation smooth.
unsafe extern "C" fn frame_timer_callback(_timer: *mut c_void, info: *mut c_void) {
    // SAFETY: `info` points to the FrameTimerInfo owned by main(), which
    // outlives the timer (invalidated before it is dropped). Only shared
    // references are ever created from it, and only on the main thread.
    let ctx = unsafe { &*(info as *const FrameTimerInfo) };
    // Already borrowed means the main loop is inside a modal dialog or
    // panel; skip the tick so emulation pauses until it returns.
    let Ok(mut guard) = ctx.state.try_borrow_mut() else {
        return;
    };
    let state = &mut *guard;
    let window = &*ctx.window;

    let _pool = unsafe { objc2_foundation::NSAutoreleasePool::new() };

    state.update_input();
    let out = state.step_emulation();

    if let Some(content_view) = window.contentView() {
        state.render(window, &content_view);
    }

    if let Some((f, ms)) = state.fps.update(out.frames_emulated(), out.emu_time) {
        state.overlay_fps = f;
        state.overlay_emu_ms = ms;
    }
}

// ── AppState ─────────────────────────────────────────────────────────────────

struct AppState {
    session: Session,
    renderer: MetalRenderer,
    scale_filter: scaling::ScaleFilter,
    key_map: std::collections::HashMap<u16, u8>,
    keys_down: HashSet<u16>,
    gamepad: GamepadState,
    audio: Option<AudioOutput>,
    camera: Option<CameraCapture>,
    camera_buf: [u8; 128 * 112],
    accel_source: AccelSource,
    force_cpu: bool,
    fps: ui_util::FpsCounter,
    show_fps_overlay: bool,
    overlay_fps: f64,
    overlay_emu_ms: f64,
    frame_copy: Vec<u32>,
    bgra_buf: Vec<u32>,
    src_w: usize,
    src_h: usize,
    is_sgb: bool,
}

/// Show "Pause" or "Resume" on the Emulation menu's pause item.
fn update_pause_menu_item(app: &NSApplication, paused: bool) {
    if let Some(main_menu) = app.mainMenu()
        && let Some(emu_menu) = main_menu.itemAtIndex(3)
        && let Some(submenu) = emu_menu.submenu()
        && let Some(pause_item) = submenu.itemWithTag(MENU_TAG_PAUSE)
    {
        let label = if paused { "Resume" } else { "Pause" };
        pause_item.setTitle(&NSString::from_str(label));
    }
}

/// Window title for a loaded ROM.
fn window_title(rom_path: &std::path::Path) -> String {
    format!(
        "VibeBoy \u{2014} {}",
        rom_path.file_name().unwrap_or_default().to_string_lossy()
    )
}

impl AppState {
    /// Sync frontend state with a freshly built emulator (startup, ROM
    /// switch, reset, model change): source dimensions, the peripherals the
    /// cartridge needs, and the pause menu item (a new emulator runs).
    fn on_new_emulator(&mut self, app: &NSApplication) {
        let emu = &self.session.emu;
        self.is_sgb = emu.is_sgb();
        (self.src_w, self.src_h) = if self.is_sgb { (256, 224) } else { (160, 144) };

        // Webcam for the Pocket Camera
        if !emu.has_camera() {
            self.camera = None;
        } else if self.camera.is_none() {
            self.camera = CameraCapture::start();
        }

        // Accelerometer for MBC7
        if !emu.has_accelerometer() {
            close_accel(&self.accel_source);
            self.accel_source = AccelSource::None;
        } else if matches!(self.accel_source, AccelSource::None) {
            self.accel_source = init_accel();
        }

        update_pause_menu_item(app, self.session.paused());
    }

    /// Switch to a new ROM. Updates window title and recent ROMs.
    fn load_rom(
        &mut self,
        path: PathBuf,
        mtm: MainThreadMarker,
        app: &NSApplication,
        window: &NSWindow,
    ) {
        if let Err(e) = self.session.load_rom(&path) {
            eprintln!("Failed to read ROM '{}': {}", path.display(), e);
            return;
        }
        window.setTitle(&NSString::from_str(&window_title(&path)));
        add_recent_rom(&path.to_string_lossy());
        rebuild_recent_menu(mtm, app, &load_recent_roms());
        self.on_new_emulator(app);
    }

    fn toggle_pause(&mut self, app: &NSApplication) {
        let paused = self.session.toggle_pause();
        update_pause_menu_item(app, paused);
    }

    fn select_slot(&mut self, app: &NSApplication, slot: usize) {
        self.session.select_slot(slot);
        update_slot_checkmarks(app, slot);
    }

    /// Handle all pending menu actions.
    fn handle_menu_actions(
        &mut self,
        actions: MenuActions,
        mtm: MainThreadMarker,
        app: &NSApplication,
        window: &NSWindow,
    ) {
        if actions.open_rom
            && let Some(path) = open_rom_dialog()
        {
            self.load_rom(path, mtm, app, window);
        }

        if actions.pause_toggle {
            self.toggle_pause(app);
        }

        if actions.reset {
            self.session.reset();
            self.on_new_emulator(app);
        }

        if actions.save_state {
            self.session.save_state(self.session.slot());
        }

        if actions.load_state {
            self.session.load_state(self.session.slot());
        }

        if let Some(slot) = actions.select_slot {
            self.select_slot(app, slot);
        }

        if let Some(tag) = actions.select_model
            && let Some(new_model) = model_tag_to_model(tag)
        {
            self.session.set_model(new_model);
            update_model_checkmarks(app, tag);
            self.on_new_emulator(app);
        }

        if let Some(tag) = actions.select_filter
            && let Some(new_filter) = filter_tag_to_filter(tag)
        {
            self.scale_filter = new_filter;
            update_filter_checkmarks(app, tag);
            eprintln!("Filter: {:?}", self.scale_filter);
        }

        if actions.toggle_force_cpu {
            self.force_cpu = !self.force_cpu;
            update_force_cpu_checkmark(app, self.force_cpu);
            eprintln!("Force CPU: {}", if self.force_cpu { "on" } else { "off" });
        }

        if actions.toggle_printer {
            let on = !self.session.printer_attached();
            self.session.set_printer(on);
            update_printer_checkmark(app, on);
        }

        if actions.toggle_fps {
            self.show_fps_overlay = !self.show_fps_overlay;
            if let Some(main_menu) = app.mainMenu()
                && let Some(view_menu_item) = main_menu.itemAtIndex(2)
                && let Some(view_submenu) = view_menu_item.submenu()
                && let Some(fps_item) = view_submenu.itemWithTag(MENU_TAG_SHOW_FPS)
            {
                let state = if self.show_fps_overlay {
                    NSControlStateValueOn
                } else {
                    NSControlStateValueOff
                };
                fps_item.setState(state);
            }
        }

        if actions.open_controls {
            show_controls_panel(&mut self.key_map);
        }

        if let Some(idx) = actions.open_recent {
            let recents = load_recent_roms();
            if let Some(path_str) = recents.get(idx) {
                self.load_rom(PathBuf::from(path_str), mtm, app, window);
            }
        }

        if actions.clear_recent {
            save_recent_roms(&[]);
            rebuild_recent_menu(mtm, app, &[]);
            eprintln!("Recent ROMs cleared");
        }
    }

    /// Update gamepad, camera, and accelerometer input.
    fn update_input(&mut self) {
        let emu = &mut self.session.emu;
        // Gamepad
        self.gamepad.poll();
        if emu.has_rumble() {
            self.gamepad.ensure_haptics_ready();
        }
        self.gamepad
            .apply_to_emu(emu, &self.key_map, &self.keys_down);

        // Camera
        if let Some(ref cam) = self.camera
            && cam.read_frame(&mut self.camera_buf)
        {
            emu.set_camera_image(&self.camera_buf);
        }

        // Accelerometer: prioritize gamepad, fall back to MacBook built-in
        {
            const CENTER: f32 = 0x81D0_u16 as f32;
            const RANGE: f32 = 0x70_u16 as f32;
            let reading = if let Some((x, y, _z)) = self.gamepad.accel {
                // GCMotion uses Apple coordinate system: negate X for MBC7
                Some((-x, y))
            } else if let Some((x, y, _z)) = poll_accel(&self.accel_source) {
                // MacBook built-in: same Apple coordinate system, negate X
                Some((-x, y))
            } else {
                None
            };
            if let Some((gx, gy)) = reading {
                let mbc7_x = (CENTER + gx * RANGE).clamp(0.0, 65535.0) as u16;
                let mbc7_y = (CENTER + gy * RANGE).clamp(0.0, 65535.0) as u16;
                emu.set_accelerometer(mbc7_x, mbc7_y);
            }
        }
    }

    /// Run one session tick (rewind, fast-forward, slow motion, pause or
    /// audio-paced stepping), then queue its audio and drive rumble.
    fn step_emulation(&mut self) -> TickOutput {
        let hold = HoldInputs {
            rewind: self.keys_down.contains(&K_DELETE) || self.gamepad.l_shoulder,
            fast_forward: self.keys_down.contains(&K_TAB) || self.gamepad.r_shoulder,
            slow_motion: self.keys_down.contains(&K_MINUS),
        };
        let queued = self.audio.as_ref().map(|a| a.queued_frames());
        let out = self.session.tick(&hold, queued);
        if let Some(audio) = self.audio.as_mut() {
            audio.push(&out.audio);
        }

        // Rumble
        if self.session.emu.has_rumble() {
            self.gamepad.set_rumble(self.session.emu.drain_rumble());
        }
        out
    }

    /// Render the current frame. Handles occlusion check, filter dispatch, and Metal rendering.
    fn render(&mut self, window: &NSWindow, content_view: &objc2_app_kit::NSView) {
        let occluded = !window
            .occlusionState()
            .contains(objc2_app_kit::NSWindowOcclusionState::Visible);

        // Update drawable size on resize (use backing pixels for Retina)
        let (disp_w, disp_h);
        {
            let bounds = content_view.bounds();
            let scale = window.backingScaleFactor();
            disp_w = (bounds.size.width * scale) as usize;
            disp_h = (bounds.size.height * scale) as usize;
            if !occluded {
                self.renderer.layer.setContentsScale(scale);
                self.renderer.layer.setDrawableSize(NSSize::new(
                    bounds.size.width * scale,
                    bounds.size.height * scale,
                ));
            }
        }

        if occluded {
            return;
        }

        // Copy frame data to a persistent buffer to avoid borrowing self.emu across &mut self calls.
        let src_len = self.src_w * self.src_h;
        let raw_src: &[u32] = if self.is_sgb {
            self.session.emu.sgb_composited_frame()
        } else {
            self.session.emu.frame_buffer()
        };
        self.frame_copy.clear();
        self.frame_copy.extend_from_slice(&raw_src[..src_len]);

        // Take frame_copy out of self to avoid borrow conflict with &mut self methods
        let frame_copy = std::mem::take(&mut self.frame_copy);

        let gpu_rendered = if self.force_cpu {
            false
        } else {
            self.render_with_filter(&frame_copy, disp_w, disp_h)
        };

        if !gpu_rendered {
            self.render_cpu_fallback(&frame_copy, disp_w, disp_h);
        }

        // Put it back for reuse next frame
        self.frame_copy = frame_copy;
    }

    /// Try GPU-accelerated rendering for the current filter.
    /// Returns true if GPU-rendered (so CPU fallback can be skipped).
    fn render_with_filter(&mut self, raw_src: &[u32], disp_w: usize, disp_h: usize) -> bool {
        unsafe {
            let src_w = self.src_w;
            let src_h = self.src_h;

            // Vectorize: full 6-stage Metal compute pipeline
            if self.scale_filter == scaling::ScaleFilter::Vectorize {
                if self.renderer.vectorize_pipeline.is_none() {
                    self.renderer.vectorize_pipeline =
                        MetalVectorizePipeline::new(&self.renderer.device);
                }
                if let Some(ref mut vp) = self.renderer.vectorize_pipeline {
                    let s = (disp_w as f64 / src_w as f64).min(disp_h as f64 / src_h as f64) as f32;
                    let gw = (src_w as f32 * s).round() as u32;
                    let gh = (src_h as f32 * s).round() as u32;
                    if self.renderer.compute_out_w != gw || self.renderer.compute_out_h != gh {
                        let desc = MTLTextureDescriptor::new();
                        desc.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
                        desc.setWidth(gw as usize);
                        desc.setHeight(gh as usize);
                        desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
                        self.renderer.compute_out_tex = Some(
                            self.renderer
                                .device
                                .newTextureWithDescriptor(&desc)
                                .unwrap(),
                        );
                        self.renderer.compute_out_w = gw;
                        self.renderer.compute_out_h = gh;
                    }
                    let out_tex = self.renderer.compute_out_tex.as_ref().unwrap();
                    vp.run(
                        &self.renderer.device,
                        &self.renderer.command_queue,
                        raw_src,
                        src_w as u32,
                        src_h as u32,
                        gw,
                        gh,
                        s,
                        out_tex,
                    );
                    self.renderer.tex_w = gw;
                    self.renderer.tex_h = gh;
                    self.renderer.texture = out_tex.clone();
                    self.renderer.use_linear_blit = true;
                    self.renderer.render();
                    return true;
                }
            }

            // GPU compute scaling filters (Vectorize handled above)
            if self.scale_filter != scaling::ScaleFilter::Vectorize
                && let Some((_tex, gw, gh)) = self.renderer.run_scale_compute(
                    self.scale_filter,
                    raw_src,
                    src_w as u32,
                    src_h as u32,
                    disp_w as u32,
                    disp_h as u32,
                )
            {
                self.renderer.tex_w = gw;
                self.renderer.tex_h = gh;
                self.renderer.texture = self.renderer.compute_out_tex.as_ref().unwrap().clone();
                self.renderer.use_linear_blit = false;
                self.renderer.render();
                return true;
            }

            false
        }
    }

    /// CPU fallback rendering path for filters not handled by the GPU.
    fn render_cpu_fallback(&mut self, raw_src: &[u32], disp_w: usize, disp_h: usize) {
        unsafe {
            let src_w = self.src_w;
            let src_h = self.src_h;
            let gpu_rendered = false;

            let vec_scaled: Vec<u32>;
            let (frame_pixels, frame_w, frame_h): (&[u32], usize, usize) =
                if self.scale_filter == scaling::ScaleFilter::Nearest {
                    (raw_src, src_w, src_h)
                } else {
                    // Compute aspect-correct dimensions for adaptive filters
                    let scale = (disp_w as f64 / src_w as f64).min(disp_h as f64 / src_h as f64);
                    let fit_w = (src_w as f64 * scale).round() as usize;
                    let fit_h = (src_h as f64 * scale).round() as usize;
                    if let Some((s, w, h)) =
                        scaling::cpu_scale(self.scale_filter, raw_src, src_w, src_h, fit_w, fit_h)
                    {
                        vec_scaled = s;
                        (&vec_scaled, w as usize, h as usize)
                    } else {
                        (raw_src, src_w, src_h)
                    }
                };

            if gpu_rendered {
                return;
            }

            // Resize texture if dimensions changed
            if frame_w as u32 != self.renderer.tex_w || frame_h as u32 != self.renderer.tex_h {
                let tex_desc = MTLTextureDescriptor::new();
                tex_desc.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
                tex_desc.setWidth(frame_w);
                tex_desc.setHeight(frame_h);
                tex_desc.setUsage(MTLTextureUsage::ShaderRead);
                self.renderer.texture = self
                    .renderer
                    .device
                    .newTextureWithDescriptor(&tex_desc)
                    .unwrap();
                self.renderer.tex_w = frame_w as u32;
                self.renderer.tex_h = frame_h as u32;
            }

            // Convert 0x00RRGGBB -> BGRA8Unorm (set alpha to 0xFF)
            self.bgra_buf.resize(frame_w * frame_h, 0u32);
            for i in 0..(frame_w * frame_h) {
                self.bgra_buf[i] = 0xFF00_0000 | frame_pixels[i];
            }

            // Draw FPS overlay into pixel buffer
            if self.show_fps_overlay {
                let text = format!("FPS: {:.1}  {:.2}ms", self.overlay_fps, self.overlay_emu_ms);
                let scale = (frame_w / 160).clamp(1, 4);
                let fg = 0xFF00FF00;
                let bg = 0xC0000000;
                tiny_font::draw_string(
                    &mut self.bgra_buf,
                    frame_w,
                    frame_h,
                    &text,
                    2 * scale,
                    2 * scale,
                    fg,
                    bg,
                    scale,
                );
            }

            self.renderer.update_texture(&self.bgra_buf);
            self.renderer.use_linear_blit = false;
            self.renderer.render();
        }
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    // SAFETY: we're on the main thread
    let mtm = unsafe { MainThreadMarker::new_unchecked() };

    unsafe {
        let _pool = objc2_foundation::NSAutoreleasePool::new();

        let app = NSApplication::sharedApplication(mtm);
        app.setActivationPolicy(NSApplicationActivationPolicy::Regular);

        // Set up menu bar and action handler
        create_menu_bar(mtm, &app);
        let (_menu_handler, menu_actions) = menu_handler::create(&app);

        // Resolve ROM path
        let rom_path: PathBuf = if let Some(ref p) = cli.rom {
            p.clone()
        } else {
            #[allow(deprecated)] // NSApp.activate() needs macOS 14
            app.activateIgnoringOtherApps(true);
            open_rom_dialog().unwrap_or_else(|| std::process::exit(0))
        };

        ui_util::print_controls();
        eprintln!();

        let session = Session::new(
            &rom_path,
            SessionConfig {
                model: cli.model,
                bootrom: cli.bootrom.clone(),
                no_boot: cli.no_boot,
                lle: cli.lle,
                snes_rom: cli.snes_rom.clone(),
                printer: cli.printer,
                sample_rate: AUDIO_SAMPLE_RATE,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| {
            eprintln!("Failed to load '{}': {}", rom_path.display(), e);
            std::process::exit(1);
        });
        let frame_dur = session.frame_duration();

        // Load custom key mappings
        let key_map = load_key_map();

        // Initialize recent ROMs list and populate menu
        add_recent_rom(&rom_path.to_string_lossy());
        rebuild_recent_menu(mtm, &app, &load_recent_roms());

        update_printer_checkmark(&app, session.printer_attached());

        // Scaling filter
        let scale_filter = string_to_filter(&cli.filter);
        if scale_filter != scaling::ScaleFilter::Nearest {
            eprintln!("  Filter: {:?}", scale_filter);
        }
        // Set initial filter checkmark
        {
            let entries = filter_entries();
            for (i, (_, f)) in entries.iter().enumerate() {
                if *f == scale_filter {
                    update_filter_checkmarks(&app, MENU_TAG_FILTER_BASE + i as isize);
                    break;
                }
            }
        }

        let is_sgb = session.emu.is_sgb();
        let (tex_w, tex_h): (u32, u32) = if is_sgb { (256, 224) } else { (160, 144) };
        let src_w = tex_w as usize;
        let src_h = tex_h as usize;
        let win_w = tex_w * SCALE;
        let win_h = tex_h * SCALE;

        // ── Metal renderer ───────────────────────────────────────────────────
        let renderer = MetalRenderer::new(tex_w, tex_h);

        // ── Window ───────────────────────────────────────────────────────────
        let style = NSWindowStyleMask::Titled
            | NSWindowStyleMask::Closable
            | NSWindowStyleMask::Miniaturizable
            | NSWindowStyleMask::Resizable;

        let window = NSWindow::initWithContentRect_styleMask_backing_defer(
            NSWindow::alloc(mtm),
            NSRect::new(
                NSPoint::new(0.0, 0.0),
                NSSize::new(win_w as f64, win_h as f64),
            ),
            style,
            NSBackingStoreType::Buffered,
            false,
        );
        // `window` is an owning Retained; stop AppKit from also releasing
        // the window when the user closes it from the title bar.
        window.setReleasedWhenClosed(false);

        window.setTitle(&NSString::from_str(&window_title(&rom_path)));
        window.center();

        // Create a custom NSView subclass that suppresses key repeat sounds
        {
            let class_name = c"VBGameView";
            if AnyClass::get(class_name).is_none() {
                let superclass = AnyClass::get(c"NSView").unwrap();
                let mut builder = ClassBuilder::new(class_name, superclass).unwrap();
                unsafe extern "C" fn accepts_first_responder(
                    _this: *mut AnyObject,
                    _sel: Sel,
                ) -> Bool {
                    Bool::YES
                }
                unsafe extern "C" fn key_down(
                    _this: *mut AnyObject,
                    _sel: Sel,
                    _event: *mut AnyObject,
                ) { /* swallow */
                }
                builder.add_method(
                    sel!(acceptsFirstResponder),
                    accepts_first_responder as unsafe extern "C" fn(*mut AnyObject, Sel) -> Bool,
                );
                builder.add_method(
                    sel!(keyDown:),
                    key_down as unsafe extern "C" fn(*mut AnyObject, Sel, *mut AnyObject),
                );
                let _ = builder.register();
            }
            let game_view_class = AnyClass::get(class_name).unwrap();
            let content_rect = window.frame();
            let game_view: *mut AnyObject = msg_send![game_view_class, alloc];
            let game_view: *mut AnyObject = msg_send![game_view, initWithFrame: content_rect];
            // Cast to NSView for typed setContentView/makeFirstResponder
            let game_view_ref: &objc2_app_kit::NSView =
                &*(game_view as *const objc2_app_kit::NSView);
            window.setContentView(Some(game_view_ref));
            window.makeFirstResponder(Some(game_view_ref));
        }

        // Attach Metal layer to content view
        let content_view = window
            .contentView()
            .expect("window must have a content view");
        content_view.setWantsLayer(true);

        // Set the Metal layer
        use objc2_quartz_core::CALayer;
        let layer_ref: &CALayer = &renderer.layer;
        content_view.setLayer(Some(layer_ref));

        // Set layer scale and drawable size to backing pixels (Retina)
        let backing_scale = window.backingScaleFactor();
        renderer.layer.setContentsScale(backing_scale);
        renderer.layer.setDrawableSize(NSSize::new(
            win_w as f64 * backing_scale,
            win_h as f64 * backing_scale,
        ));

        window.makeKeyAndOrderFront(None);
        #[allow(deprecated)] // NSApp.activate() needs macOS 14
        app.activateIgnoringOtherApps(true);

        // ── Audio ────────────────────────────────────────────────────────────
        let audio = AudioOutput::start(AUDIO_SAMPLE_RATE);

        // ── Build AppState ───────────────────────────────────────────────────
        let state = Rc::new(RefCell::new(AppState {
            session,
            renderer,
            scale_filter,
            key_map,
            keys_down: HashSet::new(),
            gamepad: GamepadState::new(),
            audio,
            camera: None,
            camera_buf: [0u8; 128 * 112],
            accel_source: AccelSource::None,
            force_cpu: false,
            fps: ui_util::FpsCounter::new(),
            show_fps_overlay: false,
            overlay_fps: 0.0,
            overlay_emu_ms: 0.0,
            frame_copy: Vec::with_capacity(src_w * src_h),
            bgra_buf: Vec::with_capacity((tex_w * tex_h) as usize),
            src_w,
            src_h,
            is_sgb,
        }));
        state.borrow_mut().on_new_emulator(&app);

        // ── Frame timer ─────────────────────────────────────────────────────
        // A CFRunLoopTimer on kCFRunLoopCommonModes drives emulation + render.
        // This fires during both normal operation and menu tracking, keeping
        // audio-driven emulation smooth when macOS blocks the main thread for
        // modal menu interaction.
        let timer_info = FrameTimerInfo {
            state: Rc::clone(&state),
            window: window.clone(),
        };
        let timer = {
            let mut ctx = CFRunLoopTimerContext {
                version: 0,
                info: &raw const timer_info as *mut c_void,
                retain: None,
                release: None,
                copy_description: None,
            };
            let t = CFRunLoopTimerCreate(
                kCFAllocatorDefault,
                CFAbsoluteTimeGetCurrent(),
                frame_dur.as_secs_f64(),
                0,
                0,
                frame_timer_callback,
                &mut ctx,
            );
            CFRunLoopAddTimer(CFRunLoopGetMain(), t, kCFRunLoopCommonModes);
            t
        };

        // ── Main loop (events only) ─────────────────────────────────────────
        'running: loop {
            let _pool = objc2_foundation::NSAutoreleasePool::new();

            // Block until the next event or the frame timer fires.
            // NSDate with frame_dur timeout avoids busy-waiting while still
            // allowing the timer to wake us for the next frame.
            let wait_until =
                objc2_foundation::NSDate::dateWithTimeIntervalSinceNow(frame_dur.as_secs_f64());
            loop {
                let Some(event) = app.nextEventMatchingMask_untilDate_inMode_dequeue(
                    NSEventMask::Any,
                    Some(&wait_until),
                    NSDefaultRunLoopMode,
                    true,
                ) else {
                    break;
                };

                let event_type = event.r#type();
                let keycode: u16 = if event_type == NSEventType::KeyDown
                    || event_type == NSEventType::KeyUp
                    || event_type == NSEventType::FlagsChanged
                {
                    event.keyCode()
                } else {
                    0
                };

                // Released before sendEvent below: menu tracking runs inside
                // it and the frame timer must be able to borrow the state
                // meanwhile.
                let mut guard = state.borrow_mut();
                let state = &mut *guard;
                if event_type == NSEventType::KeyDown {
                    if keycode == K_ESCAPE {
                        break 'running;
                    }

                    state.keys_down.insert(keycode);

                    // Hotkeys ignore key auto-repeat, except frame advance,
                    // which steps repeatedly while Period is held.
                    if keycode == K_PERIOD {
                        state.session.request_frame_advance();
                    } else if event.isARepeat() {
                        // Auto-repeat of any other hotkey: ignore.
                    } else if keycode == K_SPACE {
                        state.toggle_pause(&app);
                    } else if keycode == K_F5 {
                        state.session.save_state(state.session.slot());
                    } else if keycode == K_F7 {
                        state.session.load_state(state.session.slot());
                    } else if let Some(slot) = keycode_to_slot(keycode) {
                        state.select_slot(&app, slot);
                    }

                    if let Some(btn) = state.key_map.get(&keycode).copied() {
                        state.session.emu.set_button(btn, true);
                    }
                } else if event_type == NSEventType::KeyUp {
                    state.keys_down.remove(&keycode);
                    if let Some(btn) = state.key_map.get(&keycode).copied() {
                        state.session.emu.set_button(btn, false);
                    }
                } else if event_type == NSEventType::FlagsChanged
                    && let Some(down) = modifier_key_down(keycode, event.modifierFlags().0)
                {
                    // Modifiers (e.g. Right Shift for Select) only act as
                    // mapped buttons, never as hotkeys.
                    if down {
                        state.keys_down.insert(keycode);
                    } else {
                        state.keys_down.remove(&keycode);
                    }
                    if let Some(btn) = state.key_map.get(&keycode).copied() {
                        state.session.emu.set_button(btn, down);
                    }
                }
                drop(guard);

                // Dispatch events so menus and window chrome work.
                // During menu tracking, sendEvent blocks — but the frame timer
                // fires on kCFRunLoopCommonModes and keeps emulation running.
                app.sendEvent(&event);
            }

            // Handle menu actions
            let actions = menu_actions.borrow_mut().take_all();
            if actions.quit {
                break 'running;
            }
            // The Open dialog and controls panel run nested modal loops
            // inside this call; the frame timer skips its ticks meanwhile.
            state
                .borrow_mut()
                .handle_menu_actions(actions, mtm, &app, &window);

            // Check if window was closed
            if !window.isVisible() {
                break 'running;
            }
        }

        // Cleanup
        CFRunLoopTimerInvalidate(timer);
        CFRelease(timer);
        drop(timer_info);

        let mut guard = state.borrow_mut();
        let state = &mut *guard;
        close_accel(&state.accel_source);
        drop(state.camera.take());

        state.session.flush_save();
    }
}
