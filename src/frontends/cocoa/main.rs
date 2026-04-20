use vibeboy::*;

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
use emulator::Emulator;
use model::GbModel;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::ffi::c_void;
use std::time::{Duration, Instant};

use objc2::{msg_send, sel, MainThreadOnly};
use objc2::rc::Retained;
use objc2::runtime::{AnyClass, AnyObject, Bool, ClassBuilder, Sel};
use objc2_app_kit::{
    NSApplication, NSApplicationActivationPolicy, NSBackingStoreType, NSControlStateValueOff,
    NSControlStateValueOn, NSEventMask, NSEventType, NSWindow, NSWindowStyleMask,
};
use objc2_foundation::{
    MainThreadMarker, NSDefaultRunLoopMode, NSPoint, NSRect, NSSize, NSString,
};
use objc2_metal::*;

use ui_util::{frame_duration, parse_filter};

use accel::{init_accel, poll_accel, close_accel};
use audio::{AudioRingBuffer, SharedAudioBuffer, setup_audio};
use camera::CameraCapture;
use controls::{show_controls_panel, open_rom_dialog};
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

use ui_util::auto_detect_model;

#[derive(Parser)]
#[command(name = "vibeboy_cocoa", about = "Game Boy / Game Boy Color emulator (macOS native)")]
struct Cli {
    rom: Option<PathBuf>,
    #[arg(long)]
    bootrom: Option<PathBuf>,
    #[arg(long, value_parser = ui_util::parse_model)]
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
    fn CFRunLoopRemoveTimer(rl: *mut c_void, timer: *mut c_void, mode: *const c_void);
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

/// Context passed to the frame timer callback.
struct FrameTimerInfo {
    state: *mut AppState,
    window: *const NSWindow,
    frame_start: *mut Instant,
    running: bool,
}

/// Called by CFRunLoopTimer at frame rate. Steps emulation, renders, updates
/// FPS, and flushes saves. Fires on kCFRunLoopCommonModes so it continues
/// during menu tracking, keeping audio-driven emulation smooth.
unsafe extern "C" fn frame_timer_callback(_timer: *mut c_void, info: *mut c_void) {
    let ctx = &mut *(info as *mut FrameTimerInfo);
    if !ctx.running { return; }
    let state = &mut *ctx.state;
    let window = &*ctx.window;
    let frame_start = &mut *ctx.frame_start;

    let _pool = objc2_foundation::NSAutoreleasePool::new();

    state.update_input();
    state.step_emulation(frame_start);

    if let Some(content_view) = window.contentView() {
        state.render(window, &content_view);
    }

    let emu_time = frame_start.elapsed();
    if let Some((f, ms)) = state.fps.update(1, emu_time) {
        state.overlay_fps = f;
        state.overlay_emu_ms = ms;
    }

    state.sav_flusher.poll(&state.emu);
}

// ── AppState ─────────────────────────────────────────────────────────────────

struct AppState {
    emu: Emulator,
    rom: std::sync::Arc<[u8]>,
    rom_path: PathBuf,
    model: GbModel,
    forced_model: Option<GbModel>,
    renderer: MetalRenderer,
    scale_filter: scaling::ScaleFilter,
    key_map: std::collections::HashMap<u16, u8>,
    keys_down: HashSet<u16>,
    gamepad: GamepadState,
    // audio_unit must be declared before audio_ring so it's dropped first
    // (stops the callback before the ring buffer is freed)
    audio_unit: Option<audio::AudioUnitHandle>,
    audio_ring: SharedAudioBuffer,
    camera: Option<CameraCapture>,
    camera_buf: [u8; 128 * 112],
    accel_source: AccelSource,
    sav_flusher: ui_util::SavFlusher,
    paused: bool,
    step_one_frame: bool,
    current_slot: usize,
    force_cpu: bool,
    fps: ui_util::FpsCounter,
    show_fps_overlay: bool,
    overlay_fps: f64,
    overlay_emu_ms: f64,
    emu_time_debt: Duration,
    frame_dur: Duration,
    frame_copy: Vec<u32>,
    bgra_buf: Vec<u32>,
    src_w: usize,
    src_h: usize,
    is_sgb: bool,
    no_boot: bool,
}

impl AppState {
    /// Update SGB state and source dimensions after creating a new emulator.
    fn update_src_dims(&mut self) {
        self.is_sgb = self.emu.is_sgb();
        if self.is_sgb {
            self.src_w = 256;
            self.src_h = 224;
        } else {
            self.src_w = 160;
            self.src_h = 144;
        }
    }

    /// Load a new ROM, resetting emulator state. Updates window title and recent ROMs.
    unsafe fn load_rom(
        &mut self,
        path: PathBuf,
        rom_data: impl Into<std::sync::Arc<[u8]>>,
        mtm: MainThreadMarker,
        app: &NSApplication,
        window: &NSWindow,
    ) {
        let title_str = format!("VibeBoy \u{2014} {}",
            path.file_name().unwrap_or_default().to_string_lossy());
        let title = NSString::from_str(&title_str);
        window.setTitle(&title);
        add_recent_rom(&path.to_string_lossy());
        rebuild_recent_menu(mtm, app, &load_recent_roms());
        self.rom = rom_data.into();
        self.rom_path = path;
        self.model = self.forced_model.unwrap_or_else(|| auto_detect_model(&self.rom));
        let boot_rom = ui_util::load_boot_rom(self.model, None, self.no_boot);
        self.emu = Emulator::new(self.rom.clone(), boot_rom, self.model, None, clock::default_clock(), AUDIO_SAMPLE_RATE);
        self.update_src_dims();
        ui_util::load_sav(&mut self.emu, &self.rom_path);
        self.sav_flusher = ui_util::SavFlusher::new(&self.emu, &self.rom_path);
        self.paused = false;
        eprintln!("Loaded: {}", self.rom_path.display());
    }

    /// Handle all pending menu actions.
    unsafe fn handle_menu_actions(
        &mut self,
        actions: MenuActions,
        mtm: MainThreadMarker,
        app: &NSApplication,
        window: &NSWindow,
    ) {
        if actions.open_rom {
            if let Some(path) = open_rom_dialog() {
                if let Ok(rom_data) = fs::read(&path) {
                    self.load_rom(path, rom_data, mtm, app, window);
                }
            }
        }

        if actions.pause_toggle {
            self.paused = !self.paused;
            eprintln!("{}", if self.paused { "Paused" } else { "Resumed" });
            if let Some(main_menu) = app.mainMenu() {
                if let Some(emu_menu) = main_menu.itemAtIndex(3) {
                    if let Some(submenu) = emu_menu.submenu() {
                        if let Some(pause_item) = submenu.itemWithTag(MENU_TAG_PAUSE) {
                            let label = if self.paused { "Resume" } else { "Pause" };
                            pause_item.setTitle(&NSString::from_str(label));
                        }
                    }
                }
            }
        }

        if actions.reset {
            let boot_rom = ui_util::load_boot_rom(self.model, None, self.no_boot);
            self.emu = Emulator::new(self.rom.clone(), boot_rom, self.model, None, clock::default_clock(), AUDIO_SAMPLE_RATE);
            self.update_src_dims();
            ui_util::load_sav(&mut self.emu, &self.rom_path);
            self.sav_flusher = ui_util::SavFlusher::new(&self.emu, &self.rom_path);
            self.paused = false;
            eprintln!("Reset");
        }

        if actions.save_state {
            ui_util::save_state_to_slot(&mut self.emu, &self.rom_path, self.current_slot);
        }

        if actions.load_state {
            ui_util::load_state_from_slot(&mut self.emu, &self.rom_path, self.current_slot);
        }

        if let Some(slot) = actions.select_slot {
            self.current_slot = slot;
            eprintln!("Slot {} selected", self.current_slot);
            update_slot_checkmarks(app, slot);
        }

        if let Some(tag) = actions.select_model {
            if let Some(new_model) = model_tag_to_model(tag) {
                self.forced_model = new_model;
                self.model = self.forced_model.unwrap_or_else(|| auto_detect_model(&self.rom));
                // Use auto-detected boot ROM for the new model (ignore explicit --bootrom)
                let boot_rom = ui_util::load_boot_rom(self.model, None, self.no_boot);
                let model_name = self.forced_model.map(|m| format!("{}", m)).unwrap_or_else(|| "Auto".to_string());
                eprintln!("Hardware model: {} (boot ROM: {})", model_name,
                    if boot_rom.is_some() { "loaded" } else { "none" });
                self.emu = Emulator::new(self.rom.clone(), boot_rom, self.model, None, clock::default_clock(), AUDIO_SAMPLE_RATE);
                self.update_src_dims();
                ui_util::load_sav(&mut self.emu, &self.rom_path);
                self.sav_flusher = ui_util::SavFlusher::new(&self.emu, &self.rom_path);
                update_model_checkmarks(app, tag);
                self.paused = false;
            }
        }

        if let Some(tag) = actions.select_filter {
            if let Some(new_filter) = filter_tag_to_filter(tag) {
                self.scale_filter = new_filter;
                update_filter_checkmarks(app, tag);
                eprintln!("Filter: {:?}", self.scale_filter);
            }
        }

        if actions.toggle_force_cpu {
            self.force_cpu = !self.force_cpu;
            update_force_cpu_checkmark(app, self.force_cpu);
            eprintln!("Force CPU: {}", if self.force_cpu { "on" } else { "off" });
        }

        if actions.toggle_printer {
            let is_printer = self.emu.serial_device_as_any().is::<printer::Printer>();
            if is_printer {
                self.emu.attach_serial_device(Box::new(serial::Disconnected));
                eprintln!("Game Boy Printer disconnected");
            } else {
                self.emu.attach_serial_device(
                    Box::new(printer::Printer::new(self.model.cpu_clock_rate()))
                );
                eprintln!("Game Boy Printer connected");
            }
            if let Some(main_menu) = app.mainMenu() {
                if let Some(emu_menu_item) = main_menu.itemAtIndex(3) {
                    if let Some(emu_submenu) = emu_menu_item.submenu() {
                        if let Some(printer_menu_item) = emu_submenu.itemWithTag(MENU_TAG_PRINTER) {
                            let state = if !is_printer { NSControlStateValueOn } else { NSControlStateValueOff };
                            printer_menu_item.setState(state);
                        }
                    }
                }
            }
        }

        if actions.toggle_fps {
            self.show_fps_overlay = !self.show_fps_overlay;
            if let Some(main_menu) = app.mainMenu() {
                if let Some(view_menu_item) = main_menu.itemAtIndex(2) {
                    if let Some(view_submenu) = view_menu_item.submenu() {
                        if let Some(fps_item) = view_submenu.itemWithTag(MENU_TAG_SHOW_FPS) {
                            let state = if self.show_fps_overlay { NSControlStateValueOn } else { NSControlStateValueOff };
                            fps_item.setState(state);
                        }
                    }
                }
            }
        }

        if actions.open_controls {
            show_controls_panel(&mut self.key_map);
        }

        if let Some(idx) = actions.open_recent {
            let recents = load_recent_roms();
            if let Some(path_str) = recents.get(idx) {
                let path = PathBuf::from(path_str);
                if let Ok(rom_data) = fs::read(&path) {
                    self.load_rom(path, rom_data, mtm, app, window);
                } else {
                    eprintln!("Failed to read: {}", path_str);
                }
            }
        }

        if actions.clear_recent {
            save_recent_roms(&[]);
            rebuild_recent_menu(mtm, app, &[]);
            eprintln!("Recent ROMs cleared");
        }
    }

    /// Update gamepad, camera, and accelerometer input.
    unsafe fn update_input(&mut self) {
        // Gamepad
        self.gamepad.poll();
        if self.emu.has_rumble() {
            self.gamepad.ensure_haptics_ready();
        }
        self.gamepad.apply_to_emu(&mut self.emu, &self.key_map, &self.keys_down);

        // Camera
        if let Some(ref cam) = self.camera {
            if cam.read_frame(&mut self.camera_buf) {
                self.emu.set_camera_image(&self.camera_buf);
            }
        }

        // Accelerometer: prioritize gamepad, fall back to MacBook built-in
        {
            const CENTER: f32 = 0x81D0 as u16 as f32;
            const RANGE: f32 = 0x70 as u16 as f32;
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
                self.emu.set_accelerometer(mbc7_x, mbc7_y);
            }
        }
    }

    /// Step emulation: rewind, fast-forward, or normal frame stepping + audio drain.
    unsafe fn step_emulation(&mut self, frame_start: &mut Instant) {
        let backspace_held = self.keys_down.contains(&K_DELETE) || self.gamepad.l_shoulder;
        let fast_forward = self.keys_down.contains(&K_TAB) || self.gamepad.r_shoulder;
        let slow_motion = self.keys_down.contains(&K_MINUS);
        self.emu.set_rewinding(backspace_held);

        // Accumulate elapsed time, capped to prevent catch-up bursts if the app
        // is backgrounded or otherwise stalled for an extended period.
        self.emu_time_debt += frame_start.elapsed();
        *frame_start = Instant::now();
        let max_debt = self.frame_dur * 3;
        if self.emu_time_debt > max_debt {
            self.emu_time_debt = max_debt;
        }

        if self.paused && self.step_one_frame {
            self.emu.step_frame();
            self.step_one_frame = false;
            self.emu_time_debt = Duration::ZERO;
        } else if self.paused {
            self.emu_time_debt = Duration::ZERO;
        } else if backspace_held {
            let mut all_audio = Vec::with_capacity(19200);
            for _ in 0..3 {
                self.emu.rewind_one_frame();
                all_audio.extend_from_slice(&self.emu.drain_audio_samples());
            }
            ui_util::reverse_audio(&mut all_audio);
            let resampled = ui_util::downsample_audio(&all_audio, 3);
            if let Ok(mut ring) = self.audio_ring.lock() {
                ring.write(&resampled);
            }
            self.emu_time_debt = Duration::ZERO;
        } else if fast_forward {
            for _ in 0..4 {
                self.emu.step_frame();
            }
            self.emu_time_debt = Duration::ZERO;
        } else if slow_motion {
            // Half speed: step at 2x the normal frame duration
            let slow_dur = self.frame_dur * 2;
            while self.emu_time_debt >= slow_dur {
                self.emu.step_frame();
                self.emu_time_debt -= slow_dur;
            }
        } else {
            // Audio-driven timing: use ring buffer fill level to decide
            // how many frames to step, synchronizing to the audio device's
            // clock. Time debt is still consumed for sleep-based pacing.
            let samples_per_frame = AUDIO_SAMPLE_RATE as usize / 60 * 2; // stereo
            let target_fill = samples_per_frame * 3; // ~50ms
            let max_fill = samples_per_frame * 8;    // ~133ms
            let queued = self.audio_ring.lock().map(|r| r.len()).unwrap_or(target_fill);

            let frames_needed = if queued < target_fill / 2 {
                2u32
            } else if queued < target_fill {
                1
            } else if queued > max_fill {
                0
            } else {
                1
            };

            for _ in 0..frames_needed {
                self.emu.step_frame();
            }
            // Always consume time debt so sleep pacing still works
            self.emu_time_debt = self.emu_time_debt.saturating_sub(self.frame_dur);
        }

        // Printer
        ui_util::check_and_save_prints(&mut self.emu);

        // Rumble
        if self.emu.has_rumble() {
            self.gamepad.set_rumble(self.emu.drain_rumble());
        }

        // Audio
        let samples = self.emu.drain_audio_samples();
        if !samples.is_empty() {
            let to_write: std::borrow::Cow<[f32]> = if fast_forward {
                std::borrow::Cow::Owned(ui_util::downsample_audio(&samples, 4))
            } else {
                std::borrow::Cow::Borrowed(&samples[..])
            };
            if let Ok(mut ring) = self.audio_ring.lock() {
                ring.write(&to_write);
            }
        }
    }

    /// Render the current frame. Handles occlusion check, filter dispatch, and Metal rendering.
    unsafe fn render(&mut self, window: &NSWindow, content_view: &objc2_app_kit::NSView) {
        let occluded = !window.occlusionState().contains(objc2_app_kit::NSWindowOcclusionState::Visible);

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
                    bounds.size.width * scale, bounds.size.height * scale,
                ));
            }
        }

        if occluded {
            return;
        }

        // Copy frame data to a persistent buffer to avoid borrowing self.emu across &mut self calls.
        let src_len = self.src_w * self.src_h;
        let raw_src: &[u32] = if self.is_sgb {
            self.emu.sgb_composited_frame()
        } else {
            self.emu.frame_buffer()
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
    unsafe fn render_with_filter(&mut self, raw_src: &[u32], disp_w: usize, disp_h: usize) -> bool {
        let src_w = self.src_w;
        let src_h = self.src_h;

        // Vectorize: full 6-stage Metal compute pipeline
        if self.scale_filter == scaling::ScaleFilter::Vectorize {
            if self.renderer.vectorize_pipeline.is_none() {
                self.renderer.vectorize_pipeline = MetalVectorizePipeline::new(&self.renderer.device);
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
                    self.renderer.compute_out_tex = Some(self.renderer.device.newTextureWithDescriptor(&desc).unwrap());
                    self.renderer.compute_out_w = gw;
                    self.renderer.compute_out_h = gh;
                }
                let out_tex = self.renderer.compute_out_tex.as_ref().unwrap();
                vp.run(&self.renderer.device, &self.renderer.command_queue, raw_src,
                    src_w as u32, src_h as u32, gw, gh, s, out_tex);
                self.renderer.tex_w = gw;
                self.renderer.tex_h = gh;
                self.renderer.texture = out_tex.clone();
                self.renderer.use_linear_blit = true;
                self.renderer.render();
                return true;
            }
        }

        // GPU compute scaling filters (Vectorize handled above)
        if self.scale_filter != scaling::ScaleFilter::Vectorize {
            if let Some((_tex, gw, gh)) = self.renderer.run_scale_compute(
                self.scale_filter, raw_src, src_w as u32, src_h as u32,
                disp_w as u32, disp_h as u32,
            ) {
                self.renderer.tex_w = gw;
                self.renderer.tex_h = gh;
                self.renderer.texture = self.renderer.compute_out_tex.as_ref().unwrap().clone();
                self.renderer.use_linear_blit = false;
                self.renderer.render();
                return true;
            }
        }

        false
    }

    /// CPU fallback rendering path for filters not handled by the GPU.
    unsafe fn render_cpu_fallback(&mut self, raw_src: &[u32], disp_w: usize, disp_h: usize) {
        let src_w = self.src_w;
        let src_h = self.src_h;
        let mut gpu_rendered = false;

        let mut vec_scaled: Vec<u32>;
        let (frame_pixels, frame_w, frame_h): (&[u32], usize, usize) =
            if self.scale_filter == scaling::ScaleFilter::Nearest {
                (raw_src, src_w, src_h)
            } else {
                // Compute aspect-correct dimensions for adaptive filters
                let scale = (disp_w as f64 / src_w as f64).min(disp_h as f64 / src_h as f64);
                let fit_w = (src_w as f64 * scale).round() as usize;
                let fit_h = (src_h as f64 * scale).round() as usize;
                if let Some((s, w, h)) = scaling::cpu_scale(
                    self.scale_filter, raw_src, src_w, src_h, fit_w, fit_h,
                ) {
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
            tex_desc.setWidth(frame_w as usize);
            tex_desc.setHeight(frame_h as usize);
            tex_desc.setUsage(MTLTextureUsage::ShaderRead);
            self.renderer.texture = self.renderer.device.newTextureWithDescriptor(&tex_desc).unwrap();
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
            let scale = ((frame_w / 160).max(1)).min(4);
            let fg = 0xFF00FF00;
            let bg = 0xC0000000;
            tiny_font::draw_string(
                &mut self.bgra_buf, frame_w, frame_h,
                &text, 2 * scale, 2 * scale, fg, bg, scale,
            );
        }

        self.renderer.update_texture(&self.bgra_buf);
        self.renderer.use_linear_blit = false;
        self.renderer.render();
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
        let (_menu_handler, menu_actions_ptr) = menu_handler::create(&app);
        let menu_actions = &mut *menu_actions_ptr;

        // Resolve ROM path
        let rom_path: PathBuf = if let Some(ref p) = cli.rom {
            p.clone()
        } else {
            app.activateIgnoringOtherApps(true);
            open_rom_dialog().unwrap_or_else(|| std::process::exit(0))
        };

        let rom: std::sync::Arc<[u8]> = fs::read(&rom_path).unwrap_or_else(|e| {
            eprintln!("Failed to read ROM '{}': {}", rom_path.display(), e);
            std::process::exit(1);
        }).into();

        let forced_model: Option<GbModel> = cli.model;
        let model = forced_model.unwrap_or_else(|| auto_detect_model(&rom));
        let frame_dur = frame_duration(model);

        let boot_rom = ui_util::load_boot_rom(model, cli.bootrom.as_deref(), cli.no_boot);

        if boot_rom.is_some() {
            eprintln!("Boot ROM loaded — executing boot sequence.");
        }

        let snes_rom: Option<Vec<u8>> = if model.is_sgb() && cli.lle {
            if let Some(ref p) = cli.snes_rom {
                Some(fs::read(p).unwrap_or_else(|e| {
                    eprintln!("Failed to read SNES ROM '{}': {}", p.display(), e);
                    std::process::exit(1);
                }))
            } else {
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

        ui_util::print_controls();
        eprintln!();

        let mut emu = Emulator::new(rom.clone(), boot_rom, model, snes_rom, clock::default_clock(), AUDIO_SAMPLE_RATE);
        ui_util::load_sav(&mut emu, &rom_path);
        let sav_flusher = ui_util::SavFlusher::new(&emu, &rom_path);

        // Load custom key mappings
        let key_map = load_key_map();

        // Initialize recent ROMs list and populate menu
        add_recent_rom(&rom_path.to_string_lossy());
        rebuild_recent_menu(mtm, &app, &load_recent_roms());

        if cli.printer {
            emu.attach_serial_device(
                Box::new(printer::Printer::new(model.cpu_clock_rate())));
            eprintln!("Game Boy Printer connected — images will be saved to prints/");
            // Set checkmark on printer menu item
            if let Some(main_menu) = app.mainMenu() {
                if let Some(emu_menu_item) = main_menu.itemAtIndex(3) {
                    if let Some(emu_submenu) = emu_menu_item.submenu() {
                        if let Some(printer_menu_item) = emu_submenu.itemWithTag(MENU_TAG_PRINTER) {
                            printer_menu_item.setState(NSControlStateValueOn);
                        }
                    }
                }
            }
        }

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

        let is_sgb = emu.is_sgb();
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
            NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(win_w as f64, win_h as f64)),
            style,
            NSBackingStoreType::Buffered,
            false,
        );

        let title_str = format!("VibeBoy \u{2014} {}",
            rom_path.file_name().unwrap_or_default().to_string_lossy());
        let title = NSString::from_str(&title_str);
        window.setTitle(&title);
        window.center();

        // Create a custom NSView subclass that suppresses key repeat sounds
        {
            let class_name = c"VBGameView";
            if AnyClass::get(class_name).is_none() {
                let superclass = AnyClass::get(c"NSView").unwrap();
                let mut builder = ClassBuilder::new(class_name, superclass).unwrap();
                unsafe extern "C" fn accepts_first_responder(_this: *mut AnyObject, _sel: Sel) -> Bool { Bool::YES }
                unsafe extern "C" fn key_down(_this: *mut AnyObject, _sel: Sel, _event: *mut AnyObject) { /* swallow */ }
                builder.add_method(sel!(acceptsFirstResponder), accepts_first_responder as unsafe extern "C" fn(*mut AnyObject, Sel) -> Bool);
                builder.add_method(sel!(keyDown:), key_down as unsafe extern "C" fn(*mut AnyObject, Sel, *mut AnyObject));
                let _ = builder.register();
            }
            let game_view_class = AnyClass::get(class_name).unwrap();
            let content_rect = window.frame();
            let game_view: *mut AnyObject = msg_send![game_view_class, alloc];
            let game_view: *mut AnyObject = msg_send![game_view, initWithFrame: content_rect];
            // Cast to NSView for typed setContentView/makeFirstResponder
            let game_view_ref: &objc2_app_kit::NSView = &*(game_view as *const objc2_app_kit::NSView);
            window.setContentView(Some(game_view_ref));
            window.makeFirstResponder(Some(game_view_ref));
        }

        // Attach Metal layer to content view
        let content_view = window.contentView().expect("window must have a content view");
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
        app.activateIgnoringOtherApps(true);

        // ── Audio ────────────────────────────────────────────────────────────
        let audio_ring: SharedAudioBuffer =
            Arc::new(Mutex::new(AudioRingBuffer::new(96_000 / 60 * 4 * 2))); // ~4 frames stereo
        let audio_unit = setup_audio(&audio_ring);

        // ── Camera ───────────────────────────────────────────────────────────
        let camera = if emu.has_camera() {
            CameraCapture::start()
        } else {
            None
        };

        // ── Accelerometer ────────────────────────────────────────────────────
        let accel_source = if emu.has_accelerometer() {
            init_accel()
        } else {
            AccelSource::None
        };

        // ── Build AppState ───────────────────────────────────────────────────
        let mut state = AppState {
            emu,
            rom: rom,
            rom_path,
            model,
            forced_model,
            renderer,
            scale_filter,
            key_map,
            keys_down: HashSet::new(),
            gamepad: GamepadState::new(),
            audio_unit,
            audio_ring,
            camera,
            camera_buf: [0u8; 128 * 112],
            accel_source,
            sav_flusher,
            paused: false,
            step_one_frame: false,
            current_slot: 0,
            force_cpu: false,
            fps: ui_util::FpsCounter::new(),
            show_fps_overlay: false,
            overlay_fps: 0.0,
            overlay_emu_ms: 0.0,
            emu_time_debt: Duration::ZERO,
            frame_dur,
            frame_copy: Vec::with_capacity((src_w * src_h) as usize),
            bgra_buf: Vec::with_capacity((tex_w * tex_h) as usize),
            src_w,
            src_h,
            is_sgb,
            no_boot: cli.no_boot,
        };

        let mut frame_start = Instant::now();

        // ── Frame timer ─────────────────────────────────────────────────────
        // A CFRunLoopTimer on kCFRunLoopCommonModes drives emulation + render.
        // This fires during both normal operation and menu tracking, keeping
        // audio-driven emulation smooth when macOS blocks the main thread for
        // modal menu interaction.
        let mut timer_info = FrameTimerInfo {
            state: &mut state as *mut AppState,
            window: &*window as *const NSWindow,
            frame_start: &mut frame_start as *mut Instant,
            running: true,
        };
        let timer = {
            let mut ctx = CFRunLoopTimerContext {
                version: 0,
                info: &mut timer_info as *mut FrameTimerInfo as *mut c_void,
                retain: None,
                release: None,
                copy_description: None,
            };
            let t = CFRunLoopTimerCreate(
                kCFAllocatorDefault,
                CFAbsoluteTimeGetCurrent(),
                frame_dur.as_secs_f64(),
                0, 0,
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
            let wait_until = objc2_foundation::NSDate::dateWithTimeIntervalSinceNow(
                frame_dur.as_secs_f64(),
            );
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

                if event_type == NSEventType::KeyDown {
                    if keycode == K_ESCAPE {
                        break 'running;
                    }

                    state.keys_down.insert(keycode);

                    if keycode == K_SPACE {
                        state.paused = !state.paused;
                        eprintln!("{}", if state.paused { "Paused" } else { "Resumed" });
                    } else if keycode == K_PERIOD {
                        if state.paused { state.step_one_frame = true; }
                    } else if keycode == K_F5 {
                        ui_util::save_state_to_slot(&mut state.emu, &state.rom_path, state.current_slot);
                    } else if keycode == K_F7 {
                        ui_util::load_state_from_slot(&mut state.emu, &state.rom_path, state.current_slot);
                    } else if let Some(slot) = keycode_to_slot(keycode) {
                        state.current_slot = slot;
                        eprintln!("Slot {} selected", state.current_slot);
                        update_slot_checkmarks(&app, slot);
                    }

                    if let Some(btn) = state.key_map.get(&keycode).copied() {
                        state.emu.set_button(btn, true);
                    }
                } else if event_type == NSEventType::KeyUp {
                    state.keys_down.remove(&keycode);
                    if let Some(btn) = state.key_map.get(&keycode).copied() {
                        state.emu.set_button(btn, false);
                    }
                }

                // Dispatch events so menus and window chrome work.
                // During menu tracking, sendEvent blocks — but the frame timer
                // fires on kCFRunLoopCommonModes and keeps emulation running.
                app.sendEvent(&event);
            }

            // Handle menu actions
            let actions = menu_actions.take_all();
            state.handle_menu_actions(actions, mtm, &app, &window);

            // Check if window was closed
            if !window.isVisible() {
                break 'running;
            }
        }

        // Cleanup
        timer_info.running = false;
        CFRunLoopTimerInvalidate(timer);
        CFRelease(timer);

        close_accel(&state.accel_source);
        drop(state.camera.take());

        state.sav_flusher.flush(&state.emu);
        drop(Box::from_raw(menu_actions_ptr));
    }
}
