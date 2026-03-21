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
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use objc2::{class, msg_send, sel, ClassType};
use objc2::rc::Retained;
use objc2::runtime::{AnyClass, AnyObject, Bool, ClassBuilder, Sel};
use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy, NSWindow};
use objc2_foundation::{MainThreadMarker, NSPoint, NSRect, NSSize, NSString};
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

fn keycode_to_slot(keycode: u16) -> Option<usize> {
    match keycode {
        18 => Some(0), // 1
        19 => Some(1), // 2
        20 => Some(2), // 3
        21 => Some(3), // 4
        23 => Some(4), // 5
        22 => Some(5), // 6
        26 => Some(6), // 7
        28 => Some(7), // 8
        25 => Some(8), // 9
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

        let rom = fs::read(&rom_path).unwrap_or_else(|e| {
            eprintln!("Failed to read ROM '{}': {}", rom_path.display(), e);
            std::process::exit(1);
        });

        let mut forced_model: Option<GbModel> = cli.model;
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

        let mut current_rom = rom;
        let mut current_rom_path = rom_path;
        let mut current_model = model;
        let mut emu = Emulator::new(current_rom.clone(), boot_rom, current_model, snes_rom);
        ui_util::load_sav(&mut emu, &current_rom_path);
        let mut sav_flusher = ui_util::SavFlusher::new(&emu, &current_rom_path);

        // Load custom key mappings
        let mut key_map = load_key_map();

        // Initialize recent ROMs list and populate menu
        add_recent_rom(&current_rom_path.to_string_lossy());
        rebuild_recent_menu(mtm, &app, &load_recent_roms());

        if cli.printer {
            let output_dir = std::path::Path::new("prints");
            emu.attach_serial_device(
                Box::new(printer::Printer::new(output_dir, model.cpu_clock_rate())));
            eprintln!("Game Boy Printer connected — images will be saved to prints/");
            // Set checkmark on printer menu item
            let main_menu: *mut AnyObject = msg_send![&*app, mainMenu];
            let emu_menu_item: *mut AnyObject = msg_send![main_menu, itemAtIndex: 3isize];
            let emu_submenu: *mut AnyObject = msg_send![emu_menu_item, submenu];
            let printer_menu_item: *mut AnyObject = msg_send![emu_submenu, itemWithTag: MENU_TAG_PRINTER];
            let _: () = msg_send![printer_menu_item, setState: 1isize];
        }

        // Scaling filter
        let mut scale_filter = string_to_filter(&cli.filter);
        let mut vec_cache: Option<vectorize::VectorizeCache> = match scale_filter {
            scaling::ScaleFilter::VectorizeLegacy => Some(vectorize::VectorizeCache::new_legacy(false)),
            scaling::ScaleFilter::VectorizeLegacyAdaptive => Some(vectorize::VectorizeCache::new_legacy(true)),
            _ => None,
        };
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

        // FPS overlay
        let mut show_fps_overlay = false;
        let mut overlay_fps: f64 = 0.0;
        let mut overlay_emu_ms: f64 = 0.0;

        let is_sgb = emu.is_sgb();
        let (tex_w, tex_h): (u32, u32) = if is_sgb { (256, 224) } else { (160, 144) };
        let src_w = tex_w as usize;
        let src_h = tex_h as usize;
        let win_w = tex_w * SCALE;
        let win_h = tex_h * SCALE;

        // ── Metal renderer ───────────────────────────────────────────────────
        let mut renderer = MetalRenderer::new(tex_w, tex_h);

        // ── Window ───────────────────────────────────────────────────────────
        // NSWindowStyleMask: Titled=1, Closable=2, Miniaturizable=4, Resizable=8
        let style: usize = 1 | 2 | 4 | 8;

        let window: *mut AnyObject = msg_send![class!(NSWindow), alloc];
        let window: *mut AnyObject = msg_send![window,
            initWithContentRect: NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(win_w as f64, win_h as f64))
            styleMask: style
            backing: 2usize  // NSBackingStoreBuffered
            defer: false
        ];

        let title_str = format!("VibeBoy \u{2014} {}",
            current_rom_path.file_name().unwrap_or_default().to_string_lossy());
        let title = NSString::from_str(&title_str);
        let _: () = msg_send![window, setTitle: &*title];
        let _: () = msg_send![window, center];

        // Create a custom NSView subclass that suppresses key repeat sounds
        {
            let class_name = c"VBGameView";
            if AnyClass::get(class_name).is_none() {
                let superclass = AnyClass::get(c"NSView").unwrap();
                let mut builder = ClassBuilder::new(class_name, superclass).unwrap();
                unsafe extern "C" fn accepts_first_responder(_this: *mut AnyObject, _sel: Sel) -> Bool { Bool::YES }
                unsafe extern "C" fn key_down(_this: *mut AnyObject, _sel: Sel, _event: *mut AnyObject) { /* swallow */ }
                unsafe {
                    builder.add_method(sel!(acceptsFirstResponder), accepts_first_responder as unsafe extern "C" fn(*mut AnyObject, Sel) -> Bool);
                    builder.add_method(sel!(keyDown:), key_down as unsafe extern "C" fn(*mut AnyObject, Sel, *mut AnyObject));
                }
                let _ = builder.register();
            }
            let game_view_class = AnyClass::get(class_name).unwrap();
            let content_rect: NSRect = msg_send![window, frame];
            let game_view: *mut AnyObject = msg_send![game_view_class, alloc];
            let game_view: *mut AnyObject = msg_send![game_view, initWithFrame: content_rect];
            let _: () = msg_send![window, setContentView: game_view];
            let _: () = msg_send![window, makeFirstResponder: game_view];
        }

        // Attach Metal layer to content view
        let content_view: *mut AnyObject = msg_send![window, contentView];
        let _: () = msg_send![content_view, setWantsLayer: true];

        // Set the Metal layer
        let raw_layer: *mut AnyObject = Retained::as_ptr(&renderer.layer) as *mut AnyObject;
        let _: () = msg_send![content_view, setLayer: raw_layer];

        // Set drawable size to logical points
        renderer.layer.setDrawableSize(NSSize::new(
            win_w as f64,
            win_h as f64,
        ));

        let _: () = msg_send![window, makeKeyAndOrderFront: std::ptr::null::<AnyObject>()];
        app.activateIgnoringOtherApps(true);

        // ── Audio ────────────────────────────────────────────────────────────
        let audio_ring: SharedAudioBuffer =
            Arc::new(Mutex::new(AudioRingBuffer::new(96_000 / 60 * 4 * 2))); // ~4 frames stereo
        let _audio_unit = setup_audio(&audio_ring);

        // ── Camera ───────────────────────────────────────────────────────────
        let camera = if emu.has_camera() {
            CameraCapture::start()
        } else {
            None
        };
        let mut camera_buf = [0u8; 128 * 112];

        // ── Accelerometer ────────────────────────────────────────────────────
        let accel_source = if emu.has_accelerometer() {
            init_accel()
        } else {
            AccelSource::None
        };

        // ── Key state + frame loop ───────────────────────────────────────────
        let mut keys_down = std::collections::HashSet::<u16>::new();
        let mut gamepad_state = GamepadState::new();
        let mut current_slot: usize = 0;
        let mut paused = false;
        let mut frame_start = Instant::now();
        let mut fps_counter = ui_util::FpsCounter::new();
        let mut bgra_buf: Vec<u32> = Vec::with_capacity((tex_w * tex_h) as usize);

        let mode = NSString::from_str("kCFRunLoopDefaultMode");

        'running: loop {
            let _pool = objc2_foundation::NSAutoreleasePool::new();

            // Poll events
            loop {
                let event: *mut AnyObject = msg_send![&*app,
                    nextEventMatchingMask: u64::MAX
                    untilDate: std::ptr::null::<AnyObject>() // don't wait
                    inMode: &*mode
                    dequeue: true
                ];

                if event.is_null() {
                    break;
                }

                let event_type: u64 = msg_send![event, type];
                // NSKeyDown=10, NSKeyUp=11, NSFlagsChanged=12
                let keycode: u16 = if event_type == 10
                    || event_type == 11
                    || event_type == 12
                {
                    msg_send![event, keyCode]
                } else {
                    0
                };

                if event_type == 10 { // NSKeyDown
                    if keycode == K_ESCAPE {
                        break 'running;
                    }

                    keys_down.insert(keycode);

                    if keycode == K_F5 {
                        ui_util::save_state_to_slot(&mut emu, &current_rom_path, current_slot);
                    } else if keycode == K_F7 {
                        ui_util::load_state_from_slot(&mut emu, &current_rom_path, current_slot);
                    } else if let Some(slot) = keycode_to_slot(keycode) {
                        current_slot = slot;
                        eprintln!("Slot {} selected", current_slot + 1);
                    }

                    if let Some(btn) = key_map.get(&keycode).copied() {
                        emu.set_button(btn, true);
                    }
                } else if event_type == 11 { // NSKeyUp
                    keys_down.remove(&keycode);
                    if let Some(btn) = key_map.get(&keycode).copied() {
                        emu.set_button(btn, false);
                    }
                }

                // Always dispatch events so menus and window chrome work
                let _: () = msg_send![&*app, sendEvent: event];
            }

            // ── Handle menu actions ──────────────────────────────────────────
            {
                let actions = menu_actions.take_all();

                if actions.open_rom {
                    if let Some(path) = open_rom_dialog() {
                        if let Ok(rom_data) = fs::read(&path) {
                            let title_str = format!("VibeBoy \u{2014} {}",
                                path.file_name().unwrap_or_default().to_string_lossy());
                            let title = NSString::from_str(&title_str);
                            let _: () = msg_send![window, setTitle: &*title];
                            add_recent_rom(&path.to_string_lossy());
                            rebuild_recent_menu(mtm, &app, &load_recent_roms());
                            current_rom = rom_data;
                            current_rom_path = path;
                            current_model = forced_model.unwrap_or_else(|| auto_detect_model(&current_rom));
                            emu = Emulator::new(current_rom.clone(), None, current_model, None);
                            ui_util::load_sav(&mut emu, &current_rom_path);
                            sav_flusher = ui_util::SavFlusher::new(&emu, &current_rom_path);
                            paused = false;
                            eprintln!("Loaded: {}", current_rom_path.display());
                        }
                    }
                }

                if actions.pause_toggle {
                    paused = !paused;
                    eprintln!("{}", if paused { "Paused" } else { "Resumed" });
                    let main_menu: *mut AnyObject = msg_send![&*app, mainMenu];
                    let emu_menu: *mut AnyObject = msg_send![main_menu, itemAtIndex: 3isize];
                    let submenu: *mut AnyObject = msg_send![emu_menu, submenu];
                    let pause_item: *mut AnyObject = msg_send![submenu, itemWithTag: MENU_TAG_PAUSE];
                    let label = if paused { "Resume" } else { "Pause" };
                    let label_ns = NSString::from_str(label);
                    let _: () = msg_send![pause_item, setTitle: &*label_ns];
                }

                if actions.reset {
                    emu = Emulator::new(current_rom.clone(), None, current_model, None);
                    ui_util::load_sav(&mut emu, &current_rom_path);
                    sav_flusher = ui_util::SavFlusher::new(&emu, &current_rom_path);
                    paused = false;
                    eprintln!("Reset");
                }

                if actions.save_state {
                    ui_util::save_state_to_slot(&mut emu, &current_rom_path, current_slot);
                }

                if actions.load_state {
                    ui_util::load_state_from_slot(&mut emu, &current_rom_path, current_slot);
                }

                if let Some(slot) = actions.select_slot {
                    current_slot = slot;
                    eprintln!("Slot {} selected", current_slot + 1);
                }

                if let Some(tag) = actions.select_model {
                    if let Some(new_model) = model_tag_to_model(tag) {
                        forced_model = new_model;
                        current_model = forced_model.unwrap_or_else(|| auto_detect_model(&current_rom));
                        emu = Emulator::new(current_rom.clone(), None, current_model, None);
                        ui_util::load_sav(&mut emu, &current_rom_path);
                        sav_flusher = ui_util::SavFlusher::new(&emu, &current_rom_path);
                        update_model_checkmarks(&app, tag);
                        paused = false;
                        let model_name = forced_model.map(|m| format!("{}", m)).unwrap_or_else(|| "Auto".to_string());
                        eprintln!("Hardware model: {}", model_name);
                    }
                }

                if let Some(tag) = actions.select_filter {
                    if let Some(new_filter) = filter_tag_to_filter(tag) {
                        scale_filter = new_filter;
                        vec_cache = match scale_filter {
                            scaling::ScaleFilter::VectorizeLegacy => Some(vectorize::VectorizeCache::new_legacy(false)),
                            scaling::ScaleFilter::VectorizeLegacyAdaptive => Some(vectorize::VectorizeCache::new_legacy(true)),
                            _ => None,
                        };
                        update_filter_checkmarks(&app, tag);
                        eprintln!("Filter: {:?}", scale_filter);
                    }
                }

                if actions.toggle_printer {
                    let is_printer = emu.serial_device_as_any().is::<printer::Printer>();
                    if is_printer {
                        emu.attach_serial_device(Box::new(serial::Disconnected));
                        eprintln!("Game Boy Printer disconnected");
                    } else {
                        let output_dir = std::path::Path::new("prints");
                        emu.attach_serial_device(
                            Box::new(printer::Printer::new(output_dir, current_model.cpu_clock_rate()))
                        );
                        eprintln!("Game Boy Printer connected");
                    }
                    // Update checkmark
                    let main_menu: *mut AnyObject = msg_send![&*app, mainMenu];
                    let emu_menu_item: *mut AnyObject = msg_send![main_menu, itemAtIndex: 3isize];
                    let emu_submenu: *mut AnyObject = msg_send![emu_menu_item, submenu];
                    let printer_menu_item: *mut AnyObject = msg_send![emu_submenu, itemWithTag: MENU_TAG_PRINTER];
                    let state: isize = if !is_printer { 1 } else { 0 };
                    let _: () = msg_send![printer_menu_item, setState: state];
                }

                if actions.toggle_fps {
                    show_fps_overlay = !show_fps_overlay;
                    let main_menu: *mut AnyObject = msg_send![&*app, mainMenu];
                    let view_menu_item: *mut AnyObject = msg_send![main_menu, itemAtIndex: 4isize];
                    let view_submenu: *mut AnyObject = msg_send![view_menu_item, submenu];
                    let fps_item: *mut AnyObject = msg_send![view_submenu, itemWithTag: MENU_TAG_SHOW_FPS];
                    let state: isize = if show_fps_overlay { 1 } else { 0 };
                    let _: () = msg_send![fps_item, setState: state];
                }

                if actions.open_controls {
                    show_controls_panel(&mut key_map);
                }

                if let Some(idx) = actions.open_recent {
                    let recents = load_recent_roms();
                    if let Some(path_str) = recents.get(idx) {
                        let path = PathBuf::from(path_str);
                        if let Ok(rom_data) = fs::read(&path) {
                            let title_str = format!("VibeBoy \u{2014} {}",
                                path.file_name().unwrap_or_default().to_string_lossy());
                            let title = NSString::from_str(&title_str);
                            let _: () = msg_send![window, setTitle: &*title];
                            add_recent_rom(path_str);
                            rebuild_recent_menu(mtm, &app, &load_recent_roms());
                            current_rom = rom_data;
                            current_rom_path = path;
                            current_model = forced_model.unwrap_or_else(|| auto_detect_model(&current_rom));
                            emu = Emulator::new(current_rom.clone(), None, current_model, None);
                            ui_util::load_sav(&mut emu, &current_rom_path);
                            sav_flusher = ui_util::SavFlusher::new(&emu, &current_rom_path);
                            paused = false;
                            eprintln!("Loaded: {}", current_rom_path.display());
                        } else {
                            eprintln!("Failed to read: {}", path_str);
                        }
                    }
                }

                if actions.clear_recent {
                    save_recent_roms(&[]);
                    rebuild_recent_menu(mtm, &app, &[]);
                    eprintln!("Recent ROMs cleared");
                }
            }

            // Check if window was closed
            let visible: bool = msg_send![window, isVisible];
            if !visible {
                break 'running;
            }

            // ── Gamepad ──────────────────────────────────────────────────────
            gamepad_state.poll();
            if emu.has_rumble() {
                gamepad_state.ensure_haptics_ready();
            }
            gamepad_state.apply_to_emu(&mut emu, &key_map, &keys_down);

            // ── Camera ───────────────────────────────────────────────────────
            if let Some(ref cam) = camera {
                if cam.read_frame(&mut camera_buf) {
                    emu.set_camera_image(&camera_buf);
                }
            }

            // ── Accelerometer ────────────────────────────────────────────────
            {
                const CENTER: f32 = 0x81D0 as u16 as f32;
                const RANGE: f32 = 0x70 as u16 as f32;
                let accel_reading = gamepad_state.accel
                    .map(|(x, y, z)| (x, y, z))
                    .or_else(|| poll_accel(&accel_source));
                if let Some((x, y, _z)) = accel_reading {
                    let mbc7_x = (CENTER + (-x) * RANGE).clamp(0.0, 65535.0) as u16;
                    let mbc7_y = (CENTER + y * RANGE).clamp(0.0, 65535.0) as u16;
                    emu.set_accelerometer(mbc7_x, mbc7_y);
                }
            }

            // ── Rewind / Fast-forward ────────────────────────────────────────
            let backspace_held = keys_down.contains(&K_DELETE) || gamepad_state.l_shoulder;
            let fast_forward = keys_down.contains(&K_TAB) || gamepad_state.r_shoulder;
            emu.set_rewinding(backspace_held);

            if !paused {
                if backspace_held {
                    let mut all_audio = Vec::new();
                    for _ in 0..3 {
                        emu.rewind_one_frame();
                        all_audio.extend_from_slice(&emu.drain_audio_samples());
                    }
                    ui_util::reverse_audio(&mut all_audio);
                    let resampled = ui_util::downsample_audio(&all_audio, 3);
                    if let Ok(mut ring) = audio_ring.lock() {
                        ring.write(&resampled);
                    }
                } else if fast_forward {
                    for _ in 0..4 {
                        emu.step_frame();
                    }
                } else {
                    emu.step_frame();
                }
            }

            // ── Rumble ───────────────────────────────────────────────────────
            if emu.has_rumble() {
                gamepad_state.set_rumble(emu.drain_rumble());
            }

            // ── Audio ────────────────────────────────────────────────────────
            let samples = emu.drain_audio_samples();
            if !samples.is_empty() {
                let to_write: std::borrow::Cow<[f32]> = if fast_forward {
                    std::borrow::Cow::Owned(ui_util::downsample_audio(&samples, 4))
                } else {
                    let max_samples = 3200 * 2;
                    if samples.len() <= max_samples {
                        std::borrow::Cow::Borrowed(&samples[..])
                    } else {
                        std::borrow::Cow::Borrowed(&samples[samples.len() - max_samples..])
                    }
                };
                if let Ok(mut ring) = audio_ring.lock() {
                    ring.write(&to_write);
                }
            }

            // ── Update drawable size on resize ─────────────────────────────
            let (disp_w, disp_h);
            {
                let bounds: NSRect = msg_send![content_view, bounds];
                disp_w = bounds.size.width as usize;
                disp_h = bounds.size.height as usize;
                renderer.layer.setDrawableSize(NSSize::new(
                    bounds.size.width, bounds.size.height,
                ));
            }

            // ── Render ───────────────────────────────────────────────────────
            {
                let raw_src: &[u32] = if is_sgb {
                    emu.sgb_composited_frame()
                } else {
                    emu.frame_buffer()
                };

                let mut gpu_rendered = false;
                // VectorizeGpu: full 6-stage Metal compute pipeline
                if scale_filter == scaling::ScaleFilter::VectorizeGpu {
                    if renderer.vectorize_pipeline.is_none() {
                        renderer.vectorize_pipeline = MetalVectorizePipeline::new(&renderer.device);
                    }
                    if let Some(ref mut vp) = renderer.vectorize_pipeline {
                        let s = (disp_w as f64 / src_w as f64).min(disp_h as f64 / src_h as f64) as f32;
                        let gw = (src_w as f32 * s).round() as u32;
                        let gh = (src_h as f32 * s).round() as u32;
                        if renderer.compute_out_w != gw || renderer.compute_out_h != gh {
                            let desc = MTLTextureDescriptor::new();
                            desc.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
                            desc.setWidth(gw as usize);
                            desc.setHeight(gh as usize);
                            desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
                            renderer.compute_out_tex = Some(renderer.device.newTextureWithDescriptor(&desc).unwrap());
                            renderer.compute_out_w = gw;
                            renderer.compute_out_h = gh;
                        }
                        let out_tex = renderer.compute_out_tex.as_ref().unwrap();
                        vp.run(&renderer.device, &renderer.command_queue, raw_src,
                            src_w as u32, src_h as u32, gw, gh, s, out_tex);
                        renderer.tex_w = gw;
                        renderer.tex_h = gh;
                        renderer.texture = out_tex.clone();
                        renderer.render();
                        gpu_rendered = true;
                    }
                }

                if !gpu_rendered && !matches!(scale_filter,
                    scaling::ScaleFilter::Nearest | scaling::ScaleFilter::Bilinear
                    | scaling::ScaleFilter::VectorizeLegacy | scaling::ScaleFilter::VectorizeLegacyAdaptive
                    | scaling::ScaleFilter::VectorizeDiffusion | scaling::ScaleFilter::VectorizeSplineDiffusion
                    | scaling::ScaleFilter::VectorizeSplineDiffusionAdaptive
                    | scaling::ScaleFilter::Vectorize | scaling::ScaleFilter::VectorizeAdaptive
                    | scaling::ScaleFilter::VectorizeGpu)
                {
                    if let Some((_tex, gw, gh)) = renderer.run_scale_compute(
                        scale_filter, raw_src, src_w as u32, src_h as u32,
                        disp_w as u32, disp_h as u32,
                    ) {
                        renderer.tex_w = gw;
                        renderer.tex_h = gh;
                        renderer.texture = renderer.compute_out_tex.as_ref().unwrap().clone();
                        renderer.render();
                        gpu_rendered = true;
                    }
                }

                // CPU fallback
                if !gpu_rendered {

                let mut vec_scaled: Vec<u32>;
                let (frame_pixels, frame_w, frame_h): (&[u32], usize, usize) =
                    if scale_filter == scaling::ScaleFilter::Nearest {
                        (raw_src, src_w, src_h)
                    } else if matches!(scale_filter,
                        scaling::ScaleFilter::VectorizeLegacy | scaling::ScaleFilter::VectorizeLegacyAdaptive)
                    {
                        let s = (disp_w as f64 / src_w as f64).min(disp_h as f64 / src_h as f64);
                        let adaptive = matches!(scale_filter, scaling::ScaleFilter::VectorizeLegacyAdaptive);
                        let cache = vec_cache.get_or_insert_with(|| vectorize::VectorizeCache::new_legacy(adaptive));
                        let (paths, bg) = cache.get_paths(raw_src, src_w, src_h);
                        let (gpu_edges, row_ranges, edge_indices, ow, oh) =
                            vectorize::rasterize::prepare_gpu_edges_v2(paths, bg, s, src_w, src_h);
                        if ow > 0 && oh > 0 {
                            renderer.run_scanline_rasterize(&gpu_edges, &row_ranges, &edge_indices, ow, oh, bg);
                            renderer.render();
                            gpu_rendered = true;
                        }
                        vec_scaled = Vec::new();
                        (&[] as &[u32], 0, 0)
                    } else if scale_filter == scaling::ScaleFilter::VectorizeDiffusion {
                        let s = (disp_w as f64 / src_w as f64).min(disp_h as f64 / src_h as f64);
                        let scale = s.round().max(1.0) as u32;
                        let ow = src_w as u32 * scale;
                        let oh = src_h as u32 * scale;
                        renderer.run_diffusion_rasterize(raw_src, src_w as u32, src_h as u32, ow, oh, scale);
                        renderer.render();
                        gpu_rendered = true;
                        vec_scaled = Vec::new();
                        (&[] as &[u32], 0, 0)
                    } else if matches!(scale_filter,
                        scaling::ScaleFilter::VectorizeSplineDiffusion
                        | scaling::ScaleFilter::VectorizeSplineDiffusionAdaptive)
                    {
                        let s = (disp_w as f64 / src_w as f64).min(disp_h as f64 / src_h as f64);
                        let scale = s.round().max(1.0) as u32;
                        let adaptive = matches!(scale_filter, scaling::ScaleFilter::VectorizeSplineDiffusionAdaptive);
                        let cache = vec_cache.get_or_insert_with(|| vectorize::VectorizeCache::new_legacy(adaptive));
                        let (paths, bg) = cache.get_paths(raw_src, src_w, src_h);
                        let (gpu_edges, row_ranges, edge_indices, ow, oh) =
                            vectorize::rasterize::prepare_gpu_edges_v2(paths, bg, scale as f64, src_w, src_h);
                        if ow > 0 && oh > 0 {
                            renderer.run_spline_diffusion(
                                &gpu_edges, &row_ranges, &edge_indices, raw_src,
                                ow, oh, src_w as u32, src_h as u32, bg, scale,
                            );
                            renderer.render();
                            gpu_rendered = true;
                        }
                        vec_scaled = Vec::new();
                        (&[] as &[u32], 0, 0)
                    } else if matches!(scale_filter,
                        scaling::ScaleFilter::Vectorize | scaling::ScaleFilter::VectorizeAdaptive)
                    {
                        let s = (disp_w as f64 / src_w as f64).min(disp_h as f64 / src_h as f64);
                        let adaptive = matches!(scale_filter, scaling::ScaleFilter::VectorizeAdaptive);
                        let cache = vec_cache.get_or_insert_with(|| vectorize::VectorizeCache::new(adaptive));
                        let (paths, bg) = cache.get_paths(raw_src, src_w, src_h);
                        let (gpu_edges, row_ranges, edge_indices, ow, oh) =
                            vectorize::rasterize::prepare_gpu_edges_v2(paths, bg, s, src_w, src_h);
                        if ow > 0 && oh > 0 {
                            renderer.run_scanline_rasterize(&gpu_edges, &row_ranges, &edge_indices, ow, oh, bg);
                            renderer.render();
                            gpu_rendered = true;
                        }
                        vec_scaled = Vec::new();
                        (&[] as &[u32], 0, 0)
                    } else if let Some((s, w, h)) = scaling::cpu_scale(
                        scale_filter, raw_src, src_w, src_h, disp_w, disp_h,
                    ) {
                        vec_scaled = s;
                        (&vec_scaled, w as usize, h as usize)
                    } else {
                        (raw_src, src_w, src_h)
                    };

                // Resize texture if dimensions changed
                if !gpu_rendered && (frame_w as u32 != renderer.tex_w || frame_h as u32 != renderer.tex_h) {
                    let tex_desc = MTLTextureDescriptor::new();
                    tex_desc.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
                    tex_desc.setWidth(frame_w as usize);
                    tex_desc.setHeight(frame_h as usize);
                    tex_desc.setUsage(MTLTextureUsage::ShaderRead);
                    renderer.texture = renderer.device.newTextureWithDescriptor(&tex_desc).unwrap();
                    renderer.tex_w = frame_w as u32;
                    renderer.tex_h = frame_h as u32;
                }

                // Convert 0x00RRGGBB -> BGRA8Unorm (set alpha to 0xFF)
                bgra_buf.resize(frame_w * frame_h, 0u32);
                for i in 0..(frame_w * frame_h) {
                    bgra_buf[i] = 0xFF00_0000 | frame_pixels[i];
                }

                // Draw FPS overlay into pixel buffer
                if show_fps_overlay {
                    let text = format!("FPS: {:.1}  {:.2}ms", overlay_fps, overlay_emu_ms);
                    let scale = ((frame_w / 160).max(1)).min(4);
                    let fg = 0xFF00FF00;
                    let bg = 0xC0000000;
                    tiny_font::draw_string(
                        &mut bgra_buf, frame_w, frame_h,
                        &text, 2 * scale, 2 * scale, fg, bg, scale,
                    );
                }

                if !gpu_rendered {
                    renderer.update_texture(&bgra_buf);
                    renderer.render();
                }
                } // end if !gpu_rendered
            }

            // ── FPS counter ──────────────────────────────────────────────────
            let emu_time = frame_start.elapsed();
            if let Some((f, ms)) = fps_counter.update(1, emu_time) {
                overlay_fps = f;
                overlay_emu_ms = ms;
            }

            // ── Periodic save RAM flush ──────────────────────────────────────
            sav_flusher.poll(&emu);

            // ── Frame rate cap ───────────────────────────────────────────────
            let remaining = frame_dur.saturating_sub(frame_start.elapsed());
            if remaining > Duration::from_millis(2) {
                std::thread::sleep(remaining - Duration::from_millis(2));
            }
            while frame_start.elapsed() < frame_dur {
                std::hint::spin_loop();
            }
            frame_start = Instant::now();
        }

        // Cleanup
        close_accel(&accel_source);
        drop(camera);

        sav_flusher.flush(&emu);
    }
}
