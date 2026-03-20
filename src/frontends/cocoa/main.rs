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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cocoa::appkit::{
    NSApp, NSApplication, NSApplicationActivationPolicy,
    NSBackingStoreType, NSEvent, NSEventType, NSWindow, NSWindowStyleMask,
    NSMenu, NSMenuItem,
};
use cocoa::base::{id, nil, YES, NO, SEL};
use cocoa::foundation::{NSAutoreleasePool, NSPoint, NSRect, NSSize, NSString};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::rc::autoreleasepool;
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};

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
    CoreMotion(id),
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

    unsafe {
        let _pool = NSAutoreleasePool::new(nil);

        let app = NSApp();
        app.setActivationPolicy_(NSApplicationActivationPolicy::NSApplicationActivationPolicyRegular);

        // Set up menu bar and action handler
        create_menu_bar(app);
        let (_menu_handler, menu_actions_ptr) = menu_handler::create(app);
        let menu_actions = &mut *menu_actions_ptr;

        // Resolve ROM path
        let rom_path: PathBuf = if let Some(ref p) = cli.rom {
            p.clone()
        } else {
            app.activateIgnoringOtherApps_(YES);
            open_rom_dialog().unwrap_or_else(|| std::process::exit(0))
        };

        let rom = fs::read(&rom_path).unwrap_or_else(|e| {
            eprintln!("Failed to read ROM '{}': {}", rom_path.display(), e);
            std::process::exit(1);
        });

        let mut forced_model: Option<GbModel> = cli.model;
        let model = forced_model.unwrap_or_else(|| auto_detect_model(&rom));
        let frame_dur = frame_duration(model);

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

        eprintln!("\nControls:");
        eprintln!("  Arrow keys  — D-pad");
        eprintln!("  Z / X       — A / B");
        eprintln!("  Enter       — Start");
        eprintln!("  Right Shift — Select");
        eprintln!("  Backspace   — Rewind");
        eprintln!("  Tab         — Fast forward (4x)");
        eprintln!("  F5 / F7     — Save / Load state");
        eprintln!("  1-9         — Select state slot");
        eprintln!("  Escape      — Quit");
        eprintln!();

        let mut current_rom = rom;
        let mut current_rom_path = rom_path;
        let mut current_model = model;
        let mut emu = Emulator::new(current_rom.clone(), boot_rom, Some(current_rom_path.as_path()), current_model, snes_rom);

        // Load custom key mappings
        let mut key_map = load_key_map();

        // Initialize recent ROMs list and populate menu
        add_recent_rom(&current_rom_path.to_string_lossy());
        rebuild_recent_menu(app, &load_recent_roms());

        if cli.printer {
            let output_dir = std::path::Path::new("prints");
            emu.attach_serial_device(
                Box::new(printer::Printer::new(output_dir, model.cpu_clock_rate())));
            eprintln!("Game Boy Printer connected — images will be saved to prints/");
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
                    update_filter_checkmarks(app, MENU_TAG_FILTER_BASE + i as isize);
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
        let style = NSWindowStyleMask::NSTitledWindowMask
            | NSWindowStyleMask::NSClosableWindowMask
            | NSWindowStyleMask::NSMiniaturizableWindowMask
            | NSWindowStyleMask::NSResizableWindowMask;

        let window = NSWindow::alloc(nil).initWithContentRect_styleMask_backing_defer_(
            NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(win_w as f64, win_h as f64)),
            style,
            NSBackingStoreType::NSBackingStoreBuffered,
            NO,
        );

        let title = format!("VibeBoy \u{2014} {}",
            current_rom_path.file_name().unwrap_or_default().to_string_lossy());
        window.setTitle_(NSString::alloc(nil).init_str(&title));
        window.center();

        // Create a custom NSView subclass that suppresses key repeat sounds
        {
            use objc::declare::ClassDecl;
            use objc::runtime::{Class, Sel};
            let class_name = "VBGameView";
            if Class::get(class_name).is_none() {
                let superclass = Class::get("NSView").unwrap();
                let mut decl = ClassDecl::new(class_name, superclass).unwrap();
                extern "C" fn accepts_first_responder(_this: &Object, _sel: Sel) -> bool { true }
                extern "C" fn key_down(_this: &Object, _sel: Sel, _event: id) { /* swallow */ }
                unsafe {
                    decl.add_method(sel!(acceptsFirstResponder), accepts_first_responder as extern "C" fn(&Object, Sel) -> bool);
                    decl.add_method(sel!(keyDown:), key_down as extern "C" fn(&Object, Sel, id));
                }
                decl.register();
            }
            let game_view_class = Class::get(class_name).unwrap();
            let content_rect: NSRect = msg_send![window, frame];
            let game_view: id = msg_send![game_view_class, alloc];
            let game_view: id = msg_send![game_view, initWithFrame: content_rect];
            let _: () = msg_send![window, setContentView: game_view];
            let _: () = msg_send![window, makeFirstResponder: game_view];
        }

        // Attach Metal layer to content view
        let content_view: id = msg_send![window, contentView];
        let _: () = msg_send![content_view, setWantsLayer: YES];

        // Set the Metal layer
        let raw_layer: id = std::mem::transmute_copy(&renderer.layer);
        let _: () = msg_send![content_view, setLayer: raw_layer];

        // Set drawable size to match window backing pixels
        let backing_size: NSSize = msg_send![content_view,
            convertSizeToBacking: NSSize::new(win_w as f64, win_h as f64)];
        renderer.layer.set_drawable_size(CGSize::new(
            backing_size.width,
            backing_size.height,
        ));

        window.makeKeyAndOrderFront_(nil);
        app.activateIgnoringOtherApps_(YES);

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
        let mut fps_timer = Instant::now();
        let mut fps_count = 0u32;
        let mut fps_emu_total = Duration::ZERO;
        let mut bgra_buf: Vec<u32> = Vec::with_capacity((tex_w * tex_h) as usize);

        'running: loop {
            let _pool = NSAutoreleasePool::new(nil);

            // Poll events
            loop {
                let event: id = msg_send![app,
                    nextEventMatchingMask: u64::MAX
                    untilDate: nil // don't wait
                    inMode: NSString::alloc(nil).init_str("kCFRunLoopDefaultMode")
                    dequeue: YES
                ];

                if event == nil {
                    break;
                }

                let event_type: u64 = msg_send![event, type];
                let keycode: u16 = if event_type == NSEventType::NSKeyDown as u64
                    || event_type == NSEventType::NSKeyUp as u64
                    || event_type == NSEventType::NSFlagsChanged as u64
                {
                    msg_send![event, keyCode]
                } else {
                    0
                };

                if event_type == NSEventType::NSKeyDown as u64 {
                    if keycode == K_ESCAPE {
                        break 'running;
                    }

                    keys_down.insert(keycode);

                    if keycode == K_F5 {
                        emu.save_state(current_slot);
                        eprintln!("State saved to slot {}", current_slot + 1);
                    } else if keycode == K_F7 {
                        if emu.load_state(current_slot) {
                            eprintln!("State loaded from slot {}", current_slot + 1);
                        } else {
                            eprintln!("Slot {} is empty", current_slot + 1);
                        }
                    } else if let Some(slot) = keycode_to_slot(keycode) {
                        current_slot = slot;
                        eprintln!("Slot {} selected", current_slot + 1);
                    }

                    if let Some(btn) = key_map.get(&keycode).copied() {
                        emu.set_button(btn, true);
                    }
                } else if event_type == NSEventType::NSKeyUp as u64 {
                    keys_down.remove(&keycode);
                    if let Some(btn) = key_map.get(&keycode).copied() {
                        emu.set_button(btn, false);
                    }
                }

                // Always dispatch events so menus and window chrome work
                let _: () = msg_send![app, sendEvent: event];
            }

            // ── Handle menu actions ──────────────────────────────────────────
            {
                let actions = menu_actions.take_all();

                if actions.open_rom {
                    if let Some(path) = open_rom_dialog() {
                        if let Ok(rom_data) = fs::read(&path) {
                            let title = format!("VibeBoy \u{2014} {}",
                                path.file_name().unwrap_or_default().to_string_lossy());
                            window.setTitle_(NSString::alloc(nil).init_str(&title));
                            add_recent_rom(&path.to_string_lossy());
                            rebuild_recent_menu(app, &load_recent_roms());
                            current_rom = rom_data;
                            current_rom_path = path;
                            current_model = forced_model.unwrap_or_else(|| auto_detect_model(&current_rom));
                            emu = Emulator::new(current_rom.clone(), None, Some(current_rom_path.as_path()), current_model, None);
                            paused = false;
                            eprintln!("Loaded: {}", current_rom_path.display());
                        }
                    }
                }

                if actions.pause_toggle {
                    paused = !paused;
                    eprintln!("{}", if paused { "Paused" } else { "Resumed" });
                    // Update menu item title
                    let emu_menu: id = msg_send![app.mainMenu(), itemAtIndex: 3isize];
                    let submenu: id = msg_send![emu_menu, submenu];
                    let pause_item: id = msg_send![submenu, itemWithTag: MENU_TAG_PAUSE];
                    let label = if paused { "Resume" } else { "Pause" };
                    let _: () = msg_send![pause_item, setTitle: NSString::alloc(nil).init_str(label)];
                }

                if actions.reset {
                    emu = Emulator::new(current_rom.clone(), None, Some(current_rom_path.as_path()), current_model, None);
                    paused = false;
                    eprintln!("Reset");
                }

                if actions.save_state {
                    emu.save_state(current_slot);
                    if let Some(data) = emu.save_state_to_bytes(current_slot) {
                        let path = current_rom_path.with_extension(format!("{}.ss", current_slot + 1));
                        match std::fs::write(&path, &data) {
                            Ok(_) => eprintln!("State saved to slot {} ({})", current_slot + 1, path.display()),
                            Err(e) => eprintln!("State saved to slot {} (disk write failed: {})", current_slot + 1, e),
                        }
                    } else {
                        eprintln!("State saved to slot {}", current_slot + 1);
                    }
                }

                if actions.load_state {
                    if emu.load_state(current_slot) {
                        eprintln!("State loaded from slot {}", current_slot + 1);
                    } else {
                        let path = current_rom_path.with_extension(format!("{}.ss", current_slot + 1));
                        if let Ok(data) = std::fs::read(&path) {
                            if emu.load_state_from_bytes(current_slot, &data) {
                                eprintln!("State loaded from disk: {}", path.display());
                            } else {
                                eprintln!("Failed to load state from {}", path.display());
                            }
                        } else {
                            eprintln!("Slot {} is empty", current_slot + 1);
                        }
                    }
                }

                if let Some(slot) = actions.select_slot {
                    current_slot = slot;
                    eprintln!("Slot {} selected", current_slot + 1);
                }

                if let Some(tag) = actions.select_model {
                    if let Some(new_model) = model_tag_to_model(tag) {
                        forced_model = new_model;
                        current_model = forced_model.unwrap_or_else(|| auto_detect_model(&current_rom));
                        emu = Emulator::new(current_rom.clone(), None, Some(current_rom_path.as_path()), current_model, None);
                        update_model_checkmarks(app, tag);
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
                        update_filter_checkmarks(app, tag);
                        eprintln!("Filter: {:?}", scale_filter);
                    }
                }

                if actions.toggle_fps {
                    show_fps_overlay = !show_fps_overlay;
                    // Update menu checkmark
                    let view_menu_item: id = msg_send![app.mainMenu(), itemAtIndex: 4isize];
                    let view_submenu: id = msg_send![view_menu_item, submenu];
                    let fps_item: id = msg_send![view_submenu, itemWithTag: MENU_TAG_SHOW_FPS];
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
                            let title = format!("VibeBoy \u{2014} {}",
                                path.file_name().unwrap_or_default().to_string_lossy());
                            window.setTitle_(NSString::alloc(nil).init_str(&title));
                            add_recent_rom(path_str);
                            rebuild_recent_menu(app, &load_recent_roms());
                            current_rom = rom_data;
                            current_rom_path = path;
                            current_model = forced_model.unwrap_or_else(|| auto_detect_model(&current_rom));
                            emu = Emulator::new(current_rom.clone(), None, Some(current_rom_path.as_path()), current_model, None);
                            paused = false;
                            eprintln!("Loaded: {}", current_rom_path.display());
                        } else {
                            eprintln!("Failed to read: {}", path_str);
                        }
                    }
                }

                if actions.clear_recent {
                    save_recent_roms(&[]);
                    rebuild_recent_menu(app, &[]);
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
                // Gamepad accelerometer takes priority
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
                    emu.rewind_one_frame();
                    emu.drain_audio_samples();
                } else if fast_forward {
                    for _ in 0..3 {
                        emu.step_frame();
                        emu.drain_audio_samples();
                    }
                    emu.step_frame();
                } else {
                    emu.step_frame();
                }
            }

            // ── Audio ────────────────────────────────────────────────────────
            let samples = emu.drain_audio_samples();
            if !samples.is_empty() && !fast_forward {
                let max_samples = 3200 * 2;
                let to_write = if samples.len() <= max_samples {
                    &samples[..]
                } else {
                    &samples[samples.len() - max_samples..]
                };
                if let Ok(mut ring) = audio_ring.lock() {
                    ring.write(to_write);
                }
            }

            // ── Update drawable size on resize ─────────────────────────────
            let (disp_w, disp_h);
            {
                let bounds: NSRect = msg_send![content_view, bounds];
                let backing: NSSize = msg_send![content_view,
                    convertSizeToBacking: bounds.size];
                disp_w = backing.width as usize;
                disp_h = backing.height as usize;
                renderer.layer.set_drawable_size(CGSize::new(
                    backing.width, backing.height,
                ));
            }

            // ── Render ───────────────────────────────────────────────────────
            {
                let raw_src: &[u32] = if is_sgb {
                    emu.sgb_composited_frame()
                } else {
                    emu.frame_buffer()
                };

                // Try GPU compute scaling first
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
                            let desc = TextureDescriptor::new();
                            desc.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
                            desc.set_width(gw as u64);
                            desc.set_height(gh as u64);
                            desc.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
                            renderer.compute_out_tex = Some(renderer.device.new_texture(&desc));
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

                // Apply scaling filter
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
                    let tex_desc = TextureDescriptor::new();
                    tex_desc.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
                    tex_desc.set_width(frame_w as u64);
                    tex_desc.set_height(frame_h as u64);
                    tex_desc.set_usage(MTLTextureUsage::ShaderRead);
                    renderer.texture = renderer.device.new_texture(&tex_desc);
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
                    let fg = 0xFF00FF00; // green on BGRA LE = green
                    let bg = 0xC0000000; // semi-transparent black
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
            fps_count += 1;
            fps_emu_total += emu_time;
            let fps_elapsed = fps_timer.elapsed();
            if fps_elapsed >= Duration::from_secs(1) {
                let fps = fps_count as f64 / fps_elapsed.as_secs_f64();
                let avg_emu_ms = fps_emu_total.as_secs_f64() * 1000.0 / fps_count as f64;
                overlay_fps = fps;
                overlay_emu_ms = avg_emu_ms;
                eprintln!("FPS: {:.1}  emu: {:.2}ms/frame", fps, avg_emu_ms);
                fps_count = 0;
                fps_emu_total = Duration::ZERO;
                fps_timer = Instant::now();
            }

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

        emu.save();
    }
}
