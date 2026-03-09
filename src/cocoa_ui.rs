mod apu;
mod bus;
mod cartridge;
mod cpu;
mod emulator;
mod joypad;
mod model;
mod ppu;
mod printer;
mod serial;
mod sgb;
mod snapshot;
mod snes;
mod timer;
#[cfg(target_os = "macos")]
mod macos_accel;

use clap::Parser;
use emulator::Emulator;
use model::GbModel;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cocoa::appkit::{
    NSApp, NSApplication, NSApplicationActivationPolicy,
    NSBackingStoreType, NSEvent, NSEventType, NSWindow, NSWindowStyleMask,
};
use cocoa::base::{id, nil, YES, NO};
use cocoa::foundation::{NSAutoreleasePool, NSPoint, NSRect, NSSize, NSString};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::rc::autoreleasepool;
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};

const SCALE: u32 = 3;
const AUDIO_SAMPLE_RATE: u32 = 96_000;

fn frame_duration(model: GbModel) -> Duration {
    let nanos = 70_224u64 * 1_000_000_000 / model.cpu_clock_rate() as u64;
    Duration::from_nanos(nanos)
}

#[derive(Parser)]
#[command(name = "vibeboy_cocoa", about = "Game Boy / Game Boy Color emulator (macOS native)")]
struct Cli {
    rom: Option<PathBuf>,
    #[arg(long)]
    bootrom: Option<PathBuf>,
    #[arg(long, default_value = "auto")]
    model: String,
    #[arg(long)]
    snes_rom: Option<PathBuf>,
    #[arg(long)]
    lle: bool,
    #[arg(long)]
    no_bootrom: bool,
    #[arg(long)]
    printer: bool,
}

// ── CoreAudio FFI ────────────────────────────────────────────────────────────

mod core_audio {
    use std::os::raw::c_void;

    pub type OSStatus = i32;
    pub type AudioUnit = *mut c_void;
    pub type AudioComponent = *mut c_void;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct AudioComponentDescription {
        pub component_type: u32,
        pub component_sub_type: u32,
        pub component_manufacturer: u32,
        pub component_flags: u32,
        pub component_flags_mask: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct AudioStreamBasicDescription {
        pub sample_rate: f64,
        pub format_id: u32,
        pub format_flags: u32,
        pub bytes_per_packet: u32,
        pub frames_per_packet: u32,
        pub bytes_per_frame: u32,
        pub channels_per_frame: u32,
        pub bits_per_channel: u32,
        pub reserved: u32,
    }

    #[repr(C)]
    pub struct AudioBufferList {
        pub number_buffers: u32,
        pub buffers: [AudioBuffer; 1],
    }

    #[repr(C)]
    pub struct AudioBuffer {
        pub number_channels: u32,
        pub data_byte_size: u32,
        pub data: *mut c_void,
    }

    #[repr(C)]
    pub struct AudioTimeStamp {
        pub sample_time: f64,
        pub host_time: u64,
        pub rate_scalar: f64,
        pub word_clock_time: u64,
        pub smpte_time: [u8; 24],
        pub flags: u32,
        pub reserved: u32,
    }

    pub type AURenderCallback = unsafe extern "C" fn(
        in_ref_con: *mut c_void,
        io_action_flags: *mut u32,
        in_time_stamp: *const AudioTimeStamp,
        in_bus_number: u32,
        in_number_frames: u32,
        io_data: *mut AudioBufferList,
    ) -> OSStatus;

    #[repr(C)]
    pub struct AURenderCallbackStruct {
        pub input_proc: AURenderCallback,
        pub input_proc_ref_con: *mut c_void,
    }

    pub const K_AUDIO_UNIT_TYPE_OUTPUT: u32 = u32::from_be_bytes(*b"auou");
    pub const K_AUDIO_UNIT_SUB_TYPE_DEFAULT_OUTPUT: u32 = u32::from_be_bytes(*b"def ");
    pub const K_AUDIO_UNIT_MANUFACTURER_APPLE: u32 = u32::from_be_bytes(*b"appl");
    pub const K_AUDIO_FORMAT_LINEAR_PCM: u32 = u32::from_be_bytes(*b"lpcm");
    pub const K_AUDIO_FORMAT_FLAG_IS_FLOAT: u32 = 1;
    pub const K_AUDIO_FORMAT_FLAG_IS_PACKED: u32 = 8;
    pub const K_AUDIO_UNIT_SCOPE_INPUT: u32 = 1;
    pub const K_AUDIO_UNIT_PROPERTY_STREAM_FORMAT: u32 = 8;
    pub const K_AUDIO_UNIT_PROPERTY_SET_RENDER_CALLBACK: u32 = 23;

    #[link(name = "AudioToolbox", kind = "framework")]
    unsafe extern "C" {
        pub fn AudioComponentFindNext(
            component: AudioComponent,
            desc: *const AudioComponentDescription,
        ) -> AudioComponent;
        pub fn AudioComponentInstanceNew(
            component: AudioComponent,
            out: *mut AudioUnit,
        ) -> OSStatus;
        pub fn AudioUnitSetProperty(
            unit: AudioUnit,
            property_id: u32,
            scope: u32,
            element: u32,
            data: *const c_void,
            data_size: u32,
        ) -> OSStatus;
        pub fn AudioUnitInitialize(unit: AudioUnit) -> OSStatus;
        pub fn AudioOutputUnitStart(unit: AudioUnit) -> OSStatus;
        pub fn AudioOutputUnitStop(unit: AudioUnit) -> OSStatus;
        pub fn AudioComponentInstanceDispose(unit: AudioUnit) -> OSStatus;
    }
}

// ── Audio ring buffer ────────────────────────────────────────────────────────

struct AudioRingBuffer {
    buffer: Vec<f32>,
    write_pos: usize,
    read_pos: usize,
    capacity: usize,
}

impl AudioRingBuffer {
    fn new(capacity: usize) -> Self {
        AudioRingBuffer {
            buffer: vec![0.0; capacity],
            write_pos: 0,
            read_pos: 0,
            capacity,
        }
    }

    fn write(&mut self, data: &[f32]) {
        for &sample in data {
            let next = (self.write_pos + 1) % self.capacity;
            if next == self.read_pos {
                self.read_pos = (self.read_pos + 1) % self.capacity;
            }
            self.buffer[self.write_pos] = sample;
            self.write_pos = next;
        }
    }

    fn read(&mut self, out: &mut [f32]) {
        for sample in out.iter_mut() {
            if self.read_pos == self.write_pos {
                *sample = 0.0;
            } else {
                *sample = self.buffer[self.read_pos];
                self.read_pos = (self.read_pos + 1) % self.capacity;
            }
        }
    }
}

type SharedAudioBuffer = Arc<Mutex<AudioRingBuffer>>;

unsafe extern "C" fn audio_render_callback(
    in_ref_con: *mut std::os::raw::c_void,
    _io_action_flags: *mut u32,
    _in_time_stamp: *const core_audio::AudioTimeStamp,
    _in_bus_number: u32,
    in_number_frames: u32,
    io_data: *mut core_audio::AudioBufferList,
) -> i32 {
    unsafe {
        let ring = &*(in_ref_con as *const Mutex<AudioRingBuffer>);
        let buf_list = &mut *io_data;
        let ab = &mut buf_list.buffers[0];
        let out_ptr = ab.data as *mut f32;
        let sample_count = in_number_frames as usize * 2;
        let out_slice = std::slice::from_raw_parts_mut(out_ptr, sample_count);

        if let Ok(mut guard) = ring.lock() {
            guard.read(out_slice);
        } else {
            out_slice.fill(0.0);
        }
        0
    }
}

fn setup_audio(ring_buffer: &SharedAudioBuffer) -> Option<core_audio::AudioUnit> {
    unsafe {
        let desc = core_audio::AudioComponentDescription {
            component_type: core_audio::K_AUDIO_UNIT_TYPE_OUTPUT,
            component_sub_type: core_audio::K_AUDIO_UNIT_SUB_TYPE_DEFAULT_OUTPUT,
            component_manufacturer: core_audio::K_AUDIO_UNIT_MANUFACTURER_APPLE,
            component_flags: 0,
            component_flags_mask: 0,
        };

        let comp = core_audio::AudioComponentFindNext(std::ptr::null_mut(), &desc);
        if comp.is_null() {
            eprintln!("Failed to find audio output component");
            return None;
        }

        let mut audio_unit: core_audio::AudioUnit = std::ptr::null_mut();
        if core_audio::AudioComponentInstanceNew(comp, &mut audio_unit) != 0 {
            eprintln!("Failed to create audio unit");
            return None;
        }

        let stream_desc = core_audio::AudioStreamBasicDescription {
            sample_rate: AUDIO_SAMPLE_RATE as f64,
            format_id: core_audio::K_AUDIO_FORMAT_LINEAR_PCM,
            format_flags: core_audio::K_AUDIO_FORMAT_FLAG_IS_FLOAT
                | core_audio::K_AUDIO_FORMAT_FLAG_IS_PACKED,
            bytes_per_packet: 8,
            frames_per_packet: 1,
            bytes_per_frame: 8,
            channels_per_frame: 2,
            bits_per_channel: 32,
            reserved: 0,
        };

        core_audio::AudioUnitSetProperty(
            audio_unit,
            core_audio::K_AUDIO_UNIT_PROPERTY_STREAM_FORMAT,
            core_audio::K_AUDIO_UNIT_SCOPE_INPUT,
            0,
            &stream_desc as *const _ as *const _,
            std::mem::size_of::<core_audio::AudioStreamBasicDescription>() as u32,
        );

        let callback_struct = core_audio::AURenderCallbackStruct {
            input_proc: audio_render_callback,
            input_proc_ref_con: Arc::as_ptr(ring_buffer) as *mut _,
        };

        core_audio::AudioUnitSetProperty(
            audio_unit,
            core_audio::K_AUDIO_UNIT_PROPERTY_SET_RENDER_CALLBACK,
            core_audio::K_AUDIO_UNIT_SCOPE_INPUT,
            0,
            &callback_struct as *const _ as *const _,
            std::mem::size_of::<core_audio::AURenderCallbackStruct>() as u32,
        );

        if core_audio::AudioUnitInitialize(audio_unit) != 0 {
            eprintln!("Failed to initialize audio unit");
            core_audio::AudioComponentInstanceDispose(audio_unit);
            return None;
        }

        if core_audio::AudioOutputUnitStart(audio_unit) != 0 {
            eprintln!("Failed to start audio unit");
            core_audio::AudioComponentInstanceDispose(audio_unit);
            return None;
        }

        Some(audio_unit)
    }
}

// ── NSOpenPanel ──────────────────────────────────────────────────────────────

fn pick_rom_file() -> PathBuf {
    unsafe {
        let panel: id = msg_send![class!(NSOpenPanel), openPanel];
        let _: () = msg_send![panel, setCanChooseFiles: YES];
        let _: () = msg_send![panel, setCanChooseDirectories: NO];
        let _: () = msg_send![panel, setAllowsMultipleSelection: NO];

        // Set allowed file types
        let gb = NSString::alloc(nil).init_str("gb");
        let gbc = NSString::alloc(nil).init_str("gbc");
        let types: id = msg_send![class!(NSMutableArray), arrayWithCapacity: 2usize];
        let _: () = msg_send![types, addObject: gb];
        let _: () = msg_send![types, addObject: gbc];
        let _: () = msg_send![panel, setAllowedFileTypes: types];

        let response: isize = msg_send![panel, runModal];
        if response != 1 {
            std::process::exit(0);
        }

        let url: id = msg_send![panel, URL];
        let path: id = msg_send![url, path];
        let path_str: *const i8 = msg_send![path, UTF8String];
        let path = std::ffi::CStr::from_ptr(path_str)
            .to_str()
            .unwrap()
            .to_string();
        PathBuf::from(path)
    }
}

// ── Key mapping ──────────────────────────────────────────────────────────────

fn keycode_to_button(keycode: u16) -> Option<u8> {
    match keycode {
        6 => Some(Emulator::BTN_B),      // Z
        7 => Some(Emulator::BTN_A),      // X
        36 => Some(Emulator::BTN_START), // Return
        60 => Some(Emulator::BTN_SELECT), // Right Shift
        124 => Some(Emulator::BTN_RIGHT), // Right arrow
        123 => Some(Emulator::BTN_LEFT),  // Left arrow
        126 => Some(Emulator::BTN_UP),    // Up arrow
        125 => Some(Emulator::BTN_DOWN),  // Down arrow
        _ => None,
    }
}

const K_ESCAPE: u16 = 53;
const K_F5: u16 = 96;
const K_F7: u16 = 98;
const K_TAB: u16 = 48;
const K_DELETE: u16 = 51;

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

// ── Camera placeholder ───────────────────────────────────────────────────────

struct CameraCapture {
    buffer: Arc<Mutex<[u8; 128 * 112]>>,
    has_new_frame: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl CameraCapture {
    fn start() -> Option<Self> {
        let buffer = Arc::new(Mutex::new([0u8; 128 * 112]));
        let has_new_frame = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));

        let buf_clone = Arc::clone(&buffer);
        let frame_clone = Arc::clone(&has_new_frame);
        let stop_clone = Arc::clone(&stop);

        let handle = std::thread::Builder::new()
            .name("camera-avf".into())
            .spawn(move || {
                // Noise generator fallback (AVFoundation TODO)
                let mut rng: u32 = 0xDEAD_BEEF;
                while !stop_clone.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_millis(33));
                    let mut frame = [0u8; 128 * 112];
                    for pixel in frame.iter_mut() {
                        rng ^= rng << 13;
                        rng ^= rng >> 17;
                        rng ^= rng << 5;
                        *pixel = (rng >> 24) as u8;
                    }
                    if let Ok(mut lock) = buf_clone.lock() {
                        lock.copy_from_slice(&frame);
                    }
                    frame_clone.store(true, Ordering::Release);
                }
            })
            .ok()?;

        Some(CameraCapture {
            buffer,
            has_new_frame,
            stop,
            handle: Some(handle),
        })
    }

    fn read_frame(&self, buf: &mut [u8; 128 * 112]) -> bool {
        if self.has_new_frame.swap(false, Ordering::Acquire) {
            let lock = self.buffer.lock().unwrap();
            buf.copy_from_slice(&*lock);
            true
        } else {
            false
        }
    }
}

impl Drop for CameraCapture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

// ── Metal renderer ───────────────────────────────────────────────────────────

struct MetalRenderer {
    device: Device,
    layer: MetalLayer,
    command_queue: CommandQueue,
    texture: Texture,
    tex_w: u32,
    tex_h: u32,
}

impl MetalRenderer {
    fn new(tex_w: u32, tex_h: u32) -> Self {
        let device = Device::system_default().expect("No Metal device found");
        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_presents_with_transaction(false);
        unsafe {
            let layer_id: id = std::mem::transmute_copy(&layer);
            let _: () = msg_send![layer_id, setMagnificationFilter:
                NSString::alloc(nil).init_str("nearest")];
        }

        let command_queue = device.new_command_queue();

        let tex_desc = TextureDescriptor::new();
        tex_desc.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        tex_desc.set_width(tex_w as u64);
        tex_desc.set_height(tex_h as u64);
        tex_desc.set_usage(MTLTextureUsage::ShaderRead);

        let texture = device.new_texture(&tex_desc);

        MetalRenderer {
            device,
            layer,
            command_queue,
            texture,
            tex_w,
            tex_h,
        }
    }

    fn update_texture(&self, pixels: &[u32]) {
        let region = MTLRegion::new_2d(0, 0, self.tex_w as u64, self.tex_h as u64);
        self.texture.replace_region(
            region,
            0,
            pixels.as_ptr() as *const _,
            (self.tex_w * 4) as u64,
        );
    }

    fn render(&self) {
        autoreleasepool(|| {
            let drawable = match self.layer.next_drawable() {
                Some(d) => d,
                None => return,
            };

            let cmd_buf = self.command_queue.new_command_buffer();

            // Blit our texture to the drawable's texture
            let blit = cmd_buf.new_blit_command_encoder();

            let src_size = MTLSize::new(self.tex_w as u64, self.tex_h as u64, 1);
            let src_origin = MTLOrigin { x: 0, y: 0, z: 0 };

            let dst_tex = drawable.texture();
            let dst_w = dst_tex.width();
            let dst_h = dst_tex.height();

            // Aspect-ratio-correct scaling: compute destination origin
            let scale_x = dst_w as f64 / self.tex_w as f64;
            let scale_y = dst_h as f64 / self.tex_h as f64;
            let scale = scale_x.min(scale_y);
            let out_w = (self.tex_w as f64 * scale) as u64;
            let out_h = (self.tex_h as f64 * scale) as u64;
            let off_x = (dst_w - out_w) / 2;
            let off_y = (dst_h - out_h) / 2;

            // Blit doesn't scale — if sizes differ we need a render pass.
            // For now, just copy to center (works when layer drawable size = tex_w * SCALE
            // and layer contentsScale handles pixel doubling)
            blit.copy_from_texture(
                &self.texture,
                0,
                0,
                src_origin,
                src_size,
                &dst_tex,
                0,
                0,
                MTLOrigin { x: off_x, y: off_y, z: 0 },
            );

            blit.end_encoding();
            cmd_buf.present_drawable(&drawable);
            cmd_buf.commit();
        });
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    unsafe {
        let _pool = NSAutoreleasePool::new(nil);

        let app = NSApp();
        app.setActivationPolicy_(NSApplicationActivationPolicy::NSApplicationActivationPolicyRegular);

        // Resolve ROM path
        let rom_path: PathBuf = if let Some(ref p) = cli.rom {
            p.clone()
        } else {
            app.activateIgnoringOtherApps_(YES);
            pick_rom_file()
        };

        let rom = fs::read(&rom_path).unwrap_or_else(|e| {
            eprintln!("Failed to read ROM '{}': {}", rom_path.display(), e);
            std::process::exit(1);
        });

        let model = if cli.model == "auto" {
            GbModel::Cgb
        } else {
            cli.model.parse::<GbModel>().unwrap_or_else(|e| {
                eprintln!("{}", e);
                std::process::exit(1);
            })
        };

        let frame_dur = frame_duration(model);

        let boot_rom: Option<Vec<u8>> = if cli.no_bootrom {
            None
        } else if let Some(ref p) = cli.bootrom {
            Some(fs::read(p).unwrap_or_else(|e| {
                eprintln!("Failed to read boot ROM '{}': {}", p.display(), e);
                std::process::exit(1);
            }))
        } else {
            let candidates: &[&str] = match model {
                GbModel::Dmg0 => &["dmg0_boot.bin", "bootroms/dmg0_boot.bin", "gb_bios.bin"],
                GbModel::Dmg => &["dmg_boot.bin", "bootroms/dmg_boot.bin", "gb_bios.bin"],
                GbModel::Mgb => &["mgb_boot.bin", "bootroms/mgb_boot.bin", "gb_bios.bin"],
                GbModel::Sgb => &["sgb_boot.bin", "bootroms/sgb_boot.bin", "sgb_bios.bin"],
                GbModel::Sgb2 => &["sgb2_boot.bin", "bootroms/sgb2_boot.bin", "sgb2_bios.bin"],
                GbModel::Cgb0 => &["cgb0_boot.bin", "bootroms/cgb0_boot.bin", "gbc_bios.bin"],
                GbModel::Cgb | GbModel::Agb => {
                    &["cgb_boot.bin", "bootroms/cgb_boot.bin", "gbc_bios.bin"]
                }
            };
            candidates.iter().find_map(|name| fs::read(name).ok())
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

        let mut emu = Emulator::new(rom, boot_rom, Some(rom_path.as_path()), model, snes_rom);

        if cli.printer {
            let output_dir = std::path::Path::new("prints");
            emu.bus.serial.device =
                Box::new(printer::Printer::new(output_dir, model.cpu_clock_rate()));
            eprintln!("Game Boy Printer connected — images will be saved to prints/");
        }

        let is_sgb = emu.is_sgb();
        let (tex_w, tex_h): (u32, u32) = if is_sgb { (256, 224) } else { (160, 144) };
        let win_w = tex_w * SCALE;
        let win_h = tex_h * SCALE;

        // ── Metal renderer ───────────────────────────────────────────────────
        let renderer = MetalRenderer::new(tex_w, tex_h);

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

        window.setTitle_(NSString::alloc(nil).init_str("GBC Emulator"));
        window.center();

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
            Arc::new(Mutex::new(AudioRingBuffer::new(96_000)));
        let _audio_unit = setup_audio(&audio_ring);

        // ── Camera ───────────────────────────────────────────────────────────
        let camera = if emu.bus.cart.has_camera() {
            CameraCapture::start()
        } else {
            None
        };
        let mut camera_buf = [0u8; 128 * 112];

        // ── Accelerometer ────────────────────────────────────────────────────
        let has_accel = emu.bus.cart.has_accelerometer();
        #[cfg(target_os = "macos")]
        if has_accel {
            if macos_accel::init() {
                eprintln!("Accelerometer: macOS native (Apple Silicon)");
            }
        }

        // ── Key state + frame loop ───────────────────────────────────────────
        let mut keys_down = std::collections::HashSet::<u16>::new();
        let mut current_slot: usize = 0;
        let mut frame_start = Instant::now();
        let mut fps_timer = Instant::now();
        let mut fps_count = 0u32;
        let mut fps_emu_total = Duration::ZERO;

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

                    if let Some(btn) = keycode_to_button(keycode) {
                        emu.set_button(btn, true);
                    }
                } else if event_type == NSEventType::NSKeyUp as u64 {
                    keys_down.remove(&keycode);
                    if let Some(btn) = keycode_to_button(keycode) {
                        emu.set_button(btn, false);
                    }
                } else {
                    let _: () = msg_send![app, sendEvent: event];
                }
            }

            // Check if window was closed
            let visible: bool = msg_send![window, isVisible];
            if !visible {
                break 'running;
            }

            // ── Camera ───────────────────────────────────────────────────────
            if let Some(ref cam) = camera {
                if cam.read_frame(&mut camera_buf) {
                    emu.bus.cart.set_camera_image(&camera_buf);
                }
            }

            // ── Accelerometer ────────────────────────────────────────────────
            #[cfg(target_os = "macos")]
            if has_accel {
                if let Some((x, y, _z)) = macos_accel::poll() {
                    const CENTER: f32 = 0x81D0 as u16 as f32;
                    const RANGE: f32 = 0x70 as u16 as f32;
                    let mbc7_x = (CENTER + (-x) * RANGE).clamp(0.0, 65535.0) as u16;
                    let mbc7_y = (CENTER + y * RANGE).clamp(0.0, 65535.0) as u16;
                    emu.bus.cart.set_accelerometer(mbc7_x, mbc7_y);
                }
            }

            // ── Rewind / Fast-forward ────────────────────────────────────────
            let backspace_held = keys_down.contains(&K_DELETE);
            let fast_forward = keys_down.contains(&K_TAB);
            emu.rewinding = backspace_held;

            if backspace_held {
                emu.rewind_one_frame();
                emu.bus.apu.drain_samples();
            } else if fast_forward {
                for _ in 0..3 {
                    emu.step_frame();
                    emu.bus.apu.drain_samples();
                }
                emu.step_frame();
            } else {
                emu.step_frame();
            }

            // ── Audio ────────────────────────────────────────────────────────
            let samples = emu.bus.apu.drain_samples();
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

            // ── Render ───────────────────────────────────────────────────────
            {
                let (src, w, h) = if is_sgb {
                    (emu.sgb_composited_frame(), 256usize, 224usize)
                } else {
                    (emu.frame_buffer(), 160usize, 144usize)
                };

                // Convert 0x00RRGGBB → BGRA8 (LE bytes: B, G, R, A)
                let mut bgra = vec![0u32; w * h];
                for i in 0..(w * h) {
                    let rgb = src[i];
                    let r = (rgb >> 16) & 0xFF;
                    let g = (rgb >> 8) & 0xFF;
                    let b = rgb & 0xFF;
                    // BGRA8Unorm on LE: byte0=B byte1=G byte2=R byte3=A
                    // As u32 LE: value = A<<24 | R<<16 | G<<8 | B
                    bgra[i] = 0xFF00_0000 | (r << 16) | (g << 8) | b;
                }

                renderer.update_texture(&bgra);
                renderer.render();
            }

            // ── FPS counter ──────────────────────────────────────────────────
            let emu_time = frame_start.elapsed();
            fps_count += 1;
            fps_emu_total += emu_time;
            let fps_elapsed = fps_timer.elapsed();
            if fps_elapsed >= Duration::from_secs(1) {
                let fps = fps_count as f64 / fps_elapsed.as_secs_f64();
                let avg_emu_ms = fps_emu_total.as_secs_f64() * 1000.0 / fps_count as f64;
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
        #[cfg(target_os = "macos")]
        if has_accel {
            macos_accel::close();
        }
        drop(camera);

        emu.save();
    }
}
