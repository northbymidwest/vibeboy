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

const SCALE: u32 = 3;
const AUDIO_SAMPLE_RATE: u32 = 96_000;

// ── Accelerometer source tracking ─────────────────────────────────────────────

enum AccelSource {
    None,
    /// IOKit HID (Apple Silicon built-in accelerometer)
    IoKit,
    /// CoreMotion CMMotionManager fallback
    CoreMotion(id),
}

#[link(name = "CoreMotion", kind = "framework")]
unsafe extern "C" {}

fn init_accel() -> AccelSource {
    // Try IOKit HID first (Apple Silicon native)
    #[cfg(target_os = "macos")]
    {
        if macos_accel::init() {
            eprintln!("Accelerometer: macOS native (Apple Silicon)");
            return AccelSource::IoKit;
        }
    }

    // Fallback: CMMotionManager
    unsafe {
        let cm_class = class!(CMMotionManager);
        let manager: id = msg_send![cm_class, alloc];
        let manager: id = msg_send![manager, init];
        if manager.is_null() {
            log::info!("CMMotionManager init failed — accelerometer disabled");
            return AccelSource::None;
        }
        let available: bool = msg_send![manager, isAccelerometerAvailable];
        if !available {
            let () = msg_send![manager, release];
            log::info!("CMMotionManager: no accelerometer available");
            return AccelSource::None;
        }
        // Set update interval (~60 Hz)
        let interval: f64 = 1.0 / 60.0;
        let () = msg_send![manager, setAccelerometerUpdateInterval: interval];
        let () = msg_send![manager, startAccelerometerUpdates];
        eprintln!("Accelerometer: CoreMotion (CMMotionManager)");
        AccelSource::CoreMotion(manager)
    }
}

fn poll_accel(source: &AccelSource) -> Option<(f32, f32, f32)> {
    match source {
        AccelSource::None => None,
        AccelSource::IoKit => macos_accel::poll(),
        AccelSource::CoreMotion(manager) => unsafe {
            let data: id = msg_send![*manager, accelerometerData];
            if data.is_null() {
                return None;
            }
            // CMAcceleration is a struct { x: f64, y: f64, z: f64 }
            // -[CMAccelerometerData acceleration] returns it by value
            #[repr(C)]
            struct CMAcceleration {
                x: f64,
                y: f64,
                z: f64,
            }
            let accel: CMAcceleration = msg_send![data, acceleration];
            Some((accel.x as f32, accel.y as f32, accel.z as f32))
        },
    }
}

fn close_accel(source: &AccelSource) {
    match source {
        AccelSource::None => {}
        AccelSource::IoKit => {
            #[cfg(target_os = "macos")]
            macos_accel::close();
        }
        AccelSource::CoreMotion(manager) => unsafe {
            let () = msg_send![*manager, stopAccelerometerUpdates];
            let () = msg_send![*manager, release];
        },
    }
}

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

// ── Key mapping ──────────────────────────────────────────────────────────────


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

// ── AVFoundation Camera Capture ──────────────────────────────────────────────

mod avf_camera {
    use super::*;
    use objc::declare::ClassDecl;
    use objc::runtime::{Class, Object, Sel};
    use std::os::raw::c_void;

    // CoreMedia / CoreVideo FFI
    type CMSampleBufferRef = *mut c_void;
    type CVImageBufferRef = *mut c_void;
    type CVPixelBufferRef = CVImageBufferRef;
    type CVReturn = i32;

    #[link(name = "CoreMedia", kind = "framework")]
    unsafe extern "C" {
        fn CMSampleBufferGetImageBuffer(sbuf: CMSampleBufferRef) -> CVImageBufferRef;
    }

    #[link(name = "CoreVideo", kind = "framework")]
    unsafe extern "C" {
        fn CVPixelBufferLockBaseAddress(buf: CVPixelBufferRef, flags: u64) -> CVReturn;
        fn CVPixelBufferUnlockBaseAddress(buf: CVPixelBufferRef, flags: u64) -> CVReturn;
        fn CVPixelBufferGetBaseAddress(buf: CVPixelBufferRef) -> *mut u8;
        fn CVPixelBufferGetWidth(buf: CVPixelBufferRef) -> usize;
        fn CVPixelBufferGetHeight(buf: CVPixelBufferRef) -> usize;
        fn CVPixelBufferGetBytesPerRow(buf: CVPixelBufferRef) -> usize;
        static kCVPixelBufferPixelFormatTypeKey: id;
    }

    unsafe fn get_cv_pixel_format_key() -> id {
        unsafe { kCVPixelBufferPixelFormatTypeKey }
    }

    // Shared state passed to the delegate callback via an ivar
    struct DelegateContext {
        buffer: Arc<Mutex<[u8; 128 * 112]>>,
        has_new_frame: Arc<AtomicBool>,
    }

    // Register a runtime ObjC class that implements the sample buffer delegate
    fn register_delegate_class() -> &'static Class {
        use std::sync::Once;
        static REGISTER: Once = Once::new();
        REGISTER.call_once(|| {
            let superclass = Class::get("NSObject").unwrap();
            let mut decl = ClassDecl::new("VBCameraDelegate", superclass).unwrap();

            decl.add_ivar::<*mut c_void>("_context");

            unsafe {
                // captureOutput:didOutputSampleBuffer:fromConnection:
                decl.add_method(
                    sel!(captureOutput:didOutputSampleBuffer:fromConnection:),
                    capture_output as extern "C" fn(&Object, Sel, id, id, id),
                );
            }

            decl.register();
        });
        Class::get("VBCameraDelegate").unwrap()
    }

    extern "C" fn capture_output(
        this: &Object,
        _sel: Sel,
        _output: id,
        sample_buffer: id,
        _connection: id,
    ) {
        unsafe {
            let ctx_ptr: *mut c_void = *this.get_ivar("_context");
            if ctx_ptr.is_null() {
                return;
            }
            let ctx = &*(ctx_ptr as *const DelegateContext);

            let pixel_buf = CMSampleBufferGetImageBuffer(sample_buffer as *mut c_void);
            if pixel_buf.is_null() {
                return;
            }

            CVPixelBufferLockBaseAddress(pixel_buf, 1); // kCVPixelBufferLock_ReadOnly = 1
            let base = CVPixelBufferGetBaseAddress(pixel_buf);
            let w = CVPixelBufferGetWidth(pixel_buf);
            let h = CVPixelBufferGetHeight(pixel_buf);
            let stride = CVPixelBufferGetBytesPerRow(pixel_buf);

            if !base.is_null() && w > 0 && h > 0 {
                process_bgra_frame(base, w, h, stride, &ctx.buffer, &ctx.has_new_frame);
            }

            CVPixelBufferUnlockBaseAddress(pixel_buf, 1);
        }
    }

    fn process_bgra_frame(
        base: *const u8,
        w: usize,
        h: usize,
        stride: usize,
        buffer: &Arc<Mutex<[u8; 128 * 112]>>,
        has_new_frame: &Arc<AtomicBool>,
    ) {
        // Build RGBA image from BGRA pixel buffer
        let mut rgba = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let src_off = y * stride + x * 4;
                let dst_off = (y * w + x) * 4;
                unsafe {
                    let b = *base.add(src_off);
                    let g = *base.add(src_off + 1);
                    let r = *base.add(src_off + 2);
                    let a = *base.add(src_off + 3);
                    rgba[dst_off] = r;
                    rgba[dst_off + 1] = g;
                    rgba[dst_off + 2] = b;
                    rgba[dst_off + 3] = a;
                }
            }
        }

        let img = match image::RgbaImage::from_raw(w as u32, h as u32, rgba) {
            Some(i) => i,
            None => return,
        };

        // Crop to 8:7 aspect ratio (128:112)
        let target_ratio = 128.0 / 112.0;
        let src_ratio = w as f64 / h as f64;
        let (crop_w, crop_h) = if src_ratio > target_ratio {
            let cw = (h as f64 * target_ratio) as u32;
            (cw, h as u32)
        } else {
            let ch = (w as f64 / target_ratio) as u32;
            (w as u32, ch)
        };
        let crop_x = (w as u32 - crop_w) / 2;
        let crop_y = (h as u32 - crop_h) / 2;
        let cropped =
            image::imageops::crop_imm(&img, crop_x, crop_y, crop_w, crop_h).to_image();

        // Convert to grayscale and resize with Lanczos3
        let gray = image::imageops::grayscale(&cropped);
        let resized =
            image::imageops::resize(&gray, 128, 112, image::imageops::FilterType::Lanczos3);

        // Write to shared buffer
        if let Ok(mut lock) = buffer.lock() {
            lock.copy_from_slice(resized.as_raw());
        }
        has_new_frame.store(true, Ordering::Release);
    }

    pub struct CameraCapture {
        buffer: Arc<Mutex<[u8; 128 * 112]>>,
        has_new_frame: Arc<AtomicBool>,
        session: id,          // AVCaptureSession (retained)
        _delegate: id,        // VBCameraDelegate (retained)
        _context: *mut DelegateContext, // leaked; freed on drop
    }

    unsafe impl Send for CameraCapture {}

    impl CameraCapture {
        pub fn start() -> Option<Self> {
            unsafe {
                // Get default video capture device
                let av_capture_device = Class::get("AVCaptureDevice")?;
                let device: id = msg_send![av_capture_device,
                    defaultDeviceWithMediaType: NSString::alloc(nil).init_str("vide")];
                if device == nil {
                    log::info!("No camera found — using noise generator for Pocket Camera");
                    return None;
                }

                // Create input
                let av_capture_input = Class::get("AVCaptureDeviceInput")?;
                let mut error: id = nil;
                let input: id = msg_send![av_capture_input,
                    deviceInputWithDevice: device error: &mut error];
                if input == nil || error != nil {
                    log::warn!("Failed to create camera input");
                    return None;
                }

                // Create session
                let session: id = msg_send![Class::get("AVCaptureSession")?, new];
                // AVCaptureSessionPreset640x480 = "AVCaptureSessionPreset640x480"
                // The preset string literal IS the constant value on macOS
                let _: () = msg_send![session, setSessionPreset:
                    NSString::alloc(nil).init_str("AVCaptureSessionPreset640x480")];

                let can_add_input: bool = msg_send![session, canAddInput: input];
                if !can_add_input {
                    log::warn!("Cannot add camera input to session");
                    let _: () = msg_send![session, release];
                    return None;
                }
                let _: () = msg_send![session, addInput: input];

                // Create video data output
                let output: id = msg_send![Class::get("AVCaptureVideoDataOutput")?, new];

                // Request BGRA pixel format
                // kCVPixelBufferPixelFormatTypeKey = "PixelFormatType" (CoreVideo constant)
                let pixel_format_key: id = get_cv_pixel_format_key();
                let bgra_value: id = msg_send![Class::get("NSNumber")?,
                    numberWithUnsignedInt: 0x42475241u32]; // kCVPixelFormatType_32BGRA = 'BGRA'
                let settings: id = msg_send![Class::get("NSDictionary")?,
                    dictionaryWithObject: bgra_value forKey: pixel_format_key];
                let _: () = msg_send![output, setVideoSettings: settings];
                let _: () = msg_send![output, setAlwaysDiscardsLateVideoFrames: YES];

                // Create delegate
                let delegate_class = register_delegate_class();
                let delegate: id = msg_send![delegate_class, new];

                let buffer = Arc::new(Mutex::new([0u8; 128 * 112]));
                let has_new_frame = Arc::new(AtomicBool::new(false));

                let context = Box::into_raw(Box::new(DelegateContext {
                    buffer: Arc::clone(&buffer),
                    has_new_frame: Arc::clone(&has_new_frame),
                }));
                (*delegate).set_ivar("_context", context as *mut c_void);

                // Create a dispatch queue for callbacks
                let queue_label = b"com.vibeboy.camera\0".as_ptr() as *const i8;
                let queue: id = dispatch_queue_create(queue_label, std::ptr::null());

                let _: () = msg_send![output, setSampleBufferDelegate: delegate queue: queue];

                let can_add_output: bool = msg_send![session, canAddOutput: output];
                if !can_add_output {
                    log::warn!("Cannot add video output to session");
                    let _: () = msg_send![session, release];
                    let _: () = msg_send![delegate, release];
                    let _: () = msg_send![output, release];
                    drop(Box::from_raw(context));
                    return None;
                }
                let _: () = msg_send![session, addOutput: output];

                // Start capture
                let _: () = msg_send![session, startRunning];

                log::info!("AVFoundation camera capture started");

                Some(CameraCapture {
                    buffer,
                    has_new_frame,
                    session,
                    _delegate: delegate,
                    _context: context,
                })
            }
        }

        pub fn read_frame(&self, buf: &mut [u8; 128 * 112]) -> bool {
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
            unsafe {
                let _: () = msg_send![self.session, stopRunning];
                let _: () = msg_send![self.session, release];
                let _: () = msg_send![self._delegate, release];
                drop(Box::from_raw(self._context));
            }
            log::info!("AVFoundation camera capture stopped");
        }
    }

    // Force-link AVFoundation so the ObjC runtime finds its classes
    #[link(name = "AVFoundation", kind = "framework")]
    unsafe extern "C" {}

    #[link(name = "System", kind = "dylib")]
    unsafe extern "C" {
        fn dispatch_queue_create(label: *const i8, attr: *const c_void) -> id;
    }
}

use avf_camera::CameraCapture;

// ── Metal renderer ───────────────────────────────────────────────────────────

const METAL_SHADERS: &str = "
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 texcoord;
};

vertex VertexOut vertex_main(uint vid [[vertex_id]],
                             constant float4 *viewport [[buffer(0)]]) {
    // viewport.x = x_offset (NDC), viewport.y = y_offset (NDC)
    // viewport.z = width (NDC), viewport.w = height (NDC)
    float4 vp = viewport[0];
    float2 positions[4] = {
        float2(vp.x,        vp.y),
        float2(vp.x + vp.z, vp.y),
        float2(vp.x,        vp.y + vp.w),
        float2(vp.x + vp.z, vp.y + vp.w),
    };
    float2 texcoords[4] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 0.0),
    };
    VertexOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.texcoord = texcoords[vid];
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]],
                              texture2d<float> tex [[texture(0)]]) {
    constexpr sampler s(mag_filter::nearest, min_filter::nearest);
    return tex.sample(s, in.texcoord);
}
";

struct MetalRenderer {
    device: Device,
    layer: MetalLayer,
    command_queue: CommandQueue,
    pipeline_state: RenderPipelineState,
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

        let command_queue = device.new_command_queue();

        // Compile shaders
        let library = device
            .new_library_with_source(METAL_SHADERS, &CompileOptions::new())
            .expect("Failed to compile Metal shaders");
        let vert_fn = library.get_function("vertex_main", None).unwrap();
        let frag_fn = library.get_function("fragment_main", None).unwrap();

        let pipeline_desc = RenderPipelineDescriptor::new();
        pipeline_desc.set_vertex_function(Some(&vert_fn));
        pipeline_desc.set_fragment_function(Some(&frag_fn));
        pipeline_desc
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);

        let pipeline_state = device
            .new_render_pipeline_state(&pipeline_desc)
            .expect("Failed to create render pipeline state");

        // Create framebuffer texture
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
            pipeline_state,
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

            let dst_tex = drawable.texture();
            let dst_w = dst_tex.width() as f32;
            let dst_h = dst_tex.height() as f32;

            // Compute aspect-ratio-correct viewport in NDC (-1..1)
            let tex_aspect = self.tex_w as f32 / self.tex_h as f32;
            let dst_aspect = dst_w / dst_h;
            let (ndc_w, ndc_h) = if dst_aspect > tex_aspect {
                // Window wider than texture: pillarbox
                (2.0 * tex_aspect / dst_aspect, 2.0)
            } else {
                // Window taller than texture: letterbox
                (2.0, 2.0 * dst_aspect / tex_aspect)
            };
            let ndc_x = -ndc_w / 2.0;
            let ndc_y = -ndc_h / 2.0;
            let viewport: [f32; 4] = [ndc_x, ndc_y, ndc_w, ndc_h];

            let rpd = RenderPassDescriptor::new();
            let ca = rpd.color_attachments().object_at(0).unwrap();
            ca.set_texture(Some(dst_tex));
            ca.set_load_action(MTLLoadAction::Clear);
            ca.set_store_action(MTLStoreAction::Store);
            ca.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));

            let cmd_buf = self.command_queue.new_command_buffer();
            let encoder = cmd_buf.new_render_command_encoder(rpd);

            encoder.set_render_pipeline_state(&self.pipeline_state);
            encoder.set_vertex_bytes(
                0,
                std::mem::size_of::<[f32; 4]>() as u64,
                viewport.as_ptr() as *const _,
            );
            encoder.set_fragment_texture(0, Some(&self.texture));
            encoder.draw_primitives(MTLPrimitiveType::TriangleStrip, 0, 4);
            encoder.end_encoding();

            cmd_buf.present_drawable(&drawable);
            cmd_buf.commit();
        });
    }
}

// ── Menu bar ─────────────────────────────────────────────────────────────────

// Menu item tags for action detection
const MENU_TAG_OPEN: isize = 100;
const MENU_TAG_PAUSE: isize = 101;
const MENU_TAG_RESET: isize = 102;
const MENU_TAG_SAVE_STATE: isize = 103;
const MENU_TAG_LOAD_STATE: isize = 104;
const MENU_TAG_SLOT_BASE: isize = 200; // 200..208 for slots 1-9
const MENU_TAG_MODEL_AUTO: isize = 300;
const MENU_TAG_MODEL_DMG0: isize = 301;
const MENU_TAG_MODEL_DMG: isize = 302;
const MENU_TAG_MODEL_MGB: isize = 303;
const MENU_TAG_MODEL_SGB: isize = 304;
const MENU_TAG_MODEL_SGB2: isize = 305;
const MENU_TAG_MODEL_CGB: isize = 306;
const MENU_TAG_MODEL_AGB: isize = 307;
const MENU_TAG_CONTROLS: isize = 400;
const MENU_TAG_RECENT_BASE: isize = 500; // 500..509 for recent ROMs
const MENU_TAG_CLEAR_RECENT: isize = 510;

// ── Custom key mappings ──────────────────────────────────────────────────────

fn default_key_map() -> HashMap<u16, u8> {
    let mut m = HashMap::new();
    m.insert(6, Emulator::BTN_B);       // Z
    m.insert(7, Emulator::BTN_A);       // X
    m.insert(36, Emulator::BTN_START);   // Return
    m.insert(60, Emulator::BTN_SELECT);  // Right Shift
    m.insert(124, Emulator::BTN_RIGHT);  // Right arrow
    m.insert(123, Emulator::BTN_LEFT);   // Left arrow
    m.insert(126, Emulator::BTN_UP);     // Up arrow
    m.insert(125, Emulator::BTN_DOWN);   // Down arrow
    m
}


fn keycode_name(code: u16) -> &'static str {
    match code {
        0 => "A", 1 => "S", 2 => "D", 3 => "F", 4 => "H", 5 => "G",
        6 => "Z", 7 => "X", 8 => "C", 9 => "V", 11 => "B", 12 => "Q",
        13 => "W", 14 => "E", 15 => "R", 16 => "Y", 17 => "T",
        31 => "O", 32 => "U", 34 => "I", 35 => "P", 37 => "L",
        38 => "J", 40 => "K", 41 => ";", 45 => "N", 46 => "M",
        36 => "Return", 48 => "Tab", 49 => "Space", 51 => "Delete",
        53 => "Escape", 56 => "LShift", 60 => "RShift",
        123 => "Left", 124 => "Right", 125 => "Down", 126 => "Up",
        96 => "F5", 97 => "F6", 98 => "F7", 99 => "F3",
        _ => "?",
    }
}


fn load_key_map() -> HashMap<u16, u8> {
    unsafe {
        let defaults: id = msg_send![class!(NSUserDefaults), standardUserDefaults];
        let key = NSString::alloc(nil).init_str("ControlMappings");
        let dict: id = msg_send![defaults, dictionaryForKey: key];
        if dict == nil {
            return default_key_map();
        }
        let mut map = HashMap::new();
        let keys: id = msg_send![dict, allKeys];
        let count: usize = msg_send![keys, count];
        for i in 0..count {
            let k: id = msg_send![keys, objectAtIndex: i];
            let v: id = msg_send![dict, objectForKey: k];
            let k_str: *const i8 = msg_send![k, UTF8String];
            let v_int: i64 = msg_send![v, integerValue];
            let k_val: u16 = std::ffi::CStr::from_ptr(k_str)
                .to_str().unwrap_or("0").parse().unwrap_or(0);
            map.insert(k_val, v_int as u8);
        }
        if map.is_empty() { default_key_map() } else { map }
    }
}

fn save_key_map(map: &HashMap<u16, u8>) {
    unsafe {
        let dict: id = msg_send![class!(NSMutableDictionary), new];
        for (&keycode, &btn) in map {
            let k = NSString::alloc(nil).init_str(&keycode.to_string());
            let v: id = msg_send![class!(NSNumber), numberWithInteger: btn as isize];
            let _: () = msg_send![dict, setObject: v forKey: k];
        }
        let defaults: id = msg_send![class!(NSUserDefaults), standardUserDefaults];
        let key = NSString::alloc(nil).init_str("ControlMappings");
        let _: () = msg_send![defaults, setObject: dict forKey: key];
        let _: () = msg_send![dict, release];
    }
}

// ── Recent ROMs ─────────────────────────────────────────────────────────────

fn load_recent_roms() -> Vec<String> {
    unsafe {
        let defaults: id = msg_send![class!(NSUserDefaults), standardUserDefaults];
        let key = NSString::alloc(nil).init_str("RecentROMs");
        let arr: id = msg_send![defaults, arrayForKey: key];
        if arr == nil {
            return Vec::new();
        }
        let count: usize = msg_send![arr, count];
        let mut result = Vec::new();
        for i in 0..count {
            let s: id = msg_send![arr, objectAtIndex: i];
            let cstr: *const i8 = msg_send![s, UTF8String];
            let path = std::ffi::CStr::from_ptr(cstr).to_str().unwrap_or("").to_string();
            if !path.is_empty() {
                result.push(path);
            }
        }
        result
    }
}

fn save_recent_roms(roms: &[String]) {
    unsafe {
        let arr: id = msg_send![class!(NSMutableArray), arrayWithCapacity: roms.len()];
        for path in roms {
            let s = NSString::alloc(nil).init_str(path);
            let _: () = msg_send![arr, addObject: s];
        }
        let defaults: id = msg_send![class!(NSUserDefaults), standardUserDefaults];
        let key = NSString::alloc(nil).init_str("RecentROMs");
        let _: () = msg_send![defaults, setObject: arr forKey: key];
    }
}

fn add_recent_rom(path: &str) {
    let mut recents = load_recent_roms();
    recents.retain(|p| p != path);
    recents.insert(0, path.to_string());
    recents.truncate(10);
    save_recent_roms(&recents);
}

fn model_tag_to_model(tag: isize) -> Option<Option<GbModel>> {
    match tag {
        MENU_TAG_MODEL_AUTO => Some(None), // Auto
        MENU_TAG_MODEL_DMG0 => Some(Some(GbModel::Dmg0)),
        MENU_TAG_MODEL_DMG => Some(Some(GbModel::Dmg)),
        MENU_TAG_MODEL_MGB => Some(Some(GbModel::Mgb)),
        MENU_TAG_MODEL_SGB => Some(Some(GbModel::Sgb)),
        MENU_TAG_MODEL_SGB2 => Some(Some(GbModel::Sgb2)),
        MENU_TAG_MODEL_CGB => Some(Some(GbModel::Cgb)),
        MENU_TAG_MODEL_AGB => Some(Some(GbModel::Agb)),
        _ => None,
    }
}

fn auto_detect_model(rom: &[u8]) -> GbModel {
    let cgb_flag = rom.get(0x0143).copied().unwrap_or(0);
    if cgb_flag == 0x80 || cgb_flag == 0xC0 {
        GbModel::Cgb
    } else {
        GbModel::Dmg
    }
}

fn update_model_checkmarks(app: id, selected_tag: isize) {
    unsafe {
        // Emulation menu is at index 2, Hardware submenu is item 2 (0-indexed: after Pause, Reset)
        let emu_menu_item: id = msg_send![app.mainMenu(), itemAtIndex: 2isize];
        let emu_submenu: id = msg_send![emu_menu_item, submenu];
        let hw_item: id = msg_send![emu_submenu, itemAtIndex: 3isize]; // after pause, reset, separator
        let hw_submenu: id = msg_send![hw_item, submenu];
        if hw_submenu == nil { return; }
        let count: isize = msg_send![hw_submenu, numberOfItems];
        for i in 0..count {
            let item: id = msg_send![hw_submenu, itemAtIndex: i];
            let tag: isize = msg_send![item, tag];
            let state: isize = if tag == selected_tag { 1 } else { 0 }; // NSOnState=1, NSOffState=0
            let _: () = msg_send![item, setState: state];
        }
    }
}

fn rebuild_recent_menu(app: id, recents: &[String]) {
    unsafe {
        // File menu is at index 1, "Recent ROMs" submenu is item at index 2 (after Open, separator)
        let file_menu_item: id = msg_send![app.mainMenu(), itemAtIndex: 1isize];
        let file_submenu: id = msg_send![file_menu_item, submenu];
        let recent_item: id = msg_send![file_submenu, itemAtIndex: 2isize];
        let recent_submenu: id = msg_send![recent_item, submenu];
        if recent_submenu == nil { return; }
        let _: () = msg_send![recent_submenu, removeAllItems];
        for (i, path) in recents.iter().enumerate() {
            let display = PathBuf::from(path);
            let name = display.file_name().unwrap_or_default().to_string_lossy();
            let item = menu_item_with_tag(
                &name, sel!(menuAction:), "",
                MENU_TAG_RECENT_BASE + i as isize,
            );
            let _: () = msg_send![recent_submenu, addItem: item];
        }
        if !recents.is_empty() {
            let _: () = msg_send![recent_submenu, addItem: NSMenuItem::separatorItem(nil)];
        }
        let clear = menu_item_with_tag("Clear Recent", sel!(menuAction:), "", MENU_TAG_CLEAR_RECENT);
        let _: () = msg_send![recent_submenu, addItem: clear];
    }
}

unsafe fn create_menu_bar(app: id) {
    let main_menu = NSMenu::new(nil).autorelease();

    // ── VibeBoy menu ─────────────────────────────────────────────────────
    let app_menu = NSMenu::new(nil).autorelease();
    let _: () = msg_send![app_menu, setTitle: NSString::alloc(nil).init_str("VibeBoy")];

    let about_item = menu_item("About VibeBoy", sel!(orderFrontStandardAboutPanel:), "");
    let _: () = msg_send![app_menu, addItem: about_item];
    let _: () = msg_send![app_menu, addItem: NSMenuItem::separatorItem(nil)];

    let quit_item = menu_item("Quit VibeBoy", sel!(terminate:), "q");
    let _: () = msg_send![app_menu, addItem: quit_item];

    let app_menu_item = NSMenuItem::new(nil).autorelease();
    let _: () = msg_send![app_menu_item, setSubmenu: app_menu];
    let _: () = msg_send![main_menu, addItem: app_menu_item];

    // ── File menu ────────────────────────────────────────────────────────
    let file_menu = NSMenu::new(nil).autorelease();
    let _: () = msg_send![file_menu, setTitle: NSString::alloc(nil).init_str("File")];

    let open_item = menu_item_with_tag("Open ROM\u{2026}", sel!(menuAction:), "o", MENU_TAG_OPEN);
    let _: () = msg_send![file_menu, addItem: open_item];
    let _: () = msg_send![file_menu, addItem: NSMenuItem::separatorItem(nil)];

    // Recent ROMs submenu
    let recent_submenu = NSMenu::new(nil).autorelease();
    let _: () = msg_send![recent_submenu, setTitle: NSString::alloc(nil).init_str("Recent ROMs")];
    let clear_item = menu_item_with_tag("Clear Recent", sel!(menuAction:), "", MENU_TAG_CLEAR_RECENT);
    let _: () = msg_send![recent_submenu, addItem: clear_item];
    let recent_menu_item = NSMenuItem::new(nil).autorelease();
    let _: () = msg_send![recent_menu_item, setTitle: NSString::alloc(nil).init_str("Recent ROMs")];
    let _: () = msg_send![recent_menu_item, setSubmenu: recent_submenu];
    let _: () = msg_send![file_menu, addItem: recent_menu_item];
    let _: () = msg_send![file_menu, addItem: NSMenuItem::separatorItem(nil)];

    let close_item = menu_item("Close Window", sel!(performClose:), "w");
    let _: () = msg_send![file_menu, addItem: close_item];

    let file_menu_item = NSMenuItem::new(nil).autorelease();
    let _: () = msg_send![file_menu_item, setSubmenu: file_menu];
    let _: () = msg_send![main_menu, addItem: file_menu_item];

    // ── Emulation menu ───────────────────────────────────────────────────
    let emu_menu = NSMenu::new(nil).autorelease();
    let _: () = msg_send![emu_menu, setTitle: NSString::alloc(nil).init_str("Emulation")];

    let pause_item = menu_item_with_tag("Pause", sel!(menuAction:), "p", MENU_TAG_PAUSE);
    let _: () = msg_send![emu_menu, addItem: pause_item];

    let reset_item = menu_item_with_tag("Reset", sel!(menuAction:), "r", MENU_TAG_RESET);
    let _: () = msg_send![emu_menu, addItem: reset_item];
    let _: () = msg_send![emu_menu, addItem: NSMenuItem::separatorItem(nil)];

    // Hardware submenu
    let hw_submenu = NSMenu::new(nil).autorelease();
    let _: () = msg_send![hw_submenu, setTitle: NSString::alloc(nil).init_str("Hardware")];
    let models = [
        ("Auto", MENU_TAG_MODEL_AUTO),
        ("DMG0", MENU_TAG_MODEL_DMG0),
        ("DMG", MENU_TAG_MODEL_DMG),
        ("MGB", MENU_TAG_MODEL_MGB),
        ("SGB", MENU_TAG_MODEL_SGB),
        ("SGB2", MENU_TAG_MODEL_SGB2),
        ("CGB", MENU_TAG_MODEL_CGB),
        ("AGB", MENU_TAG_MODEL_AGB),
    ];
    for (name, tag) in &models {
        let item = menu_item_with_tag(name, sel!(menuAction:), "", *tag);
        if *tag == MENU_TAG_MODEL_AUTO {
            let _: () = msg_send![item, setState: 1isize]; // NSOnState — checked by default
        }
        let _: () = msg_send![hw_submenu, addItem: item];
    }
    let hw_menu_item = NSMenuItem::new(nil).autorelease();
    let _: () = msg_send![hw_menu_item, setTitle: NSString::alloc(nil).init_str("Hardware")];
    let _: () = msg_send![hw_menu_item, setSubmenu: hw_submenu];
    let _: () = msg_send![emu_menu, addItem: hw_menu_item];

    let _: () = msg_send![emu_menu, addItem: NSMenuItem::separatorItem(nil)];
    let controls_item = menu_item_with_tag("Controls\u{2026}", sel!(menuAction:), "", MENU_TAG_CONTROLS);
    let _: () = msg_send![emu_menu, addItem: controls_item];

    let emu_menu_item = NSMenuItem::new(nil).autorelease();
    let _: () = msg_send![emu_menu_item, setSubmenu: emu_menu];
    let _: () = msg_send![main_menu, addItem: emu_menu_item];

    // ── State menu ───────────────────────────────────────────────────────
    let state_menu = NSMenu::new(nil).autorelease();
    let _: () = msg_send![state_menu, setTitle: NSString::alloc(nil).init_str("State")];

    let save_item = menu_item_with_tag_and_key("Save State", sel!(menuAction:), MENU_TAG_SAVE_STATE, K_F5_EQUIV);
    let _: () = msg_send![state_menu, addItem: save_item];
    let load_item = menu_item_with_tag_and_key("Load State", sel!(menuAction:), MENU_TAG_LOAD_STATE, K_F7_EQUIV);
    let _: () = msg_send![state_menu, addItem: load_item];
    let _: () = msg_send![state_menu, addItem: NSMenuItem::separatorItem(nil)];

    for slot in 1..=9usize {
        let title = format!("Slot {}", slot);
        let key = format!("{}", slot);
        let item = menu_item_with_tag(&title, sel!(menuAction:), &key, MENU_TAG_SLOT_BASE + slot as isize - 1);
        let _: () = msg_send![item, setKeyEquivalentModifierMask: 0u64]; // no modifier
        let _: () = msg_send![state_menu, addItem: item];
    }

    let state_menu_item = NSMenuItem::new(nil).autorelease();
    let _: () = msg_send![state_menu_item, setSubmenu: state_menu];
    let _: () = msg_send![main_menu, addItem: state_menu_item];

    // ── Window menu ──────────────────────────────────────────────────────
    let window_menu = NSMenu::new(nil).autorelease();
    let _: () = msg_send![window_menu, setTitle: NSString::alloc(nil).init_str("Window")];

    let minimize_item = menu_item("Minimize", sel!(performMiniaturize:), "m");
    let _: () = msg_send![window_menu, addItem: minimize_item];
    let zoom_item = menu_item("Zoom", sel!(performZoom:), "");
    let _: () = msg_send![window_menu, addItem: zoom_item];
    let _: () = msg_send![window_menu, addItem: NSMenuItem::separatorItem(nil)];
    let front_item = menu_item("Bring All to Front", sel!(arrangeInFront:), "");
    let _: () = msg_send![window_menu, addItem: front_item];

    let window_menu_item = NSMenuItem::new(nil).autorelease();
    let _: () = msg_send![window_menu_item, setSubmenu: window_menu];
    let _: () = msg_send![main_menu, addItem: window_menu_item];

    app.setMainMenu_(main_menu);
    let _: () = msg_send![app, setWindowsMenu: window_menu];
}

// Function key equivalents use Unicode private-use characters
const K_F5_EQUIV: &str = "\u{F708}";  // NSF5FunctionKey
const K_F7_EQUIV: &str = "\u{F70A}";  // NSF7FunctionKey

unsafe fn menu_item(title: &str, action: SEL, key: &str) -> id {
    NSMenuItem::alloc(nil).initWithTitle_action_keyEquivalent_(
        NSString::alloc(nil).init_str(title),
        action,
        NSString::alloc(nil).init_str(key),
    ).autorelease()
}

unsafe fn menu_item_with_tag(title: &str, action: SEL, key: &str, tag: isize) -> id {
    let item = NSMenuItem::alloc(nil).initWithTitle_action_keyEquivalent_(
        NSString::alloc(nil).init_str(title),
        action,
        NSString::alloc(nil).init_str(key),
    ).autorelease();
    let _: () = msg_send![item, setTag: tag];
    // Target the app delegate (first responder chain will route to us)
    item
}

unsafe fn menu_item_with_tag_and_key(title: &str, action: SEL, tag: isize, key: &str) -> id {
    let item = NSMenuItem::alloc(nil).initWithTitle_action_keyEquivalent_(
        NSString::alloc(nil).init_str(title),
        action,
        NSString::alloc(nil).init_str(key),
    ).autorelease();
    let _: () = msg_send![item, setTag: tag];
    // Function keys need NSFunctionKeyMask
    let ns_function_key_mask: u64 = 1 << 23;
    let _: () = msg_send![item, setKeyEquivalentModifierMask: ns_function_key_mask];
    item
}

// Register an ObjC class to handle menu actions
mod menu_handler {
    use super::*;
    use objc::declare::ClassDecl;
    use objc::runtime::{Class, Object, Sel};
    use std::os::raw::c_void;
    use std::sync::Once;

    // Action flags polled by the main loop
    pub struct MenuActions {
        pub open_rom: bool,
        pub pause_toggle: bool,
        pub reset: bool,
        pub save_state: bool,
        pub load_state: bool,
        pub select_slot: Option<usize>,
        pub select_model: Option<isize>,  // tag of selected model
        pub open_controls: bool,
        pub open_recent: Option<usize>,   // index into recent ROMs list
        pub clear_recent: bool,
    }

    impl MenuActions {
        pub fn new() -> Self {
            MenuActions {
                open_rom: false,
                pause_toggle: false,
                reset: false,
                save_state: false,
                load_state: false,
                select_slot: None,
                select_model: None,
                open_controls: false,
                open_recent: None,
                clear_recent: false,
            }
        }

        pub fn take_all(&mut self) -> MenuActions {
            std::mem::replace(self, MenuActions::new())
        }
    }

    static REGISTER: Once = Once::new();

    pub fn register_class() -> &'static Class {
        REGISTER.call_once(|| {
            let superclass = Class::get("NSObject").unwrap();
            let mut decl = ClassDecl::new("VBMenuHandler", superclass).unwrap();

            decl.add_ivar::<*mut c_void>("_actions");

            unsafe {
                decl.add_method(
                    sel!(menuAction:),
                    handle_menu_action as extern "C" fn(&Object, Sel, id),
                );
                // Respond YES to validateMenuItem: so our items are always enabled
                decl.add_method(
                    sel!(validateMenuItem:),
                    validate_menu_item as extern "C" fn(&Object, Sel, id) -> bool,
                );
                decl.add_method(
                    sel!(applicationDockMenu:),
                    application_dock_menu as extern "C" fn(&Object, Sel, id) -> id,
                );
            }

            decl.register();
        });
        Class::get("VBMenuHandler").unwrap()
    }

    extern "C" fn validate_menu_item(_this: &Object, _sel: Sel, _item: id) -> bool {
        true
    }

    extern "C" fn handle_menu_action(this: &Object, _sel: Sel, sender: id) {
        unsafe {
            let ctx_ptr: *mut c_void = *this.get_ivar("_actions");
            if ctx_ptr.is_null() {
                return;
            }
            let actions = &mut *(ctx_ptr as *mut MenuActions);

            let tag: isize = msg_send![sender, tag];
            match tag {
                super::MENU_TAG_OPEN => actions.open_rom = true,
                super::MENU_TAG_PAUSE => actions.pause_toggle = true,
                super::MENU_TAG_RESET => actions.reset = true,
                super::MENU_TAG_SAVE_STATE => actions.save_state = true,
                super::MENU_TAG_LOAD_STATE => actions.load_state = true,
                t if t >= super::MENU_TAG_SLOT_BASE && t < super::MENU_TAG_SLOT_BASE + 9 => {
                    actions.select_slot = Some((t - super::MENU_TAG_SLOT_BASE) as usize);
                }
                t if t >= super::MENU_TAG_MODEL_AUTO && t <= super::MENU_TAG_MODEL_AGB => {
                    actions.select_model = Some(t);
                }
                super::MENU_TAG_CONTROLS => actions.open_controls = true,
                t if t >= super::MENU_TAG_RECENT_BASE && t < super::MENU_TAG_RECENT_BASE + 10 => {
                    actions.open_recent = Some((t - super::MENU_TAG_RECENT_BASE) as usize);
                }
                super::MENU_TAG_CLEAR_RECENT => actions.clear_recent = true,
                _ => {}
            }
        }
    }

    extern "C" fn application_dock_menu(_this: &Object, _sel: Sel, _app: id) -> id {
        unsafe {
            let recents = super::load_recent_roms();
            if recents.is_empty() {
                return nil;
            }
            let menu = NSMenu::new(nil).autorelease();
            for (i, path) in recents.iter().enumerate() {
                let display = PathBuf::from(path);
                let name = display.file_name().unwrap_or_default().to_string_lossy();
                let item = super::menu_item_with_tag(
                    &name, sel!(menuAction:), "",
                    super::MENU_TAG_RECENT_BASE + i as isize,
                );
                let _: () = msg_send![menu, addItem: item];
            }
            menu
        }
    }

    /// Create a menu handler instance and wire it up. Returns (handler_id, actions_ptr).
    pub unsafe fn create(app: id) -> (id, *mut MenuActions) {
        let class = register_class();
        let handler: id = msg_send![class, new];

        let actions = Box::into_raw(Box::new(MenuActions::new()));
        (*handler).set_ivar("_actions", actions as *mut c_void);

        // Set as first responder target for menu items that use menuAction: selector.
        // We do this by making the handler the app's delegate — the responder chain
        // sends unhandled actions up to the app delegate.
        let _: () = msg_send![app, setDelegate: handler];

        (handler, actions)
    }
}

fn open_rom_dialog() -> Option<PathBuf> {
    unsafe {
        let panel: id = msg_send![class!(NSOpenPanel), openPanel];
        let _: () = msg_send![panel, setCanChooseFiles: YES];
        let _: () = msg_send![panel, setCanChooseDirectories: NO];
        let _: () = msg_send![panel, setAllowsMultipleSelection: NO];

        let gb = NSString::alloc(nil).init_str("gb");
        let gbc = NSString::alloc(nil).init_str("gbc");
        let types: id = msg_send![class!(NSMutableArray), arrayWithCapacity: 2usize];
        let _: () = msg_send![types, addObject: gb];
        let _: () = msg_send![types, addObject: gbc];
        let _: () = msg_send![panel, setAllowedFileTypes: types];

        let response: isize = msg_send![panel, runModal];
        if response != 1 {
            return None;
        }

        let url: id = msg_send![panel, URL];
        let path: id = msg_send![url, path];
        let path_str: *const i8 = msg_send![path, UTF8String];
        let path = std::ffi::CStr::from_ptr(path_str)
            .to_str()
            .unwrap()
            .to_string();
        Some(PathBuf::from(path))
    }
}

// ── Controls Panel ───────────────────────────────────────────────────────────

fn show_controls_panel(key_map: &mut HashMap<u16, u8>) {
    unsafe {
        let panel_rect = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(360.0, 340.0));
        let style = NSWindowStyleMask::NSTitledWindowMask
            | NSWindowStyleMask::NSClosableWindowMask;
        let panel: id = msg_send![class!(NSPanel),
            alloc];
        let panel: id = NSWindow::initWithContentRect_styleMask_backing_defer_(
            panel, panel_rect, style,
            NSBackingStoreType::NSBackingStoreBuffered, NO,
        );
        let _: () = msg_send![panel, setTitle: NSString::alloc(nil).init_str("Controls")];
        let _: () = msg_send![panel, center];

        let content: id = msg_send![panel, contentView];

        // Build reverse map: button -> keycode
        let mut btn_to_key: HashMap<u8, u16> = HashMap::new();
        for (&keycode, &btn) in key_map.iter() {
            btn_to_key.insert(btn, keycode);
        }

        // Create labels for each button
        let button_order = [
            (Emulator::BTN_UP, "Up"),
            (Emulator::BTN_DOWN, "Down"),
            (Emulator::BTN_LEFT, "Left"),
            (Emulator::BTN_RIGHT, "Right"),
            (Emulator::BTN_A, "A"),
            (Emulator::BTN_B, "B"),
            (Emulator::BTN_START, "Start"),
            (Emulator::BTN_SELECT, "Select"),
        ];

        let header = NSString::alloc(nil).init_str("Click a key binding, then press a new key to reassign.\nPress Escape to cancel.");
        let header_frame = NSRect::new(NSPoint::new(20.0, 290.0), NSSize::new(320.0, 40.0));
        let header_label: id = msg_send![class!(NSTextField), alloc];
        let header_label: id = msg_send![header_label, initWithFrame: header_frame];
        let _: () = msg_send![header_label, setStringValue: header];
        let _: () = msg_send![header_label, setBezeled: NO];
        let _: () = msg_send![header_label, setDrawsBackground: NO];
        let _: () = msg_send![header_label, setEditable: NO];
        let _: () = msg_send![header_label, setSelectable: NO];
        let font: id = msg_send![class!(NSFont), systemFontOfSize: 11.0f64];
        let _: () = msg_send![header_label, setFont: font];
        let _: () = msg_send![content, addSubview: header_label];

        let mut key_labels: Vec<(u8, id)> = Vec::new();

        for (i, &(btn, name)) in button_order.iter().enumerate() {
            let y = 250.0 - (i as f64 * 30.0);

            // Action name label
            let name_frame = NSRect::new(NSPoint::new(30.0, y), NSSize::new(100.0, 24.0));
            let name_label: id = msg_send![class!(NSTextField), alloc];
            let name_label: id = msg_send![name_label, initWithFrame: name_frame];
            let _: () = msg_send![name_label, setStringValue: NSString::alloc(nil).init_str(name)];
            let _: () = msg_send![name_label, setBezeled: NO];
            let _: () = msg_send![name_label, setDrawsBackground: NO];
            let _: () = msg_send![name_label, setEditable: NO];
            let _: () = msg_send![name_label, setSelectable: NO];
            let bold_font: id = msg_send![class!(NSFont), boldSystemFontOfSize: 13.0f64];
            let _: () = msg_send![name_label, setFont: bold_font];
            let _: () = msg_send![content, addSubview: name_label];

            // Key binding button
            let key_name = btn_to_key.get(&btn)
                .map(|&k| keycode_name(k))
                .unwrap_or("(none)");
            let btn_frame = NSRect::new(NSPoint::new(150.0, y), NSSize::new(120.0, 24.0));
            let btn_view: id = msg_send![class!(NSButton), alloc];
            let btn_view: id = msg_send![btn_view, initWithFrame: btn_frame];
            let _: () = msg_send![btn_view, setTitle: NSString::alloc(nil).init_str(key_name)];
            let _: () = msg_send![btn_view, setBezelStyle: 1isize]; // NSRoundedBezelStyle
            let _: () = msg_send![btn_view, setTag: btn as isize];
            let _: () = msg_send![content, addSubview: btn_view];

            key_labels.push((btn, btn_view));
        }

        // Reset to Defaults button
        let reset_frame = NSRect::new(NSPoint::new(115.0, 10.0), NSSize::new(130.0, 30.0));
        let reset_btn: id = msg_send![class!(NSButton), alloc];
        let reset_btn: id = msg_send![reset_btn, initWithFrame: reset_frame];
        let _: () = msg_send![reset_btn, setTitle: NSString::alloc(nil).init_str("Reset to Defaults")];
        let _: () = msg_send![reset_btn, setBezelStyle: 1isize];
        let _: () = msg_send![content, addSubview: reset_btn];

        // Run as modal, handle key presses for remapping
        let _: () = msg_send![panel, makeKeyAndOrderFront: nil];

        let app: id = msg_send![class!(NSApplication), sharedApplication];

        // Simple modal loop: click a button, then press a key
        let mut waiting_for_key: Option<u8> = None;

        loop {
            let event: id = msg_send![app,
                nextEventMatchingMask: u64::MAX
                untilDate: { let d: id = msg_send![class!(NSDate), distantFuture]; d }
                inMode: NSString::alloc(nil).init_str("kCFRunLoopDefaultMode")
                dequeue: YES
            ];

            if event == nil { continue; }

            let event_type: u64 = msg_send![event, type];

            // Check if panel was closed
            let visible: bool = msg_send![panel, isVisible];
            if !visible { break; }

            if event_type == NSEventType::NSKeyDown as u64 {
                let keycode: u16 = msg_send![event, keyCode];

                if let Some(btn) = waiting_for_key {
                    if keycode == K_ESCAPE {
                        // Cancel remapping
                        waiting_for_key = None;
                        // Restore button title
                        for &(b, label) in &key_labels {
                            if b == btn {
                                let cur_name = btn_to_key.get(&b)
                                    .map(|&k| keycode_name(k))
                                    .unwrap_or("(none)");
                                let _: () = msg_send![label, setTitle:
                                    NSString::alloc(nil).init_str(cur_name)];
                            }
                        }
                        continue;
                    }

                    // Remove old mapping for this button
                    key_map.retain(|_, &mut v| v != btn);
                    // Remove any existing mapping for this keycode
                    key_map.remove(&keycode);
                    // Set new mapping
                    key_map.insert(keycode, btn);
                    btn_to_key.insert(btn, keycode);
                    save_key_map(key_map);

                    // Update button title
                    for &(b, label) in &key_labels {
                        if b == btn {
                            let _: () = msg_send![label, setTitle:
                                NSString::alloc(nil).init_str(keycode_name(keycode))];
                        }
                    }
                    waiting_for_key = None;
                    continue;
                }

                if keycode == K_ESCAPE {
                    break;
                }
            } else if event_type == NSEventType::NSLeftMouseUp as u64 {
                // Check if a key label button was clicked
                let location: NSPoint = msg_send![event, locationInWindow];
                for &(btn, label) in &key_labels {
                    let frame: NSRect = msg_send![label, frame];
                    if location.x >= frame.origin.x
                        && location.x <= frame.origin.x + frame.size.width
                        && location.y >= frame.origin.y
                        && location.y <= frame.origin.y + frame.size.height
                    {
                        waiting_for_key = Some(btn);
                        let _: () = msg_send![label, setTitle:
                            NSString::alloc(nil).init_str("Press a key...")];
                        break;
                    }
                }

                // Check if Reset to Defaults was clicked
                let reset_frame: NSRect = msg_send![reset_btn, frame];
                if location.x >= reset_frame.origin.x
                    && location.x <= reset_frame.origin.x + reset_frame.size.width
                    && location.y >= reset_frame.origin.y
                    && location.y <= reset_frame.origin.y + reset_frame.size.height
                {
                    *key_map = default_key_map();
                    save_key_map(key_map);
                    btn_to_key.clear();
                    for (&keycode, &btn) in key_map.iter() {
                        btn_to_key.insert(btn, keycode);
                    }
                    for &(btn, label) in &key_labels {
                        let name = btn_to_key.get(&btn)
                            .map(|&k| keycode_name(k))
                            .unwrap_or("(none)");
                        let _: () = msg_send![label, setTitle:
                            NSString::alloc(nil).init_str(name)];
                    }
                    waiting_for_key = None;
                }
            }

            let _: () = msg_send![app, sendEvent: event];
        }

        let _: () = msg_send![panel, close];
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

        let cli_model: Option<GbModel> = if cli.model == "auto" {
            None
        } else {
            Some(cli.model.parse::<GbModel>().unwrap_or_else(|e| {
                eprintln!("{}", e);
                std::process::exit(1);
            }))
        };

        // forced_model: None = auto-detect, Some(m) = user override
        let mut forced_model: Option<GbModel> = cli_model;
        let model = forced_model.unwrap_or_else(|| auto_detect_model(&rom));
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

        let title = format!("VibeBoy \u{2014} {}",
            current_rom_path.file_name().unwrap_or_default().to_string_lossy());
        window.setTitle_(NSString::alloc(nil).init_str(&title));
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
        let accel_source = if emu.bus.cart.has_accelerometer() {
            init_accel()
        } else {
            AccelSource::None
        };

        // ── Key state + frame loop ───────────────────────────────────────────
        let mut keys_down = std::collections::HashSet::<u16>::new();
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
                    let emu_menu: id = msg_send![app.mainMenu(), itemAtIndex: 2isize];
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
                    eprintln!("State saved to slot {}", current_slot + 1);
                }

                if actions.load_state {
                    if emu.load_state(current_slot) {
                        eprintln!("State loaded from slot {}", current_slot + 1);
                    } else {
                        eprintln!("Slot {} is empty", current_slot + 1);
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

            // ── Camera ───────────────────────────────────────────────────────
            if let Some(ref cam) = camera {
                if cam.read_frame(&mut camera_buf) {
                    emu.bus.cart.set_camera_image(&camera_buf);
                }
            }

            // ── Accelerometer ────────────────────────────────────────────────
            if let Some((x, y, _z)) = poll_accel(&accel_source) {
                const CENTER: f32 = 0x81D0 as u16 as f32;
                const RANGE: f32 = 0x70 as u16 as f32;
                let mbc7_x = (CENTER + (-x) * RANGE).clamp(0.0, 65535.0) as u16;
                let mbc7_y = (CENTER + y * RANGE).clamp(0.0, 65535.0) as u16;
                emu.bus.cart.set_accelerometer(mbc7_x, mbc7_y);
            }

            // ── Rewind / Fast-forward ────────────────────────────────────────
            let backspace_held = keys_down.contains(&K_DELETE);
            let fast_forward = keys_down.contains(&K_TAB);
            emu.rewinding = backspace_held;

            if !paused {
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

            // ── Update drawable size on resize ─────────────────────────────
            {
                let bounds: NSRect = msg_send![content_view, bounds];
                let backing: NSSize = msg_send![content_view,
                    convertSizeToBacking: bounds.size];
                renderer.layer.set_drawable_size(CGSize::new(
                    backing.width, backing.height,
                ));
            }

            // ── Render ───────────────────────────────────────────────────────
            {
                let (src, w, h) = if is_sgb {
                    (emu.sgb_composited_frame(), 256usize, 224usize)
                } else {
                    (emu.frame_buffer(), 160usize, 144usize)
                };

                // Convert 0x00RRGGBB → BGRA8Unorm (just set alpha to 0xFF)
                // BGRA8 on LE: u32 = A<<24 | R<<16 | G<<8 | B = 0xFF000000 | src
                bgra_buf.resize(w * h, 0u32);
                for i in 0..(w * h) {
                    bgra_buf[i] = 0xFF00_0000 | src[i];
                }

                renderer.update_texture(&bgra_buf);
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
        close_accel(&accel_source);
        drop(camera);

        emu.save();
    }
}
