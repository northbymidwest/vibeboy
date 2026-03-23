//! libretro core implementation for VibeBoy.
//!
//! Implements the libretro API as extern "C" functions. The core is compiled as a
//! cdylib (vibeboy_libretro.so/.dll/.dylib) and loaded by RetroArch or other
//! libretro frontends.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_uint, c_void};
use std::ptr;
use std::sync::OnceLock;

use crate::clock;
use crate::emulator::Emulator;
use crate::model::GbModel;
use crate::ui_util;

// ── libretro constants ──────────────────────────────────────────────────────

const RETRO_API_VERSION: c_uint = 1;

// Pixel format
const RETRO_PIXEL_FORMAT_XRGB8888: c_uint = 1;

// Device types
const RETRO_DEVICE_JOYPAD: c_uint = 1;

// Joypad buttons
const RETRO_DEVICE_ID_JOYPAD_B: c_uint = 0;
const RETRO_DEVICE_ID_JOYPAD_Y: c_uint = 1;
const RETRO_DEVICE_ID_JOYPAD_SELECT: c_uint = 2;
const RETRO_DEVICE_ID_JOYPAD_START: c_uint = 3;
const RETRO_DEVICE_ID_JOYPAD_UP: c_uint = 4;
const RETRO_DEVICE_ID_JOYPAD_DOWN: c_uint = 5;
const RETRO_DEVICE_ID_JOYPAD_LEFT: c_uint = 6;
const RETRO_DEVICE_ID_JOYPAD_RIGHT: c_uint = 7;
const RETRO_DEVICE_ID_JOYPAD_A: c_uint = 8;

// Environment commands
const RETRO_ENVIRONMENT_SET_PIXEL_FORMAT: c_uint = 10;
const RETRO_ENVIRONMENT_SET_VARIABLES: c_uint = 16;
const RETRO_ENVIRONMENT_GET_VARIABLE: c_uint = 15;
const RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY: c_uint = 9;
const RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY: c_uint = 31;

// Memory types
const RETRO_MEMORY_SAVE_RAM: c_uint = 0;

// ── libretro callback types ─────────────────────────────────────────────────

type EnvironmentFn = unsafe extern "C" fn(cmd: c_uint, data: *mut c_void) -> bool;
type VideoRefreshFn = unsafe extern "C" fn(data: *const c_void, width: c_uint, height: c_uint, pitch: usize);
type AudioSampleFn = unsafe extern "C" fn(left: i16, right: i16);
type AudioSampleBatchFn = unsafe extern "C" fn(data: *const i16, frames: usize) -> usize;
type InputPollFn = unsafe extern "C" fn();
type InputStateFn = unsafe extern "C" fn(port: c_uint, device: c_uint, index: c_uint, id: c_uint) -> i16;

// ── libretro structs ────────────────────────────────────────────────────────

#[repr(C)]
pub struct RetroSystemInfo {
    library_name: *const c_char,
    library_version: *const c_char,
    valid_extensions: *const c_char,
    need_fullpath: bool,
    block_extract: bool,
}

#[repr(C)]
pub struct RetroGameGeometry {
    base_width: c_uint,
    base_height: c_uint,
    max_width: c_uint,
    max_height: c_uint,
    aspect_ratio: f32,
}

#[repr(C)]
pub struct RetroSystemTiming {
    fps: f64,
    sample_rate: f64,
}

#[repr(C)]
pub struct RetroSystemAvInfo {
    geometry: RetroGameGeometry,
    timing: RetroSystemTiming,
}

#[repr(C)]
pub struct RetroGameInfo {
    path: *const c_char,
    data: *const c_void,
    size: usize,
    meta: *const c_char,
}

#[repr(C)]
struct RetroVariable {
    key: *const c_char,
    value: *const c_char,
}

// ── Core state ──────────────────────────────────────────────────────────────

struct CoreState {
    emu: Emulator,
    rom: std::sync::Arc<[u8]>,
    audio_buf_i16: Vec<i16>,
    video_buf: Vec<u32>,
    model: GbModel,
    use_xrgb8888: bool,
}

static LIB_NAME: &[u8] = b"VibeBoy\0";
static LIB_VERSION: &[u8] = b"0.1.0\0";
static EXTENSIONS: &[u8] = b"gb|gbc|sgb\0";

// Audio output rate (libretro standard). We downsample from 96kHz.
const AUDIO_RATE: f64 = 48000.0;
const DOWNSAMPLE_FACTOR: usize = 2; // 96kHz / 2 = 48kHz

// Global state (libretro API is C-style, single-threaded, no context object).
// We use raw pointer access to avoid Rust 2024's static_mut_refs lint.
static mut CORE: Option<CoreState> = None;
static mut CB_ENVIRONMENT: Option<EnvironmentFn> = None;
static mut CB_VIDEO_REFRESH: Option<VideoRefreshFn> = None;
static mut CB_AUDIO_SAMPLE: Option<AudioSampleFn> = None;
static mut CB_AUDIO_SAMPLE_BATCH: Option<AudioSampleBatchFn> = None;
static mut CB_INPUT_POLL: Option<InputPollFn> = None;
static mut CB_INPUT_STATE: Option<InputStateFn> = None;

/// Access global core state (libretro is single-threaded).
unsafe fn core_mut() -> Option<&'static mut CoreState> {
    unsafe { std::ptr::addr_of_mut!(CORE).as_mut().and_then(|o| o.as_mut()) }
}

unsafe fn env_cb() -> Option<EnvironmentFn> {
    unsafe { std::ptr::addr_of!(CB_ENVIRONMENT).read() }
}

unsafe fn video_cb() -> Option<VideoRefreshFn> {
    unsafe { std::ptr::addr_of!(CB_VIDEO_REFRESH).read() }
}

unsafe fn audio_batch_cb() -> Option<AudioSampleBatchFn> {
    unsafe { std::ptr::addr_of!(CB_AUDIO_SAMPLE_BATCH).read() }
}

unsafe fn input_poll_cb() -> Option<InputPollFn> {
    unsafe { std::ptr::addr_of!(CB_INPUT_POLL).read() }
}

unsafe fn input_state_cb() -> Option<InputStateFn> {
    unsafe { std::ptr::addr_of!(CB_INPUT_STATE).read() }
}

// Static CStrings for variables
static VARS_MODEL_KEY: &[u8] = b"vibeboy_model\0";
static VARS_MODEL_DESC: &[u8] = b"Hardware Model; Auto|DMG|DMG0|MGB|SGB|SGB2|CGB|AGB\0";

// ── Helper functions ────────────────────────────────────────────────────────

fn detect_model(rom: &[u8]) -> GbModel {
    ui_util::auto_detect_model(rom)
}

fn get_model_from_options() -> Option<GbModel> {
    unsafe {
        let env = env_cb()?;
        let key = VARS_MODEL_KEY.as_ptr() as *const c_char;
        let mut var = RetroVariable { key, value: ptr::null() };
        if env(RETRO_ENVIRONMENT_GET_VARIABLE, &mut var as *mut _ as *mut c_void) && !var.value.is_null() {
            let val = CStr::from_ptr(var.value).to_str().ok()?;
            match val {
                "DMG" => Some(GbModel::Dmg),
                "DMG0" => Some(GbModel::Dmg0),
                "MGB" => Some(GbModel::Mgb),
                "SGB" => Some(GbModel::Sgb),
                "SGB2" => Some(GbModel::Sgb2),
                "CGB" => Some(GbModel::Cgb),
                "AGB" => Some(GbModel::Agb),
                _ => None, // "Auto"
            }
        } else {
            None
        }
    }
}

fn get_system_dir() -> Option<std::path::PathBuf> {
    unsafe {
        let env = env_cb()?;
        let mut dir: *const c_char = ptr::null();
        if env(RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY, &mut dir as *mut _ as *mut c_void) && !dir.is_null() {
            Some(std::path::PathBuf::from(CStr::from_ptr(dir).to_str().ok()?))
        } else {
            None
        }
    }
}

fn load_boot_rom(model: GbModel) -> Option<Vec<u8>> {
    let sys_dir = get_system_dir()?;
    let filename = match model {
        GbModel::Dmg0 => "dmg0_boot.bin",
        GbModel::Dmg => "dmg_boot.bin",
        GbModel::Mgb => "mgb_boot.bin",
        GbModel::Sgb => "sgb_boot.bin",
        GbModel::Sgb2 => "sgb2_boot.bin",
        GbModel::Cgb0 => "cgb0_boot.bin",
        GbModel::Cgb => "cgb_boot.bin",
        GbModel::Agb => "cgb_agb_boot.bin",
    };
    std::fs::read(sys_dir.join("vibeboy").join(filename)).ok()
        .or_else(|| std::fs::read(sys_dir.join(filename)).ok())
}

// ── libretro API implementation ─────────────────────────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn retro_api_version() -> c_uint {
    RETRO_API_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn retro_get_system_info(info: *mut RetroSystemInfo) {
    unsafe {
        (*info).library_name = LIB_NAME.as_ptr() as *const c_char;
        (*info).library_version = LIB_VERSION.as_ptr() as *const c_char;
        (*info).valid_extensions = EXTENSIONS.as_ptr() as *const c_char;
        (*info).need_fullpath = false;
        (*info).block_extract = false;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn retro_get_system_av_info(info: *mut RetroSystemAvInfo) {
    let (w, h) = unsafe {
        if let Some(core) = core_mut() {
            if core.emu.is_sgb() { (256u32, 224u32) } else { (160, 144) }
        } else {
            (160, 144)
        }
    };
    unsafe {
        (*info).geometry = RetroGameGeometry {
            base_width: w,
            base_height: h,
            max_width: 256,  // SGB max
            max_height: 224, // SGB max
            aspect_ratio: w as f32 / h as f32,
        };
        (*info).timing = RetroSystemTiming {
            fps: 59.7275,
            sample_rate: AUDIO_RATE,
        };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn retro_set_environment(cb: EnvironmentFn) {
    unsafe {
        std::ptr::addr_of_mut!(CB_ENVIRONMENT).write(Some(cb));

        // Set pixel format to XRGB8888 (must be set before retro_load_game)
        let mut fmt = RETRO_PIXEL_FORMAT_XRGB8888;
        let accepted = cb(RETRO_ENVIRONMENT_SET_PIXEL_FORMAT, &mut fmt as *mut _ as *mut c_void);
        if !accepted {
            log::warn!("RetroArch did not accept XRGB8888 pixel format");
        }

        // Declare core options
        let vars: [RetroVariable; 2] = [
            RetroVariable {
                key: VARS_MODEL_KEY.as_ptr() as *const c_char,
                value: VARS_MODEL_DESC.as_ptr() as *const c_char,
            },
            RetroVariable {
                key: ptr::null(),
                value: ptr::null(),
            },
        ];
        cb(RETRO_ENVIRONMENT_SET_VARIABLES, vars.as_ptr() as *mut c_void);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn retro_set_video_refresh(cb: VideoRefreshFn) {
    unsafe { std::ptr::addr_of_mut!(CB_VIDEO_REFRESH).write(Some(cb)); }
}

#[unsafe(no_mangle)]
pub extern "C" fn retro_set_audio_sample(cb: AudioSampleFn) {
    unsafe { std::ptr::addr_of_mut!(CB_AUDIO_SAMPLE).write(Some(cb)); }
}

#[unsafe(no_mangle)]
pub extern "C" fn retro_set_audio_sample_batch(cb: AudioSampleBatchFn) {
    unsafe { std::ptr::addr_of_mut!(CB_AUDIO_SAMPLE_BATCH).write(Some(cb)); }
}

#[unsafe(no_mangle)]
pub extern "C" fn retro_set_input_poll(cb: InputPollFn) {
    unsafe { std::ptr::addr_of_mut!(CB_INPUT_POLL).write(Some(cb)); }
}

#[unsafe(no_mangle)]
pub extern "C" fn retro_set_input_state(cb: InputStateFn) {
    unsafe { std::ptr::addr_of_mut!(CB_INPUT_STATE).write(Some(cb)); }
}

#[unsafe(no_mangle)]
pub extern "C" fn retro_init() {}

#[unsafe(no_mangle)]
pub extern "C" fn retro_deinit() {
    unsafe { std::ptr::addr_of_mut!(CORE).write(None); }
}

#[unsafe(no_mangle)]
pub extern "C" fn retro_load_game(game: *const RetroGameInfo) -> bool {
    unsafe {
        if game.is_null() || (*game).data.is_null() || (*game).size == 0 {
            return false;
        }

        let rom_data = std::slice::from_raw_parts((*game).data as *const u8, (*game).size);
        if rom_data.len() < 0x150 {
            return false;
        }

        let rom: std::sync::Arc<[u8]> = rom_data.to_vec().into();
        let model = get_model_from_options().unwrap_or_else(|| detect_model(&rom));
        let boot_rom = load_boot_rom(model);

        let emu = Emulator::new(rom.clone(), boot_rom, model, None, clock::default_clock());

        std::ptr::addr_of_mut!(CORE).write(Some(CoreState {
            emu,
            rom,
            audio_buf_i16: Vec::with_capacity(4096),
            video_buf: vec![0u32; 256 * 224], // max SGB size
            model,
            use_xrgb8888: true,
        }));

        true
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn retro_load_game_special(_type: c_uint, _info: *const RetroGameInfo, _num: usize) -> bool {
    false
}

#[unsafe(no_mangle)]
pub extern "C" fn retro_unload_game() {
    unsafe { std::ptr::addr_of_mut!(CORE).write(None); }
}

#[unsafe(no_mangle)]
pub extern "C" fn retro_run() {
    unsafe {
        let core = match core_mut() {
            Some(c) => c,
            None => return,
        };

        // Poll input
        if let Some(poll) = input_poll_cb() { poll(); }

        if let Some(input_state) = input_state_cb() {
            let btn = |id: c_uint| -> bool {
                input_state(0, RETRO_DEVICE_JOYPAD, 0, id) != 0
            };
            core.emu.set_button(Emulator::BTN_A,      btn(RETRO_DEVICE_ID_JOYPAD_A));
            core.emu.set_button(Emulator::BTN_B,      btn(RETRO_DEVICE_ID_JOYPAD_B));
            core.emu.set_button(Emulator::BTN_SELECT,  btn(RETRO_DEVICE_ID_JOYPAD_SELECT));
            core.emu.set_button(Emulator::BTN_START,   btn(RETRO_DEVICE_ID_JOYPAD_START));
            core.emu.set_button(Emulator::BTN_UP,      btn(RETRO_DEVICE_ID_JOYPAD_UP));
            core.emu.set_button(Emulator::BTN_DOWN,    btn(RETRO_DEVICE_ID_JOYPAD_DOWN));
            core.emu.set_button(Emulator::BTN_LEFT,    btn(RETRO_DEVICE_ID_JOYPAD_LEFT));
            core.emu.set_button(Emulator::BTN_RIGHT,   btn(RETRO_DEVICE_ID_JOYPAD_RIGHT));
        }

        // Step one frame
        core.emu.step_frame();

        // Video: send frame buffer
        if let Some(vcb) = video_cb() {
            let is_sgb = core.emu.is_sgb();
            let (w, h): (u32, u32) = if is_sgb { (256, 224) } else { (160, 144) };
            let fb = if is_sgb {
                core.emu.sgb_composited_frame()
            } else {
                core.emu.frame_buffer()
            };
            // Emulator outputs 0x00RRGGBB which matches libretro XRGB8888 directly.
            vcb(
                fb.as_ptr() as *const c_void,
                w, h,
                (w as usize) * std::mem::size_of::<u32>(),
            );
        }

        // Audio: drain samples, downsample from 96kHz to 48kHz, convert to i16
        let samples = core.emu.drain_audio_samples();
        if !samples.is_empty() {
            if let Some(batch_cb) = audio_batch_cb() {
                let downsampled = ui_util::downsample_audio(&samples, DOWNSAMPLE_FACTOR);

                core.audio_buf_i16.clear();
                for &s in &downsampled {
                    core.audio_buf_i16.push((s.clamp(-1.0, 1.0) * 32767.0) as i16);
                }

                let frames = core.audio_buf_i16.len() / 2;
                batch_cb(core.audio_buf_i16.as_ptr(), frames);
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn retro_reset() {
    unsafe {
        if let Some(core) = core_mut() {
            let model = get_model_from_options().unwrap_or_else(|| detect_model(&core.rom));
            let boot_rom = load_boot_rom(model);
            core.emu = Emulator::new(core.rom.clone(), boot_rom, model, None, clock::default_clock());
            core.model = model;
        }
    }
}

// ── Save states ─────────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn retro_serialize_size() -> usize {
    unsafe {
        if let Some(core) = core_mut() {
            // Estimate: serialize to get actual size. Cache if needed.
            let snap = core.emu.save_snapshot();
            crate::savestate::serialize(&snap).len()
        } else {
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn retro_serialize(data: *mut c_void, size: usize) -> bool {
    unsafe {
        let core = match core_mut() {
            Some(c) => c,
            None => return false,
        };
        let snap = core.emu.save_snapshot();
        let bytes = crate::savestate::serialize(&snap);
        if bytes.len() > size {
            return false;
        }
        ptr::copy_nonoverlapping(bytes.as_ptr(), data as *mut u8, bytes.len());
        true
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn retro_unserialize(data: *const c_void, size: usize) -> bool {
    unsafe {
        let core = match core_mut() {
            Some(c) => c,
            None => return false,
        };
        let bytes = std::slice::from_raw_parts(data as *const u8, size);
        match crate::savestate::deserialize(bytes) {
            Ok(snap) => {
                core.emu.restore_snapshot(&snap);
                true
            }
            Err(_) => false,
        }
    }
}

// ── Save RAM ────────────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn retro_get_memory_data(id: c_uint) -> *mut c_void {
    // libretro manages save RAM via retro_get_memory_data/size.
    // We can't return a pointer to internal cart RAM directly since it may
    // be resized. Return null and handle via serialize/unserialize instead.
    // RetroArch will fall back to save states for persistence.
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub extern "C" fn retro_get_memory_size(id: c_uint) -> usize {
    0
}

// ── Stubs for unused callbacks ──────────────────────────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn retro_set_controller_port_device(_port: c_uint, _device: c_uint) {}

#[unsafe(no_mangle)]
pub extern "C" fn retro_cheat_reset() {}

#[unsafe(no_mangle)]
pub extern "C" fn retro_cheat_set(_index: c_uint, _enabled: bool, _code: *const c_char) {}

#[unsafe(no_mangle)]
pub extern "C" fn retro_get_region() -> c_uint { 0 } // NTSC
