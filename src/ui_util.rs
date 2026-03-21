//! Shared utility functions used by multiple UI frontends (main.rs, cocoa_ui.rs, winit_ui.rs)
//! and the test runner.

use crate::model::GbModel;
#[cfg(not(target_arch = "wasm32"))]
use crate::scaling;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::time::{Duration, Instant};

/// Target frame time: 70224 T-cycles / cpu_clock_rate.
/// Standard: ~16.74ms (~59.73 fps). SGB1: ~16.35ms (~61.17 fps).
pub fn frame_duration(model: GbModel) -> Duration {
    let nanos = 70_224u64 * 1_000_000_000 / model.cpu_clock_rate() as u64;
    Duration::from_nanos(nanos)
}

/// Parse a model string for clap value_parser.
pub fn parse_model(s: &str) -> Result<GbModel, String> {
    s.parse::<GbModel>()
}

/// Parse a filter string for clap value_parser.
#[cfg(not(target_arch = "wasm32"))]
pub fn parse_filter(s: &str) -> Result<String, String> {
    scaling::ScaleFilter::validate_name(s)
}

/// Auto-detect hardware model from ROM header CGB flag.
pub fn auto_detect_model(rom: &[u8]) -> GbModel {
    let cgb_flag = rom.get(0x0143).copied().unwrap_or(0);
    if cgb_flag == 0x80 || cgb_flag == 0xC0 {
        GbModel::Cgb
    } else {
        GbModel::Dmg
    }
}

// ── Boot ROM loading ──────────────────────────────────────────────────────

/// Default boot ROM path for each hardware model.
#[cfg(not(target_arch = "wasm32"))]
pub fn boot_rom_path(model: GbModel) -> &'static str {
    match model {
        GbModel::Dmg0 => "bootroms/dmg0_boot.bin",
        GbModel::Dmg => "bootroms/dmg_boot.bin",
        GbModel::Mgb => "bootroms/mgb_boot.bin",
        GbModel::Sgb => "bootroms/sgb_boot.bin",
        GbModel::Sgb2 => "bootroms/sgb2_boot.bin",
        GbModel::Cgb0 => "bootroms/cgb0_boot.bin",
        GbModel::Cgb => "bootroms/cgb_boot.bin",
        GbModel::Agb => "bootroms/cgb_agb_boot.bin",
    }
}

/// Load a boot ROM: explicit path > auto-detect by model > None.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_boot_rom(model: GbModel, bootrom_path: Option<&Path>, no_boot: bool) -> Option<Vec<u8>> {
    if no_boot {
        return None;
    }
    if let Some(p) = bootrom_path {
        return std::fs::read(p).ok();
    }
    std::fs::read(boot_rom_path(model)).ok()
}

// ── Controls printout ─────────────────────────────────────────────────────

/// Print the standard controls help to stderr.
#[cfg(not(target_arch = "wasm32"))]
pub fn print_controls() {
    eprintln!("\nControls:");
    eprintln!("  Arrow keys  — D-pad         Gamepad D-pad / Left stick");
    eprintln!("  Z / X       — B / A         Gamepad South / East");
    eprintln!("  Enter       — Start         Gamepad Start");
    eprintln!("  Right Shift — Select        Gamepad Back");
    eprintln!("  Backspace   — Rewind        Gamepad L Shoulder");
    eprintln!("  Tab         — Fast fwd (4x) Gamepad R Shoulder");
    eprintln!("  Minus       — Slow motion   (hold for half speed)");
    eprintln!("  Space       — Pause         (toggle)");
    eprintln!("  Period      — Frame advance  (step one frame while paused)");
    eprintln!("  F5 / F7     — Save / Load state");
    eprintln!("  F9          — Screenshot (raw + scaled)");
    eprintln!("  1-9         — Select state slot");
    eprintln!("  Escape      — Quit");
}

// ── FPS counter ───────────────────────────────────────────────────────────

/// Tracks frame rate and average emulation time, printing once per second.
pub struct FpsCounter {
    timer: Instant,
    count: u32,
    emu_total: Duration,
}

impl FpsCounter {
    pub fn new() -> Self {
        Self { timer: Instant::now(), count: 0, emu_total: Duration::ZERO }
    }

    /// Record frames and emulation time. Prints and resets every second.
    /// Returns `Some((fps, avg_emu_ms))` on print, `None` otherwise.
    pub fn update(&mut self, frames_stepped: u32, emu_elapsed: Duration) -> Option<(f64, f64)> {
        self.count += frames_stepped;
        if frames_stepped > 0 {
            self.emu_total += emu_elapsed;
        }
        let elapsed = self.timer.elapsed();
        if elapsed >= Duration::from_secs(1) && self.count > 0 {
            let fps = self.count as f64 / elapsed.as_secs_f64();
            let avg_ms = self.emu_total.as_secs_f64() * 1000.0 / self.count as f64;
            eprintln!("FPS: {:.1}  emu: {:.2}ms/frame", fps, avg_ms);
            self.count = 0;
            self.emu_total = Duration::ZERO;
            self.timer = Instant::now();
            Some((fps, avg_ms))
        } else {
            None
        }
    }
}

// ── Save state management ─────────────────────────────────────────────────

/// Save/load emulator state to/from numbered `.ss` files on disk.
#[cfg(not(target_arch = "wasm32"))]
pub fn save_state_to_slot(
    emu: &mut crate::emulator::Emulator,
    rom_path: &Path,
    slot: usize,
) {
    emu.save_state(slot);
    if let Some(data) = emu.save_state_to_bytes(slot) {
        let path = rom_path.with_extension(format!("{}.ss", slot + 1));
        match std::fs::write(&path, &data) {
            Ok(_) => eprintln!("State saved to slot {} ({})", slot + 1, path.display()),
            Err(e) => eprintln!("State saved to slot {} (disk write failed: {})", slot + 1, e),
        }
    }
}

/// Load state from slot: tries in-memory first, then disk.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_state_from_slot(
    emu: &mut crate::emulator::Emulator,
    rom_path: &Path,
    slot: usize,
) {
    if emu.load_state(slot) {
        eprintln!("State loaded from slot {}", slot + 1);
    } else {
        let path = rom_path.with_extension(format!("{}.ss", slot + 1));
        if let Ok(data) = std::fs::read(&path) {
            if emu.load_state_from_bytes(slot, &data) {
                eprintln!("State loaded from disk: {}", path.display());
            } else {
                eprintln!("Failed to load state from {}", path.display());
            }
        } else {
            eprintln!("Slot {} is empty", slot + 1);
        }
    }
}

// ── Gamepad polling ───────────────────────────────────────────────────────

/// Result of polling the gamepad each frame.
#[cfg(feature = "gilrs")]
pub struct GamepadState {
    /// Bitmask of Game Boy buttons currently held on the gamepad.
    pub buttons: u8,
    /// Whether L shoulder is pressed (rewind).
    pub rewind: bool,
    /// Whether R shoulder is pressed (fast-forward).
    pub fast_forward: bool,
}

/// Polls a gilrs gamepad, handling connect/disconnect and button/stick mapping.
#[cfg(feature = "gilrs")]
pub struct GamepadPoller {
    pub gilrs: gilrs::Gilrs,
    pub active_gamepad: Option<gilrs::GamepadId>,
}

#[cfg(feature = "gilrs")]
impl GamepadPoller {
    pub fn new() -> Option<Self> {
        gilrs::Gilrs::new().ok().map(|g| Self { gilrs: g, active_gamepad: None })
    }

    /// Drain events and read current state. Returns the gamepad state.
    pub fn poll(&mut self) -> GamepadState {
        use gilrs::{Button as B, Axis as A};
        use crate::emulator::Emulator;

        // Drain events
        while let Some(ev) = self.gilrs.next_event() {
            match ev.event {
                gilrs::EventType::Connected => {
                    if self.active_gamepad.is_none() {
                        self.active_gamepad = Some(ev.id);
                        eprintln!("Gamepad connected: {}", self.gilrs.gamepad(ev.id).name());
                    }
                }
                gilrs::EventType::Disconnected => {
                    if self.active_gamepad == Some(ev.id) {
                        self.active_gamepad = None;
                        eprintln!("Gamepad disconnected");
                    }
                }
                _ => {}
            }
        }

        let gp_id = match self.active_gamepad {
            Some(id) => id,
            None => return GamepadState { buttons: 0, rewind: false, fast_forward: false },
        };

        let gp = self.gilrs.gamepad(gp_id);
        const DEADZONE: f32 = 0.3;

        let lx = gp.axis_data(A::LeftStickX).map_or(0.0, |a| a.value());
        let ly = gp.axis_data(A::LeftStickY).map_or(0.0, |a| a.value());

        let mut bits: u8 = 0;
        let map: &[(B, u8)] = &[
            (B::East,      Emulator::BTN_A),
            (B::South,     Emulator::BTN_B),
            (B::Start,     Emulator::BTN_START),
            (B::Select,    Emulator::BTN_SELECT),
            (B::DPadUp,    Emulator::BTN_UP),
            (B::DPadDown,  Emulator::BTN_DOWN),
            (B::DPadLeft,  Emulator::BTN_LEFT),
            (B::DPadRight, Emulator::BTN_RIGHT),
        ];
        for &(gb, btn) in map {
            if gp.is_pressed(gb) { bits |= btn; }
        }
        if lx < -DEADZONE { bits |= Emulator::BTN_LEFT; }
        if lx > DEADZONE  { bits |= Emulator::BTN_RIGHT; }
        if ly < -DEADZONE { bits |= Emulator::BTN_DOWN; }
        if ly > DEADZONE  { bits |= Emulator::BTN_UP; }

        GamepadState {
            buttons: bits,
            rewind: gp.is_pressed(B::LeftTrigger),
            fast_forward: gp.is_pressed(B::RightTrigger),
        }
    }
}

// ── Audio ─────────────────────────────────────────────────────────────────

/// Reverse stereo interleaved audio in-place.
/// Swaps stereo sample pairs so the audio plays backward.
pub fn reverse_audio(samples: &mut [f32]) {
    let n = samples.len() / 2;
    for i in 0..n / 2 {
        let j = n - 1 - i;
        samples.swap(i * 2, j * 2);
        samples.swap(i * 2 + 1, j * 2 + 1);
    }
}

/// Apply a short cosine fade-in to the start of each frame's audio to
/// suppress transient pops from cold APU filter state on snapshot restore.
/// `frame_len` is the number of stereo sample pairs per frame.
pub fn fade_frame_boundaries(samples: &mut [f32], frame_len: usize) {
    if frame_len == 0 { return; }
    // Fade the first ~64 stereo pairs of each frame
    let fade_len = 64.min(frame_len);
    let mut offset = 0;
    while offset + frame_len * 2 <= samples.len() {
        for i in 0..fade_len {
            let t = i as f32 / fade_len as f32;
            // Cosine ease-in: 0 at start, 1 at fade_len
            let gain = 0.5 - 0.5 * (std::f32::consts::PI * t).cos();
            samples[offset + i * 2] *= gain;
            samples[offset + i * 2 + 1] *= gain;
        }
        offset += frame_len * 2;
    }
}

/// Downsample stereo interleaved audio by an integer factor using a
/// Blackman-windowed sinc FIR low-pass filter followed by decimation.
///
/// The filter removes frequencies above the new Nyquist (original_rate / 2*factor)
/// before decimation, preventing aliasing artifacts.
///
/// Input/output: interleaved [L, R, L, R, ...].
pub fn downsample_audio(samples: &[f32], factor: usize) -> Vec<f32> {
    if factor <= 1 || samples.len() < 2 {
        return samples.to_vec();
    }

    // Precompute FIR kernel: Blackman-windowed sinc, 33 taps
    const HALF_TAPS: i32 = 16;
    const TAPS: usize = (HALF_TAPS * 2 + 1) as usize;
    let cutoff = 1.0 / (2.0 * factor as f64);
    let mut kernel = [0.0f32; TAPS];
    let mut sum = 0.0f64;
    for i in 0..TAPS {
        let n = i as f64 - HALF_TAPS as f64;
        let sinc = if n.abs() < 1e-10 {
            2.0 * std::f64::consts::PI * cutoff
        } else {
            (2.0 * std::f64::consts::PI * cutoff * n).sin() / n
        };
        let t = i as f64 / (TAPS - 1) as f64;
        let window = 0.42 - 0.5 * (2.0 * std::f64::consts::PI * t).cos()
                          + 0.08 * (4.0 * std::f64::consts::PI * t).cos();
        let v = sinc * window;
        kernel[i] = v as f32;
        sum += v;
    }
    let inv = 1.0 / sum as f32;
    for k in &mut kernel { *k *= inv; }

    let stereo_frames = samples.len() / 2;
    let out_frames = stereo_frames / factor;
    let mut out = Vec::with_capacity(out_frames * 2);

    for i in 0..out_frames {
        let center = i * factor;
        let mut l = 0.0f32;
        let mut r = 0.0f32;
        for (j, &k) in kernel.iter().enumerate() {
            let src = (center as i32 + j as i32 - HALF_TAPS)
                .clamp(0, stereo_frames as i32 - 1) as usize;
            l += samples[src * 2] * k;
            r += samples[src * 2 + 1] * k;
        }
        out.push(l);
        out.push(r);
    }

    out
}

/// Load battery-backed save RAM from a `.sav` file next to the ROM.
/// Call this after creating the emulator, before the first frame.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_sav(emu: &mut crate::emulator::Emulator, rom_path: &Path) {
    if !emu.has_battery() {
        return;
    }
    let sav_path = rom_path.with_extension("sav");
    if let Ok(data) = std::fs::read(&sav_path) {
        log::info!("Loaded save from {}", sav_path.display());
        emu.load_ram(&data);
    }
}

/// Write battery-backed save RAM to a `.sav` file next to the ROM unconditionally.
/// Use this for final on-quit flush.
#[cfg(not(target_arch = "wasm32"))]
pub fn flush_sav(emu: &crate::emulator::Emulator, rom_path: &Path) {
    if !emu.has_battery() {
        return;
    }
    let data = emu.save_data();
    if data.is_empty() {
        return;
    }
    let sav_path = rom_path.with_extension("sav");
    if let Err(e) = std::fs::write(&sav_path, &data) {
        log::error!("Failed to write save file '{}': {}", sav_path.display(), e);
    }
}

/// Tracks save RAM state for periodic flushing.
/// Only writes to disk when RAM has changed since the last flush and has been
/// stable (unchanged) for at least 1 second, avoiding writes during active
/// save operations by the game.
#[cfg(not(target_arch = "wasm32"))]
pub struct SavFlusher {
    rom_path: std::path::PathBuf,
    last_flushed: Vec<u8>,
    dirty_since: Option<std::time::Instant>,
}

#[cfg(not(target_arch = "wasm32"))]
impl SavFlusher {
    /// Create a new flusher for the given ROM path.
    /// Initializes `last_flushed` to the current save data so we don't
    /// immediately write on startup.
    pub fn new(emu: &crate::emulator::Emulator, rom_path: &Path) -> Self {
        Self {
            rom_path: rom_path.to_path_buf(),
            last_flushed: if emu.has_battery() { emu.save_data() } else { Vec::new() },
            dirty_since: None,
        }
    }

    /// Check if save RAM has changed and flush if stable for >= 1 second.
    /// Call this every frame or every few frames.
    pub fn poll(&mut self, emu: &crate::emulator::Emulator) {
        if !emu.has_battery() {
            return;
        }
        let data = emu.save_data();
        if data == self.last_flushed {
            // RAM matches last flush — not dirty
            self.dirty_since = None;
            return;
        }
        // RAM has changed
        let now = std::time::Instant::now();
        match self.dirty_since {
            None => {
                // Just became dirty — start the stability timer
                self.dirty_since = Some(now);
            }
            Some(since) if now.duration_since(since) >= Duration::from_secs(1) => {
                // Dirty and stable for >= 1 second — flush
                let sav_path = self.rom_path.with_extension("sav");
                if let Err(e) = std::fs::write(&sav_path, &data) {
                    log::error!("Failed to write save file '{}': {}", sav_path.display(), e);
                }
                self.last_flushed = data;
                self.dirty_since = None;
            }
            _ => {
                // Dirty but not yet stable — wait
            }
        }
    }

    /// Force flush on quit (unconditional if dirty).
    pub fn flush(&mut self, emu: &crate::emulator::Emulator) {
        flush_sav(emu, &self.rom_path);
    }
}
