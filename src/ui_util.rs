//! Frontend utility functions: filesystem I/O, UI helpers, gamepad polling.
//! These are used by native frontends and the test runner, NOT by the core emulator.
//!
//! For pure (no-I/O) utilities, see `crate::util`.

// Re-export pure utility functions for backwards compatibility.
// New code should use `vibeboy::util::*` directly.
pub use crate::util::{
    frame_duration, parse_model, auto_detect_model,
    reverse_audio, fade_frame_boundaries, downsample_audio,
};

use crate::model::GbModel;
#[cfg(not(target_arch = "wasm32"))]
use crate::scaling;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::time::{Duration, Instant};

/// Parse a filter string for clap value_parser.
#[cfg(not(target_arch = "wasm32"))]
pub fn parse_filter(s: &str) -> Result<String, String> {
    scaling::ScaleFilter::validate_name(s)
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

/// Load a boot ROM: explicit path > file on disk > built-in > None.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_boot_rom(model: GbModel, bootrom_path: Option<&Path>, no_boot: bool) -> Option<Vec<u8>> {
    if no_boot {
        return None;
    }
    if let Some(p) = bootrom_path {
        return std::fs::read(p).ok();
    }
    // Try external file first (official/custom boot ROMs)
    if let Ok(data) = std::fs::read(boot_rom_path(model)) {
        return Some(data);
    }
    // Fall back to built-in boot ROM
    crate::bootrom::builtin(model).map(|b| b.to_vec())
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
    eprintln!("  0-9         — Select state slot");
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
        let path = rom_path.with_extension(format!("{}.ss", slot));
        match std::fs::write(&path, &data) {
            Ok(_) => eprintln!("State saved to slot {} ({})", slot, path.display()),
            Err(e) => eprintln!("State saved to slot {} (disk write failed: {})", slot, e),
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
        eprintln!("State loaded from slot {}", slot);
    } else {
        let path = rom_path.with_extension(format!("{}.ss", slot));
        if let Ok(data) = std::fs::read(&path) {
            if emu.load_state_from_bytes(slot, &data) {
                eprintln!("State loaded from disk: {}", path.display());
            } else {
                eprintln!("Failed to load state from {}", path.display());
            }
        } else {
            eprintln!("Slot {} is empty", slot);
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

/// Polls a gilrs gamepad, handling connect/disconnect, button/stick mapping,
/// and rumble force-feedback.
#[cfg(feature = "gilrs")]
pub struct GamepadPoller {
    pub gilrs: gilrs::Gilrs,
    pub active_gamepad: Option<gilrs::GamepadId>,
    rumble_effect: Option<gilrs::ff::Effect>,
    rumble_gamepad: Option<gilrs::GamepadId>,
    rumble_on: bool,
}

#[cfg(feature = "gilrs")]
impl GamepadPoller {
    pub fn new() -> Option<Self> {
        gilrs::Gilrs::new().ok().map(|g| Self {
            gilrs: g,
            active_gamepad: None,
            rumble_effect: None,
            rumble_gamepad: None,
            rumble_on: false,
        })
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
                        self.rumble_effect = None;
                        self.rumble_gamepad = None;
                        self.rumble_on = false;
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

    /// Set up a rumble effect for the active gamepad (if force-feedback is supported).
    /// Call once when a rumble-capable cart is loaded.
    pub fn ensure_rumble(&mut self) {
        use gilrs::ff::{EffectBuilder, BaseEffect, BaseEffectType, Replay, Repeat, Ticks};

        let gp_id = match self.active_gamepad {
            Some(id) => id,
            None => return,
        };
        if self.rumble_gamepad == Some(gp_id) && self.rumble_effect.is_some() {
            return;
        }
        // Drop old effect
        self.rumble_effect = None;
        self.rumble_gamepad = None;
        self.rumble_on = false;

        if !self.gilrs.gamepad(gp_id).is_ff_supported() {
            return;
        }

        let effect = EffectBuilder::new()
            .add_effect(BaseEffect {
                kind: BaseEffectType::Strong { magnitude: 40_000 },
                scheduling: Replay {
                    play_for: Ticks::from_ms(u32::MAX),
                    ..Default::default()
                },
                envelope: Default::default(),
            })
            .repeat(Repeat::Infinitely)
            .gamepads(&[gp_id])
            .finish(&mut self.gilrs);

        match effect {
            Ok(e) => {
                self.rumble_effect = Some(e);
                self.rumble_gamepad = Some(gp_id);
            }
            Err(e) => {
                log::warn!("Failed to create rumble effect: {}", e);
            }
        }
    }

    /// Start or stop rumble. No-op if unchanged or no effect is loaded.
    pub fn set_rumble(&mut self, on: bool) {
        if on == self.rumble_on {
            return;
        }
        self.rumble_on = on;
        if let Some(ref effect) = self.rumble_effect {
            if on {
                let _ = effect.play();
            } else {
                let _ = effect.stop();
            }
        }
    }
}

/// Poll the printer for completed prints and save them as PNG files to `prints/`.
/// Call this after stepping the emulator each frame.
#[cfg(all(not(target_arch = "wasm32"), feature = "image"))]
pub fn check_and_save_prints(emu: &mut crate::emulator::Emulator) {
    use crate::printer::Printer;
    if let Some(printer) = emu.serial_device_as_any_mut().downcast_mut::<Printer>() {
        while let Some((rgba, w, h)) = printer.take_print() {
            let dir = Path::new("prints");
            if !dir.exists() {
                let _ = std::fs::create_dir_all(dir);
            }
            // Find next available filename
            let mut idx = 0u32;
            let path = loop {
                let p = dir.join(format!("print_{:04}.png", idx));
                if !p.exists() { break p; }
                idx += 1;
            };
            match image::save_buffer(&path, &rgba, w, h, image::ColorType::Rgba8) {
                Ok(_) => eprintln!("Printer: saved {}", path.display()),
                Err(e) => eprintln!("Printer: failed to save {}: {}", path.display(), e),
            }
        }
    }
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
