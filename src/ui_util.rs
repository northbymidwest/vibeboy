//! Frontend utility functions: filesystem I/O, UI helpers, gamepad polling.
//! These are used by native frontends and the test runner, NOT by the core emulator.
//!
//! For pure (no-I/O) utilities, see `crate::util`.

#[cfg(not(target_arch = "wasm32"))]
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
pub fn load_boot_rom(
    model: GbModel,
    bootrom_path: Option<&Path>,
    no_boot: bool,
) -> Option<Vec<u8>> {
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

impl Default for FpsCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl FpsCounter {
    pub fn new() -> Self {
        Self {
            timer: Instant::now(),
            count: 0,
            emu_total: Duration::ZERO,
        }
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
pub fn save_state_to_slot(emu: &mut crate::emulator::Emulator, rom_path: &Path, slot: usize) {
    emu.save_state(slot);
    if let Some(data) = emu.save_state_to_bytes(slot) {
        let path = rom_path.with_extension(format!("{}.ss", slot));
        match write_atomic(&path, &data) {
            Ok(_) => eprintln!("State saved to slot {} ({})", slot, path.display()),
            Err(e) => eprintln!("State saved to slot {} (disk write failed: {})", slot, e),
        }
    }
}

/// Load state from slot: tries in-memory first, then disk.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_state_from_slot(emu: &mut crate::emulator::Emulator, rom_path: &Path, slot: usize) {
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
        use crate::emulator::Emulator;
        use gilrs::{Axis as A, Button as B};

        // Drain events
        while let Some(ev) = self.gilrs.next_event() {
            match ev.event {
                gilrs::EventType::Connected => {
                    if self.active_gamepad.is_none() {
                        self.active_gamepad = Some(ev.id);
                        eprintln!("Gamepad connected: {}", self.gilrs.gamepad(ev.id).name());
                    }
                }
                gilrs::EventType::Disconnected if self.active_gamepad == Some(ev.id) => {
                    self.active_gamepad = None;
                    self.rumble_effect = None;
                    self.rumble_gamepad = None;
                    self.rumble_on = false;
                    eprintln!("Gamepad disconnected");
                }
                _ => {}
            }
        }

        let gp_id = match self.active_gamepad {
            Some(id) => id,
            None => {
                return GamepadState {
                    buttons: 0,
                    rewind: false,
                    fast_forward: false,
                };
            }
        };

        let gp = self.gilrs.gamepad(gp_id);
        const DEADZONE: f32 = 0.3;

        let lx = gp.axis_data(A::LeftStickX).map_or(0.0, |a| a.value());
        let ly = gp.axis_data(A::LeftStickY).map_or(0.0, |a| a.value());

        let mut bits: u8 = 0;
        let map: &[(B, u8)] = &[
            (B::East, Emulator::BTN_A),
            (B::South, Emulator::BTN_B),
            (B::Start, Emulator::BTN_START),
            (B::Select, Emulator::BTN_SELECT),
            (B::DPadUp, Emulator::BTN_UP),
            (B::DPadDown, Emulator::BTN_DOWN),
            (B::DPadLeft, Emulator::BTN_LEFT),
            (B::DPadRight, Emulator::BTN_RIGHT),
        ];
        for &(gb, btn) in map {
            if gp.is_pressed(gb) {
                bits |= btn;
            }
        }
        if lx < -DEADZONE {
            bits |= Emulator::BTN_LEFT;
        }
        if lx > DEADZONE {
            bits |= Emulator::BTN_RIGHT;
        }
        if ly < -DEADZONE {
            bits |= Emulator::BTN_DOWN;
        }
        if ly > DEADZONE {
            bits |= Emulator::BTN_UP;
        }

        GamepadState {
            buttons: bits,
            rewind: gp.is_pressed(B::LeftTrigger),
            fast_forward: gp.is_pressed(B::RightTrigger),
        }
    }

    /// Set up a rumble effect for the active gamepad (if force-feedback is supported).
    /// Call once when a rumble-capable cart is loaded.
    pub fn ensure_rumble(&mut self) {
        use gilrs::ff::{BaseEffect, BaseEffectType, EffectBuilder, Repeat, Replay, Ticks};

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
                if !p.exists() {
                    break p;
                }
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

/// Write `data` to `path` so that a crash never leaves a truncated file: the
/// bytes go to a temporary file in the same directory, are synced to disk,
/// and the temporary file is then renamed over the destination.
#[cfg(not(target_arch = "wasm32"))]
pub fn write_atomic(path: &Path, data: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut tmp_name = path.file_name().unwrap_or_default().to_os_string();
    tmp_name.push(".tmp");
    let tmp_path = path.with_file_name(tmp_name);
    let result = (|| {
        let mut file = std::fs::File::create(&tmp_path)?;
        file.write_all(data)?;
        file.sync_all()?;
        std::fs::rename(&tmp_path, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    result
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
    if let Err(e) = write_atomic(&sav_path, &data) {
        log::error!("Failed to write save file '{}': {}", sav_path.display(), e);
    }
}

/// Tracks battery save state for periodic flushing.
///
/// Dirtiness comes from `Emulator::save_generation()`, which only changes when
/// the game writes the cartridge RAM window or a snapshot is restored. That
/// keeps RTC carts, whose `save_data()` embeds a wall-clock timestamp, from
/// being rewritten every second.
///
/// A dirty save is written once the generation has been stable for
/// `SETTLE`, so a multi-frame save routine is not captured half-way. Games
/// that keep writing cart RAM (some use it as work RAM) are still flushed
/// every `MAX_DELAY`.
#[cfg(not(target_arch = "wasm32"))]
pub struct SavFlusher {
    rom_path: std::path::PathBuf,
    flushed_generation: u64,
    seen_generation: u64,
    /// When the save first became dirty and when the generation last changed.
    dirty: Option<(Instant, Instant)>,
}

#[cfg(not(target_arch = "wasm32"))]
impl SavFlusher {
    const SETTLE: Duration = Duration::from_secs(1);
    const MAX_DELAY: Duration = Duration::from_secs(10);

    /// Create a new flusher for the given ROM path. Call this right after
    /// `load_sav`, so the freshly loaded state counts as already flushed.
    pub fn new(emu: &crate::emulator::Emulator, rom_path: &Path) -> Self {
        let generation = emu.save_generation();
        Self {
            rom_path: rom_path.to_path_buf(),
            flushed_generation: generation,
            seen_generation: generation,
            dirty: None,
        }
    }

    /// Flush the save if it is dirty and has settled. Call this every frame.
    pub fn poll(&mut self, emu: &crate::emulator::Emulator) {
        if !emu.has_battery() {
            return;
        }
        let generation = emu.save_generation();
        if generation == self.flushed_generation {
            self.dirty = None;
            return;
        }
        let now = Instant::now();
        let (first_dirty, last_change) = self.dirty.get_or_insert((now, now));
        if generation != self.seen_generation {
            self.seen_generation = generation;
            *last_change = now;
        }
        if now.duration_since(*last_change) >= Self::SETTLE
            || now.duration_since(*first_dirty) >= Self::MAX_DELAY
        {
            self.flush(emu);
        }
    }

    /// Write the save unconditionally. Use on quit, and before replacing the
    /// emulator (ROM switch, reset, model change).
    pub fn flush(&mut self, emu: &crate::emulator::Emulator) {
        flush_sav(emu, &self.rom_path);
        self.flushed_generation = emu.save_generation();
        self.seen_generation = self.flushed_generation;
        self.dirty = None;
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn write_atomic_replaces_file_and_leaves_no_temp() {
        let dir = std::env::temp_dir().join(format!("vibeboy_atomic_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("game.sav");
        std::fs::write(&path, b"old contents that are longer").unwrap();
        write_atomic(&path, b"new").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        assert!(!dir.join("game.sav.tmp").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
