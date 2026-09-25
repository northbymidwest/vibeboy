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

// ── Emulation session ─────────────────────────────────────────────────────

/// Rewind plays back this many frames per tick.
#[cfg(not(target_arch = "wasm32"))]
const REWIND_SPEED: usize = 3;
/// Fast-forward emulates this many frames per tick.
#[cfg(not(target_arch = "wasm32"))]
const FAST_FORWARD_SPEED: usize = 4;
/// Audio-driven pacing keeps about this many video frames' worth of audio
/// queued ahead of the device (~50ms): below half of it the session steps
/// two frames per tick to catch up.
#[cfg(not(target_arch = "wasm32"))]
const AUDIO_TARGET_FRAMES: usize = 3;
/// Above this many video frames' worth of queued audio (~133ms) the session
/// skips stepping so the queue can drain.
#[cfg(not(target_arch = "wasm32"))]
const AUDIO_MAX_FRAMES: usize = 8;
/// Capacity of an audio ring, in video frames' worth of audio (~267ms).
/// It must stay well above `AUDIO_MAX_FRAMES` so the skip threshold is
/// reachable, with room for a fast-forward or catch-up tick on top.
#[cfg(not(target_arch = "wasm32"))]
pub const AUDIO_RING_FRAMES: usize = 16;

/// Stereo frames of audio per video frame at `sample_rate`.
#[cfg(not(target_arch = "wasm32"))]
fn audio_frames_per_video_frame(sample_rate: u32) -> usize {
    sample_rate as usize / 60
}

/// Capacity, in stereo frames, for an audio ring fed at `sample_rate`.
#[cfg(not(target_arch = "wasm32"))]
pub fn audio_ring_capacity(sample_rate: u32) -> usize {
    audio_frames_per_video_frame(sample_rate) * AUDIO_RING_FRAMES
}

/// How many frames to emulate this tick, given `queued` stereo frames of
/// audio (at `sample_rate`) waiting in the output device's queue. Stepping
/// by the queue fill slaves emulation speed to the audio device's clock,
/// which avoids both underruns (crackling) and overruns (latency).
#[cfg(not(target_arch = "wasm32"))]
fn frames_for_audio_queue(queued: usize, sample_rate: u32) -> u32 {
    let per_frame = audio_frames_per_video_frame(sample_rate);
    if queued < per_frame * AUDIO_TARGET_FRAMES / 2 {
        2
    } else if queued > per_frame * AUDIO_MAX_FRAMES {
        0
    } else {
        1
    }
}

/// Options that shape every emulator a [`Session`] builds.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug)]
pub struct SessionConfig {
    /// Hardware model to force; `None` auto-detects it from each ROM header.
    pub model: Option<GbModel>,
    /// Explicit boot ROM file. It was chosen for the startup model, so it is
    /// dropped once the model is changed with [`Session::set_model`].
    pub bootrom: Option<std::path::PathBuf>,
    /// Skip the boot ROM and start at PC=0x100 with post-boot state.
    pub no_boot: bool,
    /// Run SGB games with the SNES CPU (LLE) when a program ROM is found.
    pub lle: bool,
    /// Explicit SNES program ROM for LLE (auto-detected if `None`).
    pub snes_rom: Option<std::path::PathBuf>,
    /// Attach a Game Boy Printer to the emulator.
    pub printer: bool,
    /// Run-ahead frames for the displayed frame (0 disables run-ahead).
    pub runahead: u32,
    /// APU output rate in Hz.
    pub sample_rate: u32,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            model: None,
            bootrom: None,
            no_boot: false,
            lle: false,
            snes_rom: None,
            printer: false,
            runahead: 0,
            sample_rate: 96_000,
        }
    }
}

/// Hotkeys that are held down rather than pressed, sampled every tick.
#[derive(Clone, Copy, Debug, Default)]
pub struct HoldInputs {
    pub rewind: bool,
    pub fast_forward: bool,
    pub slow_motion: bool,
}

/// What one [`Session::tick`] produced.
#[derive(Debug, Default)]
pub struct TickOutput {
    /// Interleaved stereo samples at the session's sample rate, ready to
    /// queue as is: rewind audio is already reversed and both rewind and
    /// fast-forward audio are already downsampled to real-time length.
    pub audio: Vec<f32>,
    /// Frames emulated forward.
    pub frames_stepped: u32,
    /// Frames played back from the rewind buffer.
    pub frames_rewound: u32,
    /// Time spent emulating.
    pub emu_time: Duration,
}

impl TickOutput {
    /// Frames emulated in either direction, for frame rate counters.
    pub fn frames_emulated(&self) -> u32 {
        self.frames_stepped + self.frames_rewound
    }
}

/// A loaded game and the per-tick emulation logic shared by the native
/// frontends: pacing, rewind, fast-forward, slow motion, pause and frame
/// advance, save states, the printer and battery saves. Frontends own the
/// window, input devices and audio device, and feed this each tick.
#[cfg(not(target_arch = "wasm32"))]
pub struct Session {
    pub emu: crate::emulator::Emulator,
    rom_path: std::path::PathBuf,
    rom: std::sync::Arc<[u8]>,
    model: GbModel,
    config: SessionConfig,
    /// Contents of `config.snes_rom`, read once up front.
    snes_rom: Option<Vec<u8>>,
    sav_flusher: SavFlusher,
    slot: usize,
    paused: bool,
    step_one_frame: bool,
    /// Wall time owed to emulation, for slow motion and for pacing without
    /// an audio device.
    time_debt: Duration,
    last_tick: Option<Instant>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Session {
    /// Load the ROM at `rom_path` and build an emulator for it. Fails if the
    /// ROM, or an explicitly given SNES program ROM, cannot be read.
    pub fn new(rom_path: &Path, config: SessionConfig) -> std::io::Result<Self> {
        let rom: std::sync::Arc<[u8]> = std::fs::read(rom_path)?.into();
        let snes_rom = match (&config.snes_rom, config.lle) {
            (Some(p), true) => Some(std::fs::read(p).map_err(|e| {
                std::io::Error::new(e.kind(), format!("SNES ROM '{}': {}", p.display(), e))
            })?),
            _ => None,
        };
        let model = config
            .model
            .unwrap_or_else(|| crate::util::auto_detect_model(&rom));
        let emu = build_emulator(&rom, model, &config, snes_rom.as_deref());
        let emu = prepare_emulator(emu, rom_path, &config);
        let sav_flusher = SavFlusher::new(&emu, rom_path);
        if config.printer {
            eprintln!("Game Boy Printer connected, images will be saved to prints/");
        }
        Ok(Self {
            emu,
            rom_path: rom_path.to_path_buf(),
            rom,
            model,
            config,
            snes_rom,
            sav_flusher,
            slot: 0,
            paused: false,
            step_one_frame: false,
            time_debt: Duration::ZERO,
            last_tick: None,
        })
    }

    pub fn rom_path(&self) -> &Path {
        &self.rom_path
    }

    /// The model the current emulator runs as.
    pub fn model(&self) -> GbModel {
        self.model
    }

    /// The forced model, or `None` when auto-detecting.
    pub fn forced_model(&self) -> Option<GbModel> {
        self.config.model
    }

    /// Duration of one emulated frame for the current model.
    pub fn frame_duration(&self) -> Duration {
        crate::util::frame_duration(self.model)
    }

    /// Switch to the ROM at `path`. The outgoing game's battery save is
    /// written first. On a read error nothing changes.
    pub fn load_rom(&mut self, path: &Path) -> std::io::Result<()> {
        let rom = std::fs::read(path)?;
        self.rom = rom.into();
        self.rom_path = path.to_path_buf();
        self.model = self
            .config
            .model
            .unwrap_or_else(|| crate::util::auto_detect_model(&self.rom));
        self.rebuild();
        eprintln!("Loaded: {}", self.rom_path.display());
        Ok(())
    }

    /// Power-cycle the current game, writing its battery save first.
    pub fn reset(&mut self) {
        self.rebuild();
        eprintln!("Reset");
    }

    /// Force a hardware model (`None` = auto-detect) and power-cycle. An
    /// explicit boot ROM from the command line is dropped, since it was
    /// chosen for the startup model.
    pub fn set_model(&mut self, model: Option<GbModel>) {
        self.config.model = model;
        self.config.bootrom = None;
        self.model = model.unwrap_or_else(|| crate::util::auto_detect_model(&self.rom));
        self.rebuild();
        match model {
            Some(m) => eprintln!("Hardware model: {}", m),
            None => eprintln!("Hardware model: Auto ({})", self.model),
        }
    }

    /// Replace the emulator with a freshly built one for the current ROM and
    /// model, keeping the attached peripherals.
    fn rebuild(&mut self) {
        // Persist the outgoing battery save before replacing the emulator.
        self.sav_flusher.flush(&self.emu);
        let emu = build_emulator(
            &self.rom,
            self.model,
            &self.config,
            self.snes_rom.as_deref(),
        );
        self.emu = prepare_emulator(emu, &self.rom_path, &self.config);
        self.sav_flusher = SavFlusher::new(&self.emu, &self.rom_path);
        self.paused = false;
        self.step_one_frame = false;
        self.time_debt = Duration::ZERO;
    }

    pub fn printer_attached(&self) -> bool {
        self.config.printer
    }

    /// Connect or disconnect the Game Boy Printer. The choice sticks across
    /// ROM switches, resets and model changes.
    pub fn set_printer(&mut self, on: bool) {
        self.config.printer = on;
        if on {
            attach_printer(&mut self.emu, self.model);
            eprintln!("Game Boy Printer connected");
        } else {
            self.emu
                .attach_serial_device(Box::new(crate::serial::Disconnected));
            eprintln!("Game Boy Printer disconnected");
        }
    }

    pub fn paused(&self) -> bool {
        self.paused
    }

    /// Toggle pause. Returns the new paused state.
    pub fn toggle_pause(&mut self) -> bool {
        self.paused = !self.paused;
        self.step_one_frame = false;
        eprintln!("{}", if self.paused { "Paused" } else { "Resumed" });
        self.paused
    }

    /// While paused, emulate exactly one frame on the next tick.
    pub fn request_frame_advance(&mut self) {
        if self.paused {
            self.step_one_frame = true;
        }
    }

    /// The save state slot the frontend's save/load hotkeys use.
    pub fn slot(&self) -> usize {
        self.slot
    }

    pub fn select_slot(&mut self, slot: usize) {
        self.slot = slot;
        eprintln!("Slot {} selected", slot);
    }

    pub fn save_state(&mut self, slot: usize) {
        save_state_to_slot(&mut self.emu, &self.rom_path, slot);
    }

    pub fn load_state(&mut self, slot: usize) {
        load_state_from_slot(&mut self.emu, &self.rom_path, slot);
    }

    /// Write the battery save now. Call on quit.
    pub fn flush_save(&mut self) {
        self.sav_flusher.flush(&self.emu);
    }

    /// Advance emulation by one frontend tick. `audio_queued` is how many
    /// stereo frames (at the session's sample rate) are waiting in the audio
    /// device's queue, or `None` without an audio device, in which case
    /// normal speed is paced by wall-clock time between ticks.
    ///
    /// Also saves completed printer pages and flushes a settled battery save.
    pub fn tick(&mut self, hold: &HoldInputs, audio_queued: Option<usize>) -> TickOutput {
        self.tick_at(hold, audio_queued, Instant::now())
    }

    fn tick_at(
        &mut self,
        hold: &HoldInputs,
        audio_queued: Option<usize>,
        now: Instant,
    ) -> TickOutput {
        let frame_dur = self.frame_duration();
        let elapsed = self
            .last_tick
            .map_or(Duration::ZERO, |t| now.saturating_duration_since(t));
        self.last_tick = Some(now);
        // Cap the debt so a stall (window drag, backgrounding) does not
        // turn into a catch-up burst.
        self.time_debt = (self.time_debt + elapsed).min(frame_dur * 4);

        let mut out = TickOutput::default();
        let emu_start = Instant::now();
        let rewinding = hold.rewind && !self.paused;
        self.emu.set_rewinding(rewinding);

        if self.paused {
            self.time_debt = Duration::ZERO;
            if std::mem::take(&mut self.step_one_frame) {
                self.step_frames(1);
                out.frames_stepped = 1;
                out.audio = self.emu.drain_audio_samples();
            }
        } else if rewinding {
            self.time_debt = Duration::ZERO;
            let mut audio = Vec::new();
            for _ in 0..REWIND_SPEED {
                if !self.emu.rewind_one_frame() {
                    break;
                }
                out.frames_rewound += 1;
                audio.extend_from_slice(&self.emu.drain_audio_samples());
            }
            crate::util::reverse_audio(&mut audio);
            out.audio = crate::util::downsample_audio(&audio, REWIND_SPEED);
        } else if hold.fast_forward {
            self.time_debt = Duration::ZERO;
            for _ in 0..FAST_FORWARD_SPEED {
                self.emu.step_frame();
            }
            out.frames_stepped = FAST_FORWARD_SPEED as u32;
            let audio = self.emu.drain_audio_samples();
            out.audio = crate::util::downsample_audio(&audio, FAST_FORWARD_SPEED);
        } else if hold.slow_motion {
            // Half speed: one frame per two frame durations of wall time.
            let slow_dur = frame_dur * 2;
            let mut frames = 0;
            while self.time_debt >= slow_dur {
                self.time_debt -= slow_dur;
                frames += 1;
            }
            self.step_frames(frames);
            out.frames_stepped = frames;
            out.audio = self.emu.drain_audio_samples();
        } else {
            let frames = match audio_queued {
                Some(queued) => {
                    self.time_debt = Duration::ZERO;
                    frames_for_audio_queue(queued, self.config.sample_rate)
                }
                None => {
                    let mut frames = 0;
                    while self.time_debt >= frame_dur && frames < 2 {
                        self.time_debt -= frame_dur;
                        frames += 1;
                    }
                    frames
                }
            };
            self.step_frames(frames);
            out.frames_stepped = frames;
            out.audio = self.emu.drain_audio_samples();
        }
        out.emu_time = emu_start.elapsed();

        #[cfg(feature = "image")]
        check_and_save_prints(&mut self.emu);
        self.sav_flusher.poll(&self.emu);
        out
    }

    /// Emulate `frames` frames, running ahead only on the last one (the one
    /// that is displayed).
    fn step_frames(&mut self, frames: u32) {
        for i in 0..frames {
            if i + 1 == frames {
                self.emu.step_frame_runahead(self.config.runahead);
            } else {
                self.emu.step_frame();
            }
        }
    }
}

/// Find the SNES program ROM for SGB LLE: the explicit one, else a known
/// file name for the model in the working directory.
#[cfg(not(target_arch = "wasm32"))]
fn find_snes_rom(model: GbModel, explicit: Option<&[u8]>) -> Option<Vec<u8>> {
    if let Some(data) = explicit {
        return Some(data.to_vec());
    }
    let candidates: &[&str] = match model {
        GbModel::Sgb2 => &["sgb2.program.rom", "sgb2.sfc"],
        GbModel::Sgb => &["sgb1.program.rom", "sgb.sfc"],
        _ => &[],
    };
    candidates.iter().find_map(|name| std::fs::read(name).ok())
}

#[cfg(not(target_arch = "wasm32"))]
fn build_emulator(
    rom: &std::sync::Arc<[u8]>,
    model: GbModel,
    config: &SessionConfig,
    snes_rom: Option<&[u8]>,
) -> crate::emulator::Emulator {
    let boot_rom = load_boot_rom(model, config.bootrom.as_deref(), config.no_boot);
    if boot_rom.is_some() {
        eprintln!("Boot ROM loaded, executing boot sequence.");
    }
    let snes_rom = if model.is_sgb() && config.lle {
        find_snes_rom(model, snes_rom)
    } else {
        None
    };
    if snes_rom.is_some() {
        eprintln!("SNES program ROM loaded, SGB LLE mode active.");
    }
    crate::emulator::Emulator::new(
        rom.clone(),
        boot_rom,
        model,
        snes_rom,
        crate::clock::default_clock(),
        config.sample_rate,
    )
}

/// Load the battery save and attach the configured peripherals.
#[cfg(not(target_arch = "wasm32"))]
fn prepare_emulator(
    mut emu: crate::emulator::Emulator,
    rom_path: &Path,
    config: &SessionConfig,
) -> crate::emulator::Emulator {
    load_sav(&mut emu, rom_path);
    if config.printer {
        let model = emu.model();
        attach_printer(&mut emu, model);
    }
    emu
}

#[cfg(not(target_arch = "wasm32"))]
fn attach_printer(emu: &mut crate::emulator::Emulator, model: GbModel) {
    emu.attach_serial_device(Box::new(crate::printer::Printer::new(
        model.cpu_clock_rate(),
    )));
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

    const RATE: u32 = 48_000;

    /// A queue fill that makes audio-driven pacing step exactly one frame.
    const STEADY_QUEUE: Option<usize> = Some(RATE as usize / 60 * 4);

    /// 32 KiB MBC1+RAM+BATTERY image whose VBlank handler increments the
    /// WRAM byte at $C000, so the byte counts emulated frames.
    fn counter_rom() -> Vec<u8> {
        let mut rom = vec![0u8; 0x8000];
        // VBlank vector: inc (hl) / reti
        rom[0x40..0x42].copy_from_slice(&[0x34, 0xD9]);
        // ld hl,$C000 / ld a,1 / ldh ($FF),a / ei / halt / jr -3 (to halt)
        rom[0x100..0x10B].copy_from_slice(&[
            0x21, 0x00, 0xC0, 0x3E, 0x01, 0xE0, 0xFF, 0xFB, 0x76, 0x18, 0xFD,
        ]);
        rom[0x147] = 0x03;
        rom[0x149] = 0x02;
        rom
    }

    /// A fresh directory holding `names` as copies of the counter ROM.
    fn temp_roms(test: &str, names: &[&str]) -> (std::path::PathBuf, Vec<std::path::PathBuf>) {
        let dir =
            std::env::temp_dir().join(format!("vibeboy_session_{}_{}", test, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let paths = names
            .iter()
            .map(|n| {
                let p = dir.join(n);
                std::fs::write(&p, counter_rom()).unwrap();
                p
            })
            .collect();
        (dir, paths)
    }

    fn session(path: &Path) -> Session {
        Session::new(
            path,
            SessionConfig {
                no_boot: true,
                sample_rate: RATE,
                ..Default::default()
            },
        )
        .unwrap()
    }

    fn counter(s: &Session) -> u8 {
        s.emu.bus().wram[0][0]
    }

    /// Write `val` to the first byte of cartridge RAM.
    fn write_cart_ram(s: &mut Session, val: u8) {
        s.emu.bus_mut().write_byte(0x0000, 0x0A);
        s.emu.bus_mut().write_byte(0xA000, val);
    }

    fn sav_byte(rom: &Path) -> u8 {
        std::fs::read(rom.with_extension("sav")).unwrap()[0]
    }

    #[test]
    fn audio_pacing_thresholds_are_reachable() {
        let per_frame = RATE as usize / 60;
        assert_eq!(frames_for_audio_queue(0, RATE), 2);
        assert_eq!(frames_for_audio_queue(per_frame * 4, RATE), 1);
        assert_eq!(frames_for_audio_queue(per_frame * 9, RATE), 0);
        assert!(audio_ring_capacity(RATE) > per_frame * AUDIO_MAX_FRAMES);
    }

    #[test]
    fn normal_tick_follows_audio_queue() {
        let (dir, roms) = temp_roms("pacing", &["game.gb"]);
        let mut s = session(&roms[0]);
        let hold = HoldInputs::default();
        let start = counter(&s);
        assert_eq!(s.tick(&hold, Some(0)).frames_stepped, 2);
        assert_eq!(s.tick(&hold, STEADY_QUEUE).frames_stepped, 1);
        assert_eq!(s.tick(&hold, Some(RATE as usize)).frames_stepped, 0);
        assert_eq!(counter(&s), start.wrapping_add(3));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn fast_forward_steps_four_frames() {
        let (dir, roms) = temp_roms("ff", &["game.gb"]);
        let mut s = session(&roms[0]);
        let start = counter(&s);
        let out = s.tick(
            &HoldInputs {
                fast_forward: true,
                ..Default::default()
            },
            STEADY_QUEUE,
        );
        assert_eq!(out.frames_stepped, 4);
        assert_eq!(counter(&s), start.wrapping_add(4));
        // Four frames of audio, downsampled to about one frame's length.
        let one_frame = RATE as usize / 60 * 2;
        assert!(out.audio.len() > one_frame / 2 && out.audio.len() < one_frame * 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rewind_goes_back_without_stepping_forward() {
        let (dir, roms) = temp_roms("rewind", &["game.gb"]);
        let mut s = session(&roms[0]);
        for _ in 0..10 {
            s.tick(&HoldInputs::default(), STEADY_QUEUE);
        }
        let before = counter(&s);
        let out = s.tick(
            &HoldInputs {
                rewind: true,
                ..Default::default()
            },
            // An empty queue would make normal pacing step two frames.
            Some(0),
        );
        assert_eq!(out.frames_stepped, 0);
        assert_eq!(out.frames_rewound, 3);
        assert!(!out.audio.is_empty());
        // Rewinding one frame restores the state before a frame and
        // re-runs it, so three rewinds land two frames back.
        assert_eq!(counter(&s), before.wrapping_sub(2));
        assert!(s.emu.is_rewinding());
        s.tick(&HoldInputs::default(), STEADY_QUEUE);
        assert!(!s.emu.is_rewinding());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn pause_holds_until_frame_advance() {
        let (dir, roms) = temp_roms("pause", &["game.gb"]);
        let mut s = session(&roms[0]);
        s.tick(&HoldInputs::default(), STEADY_QUEUE);
        assert!(s.toggle_pause());
        let start = counter(&s);
        let held = HoldInputs {
            rewind: true,
            fast_forward: true,
            slow_motion: true,
        };
        assert_eq!(s.tick(&held, Some(0)).frames_emulated(), 0);
        assert_eq!(counter(&s), start);

        s.request_frame_advance();
        let out = s.tick(&HoldInputs::default(), Some(0));
        assert_eq!(out.frames_stepped, 1);
        assert_eq!(counter(&s), start.wrapping_add(1));
        assert_eq!(s.tick(&HoldInputs::default(), Some(0)).frames_stepped, 0);

        assert!(!s.toggle_pause());
        s.request_frame_advance(); // ignored while running
        assert_eq!(
            s.tick(&HoldInputs::default(), STEADY_QUEUE).frames_stepped,
            1
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn slow_motion_runs_at_half_speed() {
        let (dir, roms) = temp_roms("slow", &["game.gb"]);
        let mut s = session(&roms[0]);
        let hold = HoldInputs {
            slow_motion: true,
            ..Default::default()
        };
        let frame = s.frame_duration();
        let t0 = Instant::now();
        s.tick_at(&hold, STEADY_QUEUE, t0);
        let mut frames = 0;
        for i in 1..=4 {
            frames += s
                .tick_at(&hold, STEADY_QUEUE, t0 + frame * i)
                .frames_stepped;
        }
        assert_eq!(frames, 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn no_audio_device_paces_by_wall_clock() {
        let (dir, roms) = temp_roms("wallclock", &["game.gb"]);
        let mut s = session(&roms[0]);
        let hold = HoldInputs::default();
        let frame = s.frame_duration();
        let t0 = Instant::now();
        assert_eq!(s.tick_at(&hold, None, t0).frames_stepped, 0);
        assert_eq!(s.tick_at(&hold, None, t0 + frame).frames_stepped, 1);
        assert_eq!(s.tick_at(&hold, None, t0 + frame * 3).frames_stepped, 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn reset_and_rom_switch_flush_the_save_first() {
        let (dir, roms) = temp_roms("flush", &["a.gb", "b.gb"]);
        let mut s = session(&roms[0]);
        s.set_printer(true);

        write_cart_ram(&mut s, 0x55);
        s.reset();
        assert_eq!(sav_byte(&roms[0]), 0x55);
        assert_eq!(s.emu.save_data()[0], 0x55, "reset reloads the save");
        assert!(s.emu.serial_device_as_any().is::<crate::printer::Printer>());

        write_cart_ram(&mut s, 0x66);
        s.set_model(Some(GbModel::Mgb));
        assert_eq!(sav_byte(&roms[0]), 0x66);
        assert_eq!(s.emu.model(), GbModel::Mgb);

        write_cart_ram(&mut s, 0x77);
        s.load_rom(&roms[1]).unwrap();
        assert_eq!(sav_byte(&roms[0]), 0x77);
        assert_eq!(s.rom_path(), roms[1].as_path());
        assert!(!roms[1].with_extension("sav").exists());
        assert!(s.emu.serial_device_as_any().is::<crate::printer::Printer>());

        assert!(s.load_rom(&dir.join("missing.gb")).is_err());
        assert_eq!(s.rom_path(), roms[1].as_path());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
