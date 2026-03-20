use crate::bus::Bus;
use crate::cpu::{Cpu, Registers};
use crate::joypad::{
    BTN_A, BTN_B, BTN_DOWN, BTN_LEFT, BTN_RIGHT, BTN_SELECT, BTN_START, BTN_UP,
};
use crate::model::GbModel;
use crate::sgb::Sgb;
use crate::snapshot::Snapshot;
use crate::snes::SnesSys;
use std::collections::VecDeque;
use std::path::Path;

/// T-cycles per frame at normal speed (70224 = 456 × 154).
pub const CYCLES_PER_FRAME: u32 = 70_224;

/// Maximum rewind buffer depth (~10 seconds at 60fps).
const REWIND_BUFFER_CAPACITY: usize = 600;

pub struct Emulator {
    pub cpu: Cpu,
    pub bus: Bus,
    model: GbModel,
    /// Composited SGB output buffer (256×224): border + game area.
    /// Border pixels persist across frames; only the game area is updated each frame.
    sgb_output: Vec<u32>,
    /// SNES subsystem for SGB LLE (None = HLE fallback)
    snes: Option<SnesSys>,
    /// Packets queued while SNES BIOS was initializing (with shade buffer snapshots)
    snes_packet_queue: Vec<([u8; 16], Vec<u8>)>,
    frame_count: u64,
    /// Ring buffer of snapshots for rewind (most recent at back).
    rewind_buffer: VecDeque<Snapshot>,
    /// In-memory save state slots (1-9, index 0 = slot 1).
    save_slots: [Option<Box<Snapshot>>; 9],
    /// True while the user is holding the rewind key.
    pub rewinding: bool,
    /// When true, skip rewind snapshots and audio accumulation (test runner mode).
    pub headless: bool,
}

impl Emulator {
    /// Create a new emulator. If `boot_rom` is Some, the CPU starts at PC=0x0000
    /// with hardware reset registers and executes the boot ROM. Otherwise the CPU
    /// starts at PC=0x0100 with post-boot register values for the given model.
    pub fn new(
        rom: Vec<u8>,
        boot_rom: Option<Vec<u8>>,
        rom_path: Option<&Path>,
        model: GbModel,
        snes_rom: Option<Vec<u8>>,
    ) -> Self {
        let has_boot = boot_rom.is_some();
        let mut cpu = Cpu::new();
        if has_boot {
            cpu.regs = Registers::reset();
        } else {
            cpu.regs = Registers::post_boot_with_rom(model, Some(&rom));
        }
        let mut bus = Bus::new(rom, boot_rom, rom_path, model);

        // When no boot ROM, set post-boot IO register values
        if !has_boot {
            // LCDC=0x91 (LCD on, BG on, tile data $8000)
            bus.write_byte(0xFF40, 0x91);
            // BGP=0xFC (standard DMG palette)
            bus.write_byte(0xFF47, 0xFC);
            // SGB boot ROM doesn't trigger CH1, so NR52=$F0; all others NR52=$F1
            if model.is_sgb() {
                bus.write_byte(0xFF26, 0xF0);
            } else {
                // NR52=$F1 (sound on, CH1 active)
                bus.write_byte(0xFF26, 0xF1);
                bus.write_byte(0xFF11, 0xBF);
                bus.write_byte(0xFF12, 0xF3);
                bus.write_byte(0xFF14, 0xBF);
                // On real hardware, CH1's envelope has fully decayed to 0
                // by the time the boot ROM finishes, so zero the volume.
                bus.apu.set_ch1_post_boot_volume();
            }
            bus.write_byte(0xFF24, 0x77);
            bus.write_byte(0xFF25, 0xF3);
        }

        // Enable LLE mode if we have a SNES program ROM
        let snes = if model.is_sgb() && snes_rom.is_some() {
            if let Some(ref mut sgb) = bus.sgb {
                sgb.lle_mode = true;
            }
            let snes_sys = SnesSys::new(snes_rom.unwrap());
            log::info!("SGB LLE: SNES CPU active (PC=${:04X})", snes_sys.cpu.pc);
            Some(snes_sys)
        } else {
            None
        };

        Emulator {
            cpu,
            bus,
            model,
            sgb_output: vec![0u32; 256 * 224],
            snes,
            snes_packet_queue: Vec::new(),
            frame_count: 0,
            rewind_buffer: VecDeque::with_capacity(REWIND_BUFFER_CAPACITY),
            save_slots: Default::default(),
            rewinding: false,
            headless: false,
        }
    }

    /// Persist battery-backed cartridge RAM to disk.
    pub fn save(&self) {
        self.bus.save_to_disk();
    }

    /// Run until one full frame has been rendered (VBlank).
    pub fn step_frame(&mut self) {
        // Push a snapshot for rewind (before emulating this frame)
        if !self.rewinding && !self.headless {
            let snap = self.save_snapshot();
            if self.rewind_buffer.len() >= REWIND_BUFFER_CAPACITY {
                self.rewind_buffer.pop_front();
            }
            self.rewind_buffer.push_back(snap);
        }

        self.bus.clear_frame_ready();
        self.frame_count += 1;
        let mut cycles = 0u32;
        while !self.bus.frame_ready() {
            self.step();
            cycles += 4;
            // Safety valve: if the ROM toggles LCD off before line 153,
            // frame_ready never fires. Break after 2 frames' worth of cycles.
            if cycles >= CYCLES_PER_FRAME * 2 {
                break;
            }
        }

        // SGB post-processing
        if self.model.is_sgb() {
            if self.snes.is_some() {
                self.step_snes_frame();
            } else {
                self.bus.apply_sgb_palettes();
                self.bus.check_sgb_transfer();
                self.bus.capture_sgb_freeze();
            }

        }
    }

    // ── Snapshot / Rewind / Save State ─────────────────────────────────────────

    /// Capture the full emulator state into a Snapshot.
    pub fn save_snapshot(&mut self) -> Snapshot {
        Snapshot {
            cpu: self.cpu.clone(),
            bus: self.bus.take_snapshot(),
            snes: self.snes.as_ref().map(|s| s.take_snapshot()),
            frame_count: self.frame_count,
        }
    }

    /// Restore full emulator state from a Snapshot.
    pub fn restore_snapshot(&mut self, snap: &Snapshot) {
        self.cpu = snap.cpu.clone();
        self.bus.apply_snapshot(&snap.bus);
        self.frame_count = snap.frame_count;
        if let Some(ref snes_snap) = snap.snes {
            if let Some(ref mut snes) = self.snes {
                snes.apply_snapshot(snes_snap);
            }
        }
        self.snes_packet_queue.clear();
    }

    /// Pop one frame from the rewind buffer and restore it. Returns true if rewound.
    pub fn rewind_one_frame(&mut self) -> bool {
        if let Some(snap) = self.rewind_buffer.pop_back() {
            self.restore_snapshot(&snap);
            true
        } else {
            false
        }
    }

    /// Save current state to the given slot (0-indexed, 0-8 for slots 1-9).
    pub fn save_state(&mut self, slot: usize) {
        if slot < 9 {
            self.save_slots[slot] = Some(Box::new(self.save_snapshot()));
            log::info!("State saved to slot {}", slot + 1);
        }
    }

    /// Load state from the given slot. Returns true if a state was loaded.
    pub fn load_state(&mut self, slot: usize) -> bool {
        if slot < 9 {
            if let Some(snap) = self.save_slots[slot].take() {
                self.restore_snapshot(&snap);
                self.save_slots[slot] = Some(snap);
                self.rewind_buffer.clear();
                log::info!("State loaded from slot {}", slot + 1);
                return true;
            }
        }
        false
    }

    /// Serialize the given slot to bytes for disk/network storage.
    /// Returns None if the slot is empty.
    pub fn save_state_to_bytes(&mut self, slot: usize) -> Option<Vec<u8>> {
        if slot < 9 {
            if let Some(ref snap) = self.save_slots[slot] {
                return Some(crate::savestate::serialize(snap));
            }
        }
        None
    }

    /// Load a save state from bytes into the given slot and restore it.
    /// Returns true on success.
    pub fn load_state_from_bytes(&mut self, slot: usize, data: &[u8]) -> bool {
        match crate::savestate::deserialize(data) {
            Ok(snap) => {
                self.restore_snapshot(&snap);
                self.save_slots[slot] = Some(Box::new(snap));
                self.rewind_buffer.clear();
                true
            }
            Err(e) => {
                log::error!("Failed to load save state: {}", e);
                false
            }
        }
    }

    /// Run the SNES subsystem for one frame (LLE mode).
    fn step_snes_frame(&mut self) {
        let snes = self.snes.as_mut().unwrap();

        // 1. Drain pending SGB packets from the GB side
        let new_packets: Vec<[u8; 16]> = if let Some(ref mut sgb) = self.bus.sgb {
            std::mem::take(&mut sgb.pending_packets)
        } else {
            vec![]
        };

        // 2. Snapshot current shade buffer for any new packets
        let shade_snapshot = self.bus.ppu.shade_buffer.clone();

        // Queue packets during first ~30 frames (SPC upload + init), then replay
        if snes.frame_count < 32 {
            // Store each packet with its shade buffer snapshot
            for pkt in &new_packets {
                self.snes_packet_queue.push((*pkt, shade_snapshot.clone()));
            }
            // Feed current shade buffer and run one SNES frame (SPC upload)
            snes.feed_scanlines(&shade_snapshot);
            snes.run_frame();
        } else {
            // Replay any queued packets from during SPC upload, then process new ones
            let queued = std::mem::take(&mut self.snes_packet_queue);
            let has_packets = !queued.is_empty() || !new_packets.is_empty();

            if !has_packets {
                snes.feed_scanlines(&shade_snapshot);
                snes.run_frame();
            } else {
                // Process queued packets (from SPC upload phase) with their saved shades
                for (pkt, shade) in &queued {
                    snes.feed_scanlines(shade);
                    snes.feed_packet(pkt);
                    snes.run_frame();
                }
                // Process new packets with the current shade buffer
                for pkt in &new_packets {
                    snes.feed_scanlines(&shade_snapshot);
                    snes.feed_packet(pkt);
                    snes.run_frame();
                }
            }

            // The BIOS display pipeline doesn't fully reach the DATA_SND patched
            // code due to incomplete scanline timing, so the SNES VRAM tilemap
            // doesn't contain meaningful per-tile palette data.
            // HLE commands (ATTR_SET, PAL_SET+attr file) already set attr_map
            // correctly via sgb.rs, so we don't override from SNES VRAM.
        }

        // Apply palettes and handle masking (same as HLE path)
        self.bus.apply_sgb_palettes();
        self.bus.check_sgb_transfer();
        self.bus.capture_sgb_freeze();
    }

    /// Execute one CPU instruction and advance bus components.
    fn step(&mut self) {
        self.cpu.step(&mut self.bus);
        // Ticking is now done inline by CPU during each M-cycle
    }

    /// Run until the Mooneye LD B,B breakpoint (opcode 0x40 at PC) or
    /// `max_frames` frames elapse. Returns Some((b,c,d,e,h,l)) on breakpoint.
    pub fn run_until_breakpoint(&mut self, max_frames: u32) -> Option<[u8; 6]> {
        let cycles_per_frame = 70_224u32;
        let mut cycles = 0u32;
        let limit = cycles_per_frame * max_frames;
        loop {
            // Check for Mooneye breakpoints before executing:
            // - LD B,B ($40): modern mooneye-test-suite
            // - NOP; JR -3 ($00 $18 $FD): older mooneye-gb halt_execution loop
            //   We detect at the NOP so we catch it on the first iteration
            let opcode = self.bus.read_byte(self.cpu.regs.pc);
            if opcode == 0x40
                || (opcode == 0x00
                    && self.bus.read_byte(self.cpu.regs.pc.wrapping_add(1)) == 0x18
                    && self.bus.read_byte(self.cpu.regs.pc.wrapping_add(2)) == 0xFD)
            {
                let r = &self.cpu.regs;
                return Some([r.b, r.c, r.d, r.e, r.h, r.l]);
            }
            self.cpu.step(&mut self.bus);
            cycles += 4; // approximate; good enough for frame counting
            if cycles >= limit {
                return None;
            }
        }
    }

    /// Run until serial output contains "Passed" or "Failed", or the Blargg
    /// done loop (JR -2) is hit, or max_frames elapse.
    /// Returns the serial output as a string.
    pub fn run_until_serial_result(&mut self, max_frames: u32) -> String {
        let limit = 70_224u64 * max_frames as u64;
        let mut total_cycles = 0u64;
        loop {
            let pc = self.cpu.regs.pc;
            self.cpu.step(&mut self.bus);
            total_cycles += 4; // approximate; good enough for timeout

            // Detect Blargg done loop: JR -2 (opcode 18 FE)
            if self.bus.read_byte(pc) == 0x18 && self.bus.read_byte(pc.wrapping_add(1)) == 0xFE {
                let serial = String::from_utf8_lossy(&self.bus.serial.serial_output).into_owned();
                let result = self.bus.read_byte(0xA000);
                if result == 0 {
                    if serial.is_empty() { return "Passed".to_string(); }
                    return format!("Passed\n{}", serial);
                } else {
                    if serial.is_empty() { return format!("Failed #{}", result); }
                    return format!("Failed #{}\n{}", result, serial);
                }
            }

            // Check periodically for serial output or timeout
            if total_cycles % 4096 == 0 {
                let output = String::from_utf8_lossy(&self.bus.serial.serial_output);
                if output.contains("Passed") || output.contains("Failed") {
                    return output.into_owned();
                }
                if total_cycles >= limit {
                    return output.into_owned();
                }
            }
        }
    }

    /// Return the current frame buffer (160 × 144 pixels, 0x00RRGGBB).
    pub fn frame_buffer(&self) -> &[u32] {
        // For SGB with freeze mask, return frozen buffer
        if let Some(ref sgb) = self.bus.sgb {
            if sgb.mask_mode == 1 {
                if let Some(ref frozen) = sgb.frozen_buffer {
                    return frozen;
                }
            }
        }
        self.bus.ppu.frame_buffer()
    }

    pub fn is_sgb(&self) -> bool {
        self.model.is_sgb()
    }

    /// Get the composited 256×224 SGB frame (border + game).
    pub fn sgb_composited_frame(&mut self) -> &[u32] {
        // Re-render border directly into sgb_output only when dirty.
        // The border pixels persist across frames, avoiding a 230KB memcpy.
        if let Some(ref sgb) = self.bus.sgb {
            if sgb.border_dirty {
                sgb.render_border(&mut self.sgb_output);
            }
        }
        if let Some(ref mut sgb) = self.bus.sgb {
            sgb.border_dirty = false;
        }

        // Composite game frame into the game area (48,40)-(208,184)
        let boot_pending = self.bus.sgb.as_ref().map_or(false, |s| s.boot_pending);
        if !boot_pending {
            let game_buf = if let Some(ref sgb) = self.bus.sgb {
                if sgb.mask_mode == 1 {
                    if let Some(ref frozen) = sgb.frozen_buffer {
                        frozen.as_slice()
                    } else {
                        self.bus.ppu.frame_buffer()
                    }
                } else {
                    self.bus.ppu.frame_buffer()
                }
            } else {
                self.bus.ppu.frame_buffer()
            };
            Sgb::composite_frame(&mut self.sgb_output, game_buf);
        } else {
            // Black out the game area during boot
            for y in 0..144usize {
                let row_start = (y + 40) * 256 + 48;
                for x in 0..160usize {
                    self.sgb_output[row_start + x] = 0;
                }
            }
        }

        &self.sgb_output
    }

    // ── Input handling ────────────────────────────────────────────────────────

    pub fn set_button(&mut self, button: u8, pressed: bool) {
        self.bus.joypad.set_button(button, pressed);
    }

    // Expose button constants for main.rs
    pub const BTN_RIGHT:  u8 = BTN_RIGHT;
    pub const BTN_LEFT:   u8 = BTN_LEFT;
    pub const BTN_UP:     u8 = BTN_UP;
    pub const BTN_DOWN:   u8 = BTN_DOWN;
    pub const BTN_A:      u8 = BTN_A;
    pub const BTN_B:      u8 = BTN_B;
    pub const BTN_SELECT: u8 = BTN_SELECT;
    pub const BTN_START:  u8 = BTN_START;
}
