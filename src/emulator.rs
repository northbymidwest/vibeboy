use crate::bus::Bus;
use crate::clock::Clock;
use crate::cpu::{Cpu, McycleOp, Registers};
use crate::joypad::{
    BTN_A, BTN_B, BTN_DOWN, BTN_LEFT, BTN_RIGHT, BTN_SELECT, BTN_START, BTN_UP,
};
use crate::model::GbModel;
use crate::rewind::RewindBuffer;
use crate::sgb::Sgb;
use crate::snapshot::Snapshot;
use crate::snes::SnesSys;

/// T-cycles per frame at normal speed (70224 = 456 × 154).
pub const CYCLES_PER_FRAME: u32 = 70_224;

/// Maximum rewind buffer depth (~10 minutes at 60fps).
const REWIND_BUFFER_CAPACITY: usize = 36_000;

pub struct Emulator {
    /// CPU state — use facade methods for normal access. Direct access
    /// available via `cpu()` / `cpu_mut()` for debuggers and test harnesses.
    cpu: Cpu,
    /// Bus (memory + subsystems) — use facade methods for normal access.
    /// Direct access available via `bus()` / `bus_mut()` for debuggers.
    bus: Bus,
    model: GbModel,
    /// Composited SGB output buffer (256×224): border + game area.
    /// Border pixels persist across frames; only the game area is updated each frame.
    sgb_output: Vec<u32>,
    /// SNES subsystem for SGB LLE (None = HLE fallback)
    snes: Option<SnesSys>,
    /// Packets queued while SNES BIOS was initializing (with shade buffer snapshots)
    snes_packet_queue: Vec<([u8; 16], std::sync::Arc<[u8]>)>,
    frame_count: u64,
    /// Delta-compressed rewind buffer.
    rewind_buffer: RewindBuffer,
    /// In-memory save state slots (0-9).
    save_slots: [Option<Box<Snapshot>>; 10],
    /// True while the user is holding the rewind key.
    rewinding: bool,
    /// When true, skip rewind snapshots and audio accumulation (test runner mode).
    headless: bool,
}

impl Emulator {
    /// Create a new emulator. If `boot_rom` is Some, the CPU starts at PC=0x0000
    /// with hardware reset registers and executes the boot ROM. Otherwise the CPU
    /// starts at PC=0x0100 with post-boot register values for the given model.
    pub fn new(
        rom: impl Into<std::sync::Arc<[u8]>>,
        boot_rom: Option<Vec<u8>>,
        model: GbModel,
        snes_rom: Option<Vec<u8>>,
        clock: std::sync::Arc<dyn Clock>,
    ) -> Self {
        let rom: std::sync::Arc<[u8]> = rom.into();
        let has_boot = boot_rom.is_some();
        let mut cpu = Cpu::new();
        if has_boot {
            cpu.regs = Registers::reset();
        } else {
            cpu.regs = Registers::post_boot_with_rom(model, Some(&rom));
        }
        let mut bus = Bus::new(rom, boot_rom, model, clock);

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
            rewind_buffer: RewindBuffer::new(REWIND_BUFFER_CAPACITY),
            save_slots: Default::default(),
            rewinding: false,
            headless: false,
        }
    }

    /// Run until one full frame has been rendered (VBlank).
    pub fn step_frame(&mut self) {
        // Push a snapshot for rewind (before emulating this frame).
        // Strip frame_buffer/shade_buffer before serializing — they're output
        // data regenerated each frame, not state. Saves ~92KB per delta.
        if !self.rewinding && !self.headless {
            let mut snap = self.save_snapshot();
            snap.bus.ppu.frame_buffer.clear();
            snap.bus.ppu.shade_buffer.clear();
            self.rewind_buffer.push(&snap);
        }

        self.bus.clear_frame_ready();
        self.frame_count += 1;
        let mut cycles = 0u32;
        while !self.bus.frame_ready() {
            cycles += self.step();
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

    /// Run one frame, then speculatively run `ahead` frames to reduce input
    /// latency. Displays the frame from `ahead` frames in the future but keeps
    /// the emulator state at the current frame. Audio comes from the real frame.
    pub fn step_frame_runahead(&mut self, ahead: u32) {
        // Run the real frame (generates audio, pushes rewind snapshot)
        self.step_frame();

        if ahead == 0 {
            return;
        }

        // Drain real audio before running ahead (preserve it across restore)
        let real_audio = std::mem::take(&mut self.bus.apu.sample_buf);
        let filter_state = self.bus.apu.save_filter_state();

        // Save state before running ahead
        let snap = self.save_snapshot();

        // Run ahead frames silently (no audio, no rewind snapshots)
        let was_headless = self.headless;
        self.headless = true;
        for _ in 0..ahead {
            self.bus.clear_frame_ready();
            self.frame_count += 1;
            let mut cycles = 0u32;
            while !self.bus.frame_ready() {
                cycles += self.step();
                if cycles >= CYCLES_PER_FRAME * 2 { break; }
            }
        }
        self.headless = was_headless;

        // Grab the ahead frame buffer
        let ahead_fb = self.bus.ppu.frame_buffer.clone();
        let ahead_shade = if self.bus.ppu.sgb_mode {
            self.bus.ppu.shade_buffer.clone()
        } else {
            Vec::new()
        };

        // Restore real state
        self.restore_snapshot(&snap);

        // Restore APU filter state and audio buffer
        self.bus.apu.sample_buf = real_audio;
        self.bus.apu.restore_filter_state(&filter_state);

        // Overwrite frame buffer with the ahead version for display
        self.bus.ppu.frame_buffer = ahead_fb;
        if self.bus.ppu.sgb_mode {
            self.bus.ppu.shade_buffer = ahead_shade;
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
    /// Runs one frame after restoring to regenerate the frame buffer (which is
    /// excluded from rewind snapshots to save memory).
    pub fn rewind_one_frame(&mut self) -> bool {
        if let Some(snap) = self.rewind_buffer.pop() {
            // Preserve APU filter state across restore for clean audio
            let filter_state = self.bus.apu.save_filter_state();
            self.restore_snapshot(&snap);
            self.bus.apu.restore_filter_state(&filter_state);
            // Re-allocate output buffers (cleared before serialization to save space)
            let (w, h) = if self.bus.ppu.sgb_mode { (256, 224) } else { (160, 144) };
            if self.bus.ppu.frame_buffer.len() != w * h {
                self.bus.ppu.frame_buffer.resize(w * h, 0);
            }
            if self.bus.ppu.sgb_mode && self.bus.ppu.shade_buffer.len() != 160 * 144 {
                self.bus.ppu.shade_buffer.resize(160 * 144, 0);
            }
            // Regenerate frame buffer (and audio) from restored VRAM/PPU state
            self.bus.clear_frame_ready();
            let mut cycles = 0u32;
            while !self.bus.frame_ready() {
                cycles += self.step();
                if cycles >= CYCLES_PER_FRAME * 2 { break; }
            }
            true
        } else {
            false
        }
    }

    /// Save current state to the given slot (0-indexed, 0-8 for slots 1-9).
    pub fn save_state(&mut self, slot: usize) {
        if slot < 10 {
            self.save_slots[slot] = Some(Box::new(self.save_snapshot()));
            log::info!("State saved to slot {}", (slot + 1) % 10);
        }
    }

    /// Load state from the given slot. Returns true if a state was loaded.
    pub fn load_state(&mut self, slot: usize) -> bool {
        if slot < 10 {
            if let Some(snap) = self.save_slots[slot].take() {
                self.restore_snapshot(&snap);
                self.save_slots[slot] = Some(snap);
                self.rewind_buffer.clear();
                log::info!("State loaded from slot {}", (slot + 1) % 10);
                return true;
            }
        }
        false
    }

    /// Serialize the given slot to bytes for disk/network storage.
    /// Returns None if the slot is empty.
    pub fn save_state_to_bytes(&mut self, slot: usize) -> Option<Vec<u8>> {
        if slot < 10 {
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

        // 2. Snapshot current shade buffer for any new packets (Arc = cheap clone)
        let shade_snapshot: std::sync::Arc<[u8]> = self.bus.ppu.shade_buffer.clone().into();

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

    /// Execute one CPU instruction via the M-cycle state machine.
    /// Returns the total T-cycles consumed (including any DMA halt cycles).
    fn step(&mut self) -> u32 {
        // Check for pending interrupts at the START of each step, matching
        // hardware behavior where the CPU checks IE & IF before fetching
        // the next opcode. This must happen before mcycle() so the CPU
        // enters interrupt dispatch instead of opcode fetch.
        if !self.cpu.in_interrupt && !self.cpu.halted
            && self.cpu.ime && self.cpu.speed_switch_remaining == 0
            && self.cpu.phase == 0
        {
            self.bus.flush_ppu_deferred();
            let pending = self.bus.ie & self.bus.if_ & 0x1F;
            if pending != 0 {
                self.cpu.begin_interrupt_dispatch();
            }
        }

        let mut total = 0u32;
        loop {
            let op = self.cpu.mcycle();
            let is_done = op == McycleOp::Done
                || op == McycleOp::HaltNop;

            // Handle OAM bugs from this mcycle() call BEFORE ticking.
            // This matches the old model where trigger_oam_bug runs between
            // the CPU state change and the bus tick.
            if let Some(addr) = self.cpu.oam_bug_addr.take() {
                self.bus.trigger_oam_bug(addr);
                // Interrupt dispatch phase 1 also triggers OAM bug on SP.
                // CPU stores SP in tmp16 at phase 1 for this purpose.
                if self.cpu.in_interrupt && self.cpu.interrupt_phase == 2 {
                    self.bus.trigger_oam_bug(self.cpu.tmp16);
                }
            }
            if let Some(addr) = self.cpu.oam_bug_read_addr.take() {
                self.bus.trigger_oam_bug_read(addr);
            }

            match op {
                McycleOp::Done => {}
                McycleOp::Read { addr } => {
                    self.cpu.data_latch = self.bus.tick_read(addr);
                    total += 4;
                }
                McycleOp::Write { addr, val } => {
                    self.bus.tick_write(addr, val);
                    total += 4;
                }
                McycleOp::Internal => {
                    self.bus.tick_internal();
                    total += 4;
                }
                McycleOp::HaltNop => {
                    self.bus.tick_half();
                    self.bus.flush_ppu_deferred();
                    let pending = self.bus.ie() & self.bus.if_ & 0x1F;
                    self.bus.tick_half_post();
                    total += 4;
                    if pending != 0 {
                        self.cpu.halted = false;
                        if self.cpu.ime {
                            self.cpu.begin_interrupt_dispatch();
                        } else {
                            self.cpu.halt_bug = true;
                        }
                    }
                    break;
                }
                McycleOp::SpeedSwitchIdle => {
                    self.bus.tick_speed_switch_idle();
                    self.cpu.speed_switch_remaining -= 1;
                    if self.cpu.speed_switch_remaining == self.cpu.speed_switch_toggle_at {
                        self.bus.do_speed_toggle();
                    }
                    if self.cpu.speed_switch_remaining == 0 {
                        self.cpu.halted = false;
                    }
                    total += 4;
                    break;
                }
            }

            // Interrupt vector resolution: after push-high-byte tick in interrupt dispatch,
            // resolve IE & IF to determine the vector address (between phases 2 and 3).
            if self.cpu.in_interrupt && self.cpu.interrupt_phase == 3 {
                self.bus.flush_ppu_deferred();
                let pending = self.bus.ie & self.bus.if_ & 0x1F;
                if pending != 0 {
                    let bit = pending.trailing_zeros() as u8;
                    self.bus.if_ &= !(1 << bit);
                    self.cpu.tmp16 = match bit {
                        0 => 0x0040,
                        1 => 0x0048,
                        2 => 0x0050,
                        3 => 0x0058,
                        4 => 0x0060,
                        _ => 0x0040,
                    };
                } else {
                    self.cpu.tmp16 = 0x0000; // cancelled dispatch
                }
            }

            // Done means instruction is complete — break after handling
            // any pending interrupts at this instruction boundary.
            if is_done {
                // Undefined opcodes: CPU locks (halted + IME=false + IE=0)
                // Must be checked BEFORE halt_bug detection so IE is cleared first.
                if self.cpu.halted && !self.cpu.ime && matches!(self.cpu.opcode,
                    0xD3 | 0xDB | 0xDD | 0xE3 | 0xE4 | 0xEB | 0xEC | 0xED | 0xF4 | 0xFC | 0xFD)
                {
                    self.bus.ie = 0;
                }

                // HALT with IME=false: detect halt_bug (pending interrupt skips next opcode fetch)
                if self.cpu.halted && !self.cpu.ime {
                    self.bus.flush_ppu_deferred();
                    let pending = self.bus.ie & self.bus.if_ & 0x1F;
                    if pending != 0 {
                        self.cpu.halt_bug = true;
                        self.cpu.halted = false;
                    }
                }

                // STOP: handle speed switch if armed
                if self.cpu.opcode == 0x10 && self.cpu.speed_switch_remaining == 0
                    && self.bus.speed_switch_armed()
                {
                    self.bus.do_speed_switch_prepare();
                    self.cpu.speed_switch_remaining = 2050;
                    self.cpu.speed_switch_toggle_at = 2050 / 2;
                    self.cpu.halted = true;
                }

                // Interrupt check moved to start of step() to match hardware
                // timing (CPU checks IE & IF before fetching next opcode).
                break;
            }
        }
        let dma_extra = self.bus.dma_halt_cycles;
        self.bus.dma_halt_cycles = 0;
        total + dma_extra
    }

    /// Run until the Mooneye LD B,B breakpoint (opcode 0x40 at PC) or
    /// `max_frames` frames elapse. Returns Some((b,c,d,e,h,l)) on breakpoint.
    pub fn run_until_breakpoint(&mut self, max_frames: u32) -> Option<[u8; 6]> {
        let cycles_per_frame = 70_224u32;
        let mut cycles = 0u32;
        let limit = cycles_per_frame * max_frames;
        let mut iter_count = 0u64;
        loop {
            iter_count += 1;
            if iter_count == 1_000_000 {
                eprintln!("1M iters: PC={:04X} cycles={} halted={} ime={} phase={} in_int={}",
                    self.cpu.regs.pc, cycles, self.cpu.halted, self.cpu.ime,
                    self.cpu.phase, self.cpu.in_interrupt);
            }
            if iter_count == 10_000_000 {
                eprintln!("10M iters: PC={:04X} cycles={} halted={} ime={} phase={} in_int={}",
                    self.cpu.regs.pc, cycles, self.cpu.halted, self.cpu.ime,
                    self.cpu.phase, self.cpu.in_interrupt);
            }
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
            cycles += self.step();
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
            total_cycles += self.step() as u64;

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

    pub fn model(&self) -> crate::model::GbModel {
        self.model
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

    // ── Audio ─────────────────────────────────────────────────────────────────

    /// Drain and return all pending audio samples (interleaved L/R f32).
    pub fn drain_audio_samples(&mut self) -> Vec<f32> {
        self.bus.apu.drain_samples()
    }

    /// Enable or disable audio sample generation.
    pub fn set_audio_enabled(&mut self, enabled: bool) {
        self.bus.apu.headless = !enabled;
    }

    // ── Serial / Printer ──────────────────────────────────────────────────────

    /// Attach a device to the serial port (e.g. printer).
    pub fn attach_serial_device(&mut self, device: Box<dyn crate::serial::SerialDevice>) {
        self.bus.serial.device = device;
    }

    /// Access the serial device as `&dyn Any` for downcasting.
    pub fn serial_device_as_any(&self) -> &dyn std::any::Any {
        self.bus.serial.device.as_any()
    }

    /// Access the serial device as `&mut dyn Any` for downcasting.
    pub fn serial_device_as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self.bus.serial.device.as_any_mut()
    }

    // ── Cartridge queries ─────────────────────────────────────────────────────

    pub fn has_camera(&self) -> bool { self.bus.cart.has_camera() }
    pub fn has_accelerometer(&self) -> bool { self.bus.cart.has_accelerometer() }
    pub fn has_rumble(&self) -> bool { self.bus.cart.has_rumble() }
    pub fn rumble_active(&self) -> bool { self.bus.cart.rumble_active() }
    /// Returns true if rumble was active at any point since the last call, then clears.
    pub fn drain_rumble(&mut self) -> bool { self.bus.cart.drain_rumble() }
    pub fn has_battery(&self) -> bool { self.bus.cart.has_battery() }

    pub fn set_camera_image(&mut self, data: &[u8; 128 * 112]) { self.bus.cart.set_camera_image(data); }
    pub fn set_accelerometer(&mut self, x: u16, y: u16) { self.bus.cart.set_accelerometer(x, y); }

    pub fn save_data(&self) -> Vec<u8> { self.bus.cart.save_data() }
    pub fn load_ram(&mut self, data: &[u8]) { self.bus.cart.load_ram(data); }

    // ── Bus state queries ─────────────────────────────────────────────────────

    pub fn is_double_speed(&self) -> bool { self.bus.double_speed }

    // ── Direct access (for debuggers and test harnesses) ──────────────────────

    pub fn cpu(&self) -> &Cpu { &self.cpu }
    pub fn cpu_mut(&mut self) -> &mut Cpu { &mut self.cpu }
    pub fn bus(&self) -> &Bus { &self.bus }
    pub fn bus_mut(&mut self) -> &mut Bus { &mut self.bus }

    /// Execute one CPU instruction, advancing bus subsystems accordingly.
    pub fn cpu_step(&mut self) { self.step(); }

    /// Enable headless mode: disables audio accumulation and rewind snapshots.
    /// Used by test runner and calibration tools.
    pub fn set_headless(&mut self, headless: bool) {
        self.headless = headless;
        self.bus.apu.headless = headless;
    }

    // ── Rewind state ──────────────────────────────────────────────────────────

    pub fn set_rewinding(&mut self, active: bool) { self.rewinding = active; }
    pub fn is_rewinding(&self) -> bool { self.rewinding }
    pub fn rewind_memory_usage(&self) -> usize { self.rewind_buffer.memory_usage() }
}
