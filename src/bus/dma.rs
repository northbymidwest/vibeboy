/// OAM DMA, HDMA, and bus tick/timing.

use super::Bus;

impl Bus {
    // ── OAM DMA ───────────────────────────────────────────────────────────────

    pub(super) fn start_oam_dma(&mut self, source_page: u8) {
        // Store source page in PPU register so 0xFF46 reads back correctly.
        self.ppu.write(0xFF46, source_page);
        // Use the pre-step blocking state captured at the start of this M-cycle.
        let was_blocking = self.oam_dma.blocking;
        // Schedule DMA: 1 M-cycle delay before blocking starts, then 160 transfers.
        self.oam_dma = super::OamDma {
            active:       true,
            source:       (source_page as u16) << 8,
            progress:     0,
            delay:        1,
            was_blocking,
            blocking:     was_blocking,
        };
    }

    /// Advance OAM DMA by one M-cycle. Called from tick_mcycle().
    /// DMA takes 162 M-cycles total: 1 warmup (delay) + 160 copies + 1 teardown.
    /// During teardown, bus is still blocked but no data is transferred.
    pub fn step_oam_dma(&mut self) {
        if !self.oam_dma.active { return; }
        if self.oam_dma.delay > 0 {
            self.oam_dma.delay -= 1;
            return;
        }
        if self.oam_dma.progress < 160 {
            // Copy one byte from source to OAM.
            let mut src = self.oam_dma.source + self.oam_dma.progress as u16;
            let byte = if self.model.is_cgb() && src >= 0xE000 {
                // CGB: source >= $E000 reads as 0xFF
                0xFF
            } else {
                // DMG: $FE00-$FFFF maps to echo WRAM ($DE00-$DFFF)
                if !self.model.is_cgb() && src >= 0xFE00 {
                    src -= 0x2000;
                }
                self.read_byte_raw(src)
            };
            self.ppu.oam[self.oam_dma.progress as usize] = byte;
        }
        self.oam_dma.progress += 1;
        // CGB has a teardown M-cycle at progress=160 (bus still blocked, no copy)
        // DMG DMA ends immediately after the last byte
        let end = if self.model.is_cgb() { 161 } else { 160 };
        if self.oam_dma.progress >= end {
            self.oam_dma.active = false;
        }
    }

    // ── HDMA ──────────────────────────────────────────────────────────────────

    pub(super) fn start_hdma(&mut self, val: u8) {
        let mode = (val >> 7) & 1;

        // Cancel active H-Blank DMA by writing with bit 7 = 0
        if mode == 0 && self.hdma.active && self.hdma.mode == 1 {
            self.hdma.active = false;
            return;
        }

        let blocks = (val & 0x7F) + 1;
        self.hdma.blocks = blocks;
        self.hdma.mode = mode;
        self.hdma.active = true;

        if mode == 0 {
            // General purpose DMA: transfer all blocks, ticking the bus
            self.do_gdma(blocks);
            self.hdma.active = false;
        } else {
            // H-Blank DMA: if PPU is already in mode 0 (HBlank), transfer
            // the first block immediately per hardware behavior (jsgroth tests).
            self.ppu.hblank_entered = false;
            if self.ppu.mode == 0 {
                self.do_hdma_one_block();
            }
        }
    }

    /// GDMA: transfer all blocks with per-M-cycle subsystem interleaving.
    /// Setup: 2 bus M-cycles (normal) or 1 bus M-cycle (DS).
    /// Transfer: 8 bus M-cycles per 16-byte block (2 bytes per M-cycle).
    ///
    /// Each M-cycle ticks DIV/Timer/Serial/APU/PPU so that events (APU
    /// frame sequencer, timer overflow, etc.) fire at the correct time
    /// during the transfer instead of being batched at the end.
    fn do_gdma(&mut self, blocks: u8) {
        let ds = self.double_speed;
        let timer_per_mcycle: u32 = if ds { 8 } else { 4 };
        let bus_per_mcycle: u32 = if ds { 2 } else { 4 };
        self.hdma.in_transfer = true;

        // Setup M-cycles (no data transfer)
        let setup_mcycles: u32 = if ds { 1 } else { 2 };
        for _ in 0..setup_mcycles {
            self.tick_dma_mcycle(timer_per_mcycle, bus_per_mcycle);
        }

        // Transfer: 8 M-cycles per block, 2 bytes per M-cycle
        for _ in 0..blocks {
            self.do_hdma_block_interleaved(timer_per_mcycle, bus_per_mcycle);
        }

        self.hdma.in_transfer = false;

        // CPU halt cycles
        let total_bus_mcycles = setup_mcycles + blocks as u32 * 8;
        self.dma_halt_cycles += total_bus_mcycles * timer_per_mcycle;
    }

    /// Copy one 16-byte HDMA block with per-M-cycle subsystem ticking.
    /// Each of the 8 M-cycles copies 2 bytes then ticks all subsystems.
    fn do_hdma_block_interleaved(&mut self, timer_per_mcycle: u32, bus_per_mcycle: u32) {
        for byte_off in (0..16u16).step_by(2) {
            let src_addr = self.hdma.src.wrapping_add(byte_off);
            let dst_addr = self.hdma.dst.wrapping_add(byte_off);
            let b0 = self.read_byte_raw(src_addr);
            self.ppu.write_vram(dst_addr, b0);
            let b1 = self.read_byte_raw(src_addr + 1);
            self.ppu.write_vram(dst_addr + 1, b1);
            self.tick_dma_mcycle(timer_per_mcycle, bus_per_mcycle);
        }
        self.hdma.src = self.hdma.src.wrapping_add(16);
        self.hdma.dst = self.hdma.dst.wrapping_add(16);
        let dst_off = (self.hdma.dst.wrapping_sub(0x8000)) & 0x1FFF;
        self.hdma.dst = 0x8000 + dst_off;
    }

    /// Tick all subsystems for one M-cycle during a DMA transfer.
    /// Also advances OAM DMA since it runs independently of HDMA.
    fn tick_dma_mcycle(&mut self, timer_cycles: u32, bus_cycles: u32) {
        self.oam_dma.blocking = self.oam_dma.compute_blocking();
        self.tick(timer_cycles, bus_cycles);
        self.step_oam_dma();
    }

    // ── Tick: advance all components by T-cycles ──────────────────────────────

    /// Tick only the PPU for one M-cycle during CGB speed switch idle.
    /// Per Pan Docs, DIV resets and stops ticking during the STOP halt;
    /// timer, serial, and APU do not advance. PPU continues rendering.
    pub fn tick_speed_switch_idle(&mut self) {
        let bus_cycles = if self.double_speed { 2u32 } else { 4 };
        let ppu_flags = self.ppu.step(bus_cycles);
        self.if_ |= ppu_flags;
    }

    /// Tick the bus by one M-cycle (4 T-cycles normal speed, 2 in double-speed).
    /// Call this once per CPU M-cycle (memory access or internal cycle).
    pub fn tick_mcycle(&mut self) {
        // Capture blocking state BEFORE advancing DMA so CPU accesses in this M-cycle
        // see the correct blocking state (e.g. last DMA copy still blocks OAM).
        self.oam_dma.blocking = self.oam_dma.compute_blocking();

        // Timer is clocked by the CPU, so always 4 T-cycles per M-cycle.
        // PPU/APU run at fixed 4MHz, so 2 T-cycles per M-cycle in double-speed.
        let bus_cycles = if self.double_speed { 2u32 } else { 4 };

        // Apply any PPU tick debt from mid-M-cycle glitch handling
        let debt = self.ppu_tick_debt;
        self.ppu_tick_debt = 0;
        let ppu_cycles = bus_cycles.saturating_sub(debt);

        // Accumulate PPU cycles for lazy flushing. PPU ticks are deferred
        // until the next read_byte/if_reg/write_byte, allowing register writes
        // to take effect at the correct mid-M-cycle point.
        self.ppu_deferred += ppu_cycles;

        // Tick everything except PPU
        self.tick_split(4, bus_cycles, 0);

        self.check_hdma_hblank();
        self.step_oam_dma();
    }

    /// Check if a prior PPU flush detected mode 0 entry and trigger
    /// HDMA mode 1 transfer. Called from tick_mcycle (after CPU read)
    /// so transfer data isn't visible until the next read.
    pub fn check_hdma_hblank(&mut self) {
        if self.ppu.hblank_entered && !self.hdma.in_transfer {
            self.ppu.hblank_entered = false;
            if self.hdma.active && self.hdma.mode == 1 {
                self.do_hdma_one_block();
            }
        }
    }

    /// Transfer one HDMA block with per-M-cycle interleaving.
    /// Used by both the HBlank trigger path and the immediate-mode-0 path.
    fn do_hdma_one_block(&mut self) {
        let ds = self.double_speed;
        let timer_per_mcycle: u32 = if ds { 8 } else { 4 };
        let bus_per_mcycle: u32 = if ds { 2 } else { 4 };
        self.hdma.in_transfer = true;

        // Setup M-cycles (no data transfer)
        let setup_mcycles: u32 = if ds { 1 } else { 2 };
        for _ in 0..setup_mcycles {
            self.tick_dma_mcycle(timer_per_mcycle, bus_per_mcycle);
        }

        // Transfer: 8 M-cycles for one 16-byte block
        self.do_hdma_block_interleaved(timer_per_mcycle, bus_per_mcycle);

        self.hdma.in_transfer = false;

        // CPU halt cycles
        let total_bus_mcycles = setup_mcycles + 8;
        self.dma_halt_cycles += total_bus_mcycles * timer_per_mcycle;

        self.hdma.blocks -= 1;
        if self.hdma.blocks == 0 {
            self.hdma.active = false;
        }
    }

    /// Tick the bus by half an M-cycle (2 T-cycles normal speed, 1 in double-speed).
    /// Used for HALT wake timing: hardware checks IF at the midpoint of the HALT NOP.
    pub fn tick_half_mcycle(&mut self) {
        self.oam_dma.blocking = self.oam_dma.compute_blocking();
        let bus_cycles = if self.double_speed { 1 } else { 2 };
        // Accumulate PPU cycles lazily (flushed at if_reg check between halves)
        self.ppu_deferred += bus_cycles;
        self.tick_split(2, bus_cycles, 0);
        // OAM DMA step deferred to the next half or full M-cycle
    }

    /// Advance all bus components. `timer_cycles` is CPU-clock T-cycles (always 4 per M-cycle).
    /// `bus_cycles` is 4MHz-rate T-cycles (4 normal, 2 double-speed).
    pub fn tick(&mut self, timer_cycles: u32, bus_cycles: u32) {
        self.tick_split(timer_cycles, bus_cycles, bus_cycles);
    }

    /// Like tick() but with a separate PPU cycle count. Used for eager PPU
    /// ticking where PPU may get different cycles than other components.
    pub(super) fn tick_split(&mut self, timer_cycles: u32, bus_cycles: u32, ppu_cycles: u32) {
        // Capture DIV counter before and after timer step for serial/APU edge detection
        let old_div = self.timer.counter();
        self.timer.step(timer_cycles);
        let new_div = self.timer.counter();
        if self.timer.interrupt {
            self.if_ |= 0x04;
            self.timer.clear_interrupt();
        }

        // Serial clock is derived from DIV counter
        self.serial.step(old_div, new_div);
        if self.serial.interrupt {
            self.if_ |= 0x08;
            self.serial.interrupt = false;
        }

        // APU frame sequencer is clocked by DIV bit 12 (or 13 in double speed)
        let apu_bit: u16 = if self.double_speed { 0x2000 } else { 0x1000 };
        let triggers = old_div & !new_div; // bits that fell
        if triggers & apu_bit != 0 {
            self.apu.div_event();
        } else {
            let secondary = !old_div & new_div; // bits that rose
            if secondary & apu_bit != 0 {
                self.apu.div_secondary_event();
            }
        }
        self.apu.set_div_counter(new_div);

        if ppu_cycles > 0 {
            let ppu_flags = self.ppu.step(ppu_cycles);
            self.if_ |= ppu_flags;
        }

        self.apu.step(bus_cycles);

        if self.joypad.interrupt {
            self.if_ |= 0x10;
            self.joypad.clear_interrupt();
        }

        // Note: HBlank HDMA is handled by check_hdma_hblank() in tick_mcycle(),
        // not here. This avoids recursive tick() calls during DMA transfers.
    }
}
