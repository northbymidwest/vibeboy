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
            active: true,
            source: (source_page as u16) << 8,
            progress: 0,
            delay: 1,
            was_blocking,
            blocking: was_blocking,
            pending_write: None,
            last_bus_byte: 0xFF,
            bus_release_dots: 0,
        };
    }

    /// Advance OAM DMA by one M-cycle. Called from each M-cycle tick.
    ///
    /// Pipelined model: DMA reads a byte from the bus in one M-cycle,
    /// then writes it to OAM in the NEXT M-cycle. This matches the
    /// 1-cycle pipeline delay observed on real hardware.
    ///
    /// Timeline (progress values at step entry):
    ///   0: read byte 0, pending=(0, byte0)
    ///   1: write pending to OAM[0], read byte 1, pending=(1, byte1)
    ///   ...
    ///   159: write pending to OAM[158], read byte 159, pending=(159, byte159)
    ///   160: write pending to OAM[159] (flush), no read
    ///   161 (CGB): teardown — bus blocked, no transfer
    ///   End: DMG at 161, CGB at 162
    pub fn step_oam_dma(&mut self) {
        if !self.oam_dma.active {
            return;
        }
        if self.oam_dma.delay > 0 {
            self.oam_dma.delay -= 1;
            return;
        }

        // Write PREVIOUS cycle's pending byte to OAM (pipeline flush)
        if let Some((idx, byte)) = self.oam_dma.pending_write.take() {
            self.ppu.oam[idx] = byte;
        }

        // Read CURRENT byte from bus for next cycle
        if self.oam_dma.progress < 160 {
            let mut src = self.oam_dma.source + self.oam_dma.progress as u16;
            let byte = if self.model.is_cgb() && src >= 0xE000 {
                0xFF
            } else {
                if !self.model.is_cgb() && src >= 0xFE00 {
                    src -= 0x2000;
                }
                self.read_byte_raw(src)
            };
            self.oam_dma.pending_write = Some((self.oam_dma.progress as usize, byte));
            self.oam_dma.last_bus_byte = byte;
        }

        self.oam_dma.progress += 1;
        // Keep original end conditions — blocking timing must not change.
        // CGB teardown at progress=160 naturally flushes the last pending byte.
        // DMG: flush remaining pending byte at end.
        let end = if self.model.is_cgb() { 161 } else { 160 };
        if self.oam_dma.progress >= end {
            // Flush any remaining pending byte before deactivating
            if let Some((idx, byte)) = self.oam_dma.pending_write.take() {
                self.ppu.oam[idx] = byte;
            }
            self.oam_dma.active = false;
            // Bus release: OAM bus holds last DMA byte for 1 M-cycle
            let release = if self.double_speed { 2u8 } else { 4 };
            self.oam_dma.bus_release_dots = release;
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

    /// CPU M-cycles of HDMA/GDMA setup before the first byte moves.
    /// The setup is 8 CPU T-cycles in both speed modes (2 M-cycles normal,
    /// 2 "fast" M-cycles double speed).
    const HDMA_SETUP_MCYCLES: u32 = 2;

    /// Bytes copied per CPU M-cycle. Per Pan Docs (CGB Registers, "Bit 7 = 0:
    /// General-Purpose DMA"), a 16-byte block takes 8 M-cycles in normal
    /// speed and 16 fast M-cycles in double speed, i.e. 32 dots (about 8 us)
    /// in both modes: the DMA engine moves bytes at a fixed 2 MHz, so a
    /// double-speed CPU M-cycle only covers one byte.
    fn hdma_bytes_per_mcycle(&self) -> u16 {
        if self.double_speed { 1 } else { 2 }
    }

    /// GDMA: transfer all blocks with per-M-cycle subsystem interleaving.
    ///
    /// Each CPU M-cycle ticks DIV/Timer/Serial/APU/PPU/OAM DMA exactly like a
    /// normal CPU M-cycle so that events (APU frame sequencer, timer overflow,
    /// mode changes, etc.) fire at the correct time during the transfer.
    fn do_gdma(&mut self, blocks: u8) {
        self.hdma.in_transfer = true;
        for _ in 0..Self::HDMA_SETUP_MCYCLES {
            self.tick_dma_mcycle();
        }
        let mut mcycles = Self::HDMA_SETUP_MCYCLES;
        for _ in 0..blocks {
            mcycles += self.do_hdma_block_interleaved();
        }
        self.hdma.in_transfer = false;
        // CPU halt cycles (CPU T-cycles, 4 per CPU M-cycle in either speed)
        self.dma_halt_cycles += mcycles * 4;
    }

    /// Copy one 16-byte HDMA block, ticking all subsystems once per CPU
    /// M-cycle. Returns the number of CPU M-cycles consumed.
    fn do_hdma_block_interleaved(&mut self) -> u32 {
        let step = self.hdma_bytes_per_mcycle();
        let mut mcycles = 0;
        for byte_off in (0..16u16).step_by(step as usize) {
            for i in 0..step {
                let src_addr = self.hdma.src.wrapping_add(byte_off + i);
                let dst_addr = self.hdma.dst.wrapping_add(byte_off + i);
                let b = self.read_byte_raw(src_addr);
                self.ppu.write_vram(dst_addr, b);
            }
            self.tick_dma_mcycle();
            mcycles += 1;
        }
        self.hdma.src = self.hdma.src.wrapping_add(16);
        self.hdma.dst = self.hdma.dst.wrapping_add(16);
        let dst_off = (self.hdma.dst.wrapping_sub(0x8000)) & 0x1FFF;
        self.hdma.dst = 0x8000 + dst_off;
        mcycles
    }

    /// Tick all subsystems for one CPU M-cycle during a DMA transfer (timer
    /// 4 CPU T-cycles, PPU/APU 4 dots normal speed or 2 double speed, the
    /// same split as tick_mcycle). Also advances OAM DMA, which runs at the
    /// CPU clock (one byte per CPU M-cycle) independently of HDMA.
    fn tick_dma_mcycle(&mut self) {
        let bus_cycles = if self.double_speed { 2 } else { 4 };
        self.oam_dma.blocking = self.oam_dma.compute_blocking();
        self.tick(4, bus_cycles);
        self.step_oam_dma();
    }

    // ── Tick: advance all components by T-cycles ──────────────────────────────

    /// Tick only the PPU for one M-cycle during CGB speed switch idle.
    /// Per Pan Docs, DIV resets and stops ticking during the STOP halt;
    /// timer, serial, and APU do not advance. PPU continues rendering.
    pub fn tick_speed_switch_idle(&mut self) {
        let bus_cycles = if self.double_speed { 2u32 } else { 4 };
        self.sync_ppu_dma_bus_byte();
        let ppu_flags = self.ppu.step(bus_cycles);
        self.if_ |= ppu_flags;
    }

    /// Check if a prior PPU flush detected mode 0 entry and trigger
    /// HDMA mode 1 transfer. Called from each M-cycle tick (after the CPU access)
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
        self.hdma.in_transfer = true;
        for _ in 0..Self::HDMA_SETUP_MCYCLES {
            self.tick_dma_mcycle();
        }
        let mcycles = Self::HDMA_SETUP_MCYCLES + self.do_hdma_block_interleaved();
        self.hdma.in_transfer = false;
        // CPU halt cycles (CPU T-cycles, 4 per CPU M-cycle in either speed)
        self.dma_halt_cycles += mcycles * 4;

        self.hdma.blocks -= 1;
        if self.hdma.blocks == 0 {
            self.hdma.active = false;
        }
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
            self.sync_ppu_dma_bus_byte();
            let ppu_flags = self.ppu.step(ppu_cycles);
            self.if_ |= ppu_flags;
        }

        self.apu.step(bus_cycles);

        if self.joypad.interrupt {
            self.if_ |= 0x10;
            self.joypad.clear_interrupt();
        }

        // Note: HBlank HDMA is handled by check_hdma_hblank() in the M-cycle tick,
        // not here. This avoids recursive tick() calls during DMA transfers.
    }
}
