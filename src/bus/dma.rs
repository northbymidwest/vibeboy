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

        if mode == 1 {
            // H-Blank DMA: clear stale hblank_entered from the current M-cycle
            self.ppu.hblank_entered = false;
        }

        if mode == 0 {
            // General purpose DMA: transfer all blocks, ticking the bus
            self.do_gdma(blocks);
            self.hdma.active = false;
        }
        // H-Blank DMA: transfer one block per HBlank, handled in tick()
    }

    /// GDMA: transfer all blocks immediately, ticking the bus.
    /// Setup: 2 bus M-cycles (normal) or 1 bus M-cycle (DS).
    /// Transfer: 8 bus M-cycles per block.
    fn do_gdma(&mut self, blocks: u8) {
        let ds = self.double_speed;
        self.hdma.in_transfer = true;
        // Setup overhead
        if ds {
            self.tick(8, 4);
        } else {
            self.tick(4, 4);
            self.tick(4, 4);
        }
        for _ in 0..blocks {
            self.do_hdma_block_ticked();
        }
        self.hdma.in_transfer = false;
        // CPU halt: 8T setup + N * 8 * cpu_t_per_bus_m
        let cpu_t_per_bus_m: u32 = if ds { 8 } else { 4 };
        self.dma_halt_cycles += 8 + blocks as u32 * 8 * cpu_t_per_bus_m;
    }

    /// Transfer one 16-byte HDMA block with bus ticking (2 bytes per bus M-cycle).
    fn do_hdma_block_ticked(&mut self) {
        let ds = self.double_speed;
        for byte_off in (0..16u16).step_by(2) {
            let src_addr = self.hdma.src.wrapping_add(byte_off);
            let dst_addr = self.hdma.dst.wrapping_add(byte_off);
            let b0 = self.read_byte_raw(src_addr);
            self.ppu.write_vram(dst_addr, b0);
            let b1 = self.read_byte_raw(src_addr + 1);
            self.ppu.write_vram(dst_addr + 1, b1);
            if ds {
                self.tick(8, 4);
            } else {
                self.tick(4, 4);
            }
        }
        self.hdma.src = self.hdma.src.wrapping_add(16);
        self.hdma.dst = self.hdma.dst.wrapping_add(16);
        let dst_off = (self.hdma.dst.wrapping_sub(0x8000)) & 0x1FFF;
        self.hdma.dst = 0x8000 + dst_off;
    }

    // ── Tick: advance all components by T-cycles ──────────────────────────────

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
                let ds = self.double_speed;
                self.hdma.in_transfer = true;
                // Setup: 2 bus M-cycles (normal) or 1 (DS)
                if ds {
                    self.tick(8, 4);
                } else {
                    self.tick(4, 4);
                    self.tick(4, 4);
                }
                self.do_hdma_block_ticked();
                self.hdma.in_transfer = false;
                let cpu_t_per_bus_m: u32 = if ds { 8 } else { 4 };
                self.dma_halt_cycles += 8 + 8 * cpu_t_per_bus_m;
                self.hdma.blocks -= 1;
                if self.hdma.blocks == 0 {
                    self.hdma.active = false;
                }
            }
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

    /// Like tick() but with a separate PPU cycle count. Used by the lazy PPU
    /// tick model where PPU gets fewer immediate ticks than other components.
    fn tick_split(&mut self, timer_cycles: u32, bus_cycles: u32, ppu_cycles: u32) {
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
        self.apu.set_double_speed(self.double_speed);

        if ppu_cycles > 0 {
            let ppu_flags = self.ppu.step(ppu_cycles);
            self.if_ |= ppu_flags;
        }

        self.apu.step(bus_cycles);

        if self.joypad.interrupt {
            self.if_ |= 0x10;
            self.joypad.clear_interrupt();
        }

        // H-Blank HDMA: transfer one block each time the PPU enters Mode 0
        // Skip during active DMA transfer to prevent re-entrant cascading
        if self.ppu.hblank_entered && !self.hdma.in_transfer {
            self.ppu.hblank_entered = false;
            if self.hdma.active && self.hdma.mode == 1 {
                let ds = self.double_speed;
                self.hdma.in_transfer = true;
                // Setup: 2 bus M-cycles (normal) or 1 (DS)
                if ds {
                    self.tick(8, 4);
                } else {
                    self.tick(4, 4);
                    self.tick(4, 4);
                }
                self.do_hdma_block_ticked();
                self.hdma.in_transfer = false;
                // CPU halt: 8T setup + 8 bus M-cycles per block
                let cpu_t_per_bus_m: u32 = if ds { 8 } else { 4 };
                self.dma_halt_cycles += 8 + 8 * cpu_t_per_bus_m;
                self.hdma.blocks -= 1;
                if self.hdma.blocks == 0 {
                    self.hdma.active = false;
                }
            }
        }
    }
}
