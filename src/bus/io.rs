/// I/O register read/write dispatch (0xFF00-0xFF7F).

use super::Bus;

impl Bus {
    pub(super) fn read_io(&self, addr: u16) -> u8 {
        match addr {
            0xFF00 => {
                if let Some(ref sgb) = self.sgb {
                    if sgb.player_count > 1 {
                        // When both select lines high, return player ID
                        let p1_select = self.joypad.read() & 0x30;
                        if p1_select == 0x30 {
                            return 0xC0 | 0x30 | sgb.read_p1_id();
                        }
                    }
                }
                self.joypad.read()
            }
            0xFF01 => self.serial.sb,
            0xFF02 => self.serial.read_sc(),
            0xFF03 => 0xFF,
            0xFF04..=0xFF07 => self.timer.read(addr),
            0xFF0F => self.if_ | 0xE0,
            0xFF10..=0xFF3F => self.apu.read(addr),
            0xFF40..=0xFF4B => self.ppu.read(addr),
            0xFF4D => {
                if !self.model.is_cgb() || self.dmg_compat { return 0xFF; }
                self.key1 | 0x7E
            }
            // VBK, BGPI, OBPI: accessible on CGB even in DMG-compat mode
            0xFF4F | 0xFF68 | 0xFF6A => {
                if !self.model.is_cgb() { return 0xFF; }
                self.ppu.read(addr)
            }
            // BGPD, OBPD: blocked in DMG-compat mode
            0xFF69 | 0xFF6B => {
                if !self.model.is_cgb() || self.dmg_compat { return 0xFF; }
                self.ppu.read(addr)
            }
            0xFF51..=0xFF54 => 0xFF, // HDMA src/dst are write-only
            0xFF55 => {
                if !self.model.is_cgb() || self.dmg_compat { return 0xFF; }
                // Bit 7: 0 = active, 1 = not active
                // Bits 0-6: remaining blocks minus 1
                let remaining = self.hdma.blocks.wrapping_sub(1) & 0x7F;
                if self.hdma.active {
                    remaining
                } else {
                    0x80 | remaining
                }
            }
            0xFF50 => if self.boot_rom_active { 0xFE } else { 0xFF },
            0xFF70 => {
                if !self.model.is_cgb() || self.dmg_compat { return 0xFF; }
                self.wram_bank as u8 | 0xF8
            }
            0xFF6C => if self.model.is_cgb() { self.ppu.opri | 0xFE } else { 0xFF },
            0xFF72 => if self.model.is_cgb() { self.ff72 } else { 0xFF },
            0xFF73 => if self.model.is_cgb() { self.ff73 } else { 0xFF },
            0xFF74 => if self.model.is_cgb() { self.ff74 } else { 0xFF },
            0xFF75 => if self.model.is_cgb() { self.ff75 | 0x8F } else { 0xFF },
            0xFF76 => if self.model.is_cgb() { self.apu.pcm12() } else { 0xFF },
            0xFF77 => if self.model.is_cgb() { self.apu.pcm34() } else { 0xFF },
            _ => 0xFF,
        }
    }

    pub(super) fn write_io(&mut self, addr: u16, val: u8) {
        match addr {
            0xFF00 => {
                self.joypad.write(val);
                if let Some(ref mut sgb) = self.sgb {
                    sgb.write_p1(val);
                }
            }
            0xFF01 => self.serial.sb = val,
            0xFF02 => {
                self.serial.write_sc(val);
            }
            0xFF04..=0xFF07 => {
                let old_div = self.timer.counter();
                self.timer.write(addr, val);
                let new_div = self.timer.counter();
                if self.timer.interrupt {
                    self.if_ |= 0x04;
                    self.timer.clear_interrupt();
                }
                // DIV reset can create falling edge for serial clock
                self.serial.step(old_div, new_div);
                if self.serial.interrupt {
                    self.if_ |= 0x08;
                    self.serial.interrupt = false;
                }
                // APU: DIV write can create falling/rising edge on APU bit
                if addr == 0xFF04 {
                    let apu_bit: u16 = if self.double_speed { 0x2000 } else { 0x1000 };
                    let triggers = old_div & !new_div;
                    if triggers & apu_bit != 0 {
                        self.apu.div_event();
                    } else {
                        let secondary = !old_div & new_div;
                        if secondary & apu_bit != 0 {
                            self.apu.div_secondary_event();
                        }
                    }
                    self.apu.set_div_counter(new_div);
                }
            }
            0xFF0F => self.if_ = val | 0xE0,
            0xFF10..=0xFF3F => {
                self.apu.set_div_counter(self.timer.counter());
                self.apu.set_double_speed(self.double_speed);
                self.apu.write(addr, val);
            }
            0xFF46 => self.start_oam_dma(val),
            // BGPD ($FF69), OBPD ($FF6B): ignore in DMG-compat mode
            0xFF69 | 0xFF6B if !self.model.is_cgb() || self.dmg_compat => {}
            // VBK, BGPI, OBPI: ignore on non-CGB only (accessible in compat)
            0xFF4F | 0xFF68 | 0xFF6A if !self.model.is_cgb() => {}
            // CGB LCDC write: handle tile_sel_glitch when TILE_SEL transitions 1→0
            0xFF40 if self.model.is_cgb() && !self.double_speed => {
                self.flush_ppu_deferred();
                let old_lcdc = self.ppu.lcdc;
                self.ppu.write(addr, val);
                if self.ppu.if_flags != 0 {
                    self.if_ |= self.ppu.if_flags;
                    self.ppu.if_flags = 0;
                }
                // TILE_SEL (bit 4) transition 1→0: 1T glitch window
                if (old_lcdc & 0x10) != 0 && (val & 0x10) == 0 {
                    self.ppu.tile_sel_glitch = true;
                    self.ppu.step(1);
                    self.ppu.tile_sel_glitch = false;
                    self.ppu_tick_debt += 1;
                    if self.ppu.if_flags != 0 {
                        self.if_ |= self.ppu.if_flags;
                        self.ppu.if_flags = 0;
                    }
                }
            }
            // CGB palette writes: write takes effect 2T early (no OR glitch).
            // Hardware: advance(pending-2), write, pending=6.
            0xFF47..=0xFF49 if self.model.is_cgb() && !self.double_speed => {
                if self.ppu_deferred > 2 {
                    let flush = self.ppu_deferred - 2;
                    let flags = self.ppu.step(flush);
                    self.if_ |= flags;
                    self.ppu_deferred = 2;
                }
                self.ppu.write(addr, val);
                if self.ppu.if_flags != 0 {
                    self.if_ |= self.ppu.if_flags;
                    self.ppu.if_flags = 0;
                }
            }
            // DMG palette writes: -2T conflict with bus glitch (old|new at T3)
            // Hardware timing: 2T old → 1T (old|new) glitch → remaining T with new
            0xFF47..=0xFF49 if !self.model.is_cgb() => {
                // Flush all deferred ticks except the current M-cycle's 4T
                if self.ppu_deferred > 4 {
                    let catchup = self.ppu_deferred - 4;
                    let flags = self.ppu.step(catchup);
                    self.if_ |= flags;
                    self.ppu_deferred = 4;
                }
                if self.ppu_deferred >= 4 {
                    // Flush 2T with old palette value
                    let flags = self.ppu.step(2);
                    self.if_ |= flags;
                    self.ppu_deferred -= 2;
                    // Set glitch palette (old | new) and tick 1T
                    let old_val = match addr {
                        0xFF47 => self.ppu.bgp_rendering,
                        0xFF48 => self.ppu.obp0_rendering,
                        _ => self.ppu.obp1_rendering,
                    };
                    let glitch = old_val | val;
                    match addr {
                        0xFF47 => self.ppu.bgp_rendering = glitch,
                        0xFF48 => self.ppu.obp0_rendering = glitch,
                        _ => self.ppu.obp1_rendering = glitch,
                    }
                    let flags = self.ppu.step(1);
                    self.if_ |= flags;
                    self.ppu_deferred -= 1;
                } else {
                    // Not enough deferred ticks for full conflict; flush all
                    self.flush_ppu_deferred();
                }
                // Write real value (remaining deferred tick uses it)
                self.ppu.write(addr, val);
                if self.ppu.if_flags != 0 {
                    self.if_ |= self.ppu.if_flags;
                    self.ppu.if_flags = 0;
                }
            }
            // DMG SCY: READ_NEW conflict (hardware: advance(pending-1), write, pending=5)
            // Flush all but 1T with old value, then write new value.
            0xFF42 if !self.model.is_cgb() => {
                if self.ppu_deferred > 1 {
                    let flush = self.ppu_deferred - 1;
                    let flags = self.ppu.step(flush);
                    self.if_ |= flags;
                    self.ppu_deferred = 1;
                }
                self.ppu.write(addr, val);
                if self.ppu.if_flags != 0 {
                    self.if_ |= self.ppu.if_flags;
                    self.ppu.if_flags = 0;
                }
            }
            // DMG/CGB-double SCX: SCX_DMG conflict (write takes effect 2T early)
            // Hardware: advance(pending-2), write, pending=6.
            0xFF43 if !self.model.is_cgb() || self.double_speed => {
                // Flush all but 2T with old SCX value
                if self.ppu_deferred > 2 {
                    let flush = self.ppu_deferred - 2;
                    let flags = self.ppu.step(flush);
                    self.if_ |= flags;
                    self.ppu_deferred = 2;
                }
                self.ppu.write(addr, val);
                if self.ppu.if_flags != 0 {
                    self.if_ |= self.ppu.if_flags;
                    self.ppu.if_flags = 0;
                }
            }
            // DMG LCDC: conflict handler with glitch timing
            // Hardware: advance(pending-2), write glitch(old | (new & BG_EN)), advance(1), write real, pending=5
            0xFF40 if !self.model.is_cgb() => {
                // Flush all deferred ticks except the current M-cycle's 4T,
                // so we check mode with up-to-date PPU state.
                if self.ppu_deferred > 4 {
                    let catchup = self.ppu_deferred - 4;
                    let flags = self.ppu.step(catchup);
                    self.if_ |= flags;
                    self.ppu_deferred = 4;
                }
                let in_mode3 = self.ppu.mode == 3;
                if in_mode3 && self.ppu_deferred >= 4 {
                    // OBJ_EN takes effect immediately when cleared —
                    // apply it before the 2T pre-write ticks so sprite
                    // pixels are suppressed during the entire write sequence.
                    let _saved_obj_en = self.ppu.lcdc & 0x02;
                    if (val & 0x02) == 0 {
                        self.ppu.lcdc &= !0x02;
                    }
                    // 2T old (with OBJ_EN already cleared if disabling)
                    let flags = self.ppu.step(2);
                    self.if_ |= flags;
                    self.ppu_deferred -= 2;
                    let old_lcdc = self.ppu.lcdc;
                    // Restore OBJ_EN for glitch calculation
                    // (the glitch uses the modified old_lcdc)
                    // Glitch LCDC: old | (new & BG_EN bit)
                    let glitch = old_lcdc | (val & 0x01);
                    let saved_lcdc = self.ppu.lcdc;
                    self.ppu.lcdc = glitch;
                    let flags = self.ppu.step(1);
                    self.if_ |= flags;
                    self.ppu_deferred -= 1;
                    self.ppu.lcdc = saved_lcdc;
                    // Window disable during window fetch: set glitch flag
                    if (saved_lcdc & 0x20) != 0 && (val & 0x20) == 0
                        && self.ppu.fetcher_is_window()
                    {
                        self.ppu.disable_window_pixel_insertion_glitch = true;
                    }
                } else {
                    self.flush_ppu_deferred();
                }
                self.ppu.write(addr, val);
                if self.ppu.if_flags != 0 {
                    self.if_ |= self.ppu.if_flags;
                    self.ppu.if_flags = 0;
                }
            }
            // TODO: CGB STAT needs split-write conflict handler (LYC enable bit
            // transitions 1T later in normal speed, HBlank enable bit in double
            // speed). Requires PPU-internal support for partial STAT writes
            // without triggering the full write handler's IRQ logic.
            // CGB LYC (normal speed): WRITE_CPU — write takes effect 1T later.
            // Hardware: advance(pending+1), write, pending=3.
            0xFF45 if self.model.is_cgb() && !self.double_speed => {
                self.flush_ppu_deferred();
                let flags = self.ppu.step(1);
                self.if_ |= flags;
                self.ppu.write(addr, val);
                if self.ppu.if_flags != 0 {
                    self.if_ |= self.ppu.if_flags;
                    self.ppu.if_flags = 0;
                }
                self.ppu_tick_debt += 1;
            }
            // DMG WX: READ_OLD + wx_just_changed flag for 1T after write
            0xFF4B if !self.model.is_cgb() => {
                self.flush_ppu_deferred();
                self.ppu.write(addr, val);
                if self.ppu.if_flags != 0 {
                    self.if_ |= self.ppu.if_flags;
                    self.ppu.if_flags = 0;
                }
                // Tick 1T with wx_just_changed to suppress window trigger
                self.ppu.wx_just_changed = true;
                let flags = self.ppu.step(1);
                self.if_ |= flags;
                self.ppu.wx_just_changed = false;
                self.ppu_tick_debt += 1;
            }
            // Default PPU registers: READ_OLD (flush all deferred, then write)
            0xFF40..=0xFF45 | 0xFF47..=0xFF4B | 0xFF4F | 0xFF68..=0xFF6B => {
                self.flush_ppu_deferred();
                self.ppu.write(addr, val);
                if self.ppu.if_flags != 0 {
                    self.if_ |= self.ppu.if_flags;
                    self.ppu.if_flags = 0;
                }
            }
            0xFF4D if !self.model.is_cgb() || self.dmg_compat => {} // ignore on DMG/compat
            0xFF4D => {
                // KEY1: prepare speed switch (bit 0 = switch request)
                self.key1 = (self.key1 & 0x80) | (val & 0x01);
            }
            0xFF51..=0xFF55 if !self.model.is_cgb() || self.dmg_compat => {} // ignore on DMG/compat
            0xFF51 => self.hdma.src = (self.hdma.src & 0x00FF) | ((val as u16) << 8),
            0xFF52 => self.hdma.src = (self.hdma.src & 0xFF00) | ((val & 0xF0) as u16),
            0xFF53 => self.hdma.dst = (self.hdma.dst & 0x00FF) | (((val & 0x1F) as u16) << 8) | 0x8000,
            0xFF54 => self.hdma.dst = (self.hdma.dst & 0xFF00) | ((val & 0xF0) as u16),
            0xFF55 => self.start_hdma(val),
            0xFF50 => {
                // Writing any non-zero value permanently disables the boot ROM.
                if val != 0 {
                    self.boot_rom_active = false;
                    // Activate SGB protocol now that boot ROM is done
                    if let Some(ref mut sgb) = self.sgb {
                        sgb.protocol_active = true;
                    }
                    // Detect DMG compat: CGB hardware running DMG game
                    let cgb_flag = self.cart.read_rom(0x0143);
                    if self.model.is_cgb() && cgb_flag != 0x80 && cgb_flag != 0xC0 {
                        self.ppu.dmg_compat = true;
                        self.dmg_compat = true;
                        self.serial.cgb_mode = false;
                        // Capture current CGB palette RAM as reference colors
                        // (the boot ROM has programmed these)
                        for i in 0..4 {
                            let off = i * 2;
                            self.ppu.dmg_bg_ref[i] = self.ppu.bcpd[off] as u16
                                | ((self.ppu.bcpd[off + 1] as u16) << 8);
                            self.ppu.dmg_obj_ref[0][i] = self.ppu.ocpd[off] as u16
                                | ((self.ppu.ocpd[off + 1] as u16) << 8);
                            self.ppu.dmg_obj_ref[1][i] = self.ppu.ocpd[8 + off] as u16
                                | ((self.ppu.ocpd[8 + off + 1] as u16) << 8);
                        }
                    }
                }
            }
            0xFF70 if !self.model.is_cgb() || self.dmg_compat => {} // ignore on DMG/compat
            0xFF70 => {
                let bank = (val & 0x07) as usize;
                self.wram_bank = if bank == 0 { 1 } else { bank };
            }
            0xFF6C if self.model.is_cgb() => { self.ppu.opri = val & 0x01; }
            0xFF72 if self.model.is_cgb() => { self.ff72 = val; }
            0xFF73 if self.model.is_cgb() => { self.ff73 = val; }
            0xFF74 if self.model.is_cgb() => { self.ff74 = val; }
            0xFF75 if self.model.is_cgb() => { self.ff75 = val & 0x70; }
            _ => {}
        }
    }
}
