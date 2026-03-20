/// PPU I/O register read/write, VRAM/OAM access, palette sync.

use super::Ppu;

impl Ppu {
    // ---- I/O Register Access ----

    pub fn read(&self, addr: u16) -> u8 {
        match addr {
            0xFF40 => self.lcdc,
            0xFF41 => self.stat | 0x80,
            0xFF42 => self.scy,
            0xFF43 => self.scx,
            0xFF44 => self.visible_ly,
            0xFF45 => self.pending_lyc.unwrap_or(self.lyc),
            0xFF46 => self.dma,
            0xFF47 => self.bgp,
            0xFF48 => self.obp0,
            0xFF49 => self.obp1,
            0xFF4A => self.wy,
            0xFF4B => self.wx,
            0xFF4F => self.vram_bank as u8 | 0xFE,
            0xFF68 => self.bcps | 0x40,
            0xFF69 => {
                if self.cgb_palettes_blocked { 0xFF }
                else { self.bcpd[(self.bcps & 0x3F) as usize] }
            }
            0xFF6A => self.ocps | 0x40,
            0xFF6B => {
                if self.cgb_palettes_blocked { 0xFF }
                else { self.ocpd[(self.ocps & 0x3F) as usize] }
            }
            _ => 0xFF,
        }
    }

    /// In DMG compat mode, sync a DMG palette register write to CGB palette RAM.
    /// Maps each 2-bit shade entry through the reference colors.
    pub fn sync_dmg_palette_to_cgb(&mut self, palette_reg: u8, is_obj: bool, pal_idx: usize) {
        for color_idx in 0..4 {
            let shade = (palette_reg >> (color_idx * 2)) & 0x03;
            let rgb555 = if is_obj {
                self.dmg_obj_ref[pal_idx][shade as usize]
            } else {
                self.dmg_bg_ref[shade as usize]
            };
            let lo = (rgb555 & 0xFF) as u8;
            let hi = (rgb555 >> 8) as u8;
            let offset = pal_idx * 8 + color_idx * 2;
            if is_obj {
                self.ocpd[offset] = lo;
                self.ocpd[offset + 1] = hi;
            } else {
                self.bcpd[offset] = lo;
                self.bcpd[offset + 1] = hi;
            }
        }
    }

    pub fn fetcher_is_window(&self) -> bool { self.fetcher.fetching_window }

    pub fn write(&mut self, addr: u16, val: u8) {
        match addr {
            0xFF40 => {
                let lcd_was_on = self.lcdc & 0x80 != 0;
                let win_was_on = self.lcdc & 0x20 != 0;
                self.lcdc = val;
                let lcd_now_on = self.lcdc & 0x80 != 0;
                let win_now_on = self.lcdc & 0x20 != 0;

                // Window enable toggled on: check WY condition
                if !win_was_on && win_now_on && lcd_now_on && self.ly == self.wy {
                    self.wy_triggered = true;
                }

                // Window enable toggled off during mode 3: deactivate window
                // immediately so the PPU switches back to BG tiles.
                // Clear deferred window trigger if window was just disabled
                if win_was_on && !win_now_on {
                    self.window_trigger_pending = false;
                    self.window_trigger_from_wx_write = false;
                }
                if win_was_on && !win_now_on && self.window_active && self.mode == 3 {
                    // DMG glitch: disabling window while fetcher is in window mode
                    // suppresses phantom window pixel insertion
                    if !self.cgb_mode && self.fetcher.fetching_window {
                        self.disable_window_pixel_insertion_glitch = true;
                    }
                    self.window_active = false;
                    self.fetcher.reset(false);
                    self.bg_fifo.clear();
                }

                // Window re-enabled during mode 3 after being recently deactivated:
                // defer reactivation to the next step boundary
                // Window re-enabled during mode 3: check if trigger point
                // was just passed (off-by-1 from mid-M-cycle write timing)
                if !win_was_on && win_now_on && self.mode == 3
                    && !self.window_active && !self.window_trigger_pending
                    && self.wy_triggered && self.position_in_line >= 0
                {
                    let wx_screen = if self.wx >= 7 { self.wx - 7 } else { 0 };
                    let tolerance = if self.double_speed { 0 } else { 1 };
                    let px = self.position_in_line as u8;
                    if px >= wx_screen && px <= wx_screen + tolerance {
                        self.window_trigger_pending = true;
                        self.window_trigger_from_wx_write = true;
                    }
                }


                if lcd_was_on && !lcd_now_on {
// LCD off: reset LY, dot, mode; preserve coincidence bit
                    // Do NOT reset stat_irq_line — hardware preserves the IRQ signal state
                    self.ly = 0;
                    self.visible_ly = 0;
                    self.ly_for_comparison = 0;
                    self.line_start_pending = false;
                    self.line_start_is_vblank = false;
                    self.mode_for_interrupt = -1;
                    self.line_153_phase = 0;
                    self.accessed_oam_row = 0xFF;
                    self.dot = 0;
                    self.mode = 0;
                    self.stat = (self.stat & !0x03) | (self.stat & 0x04); // keep bit2
                    self.oam_accessible = true;
                    self.oam_write_accessible = true;
                    self.vram_accessible = true;
                    self.vram_write_accessible = true;
                    for p in self.frame_buffer.iter_mut() {
                        *p = 0x00FFFFFF;
                    }
                } else if !lcd_was_on && lcd_now_on {
// LCD on: start at line 0, mode reads as 0 initially
                    // DMG first line is ~449T (7T shorter): 1T initial sleep + 8T
                    // phantom cycles_for_line adjustment that shortens Mode 0.
                    self.ly = 0;
                    self.visible_ly = 0;
                    self.ly_for_comparison = 0;
                    self.line_start_pending = false;
                    self.line_start_is_vblank = false;
                    self.mode_for_interrupt = -1;
                    self.line_153_phase = 0;
                    self.accessed_oam_row = 0xFF;
                    self.dot = 0;
                    self.mode = 0;
                    self.stat = self.stat & !0x03; // mode bits = 0
                    self.oam_accessible = true;
                    self.oam_write_accessible = true;
                    self.vram_accessible = true;
                    self.vram_write_accessible = true;
                    self.lcd_first_line = true;
                    self.lcd_first_line_short = false;
                    self.lcd_first_frame = true;
                    self.total_ticks = 0;
                    self.window_line_counter = 0;
                    // Check WY trigger for first line (LY=0)
                    self.wy_triggered = self.lcdc & 0x20 != 0 && self.ly == self.wy;
                    self.update_coincidence();
                    self.update_stat_irq();
                }
            }
            0xFF41 => {
                // Lower 3 bits (mode flags + coincidence) are read-only
                self.stat = (self.stat & 0x07) | (val & 0x78);
                if self.lcdc & 0x80 != 0 {
                    if self.cgb_mode {
                        // CGB: STAT writes re-evaluate IRQ, but mode sources
                        // (bits 3-5) are suppressed during mode 2/3. Only LYC
                        // source (bit 6) triggers during mode 2/3.
                        self.update_stat_irq_on_write();
                    } else {
                        // DMG STAT write glitch (Pan Docs): behaves as if $FF
                        // is written for one cycle, then the real value takes
                        // effect. Evaluate the IRQ signal with all enable bits
                        // temporarily set — any mode match or LYC coincidence
                        // produces a rising edge. Then re-evaluate with the
                        // real written value to set stat_irq_line correctly.
                        if (self.mode == 0 || self.mode == 1)
                            && !self.stat_irq_line
                        {
                            // With $FF: all mode enables set. Check if
                            // mode_for_interrupt matches ANY mode source.
                            let mode_match = self.mode_for_interrupt >= 0
                                && self.mode_for_interrupt <= 2;
                            // With $FF: LYC enable (bit 6) set. Check
                            // coincidence flag (STAT bit 2, read-only).
                            let lyc_match = self.stat & 0x04 != 0;
                            if mode_match || lyc_match {
                                self.if_flags |= 0x02;
                            }
                        }
                        self.update_stat_irq();
                    }
                }
            }
            0xFF42 => self.scy = val,
            0xFF43 => self.scx = val,
            0xFF44 => {} // LY is read-only
            0xFF45 => {
                if self.cgb_mode && self.line_start_pending {
                    // Defer until line-start handler completes, so the
                    // coincidence check at dot 3/4 uses the old LYC value.
                    self.pending_lyc = Some(val);
                } else if self.cgb_mode && self.line_153_phase > 0 && self.dot >= 8 {
                    // After LY=0 in line_153: defer so dot 12 coincidence
                    // uses the old LYC value.
                    self.pending_lyc = Some(val);
                } else if self.cgb_mode && (
                    self.line_153_phase > 0 ||
                    ((self.mode == 0 || self.mode == 1) && self.dot >= 452)
                ) {
                    // During line_153_phase (before dot 8) or near line end:
                    // apply value but suppress STAT IRQ (display_state 15/16
                    // skip behavior).
                    // Silently update stat_irq_line to prevent false rising
                    // edge at the next real update_stat_irq() call.
                    self.lyc = val;
                    if self.lcdc & 0x80 != 0 {
                        self.update_coincidence();
                        self.update_stat_irq_silent();
                    }
                } else {
                    self.lyc = val;
                    if self.lcdc & 0x80 != 0 {
                        self.update_coincidence();
                        self.update_stat_irq();
                    }
                }
            }
            0xFF46 => self.dma = val,
            0xFF47 => {
                self.bgp = val;
                self.bgp_rendering = val;
                if self.dmg_compat { self.sync_dmg_palette_to_cgb(val, false, 0); }
            }
            0xFF48 => {
                self.obp0 = val;
                self.obp0_rendering = val;
                if self.dmg_compat { self.sync_dmg_palette_to_cgb(val, true, 0); }
            }
            0xFF49 => {
                self.obp1 = val;
                self.obp1_rendering = val;
                if self.dmg_compat { self.sync_dmg_palette_to_cgb(val, true, 1); }
            }
            0xFF4A => {
                self.wy = val;
                // WY comparison runs continuously until the PPU has started
                // outputting visible pixels (position_in_line > 0 means at
                // least one visible pixel has been output).
                let can_trigger = self.mode != 3 || self.position_in_line <= 0;
                if self.lcdc & 0x20 != 0 && self.ly == val && can_trigger {
                    self.wy_triggered = true;
                }
            }
            0xFF4B => {
                self.wx = val;
                if self.mode == 3 && !self.window_active && !self.window_trigger_pending
                    && self.lcdc & 0x20 != 0 && self.wy_triggered && self.position_in_line >= 0
                {
                    let wx_screen = if val >= 7 { val - 7 } else { 0 };
                    // Allow trigger if position just passed the WX point (off-by-1
                    // from mid-M-cycle write timing in normal speed)
                    let tolerance = if self.double_speed { 0 } else { 1 };
                    let px = self.position_in_line as u8;
                    if px >= wx_screen && px <= wx_screen + tolerance {
                        self.window_trigger_pending = true;
                        self.window_trigger_from_wx_write = true;
                    }
                }
            }
            0xFF4F => self.vram_bank = (val & 0x01) as usize,
            0xFF68 => self.bcps = val & 0xBF,
            0xFF69 => {
                let idx = (self.bcps & 0x3F) as usize;
                if !self.cgb_palettes_blocked {
                    self.bcpd[idx] = val;
                }
                // Auto-increment always advances, even when palette write is blocked
                if self.bcps & 0x80 != 0 {
                    let next = (idx + 1) & 0x3F;
                    self.bcps = (self.bcps & 0x80) | next as u8;
                }
            }
            0xFF6A => self.ocps = val & 0xBF,
            0xFF6B => {
                let idx = (self.ocps & 0x3F) as usize;
                if !self.cgb_palettes_blocked {
                    self.ocpd[idx] = val;
                }
                if self.ocps & 0x80 != 0 {
                    let next = (idx + 1) & 0x3F;
                    self.ocps = (self.ocps & 0x80) | next as u8;
                }
            }
            _ => {}
        }
    }

    pub fn read_vram(&self, addr: u16) -> u8 {
        self.vram[self.vram_bank][(addr - 0x8000) as usize]
    }

    pub fn write_vram(&mut self, addr: u16, val: u8) {
        self.vram[self.vram_bank][(addr - 0x8000) as usize] = val;
    }

    pub fn read_oam(&self, addr: u16) -> u8 {
        self.oam[(addr - 0xFE00) as usize]
    }

    pub fn write_oam(&mut self, addr: u16, val: u8) {
        self.oam[(addr - 0xFE00) as usize] = val;
    }

    pub fn frame_buffer(&self) -> &[u32] {
        &self.frame_buffer
    }
}
