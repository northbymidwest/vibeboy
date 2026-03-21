/// PPU timing state machine: step/tick loop, line-start handlers, mode transitions,
/// STAT IRQ edge detection, OAM scan.

use super::Ppu;

impl Ppu {
    /// Step the PPU by `cycles` T-cycles.
    /// Returns interrupt flags (bit0=VBlank, bit1=STAT) to OR into IF.
    pub fn step(&mut self, cycles: u32) -> u8 {
        self.if_flags = 0;

        if self.lcdc & 0x80 == 0 {
            return 0;
        }

        // Deferred window activation from WX write handler: process at the
        // M-cycle boundary so the activation sees the new WX value.
        if self.window_trigger_pending && self.mode == 3 {
            self.window_trigger_pending = false;
            if self.window_trigger_from_wx_write {
                self.window_trigger_from_wx_write = false;
                if !self.window_active && self.lcdc & 0x20 != 0 && self.wy_triggered {
                    self.activate_window();
                }
            }
            // Note: non-WX-write triggers are now handled immediately in
            // render_pixel_if_possible, so no re-check needed here.
        }

        self.step_inner(cycles, true)
    }

    /// Step the PPU for deferred ticks (from previous M-cycle's lazy flush).
    /// Unlike step(), this does NOT mark the first tick as a CPU write boundary
    /// and does NOT capture oam_bug_row. Returns accumulated IF flags.
    pub fn step_deferred(&mut self, cycles: u32) -> u8 {
        self.step_inner(cycles, false)
    }

    fn step_inner(&mut self, cycles: u32, is_cpu_boundary: bool) -> u8 {
        for i in 0..cycles {
            self.first_tick_of_step = is_cpu_boundary && i == 0;
            self.last_tick_of_step = i == cycles - 1;
            self.tick();
        }

        // Only capture oam_bug_row at CPU-boundary steps (not deferred flushes)
        if is_cpu_boundary {
            self.oam_bug_row = self.accessed_oam_row;
        }

        let flags = self.if_flags;
        self.if_flags = 0;
        flags
    }

    /// Advance the PPU by one T-cycle.
    fn tick(&mut self) {
        self.dot += 1;
        self.total_ticks += 1;

        // Line-start sequence: handle delayed LY/mode transitions
        if self.line_start_pending {
            if self.cgb_mode {
                self.handle_cgb_line_start();
            } else {
                self.handle_dmg_line_start();
            }
            return;
        }

        match self.mode {
            2 => {
                // Per-entry OAM scan: check one entry every 2T during mode 2.
                // DMG scan starts at dot 4 (after line-start offset); CGB at dot 4.
                // Each entry takes 2T, so entry N is checked at dot (4 + N*2).
                let scan_start = 4u32;
                if self.dot >= scan_start && self.dot % 2 == 0 {
                    self.oam_scan_step();
                }

                // DMG OAM bug: track which OAM row the PPU is accessing.
                // accessed_oam_row updates AFTER each 2T sleep in the OAM search loop.
                // The search loop starts at dot 4 (DMG line-start offset), with 2T per entry.
                // Entry N's row is set at dot (6 + N*2), right after the 2T sleep.
                if self.dot >= 6 {
                    let oam_search_index = ((self.dot - 6) / 2) as i16;

                    if !self.cgb_mode {
                        if oam_search_index >= 38 {
                            self.accessed_oam_row = 0xFF;
                        } else {
                            self.accessed_oam_row = (oam_search_index & !1) * 4 + 8;
                        }
                    }

                    // At OAM search index 37 (~dot 80):
                    // OAM writes unblocked (reads stay blocked) on both DMG and CGB.
                    // VRAM reads blocked on DMG only (CGB keeps VRAM accessible until mode 3).
                    if oam_search_index >= 37 {
                        if !self.cgb_mode {
                            self.vram_accessible = false;
                        }
                        self.oam_write_accessible = true;
                    }
                }

                // Mode 2 → Mode 3: transition at dot 84
                let mode2_end = 84;
                if self.dot >= mode2_end {
                    self.accessed_oam_row = 0xFF;
                    self.mode3_dot = self.dot;
                    self.mode = 3;
                    self.mode_for_interrupt = 3;
                    self.oam_accessible = false;
                    self.oam_write_accessible = false;
                    self.vram_accessible = false;
                    self.vram_write_accessible = false;
                    self.init_fifo();
                    // STAT mode bits update immediately on both DMG and CGB
                    self.stat = (self.stat & !0x03) | 0x03;
                    self.update_stat_irq();
                    // Run the first mode 3 tick on the transition dot itself.
                    // The 5T priming delay includes this dot as tick 1, so
                    // tick_mode3 must run here to avoid losing 1T at the boundary.
                    self.tick_mode3();
                }
            }
            3 => {
                // CGB: palette RAM blocked 3T after mode 3 start
                if self.cgb_mode && !self.cgb_palettes_blocked && self.dot >= 87 {
                    self.cgb_palettes_blocked = true;
                }
                // Run per-pixel FIFO logic
                self.tick_mode3();
                // Check if scanline is complete
                if self.position_in_line >= 160 {
                    if self.cgb_mode {
                        // Palette stays blocked into mode 0:
                        // Single-speed: 5T after mode 3 ends
                        // Double-speed: 3T after mode 3 ends
                        let delay = if self.double_speed { 3 } else { 5 };
                        self.cgb_palette_unblock_dot = self.dot + delay;
                    } else {
                        self.cgb_palettes_blocked = false;
                    }
                    self.mode = 0;
                    self.mode_for_interrupt = 0;
                    if self.cgb_mode && self.double_speed {
                        // CGB double-speed: defer STAT bits, accessibility,
                        // and IRQ by 1T
                        self.mode0_stat_dot = self.dot + 1;
                    } else if self.cgb_mode {
                        // CGB normal speed: STAT mode bits and accessibility
                        // change immediately; STAT interrupt fires 1T later.
                        self.stat = self.stat & !0x03;
                        self.oam_accessible = true;
                        self.oam_write_accessible = true;
                        self.vram_accessible = true;
                        self.vram_write_accessible = true;
                        self.hblank_entered = true;
                        self.mode0_stat_dot = self.dot + 1;
                    } else {
                        // DMG and CGB normal speed: STAT mode bits and
                        // accessibility change immediately; only the STAT
                        // interrupt fires 1T later.
                        self.stat = self.stat & !0x03;
                        self.oam_accessible = true;
                        self.oam_write_accessible = true;
                        self.vram_accessible = true;
                        self.vram_write_accessible = true;
                        self.hblank_entered = true;
                        self.mode0_stat_dot = self.dot + 1;
                    }
                    if self.window_active {
                        self.window_line_counter = self.window_line_counter.wrapping_add(1);
                    }
                }
            }
            0 => {
                // CGB: deferred palette unblock (3T after mode 3 end)
                if self.cgb_palette_unblock_dot > 0 && self.dot >= self.cgb_palette_unblock_dot {
                    self.cgb_palette_unblock_dot = 0;
                    self.cgb_palettes_blocked = false;
                }
                // Delayed mode 0 STAT IRQ (both DMG and CGB)
                if self.mode0_stat_dot > 0 && self.dot >= self.mode0_stat_dot {
                    self.mode0_stat_dot = 0;
                    if self.cgb_mode {
                        // CGB: STAT bits and accessibility also deferred
                        self.stat = self.stat & !0x03;
                        self.oam_accessible = true;
                        self.oam_write_accessible = true;
                        self.vram_accessible = true;
                        self.vram_write_accessible = true;
                        self.hblank_entered = true;
                    }
                    // Fire the mode 0 STAT interrupt
                    self.update_stat_irq();
                }
                // LCD first-line: STAT mode bits stay 0, then skip to mode 3 at dot 79.
                // Hardware: 1T DMG sleep + 76T mode 0 + 2T OAM block = T=79 STAT mode 3.
                // mode3_start_delay=5 ensures actual pixel rendering starts at dot 84.
                if self.lcd_first_line && self.dot >= 79 {
                    self.lcd_first_line = false;
                    self.lcd_first_line_short = true;
                    self.oam_scan(); // collect sprites
                    self.transition_to_mode3();
                    return;
                }
                // Mode 0 → end of scanline at dot 456 (or shorter for first line after LCD enable).
                // First line HBlank is shortened by 8T phantom cycles_for_line augment.
                // DMG: +1T initial sleep → 456 - 8 + 1 = 449T.
                // CGB: no initial sleep → 456 - 8 = 448T.
                let line_end = if self.lcd_first_line_short {
                    if self.cgb_mode { 448 } else { 449 }
                } else {
                    456
                };
                // CGB mode 2 STAT quirk at VBlank entry: inject STAT IF 2T
                // before the line wrap so it falls in a different HALT half-cycle
                // than VBlank IF (which fires at dot 2 of line 144). This 4T
                // separation is required by vblank_stat_intr-C. We also set
                // stat_irq_line so the wrap-point update_stat_irq doesn't
                // double-fire.
                if self.cgb_mode && self.ly == 143 && self.dot == line_end - 2 {
                    if !self.stat_irq_line && self.stat & 0x20 != 0 {
                        self.if_flags |= 0x02;
                        self.stat_irq_line = true;
                    }
                }
                if self.dot >= line_end {
                    self.lcd_first_line_short = false;
                    self.dot = 0;
                    self.ly = self.ly.wrapping_add(1);

                    if self.cgb_mode {
                        // CGB: all changes (LY, coincidence, mode) are deferred
                        // to the line-start handler. ly_for_comparison and coincidence
                        // update happen at dot 3-4, not at the wrap.
                        //
                        // CGB VBlank line wraps (144-152): use direct IF injection
                        // for Mode 2 STAT source instead of mode_for_interrupt pulse.
                        // Pulsing mode_for_interrupt through 2→-1 would disturb
                        // stat_irq_line, creating spurious rising edges when the
                        // line-start handler restores mode_for_interrupt to 1.
                        if self.ly >= 144 && self.ly <= 152 {
                            if !self.stat_irq_line && self.stat & 0x20 != 0 {
                                self.if_flags |= 0x02;
                            }
                        }
                        self.line_start_pending = true;
                        self.line_start_is_vblank = self.ly >= 144;
                    } else {
                        // DMG: defer to line-start sequence
                        self.line_start_pending = true;
                        if self.ly < 144 {
                            // Active line wrap: mode_for_interrupt stays at 0
                            // (from the previous HBlank). The transition to
                            // mode 2 happens naturally in the line-start handler:
                            // dot 3 creates a gap (-1), dot 4 activates mode 2.
                            self.line_start_is_vblank = false;
                        } else {
                            // DMG VBlank entry (ly == 144): defer to line-start.
                            self.line_start_is_vblank = true;
                            self.ly_for_comparison = -1;
                            self.update_coincidence();
                            self.update_stat_irq();
                        }
                    }
                }
            }
            1 => {
                // VBlank lines

                // Line 153 extended state machine: spreads the LY 153→0
                // transition across multiple T-cycles (matching hardware timing).
                if self.line_153_phase > 0 {
                    self.tick_line_153();
                }

                // CGB line 153 early handling (non-DMG path)
                if self.cgb_mode && self.ly == 153 && self.dot == 4 && self.line_153_phase == 0 {
                    // CGB line 153: ly_for_comparison = 153 at T+4
                    self.ly_for_comparison = 153;
                    self.update_coincidence();
                    self.update_stat_irq();
                    // Start extended sequence at phase 1 (next event at dot 8)
                    self.line_153_phase = 1;
                }

                if self.dot >= 456 {
                    self.dot = 0;
                    self.line_153_phase = 0;
                    if self.ly == 0 {
                        // Line 153 already set LY=0; now start actual line 0
                        self.frame_ready = true;
                        self.window_line_counter = 0;
                        self.wy_triggered = false;
                        if self.cgb_mode {
                            // CGB: visible_ly already 0 (set by line 153 handler)
                            self.line_start_pending = true;
                            self.line_start_is_vblank = false;
                        } else {
                            // DMG: defer to line-start for line 0
                            // Note: do NOT prime mode_for_interrupt = 2 here.
                            // On hardware, the mode 2 STAT interrupt fires 1T
                            // later on line 0 than on lines 1-143 (at T+4 vs T+3).
                            // Keeping mode_for_interrupt at 1 (from VBlank) ensures
                            // the mode 2 interrupt is deferred to dot 4.
                            self.line_start_pending = true;
                            self.line_start_is_vblank = false;
                            self.ly_for_comparison = 0;
                        }
                    } else {
                        self.ly = self.ly.wrapping_add(1);
                        if self.cgb_mode {
                            // CGB VBlank line wraps (145-152): direct IF injection
                            // for Mode 2 source. Same rationale as line 144 wrap.
                            if self.ly >= 145 && self.ly <= 152 {
                                if !self.stat_irq_line && self.stat & 0x20 != 0 {
                                    self.if_flags |= 0x02;
                                }
                            }
                            self.line_start_pending = true;
                            self.line_start_is_vblank = true;
                        } else {
                            // DMG VBlank line transition
                            self.line_start_pending = true;
                            self.line_start_is_vblank = true;
                            self.ly_for_comparison = -1;
                            self.update_coincidence();
                            self.update_stat_irq();
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// DMG line-start sequence: handles the delayed LY visibility and mode transitions
    /// that occur during dots 1-5 of each new scanline on DMG hardware.
    fn handle_dmg_line_start(&mut self) {
        if !self.line_start_is_vblank {
            // Active line (0-143)
            match self.dot {
                1 | 2 => {} // idle
                3 => {
                    self.visible_ly = self.ly;
                    self.ly_for_comparison = if self.ly == 0 { 0 } else { -1 };
                    self.update_coincidence();
                    if self.ly != 0 {
                        // Lines 1-143: activate mode 2 source. The mode
                        // was 0 (HBlank) at the wrap; transitioning to 2
                        // here fires the Mode 2 IRQ via rising edge if
                        // stat_irq_line was low (mode 0 wasn't keeping it
                        // high). If mode 0 WAS keeping it high, mode 2
                        // doesn't create a rising edge → "STAT blocking."
                        self.mode_for_interrupt = 2;
                    }
                    // OAM reads blocked 1T before mode 2 (T=3 of line start)
                    self.oam_accessible = false;
                    self.stat &= !0x03;
                    self.update_stat_irq();
                }
                4 => {
                    self.mode = 2;
                    self.stat = (self.stat & !0x03) | 0x02;
                    self.mode_for_interrupt = 2;
                    self.oam_accessible = false;
                    self.oam_write_accessible = false;
                    self.vram_accessible = true;
                    self.vram_write_accessible = true;
                    self.accessed_oam_row = 0;
                    self.ly_for_comparison = self.ly as i16;
                    self.update_coincidence();
                    if self.lcdc & 0x20 != 0 && self.ly == self.wy {
                        self.wy_triggered = true;
                    }
                    self.scanline_sprites.clear();
                    self.oam_scan_index = 0;
                    self.update_stat_irq();
                    // Clear mode_for_interrupt after the mode 2 source
                    // has been evaluated. This allows other sources (like
                    // LYC coincidence) to independently trigger during
                    // mode 2 without being masked by the mode 2 source.
                    self.mode_for_interrupt = -1;
                    self.update_stat_irq();
                    self.line_start_pending = false;
                }
                _ => {}
            }
        } else {
            // VBlank line (144-153)
            match self.dot {
                1 => {} // idle
                2 => {
                    self.visible_ly = self.ly;
                    // DMG: Mode 2 STAT quirk fires early at VBlank entry
                    // (dot 2), before the mode transition at dot 5. Hardware
                    // briefly pulses the Mode 2 source as the line-start
                    // state machine begins, even on VBlank lines.
                    if self.ly == 144 {
                        if !self.stat_irq_line && self.stat & 0x20 != 0 {
                            self.if_flags |= 0x02;
                        }
                    }
                }
                3 => {} // idle
                4 => {
                    self.ly_for_comparison = self.ly as i16;
                    self.update_coincidence();
                    self.update_stat_irq();
                    // Line 153: start extended sequence (LY reset spread across dots 6-12)
                    if self.ly == 153 {
                        self.line_153_phase = 1;
                    }
                }
                5 => {
                    if self.ly == 144 || (self.ly == 0 && self.line_153_phase == 0) {
                        // VBlank entry: mode 0→1, VBlank IF, Mode 2 STAT quirk.
                        // The Mode 2 quirk also fires at the line wrap (dot 0)
                        // for early interrupt availability. The dot-5 injection
                        // here is redundant (IF bit already set) but keeps the
                        // stat_irq_line update aligned with the mode transition.
                        if self.mode != 1 {
                            if !self.stat_irq_line && self.stat & 0x20 != 0 {
                                self.if_flags |= 0x02;
                            }
                            self.stat = (self.stat & !0x03) | 0x01;
                            self.mode = 1;
                            self.if_flags |= 0x01; // VBlank IF
                            self.mode_for_interrupt = 1;
                            self.oam_accessible = true;
                            self.oam_write_accessible = true;
                            self.vram_accessible = true;
                            self.vram_write_accessible = true;
                            self.lcd_first_frame = false;
                            self.update_stat_irq();
                        }
                    }
                    self.line_start_pending = false;
                }
                _ => {}
            }
        }
    }

    /// CGB line-start sequence: handles the delayed LY visibility and mode transitions
    /// that occur during dots 1-4 of each new scanline on CGB hardware.
    /// State machine: 2T idle → 1T ly_for_comp/-1+mode2 → 1T oam_scan+lsp_clear.
    fn handle_cgb_line_start(&mut self) {
        if !self.line_start_is_vblank {
            // Active line (0-143)
            match self.dot {
                1 => {
                    // State 35 start: LY becomes visible in IO register
                    self.visible_ly = self.ly;
                }
                2 => {
                    // State 35 end: idle
                }
                3 => {
                    // State 6 equivalent: ly_for_comparison and mode update
                    if self.ly == 0 {
                        self.ly_for_comparison = 0;
                    } else {
                        self.ly_for_comparison = -1;
                        self.mode_for_interrupt = 2;
                    }
                    // Clear STAT mode bits (still mode 0 internally)
                    self.stat &= !0x03;
                    self.update_coincidence();
                    self.update_stat_irq();
                }
                4 => {
                    // State 7 equivalent: Mode 2 entry
                    self.mode = 2;
                    self.stat = (self.stat & !0x03) | 0x02;
                    self.mode_for_interrupt = 2;
                    self.oam_accessible = false;
                    self.oam_write_accessible = false;
                    self.vram_accessible = true;
                    self.vram_write_accessible = true;
                    self.accessed_oam_row = 0;
                    self.ly_for_comparison = self.ly as i16;
                    self.update_coincidence();
                    if self.lcdc & 0x20 != 0 && self.ly == self.wy {
                        self.wy_triggered = true;
                    }
                    self.scanline_sprites.clear();
                    self.oam_scan_index = 0;
                    self.update_stat_irq();
                    // Immediately clear mode_for_interrupt
                    self.mode_for_interrupt = -1;
                    self.update_stat_irq();
                    self.line_start_pending = false;
                    // Apply lsp-deferred LYC write now that line-start is done
                    if let Some(lyc) = self.pending_lyc.take() {
                        self.lyc = lyc;
                        if self.lcdc & 0x80 != 0 {
                            self.update_coincidence();
                            self.update_stat_irq();
                        }
                    }
                }
                _ => {}
            }
        } else {
            // VBlank line (144-153)
            match self.dot {
                1 => {
                    // LY becomes visible in IO register
                    self.visible_ly = self.ly;
                }
                2 => {
                    // Idle
                }
                3 => {
                    // Clear ly_for_comparison briefly (creates coincidence gap)
                    self.ly_for_comparison = -1;
                    self.update_coincidence();
                    // Mode 2 STAT source: direct IF injection for VBlank lines
                    // 145-151. Line 144's mode 2 comes from the early injection
                    // at line 143 dot line_end-2. Lines 152-153 do not fire mode 2.
                    // Uses direct injection instead of mode_for_interrupt pulse
                    // to avoid disturbing stat_irq_line (which would create
                    // spurious rising edges).
                    if self.ly >= 145 && self.ly <= 151 {
                        if !self.stat_irq_line && self.stat & 0x20 != 0 {
                            self.if_flags |= 0x02;
                        }
                    }
                    self.update_stat_irq();
                }
                4 => {
                    // ly_for_comparison update to actual line
                    self.ly_for_comparison = self.ly as i16;
                    self.update_coincidence();
                    self.update_stat_irq();
                    // Line 153: start extended sequence
                    if self.ly == 153 {
                        self.line_153_phase = 1;
                    }
                    // VBlank entry at dot 4 (after ly_for_comparison set).
                    if self.ly == 144 && self.mode != 1 {
                        self.stat = (self.stat & !0x03) | 0x01;
                        self.mode = 1;
                        self.if_flags |= 0x01; // VBlank IF
                        self.mode_for_interrupt = 1;
                        self.oam_accessible = true;
                        self.oam_write_accessible = true;
                        self.vram_accessible = true;
                        self.vram_write_accessible = true;
                        self.lcd_first_frame = false;
                        self.update_stat_irq();
                    }
                }
                5 => {
                    self.line_start_pending = false;
                    // Apply lsp-deferred LYC write now that line-start is done
                    if let Some(lyc) = self.pending_lyc.take() {
                        self.lyc = lyc;
                        if self.lcdc & 0x80 != 0 {
                            self.update_coincidence();
                            self.update_stat_irq();
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Line 153 extended state machine: spreads LY 153→0 transition across
    /// multiple T-cycles to match hardware timing verified by mooneye tests.
    ///
    /// DMG timing (from line 153 start):
    ///   dot 4: ly_for_comparison=153 (handled in line_start handler)
    ///   dot 6: LY=0, visible_ly=0
    ///   dot 8: ly_for_comparison=-1; STAT update
    ///   dot 12: ly_for_comparison=0; STAT update
    ///
    /// CGB timing (from line 153 start):
    ///   dot 4: ly_for_comparison=153 (handled in mode 1 handler)
    ///   dot 8: LY=0, visible_ly=0; ly_for_comparison stays 153; STAT update
    ///   dot 12: ly_for_comparison=0; STAT update
    fn tick_line_153(&mut self) {
        if self.cgb_mode {
            match self.dot {
                8 => {
                    // CGB: LY resets to 0; ly_for_comparison stays at 153
                    self.ly = 0;
                    self.visible_ly = 0;
                    self.update_stat_irq();
                }
                12 => {
                    // CGB: ly_for_comparison transitions to 0
                    self.ly_for_comparison = 0;
                    self.update_coincidence();
                    self.update_stat_irq();
                    self.line_153_phase = 0;
                    // Apply l153-deferred LYC write now that line 153 is done
                    if let Some(lyc) = self.pending_lyc.take() {
                        self.lyc = lyc;
                        if self.lcdc & 0x80 != 0 {
                            self.update_coincidence();
                            self.update_stat_irq();
                        }
                    }
                }
                _ => {}
            }
        } else {
            // DMG
            match self.dot {
                6 => {
                    // DMG: LY resets to 0 (visible)
                    self.ly = 0;
                    self.visible_ly = 0;
                }
                8 => {
                    // DMG: ly_for_comparison clears to -1 (brief gap)
                    self.ly_for_comparison = -1;
                    self.update_coincidence();
                    self.update_stat_irq();
                }
                12 => {
                    // DMG: ly_for_comparison transitions to 0
                    self.ly_for_comparison = 0;
                    self.update_coincidence();
                    self.update_stat_irq();
                    self.line_153_phase = 0;
                }
                _ => {}
            }
        }
    }

    // ---- Mode transitions ----

    pub(super) fn transition_to_mode2(&mut self) {
        self.mode = 2;
        self.mode_for_interrupt = 2;
        self.stat = (self.stat & !0x03) | 0x02;
        self.oam_accessible = false;
        self.oam_write_accessible = false;
        self.vram_accessible = true;
        self.vram_write_accessible = true;
        self.accessed_oam_row = 0;
        self.oam_scan();
        if self.lcdc & 0x20 != 0 && self.ly == self.wy {
            self.wy_triggered = true;
        }
        self.update_stat_irq();
    }

    pub(super) fn transition_to_mode3(&mut self) {
        self.mode3_dot = self.dot;
        self.mode = 3;
        self.mode_for_interrupt = 3;
        self.stat = (self.stat & !0x03) | 0x03;
        self.oam_accessible = false;
        self.oam_write_accessible = false;
        self.vram_accessible = false;
        self.vram_write_accessible = false;
        self.init_fifo();
        self.update_stat_irq();
        // Run the first mode 3 tick on the transition dot itself.
        // The 5T priming includes this dot as tick 1.
        self.tick_mode3();
    }

    fn transition_to_mode1(&mut self) {
        self.mode = 1;
        self.mode_for_interrupt = 1;
        self.stat = (self.stat & !0x03) | 0x01;
        self.oam_accessible = true;
        self.oam_write_accessible = true;
        self.vram_accessible = true;
        self.vram_write_accessible = true;
        // VBlank interrupt always fires
        self.if_flags |= 0x01;
        // CGB: Hardware quirk: Mode 2 source also fires at VBlank entry (one-shot)
        // DMG: handled by mode_for_interrupt priming in handle_dmg_line_start
        self.update_stat_irq_with_mode2(true);
        // Immediately re-evaluate without forced mode 2 so stat_irq_line reflects
        // the normal mode 1 state. Without this, the forced mode 2 signal lingers
        // until the next update_stat_irq call (dot 456 of next VBlank line).
        self.update_stat_irq();
    }

    // ---- Edge-triggered STAT interrupt ----

    pub(super) fn update_stat_irq(&mut self) {
        let coincidence = self.stat & 0x04 != 0;
        // Both DMG and CGB use mode_for_interrupt for STAT IRQ edge detection.
        // This decouples the interrupt source from the visible STAT mode bits,
        // allowing line-start sequences to control when mode 2 fires.
        let mode_signal = match self.mode_for_interrupt {
            0 => self.stat & 0x08 != 0,
            1 => self.stat & 0x10 != 0,
            2 => self.stat & 0x20 != 0,
            _ => false, // 3 or -1: no mode fires
        };
        let signal = mode_signal || (self.stat & 0x40 != 0 && coincidence);

        // Fire on rising edge only
        if signal && !self.stat_irq_line {
            self.if_flags |= 0x02;
        }
        self.stat_irq_line = signal;
    }

    /// Update stat_irq_line without generating IF. Used when LYC writes are
    /// suppressed (CGB line_153_phase / near-line-end) so that subsequent
    /// update_stat_irq() calls don't see a false rising edge.
    pub(super) fn update_stat_irq_silent(&mut self) {
        let coincidence = self.stat & 0x04 != 0;
        let mode_signal = match self.mode_for_interrupt {
            0 => self.stat & 0x08 != 0,
            1 => self.stat & 0x10 != 0,
            2 => self.stat & 0x20 != 0,
            _ => false,
        };
        self.stat_irq_line = mode_signal || (self.stat & 0x40 != 0 && coincidence);
    }

    /// CGB-only: STAT IRQ check with Mode 2 source forced on (VBlank entry quirk).
    fn update_stat_irq_with_mode2(&mut self, force_mode2: bool) {
        let coincidence = self.stat & 0x04 != 0;
        let signal =
            (self.stat & 0x08 != 0 && self.mode_for_interrupt == 0) ||
            (self.stat & 0x10 != 0 && self.mode_for_interrupt == 1) ||
            (self.stat & 0x20 != 0 && (self.mode_for_interrupt == 2 || force_mode2)) ||
            (self.stat & 0x40 != 0 && coincidence);

        if signal && !self.stat_irq_line {
            self.if_flags |= 0x02;
        }
        self.stat_irq_line = signal;
    }

    /// CGB-only: STAT IRQ re-evaluation triggered by a CPU write to STAT.
    /// Mode sources (bits 3-5) are suppressed during mode 2/3; only LYC
    /// source (bit 6) can trigger during those modes.
    pub(super) fn update_stat_irq_on_write(&mut self) {
        let coincidence = self.stat & 0x04 != 0;
        // Full signal: used to update stat_irq_line (prevents spurious
        // rising edges on the next normal PPU tick).
        let full_signal =
            (self.stat & 0x08 != 0 && self.mode == 0) ||
            (self.stat & 0x10 != 0 && self.mode == 1) ||
            (self.stat & 0x20 != 0 && self.mode == 2) ||
            (self.stat & 0x40 != 0 && coincidence);
        // Write signal: during mode 2/3, only LYC source can trigger an
        // interrupt from a STAT write. Mode sources are suppressed.
        let write_signal = if self.mode <= 1 {
            full_signal
        } else {
            self.stat & 0x40 != 0 && coincidence
        };

        if write_signal && !self.stat_irq_line {
            self.if_flags |= 0x02;
        }
        self.stat_irq_line = full_signal;
    }

    pub(super) fn update_coincidence(&mut self) {
        if self.ly_for_comparison >= 0 && self.ly_for_comparison as u8 == self.lyc {
            self.stat |= 0x04;
        } else {
            self.stat &= !0x04;
        }
    }

    // ---- OAM scan ----

    pub(super) fn oam_scan(&mut self) {
        self.scanline_sprites.clear();
        let sprite_height: i16 = if self.lcdc & 0x04 != 0 { 16 } else { 8 };
        let ly = self.ly as i16;

        for i in 0..40usize {
            let sprite_y = self.oam[i * 4] as i16 - 16;
            let sprite_x = self.oam[i * 4 + 1];
            let tile_idx = self.oam[i * 4 + 2];
            let attrs = self.oam[i * 4 + 3];

            if ly >= sprite_y && ly < sprite_y + sprite_height {
                self.scanline_sprites.push((self.oam[i * 4], sprite_x, tile_idx, attrs, i as u8));
                if self.scanline_sprites.len() >= 10 {
                    break;
                }
            }
        }
    }

    /// Scan one OAM entry per call. Called every 2T during mode 2.
    /// Uses current LCDC bit 2 for sprite height, enabling mid-scan size changes.
    fn oam_scan_step(&mut self) {
        if self.oam_scan_index >= 40 || self.scanline_sprites.len() >= 10 {
            return;
        }
        let i = self.oam_scan_index as usize;
        self.oam_scan_index += 1;
        let sprite_height: i16 = if self.lcdc & 0x04 != 0 { 16 } else { 8 };
        let ly = self.ly as i16;
        let sprite_y = self.oam[i * 4] as i16 - 16;
        let sprite_x = self.oam[i * 4 + 1];
        let tile_idx = self.oam[i * 4 + 2];
        let attrs = self.oam[i * 4 + 3];
        if ly >= sprite_y && ly < sprite_y + sprite_height {
            self.scanline_sprites.push((self.oam[i * 4], sprite_x, tile_idx, attrs, i as u8));
        }
    }
}
