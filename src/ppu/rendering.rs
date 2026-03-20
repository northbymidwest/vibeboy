/// Mode 3 pixel FIFO pipeline: fetcher, sprite fetch, pixel output, color conversion.

use super::{FifoPixel, Ppu};

impl Ppu {
    /// Initialize FIFO state at the start of Mode 3
    pub(super) fn init_fifo(&mut self) {
        self.bg_fifo.clear();
        self.oam_fifo.clear();
        self.fetcher.reset(false);
        // Pre-fill the FIFO with 8 junk pixels. These get consumed during
        // the junk zone (positions -16 to -9) in parallel with the fetcher's
        // first tile read. The alignment check (pos & 7) == (SCX & 7) jumps
        // position to -8, entering the SCX discard zone (-8 to -1). This
        // naturally handles fractional scroll pixel discarding.
        for _ in 0..8 {
            self.bg_fifo.push_back(FifoPixel::default());
        }
        self.position_in_line = -16;
        self.line_has_fractional_scrolling = false;
        self.window_is_being_fetched = false;
        self.sprite_fetch_active = false;
        self.sprites_fetched = 0;
        self.window_active = false;
        self.window_trigger_pending = false;
        self.window_trigger_from_wx_write = false;
        self.disable_window_pixel_insertion_glitch = false;
        self.last_sprite_slot = -1;
        // Hardware pipeline priming delay. tick_mode3() runs on the
        // transition dot itself, so the delay counter starts decrementing
        // immediately. DMG=5T, CGB=4T (CGB begins rendering 1T earlier).
        self.mode3_start_delay = if self.cgb_mode { 4 } else { 5 };
    }

    /// One T-cycle of Mode 3 pixel FIFO processing
    pub(super) fn tick_mode3(&mut self) {
        // Pipeline priming delay at start of Mode 3
        if self.mode3_start_delay > 0 {
            self.mode3_start_delay -= 1;
            return;
        }

        // Sprite fetch active: BG fetcher keeps running, pixel output paused
        if self.sprite_fetch_active {
            self.tick_bg_fetcher();
            if self.sprite_alignment_delay > 0 {
                self.sprite_alignment_delay -= 1;
                return;
            }
            self.tick_sprite_fetch();
            return;
        }

        // Pixel output runs BEFORE the fetcher advance each T-cycle,
        // matching hardware pipeline ordering where the FIFO is drained
        // before the fetcher state machine advances.
        self.render_pixel_if_possible();

        // Advance BG fetcher (runs every T-cycle regardless of pixel output)
        self.tick_bg_fetcher();
    }

    /// Try to pop a pixel from the BG FIFO and either discard it (junk/SCX
    /// zone) or render it. Also handles sprite and window trigger detection.
    fn render_pixel_if_possible(&mut self) {
        if self.bg_fifo.len() == 0 {
            return;
        }

        // Check sprite trigger (only in visible pixel zone, after SCX
        // discard is complete — sprites use absolute screen positions)
        if self.lcdc & 0x02 != 0 && self.position_in_line >= 0 {
            if let Some(sprite_idx) = self.find_sprite_at_pixel_x() {
                self.start_sprite_fetch(sprite_idx);
                if self.sprite_alignment_delay > 0 {
                    self.sprite_alignment_delay -= 1;
                } else {
                    self.tick_sprite_fetch();
                }
                return;
            }
        }

        // Window trigger check (only in visible pixel zone).
        // Activate immediately — on hardware, the window trigger takes
        // effect within the same T-cycle. The previous deferred activation
        // (first_tick_of_step guard) caused variable dead ticks that made
        // mode 3 duration depend on batch alignment rather than pixel position.
        if self.position_in_line >= 0 && self.check_window_trigger() {
            self.activate_window();
            return;
        }

        let bg_pixel = self.bg_fifo.pop_front();
        let oam_pixel = if self.oam_fifo.len() > 0 {
            Some(self.oam_fifo.pop_front())
        } else {
            None
        };

        // Junk zone: positions -16 to -9. The pre-filled junk pixels drain
        // here while the fetcher reads the first tile. The alignment check
        // (pos & 7) == (SCX & 7) jumps position to -8, transitioning to the
        // SCX discard zone. This models the hardware's tile-boundary alignment.
        if self.position_in_line >= -16 && self.position_in_line <= -9 {
            if (self.position_in_line & 7) == (self.scx as i16 & 7) {
                // Tile boundary alignment: skip remaining junk
                self.position_in_line = -8;
            } else if self.window_is_being_fetched
                && (self.position_in_line & 7) == 6
                && (self.scx & 7) == 7
            {
                // Window fetch edge case: early alignment when SCX low bits = 7
                self.position_in_line = -8;
            } else if self.position_in_line == -9 {
                // Alignment never matched (safety net), wrap back
                self.position_in_line = -16;
                return;
            } else {
                self.line_has_fractional_scrolling = true;
            }
            // Fall through to the < 0 discard check below
        }

        self.window_is_being_fetched = false;

        if self.position_in_line < 0 {
            // Discard zone (includes -8..-1 from SCX alignment and
            // post-alignment junk zone pixels): pop but don't render.
            self.position_in_line += 1;
            return;
        }

        // Visible pixel zone (0..159): render to framebuffer
        self.output_pixel(bg_pixel, oam_pixel);
    }

    /// Check if window should activate at current position_in_line
    fn check_window_trigger(&self) -> bool {
        if self.window_active {
            return false; // already active
        }
        if self.lcdc & 0x20 == 0 {
            return false; // window disabled
        }
        if !self.wy_triggered {
            return false; // WY condition not met
        }
        if self.position_in_line < 0 {
            return false; // still in junk/discard zone
        }
        let px = self.position_in_line as u8;
        // WX=0..166 maps to screen pixel (WX-7)..
        // Window triggers when position == WX-7 (for WX >= 7)
        // For WX < 7, window triggers at position == 0
        let wx_screen = if self.wx >= 7 { self.wx - 7 } else { 0 };
        if px == wx_screen {
            return true;
        }
        // DMG LCD-PPU horizontal desync: window also triggers 1 pixel late
        // (WX == position + 6), unless WX was just written this T-cycle.
        if !self.cgb_mode && !self.wx_just_changed && self.wx >= 7 {
            if px == self.wx.wrapping_sub(6) {
                return true;
            }
        }
        false
    }

    /// Activate window: clear BG FIFO, restart fetcher for window tiles.
    pub(super) fn activate_window(&mut self) {
        self.bg_fifo.clear();
        self.fetcher.reset(true);
        self.window_active = true;
        self.window_is_being_fetched = true;
    }

    /// Find a sprite that triggers at the current position_in_line
    fn find_sprite_at_pixel_x(&self) -> Option<usize> {
        if self.position_in_line < 0 {
            return None;
        }
        let px = self.position_in_line as u8;
        for (i, &(_y, x, _tile, _attrs, _oam_idx)) in self.scanline_sprites.iter().enumerate() {
            if self.sprites_fetched & (1 << i) != 0 {
                continue; // already fetched
            }
            // Sprite triggers when position reaches sprite_x - 8
            // For sprites with X < 8, they trigger at position == 0
            let trigger_x = if x >= 8 { x - 8 } else { 0 };
            if px == trigger_x {
                return Some(i);
            }
        }
        None
    }

    /// Start a sprite fetch for the given sprite index
    fn start_sprite_fetch(&mut self, sprite_idx: usize) {
        self.sprite_fetch_active = true;
        self.sprite_fetch_step = 0;
        self.sprite_fetch_tick = 0;
        self.sprite_fetch_entry = sprite_idx;
        self.sprites_fetched |= 1 << sprite_idx;
        // Compute alignment penalty based on tile slot grouping
        let sprite_x = self.scanline_sprites[sprite_idx].1;
        let adjusted = sprite_x.wrapping_add(self.scx);
        let slot = (adjusted >> 3) as i16;
        if slot == self.last_sprite_slot {
            // Same tile slot as previous sprite: just 6T fetch, no alignment penalty
            self.sprite_alignment_delay = 0;
        } else {
            // New slot: full alignment penalty
            let alignment = (adjusted & 7) as u8;
            self.sprite_alignment_delay = 5 - std::cmp::min(5, alignment);
            // Special case: sprite at OAM X=0 (completely off-screen left) always
            // gets full 5T alignment penalty regardless of SCX.
            if sprite_x == 0 {
                self.sprite_alignment_delay = 5;
            }
            self.last_sprite_slot = slot;
        }
    }

    /// Advance the sprite fetch state machine (6T total: 2T tile_id, 2T data_lo, 2T data_hi)
    fn tick_sprite_fetch(&mut self) {
        self.sprite_fetch_tick += 1;
        if self.sprite_fetch_tick < 2 {
            return; // each step takes 2T
        }
        self.sprite_fetch_tick = 0;

        let &(raw_y, raw_x, mut tile_idx, attrs, oam_index) = &self.scanline_sprites[self.sprite_fetch_entry];
        let sprite_height: i16 = if self.lcdc & 0x04 != 0 { 16 } else { 8 };

        match self.sprite_fetch_step {
            0 => {
                // Read tile ID (already have it from OAM scan, just advance)
                if sprite_height == 16 {
                    tile_idx &= 0xFE;
                }
                // Compute row
                let sprite_y = raw_y as i16 - 16;
                let y_flip = attrs & 0x40 != 0;
                let mut row = (self.ly as i16 - sprite_y) as u16;
                if y_flip {
                    row = sprite_height as u16 - 1 - row;
                }
                let actual_tile = if sprite_height == 16 {
                    if row < 8 { tile_idx & 0xFE } else { tile_idx | 0x01 }
                } else {
                    tile_idx
                };
                let row_in_tile = row % 8;
                let vram_bank_sel = if self.cgb_mode && attrs & 0x08 != 0 { 1usize } else { 0usize };
                let byte_addr = (actual_tile as u16 * 16 + row_in_tile * 2) as usize;
                // Pre-compute and store for next steps
                self.sprite_tile_data_low = self.vram[vram_bank_sel][byte_addr];
                self.sprite_tile_data_high = self.vram[vram_bank_sel][byte_addr + 1];
                self.sprite_fetch_step = 1;
            }
            1 => {
                // Data low already read in step 0
                self.sprite_fetch_step = 2;
            }
            2 => {
                // Data high already read, now mix into FIFO
                self.mix_sprite_into_fifo(raw_x, attrs, oam_index);
                self.sprite_fetch_active = false;

                // Check if another sprite triggers at the same pixel_x
                if self.lcdc & 0x02 != 0 {
                    if let Some(next_idx) = self.find_sprite_at_pixel_x() {
                        self.start_sprite_fetch(next_idx);
                    }
                }
            }
            _ => {}
        }
    }

    /// Mix fetched sprite data into the OAM FIFO.
    /// Sprite pixels are stored separately from BG pixels; priority is resolved
    /// at output time when both FIFOs are popped together.
    fn mix_sprite_into_fifo(&mut self, sprite_x: u8, attrs: u8, oam_index: u8) {
        let x_flip = attrs & 0x20 != 0;
        let bg_over = attrs & 0x80 != 0;
        let palette_idx = if self.cgb_mode && !self.dmg_compat {
            attrs & 0x07
        } else if self.dmg_compat {
            if attrs & 0x10 != 0 { 1 } else { 0 }
        } else {
            0
        };
        let dmg_pal = if attrs & 0x10 != 0 { 1u8 } else { 0u8 };

        let lo = self.sprite_tile_data_low;
        let hi = self.sprite_tile_data_high;

        // Pad OAM FIFO to exactly 8 entries for sprite overlay.
        while self.oam_fifo.len() < 8 {
            self.oam_fifo.push_back(FifoPixel::default());
        }

        // Overlay sprite pixels into OAM FIFO.
        // For left-edge sprites (X < 8), the leftmost (8-X) tile pixels are
        // clipped. FIFO pos 0 = next pixel to output (leftmost on screen).
        // Non-flipped: fifo[p] gets bit (7 - tile_pixel), where tile_pixel 0=left
        // Flipped:     fifo[p] gets bit (tile_pixel)
        let skip = if sprite_x >= 8 { 0u8 } else { 8 - sprite_x };
        let num_visible = 8 - skip;

        for p in 0..num_visible {
            let tile_pixel = skip + p; // 0=leftmost .. 7=rightmost in tile
            let bit = if x_flip { tile_pixel } else { 7 - tile_pixel };
            let color_idx = (((hi >> bit) & 1) << 1) | ((lo >> bit) & 1);
            let fifo_pos = p as usize;

            if color_idx == 0 {
                continue;
            }

            let existing = *self.oam_fifo.get(fifo_pos);
            if existing.color_index != 0 {
                // CGB with OAM-index priority (OPRI bit 0 = 0): lower OAM index wins
                if self.cgb_mode && self.opri & 0x01 == 0 && oam_index < existing.sprite_oam_index {
                    // fall through to overwrite
                } else {
                    continue;
                }
            }

            self.oam_fifo.replace(fifo_pos, FifoPixel {
                color_index: color_idx,
                palette: palette_idx,
                is_sprite: true,
                bg_priority: false,
                sprite_bg_over: bg_over,
                sprite_dmg_palette: dmg_pal,
                sprite_oam_index: oam_index,
                bg_color_index: 0,
                bg_palette: 0,
            });
        }
    }

    /// Advance the BG/window tile fetcher by one T-cycle.
    /// 7-state pipeline: each state is exactly 1T.
    ///   GetTileT1: CGB latches fetcher_y, TILE_SEL, map address
    ///   GetTileT2: reads tile ID (and CGB attributes) from VRAM
    ///   GetTileDataLowT1: CGB latches TILE_SEL, tile data address
    ///   GetTileDataLowT2: reads low byte of tile data from VRAM
    ///   GetTileDataHighT1: CGB latches TILE_SEL, tile data address
    ///   GetTileDataHighT2: reads high byte of tile data from VRAM
    ///   Push: pushes 8 pixels to BG FIFO (stalls if FIFO not empty)
    fn tick_bg_fetcher(&mut self) {
        match self.fetcher.state {
            super::FetcherState::GetTileT1 => {
                // CGB (≥CGB-D): latch registers at T1
                if self.cgb_mode {
                    self.fetcher.fetcher_y = if self.fetcher.fetching_window {
                        self.window_line_counter as u8
                    } else {
                        self.scy.wrapping_add(self.ly)
                    };
                    self.fetcher.latched_tile_sel = self.lcdc & 0x10 != 0;
                    self.fetcher.latched_map_addr = self.fetcher_map_addr();
                }
                self.fetcher.state = super::FetcherState::GetTileT2;
            }
            super::FetcherState::GetTileT2 => {
                // CGB: use latched address. DMG: compute fresh.
                let map_addr = if self.cgb_mode {
                    self.fetcher.latched_map_addr
                } else {
                    self.fetcher_map_addr()
                };
                self.fetcher.tile_id = self.vram[0][map_addr];
                self.fetcher.tile_attrs = if self.cgb_mode {
                    self.vram[1][map_addr]
                } else {
                    0
                };
                self.fetcher.state = super::FetcherState::GetTileDataLowT1;
            }
            super::FetcherState::GetTileDataLowT1 => {
                // CGB: latch TILE_SEL and tile data address
                if self.cgb_mode {
                    if self.tile_sel_glitch {
                        self.tile_sel_glitch_latched = true;
                    } else if !self.tile_sel_glitch_latched {
                        self.fetcher.latched_tile_sel = self.lcdc & 0x10 != 0;
                    }
                    let (addr, bank) = self.fetcher_tile_data_addr();
                    self.fetcher.latched_addr = addr;
                    self.fetcher.latched_bank = bank;
                }
                self.fetcher.state = super::FetcherState::GetTileDataLowT2;
            }
            super::FetcherState::GetTileDataLowT2 => {
                // CGB: use cached address. DMG: compute fresh.
                let (addr, bank) = if self.cgb_mode {
                    (self.fetcher.latched_addr, self.fetcher.latched_bank)
                } else {
                    self.fetcher_tile_data_addr()
                };
                if self.tile_sel_glitch && self.cgb_mode {
                    self.fetcher.tile_data_low = self.tile_sel_glitch_data();
                } else {
                    self.fetcher.tile_data_low = self.vram[bank][addr];
                }
                self.fetcher.state = super::FetcherState::GetTileDataHighT1;
            }
            super::FetcherState::GetTileDataHighT1 => {
                // CGB: latch TILE_SEL and tile data address
                if self.cgb_mode {
                    if self.tile_sel_glitch {
                        self.tile_sel_glitch_latched = true;
                    } else if !self.tile_sel_glitch_latched {
                        self.fetcher.latched_tile_sel = self.lcdc & 0x10 != 0;
                    }
                    let (addr, bank) = self.fetcher_tile_data_addr();
                    self.fetcher.latched_addr = addr;
                    self.fetcher.latched_bank = bank;
                }
                self.fetcher.state = super::FetcherState::GetTileDataHighT2;
            }
            super::FetcherState::GetTileDataHighT2 => {
                // CGB: use cached address. DMG: compute fresh.
                let (addr, bank) = if self.cgb_mode {
                    (self.fetcher.latched_addr, self.fetcher.latched_bank)
                } else {
                    self.fetcher_tile_data_addr()
                };
                if (self.tile_sel_glitch || self.tile_sel_glitch_latched) && self.cgb_mode {
                    self.tile_sel_glitch_latched = false;
                    self.fetcher.tile_data_high = self.tile_sel_glitch_data();
                } else {
                    self.fetcher.tile_data_high = self.vram[bank][addr + 1];
                }
                self.fetcher.state = super::FetcherState::Push;
            }
            super::FetcherState::Push => {
                if self.bg_fifo.len() == 0 {
                    self.push_bg_pixels();
                    self.fetcher.tile_x += 1;
                    self.fetcher.state = super::FetcherState::GetTileT1;
                }
                // If FIFO not empty, stall — stay in Push state.
            }
        }
    }

    /// Compute tilemap address for the current fetcher position
    fn fetcher_map_addr(&self) -> usize {
        if self.fetcher.fetching_window {
            let win_map_base: u16 = if self.lcdc & 0x40 != 0 { 0x1C00 } else { 0x1800 };
            let tile_x = self.fetcher.tile_x as u16;
            let tile_y = self.window_line_counter as u16 / 8;
            (win_map_base + tile_y * 32 + (tile_x & 0x1F)) as usize
        } else {
            let bg_map_base: u16 = if self.lcdc & 0x08 != 0 { 0x1C00 } else { 0x1800 };
            let scroll_y = self.scy.wrapping_add(self.ly);
            // Read SCX live (not cached) — hardware reads the register each tile fetch
            let tile_x = ((self.scx / 8).wrapping_add(self.fetcher.tile_x)) & 0x1F;
            let tile_y = (scroll_y as u16) / 8;
            (bg_map_base + tile_y * 32 + tile_x as u16) as usize
        }
    }

    /// Compute VRAM address and bank for tile data.
    /// On CGB, uses latched fetcher_y and TILE_SEL (cached at GetTileT1).
    /// On DMG, computes fresh from current SCY+LY and LCDC.
    fn fetcher_tile_data_addr(&self) -> (usize, usize) {
        let tile_id = self.fetcher.tile_id;
        let attrs = self.fetcher.tile_attrs;
        // CGB: use latched TILE_SEL from T1; DMG: read LCDC live at T2
        let tile_data_signed = if self.cgb_mode {
            !self.fetcher.latched_tile_sel
        } else {
            self.lcdc & 0x10 == 0
        };

        let tile_addr: u16 = if !tile_data_signed {
            tile_id as u16 * 16
        } else {
            (0x1000i32 + (tile_id as i8 as i32) * 16) as u16
        };

        let y_flip = self.cgb_mode && attrs & 0x40 != 0;
        let bank = if self.cgb_mode && attrs & 0x08 != 0 { 1 } else { 0 };

        // CGB: use latched fetcher_y from T1; DMG: read SCY+LY fresh
        let pixel_y = if self.cgb_mode {
            self.fetcher.fetcher_y & 7
        } else if self.fetcher.fetching_window {
            (self.window_line_counter & 7) as u8
        } else {
            self.scy.wrapping_add(self.ly) & 7
        };

        let row = if y_flip { 7 - pixel_y } else { pixel_y };
        let addr = (tile_addr + row as u16 * 2) as usize;
        (addr, bank)
    }

    /// CGB tile_sel_glitch: when LCDC bit 4 transitions 1→0 during Mode 3,
    /// tile data is replaced with the tile INDEX itself for tiles 0-127
    /// (non-CGB_D models).
    fn tile_sel_glitch_data(&self) -> u8 {
        if self.fetcher.latched_tile_sel {
            // last_tileset was true (TILE_SEL=1 at the preceding T1)
            // For tiles < 128: return the tile index as corrupted data
            if self.fetcher.tile_id & 0x80 == 0 {
                self.fetcher.tile_id
            } else {
                // tile >= 128: no glitch, use normal VRAM data
                let (addr, bank) = (self.fetcher.latched_addr, self.fetcher.latched_bank);
                if self.fetcher.state == super::FetcherState::GetTileDataLowT2 {
                    self.vram[bank][addr]
                } else {
                    self.vram[bank][addr + 1]
                }
            }
        } else {
            // last_tileset was false: uses a cached `data_for_sel_glitch` value.
            // For now, use normal VRAM data (this path is less common).
            let (addr, bank) = (self.fetcher.latched_addr, self.fetcher.latched_bank);
            if self.fetcher.state == super::FetcherState::GetTileDataLowT2 {
                self.vram[bank][addr]
            } else {
                self.vram[bank][addr + 1]
            }
        }
    }

    /// Push 8 pixels from fetcher data into the BG FIFO
    fn push_bg_pixels(&mut self) {
        let lo = self.fetcher.tile_data_low;
        let hi = self.fetcher.tile_data_high;
        let attrs = self.fetcher.tile_attrs;

        let x_flip = self.cgb_mode && attrs & 0x20 != 0;
        let bg_prio = self.cgb_mode && attrs & 0x80 != 0;
        let palette = if self.cgb_mode && !self.dmg_compat { attrs & 0x07 } else { 0 };

        for px in 0..8u8 {
            let bit = if x_flip { px } else { 7 - px };
            let color_idx = (((hi >> bit) & 1) << 1) | ((lo >> bit) & 1);

            self.bg_fifo.push_back(FifoPixel {
                color_index: color_idx,
                palette,
                is_sprite: false,
                bg_priority: bg_prio,
                sprite_bg_over: false,
                sprite_dmg_palette: 0,
                sprite_oam_index: 0,
                bg_color_index: 0,
                bg_palette: 0,
            });
        }
    }

    /// Output one pixel to the framebuffer, resolving BG/OAM priority.
    fn output_pixel(&mut self, bg: FifoPixel, oam: Option<FifoPixel>) {
        if self.position_in_line < 0 || self.position_in_line >= 160 {
            return;
        }
        let ly = self.ly as usize;
        if ly >= 144 {
            return;
        }

        // Determine if the sprite pixel wins over the BG pixel
        let draw_sprite = if let Some(ref oam_px) = oam {
            if oam_px.color_index == 0 {
                false // transparent OAM pixel
            } else if self.lcdc & 0x02 == 0 && !self.cgb_mode {
                // DMG: LCDC bit 1 off disables sprites entirely
                false
            } else if self.lcdc & 0x01 == 0 {
                // Both DMG and CGB: LCDC bit 0 off → sprite always wins
                // (DMG: BG disabled; CGB: BG priority disabled)
                true
            } else if oam_px.sprite_bg_over && bg.color_index != 0 {
                false // OAM attr bit 7: sprite behind non-zero BG
            } else if bg.bg_priority && bg.color_index != 0 {
                false // CGB BG attr bit 7: BG over sprite when non-zero
            } else {
                true
            }
        } else {
            false
        };

        let color32 = if draw_sprite {
            let oam_px = oam.unwrap();
            if self.cgb_mode {
                self.gbc_obj_color(oam_px.palette as usize, oam_px.color_index as usize)
            } else {
                // Use rendering palette (respects T3 write timing)
                let pal = if oam_px.sprite_dmg_palette == 1 { self.obp1_rendering } else { self.obp0_rendering };
                Self::dmg_color(pal, oam_px.color_index)
            }
        } else {
            // BG/window pixel
            if self.cgb_mode && !self.dmg_compat {
                // Native CGB: LCDC bit 0 doesn't disable BG display
                self.gbc_bg_color(bg.palette as usize, bg.color_index as usize)
            } else if self.cgb_mode && self.dmg_compat {
                // CGB DMG-compat: LCDC bit 0 disables BG display (DMG behavior)
                if self.lcdc & 0x01 == 0 {
                    self.gbc_bg_color(bg.palette as usize, 0)
                } else {
                    self.gbc_bg_color(bg.palette as usize, bg.color_index as usize)
                }
            } else if self.lcdc & 0x01 == 0 {
                // DMG: LCDC bit 0 off → BG/window draws as color 0
                Self::dmg_color(self.bgp_rendering, 0)
            } else {
                Self::dmg_color(self.bgp_rendering, bg.color_index)
            }
        };

        let fb_idx = ly * 160 + self.position_in_line as usize;

        // SGB: capture 2-bit shade index for palette remapping
        if self.sgb_mode {
            let (pal_reg, cidx) = if draw_sprite {
                let oam_px = oam.unwrap();
                let pal = if oam_px.sprite_dmg_palette == 1 { self.obp1_rendering } else { self.obp0_rendering };
                (pal, oam_px.color_index)
            } else {
                (self.bgp_rendering, bg.color_index)
            };
            let shade = (pal_reg >> (cidx * 2)) & 0x03;
            self.shade_buffer[fb_idx] = shade;
        }

        // On real DMG, the LCD doesn't display the first frame after LCD enable.
        // Suppress pixel output (render white) for that frame.
        if self.lcd_first_frame {
            self.frame_buffer[fb_idx] = 0x00FFFFFF;
        } else {
            self.frame_buffer[fb_idx] = color32;
        }
        self.position_in_line += 1;
    }

    // ---- Color conversion ----

    /// Convert GBC 15-bit palette color to 32-bit ARGB (0x00RRGGBB).
    fn gbc_15bit_to_32bit(c: u16) -> u32 {
        let r5 = (c & 0x1F) as u32;
        let g5 = ((c >> 5) & 0x1F) as u32;
        let b5 = ((c >> 10) & 0x1F) as u32;

        let r8 = (r5 << 3) | (r5 >> 2);
        let g8 = (g5 << 3) | (g5 >> 2);
        let b8 = (b5 << 3) | (b5 >> 2);

        (r8 << 16) | (g8 << 8) | b8
    }

    pub(super) fn gbc_bg_color(&self, palette_idx: usize, color_idx: usize) -> u32 {
        let offset = palette_idx * 8 + color_idx * 2;
        let lo = self.bcpd[offset] as u16;
        let hi = self.bcpd[offset + 1] as u16;
        let c = lo | (hi << 8);
        Self::gbc_15bit_to_32bit(c)
    }

    fn gbc_obj_color(&self, palette_idx: usize, color_idx: usize) -> u32 {
        let offset = palette_idx * 8 + color_idx * 2;
        let lo = self.ocpd[offset] as u16;
        let hi = self.ocpd[offset + 1] as u16;
        let c = lo | (hi << 8);
        Self::gbc_15bit_to_32bit(c)
    }

    /// DMG palette lookup: map a 2-bit color index through a palette register.
    /// Classic green Game Boy LCD colors.
    pub(super) const DMG_SHADES: [u32; 4] = [0x009BBC0F, 0x008BAC0F, 0x00306230, 0x000F380F];

    pub(super) fn dmg_color(palette_reg: u8, color_idx: u8) -> u32 {
        let shade = (palette_reg >> (color_idx * 2)) & 0x03;
        Self::DMG_SHADES[shade as usize]
    }
}
