/// Game Boy Color PPU (Pixel Processing Unit)
///
/// Timing (T-cycles):
///   Scanlines 0-143: Mode2(80) -> Mode3(172) -> Mode0(204) = 456 per line
///   Scanlines 144-153: Mode1(456 each) = 4560 total VBlank
///   Total frame: 70224 T-cycles

pub struct Ppu {
    /// VRAM banks 0 and 1 (8 KiB each)
    pub vram: [[u8; 0x2000]; 2],
    /// Current VRAM bank (0 or 1), controlled by 0xFF4F (VBK)
    pub vram_bank: usize,
    /// Object Attribute Memory (40 sprites x 4 bytes)
    pub oam: [u8; 0xA0],

    // PPU registers
    pub lcdc: u8, // 0xFF40
    pub stat: u8, // 0xFF41
    pub scy: u8,  // 0xFF42
    pub scx: u8,  // 0xFF43
    pub ly: u8,   // 0xFF44
    pub lyc: u8,  // 0xFF45
    pub bgp: u8,  // 0xFF47
    pub obp0: u8, // 0xFF48
    pub obp1: u8, // 0xFF49
    pub wy: u8,   // 0xFF4A
    pub wx: u8,   // 0xFF4B
    pub dma: u8,  // 0xFF46

    // GBC color palettes
    pub bcps: u8,      // 0xFF68: BG Color Palette Spec
    pub bcpd: [u8; 64], // 0xFF69: BG palette data (8 palettes x 4 colors x 2 bytes)
    pub ocps: u8,      // 0xFF6A: OBJ Color Palette Spec
    pub ocpd: [u8; 64], // 0xFF6B: OBJ palette data

    // Internal state
    mode: u8,
    cycles: u32,

    // Output
    pub frame_buffer: Vec<u32>,
    pub frame_ready: bool,

    // Window state
    window_line_counter: u8,
    wy_triggered: bool,

    /// Interrupt flags to set (VBlank=bit0, STAT=bit1)
    pub if_flags: u8,

    /// Set to true each time the PPU enters Mode 0 (HBlank); cleared by Bus.
    pub hblank_entered: bool,
}

impl Ppu {
    pub fn new() -> Self {
        // Default all-white GBC palettes: 0x7FFF per color (white in 15-bit BGR)
        // Each color is 2 bytes little-endian: 0xFF, 0x7F
        let mut bcpd = [0u8; 64];
        let mut ocpd = [0u8; 64];
        for i in 0..32 {
            bcpd[i * 2] = 0xFF;
            bcpd[i * 2 + 1] = 0xFF;
            ocpd[i * 2] = 0xFF;
            ocpd[i * 2 + 1] = 0xFF;
        }

        Ppu {
            vram: [[0u8; 0x2000]; 2],
            vram_bank: 0,
            oam: [0u8; 0xA0],

            lcdc: 0x91,
            stat: 0x85,
            scy: 0,
            scx: 0,
            ly: 0,
            lyc: 0,
            bgp: 0xFC,
            obp0: 0xFF,
            obp1: 0xFF,
            wy: 0,
            wx: 0,
            dma: 0,

            bcps: 0,
            bcpd,
            ocps: 0,
            ocpd,

            mode: 2,
            cycles: 0,

            frame_buffer: vec![0u32; 160 * 144],
            frame_ready: false,

            window_line_counter: 0,
            wy_triggered: false,

            if_flags: 0,
            hblank_entered: false,
        }
    }

    /// Step the PPU by `cycles` T-cycles.
    /// Returns interrupt flags (bit0=VBlank, bit1=STAT) to OR into IF.
    pub fn step(&mut self, cycles: u32) -> u8 {
        // If LCD is off, do nothing
        if self.lcdc & 0x80 == 0 {
            return 0;
        }

        self.if_flags = 0;

        for _ in 0..cycles {
            self.cycles += 1;

            match self.mode {
                // Mode 2: OAM scan (80 T-cycles)
                2 => {
                    if self.cycles >= 80 {
                        self.cycles -= 80;
                        self.enter_mode3();
                    }
                }

                // Mode 3: Drawing (172 T-cycles)
                3 => {
                    if self.cycles >= 172 {
                        self.cycles -= 172;
                        // Render scanline at END of mode 3
                        self.render_scanline();
                        self.enter_mode0();
                    }
                }

                // Mode 0: HBlank (204 T-cycles)
                0 => {
                    if self.cycles >= 204 {
                        self.cycles -= 204;
                        // Increment LY
                        self.ly = self.ly.wrapping_add(1);
                        self.check_lyc_coincidence();

                        if self.ly == 144 {
                            // Enter VBlank
                            self.enter_mode1();
                        } else {
                            // Next scanline: enter OAM scan
                            self.enter_mode2();
                        }
                    }
                }

                // Mode 1: VBlank (456 T-cycles per line, lines 144-153)
                1 => {
                    if self.cycles >= 456 {
                        self.cycles -= 456;
                        self.ly = self.ly.wrapping_add(1);

                        if self.ly > 153 {
                            // End of VBlank, back to line 0
                            self.ly = 0;
                            self.frame_ready = true;
                            self.window_line_counter = 0;
                            self.wy_triggered = false;
                            self.check_lyc_coincidence();
                            self.enter_mode2();
                        } else {
                            self.check_lyc_coincidence();
                        }
                    }
                }

                _ => {}
            }
        }

        let flags = self.if_flags;
        self.if_flags = 0;
        flags
    }

    fn enter_mode2(&mut self) {
        self.mode = 2;
        self.stat = (self.stat & !0x03) | 0x02;
        // STAT interrupt if bit5 set
        if self.stat & 0x20 != 0 {
            self.if_flags |= 0x02;
        }
    }

    fn enter_mode3(&mut self) {
        self.mode = 3;
        self.stat = (self.stat & !0x03) | 0x03;
    }

    fn enter_mode0(&mut self) {
        self.mode = 0;
        self.stat = self.stat & !0x03; // bits 0-1 = 00
        // STAT interrupt if bit3 set
        if self.stat & 0x08 != 0 {
            self.if_flags |= 0x02;
        }
        self.hblank_entered = true;
    }

    fn enter_mode1(&mut self) {
        self.mode = 1;
        self.stat = (self.stat & !0x03) | 0x01;
        // VBlank interrupt always
        self.if_flags |= 0x01;
        // STAT interrupt if bit4 set
        if self.stat & 0x10 != 0 {
            self.if_flags |= 0x02;
        }
    }

    fn check_lyc_coincidence(&mut self) {
        if self.ly == self.lyc {
            self.stat |= 0x04; // Set coincidence flag (bit2)
            if self.stat & 0x40 != 0 {
                self.if_flags |= 0x02;
            }
        } else {
            self.stat &= !0x04; // Clear coincidence flag
        }
    }

    /// Render one scanline (self.ly) into frame_buffer.
    fn render_scanline(&mut self) {
        let ly = self.ly as usize;
        if ly >= 144 {
            return;
        }

        // Per-pixel background data for sprite priority checking
        let mut bg_color: [u32; 160] = [0u32; 160];
        let mut bg_priority: [bool; 160] = [false; 160]; // attr bit7 set
        let mut bg_color_zero: [bool; 160] = [true; 160]; // color index == 0

        // ---- Background Rendering ----
        // In GBC mode we always render BG (LCDC bit0 disables BG/Window priority instead)
        {
            let bg_map_base: u16 = if self.lcdc & 0x08 != 0 { 0x1C00 } else { 0x1800 };
            let tile_data_signed = self.lcdc & 0x10 == 0;

            for x in 0u16..160 {
                let scroll_x = self.scx as u16 + x;
                let scroll_y = self.scy as u16 + self.ly as u16;

                let tile_x = (scroll_x & 0xFF) / 8;
                let tile_y = (scroll_y & 0xFF) / 8;

                let map_addr = (bg_map_base + tile_y * 32 + tile_x) as usize;

                let tile_idx = self.vram[0][map_addr];
                let tile_attrs = self.vram[1][map_addr];

                let palette_idx = (tile_attrs & 0x07) as usize;
                let vram_bank_sel = if tile_attrs & 0x08 != 0 { 1usize } else { 0usize };
                let x_flip = tile_attrs & 0x20 != 0;
                let y_flip = tile_attrs & 0x40 != 0;
                let bg_prio = tile_attrs & 0x80 != 0;

                // Tile address in VRAM
                let tile_addr: u16 = if !tile_data_signed {
                    tile_idx as u16 * 16
                } else {
                    (0x1000i32 + (tile_idx as i8 as i32) * 16) as u16
                };

                let mut pixel_y_in_tile = (scroll_y & 0xFF) as u8 & 7;
                let mut pixel_x_in_tile = (scroll_x & 0xFF) as u8 & 7;

                if y_flip {
                    pixel_y_in_tile = 7 - pixel_y_in_tile;
                }
                if x_flip {
                    pixel_x_in_tile = 7 - pixel_x_in_tile;
                }

                let byte_addr = (tile_addr + pixel_y_in_tile as u16 * 2) as usize;
                let lo = self.vram[vram_bank_sel][byte_addr];
                let hi = self.vram[vram_bank_sel][byte_addr + 1];

                let bit = 7 - pixel_x_in_tile;
                let color_idx = (((hi >> bit) & 1) << 1) | ((lo >> bit) & 1);

                let color32 = self.gbc_bg_color(palette_idx, color_idx as usize);

                bg_color[x as usize] = color32;
                bg_priority[x as usize] = bg_prio;
                bg_color_zero[x as usize] = color_idx == 0;

                // Write BG pixel to frame buffer (will be overwritten by sprite if needed)
                self.frame_buffer[ly * 160 + x as usize] = color32;
            }
        }

        // ---- Window Rendering ----
        if self.lcdc & 0x20 != 0 {
            // Check WY trigger
            if self.ly == self.wy {
                self.wy_triggered = true;
            }

            if self.wy_triggered && self.wx <= 166 {
                let win_map_base: u16 = if self.lcdc & 0x40 != 0 { 0x1C00 } else { 0x1800 };
                let tile_data_signed = self.lcdc & 0x10 == 0;

                let wx_offset = if self.wx >= 7 { (self.wx - 7) as i32 } else { 0i32 };
                let mut rendered_any = false;

                for x in wx_offset..160i32 {
                    let win_x = (x - wx_offset) as u16;
                    let win_y = self.window_line_counter as u16;

                    let tile_x = win_x / 8;
                    let tile_y = win_y / 8;

                    let map_addr = (win_map_base + tile_y * 32 + tile_x) as usize;

                    let tile_idx = self.vram[0][map_addr];
                    let tile_attrs = self.vram[1][map_addr];

                    let palette_idx = (tile_attrs & 0x07) as usize;
                    let vram_bank_sel = if tile_attrs & 0x08 != 0 { 1usize } else { 0usize };
                    let x_flip = tile_attrs & 0x20 != 0;
                    let y_flip = tile_attrs & 0x40 != 0;
                    let bg_prio = tile_attrs & 0x80 != 0;

                    let tile_addr: u16 = if !tile_data_signed {
                        tile_idx as u16 * 16
                    } else {
                        (0x1000i32 + (tile_idx as i8 as i32) * 16) as u16
                    };

                    let mut pixel_y_in_tile = (win_y & 7) as u8;
                    let mut pixel_x_in_tile = (win_x & 7) as u8;

                    if y_flip {
                        pixel_y_in_tile = 7 - pixel_y_in_tile;
                    }
                    if x_flip {
                        pixel_x_in_tile = 7 - pixel_x_in_tile;
                    }

                    let byte_addr = (tile_addr + pixel_y_in_tile as u16 * 2) as usize;
                    let lo = self.vram[vram_bank_sel][byte_addr];
                    let hi = self.vram[vram_bank_sel][byte_addr + 1];

                    let bit = 7 - pixel_x_in_tile;
                    let color_idx = (((hi >> bit) & 1) << 1) | ((lo >> bit) & 1);

                    let color32 = self.gbc_bg_color(palette_idx, color_idx as usize);

                    bg_color[x as usize] = color32;
                    bg_priority[x as usize] = bg_prio;
                    bg_color_zero[x as usize] = color_idx == 0;

                    self.frame_buffer[ly * 160 + x as usize] = color32;
                    rendered_any = true;
                }

                if rendered_any {
                    self.window_line_counter = self.window_line_counter.wrapping_add(1);
                }
            }
        }

        // ---- Sprite Rendering ----
        if self.lcdc & 0x02 != 0 {
            let sprite_height: i16 = if self.lcdc & 0x04 != 0 { 16 } else { 8 };

            // Collect up to 10 visible sprites (OAM priority: lower index wins)
            let mut sprites: Vec<(i16, i16, u8, u8)> = Vec::with_capacity(10); // (sprite_y, sprite_x, tile_idx, attrs)

            for i in 0..40usize {
                let sprite_y = self.oam[i * 4 + 0] as i16 - 16;
                let sprite_x = self.oam[i * 4 + 1] as i16 - 8;
                let mut tile_idx = self.oam[i * 4 + 2];
                let attrs = self.oam[i * 4 + 3];

                let ly_i16 = self.ly as i16;
                if ly_i16 >= sprite_y && ly_i16 < sprite_y + sprite_height {
                    // In 8x16 mode, mask LSB of tile index
                    if sprite_height == 16 {
                        tile_idx &= 0xFE;
                    }
                    sprites.push((sprite_y, sprite_x, tile_idx, attrs));
                    if sprites.len() == 10 {
                        break;
                    }
                }
            }

            // Render in reverse order so lower OAM index wins (drawn last = on top)
            for &(sprite_y, sprite_x, tile_idx, attrs) in sprites.iter().rev() {
                let palette_idx = (attrs & 0x07) as usize;
                let vram_bank_sel = if attrs & 0x08 != 0 { 1usize } else { 0usize };
                let x_flip = attrs & 0x20 != 0;
                let y_flip = attrs & 0x40 != 0;
                let bg_over_sprite = attrs & 0x80 != 0;

                let mut row_in_sprite = (self.ly as i16 - sprite_y) as u16;

                if y_flip {
                    row_in_sprite = sprite_height as u16 - 1 - row_in_sprite;
                }

                // In 8x16 mode, choose top or bottom tile
                let actual_tile_idx = if sprite_height == 16 {
                    if row_in_sprite < 8 {
                        tile_idx & 0xFE
                    } else {
                        tile_idx | 0x01
                    }
                } else {
                    tile_idx
                };
                let row_in_tile = row_in_sprite % 8;

                let byte_addr = (actual_tile_idx as u16 * 16 + row_in_tile * 2) as usize;
                let lo = self.vram[vram_bank_sel][byte_addr];
                let hi = self.vram[vram_bank_sel][byte_addr + 1];

                for px in 0u8..8 {
                    let screen_x = sprite_x + if x_flip { 7 - px as i16 } else { px as i16 };

                    if screen_x < 0 || screen_x >= 160 {
                        continue;
                    }

                    let bit = 7 - px;
                    let color_idx = (((hi >> bit) & 1) << 1) | ((lo >> bit) & 1);

                    // Color 0 is transparent for sprites
                    if color_idx == 0 {
                        continue;
                    }

                    let sx = screen_x as usize;

                    // Priority resolution:
                    // LCDC bit0 = 0 in GBC: sprites always on top (master BG/WIN disable)
                    // LCDC bit0 = 1: use per-attribute priority
                    //   - if sprite attr bit7 set AND bg pixel is not color 0: BG wins
                    //   - if bg tile attr bit7 set AND bg pixel is not color 0: BG wins
                    let sprite_wins = if self.lcdc & 0x01 == 0 {
                        // Master priority: sprites always win
                        true
                    } else if bg_over_sprite && !bg_color_zero[sx] {
                        false
                    } else if bg_priority[sx] && !bg_color_zero[sx] {
                        false
                    } else {
                        true
                    };

                    if sprite_wins {
                        let color32 = self.gbc_obj_color(palette_idx, color_idx as usize);
                        self.frame_buffer[ly * 160 + sx] = color32;
                    }
                }
            }
        }
    }

    /// Convert GBC 15-bit palette color to 32-bit ARGB (0x00RRGGBB).
    fn gbc_15bit_to_32bit(c: u16) -> u32 {
        let r5 = (c & 0x1F) as u32;
        let g5 = ((c >> 5) & 0x1F) as u32;
        let b5 = ((c >> 10) & 0x1F) as u32;

        // Expand 5-bit to 8-bit: val8 = (val5 << 3) | (val5 >> 2)
        let r8 = (r5 << 3) | (r5 >> 2);
        let g8 = (g5 << 3) | (g5 >> 2);
        let b8 = (b5 << 3) | (b5 >> 2);

        (r8 << 16) | (g8 << 8) | b8
    }

    fn gbc_bg_color(&self, palette_idx: usize, color_idx: usize) -> u32 {
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

    // ---- I/O Register Access ----

    pub fn read(&self, addr: u16) -> u8 {
        match addr {
            0xFF40 => self.lcdc,
            0xFF41 => self.stat | 0x80,
            0xFF42 => self.scy,
            0xFF43 => self.scx,
            0xFF44 => self.ly,
            0xFF45 => self.lyc,
            0xFF46 => self.dma,
            0xFF47 => self.bgp,
            0xFF48 => self.obp0,
            0xFF49 => self.obp1,
            0xFF4A => self.wy,
            0xFF4B => self.wx,
            0xFF4F => self.vram_bank as u8 | 0xFE,
            0xFF68 => self.bcps | 0x40,
            0xFF69 => self.bcpd[(self.bcps & 0x3F) as usize],
            0xFF6A => self.ocps | 0x40,
            0xFF6B => self.ocpd[(self.ocps & 0x3F) as usize],
            _ => 0xFF,
        }
    }

    pub fn write(&mut self, addr: u16, val: u8) {
        match addr {
            0xFF40 => {
                let lcd_was_on = self.lcdc & 0x80 != 0;
                self.lcdc = val;
                let lcd_now_on = self.lcdc & 0x80 != 0;
                // If LCD just turned off, reset LY and mode
                if lcd_was_on && !lcd_now_on {
                    self.ly = 0;
                    self.mode = 0;
                    self.cycles = 0;
                    self.stat &= !0x03;
                    // Blank the screen
                    for p in self.frame_buffer.iter_mut() {
                        *p = 0x00FFFFFF; // white when off
                    }
                }
            }
            0xFF41 => {
                // Lower 3 bits (mode flags + coincidence) are read-only
                self.stat = (self.stat & 0x07) | (val & 0x78);
            }
            0xFF42 => self.scy = val,
            0xFF43 => self.scx = val,
            0xFF44 => {} // LY is read-only
            0xFF45 => {
                self.lyc = val;
                self.check_lyc_coincidence();
            }
            0xFF46 => self.dma = val,
            0xFF47 => self.bgp = val,
            0xFF48 => self.obp0 = val,
            0xFF49 => self.obp1 = val,
            0xFF4A => self.wy = val,
            0xFF4B => self.wx = val,
            0xFF4F => self.vram_bank = (val & 0x01) as usize,
            0xFF68 => self.bcps = val & 0xBF,
            0xFF69 => {
                let idx = (self.bcps & 0x3F) as usize;
                self.bcpd[idx] = val;
                // Auto-increment if bit7 set
                if self.bcps & 0x80 != 0 {
                    let next = (idx + 1) & 0x3F;
                    self.bcps = (self.bcps & 0x80) | next as u8;
                }
            }
            0xFF6A => self.ocps = val & 0xBF,
            0xFF6B => {
                let idx = (self.ocps & 0x3F) as usize;
                self.ocpd[idx] = val;
                // Auto-increment if bit7 set
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

impl Default for Ppu {
    fn default() -> Self {
        Self::new()
    }
}
