/// Game Boy Color PPU (Pixel Processing Unit) — Pixel FIFO renderer
///
/// Timing (T-cycles per scanline = 456):
///   Mode 2: OAM scan        — 80 dots (dot 0..79)
///   Mode 3: Drawing          — variable, ends organically when 160 pixels output
///   Mode 0: HBlank           — remainder of 456
///   Mode 1: VBlank           — lines 144-153, 456 dots each
///   Total frame: 154 lines × 456 = 70224 T-cycles

// ---- Pixel FIFO types ----

#[derive(Clone, Copy, Default)]
struct FifoPixel {
    color_index: u8,        // 2-bit tile color (0-3)
    palette: u8,            // CGB palette (0-7), 0 for DMG
    is_sprite: bool,
    bg_priority: bool,      // CGB BG attr bit 7
    sprite_bg_over: bool,   // sprite OAM attr bit 7
    sprite_dmg_palette: u8, // DMG: captured obp0/obp1 value
    bg_color_index: u8,     // original BG color underneath sprite
    bg_palette: u8,         // original BG palette underneath sprite
}

struct PixelFifo {
    buf: [FifoPixel; 16],
    head: usize,
    count: usize,
}

impl PixelFifo {
    fn new() -> Self {
        Self {
            buf: [FifoPixel::default(); 16],
            head: 0,
            count: 0,
        }
    }

    fn len(&self) -> usize {
        self.count
    }

    fn clear(&mut self) {
        self.head = 0;
        self.count = 0;
    }

    fn push_back(&mut self, pixel: FifoPixel) {
        let idx = (self.head + self.count) & 15;
        self.buf[idx] = pixel;
        self.count += 1;
    }

    fn pop_front(&mut self) -> FifoPixel {
        let pixel = self.buf[self.head];
        self.head = (self.head + 1) & 15;
        self.count -= 1;
        pixel
    }

    fn get(&self, i: usize) -> &FifoPixel {
        &self.buf[(self.head + i) & 15]
    }

    fn replace(&mut self, i: usize, pixel: FifoPixel) {
        self.buf[(self.head + i) & 15] = pixel;
    }
}

#[derive(Clone, Copy, PartialEq)]
enum FetcherState {
    ReadTileId,
    ReadTileDataLow,
    ReadTileDataHigh,
    Push,
}

#[derive(Clone)]
struct Fetcher {
    state: FetcherState,
    tick: u8,               // counts 0-1 within each state (2T per step)
    fetching_window: bool,
    tile_x: u8,             // tiles fetched so far
    tile_id: u8,
    tile_attrs: u8,
    tile_data_low: u8,
    tile_data_high: u8,
}

impl Fetcher {
    fn new() -> Self {
        Self {
            state: FetcherState::ReadTileId,
            tick: 0,
            fetching_window: false,
            tile_x: 0,
            tile_id: 0,
            tile_attrs: 0,
            tile_data_low: 0,
            tile_data_high: 0,
        }
    }

    fn reset(&mut self, for_window: bool) {
        self.state = FetcherState::ReadTileId;
        self.tick = 0;
        self.fetching_window = for_window;
        self.tile_x = 0;
        self.tile_id = 0;
        self.tile_attrs = 0;
        self.tile_data_low = 0;
        self.tile_data_high = 0;
    }
}

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
    /// T-cycle counter within the current scanline (0..455)
    dot: u32,

    /// Previous state of internal STAT IRQ signal (for edge detection)
    stat_irq_line: bool,

    /// Sprites collected during Mode 2 OAM scan: (y, x, tile, attrs)
    scanline_sprites: Vec<(u8, u8, u8, u8)>,

    /// VRAM accessible (false during Mode 3)
    pub vram_accessible: bool,
    /// OAM accessible (false during Modes 2 and 3)
    pub oam_accessible: bool,

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

    /// True on the first scanline after LCD is enabled (special timing).
    lcd_first_line: bool,

    /// CGB: dot at which Mode 0 becomes visible in STAT (0 = not pending).
    mode0_stat_dot: u32,

    /// CGB: dot at which Mode 3 becomes visible in STAT (0 = not pending).
    mode3_stat_dot: u32,

    /// True = CGB game (uses CGB palettes), false = DMG game (uses BGP/OBP0/OBP1).
    pub cgb_mode: bool,

    /// True = CGB hardware running a DMG game (DMG compatibility mode).
    /// Uses CGB palette RAM but with DMG-style palette selection.
    pub dmg_compat: bool,

    /// Reference colors for DMG compat mode (RGB555).
    /// Set by boot ROM or default grayscale when no boot ROM.
    pub dmg_bg_ref: [u16; 4],
    pub dmg_obj_ref: [[u16; 4]; 2],

    // ---- Pixel FIFO state ----
    bg_fifo: PixelFifo,
    fetcher: Fetcher,
    /// Pixels pushed to framebuffer this scanline (0-160)
    pixel_x: u8,
    /// SCX%8 pixels to discard from first BG tile
    scx_discard: u8,
    /// Sprite fetch in progress
    sprite_fetch_active: bool,
    sprite_fetch_step: u8,   // 0=tile_id, 1=data_lo, 2=data_hi
    sprite_fetch_tick: u8,   // 0-1 within each step (2T per step)
    /// Alignment delay before sprite fetch begins (BG fetcher keeps running)
    sprite_alignment_delay: u8,
    sprite_fetch_entry: usize, // index into scanline_sprites
    sprite_tile_data_low: u8,
    sprite_tile_data_high: u8,
    /// Bitmask of already-fetched sprites (up to 10)
    sprites_fetched: u16,
    /// Window became active this scanline
    window_active: bool,
    /// Startup delay at beginning of Mode 3 (pipeline priming)
    mode3_start_delay: u8,
    /// Last sprite tile slot for same-slot grouping (or -1 if none)
    last_sprite_slot: i16,
}

impl Ppu {
    pub fn new() -> Self {
        // Default all-white GBC palettes: 0xFFFF per color
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

            mode: 1, // Post-boot: VBlank
            dot: 0,

            stat_irq_line: false,
            scanline_sprites: Vec::with_capacity(10),

            vram_accessible: true,
            oam_accessible: true, // VBlank: accessible

            frame_buffer: vec![0u32; 160 * 144],
            frame_ready: false,

            window_line_counter: 0,
            wy_triggered: false,

            if_flags: 0,
            hblank_entered: false,
            lcd_first_line: false,
            mode0_stat_dot: 0,
            mode3_stat_dot: 0,
            cgb_mode: true,
            dmg_compat: false,
            dmg_bg_ref: [0x7FFF; 4],
            dmg_obj_ref: [[0x7FFF; 4]; 2],

            bg_fifo: PixelFifo::new(),
            fetcher: Fetcher::new(),
            pixel_x: 0,
            scx_discard: 0,
            sprite_fetch_active: false,
            sprite_fetch_step: 0,
            sprite_fetch_tick: 0,
            sprite_alignment_delay: 0,
            sprite_fetch_entry: 0,
            sprite_tile_data_low: 0,
            sprite_tile_data_high: 0,
            sprites_fetched: 0,
            window_active: false,
            mode3_start_delay: 0,
            last_sprite_slot: -1,
        }
    }

    /// Reset PPU to hardware power-on state (for boot ROM execution).
    /// LCD is off, all registers zeroed, palettes zeroed.
    pub fn reset(&mut self) {
        self.lcdc = 0x00; // LCD off
        self.stat = 0x00;
        self.scy = 0;
        self.scx = 0;
        self.ly = 0;
        self.lyc = 0;
        self.bgp = 0;
        self.obp0 = 0;
        self.obp1 = 0;
        self.wy = 0;
        self.wx = 0;
        self.dma = 0;
        self.bcps = 0;
        self.bcpd = [0u8; 64];
        self.ocps = 0;
        self.ocpd = [0u8; 64];
        self.mode = 0;
        self.dot = 0;
        self.stat_irq_line = false;
        self.vram_accessible = true;
        self.oam_accessible = true;
        self.frame_ready = false;
        self.hblank_entered = false;
        self.lcd_first_line = false;
        self.mode0_stat_dot = 0;
        self.mode3_stat_dot = 0;
        self.window_line_counter = 0;
        self.wy_triggered = false;
        self.bg_fifo.clear();
        self.fetcher.reset(false);
        self.pixel_x = 0;
        self.scx_discard = 0;
        self.sprite_fetch_active = false;
        self.sprites_fetched = 0;
        self.window_active = false;
        self.mode3_start_delay = 0;
        self.last_sprite_slot = -1;
    }

    /// Step the PPU by `cycles` T-cycles.
    /// Returns interrupt flags (bit0=VBlank, bit1=STAT) to OR into IF.
    pub fn step(&mut self, cycles: u32) -> u8 {
        self.if_flags = 0;

        if self.lcdc & 0x80 == 0 {
            return 0;
        }

        for _ in 0..cycles {
            self.tick();
        }

        let flags = self.if_flags;
        self.if_flags = 0;
        flags
    }

    /// Advance the PPU by one T-cycle.
    fn tick(&mut self) {
        self.dot += 1;

        match self.mode {
            2 => {
                // Mode 2 → Mode 3: internal transition at dot 80, STAT bits update 1T later
                if self.dot >= 80 {
                    self.mode = 3;
                    self.oam_accessible = false;
                    self.vram_accessible = false;
                    self.init_fifo();
                    // Delay STAT mode bits update by 1T
                    self.mode3_stat_dot = self.dot + 1;
                    self.update_stat_irq();
                }
            }
            3 => {
                // Delayed STAT mode bits update for Mode 2→3
                if self.mode3_stat_dot > 0 && self.dot >= self.mode3_stat_dot {
                    self.mode3_stat_dot = 0;
                    self.stat = (self.stat & !0x03) | 0x03;
                }
                // Run per-pixel FIFO logic
                self.tick_mode3();
                // Check if scanline is complete
                if self.pixel_x >= 160 {
                    self.mode = 0;
                    self.mode0_stat_dot = self.dot + 1;
                    self.update_stat_irq();
                    if self.window_active {
                        self.window_line_counter = self.window_line_counter.wrapping_add(1);
                    }
                }
            }
            0 => {
                // CGB: delayed mode bit update for Mode 0
                if self.mode0_stat_dot > 0 && self.dot >= self.mode0_stat_dot {
                    self.mode0_stat_dot = 0;
                    self.stat = self.stat & !0x03; // mode bits = 0
                    self.oam_accessible = true;
                    self.vram_accessible = true;
                    self.hblank_entered = true;
                }
                // LCD first-line: STAT mode bits stay 0 for ~80 dots, then skip to mode 3
                if self.lcd_first_line && self.dot >= 80 {
                    self.lcd_first_line = false;
                    self.oam_scan(); // collect sprites
                    self.transition_to_mode3();
                    return;
                }
                // Mode 0 → end of scanline at dot 456
                if self.dot >= 456 {
                    self.dot = 0;
                    self.ly = self.ly.wrapping_add(1);
                    self.update_coincidence();

                    if self.ly == 144 {
                        self.transition_to_mode1();
                    } else {
                        self.transition_to_mode2();
                    }
                }
            }
            1 => {
                // VBlank lines
                if self.dot >= 456 {
                    self.dot = 0;
                    self.ly = self.ly.wrapping_add(1);
                    self.update_coincidence();

                    if self.ly > 153 {
                        // End of VBlank, back to line 0
                        self.ly = 0;
                        self.update_coincidence();
                        self.frame_ready = true;
                        self.window_line_counter = 0;
                        self.wy_triggered = false;
                        self.transition_to_mode2();
                    } else {
                        self.update_stat_irq();
                    }
                }
            }
            _ => {}
        }
    }

    // ---- Mode transitions ----

    fn transition_to_mode2(&mut self) {
        self.mode = 2;
        self.stat = (self.stat & !0x03) | 0x02;
        self.oam_accessible = false;
        self.vram_accessible = true;
        // OAM scan: collect sprites for this scanline
        self.oam_scan();
        // Check WY trigger at start of scanline
        if self.lcdc & 0x20 != 0 && self.ly == self.wy {
            self.wy_triggered = true;
        }
        self.update_stat_irq();
    }

    fn transition_to_mode3(&mut self) {
        self.mode = 3;
        self.stat = (self.stat & !0x03) | 0x03;
        self.oam_accessible = false;
        self.vram_accessible = false;
        self.init_fifo();
        self.update_stat_irq();
    }

    fn transition_to_mode1(&mut self) {
        self.mode = 1;
        self.stat = (self.stat & !0x03) | 0x01;
        self.oam_accessible = true;
        self.vram_accessible = true;
        // VBlank interrupt always fires
        self.if_flags |= 0x01;
        // Hardware quirk: Mode 2 source also fires at VBlank entry (one-shot)
        self.update_stat_irq_with_mode2(true);
    }

    // ---- Pixel FIFO ----

    /// Initialize FIFO state at the start of Mode 3
    fn init_fifo(&mut self) {
        self.bg_fifo.clear();
        self.fetcher.reset(false);
        self.pixel_x = 0;
        self.scx_discard = self.scx & 7;
        self.sprite_fetch_active = false;
        self.sprites_fetched = 0;
        self.window_active = false;
        // Hardware pipeline priming: 5T delay before fetcher starts
        self.mode3_start_delay = 5;
        self.last_sprite_slot = -1;
    }

    /// One T-cycle of Mode 3 pixel FIFO processing
    fn tick_mode3(&mut self) {
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

        // Advance BG fetcher
        self.tick_bg_fetcher();

        // Check sprite trigger
        if self.lcdc & 0x02 != 0 && self.bg_fifo.len() > 0 {
            if let Some(sprite_idx) = self.find_sprite_at_pixel_x() {
                self.start_sprite_fetch(sprite_idx);
                // Process first cycle immediately to avoid off-by-one penalty
                if self.sprite_alignment_delay > 0 {
                    self.sprite_alignment_delay -= 1;
                } else {
                    self.tick_sprite_fetch();
                }
                return;
            }
        }

        // Pop pixel from FIFO and output
        if self.bg_fifo.len() > 0 {
            let pixel = self.bg_fifo.pop_front();

            if self.scx_discard > 0 {
                self.scx_discard -= 1;
                return;
            }

            // Check window trigger before outputting
            if self.check_window_trigger() {
                self.bg_fifo.clear();
                self.fetcher.reset(true);
                self.window_active = true;
                self.scx_discard = 0;
                return;
            }

            self.output_pixel(pixel);
        }
    }

    /// Check if window should activate at current pixel_x
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
        // WX=0..166 maps to screen pixel (WX-7)..
        // Window triggers when pixel_x == WX-7 (for WX >= 7)
        // For WX < 7, window triggers at pixel_x == 0
        let wx_screen = if self.wx >= 7 { self.wx - 7 } else { 0 };
        self.pixel_x == wx_screen
    }

    /// Find a sprite that triggers at the current pixel_x
    fn find_sprite_at_pixel_x(&self) -> Option<usize> {
        for (i, &(_y, x, _tile, _attrs)) in self.scanline_sprites.iter().enumerate() {
            if self.sprites_fetched & (1 << i) != 0 {
                continue; // already fetched
            }
            // Sprite triggers when pixel_x reaches sprite_x - 8
            // For sprites with X < 8, they trigger at pixel_x == 0
            let trigger_x = if x >= 8 { x - 8 } else { 0 };
            if self.pixel_x == trigger_x {
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

        let &(raw_y, raw_x, mut tile_idx, attrs) = &self.scanline_sprites[self.sprite_fetch_entry];
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
                self.mix_sprite_into_fifo(raw_x, attrs);
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

    /// Mix fetched sprite data into the BG FIFO
    fn mix_sprite_into_fifo(&mut self, sprite_x: u8, attrs: u8) {
        let x_flip = attrs & 0x20 != 0;
        let bg_over = attrs & 0x80 != 0;
        let palette_idx = if self.cgb_mode && !self.dmg_compat {
            attrs & 0x07
        } else if self.dmg_compat {
            if attrs & 0x10 != 0 { 1 } else { 0 }
        } else {
            0
        };
        let dmg_pal = if attrs & 0x10 != 0 { self.obp1 } else { self.obp0 };

        let lo = self.sprite_tile_data_low;
        let hi = self.sprite_tile_data_high;

        // Determine the start offset into the FIFO
        // If sprite_x < 8, some leftmost pixels are clipped
        let start_pixel = if sprite_x < 8 { 8 - sprite_x } else { 0 };

        for px in start_pixel..8 {
            let fifo_pos = if sprite_x >= 8 {
                (sprite_x as i16 - 8 - self.pixel_x as i16 + px as i16) as usize
            } else {
                (px - start_pixel) as usize
            };

            if fifo_pos >= self.bg_fifo.len() {
                break;
            }

            let bit = if x_flip { px } else { 7 - px };
            let color_idx = (((hi >> bit) & 1) << 1) | ((lo >> bit) & 1);

            if color_idx == 0 {
                continue; // transparent sprite pixel
            }

            let existing = *self.bg_fifo.get(fifo_pos);
            if existing.is_sprite {
                continue; // first sprite wins
            }

            self.bg_fifo.replace(fifo_pos, FifoPixel {
                color_index: color_idx,
                palette: palette_idx,
                is_sprite: true,
                bg_priority: existing.bg_priority,
                sprite_bg_over: bg_over,
                sprite_dmg_palette: dmg_pal,
                bg_color_index: existing.color_index,
                bg_palette: existing.palette,
            });
        }
    }

    /// Advance the BG/window tile fetcher by one T-cycle
    fn tick_bg_fetcher(&mut self) {
        self.fetcher.tick += 1;
        if self.fetcher.tick < 2 {
            return; // each step takes 2T
        }
        self.fetcher.tick = 0;

        match self.fetcher.state {
            FetcherState::ReadTileId => {
                let map_addr = self.fetcher_map_addr();
                self.fetcher.tile_id = self.vram[0][map_addr];
                self.fetcher.tile_attrs = if self.cgb_mode {
                    self.vram[1][map_addr]
                } else {
                    0
                };
                self.fetcher.state = FetcherState::ReadTileDataLow;
            }
            FetcherState::ReadTileDataLow => {
                let (addr, bank) = self.fetcher_tile_data_addr();
                self.fetcher.tile_data_low = self.vram[bank][addr];
                self.fetcher.state = FetcherState::ReadTileDataHigh;
            }
            FetcherState::ReadTileDataHigh => {
                let (addr, bank) = self.fetcher_tile_data_addr();
                self.fetcher.tile_data_high = self.vram[bank][addr + 1];
                self.fetcher.state = FetcherState::Push;
            }
            FetcherState::Push => {
                if self.bg_fifo.len() <= 8 {
                    self.push_bg_pixels();
                    self.fetcher.tile_x += 1;
                    self.fetcher.state = FetcherState::ReadTileId;
                } else {
                    // FIFO full, stall — re-tick this state next cycle
                    self.fetcher.tick = 1; // will trigger again next T-cycle
                }
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
            let tile_x = ((self.scx / 8).wrapping_add(self.fetcher.tile_x)) & 0x1F;
            let tile_y = (scroll_y as u16) / 8;
            (bg_map_base + tile_y * 32 + tile_x as u16) as usize
        }
    }

    /// Compute VRAM address and bank for tile data
    fn fetcher_tile_data_addr(&self) -> (usize, usize) {
        let tile_id = self.fetcher.tile_id;
        let attrs = self.fetcher.tile_attrs;
        let tile_data_signed = self.lcdc & 0x10 == 0;

        let tile_addr: u16 = if !tile_data_signed {
            tile_id as u16 * 16
        } else {
            (0x1000i32 + (tile_id as i8 as i32) * 16) as u16
        };

        let y_flip = self.cgb_mode && attrs & 0x40 != 0;
        let bank = if self.cgb_mode && attrs & 0x08 != 0 { 1 } else { 0 };

        let pixel_y = if self.fetcher.fetching_window {
            (self.window_line_counter & 7) as u8
        } else {
            self.scy.wrapping_add(self.ly) & 7
        };

        let row = if y_flip { 7 - pixel_y } else { pixel_y };
        let addr = (tile_addr + row as u16 * 2) as usize;
        (addr, bank)
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
                bg_color_index: 0,
                bg_palette: 0,
            });
        }
    }

    /// Output one pixel to the framebuffer
    fn output_pixel(&mut self, pixel: FifoPixel) {
        if self.pixel_x >= 160 {
            return;
        }
        let ly = self.ly as usize;
        if ly >= 144 {
            return;
        }

        let color32 = if pixel.is_sprite {
            // Sprite pixel — check priority
            let bg_color_idx = pixel.bg_color_index;
            let bg_is_zero = bg_color_idx == 0;

            let sprite_wins = if self.lcdc & 0x01 == 0 {
                // BG/Window master disable — sprite always wins
                true
            } else if pixel.sprite_bg_over && !bg_is_zero {
                false
            } else if pixel.bg_priority && !bg_is_zero {
                false
            } else {
                true
            };

            if sprite_wins {
                if self.cgb_mode {
                    self.gbc_obj_color(pixel.palette as usize, pixel.color_index as usize)
                } else {
                    Self::dmg_color(pixel.sprite_dmg_palette, pixel.color_index)
                }
            } else {
                // BG wins — use the stored BG color
                if self.cgb_mode {
                    self.gbc_bg_color(pixel.bg_palette as usize, bg_color_idx as usize)
                } else {
                    Self::dmg_color(self.bgp, bg_color_idx)
                }
            }
        } else {
            // BG/window pixel
            if self.cgb_mode {
                self.gbc_bg_color(pixel.palette as usize, pixel.color_index as usize)
            } else {
                Self::dmg_color(self.bgp, pixel.color_index)
            }
        };

        self.frame_buffer[ly * 160 + self.pixel_x as usize] = color32;
        self.pixel_x += 1;
    }

    // ---- Edge-triggered STAT interrupt ----

    fn update_stat_irq(&mut self) {
        self.update_stat_irq_with_mode2(false);
    }

    fn update_stat_irq_with_mode2(&mut self, force_mode2: bool) {
        let coincidence = self.stat & 0x04 != 0;
        let signal =
            (self.stat & 0x08 != 0 && self.mode == 0) ||  // bit3: Mode 0
            (self.stat & 0x10 != 0 && self.mode == 1) ||  // bit4: Mode 1
            (self.stat & 0x20 != 0 && (self.mode == 2 || force_mode2)) || // bit5: Mode 2
            (self.stat & 0x40 != 0 && coincidence);        // bit6: LYC=LY

        // Fire on rising edge only
        if signal && !self.stat_irq_line {
            self.if_flags |= 0x02;
        }
        self.stat_irq_line = signal;
    }

    fn update_coincidence(&mut self) {
        if self.ly == self.lyc {
            self.stat |= 0x04;
        } else {
            self.stat &= !0x04;
        }
    }

    // ---- OAM scan ----

    fn oam_scan(&mut self) {
        self.scanline_sprites.clear();
        let sprite_height: i16 = if self.lcdc & 0x04 != 0 { 16 } else { 8 };
        let ly = self.ly as i16;

        for i in 0..40usize {
            let sprite_y = self.oam[i * 4] as i16 - 16;
            let sprite_x = self.oam[i * 4 + 1];
            let tile_idx = self.oam[i * 4 + 2];
            let attrs = self.oam[i * 4 + 3];

            if ly >= sprite_y && ly < sprite_y + sprite_height {
                self.scanline_sprites.push((self.oam[i * 4], sprite_x, tile_idx, attrs));
                if self.scanline_sprites.len() >= 10 {
                    break;
                }
            }
        }
    }

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

    /// DMG palette lookup: map a 2-bit color index through a palette register to grayscale.
    const DMG_SHADES: [u32; 4] = [0x00FFFFFF, 0x00AAAAAA, 0x00555555, 0x00000000];

    fn dmg_color(palette_reg: u8, color_idx: u8) -> u32 {
        let shade = (palette_reg >> (color_idx * 2)) & 0x03;
        Self::DMG_SHADES[shade as usize]
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

    pub fn write(&mut self, addr: u16, val: u8) {
        match addr {
            0xFF40 => {
                let lcd_was_on = self.lcdc & 0x80 != 0;
                self.lcdc = val;
                let lcd_now_on = self.lcdc & 0x80 != 0;

                if lcd_was_on && !lcd_now_on {
                    // LCD off: reset LY, dot, mode; preserve coincidence bit
                    // Do NOT reset stat_irq_line — hardware preserves the IRQ signal state
                    self.ly = 0;
                    self.dot = 0;
                    self.mode = 0;
                    self.stat = (self.stat & !0x03) | (self.stat & 0x04); // keep bit2
                    self.oam_accessible = true;
                    self.vram_accessible = true;
                    for p in self.frame_buffer.iter_mut() {
                        *p = 0x00FFFFFF;
                    }
                } else if !lcd_was_on && lcd_now_on {
                    // LCD on: start at line 0, mode reads as 0 initially
                    self.ly = 0;
                    self.dot = 0;
                    self.mode = 0;
                    self.stat = self.stat & !0x03; // mode bits = 0
                    self.oam_accessible = true;
                    self.vram_accessible = true;
                    self.lcd_first_line = true;
                    self.update_coincidence();
                    self.update_stat_irq();
                }
            }
            0xFF41 => {
                // Lower 3 bits (mode flags + coincidence) are read-only
                self.stat = (self.stat & 0x07) | (val & 0x78);
                // Writing STAT can change which sources are enabled → re-check edge
                if self.lcdc & 0x80 != 0 {
                    self.update_stat_irq();
                }
            }
            0xFF42 => self.scy = val,
            0xFF43 => self.scx = val,
            0xFF44 => {} // LY is read-only
            0xFF45 => {
                self.lyc = val;
                if self.lcdc & 0x80 != 0 {
                    self.update_coincidence();
                    self.update_stat_irq();
                }
            }
            0xFF46 => self.dma = val,
            0xFF47 => {
                self.bgp = val;
                if self.dmg_compat { self.sync_dmg_palette_to_cgb(val, false, 0); }
            }
            0xFF48 => {
                self.obp0 = val;
                if self.dmg_compat { self.sync_dmg_palette_to_cgb(val, true, 0); }
            }
            0xFF49 => {
                self.obp1 = val;
                if self.dmg_compat { self.sync_dmg_palette_to_cgb(val, true, 1); }
            }
            0xFF4A => self.wy = val,
            0xFF4B => self.wx = val,
            0xFF4F => self.vram_bank = (val & 0x01) as usize,
            0xFF68 => self.bcps = val & 0xBF,
            0xFF69 => {
                let idx = (self.bcps & 0x3F) as usize;
                self.bcpd[idx] = val;
                if self.bcps & 0x80 != 0 {
                    let next = (idx + 1) & 0x3F;
                    self.bcps = (self.bcps & 0x80) | next as u8;
                }
            }
            0xFF6A => self.ocps = val & 0xBF,
            0xFF6B => {
                let idx = (self.ocps & 0x3F) as usize;
                self.ocpd[idx] = val;
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
