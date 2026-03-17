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
    sprite_dmg_palette: u8, // DMG: 0=OBP0, 1=OBP1 (palette selected at output time)
    sprite_oam_index: u8,   // OAM entry index (0-39) for CGB priority
    bg_color_index: u8,     // original BG color underneath sprite
    bg_palette: u8,         // original BG palette underneath sprite
}

#[derive(Clone)]
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

/// BG/Window tile fetcher states. Each state takes exactly 1 T-cycle.
/// States come in pairs: T1 (latch/address) and T2 (VRAM read/execute).
/// Push stalls (repeats) if the BG FIFO is not yet empty.
#[derive(Clone, Copy, PartialEq)]
enum FetcherState {
    GetTileT1,
    GetTileT2,
    GetTileDataLowT1,
    GetTileDataLowT2,
    GetTileDataHighT1,
    GetTileDataHighT2,
    Push,
}

#[derive(Clone)]
struct Fetcher {
    state: FetcherState,
    fetching_window: bool,
    tile_x: u8,             // tiles fetched so far
    tile_id: u8,
    tile_attrs: u8,
    tile_data_low: u8,
    tile_data_high: u8,
    /// Latched tile data address (computed at T1 of data fetch, used at T2)
    latched_addr: usize,
    latched_bank: usize,
    /// CGB: fetcher_y latched at GetTileT1 (CGB-D+ caches this)
    fetcher_y: u8,
    /// CGB: LCDC TILE_SEL (bit 4) latched at GetTileT1
    latched_tile_sel: bool,
    /// CGB: BG map address latched at GetTileT1
    /// (cached at GET_TILE T1, used at T2 for VRAM read)
    latched_map_addr: usize,
}

impl Fetcher {
    fn new() -> Self {
        Self {
            state: FetcherState::GetTileT1,
            fetching_window: false,
            tile_x: 0,
            tile_id: 0,
            tile_attrs: 0,
            tile_data_low: 0,
            tile_data_high: 0,
            latched_addr: 0,
            latched_bank: 0,
            fetcher_y: 0,
            latched_tile_sel: false,
            latched_map_addr: 0,
        }
    }

    fn reset(&mut self, for_window: bool) {
        self.state = FetcherState::GetTileT1;
        self.fetching_window = for_window;
        self.tile_x = 0;
        self.tile_id = 0;
        self.tile_attrs = 0;
        self.tile_data_low = 0;
        self.tile_data_high = 0;
        self.latched_addr = 0;
        self.latched_bank = 0;
        self.fetcher_y = 0;
        self.latched_tile_sel = false;
        self.latched_map_addr = 0;
    }
}

/// Double each bit in a 4-bit nibble to produce an 8-bit value.
/// e.g., 0b1010 → 0b11001100. Matches the DMG boot ROM's logo decompression.
fn double_bits(nibble: u8) -> u8 {
    let mut result = 0u8;
    for i in 0..4 {
        let bit = (nibble >> (3 - i)) & 1;
        result |= bit << (7 - i * 2);
        result |= bit << (6 - i * 2);
    }
    result
}

#[derive(Clone)]
pub struct Ppu {
    /// VRAM banks 0 and 1 (8 KiB each)
    pub vram: [[u8; 0x2000]; 2],
    /// Current VRAM bank (0 or 1), controlled by 0xFF4F (VBK)
    pub vram_bank: usize,
    /// Object Attribute Memory (40 sprites x 4 bytes = 160 bytes).
    /// Extended to 192 bytes to handle OAM bug corruption at accessed_oam_row > 152
    pub oam: [u8; 192],

    // PPU registers
    pub lcdc: u8, // 0xFF40
    pub stat: u8, // 0xFF41
    pub scy: u8,  // 0xFF42
    pub scx: u8,  // 0xFF43
    pub ly: u8,   // 0xFF44
    pub lyc: u8,  // 0xFF45
    /// Deferred LYC value: when CPU writes LYC during CGB line_start_pending,
    /// the write is deferred until after the current step(4) completes.
    pending_lyc: Option<u8>,
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
    pub(crate) mode: u8,
    /// T-cycle counter within the current scanline (0..455)
    pub(crate) dot: u32,

    /// Debug: total ticks since LCD enable
    pub(crate) total_ticks: u64,

    /// Previous state of internal STAT IRQ signal (for edge detection)
    stat_irq_line: bool,

    /// Sprites collected during Mode 2 OAM scan: (y, x, tile, attrs, oam_index)
    scanline_sprites: Vec<(u8, u8, u8, u8, u8)>,

    /// VRAM read accessible (false during Mode 3, and late Mode 2 on DMG)
    pub vram_accessible: bool,
    /// VRAM write accessible (false during Mode 3 only)
    pub vram_write_accessible: bool,
    /// OAM read accessible (false during Modes 2 and 3)
    pub oam_accessible: bool,
    /// OAM write accessible (false during Mode 2 early and Mode 3; unblocked at Mode 2 index 37)
    pub oam_write_accessible: bool,

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
    /// True during Mode 0 of the first line after LCD enable (shortens line by 7T).
    lcd_first_line_short: bool,
    /// True for the entire first frame after LCD enable (pixels render as white).
    /// On real DMG hardware the LCD panel doesn't display the first frame.
    lcd_first_frame: bool,

    /// CGB: dot at which Mode 0 becomes visible in STAT (0 = not pending).
    mode0_stat_dot: u32,

    /// CGB: dot at which Mode 3 becomes visible in STAT (0 = not pending).
    mode3_stat_dot: u32,

    /// CGB: palette RAM blocked during most of mode 3.
    pub cgb_palettes_blocked: bool,
    /// CGB: dot at which palette blocking clears (0 = not pending).
    cgb_palette_unblock_dot: u32,

    /// True = CGB game (uses CGB palettes), false = DMG game (uses BGP/OBP0/OBP1).
    pub cgb_mode: bool,

    /// CGB double-speed mode active.
    pub double_speed: bool,

    /// SGB mode: capture 2-bit shade indices for SGB palette remapping.
    pub sgb_mode: bool,
    /// Shade buffer: 160×144 of 2-bit shade indices (written during rendering).
    pub shade_buffer: Vec<u8>,

    /// True = CGB hardware running a DMG game (DMG compatibility mode).
    /// Uses CGB palette RAM but with DMG-style palette selection.
    pub dmg_compat: bool,

    /// $FF6C OPRI: Object priority mode (CGB only)
    /// bit 0: 0 = OAM index priority (CGB default), 1 = X-coordinate priority (DMG mode)
    pub opri: u8,

    /// CGB tile_sel_glitch: set for 1T when LCDC bit 4 transitions 1→0.
    /// Causes tile data fetch to read from a glitched address.
    pub tile_sel_glitch: bool,
    /// Latched at T1 if tile_sel_glitch was true; consumed at T2 to apply glitch data.
    tile_sel_glitch_latched: bool,

    /// Reference colors for DMG compat mode (RGB555).
    /// Set by boot ROM or default grayscale when no boot ROM.
    pub dmg_bg_ref: [u16; 4],
    pub dmg_obj_ref: [[u16; 4]; 2],

    // ---- Pixel FIFO state ----
    bg_fifo: PixelFifo,
    /// Separate OAM (sprite) FIFO — popped in lockstep with bg_fifo.
    /// Sprite pixels are stored here rather than overlaid into bg_fifo,
    /// so priority is resolved correctly at output time.
    oam_fifo: PixelFifo,
    fetcher: Fetcher,
    /// Pixel position in current scanline. Tracks progress through the
    /// rendering pipeline using signed coordinates:
    ///   -16..-9 : junk pixel zone (pre-filled FIFO garbage being consumed)
    ///   -8..-1  : SCX fractional scroll discard zone
    ///   0..159  : visible screen pixels (maps to framebuffer X)
    ///   160     : scanline complete, triggers mode 0
    /// Replaces separate pixel_x + scx_discard counters with a unified model.
    pub(crate) position_in_line: i16,
    /// Sprite fetch in progress
    pub(crate) sprite_fetch_active: bool,
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
    /// Deferred window activation: set when WX matches on the last tick of
    /// a step(). Activation happens on the first tick of the next step().
    /// This allows LCDC writes between steps to cancel window activation.
    window_trigger_pending: bool,
    window_trigger_from_wx_write: bool,
    /// DMG glitch: when WIN_EN is disabled while window is being fetched,
    /// suppress the phantom window pixel insertion at the fetcher push state.
    pub(crate) disable_window_pixel_insertion_glitch: bool,
    /// True when processing the last tick of a step() call.
    last_tick_of_step: bool,
    /// True when processing the first tick of a step() call.
    first_tick_of_step: bool,
    /// Startup delay at beginning of Mode 3 (pipeline priming)
    pub(crate) mode3_start_delay: u8,
    /// Debug: dot when mode 3 started
    mode3_dot: u32,
    /// Last sprite tile slot for same-slot grouping (or -1 if none)
    last_sprite_slot: i16,

    /// DMG line-start timing: LY value visible via IO register FF44.
    /// On DMG, delayed from internal ly until dot 4 of the new scanline.
    visible_ly: u8,
    /// DMG line-start timing: LY value used for LYC comparison.
    /// -1 means "no match possible" (preserves existing coincidence bit).
    ly_for_comparison: i16,
    /// DMG line-start timing: true during dots 0-4 of a new scanline.
    line_start_pending: bool,
    /// DMG line-start timing: true if line_start_pending is for a VBlank line.
    line_start_is_vblank: bool,
    /// DMG line-start timing: mode override for STAT IRQ check (-1 = use self.mode).
    mode_for_interrupt: i8,
    /// Line 153 extended state machine: tracks the multi-step LY 153→0 transition.
    /// 0 = not active, >0 = active at this step of the sequence.
    line_153_phase: u8,

    /// Per-entry OAM scan: index of next OAM entry to check (0-40).
    /// During mode 2, one entry is checked every 2T using the current LCDC bit 2.
    oam_scan_index: u8,

    /// DMG OAM bug: the OAM row currently being accessed by the PPU during Mode 2.
    /// 0xFF = not in Mode 2 (no OAM bug possible). Only used on DMG models.
    pub accessed_oam_row: i16,

    /// OAM bug row captured at the end of each step(4) call.
    /// The CPU checks this value BEFORE the next tick_mcycle(), so it
    /// reflects the PPU state after the previous M-cycle's advancement.
    pub oam_bug_row: i16,

    // ---- DMG palette write timing ----
    // On hardware, CPU writes take effect at T3 of the M-cycle. During the
    // transition T-cycle (T3), the PPU reads (old_value | new_value) — a bus
    // conflict glitch. At T4, the real new value is used.
    // We track separate "rendering" palette values that implement this timing.
    pub(crate) bgp_rendering: u8,
    pub(crate) obp0_rendering: u8,
    pub(crate) obp1_rendering: u8,
    /// Ring buffer of last 2 pixel metadata for retroactive correction.
    /// Each entry: (fb_idx, pal_type: 0=BGP/1=OBP0/2=OBP1, color_index, bg_color_index, is_sprite).
    pixel_history: [(usize, u8, u8, u8, bool); 2],
    /// Number of valid entries in pixel_history (0, 1, or 2).
    pixel_history_count: u8,
    /// Next write position in pixel_history (0 or 1).
    pixel_history_next: usize,
    /// WX write conflict: suppresses WX+6 window trigger for 1T after WX write
    pub(crate) wx_just_changed: bool,
    /// Skip retroactive LCDC fix: set when bus handler already applied glitch timing
    pub(crate) skip_retroactive_lcdc_fix: bool,
    /// Junk zone: set when position_in_line is in [-16, -9] and SCX alignment
    /// hasn't matched yet. Used by hardware for mid-scanline SCX glitches.
    line_has_fractional_scrolling: bool,
    /// True while the fetcher is actively fetching window tiles (set when window
    /// activates, cleared by render_pixel_if_possible after first pixel pop).
    window_is_being_fetched: bool,

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
            oam: [0u8; 192],

            lcdc: 0x91,
            stat: 0x85,
            scy: 0,
            scx: 0,
            ly: 0x00,  // Updated per-model in Bus::new()
            lyc: 0,
            bgp: 0xFC,
            obp0: 0xFF,
            obp1: 0xFF,
            wy: 0,
            wx: 0,
            dma: 0xFF,

            bcps: 0,
            bcpd,
            ocps: 0,
            ocpd,

            mode: 1, // Post-boot: VBlank
            dot: 0,
            total_ticks: 0,

            stat_irq_line: false,
            scanline_sprites: Vec::with_capacity(10),

            vram_accessible: true,
            vram_write_accessible: true,
            oam_accessible: true, // VBlank: accessible
            oam_write_accessible: true,

            frame_buffer: vec![0u32; 160 * 144],
            frame_ready: false,

            window_line_counter: 0,
            wy_triggered: false,

            if_flags: 0,
            hblank_entered: false,
            lcd_first_line: false,
            lcd_first_line_short: false,
            lcd_first_frame: false,
            mode0_stat_dot: 0,
            mode3_stat_dot: 0,
            cgb_palettes_blocked: false,
            cgb_palette_unblock_dot: 0,
            cgb_mode: true,
            double_speed: false,
            sgb_mode: false,
            shade_buffer: vec![0u8; 160 * 144],
            dmg_compat: false,
            opri: 0,
            tile_sel_glitch: false,
            tile_sel_glitch_latched: false,
            dmg_bg_ref: [0x7FFF; 4],
            dmg_obj_ref: [[0x7FFF; 4]; 2],

            bg_fifo: PixelFifo::new(),
            oam_fifo: PixelFifo::new(),
            fetcher: Fetcher::new(),
            position_in_line: 0,
            sprite_fetch_active: false,
            sprite_fetch_step: 0,
            sprite_fetch_tick: 0,
            sprite_alignment_delay: 0,
            sprite_fetch_entry: 0,
            sprite_tile_data_low: 0,
            sprite_tile_data_high: 0,
            sprites_fetched: 0,
            window_active: false,
            window_trigger_pending: false,
            window_trigger_from_wx_write: false,
            disable_window_pixel_insertion_glitch: false,
            last_tick_of_step: false,
            first_tick_of_step: false,
            mode3_start_delay: 0,
            mode3_dot: 0,
            last_sprite_slot: -1,
            visible_ly: 0,
            ly_for_comparison: 0,
            line_start_pending: false,
            line_start_is_vblank: false,
            mode_for_interrupt: -1,
            line_153_phase: 0,
            oam_scan_index: 0,
            accessed_oam_row: 0xFF,
            oam_bug_row: 0xFF,
            pending_lyc: None,
            bgp_rendering: 0xFC,
            obp0_rendering: 0xFF,
            obp1_rendering: 0xFF,
            pixel_history: [(0, 0, 0, 0, false); 2],
            pixel_history_count: 0,
            pixel_history_next: 0,
            wx_just_changed: false,
            skip_retroactive_lcdc_fix: false,
            line_has_fractional_scrolling: false,
            window_is_being_fetched: false,
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
        self.obp0 = 0xFF;
        self.obp1 = 0xFF;
        self.wy = 0;
        self.wx = 0;
        self.dma = 0xFF;
        self.bcps = 0;
        self.bcpd = [0u8; 64];
        self.ocps = 0;
        self.ocpd = [0u8; 64];
        self.mode = 0;
        self.dot = 0;
        self.stat_irq_line = false;
        self.vram_accessible = true;
        self.vram_write_accessible = true;
        self.oam_accessible = true;
        self.oam_write_accessible = true;
        self.frame_ready = false;
        self.hblank_entered = false;
        self.lcd_first_line = false;
        self.lcd_first_line_short = false;
        self.lcd_first_frame = false;
        self.mode0_stat_dot = 0;
        self.mode3_stat_dot = 0;
        self.cgb_palettes_blocked = false;
        self.cgb_palette_unblock_dot = 0;
        self.window_line_counter = 0;
        self.wy_triggered = false;
        self.bg_fifo.clear();
        self.fetcher.reset(false);
        self.position_in_line = 0;
        self.sprite_fetch_active = false;
        self.sprites_fetched = 0;
        self.window_active = false;
        self.mode3_start_delay = 0;
        self.last_sprite_slot = -1;
        self.visible_ly = 0;
        self.ly_for_comparison = 0;
        self.line_start_pending = false;
        self.line_start_is_vblank = false;
        self.mode_for_interrupt = -1;
        self.line_153_phase = 0;
        self.accessed_oam_row = 0xFF;
        self.oam_bug_row = 0xFF;
        self.pending_lyc = None;
        self.bgp_rendering = 0;
        self.obp0_rendering = 0xFF;
        self.obp1_rendering = 0xFF;
        self.pixel_history = [(0, 0, 0, 0, false); 2];
        self.pixel_history_count = 0;
        self.pixel_history_next = 0;
        self.wx_just_changed = false;
    }

    /// Retroactively correct the last 2 rendered pixels when a DMG palette
    /// register is written during mode 3. On hardware, the CPU write takes
    /// effect at T3 of the M-cycle, but our eager PPU model has already
    /// rendered those T-cycles with the old palette. Fix pixel[-2] with the
    /// bus conflict glitch (old | new) and pixel[-1] with the real new value.
    fn retroactive_palette_fix(&mut self, pal_type: u8, old_val: u8, new_val: u8) {
        if self.mode != 3 || self.cgb_mode || self.pixel_history_count == 0 {
            return;
        }
        let glitch_val = old_val | new_val;

        // pixel_history_next points to the next write slot, so:
        // most recent pixel = (next + 1) % 2, older pixel = next % 2
        if self.pixel_history_count >= 2 {
            // Older pixel (T3 of the M-cycle): glitch value
            let idx = self.pixel_history_next;
            let (fb_idx, pt, cidx, _, _) = self.pixel_history[idx];
            if pt == pal_type && !self.lcd_first_frame {
                self.frame_buffer[fb_idx] = Self::dmg_color(glitch_val, cidx);
                if self.sgb_mode {
                    self.shade_buffer[fb_idx] = (glitch_val >> (cidx * 2)) & 0x03;
                }
            }
        }
        if self.pixel_history_count >= 1 {
            // Most recent pixel (T4 of the M-cycle): real new value
            let idx = (self.pixel_history_next + 1) % 2;
            let (fb_idx, pt, cidx, _, _) = self.pixel_history[idx];
            if pt == pal_type && !self.lcd_first_frame {
                self.frame_buffer[fb_idx] = Self::dmg_color(new_val, cidx);
                if self.sgb_mode {
                    self.shade_buffer[fb_idx] = (new_val >> (cidx * 2)) & 0x03;
                }
            }
        }
    }

    /// Retroactively correct the last 2 pixels for LCDC writes during mode 3.
    /// On DMG, the LCDC write has a -2T conflict similar to palette writes.
    /// The glitch cycle (T3) has BG_EN = old_BG_EN | new_BG_EN (only bit 0 is OR'd).
    /// T4 uses the real new LCDC value.
    fn retroactive_lcdc_fix(&mut self, old_lcdc: u8, new_lcdc: u8) {
        if self.mode != 3 || self.cgb_mode || self.pixel_history_count == 0 {
            return;
        }
        // When the bus handler already applied the glitch timing (2T old + 1T glitch),
        // skip the retroactive fix to avoid double-correcting pixels.
        if self.skip_retroactive_lcdc_fix {
            self.skip_retroactive_lcdc_fix = false;
            return;
        }
        let old_bg_en = old_lcdc & 0x01 != 0;
        let new_bg_en = new_lcdc & 0x01 != 0;
        if old_bg_en == new_bg_en {
            return; // BG_EN didn't change, no pixel correction needed
        }

        // Glitch LCDC: only BG_EN bit is OR'd (old | new), other bits keep old value
        let glitch_bg_en = old_bg_en || new_bg_en; // always true (one of them changed)

        // Re-render affected pixels
        for i in 0..2u8 {
            let count_needed = if i == 0 { 2 } else { 1 };
            if self.pixel_history_count < count_needed {
                continue;
            }
            let idx = if i == 0 {
                self.pixel_history_next // older pixel (T3: glitch)
            } else {
                (self.pixel_history_next + 1) % 2 // newer pixel (T4: real)
            };
            let (fb_idx, pt, _cidx, bg_cidx, is_sprite) = self.pixel_history[idx];
            if self.lcd_first_frame || is_sprite {
                continue; // sprite pixels not affected by BG_EN
            }
            let effective_bg_en = if i == 0 { glitch_bg_en } else { new_bg_en };
            let render_cidx = if effective_bg_en { bg_cidx } else { 0 };
            if pt == 0 {
                // BG pixel using BGP
                self.frame_buffer[fb_idx] = Self::dmg_color(self.bgp_rendering, render_cidx);
                if self.sgb_mode {
                    self.shade_buffer[fb_idx] = (self.bgp_rendering >> (render_cidx * 2)) & 0x03;
                }
            }
        }
    }

    /// Set post-boot PPU state for the given model (used when no boot ROM is loaded).
    /// Values determined by running actual boot ROMs and capturing PPU state at PC=$0100.
    /// For CGB/AGB, timing differs between CGB-native games and DMG compat mode.
    /// For SGB/SGB2, PPU timing depends on ROM header data (same mechanism as timer).
    pub fn set_post_boot(&mut self, model: crate::model::GbModel, is_cgb_game: bool, rom: &[u8]) {
        let (ly, dot, ticks) = match model {
            crate::model::GbModel::Dmg0 => (145u8, 99u32, 24_574_388u64),
            crate::model::GbModel::Dmg |
            crate::model::GbModel::Mgb  => (0, 403, 23_173_860u64),
            crate::model::GbModel::Sgb |
            crate::model::GbModel::Sgb2 => {
                // SGB boot ROM timing depends on ROM header data popcount.
                // Base ticks = 1686380; each 1-bit saves 4 T-cycles.
                // First line after LCD enable is 449 dots; subsequent lines are 456.
                let popcount = crate::timer::sgb_packet_popcount(rom);
                let total_ticks = 1_686_380u64 - 4 * popcount as u64;
                let remaining = total_ticks - 449;
                let lines_after_first = remaining / 456;
                let dot = (remaining % 456) as u32;
                let ly = ((1 + lines_after_first) % 154) as u8;
                (ly, dot, total_ticks)
            }
            crate::model::GbModel::Cgb0 |
            crate::model::GbModel::Cgb if is_cgb_game => (144u8, 164u32, 12_355_028u64),
            crate::model::GbModel::Cgb0 |
            crate::model::GbModel::Cgb  => (148, 352, 12_357_040u64),
            crate::model::GbModel::Agb if is_cgb_game => (144, 168, 12_355_032u64),
            crate::model::GbModel::Agb  => (148, 356, 12_357_044u64),
        };
        self.ly = ly;
        self.dot = dot;
        self.total_ticks = ticks;
        self.mode = 1;  // All models are in VBlank at $0100
        self.stat = (self.stat & 0xF8) | 1;  // Mode 1
        self.visible_ly = ly;
        self.ly_for_comparison = ly as i16;
        self.update_coincidence();
        // Set LCDC so that the subsequent write of 0x91 in emulator.rs doesn't
        // trigger a fresh LCD enable (which would reset the PPU state we just set).
        self.lcdc = 0x91;
        // CGB boot ROM leaves palette index registers at specific values.
        // BCPS reads as $C8 (auto-increment + index 8), stored as $88 (read OR's bit 6).
        // OCPS reads as $D0 (auto-increment + index 16), stored as $90.
        if model.is_cgb() {
            self.bcps = 0x88;
            self.ocps = 0x90;
        }

        // Initialize VRAM with post-boot state (Nintendo logo tiles + tilemap).
        // The boot ROM decompresses the cart header logo ($0104-$0133) into tiles 1-$18,
        // writes the ® symbol as tile $19, and sets up the tilemap at $9800.
        self.init_post_boot_vram(rom);
    }

    /// Populate VRAM bank 0 with the state left by the boot ROM.
    fn init_post_boot_vram(&mut self, rom: &[u8]) {
        // Clear VRAM first (boot ROM does this)
        self.vram[0].fill(0);

        // Decompress Nintendo logo from cart header $0104-$0133 into tiles 1-$18.
        // Each input byte encodes two 4-pixel rows (high nibble first).
        // DoubleBitsAndWriteRow doubles each bit horizontally (4px → 8px)
        // and writes the same byte to two consecutive rows (vertical doubling).
        // Only the lo plane is written; hi plane stays 0 (1bpp tiles).
        let logo_start = 0x0104;
        let logo_end = 0x0134;
        let mut vram_addr: usize = 0x10; // tile 1 starts at VRAM $8010 = offset $10
        for i in logo_start..logo_end {
            if i >= rom.len() { break; }
            let byte = rom[i];
            // High nibble → rows 0,1
            let doubled_hi = double_bits((byte >> 4) & 0x0F);
            self.vram[0][vram_addr] = doubled_hi;     // row 0 lo
            // vram_addr+1 = row 0 hi (stays 0)
            self.vram[0][vram_addr + 2] = doubled_hi;  // row 1 lo
            // vram_addr+3 = row 1 hi (stays 0)
            vram_addr += 4;
            // Low nibble → rows 2,3
            let doubled_lo = double_bits(byte & 0x0F);
            self.vram[0][vram_addr] = doubled_lo;     // row 2 lo
            self.vram[0][vram_addr + 2] = doubled_lo;  // row 3 lo
            vram_addr += 4;
        }

        // Trademark symbol (®) as tile $19 (VRAM offset $190)
        const TRADEMARK: [u8; 8] = [
            0x3C, 0x42, 0xB9, 0xA5, 0xB9, 0xA5, 0x42, 0x3C,
        ];
        let tm_offset = 0x190; // tile $19 = 25 * 16 = 0x190
        for (row, &byte) in TRADEMARK.iter().enumerate() {
            self.vram[0][tm_offset + row * 2] = byte; // lo plane only
        }

        // Set up tilemap at $9800 (VRAM offset $1800).
        // Logo is 12 tiles wide × 2 tiles tall, centered at row 8-9, cols 4-15.
        // Tiles are numbered 1-$18: top row = 1,2,...,12; bottom row = 13,14,...,24.
        // Trademark (®) tile $19 at row 8, col 16.
        let scrn0 = 0x1800usize; // tilemap base
        // Top row of logo: tiles 1-12 at tilemap row 8, cols 4-15
        for i in 0..12u8 {
            self.vram[0][scrn0 + 8 * 32 + 4 + i as usize] = i + 1;
        }
        // Bottom row of logo: tiles 13-24 at tilemap row 9, cols 4-15
        for i in 0..12u8 {
            self.vram[0][scrn0 + 9 * 32 + 4 + i as usize] = i + 13;
        }
        // Trademark symbol at row 8, col 16
        self.vram[0][scrn0 + 8 * 32 + 16] = 0x19;
    }

    /// Compute the accessed OAM row at a given dot position during Mode 2.
    /// Returns the row byte offset (8, 16, 24, ..., 152) or 0xFF if not in Mode 2
    /// or if the dot is before OAM search starts.
    pub fn oam_row_at_dot(&self, dot: u32) -> i16 {
        if self.mode != 2 || self.cgb_mode { return 0xFF; }
        if dot < 6 { return 0; }
        let mode2_end = 84u32; // DMG mode 2 ends at dot 84
        if dot >= mode2_end { return 0xFF; }
        let oam_search_index = ((dot - 6) / 2) as i16;
        (oam_search_index & !1) * 4 + 8
    }

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
                    if self.cgb_mode {
                        self.mode3_stat_dot = self.dot + 1;
                    } else {
                        self.stat = (self.stat & !0x03) | 0x03;
                    }
                    self.update_stat_irq();
                    // Run the first mode 3 tick on the transition dot itself.
                    // The 5T priming delay includes this dot as tick 1, so
                    // tick_mode3 must run here to avoid losing 1T at the boundary.
                    self.tick_mode3();
                }
            }
            3 => {
                // CGB: delayed STAT mode bits update for Mode 2→3
                if self.cgb_mode && self.mode3_stat_dot > 0 && self.dot >= self.mode3_stat_dot {
                    self.mode3_stat_dot = 0;
                    self.stat = (self.stat & !0x03) | 0x03;
                }
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
                // Mode 0 → end of scanline at dot 456 (or dot 449 for first line after LCD enable)
                // First line is 7T shorter: Mode 0 is shortened by 8T due to phantom
                // cycles_for_line adjustment, plus 1T initial DMG sleep.
                let line_end = if self.lcd_first_line_short {
                    // CGB: 450T breaks M-cycle alignment so that LY and mode
                    // changes fall in different step(4) batches.
                    // DMG: 449T (1T initial sleep + 8T phantom augment)
                    if self.cgb_mode { 450 } else { 449 }
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
                            // Prime mode_for_interrupt for next active line.
                            // Don't fire update_stat_irq here — on hardware, the
                            // Mode 2 STAT interrupt fires at T=3 from line start
                            // (in the handle_dmg_line_start dot 3 handler), not
                            // at the line wrap. Setting mode_for_interrupt now
                            // ensures the rising edge is detected at dot 3.
                            self.mode_for_interrupt = 2;
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
                        self.mode_for_interrupt = 2;
                    }
                    // OAM reads blocked 1T before mode 2 (T=3 of line start)
                    // OAM writes remain accessible until mode 2 proper
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
                    if self.ly == 144 && self.mode != 1 {
                        // VBlank entry at dot 2: mode 0→1, VBlank IF.
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

    fn transition_to_mode2(&mut self) {
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

    fn transition_to_mode3(&mut self) {
        self.mode3_dot = self.dot;
        self.mode = 3;
        self.mode_for_interrupt = 3;
        self.stat = (self.stat & !0x03) | 0x03;
        self.oam_accessible = false;
        self.oam_write_accessible = false;
        self.vram_accessible = false;
        self.vram_write_accessible = false;
        self.pixel_history_count = 0;
        self.pixel_history_next = 0;
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

    // ---- Pixel FIFO ----

    /// Initialize FIFO state at the start of Mode 3
    fn init_fifo(&mut self) {
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
    fn activate_window(&mut self) {
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
            FetcherState::GetTileT1 => {
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
                self.fetcher.state = FetcherState::GetTileT2;
            }
            FetcherState::GetTileT2 => {
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
                self.fetcher.state = FetcherState::GetTileDataLowT1;
            }
            FetcherState::GetTileDataLowT1 => {
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
                self.fetcher.state = FetcherState::GetTileDataLowT2;
            }
            FetcherState::GetTileDataLowT2 => {
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
                self.fetcher.state = FetcherState::GetTileDataHighT1;
            }
            FetcherState::GetTileDataHighT1 => {
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
                self.fetcher.state = FetcherState::GetTileDataHighT2;
            }
            FetcherState::GetTileDataHighT2 => {
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
                self.fetcher.state = FetcherState::Push;
            }
            FetcherState::Push => {
                if self.bg_fifo.len() == 0 {
                    self.push_bg_pixels();
                    self.fetcher.tile_x += 1;
                    self.fetcher.state = FetcherState::GetTileT1;
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
                if self.fetcher.state == FetcherState::GetTileDataLowT2 {
                    self.vram[bank][addr]
                } else {
                    self.vram[bank][addr + 1]
                }
            }
        } else {
            // last_tileset was false: uses a cached `data_for_sel_glitch` value.
            // For now, use normal VRAM data (this path is less common).
            let (addr, bank) = (self.fetcher.latched_addr, self.fetcher.latched_bank);
            if self.fetcher.state == FetcherState::GetTileDataLowT2 {
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

        // Record pixel metadata for retroactive correction (DMG only).
        if !self.cgb_mode {
            let (pal_type, cidx) = if draw_sprite {
                let oam_px = oam.unwrap();
                (if oam_px.sprite_dmg_palette == 1 { 2u8 } else { 1u8 }, oam_px.color_index)
            } else if self.lcdc & 0x01 == 0 {
                (0u8, 0u8) // BG disabled → color 0 through BGP
            } else {
                (0u8, bg.color_index)
            };
            self.pixel_history[self.pixel_history_next] = (fb_idx, pal_type, cidx, bg.color_index, draw_sprite);
            self.pixel_history_next = 1 - self.pixel_history_next;
            if self.pixel_history_count < 2 {
                self.pixel_history_count += 1;
            }
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

    // ---- Edge-triggered STAT interrupt ----

    fn update_stat_irq(&mut self) {
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
    fn update_stat_irq_silent(&mut self) {
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
    fn update_stat_irq_on_write(&mut self) {
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

    fn update_coincidence(&mut self) {
        if self.ly_for_comparison >= 0 && self.ly_for_comparison as u8 == self.lyc {
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
                self.retroactive_lcdc_fix(self.lcdc, val);
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
                        // DMG STAT write bug: writing STAT during Mode 0 or 1
                        // briefly drives the STAT IRQ line high, firing a
                        // spurious interrupt if the line was previously low.
                        // The glitch fires when no interrupt enable bits (3-6)
                        // are being set in the written value. Writing a value
                        // that enables any interrupt source (LYC, OAM, VBlank,
                        // HBlank) does not trigger the glitch.
                        if (self.mode == 0 || self.mode == 1)
                            && !self.stat_irq_line
                            && val & 0x78 == 0
                        {
                            self.if_flags |= 0x02;
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

impl Default for Ppu {
    fn default() -> Self {
        Self::new()
    }
}
