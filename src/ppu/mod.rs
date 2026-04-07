/// Game Boy Color PPU (Pixel Processing Unit) — Pixel FIFO renderer
///
/// Timing (T-cycles per scanline = 456):
///   Mode 2: OAM scan        — 80 dots (dot 0..79)
///   Mode 3: Drawing          — variable, ends organically when 160 pixels output
///   Mode 0: HBlank           — remainder of 456
///   Mode 1: VBlank           — lines 144-153, 456 dots each
///   Total frame: 154 lines × 456 = 70224 T-cycles

mod registers;
mod rendering;
mod timing;

/// Serde helper for VRAM: [[u8; 0x2000]; 2] (two 8KB banks).
mod serde_vram {
    use serde::{Serializer, Deserializer, Serialize, Deserialize};

    pub fn serialize<S: Serializer>(data: &[[u8; 0x2000]; 2], ser: S) -> Result<S::Ok, S::Error> {
        let combined: Vec<u8> = data[0].iter().chain(data[1].iter()).copied().collect();
        combined.serialize(ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<[[u8; 0x2000]; 2], D::Error> {
        let combined: Vec<u8> = Vec::deserialize(de)?;
        if combined.len() != 0x4000 {
            return Err(serde::de::Error::custom("expected 16384 bytes for VRAM"));
        }
        let mut result = [[0u8; 0x2000]; 2];
        result[0].copy_from_slice(&combined[..0x2000]);
        result[1].copy_from_slice(&combined[0x2000..]);
        Ok(result)
    }
}

// ---- Pixel FIFO types ----

#[derive(Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
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

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
enum FetcherState {
    GetTileT1,
    GetTileT2,
    GetTileDataLowT1,
    GetTileDataLowT2,
    GetTileDataHighT1,
    GetTileDataHighT2,
    Push,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    /// Tick counter modulo 6: tracks position in the 6-dot fetch cycle
    /// independent of Push stalls. Used for sprite alignment penalty.
    cycle_tick: u8,
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
            cycle_tick: 0,
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
        self.cycle_tick = 0;
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

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Ppu {
    /// VRAM banks 0 and 1 (8 KiB each)
    #[serde(with = "serde_vram")]
    pub vram: [[u8; 0x2000]; 2],
    /// Current VRAM bank (0 or 1), controlled by 0xFF4F (VBK)
    pub vram_bank: usize,
    /// Object Attribute Memory (40 sprites x 4 bytes = 160 bytes).
    /// Extended to 192 bytes to handle OAM bug corruption at accessed_oam_row > 152
    #[serde(with = "serde_big_array::BigArray")]
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
    #[serde(with = "serde_big_array::BigArray")]
    pub bcpd: [u8; 64], // 0xFF69: BG palette data (8 palettes x 4 colors x 2 bytes)
    pub ocps: u8,      // 0xFF6A: OBJ Color Palette Spec
    #[serde(with = "serde_big_array::BigArray")]
    pub ocpd: [u8; 64], // 0xFF6B: OBJ palette data

    // Internal state
    pub(crate) mode: u8,
    /// T-cycle counter within the current scanline (0..455)
    pub dot: u32,

    /// Debug: total ticks since LCD enable
    pub total_ticks: u64,

    /// Previous state of internal STAT IRQ signal (for edge detection)
    stat_irq_line: bool,

    /// Sprites collected during Mode 2 OAM scan: (y, x, tile, attrs, oam_index)
    pub(crate) scanline_sprites: Vec<(u8, u8, u8, u8, u8)>,

    /// VRAM read accessible (false during Mode 3, and late Mode 2 on DMG)
    pub vram_accessible: bool,
    /// VRAM write accessible (false during Mode 3 only)
    pub vram_write_accessible: bool,
    /// OAM read accessible (false during Modes 2 and 3)
    pub oam_accessible: bool,
    /// OAM write accessible (false during Mode 2 early and Mode 3; unblocked at Mode 2 index 37)
    pub oam_write_accessible: bool,
    /// During OAM DMA, the PPU reads the current DMA bus byte instead of stored
    /// OAM values. Set by Bus before each PPU step, None when DMA is not active.
    pub dma_bus_byte: Option<u8>,

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

    /// MGB (Game Boy Pocket) mode: uses grayscale palette instead of DMG green.
    pub mgb_mode: bool,

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
    /// WX write conflict: suppresses WX+6 window trigger for 1T after WX write
    pub(crate) wx_just_changed: bool,
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
            dma_bus_byte: None,

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
            mgb_mode: false,
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
            wx_just_changed: false,
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
        self.wx_just_changed = false;
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
}

impl Default for Ppu {
    fn default() -> Self {
        Self::new()
    }
}
