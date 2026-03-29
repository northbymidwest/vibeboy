//! Data tables for CGB boot ROM: fonts, palettes, checksum tables.
//!
//! Palette data is original (no copyrighted colors). Checksum/disambiguation
//! tables match the publicly documented CGB boot ROM algorithm.

/// 8x8 2bpp font for "VIBEBOY" logo, split across tiles for centering.
///
/// 7 letters = 56px on 160px screen. True center is pixel 52, which isn't
/// on a tile boundary. We split each letter at the 4px midpoint so that
/// each tile contains the right half of one letter and left half of the next,
/// with blank padding on the edges. This produces 8 tiles that, placed at
/// column 6 (pixel 48), center the visible content at pixel 52.
///
/// Each tile uses two CGB color indices so both halves keep their letter's
/// color: left half = color 2 (hi=1, lo=0), right half = color 3 (hi=1, lo=1).
/// Each tile's palette maps color 2 → left letter's color, color 3 → right letter's.
/// Tiles 0 and 7 share a palette (each only uses one color), freeing palette 7
/// for the swatch bar.
///
/// Tile layout: [pad4|V_L] [V_R|I_L] [I_R|B_L] ... [Y_R|pad4]
pub fn font_tiles() -> Vec<u8> {
    let glyphs: [[u8; 8]; 7] = [
        [0xC6, 0xC6, 0xC6, 0xC6, 0x6C, 0x6C, 0x38, 0x10], // V
        [0x7C, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x7C], // I
        [0xFC, 0xC6, 0xC6, 0xFC, 0xC6, 0xC6, 0xC6, 0xFC], // B
        [0xFE, 0xC0, 0xC0, 0xFC, 0xC0, 0xC0, 0xC0, 0xFE], // E
        [0xFC, 0xC6, 0xC6, 0xFC, 0xC6, 0xC6, 0xC6, 0xFC], // B
        [0x7C, 0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0x7C], // O
        [0xC6, 0xC6, 0x6C, 0x38, 0x10, 0x10, 0x10, 0x10], // Y
    ];
    let mut data = Vec::new();
    for t in 0..8 {
        for row in 0..8 {
            let left = if t > 0 { glyphs[t - 1][row] } else { 0 };
            let right = if t < 7 { glyphs[t][row] } else { 0 };
            // Combined glyph bits for this split tile row
            let combined = ((left & 0x0F) << 4) | (right >> 4);
            // Left half pixels → color 2 (lo=0, hi=1)
            // Right half pixels → color 3 (lo=1, hi=1)
            let lo = combined & 0x0F; // only right-half bits in lo plane
            let hi = combined; // all glyph bits in hi plane
            data.push(lo);
            data.push(hi);
        }
    }
    // Swatch tiles: solid color 1 (lo=FF,hi=00), color 2 (lo=00,hi=FF), color 3 (lo=FF,hi=FF)
    for (lo, hi) in &[(0xFF, 0x00), (0x00, 0xFF), (0xFF, 0xFF)] {
        for _ in 0..8 {
            data.push(*lo);
            data.push(*hi);
        }
    }
    data
}

/// Rainbow logo colors (RGB555 LE) — one per letter of "VIBEBOY".
pub const LOGO_COLORS: [u16; 7] = [0x001F, 0x02DF, 0x03FF, 0x03A0, 0x7E00, 0x7E0F, 0x7C1F];

// ── Official CGB palette algorithm tables ────────────────────────────────
// From publicly documented reverse engineering (ISSOtm/gb-bootroms, Pan Docs)

pub const TITLE_CHECKSUMS: &[u8] = &[
    0x00, 0x88, 0x16, 0x36, 0xD1, 0xDB, 0xF2, 0x3C, 0x8C, 0x92, 0x3D, 0x5C, 0x58, 0xC9, 0x3E, 0x70,
    0x1D, 0x59, 0x69, 0x19, 0x35, 0xA8, 0x14, 0xAA, 0x75, 0x95, 0x99, 0x34, 0x6F, 0x15, 0xFF, 0x97,
    0x4B, 0x90, 0x17, 0x10, 0x39, 0xF7, 0xF6, 0xA2, 0x49, 0x4E, 0x43, 0x68, 0xE0, 0x8B, 0xF0, 0xCE,
    0x0C, 0x29, 0xE8, 0xB7, 0x86, 0x9A, 0x52, 0x01, 0x9D, 0x71, 0x9C, 0xBD, 0x5D, 0x6D, 0x67, 0x3F,
    0x6B, 0xB3, 0x46, 0x28, 0xA5, 0xC6, 0xD3, 0x27, 0x61, 0x18, 0x66, 0x6A, 0xBF, 0x0D, 0xF4,
];

pub const FOURTH_LETTERS_R0: &[u8] = &[
    0x42, 0x45, 0x46, 0x41, 0x41, 0x52, 0x42, 0x45, 0x4B, 0x45, 0x4B, 0x20, 0x52, 0x2D,
];
pub const FOURTH_LETTERS_R1: &[u8] = &[
    0x55, 0x52, 0x41, 0x52, 0x20, 0x49, 0x4E, 0x41, 0x49, 0x4C, 0x49, 0x43, 0x45, 0x20,
];

pub const PAL_IDX_FLAGS: &[u8] = &[
    0x7C, 0x08, 0x12, 0xA3, 0xA2, 0x07, 0x87, 0x4B, 0x20, 0x12, 0x65, 0xA8, 0x16, 0xA9, 0x86, 0xB1,
    0x68, 0xA0, 0x87, 0x66, 0x12, 0xA1, 0x30, 0x3C, 0x12, 0x85, 0x12, 0x64, 0x1B, 0x07, 0x06, 0x6F,
    0x6E, 0x6E, 0xAE, 0xAF, 0x6F, 0xB2, 0xAF, 0xB2, 0xA8, 0xAB, 0x6F, 0xAF, 0x86, 0xAE, 0xA2, 0xA2,
    0x12, 0xAF, 0x13, 0x12, 0xA1, 0x6E, 0xAF, 0xAF, 0xAD, 0x06, 0x4C, 0x6E, 0xAF, 0xAF, 0x12, 0x7C,
    0xAC, 0xA8, 0x6A, 0x6E, 0x13, 0xA0, 0x2D, 0xA8, 0x2B, 0xAC, 0x64, 0xAC, 0x6D, 0x87, 0xBC, 0x60,
    0xB4, 0x13, 0x72, 0x7C, 0xB5, 0xAE, 0xAE, 0x7C, 0x7C, 0x65, 0xA2, 0x6C, 0x64, 0x85,
];

pub const PAL_TRIPLETS: &[u8] = &[
    0x80, 0xB0, 0x40, 0x88, 0x20, 0x68, 0xDE, 0x00, 0x70, 0xDE, 0x20, 0x78, 0x20, 0x20, 0x38, 0x20,
    0xB0, 0x90, 0x20, 0xB0, 0xA0, 0xE0, 0xB0, 0xC0, 0x98, 0xB6, 0x48, 0x80, 0xE0, 0x50, 0x1E, 0x1E,
    0x58, 0x20, 0xB8, 0xE0, 0x88, 0xB0, 0x10, 0x20, 0x00, 0x10, 0x20, 0xE0, 0x18, 0xE0, 0x18, 0x00,
    0x18, 0xE0, 0x20, 0xA8, 0xE0, 0x20, 0x18, 0xE0, 0x00, 0x20, 0x18, 0xD8, 0xC8, 0x18, 0xE0, 0x00,
    0xE0, 0x40, 0x28, 0x28, 0x28, 0x18, 0xE0, 0x60, 0x20, 0x18, 0xE0, 0x00, 0x00, 0x08, 0xE0, 0x18,
    0x30, 0xD0, 0xD0, 0xD0, 0x20, 0xE0, 0xE8,
];

/// Joypad combo table: combined joypad state bytes.
/// High nibble = D-pad (Down=$80,Up=$40,Left=$20,Right=$10),
/// low nibble = buttons (B=$02, A=$01).
pub const COMBOS: &[u8] = &[
    0x40, 0x41, 0x42, // Up, Up+A, Up+B
    0x20, 0x21, 0x22, // Left, Left+A, Left+B
    0x80, 0x81, 0x82, // Down, Down+A, Down+B
    0x10, 0x11, 0x12, // Right, Right+A, Right+B
];

/// Palette flags bytes for each combo (same format as PAL_IDX_FLAGS).
pub const COMBO_PAL: &[u8] = &[
    0x12, 0xB0, 0x79, // Up, Up+A, Up+B
    0xB8, 0xAD, 0x16, // Left, Left+A, Left+B
    0x17, 0x07, 0xBA, // Down, Down+A, Down+B
    0x05, 0x7C, 0x13, // Right, Right+A, Right+B
];

// ── Original palette RGB data ────────────────────────────────────────────
// 30 palettes × 4 colors × 2 bytes (RGB555 LE) = 240 bytes.
// Designed for vibeboy — no copyrighted colors.

fn rgb555(r: u8, g: u8, b: u8) -> [u8; 2] {
    let v = (r as u16 & 0x1F) | ((g as u16 & 0x1F) << 5) | ((b as u16 & 0x1F) << 10);
    [v as u8, (v >> 8) as u8]
}

fn pal(c0: (u8,u8,u8), c1: (u8,u8,u8), c2: (u8,u8,u8), c3: (u8,u8,u8)) -> [u8; 8] {
    let mut out = [0u8; 8];
    let colors = [c0, c1, c2, c3];
    for (i, &(r, g, b)) in colors.iter().enumerate() {
        let bytes = rgb555(r, g, b);
        out[i * 2] = bytes[0];
        out[i * 2 + 1] = bytes[1];
    }
    out
}

const W: (u8,u8,u8) = (31,31,31);
const K: (u8,u8,u8) = (0,0,0);

pub fn palette_rgb() -> Vec<u8> {
    let palettes: &[[u8; 8]] = &[
        //  0: Warm cream/peach
        pal((31,30,26), (24,18,10), (14,8,4),   K),
        //  1: Earthy terracotta
        pal((28,24,18), (22,14,8),  (14,6,2),   (4,2,0)),
        //  2: Steel blue
        pal(W,          (14,20,28), (6,10,18),   K),
        //  3: Green/lime
        pal(W,          (10,28,4),  (2,16,0),    K),
        //  4: Rose/red
        pal(W,          (28,10,12), (18,2,4),    K),
        //  5: Classic gray
        pal(W,          (20,20,20), (10,10,10),  K),
        //  6: Gold/amber
        pal(W,          (30,26,4),  (20,14,0),   K),
        //  7: Warm yellow
        pal(W,          (30,24,6),  (22,14,2),   K),
        //  8: Salmon/peach
        pal(W,          (26,18,16), (20,8,6),    K),
        //  9: Teal/sea
        pal((28,31,28), (4,26,22),  (0,14,12),   K),
        // 10: Dusty violet
        pal((30,28,31), (22,12,24), (14,4,16),   (4,0,6)),
        // 11: Warm brown
        pal((30,28,24), (22,16,8),  (12,8,2),    K),
        // 12: Pastel pink
        pal((31,24,26), (26,14,20), (18,6,14),   K),
        // 13: Olive/army
        pal((26,30,22), (16,22,8),  (8,14,2),    (2,4,0)),
        // 14: Inverted dark base
        pal((8,6,4),    (18,16,14), (26,24,22),  K),
        // 15: Inverted blue accent
        pal((6,8,18),   (12,16,24), (22,26,31),  W),
        // 16: Sky blue
        pal(W,          (14,22,31), (4,12,22),   K),
        // 17: Deep forest
        pal((14,22,10), (8,16,4),   (2,10,0),    (0,4,0)),
        // 18: Tangerine
        pal(W,          (30,18,4),  (22,8,0),    K),
        // 19: Dark moss
        pal((16,20,10), (10,14,4),  (4,8,0),     K),
        // 20: Grass green
        pal(W,          (10,26,6),  (2,18,0),    K),
        // 21: Harvest gold
        pal(W,          (28,22,6),  (18,12,2),   (4,2,0)),
        // 22: Aqua/cyan
        pal(W,          (6,28,26),  (0,16,16),   K),
        // 23: Bright pastel
        pal(W,          (24,31,24), (16,24,16),  (6,10,6)),
        // 24: Lime/chartreuse
        pal(W,          (22,30,4),  (12,20,0),   K),
        // 25: Dark hunter
        pal((22,26,10), (14,18,4),  (6,10,0),    K),
        // 26: Warm neutral
        pal(W,          (22,20,16), (12,10,6),   K),
        // 27: Inverted base
        pal(K,          (10,12,14), (20,22,24),  W),
        // 28: Vivid blue
        pal(W,          (12,18,30), (2,8,24),    K),
        // 29: DMG green tint
        pal(W,          (14,28,6),  (2,18,0),    K),
    ];
    palettes.iter().flat_map(|p| p.iter().copied()).collect()
}
