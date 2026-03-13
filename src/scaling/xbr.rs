//! xBR (scale By Rules) — 2x, 3x, and 4x pixel-art scaling.
//!
//! Based on Hyllian's xBR algorithm. Uses weighted color distance to detect
//! edges and applies directional interpolation rules to produce smooth,
//! anti-aliased output while preserving pixel-art detail.

use super::get;
use super::color_dist;
use super::blend_argb as blend;

/// xBR scaling factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XbrScale {
    Xbr2x,
    Xbr3x,
    Xbr4x,
}

impl XbrScale {
    pub fn factor(self) -> u32 {
        match self {
            XbrScale::Xbr2x => 2,
            XbrScale::Xbr3x => 3,
            XbrScale::Xbr4x => 4,
        }
    }
}

/// Check if two colors are equal (exact match).
#[inline(always)]
fn eq(a: u32, b: u32) -> bool {
    a == b
}

// ── Edge detection helpers ─────────────────────────────────────────────────

/// Sample the 5x5 neighborhood around (x, y):
///
/// ```text
///  a0 a1 a2 a3 a4
///  b0 b1 b2 b3 b4
///  c0 c1  e c3 c4
///  d0 d1 d2 d3 d4
///  e0 e1 e2 e3 e4
/// ```
///
/// Returns [a0..e4] as a flat 25-element array, with e = index 12.
struct Neighborhood {
    p: [u32; 25],
}

impl Neighborhood {
    fn sample(src: &[u32], w: usize, h: usize, x: isize, y: isize) -> Self {
        let mut p = [0u32; 25];
        for j in 0..5_isize {
            for i in 0..5_isize {
                p[(j * 5 + i) as usize] = get(src, w, h, x + i - 2, y + j - 2);
            }
        }
        Neighborhood { p }
    }
}

/// Hyllian xBR edge weight for a given corner direction.
///
/// 5x5 neighborhood layout:
/// ```text
///   0  1  2  3  4
///   5  6  7  8  9
///  10 11 [12] 13 14
///  15 16  17 18 19
///  20 21  22 23 24
/// ```
///
/// Returns (wd1, wd2) where wd1 < wd2 indicates a diagonal edge
/// toward the specified corner. E=12 is always the center pixel.
///
/// Formula (Hyllian):
///   wd1 = d(E,C) + d(E,G) + d(I,F4) + d(I,H4) + 4*d(H,F)
///   wd2 = d(H,D) + d(H,I4) + d(F,B) + d(F,I5) + 4*d(E,I)
///
/// Where for each corner:
///   F = neighbor along axis 1, H = neighbor along axis 2
///   I = diagonal corner, C = cross-diagonal from E
///   G = opposite cross-diagonal from E
///   F4, H4 = one beyond I in each axis direction
///   I4, I5 = one beyond I in each axis direction (for wd2)
#[inline(always)]
fn edge_weight(p: &[u32; 25], corner: Corner) -> (f32, f32) {
    let (c, f, g, h, i, f4, h4, d, i4, b, i5) = match corner {
        // BR: F=right(13), H=below(17), I=BR(18)
        Corner::Br => (8, 13, 16, 17, 18, 14, 22, 11, 23, 7, 19),
        // BL: F=left(11), H=below(17), I=BL(16)
        Corner::Bl => (6, 11, 18, 17, 16, 10, 22, 13, 21, 7, 15),
        // TR: F=right(13), H=above(7), I=TR(8)
        Corner::Tr => (18, 13, 6, 7, 8, 14, 2, 11, 3, 17, 9),
        // TL: F=left(11), H=above(7), I=TL(6)
        Corner::Tl => (16, 11, 8, 7, 6, 10, 2, 13, 1, 17, 5),
    };

    let wd1 = color_dist(p[12], p[c])
        + color_dist(p[12], p[g])
        + color_dist(p[i], p[f4])
        + color_dist(p[i], p[h4])
        + 4.0 * color_dist(p[h], p[f]);

    let wd2 = color_dist(p[h], p[d])
        + color_dist(p[h], p[i4])
        + color_dist(p[f], p[b])
        + color_dist(p[f], p[i5])
        + 4.0 * color_dist(p[12], p[i]);

    (wd1, wd2)
}

#[derive(Clone, Copy)]
enum Corner { Br, Bl, Tr, Tl }

// ── xBR2x ──────────────────────────────────────────────────────────────────

pub fn scale2x(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let dst_w = src_w * 2;
    let dst_h = src_h * 2;
    let mut dst = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let ix = x as isize;
            let iy = y as isize;
            let n = Neighborhood::sample(src, src_w, src_h, ix, iy);
            let e = n.p[12]; // center

            // Start with all center
            let mut out = [e; 4];

            // Process all 4 corners
            xbr2x_corner(&n, &mut out);

            let dx = x * 2;
            let dy = y * 2;
            dst[dy * dst_w + dx] = out[0];
            dst[dy * dst_w + dx + 1] = out[1];
            dst[(dy + 1) * dst_w + dx] = out[2];
            dst[(dy + 1) * dst_w + dx + 1] = out[3];
        }
    }
    dst
}

/// Process all 4 corners of a 2x2 output block.
fn xbr2x_corner(n: &Neighborhood, out: &mut [u32; 4]) {
    let p = &n.p;
    let e = p[12];

    // Bottom-right corner
    {
        let (wd1, wd2) = edge_weight(p, Corner::Br);
        if wd1 < wd2 && !eq(p[12], p[18]) {
            let r = p[13];
            let d = p[17];
            if eq(r, d) {
                out[3] = blend(e, r, 0.5);
            } else if color_dist(e, r) < color_dist(e, d) {
                out[3] = blend(e, r, 0.25);
            } else {
                out[3] = blend(e, d, 0.25);
            }
        }
    }

    // Bottom-left corner
    {
        let (wd1, wd2) = edge_weight(p, Corner::Bl);
        if wd1 < wd2 && !eq(p[12], p[16]) {
            let l = p[11];
            let d = p[17];
            if eq(l, d) {
                out[2] = blend(e, l, 0.5);
            } else if color_dist(e, l) < color_dist(e, d) {
                out[2] = blend(e, l, 0.25);
            } else {
                out[2] = blend(e, d, 0.25);
            }
        }
    }

    // Top-right corner
    {
        let (wd1, wd2) = edge_weight(p, Corner::Tr);
        if wd1 < wd2 && !eq(p[12], p[8]) {
            let r = p[13];
            let u = p[7];
            if eq(r, u) {
                out[1] = blend(e, r, 0.5);
            } else if color_dist(e, r) < color_dist(e, u) {
                out[1] = blend(e, r, 0.25);
            } else {
                out[1] = blend(e, u, 0.25);
            }
        }
    }

    // Top-left corner
    {
        let (wd1, wd2) = edge_weight(p, Corner::Tl);
        if wd1 < wd2 && !eq(p[12], p[6]) {
            let l = p[11];
            let u = p[7];
            if eq(l, u) {
                out[0] = blend(e, l, 0.5);
            } else if color_dist(e, l) < color_dist(e, u) {
                out[0] = blend(e, l, 0.25);
            } else {
                out[0] = blend(e, u, 0.25);
            }
        }
    }
}

// ── xBR3x ──────────────────────────────────────────────────────────────────

pub fn scale3x(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let dst_w = src_w * 3;
    let dst_h = src_h * 3;
    let mut dst = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let ix = x as isize;
            let iy = y as isize;
            let n = Neighborhood::sample(src, src_w, src_h, ix, iy);
            let e = n.p[12];

            let mut out = [e; 9];
            xbr3x_corners(&n, &mut out);

            let dx = x * 3;
            let dy = y * 3;
            for oy in 0..3 {
                for ox in 0..3 {
                    dst[(dy + oy) * dst_w + dx + ox] = out[oy * 3 + ox];
                }
            }
        }
    }
    dst
}

/// Process all 4 corners of a 3x3 output block.
fn xbr3x_corners(n: &Neighborhood, out: &mut [u32; 9]) {
    let p = &n.p;
    let e = p[12];

    // Bottom-right corner: affects out[5], out[7], out[8]
    {
        let (wd1, wd2) = edge_weight(p, Corner::Br);
        if wd1 < wd2 && !eq(p[12], p[18]) {
            let r = p[13];
            let d = p[17];
            if eq(r, d) {
                out[5] = blend(e, r, 0.25);
                out[7] = blend(e, d, 0.25);
                out[8] = blend(e, r, 0.5);
            } else if color_dist(e, r) < color_dist(e, d) {
                out[8] = blend(e, r, 0.375);
                out[5] = blend(e, r, 0.125);
            } else {
                out[8] = blend(e, d, 0.375);
                out[7] = blend(e, d, 0.125);
            }
        }
    }

    // Bottom-left corner: affects out[3], out[7], out[6]
    {
        let (wd1, wd2) = edge_weight(p, Corner::Bl);
        if wd1 < wd2 && !eq(p[12], p[16]) {
            let l = p[11];
            let d = p[17];
            if eq(l, d) {
                out[3] = blend(e, l, 0.25);
                out[7] = blend(out[7], d, 0.25);
                out[6] = blend(e, l, 0.5);
            } else if color_dist(e, l) < color_dist(e, d) {
                out[6] = blend(e, l, 0.375);
                out[3] = blend(e, l, 0.125);
            } else {
                out[6] = blend(e, d, 0.375);
                out[7] = blend(out[7], d, 0.125);
            }
        }
    }

    // Top-right corner: affects out[1], out[5], out[2]
    {
        let (wd1, wd2) = edge_weight(p, Corner::Tr);
        if wd1 < wd2 && !eq(p[12], p[8]) {
            let r = p[13];
            let u = p[7];
            if eq(r, u) {
                out[5] = blend(out[5], r, 0.25);
                out[1] = blend(e, u, 0.25);
                out[2] = blend(e, r, 0.5);
            } else if color_dist(e, r) < color_dist(e, u) {
                out[2] = blend(e, r, 0.375);
                out[5] = blend(out[5], r, 0.125);
            } else {
                out[2] = blend(e, u, 0.375);
                out[1] = blend(e, u, 0.125);
            }
        }
    }

    // Top-left corner: affects out[1], out[3], out[0]
    {
        let (wd1, wd2) = edge_weight(p, Corner::Tl);
        if wd1 < wd2 && !eq(p[12], p[6]) {
            let l = p[11];
            let u = p[7];
            if eq(l, u) {
                out[3] = blend(out[3], l, 0.25);
                out[1] = blend(out[1], u, 0.25);
                out[0] = blend(e, l, 0.5);
            } else if color_dist(e, l) < color_dist(e, u) {
                out[0] = blend(e, l, 0.375);
                out[3] = blend(out[3], l, 0.125);
            } else {
                out[0] = blend(e, u, 0.375);
                out[1] = blend(out[1], u, 0.125);
            }
        }
    }
}

// ── xBR4x ──────────────────────────────────────────────────────────────────

pub fn scale4x(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let dst_w = src_w * 4;
    let dst_h = src_h * 4;
    let mut dst = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let ix = x as isize;
            let iy = y as isize;
            let n = Neighborhood::sample(src, src_w, src_h, ix, iy);
            let e = n.p[12];

            let mut out = [e; 16];
            xbr4x_corners(&n, &mut out);

            let dx = x * 4;
            let dy = y * 4;
            for oy in 0..4 {
                for ox in 0..4 {
                    dst[(dy + oy) * dst_w + dx + ox] = out[oy * 4 + ox];
                }
            }
        }
    }
    dst
}

/// Process all 4 corners of a 4x4 output block.
fn xbr4x_corners(n: &Neighborhood, out: &mut [u32; 16]) {
    let p = &n.p;
    let e = p[12];

    // Bottom-right corner: affects out[7], out[11], out[13], out[14], out[15]
    {
        let (wd1, wd2) = edge_weight(p, Corner::Br);
        if wd1 < wd2 && !eq(p[12], p[18]) {
            let r = p[13];
            let d = p[17];
            if eq(r, d) {
                out[15] = blend(e, r, 0.5);
                out[14] = blend(e, d, 0.375);
                out[11] = blend(e, r, 0.375);
                out[13] = blend(e, d, 0.125);
                out[7] = blend(e, r, 0.125);
            } else if color_dist(e, r) < color_dist(e, d) {
                out[15] = blend(e, r, 0.5);
                out[11] = blend(e, r, 0.25);
                out[7] = blend(e, r, 0.0625);
            } else {
                out[15] = blend(e, d, 0.5);
                out[14] = blend(e, d, 0.25);
                out[13] = blend(e, d, 0.0625);
            }
        }
    }

    // Bottom-left corner
    {
        let (wd1, wd2) = edge_weight(p, Corner::Bl);
        if wd1 < wd2 && !eq(p[12], p[16]) {
            let l = p[11];
            let d = p[17];
            if eq(l, d) {
                out[12] = blend(e, l, 0.5);
                out[13] = blend(out[13], d, 0.375);
                out[8] = blend(e, l, 0.375);
                out[14] = blend(out[14], d, 0.125);
                out[4] = blend(e, l, 0.125);
            } else if color_dist(e, l) < color_dist(e, d) {
                out[12] = blend(e, l, 0.5);
                out[8] = blend(e, l, 0.25);
                out[4] = blend(e, l, 0.0625);
            } else {
                out[12] = blend(e, d, 0.5);
                out[13] = blend(out[13], d, 0.25);
                out[14] = blend(out[14], d, 0.0625);
            }
        }
    }

    // Top-right corner
    {
        let (wd1, wd2) = edge_weight(p, Corner::Tr);
        if wd1 < wd2 && !eq(p[12], p[8]) {
            let r = p[13];
            let u = p[7];
            if eq(r, u) {
                out[3] = blend(e, r, 0.5);
                out[2] = blend(e, u, 0.375);
                out[7] = blend(out[7], r, 0.375);
                out[1] = blend(e, u, 0.125);
                out[11] = blend(out[11], r, 0.125);
            } else if color_dist(e, r) < color_dist(e, u) {
                out[3] = blend(e, r, 0.5);
                out[7] = blend(out[7], r, 0.25);
                out[11] = blend(out[11], r, 0.0625);
            } else {
                out[3] = blend(e, u, 0.5);
                out[2] = blend(e, u, 0.25);
                out[1] = blend(e, u, 0.0625);
            }
        }
    }

    // Top-left corner
    {
        let (wd1, wd2) = edge_weight(p, Corner::Tl);
        if wd1 < wd2 && !eq(p[12], p[6]) {
            let l = p[11];
            let u = p[7];
            if eq(l, u) {
                out[0] = blend(e, l, 0.5);
                out[1] = blend(out[1], u, 0.375);
                out[4] = blend(out[4], l, 0.375);
                out[2] = blend(out[2], u, 0.125);
                out[8] = blend(out[8], l, 0.125);
            } else if color_dist(e, l) < color_dist(e, u) {
                out[0] = blend(e, l, 0.5);
                out[4] = blend(out[4], l, 0.25);
                out[8] = blend(out[8], l, 0.0625);
            } else {
                out[0] = blend(e, u, 0.5);
                out[1] = blend(out[1], u, 0.25);
                out[2] = blend(out[2], u, 0.0625);
            }
        }
    }
}

/// Apply the appropriate xBR scale to a source buffer.
pub fn scale(src: &[u32], src_w: usize, src_h: usize, mode: XbrScale) -> Vec<u32> {
    match mode {
        XbrScale::Xbr2x => scale2x(src, src_w, src_h),
        XbrScale::Xbr3x => scale3x(src, src_w, src_h),
        XbrScale::Xbr4x => scale4x(src, src_w, src_h),
    }
}
