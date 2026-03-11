//! xBR (scale By Rules) — 2x, 3x, and 4x pixel-art scaling.
//!
//! Based on Hyllian's xBR algorithm. Uses weighted color distance to detect
//! edges and applies directional interpolation rules to produce smooth,
//! anti-aliased output while preserving pixel-art detail.

use super::get;

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

// ── Color distance (YCbCr-weighted) ────────────────────────────────────────

/// Weighted color distance in YCbCr-like space.
#[inline(always)]
fn color_dist(a: u32, b: u32) -> f32 {
    if a == b {
        return 0.0;
    }
    let ar = ((a >> 16) & 0xFF) as f32;
    let ag = ((a >> 8) & 0xFF) as f32;
    let ab = (a & 0xFF) as f32;
    let br = ((b >> 16) & 0xFF) as f32;
    let bg = ((b >> 8) & 0xFF) as f32;
    let bb = (b & 0xFF) as f32;

    let dr = ar - br;
    let dg = ag - bg;
    let db = ab - bb;

    // Approximate YCbCr weighting
    let dy = 0.299 * dr + 0.587 * dg + 0.114 * db;
    let dcb = -0.169 * dr - 0.331 * dg + 0.500 * db;
    let dcr = 0.500 * dr - 0.419 * dg - 0.081 * db;

    48.0 * dy * dy + 7.0 * dcb * dcb + 6.0 * dcr * dcr
}

/// Blend two ARGB colors with weight alpha (0.0 = all a, 1.0 = all b).
#[inline(always)]
fn blend(a: u32, b: u32, alpha: f32) -> u32 {
    if alpha <= 0.0 { return a; }
    if alpha >= 1.0 { return b; }
    let inv = 1.0 - alpha;
    let r = (((a >> 16) & 0xFF) as f32 * inv + ((b >> 16) & 0xFF) as f32 * alpha).round() as u32;
    let g = (((a >> 8) & 0xFF) as f32 * inv + ((b >> 8) & 0xFF) as f32 * alpha).round() as u32;
    let bl = ((a & 0xFF) as f32 * inv + (b & 0xFF) as f32 * alpha).round() as u32;
    0xFF000000 | (r.min(255) << 16) | (g.min(255) << 8) | bl.min(255)
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

/// xBR edge weight for the bottom-right corner direction.
/// Compares diagonal vs orthogonal color distances to detect dominant edges.
///
/// Neighborhood layout (indexed into the 5x5 grid):
/// ```text
///   0  1  2  3  4
///   5  6  7  8  9
///  10 11 [12] 13 14
///  15 16  17 18 19
///  20 21  22 23 24
/// ```
#[inline(always)]
fn edge_weight_br(n: &Neighborhood) -> (f32, f32) {
    let p = &n.p;
    // Diagonal weight (bottom-right direction: 6→18 diagonal)
    let d_diag =
        color_dist(p[12], p[18]) +
        color_dist(p[12], p[8]) +
        color_dist(p[16], p[22]) +
        color_dist(p[7], p[14]) +
        4.0 * color_dist(p[13], p[17]);

    // Orthogonal weight (bottom and right edges)
    let d_ortho =
        color_dist(p[12], p[17]) +
        color_dist(p[12], p[13]) +
        color_dist(p[11], p[18]) +
        color_dist(p[8], p[23]) +
        4.0 * color_dist(p[7], p[16]);

    (d_diag, d_ortho)
}

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

            // Process all 4 corners by rotating the neighborhood
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
        let (d_diag, d_ortho) = edge_weight_br(n);
        if d_diag < d_ortho && !eq(p[12], p[18]) {
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

    // Bottom-left corner (mirror horizontally: swap left/right)
    {
        let d_diag =
            color_dist(p[12], p[16]) +
            color_dist(p[12], p[6]) +
            color_dist(p[18], p[20]) +
            color_dist(p[7], p[10]) +
            4.0 * color_dist(p[11], p[17]);
        let d_ortho =
            color_dist(p[12], p[17]) +
            color_dist(p[12], p[11]) +
            color_dist(p[13], p[16]) +
            color_dist(p[6], p[21]) +
            4.0 * color_dist(p[7], p[18]);

        if d_diag < d_ortho && !eq(p[12], p[16]) {
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

    // Top-right corner (mirror vertically: swap top/bottom)
    {
        let d_diag =
            color_dist(p[12], p[8]) +
            color_dist(p[12], p[18]) +
            color_dist(p[6], p[4]) +
            color_dist(p[17], p[9]) +
            4.0 * color_dist(p[13], p[7]);
        let d_ortho =
            color_dist(p[12], p[7]) +
            color_dist(p[12], p[13]) +
            color_dist(p[11], p[8]) +
            color_dist(p[18], p[3]) +
            4.0 * color_dist(p[17], p[6]);

        if d_diag < d_ortho && !eq(p[12], p[8]) {
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

    // Top-left corner (mirror both)
    {
        let d_diag =
            color_dist(p[12], p[6]) +
            color_dist(p[12], p[16]) +
            color_dist(p[8], p[0]) +
            color_dist(p[17], p[5]) +
            4.0 * color_dist(p[11], p[7]);
        let d_ortho =
            color_dist(p[12], p[7]) +
            color_dist(p[12], p[11]) +
            color_dist(p[13], p[6]) +
            color_dist(p[16], p[1]) +
            4.0 * color_dist(p[17], p[8]);

        if d_diag < d_ortho && !eq(p[12], p[6]) {
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
        let (d_diag, d_ortho) = edge_weight_br(n);
        if d_diag < d_ortho && !eq(p[12], p[18]) {
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
        let d_diag =
            color_dist(p[12], p[16]) +
            color_dist(p[12], p[6]) +
            color_dist(p[18], p[20]) +
            color_dist(p[7], p[10]) +
            4.0 * color_dist(p[11], p[17]);
        let d_ortho =
            color_dist(p[12], p[17]) +
            color_dist(p[12], p[11]) +
            color_dist(p[13], p[16]) +
            color_dist(p[6], p[21]) +
            4.0 * color_dist(p[7], p[18]);

        if d_diag < d_ortho && !eq(p[12], p[16]) {
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
        let d_diag =
            color_dist(p[12], p[8]) +
            color_dist(p[12], p[18]) +
            color_dist(p[6], p[4]) +
            color_dist(p[17], p[9]) +
            4.0 * color_dist(p[13], p[7]);
        let d_ortho =
            color_dist(p[12], p[7]) +
            color_dist(p[12], p[13]) +
            color_dist(p[11], p[8]) +
            color_dist(p[18], p[3]) +
            4.0 * color_dist(p[17], p[6]);

        if d_diag < d_ortho && !eq(p[12], p[8]) {
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
        let d_diag =
            color_dist(p[12], p[6]) +
            color_dist(p[12], p[16]) +
            color_dist(p[8], p[0]) +
            color_dist(p[17], p[5]) +
            4.0 * color_dist(p[11], p[7]);
        let d_ortho =
            color_dist(p[12], p[7]) +
            color_dist(p[12], p[11]) +
            color_dist(p[13], p[6]) +
            color_dist(p[16], p[1]) +
            4.0 * color_dist(p[17], p[8]);

        if d_diag < d_ortho && !eq(p[12], p[6]) {
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
        let (d_diag, d_ortho) = edge_weight_br(n);
        if d_diag < d_ortho && !eq(p[12], p[18]) {
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

    // Bottom-left corner: affects out[4], out[8], out[12], out[13], out[14] → mapped to BL
    {
        let d_diag =
            color_dist(p[12], p[16]) +
            color_dist(p[12], p[6]) +
            color_dist(p[18], p[20]) +
            color_dist(p[7], p[10]) +
            4.0 * color_dist(p[11], p[17]);
        let d_ortho =
            color_dist(p[12], p[17]) +
            color_dist(p[12], p[11]) +
            color_dist(p[13], p[16]) +
            color_dist(p[6], p[21]) +
            4.0 * color_dist(p[7], p[18]);

        if d_diag < d_ortho && !eq(p[12], p[16]) {
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

    // Top-right corner: affects out[2], out[3], out[7] → mapped to TR
    {
        let d_diag =
            color_dist(p[12], p[8]) +
            color_dist(p[12], p[18]) +
            color_dist(p[6], p[4]) +
            color_dist(p[17], p[9]) +
            4.0 * color_dist(p[13], p[7]);
        let d_ortho =
            color_dist(p[12], p[7]) +
            color_dist(p[12], p[13]) +
            color_dist(p[11], p[8]) +
            color_dist(p[18], p[3]) +
            4.0 * color_dist(p[17], p[6]);

        if d_diag < d_ortho && !eq(p[12], p[8]) {
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

    // Top-left corner: affects out[0], out[1], out[4] → mapped to TL
    {
        let d_diag =
            color_dist(p[12], p[6]) +
            color_dist(p[12], p[16]) +
            color_dist(p[8], p[0]) +
            color_dist(p[17], p[5]) +
            4.0 * color_dist(p[11], p[7]);
        let d_ortho =
            color_dist(p[12], p[7]) +
            color_dist(p[12], p[11]) +
            color_dist(p[13], p[6]) +
            color_dist(p[16], p[1]) +
            4.0 * color_dist(p[17], p[8]);

        if d_diag < d_ortho && !eq(p[12], p[6]) {
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
