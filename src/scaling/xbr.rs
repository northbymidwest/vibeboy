//! xBR (scale By Rules) — 2x, 3x, and 4x pixel-art scaling.
//!
//! Based on Hyllian's xBR algorithm. Uses weighted color distance to detect
//! edges and applies two-level directional interpolation: Level 1 detects
//! diagonal edges via weighted neighborhood comparison, Level 2 refines
//! steep/shallow line angles for smoother curves.

use super::get;
use super::color_dist;
use super::blend_argb;

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

// ── Neighborhood sampling ─────────────────────────────────────────────────

/// 5x5 neighborhood around a center pixel.
///
/// ```text
///   0  1  2  3  4
///   5  6  7  8  9
///  10 11 [12] 13 14
///  15 16  17 18 19
///  20 21  22 23 24
/// ```
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

// ── Corner parameters ─────────────────────────────────────────────────────

/// Indices into the 5x5 neighborhood for one corner's edge analysis.
///
/// Using Hyllian's naming: E (center=12), F/H (axis neighbors toward corner),
/// I (diagonal at corner), C/G (cross-diagonals), B/D (opposites of H/F),
/// F4/H5 (beyond F/H), I4/I5 (beyond I along F/H directions).
struct CornerParams {
    f: usize,
    h: usize,
    i: usize,
    c: usize,
    g: usize,
    b: usize,
    d: usize,
    f4: usize,
    h5: usize,
    i4: usize,
    i5: usize,
}

// BR: F=right(13), H=below(17), I=bottom-right(18)
const CORNER_BR: CornerParams = CornerParams {
    f: 13, h: 17, i: 18, c: 8, g: 16, b: 7, d: 11,
    f4: 14, h5: 22, i4: 19, i5: 23,
};
// BL: F=left(11), H=below(17), I=bottom-left(16)
const CORNER_BL: CornerParams = CornerParams {
    f: 11, h: 17, i: 16, c: 6, g: 18, b: 7, d: 13,
    f4: 10, h5: 22, i4: 15, i5: 21,
};
// TR: F=right(13), H=above(7), I=top-right(8)
const CORNER_TR: CornerParams = CornerParams {
    f: 13, h: 7, i: 8, c: 18, g: 6, b: 17, d: 11,
    f4: 14, h5: 2, i4: 9, i5: 3,
};
// TL: F=left(11), H=above(7), I=top-left(6)
const CORNER_TL: CornerParams = CornerParams {
    f: 11, h: 7, i: 6, c: 16, g: 8, b: 17, d: 13,
    f4: 10, h5: 2, i4: 5, i5: 1,
};

// ── Edge detection ────────────────────────────────────────────────────────

/// Compute xBR edge weights for a corner.
///
///   e = d(E,C) + d(E,G) + d(I,H5) + d(I,F4) + 4*d(H,F)
///   i = d(H,D) + d(H,I5) + d(F,I4) + d(F,B) + 4*d(E,I)
///
/// e < i indicates a diagonal edge toward the corner.
#[inline(always)]
fn edge_weights(p: &[u32; 25], cp: &CornerParams) -> (f32, f32) {
    let ew = color_dist(p[12], p[cp.c])
        + color_dist(p[12], p[cp.g])
        + color_dist(p[cp.i], p[cp.h5])
        + color_dist(p[cp.i], p[cp.f4])
        + 4.0 * color_dist(p[cp.h], p[cp.f]);

    let iw = color_dist(p[cp.h], p[cp.d])
        + color_dist(p[cp.h], p[cp.i5])
        + color_dist(p[cp.f], p[cp.i4])
        + color_dist(p[cp.f], p[cp.b])
        + 4.0 * color_dist(p[12], p[cp.i]);

    (ew, iw)
}

/// Select interpolation color: the axis neighbor closer to E.
#[inline(always)]
fn pick_color(p: &[u32; 25], cp: &CornerParams) -> u32 {
    if color_dist(p[12], p[cp.f]) <= color_dist(p[12], p[cp.h]) {
        p[cp.f]
    } else {
        p[cp.h]
    }
}

/// Compute steep/shallow line detection for Level 2 interpolation.
///
///   ke = d(F,G),  ki = d(H,C)
///   left  = (2*ke <= ki) && E!=G && D!=G   (steep — edge along F direction)
///   up    = (ke >= 2*ki) && E!=C && B!=C   (shallow — edge along H direction)
#[inline(always)]
fn detect_direction(p: &[u32; 25], cp: &CornerParams) -> (bool, bool) {
    let ke = color_dist(p[cp.f], p[cp.g]);
    let ki = color_dist(p[cp.h], p[cp.c]);
    let left = ke * 2.0 <= ki && !eq(p[12], p[cp.g]) && !eq(p[cp.d], p[cp.g]);
    let up = ke >= ki * 2.0 && !eq(p[12], p[cp.c]) && !eq(p[cp.b], p[cp.c]);
    (left, up)
}

/// Sub-condition for Level 2 activation (used by 2x and 4x filters).
///
/// Requires that at least one of these patterns holds:
/// - F differs from both B, and H differs from D (no flat continuation)
/// - E matches I but F/H don't match far neighbors (diagonal continuity)
/// - E matches G or C (cross-diagonal continuity)
#[inline(always)]
fn sub_condition_24(p: &[u32; 25], cp: &CornerParams) -> bool {
    (!eq(p[cp.f], p[cp.b]) && !eq(p[cp.h], p[cp.d]))
        || (eq(p[12], p[cp.i]) && !eq(p[cp.f], p[cp.i4]) && !eq(p[cp.h], p[cp.i5]))
        || eq(p[12], p[cp.g])
        || eq(p[12], p[cp.c])
}

/// Sub-condition for Level 2 activation (3x variant — more permissive).
#[inline(always)]
fn sub_condition_3(p: &[u32; 25], cp: &CornerParams) -> bool {
    (!eq(p[cp.f], p[cp.b]) && !eq(p[cp.f], p[cp.c]))
        || (!eq(p[cp.h], p[cp.d]) && !eq(p[cp.h], p[cp.g]))
        || (eq(p[12], p[cp.i])
            && ((!eq(p[cp.f], p[cp.f4]) && !eq(p[cp.f], p[cp.i4]))
                || (!eq(p[cp.h], p[cp.h5]) && !eq(p[cp.h], p[cp.i5]))))
        || eq(p[12], p[cp.g])
        || eq(p[12], p[cp.c])
}

// ── Blend weight constants ────────────────────────────────────────────────

const B32: f32 = 32.0 / 256.0;
const B64: f32 = 64.0 / 256.0;
const B128: f32 = 128.0 / 256.0;
const B192: f32 = 192.0 / 256.0;
const B224: f32 = 224.0 / 256.0;

// ── xBR2x ─────────────────────────────────────────────────────────────────

/// 2x output pixel indices per corner: (corner, left_side, up_side).
///
/// ```text
///  [0] [1]
///  [2] [3]
/// ```
const OUT2X: [(usize, usize, usize); 4] = [
    (3, 2, 1), // BR
    (2, 3, 0), // BL
    (1, 0, 3), // TR
    (0, 1, 2), // TL
];

pub fn scale2x(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let dst_w = src_w * 2;
    let dst_h = src_h * 2;
    let mut dst = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let n = Neighborhood::sample(src, src_w, src_h, x as isize, y as isize);
            let e = n.p[12];
            let mut out = [e; 4];

            let corners: [&CornerParams; 4] = [&CORNER_BR, &CORNER_BL, &CORNER_TR, &CORNER_TL];
            for (cp, &(n3, n2, n1)) in corners.iter().zip(OUT2X.iter()) {
                filt2x(&n.p, cp, &mut out, n3, n2, n1);
            }

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

/// Process one corner for 2x scaling.
///
/// n3 = corner pixel, n2 = side toward H (steep/left), n1 = side toward F (shallow/up).
#[inline(always)]
fn filt2x(p: &[u32; 25], cp: &CornerParams, out: &mut [u32; 4], n3: usize, n2: usize, n1: usize) {
    // Quick reject: center matches both axis neighbors — no edge possible
    if eq(p[12], p[cp.h]) || eq(p[12], p[cp.f]) {
        return;
    }

    let (ew, iw) = edge_weights(p, cp);
    if ew > iw {
        return;
    }

    let px = pick_color(p, cp);

    if ew < iw && sub_condition_24(p, cp) {
        // Level 2: directional blending
        let (left, up) = detect_direction(p, cp);
        if left && up {
            out[n3] = blend_argb(out[n3], px, B224);
            let side = blend_argb(out[n2], px, B64);
            out[n2] = side;
            out[n1] = side;
        } else if left {
            out[n3] = blend_argb(out[n3], px, B192);
            out[n2] = blend_argb(out[n2], px, B64);
        } else if up {
            out[n3] = blend_argb(out[n3], px, B192);
            out[n1] = blend_argb(out[n1], px, B64);
        } else {
            // Pure diagonal
            out[n3] = blend_argb(out[n3], px, B128);
        }
    } else {
        // Fallback: basic diagonal blend
        out[n3] = blend_argb(out[n3], px, B128);
    }
}

// ── xBR3x ─────────────────────────────────────────────────────────────────

/// 3x output pixel indices per corner:
/// (corner, adj_along_h, adj_along_f, ext_along_h, ext_along_f).
///
/// ```text
///  [0] [1] [2]
///  [3] [4] [5]
///  [6] [7] [8]
/// ```
const OUT3X: [(usize, usize, usize, usize, usize); 4] = [
    (8, 7, 5, 6, 2), // BR
    (6, 7, 3, 8, 0), // BL
    (2, 1, 5, 0, 8), // TR
    (0, 1, 3, 2, 6), // TL
];

pub fn scale3x(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let dst_w = src_w * 3;
    let dst_h = src_h * 3;
    let mut dst = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let n = Neighborhood::sample(src, src_w, src_h, x as isize, y as isize);
            let e = n.p[12];
            let mut out = [e; 9];

            let corners: [&CornerParams; 4] = [&CORNER_BR, &CORNER_BL, &CORNER_TR, &CORNER_TL];
            for (cp, &(n8, n7, n5, n6, n2)) in corners.iter().zip(OUT3X.iter()) {
                filt3x(&n.p, cp, &mut out, n8, n7, n5, n6, n2);
            }

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

/// Process one corner for 3x scaling.
///
/// n8 = corner, n7 = adjacent along H edge, n5 = adjacent along F edge,
/// n6 = extended along H, n2 = extended along F.
#[inline(always)]
fn filt3x(
    p: &[u32; 25],
    cp: &CornerParams,
    out: &mut [u32; 9],
    n8: usize,
    n7: usize,
    n5: usize,
    n6: usize,
    n2: usize,
) {
    if eq(p[12], p[cp.h]) || eq(p[12], p[cp.f]) {
        return;
    }

    let (ew, iw) = edge_weights(p, cp);
    if ew > iw {
        return;
    }

    let px = pick_color(p, cp);

    if ew < iw && sub_condition_3(p, cp) {
        let (left, up) = detect_direction(p, cp);
        if left && up {
            let blended_n7 = blend_argb(out[n7], px, B192);
            let blended_n6 = blend_argb(out[n6], px, B64);
            out[n7] = blended_n7;
            out[n6] = blended_n6;
            out[n5] = blended_n7;
            out[n2] = blended_n6;
            out[n8] = px;
        } else if left {
            out[n7] = blend_argb(out[n7], px, B192);
            out[n5] = blend_argb(out[n5], px, B64);
            out[n6] = blend_argb(out[n6], px, B64);
            out[n8] = px;
        } else if up {
            out[n5] = blend_argb(out[n5], px, B192);
            out[n7] = blend_argb(out[n7], px, B64);
            out[n2] = blend_argb(out[n2], px, B64);
            out[n8] = px;
        } else {
            out[n8] = blend_argb(out[n8], px, B224);
            out[n5] = blend_argb(out[n5], px, B32);
            out[n7] = blend_argb(out[n7], px, B32);
        }
    } else {
        out[n8] = blend_argb(out[n8], px, B128);
    }
}

// ── xBR4x ─────────────────────────────────────────────────────────────────

/// 4x output pixel indices per corner:
/// (n15, n14, n13, n12, n11, n10, n7, n3) — using reference FILT4 naming.
///
/// ```text
///  [ 0] [ 1] [ 2] [ 3]
///  [ 4] [ 5] [ 6] [ 7]
///  [ 8] [ 9] [10] [11]
///  [12] [13] [14] [15]
/// ```
struct Out4x {
    n15: usize,
    n14: usize,
    n13: usize,
    n12: usize,
    n11: usize,
    n10: usize,
    n7: usize,
    n3: usize,
}

const OUT4X: [Out4x; 4] = [
    // BR: default orientation
    Out4x { n15: 15, n14: 14, n13: 13, n12: 12, n11: 11, n10: 10, n7: 7, n3: 3 },
    // BL: horizontal mirror
    Out4x { n15: 12, n14: 13, n13: 14, n12: 15, n11: 8, n10: 9, n7: 4, n3: 0 },
    // TR: vertical mirror
    Out4x { n15: 3, n14: 2, n13: 1, n12: 0, n11: 7, n10: 6, n7: 11, n3: 15 },
    // TL: both mirrors
    Out4x { n15: 0, n14: 1, n13: 2, n12: 3, n11: 4, n10: 5, n7: 8, n3: 12 },
];

pub fn scale4x(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let dst_w = src_w * 4;
    let dst_h = src_h * 4;
    let mut dst = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let n = Neighborhood::sample(src, src_w, src_h, x as isize, y as isize);
            let e = n.p[12];
            let mut out = [e; 16];

            let corners: [&CornerParams; 4] = [&CORNER_BR, &CORNER_BL, &CORNER_TR, &CORNER_TL];
            for (cp, o) in corners.iter().zip(OUT4X.iter()) {
                filt4x(&n.p, cp, &mut out, o);
            }

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

/// Process one corner for 4x scaling.
#[inline(always)]
fn filt4x(p: &[u32; 25], cp: &CornerParams, out: &mut [u32; 16], o: &Out4x) {
    if eq(p[12], p[cp.h]) || eq(p[12], p[cp.f]) {
        return;
    }

    let (ew, iw) = edge_weights(p, cp);
    if ew > iw {
        return;
    }

    let px = pick_color(p, cp);

    if ew < iw && sub_condition_24(p, cp) {
        let (left, up) = detect_direction(p, cp);
        if left && up {
            let blended_n13 = blend_argb(out[o.n13], px, B192);
            let blended_n12 = blend_argb(out[o.n12], px, B64);
            out[o.n13] = blended_n13;
            out[o.n12] = blended_n12;
            out[o.n15] = px;
            out[o.n14] = px;
            out[o.n11] = px;
            out[o.n10] = blended_n12;
            out[o.n3] = blended_n12;
            out[o.n7] = blended_n13;
        } else if left {
            out[o.n11] = blend_argb(out[o.n11], px, B192);
            out[o.n13] = blend_argb(out[o.n13], px, B192);
            out[o.n10] = blend_argb(out[o.n10], px, B64);
            out[o.n12] = blend_argb(out[o.n12], px, B64);
            out[o.n14] = px;
            out[o.n15] = px;
        } else if up {
            out[o.n14] = blend_argb(out[o.n14], px, B192);
            out[o.n7] = blend_argb(out[o.n7], px, B192);
            out[o.n10] = blend_argb(out[o.n10], px, B64);
            out[o.n3] = blend_argb(out[o.n3], px, B64);
            out[o.n11] = px;
            out[o.n15] = px;
        } else {
            out[o.n11] = blend_argb(out[o.n11], px, B128);
            out[o.n14] = blend_argb(out[o.n14], px, B128);
            out[o.n15] = px;
        }
    } else {
        out[o.n15] = blend_argb(out[o.n15], px, B128);
    }
}

// ── Public entry point ────────────────────────────────────────────────────

/// Apply the appropriate xBR scale to a source buffer.
pub fn scale(src: &[u32], src_w: usize, src_h: usize, mode: XbrScale) -> Vec<u32> {
    match mode {
        XbrScale::Xbr2x => scale2x(src, src_w, src_h),
        XbrScale::Xbr3x => scale3x(src, src_w, src_h),
        XbrScale::Xbr4x => scale4x(src, src_w, src_h),
    }
}
