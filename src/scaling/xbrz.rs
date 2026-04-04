//! xBRZ — edge-detection pixel-art scaler (2x–6x) by Zenju.
//!
//! Two-phase algorithm:
//!   1. Analyze 4×4 neighborhoods to determine which corners have edges
//!      and whether those edges are dominant (strong) or normal.
//!   2. For each corner, classify the edge direction (steep/shallow/diagonal)
//!      and apply a scale-specific blend pattern to the output block.
//!
//! Every output pixel is either the center color or a weighted blend between
//! the center and one edge neighbor, preserving the pixel-art aesthetic.

use super::{get, color_dist_bt2020};

// ── Scale factor enum ───────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XbrzScale { Xbrz2x, Xbrz3x, Xbrz4x, Xbrz5x, Xbrz6x }

impl XbrzScale {
    pub fn factor(self) -> u32 {
        match self {
            Self::Xbrz2x => 2, Self::Xbrz3x => 3, Self::Xbrz4x => 4,
            Self::Xbrz5x => 5, Self::Xbrz6x => 6,
        }
    }
}

// ── Perceptual color helpers ────────────────────────────────────────────────

/// Two colors are perceptually "equal" if their BT.2020 distance is small.
const EQUAL_THRESHOLD: f32 = 30.0;

#[inline(always)]
fn dist(a: u32, b: u32) -> f32 { color_dist_bt2020(a, b) }

#[inline(always)]
fn colors_equal(a: u32, b: u32) -> bool { dist(a, b) < EQUAL_THRESHOLD }

// ── Edge strength per corner ────────────────────────────────────────────────

/// Edge strength: none, normal, or dominant (strong diagonal).
const EDGE_NONE: u8 = 0;
const EDGE_NORMAL: u8 = 1;
const EDGE_DOMINANT: u8 = 2;

/// Ratio threshold for dominant vs normal edges.
const DOMINANT_THRESHOLD: f32 = 3.6;

/// Pack four 2-bit edge values into a single byte.
/// Layout: corner0=[1:0], corner1=[3:2], corner2=[5:4], corner3=[7:6]
#[inline(always)] fn pack_edge(c0: u8, c1: u8, c2: u8, c3: u8) -> u8 {
    c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)
}
#[inline(always)] fn read_edge(packed: u8, corner: u8) -> u8 {
    (packed >> (corner * 2)) & 3
}
#[inline(always)] fn write_edge(packed: &mut u8, corner: u8, val: u8) {
    let shift = corner * 2;
    *packed = (*packed & !(3 << shift)) | (val << shift);
}

/// Rotate the packed edge info by 90° clockwise: 0→1→2→3→0
#[inline(always)]
fn rotate_edges(packed: u8) -> u8 {
    let c0 = packed & 3;
    let c1 = (packed >> 2) & 3;
    let c2 = (packed >> 4) & 3;
    let c3 = (packed >> 6) & 3;
    c1 | (c2 << 2) | (c3 << 4) | (c0 << 6)
}

/// Analyze the junction at position (jx, jy) — the meeting point of pixels
/// (jx,jy), (jx+1,jy), (jx,jy+1), (jx+1,jy+1). Returns a packed byte
/// with edge strengths for all 4 corners of the junction.
///
/// The 4×4 neighborhood centered on the junction:
/// ```
///       b  c
///   e   f  g  h
///   i   j  k  l
///       n  o
/// ```
fn analyze_junction(src: &[u32], w: usize, h: usize, jx: isize, jy: isize) -> u8 {
    let f = get(src, w, h, jx,     jy);
    let g = get(src, w, h, jx + 1, jy);
    let j = get(src, w, h, jx,     jy + 1);
    let k = get(src, w, h, jx + 1, jy + 1);

    // Uniform 2×2 blocks have no edges
    if (f == g && j == k) || (f == j && g == k) { return 0; }

    let b  = get(src, w, h, jx,     jy - 1);
    let c  = get(src, w, h, jx + 1, jy - 1);
    let e  = get(src, w, h, jx - 1, jy);
    let h_ = get(src, w, h, jx + 2, jy);
    let i  = get(src, w, h, jx - 1, jy + 1);
    let l  = get(src, w, h, jx + 2, jy + 1);
    let n  = get(src, w, h, jx,     jy + 2);
    let o  = get(src, w, h, jx + 1, jy + 2);

    let d_jg = dist(i, f) + dist(f, c) + dist(n, k) + dist(k, h_) + 4.0 * dist(j, g);
    let d_fk = dist(e, j) + dist(j, o) + dist(b, g) + dist(g, l) + 4.0 * dist(f, k);

    let mut result = 0u8;
    if d_jg < d_fk {
        let s = if DOMINANT_THRESHOLD * d_jg < d_fk { EDGE_DOMINANT } else { EDGE_NORMAL };
        // j-g diagonal wins: affects f(BR=corner2) and k(TL=corner0)
        if f != g && f != j { write_edge(&mut result, 2, s); }
        if k != j && k != g { write_edge(&mut result, 0, s); }
    } else if d_fk < d_jg {
        let s = if DOMINANT_THRESHOLD * d_fk < d_jg { EDGE_DOMINANT } else { EDGE_NORMAL };
        // f-k diagonal wins: affects g(BL=corner3) and j(TR=corner1)
        if g != f && g != k { write_edge(&mut result, 3, s); }
        if j != f && j != k { write_edge(&mut result, 1, s); }
    }
    result
}

/// For each pixel, gather edge info from the 4 surrounding junctions.
/// Each pixel's 4 corners come from 4 different junctions.
fn analyze_edges(src: &[u32], w: usize, h: usize) -> Vec<u8> {
    let mut edges = vec![0u8; w * h];
    for py in 0..h {
        for px in 0..w {
            let ix = px as isize;
            let iy = py as isize;

            // Junction at (px, py) = bottom-right junction of pixel (px, py)
            let j_br = analyze_junction(src, w, h, ix, iy);
            // Junction at (px-1, py) = bottom-left junction
            let j_bl = analyze_junction(src, w, h, ix - 1, iy);
            // Junction at (px, py-1) = top-right junction
            let j_tr = analyze_junction(src, w, h, ix, iy - 1);
            // Junction at (px-1, py-1) = top-left junction
            let j_tl = analyze_junction(src, w, h, ix - 1, iy - 1);

            // Gather: each junction contributes one corner to this pixel
            let mut info = 0u8;
            write_edge(&mut info, 0, read_edge(j_tl, 0)); // TL corner from TL junction's TL
            write_edge(&mut info, 1, read_edge(j_tr, 1)); // TR corner from TR junction's TR
            write_edge(&mut info, 2, read_edge(j_br, 2)); // BR corner from BR junction's BR
            write_edge(&mut info, 3, read_edge(j_bl, 3)); // BL corner from BL junction's BL
            edges[py * w + px] = info;
        }
    }
    edges
}

// ── Edge classification ─────────────────────────────────────────────────────

/// Ratio for steep/shallow line detection.
const STEEP_THRESHOLD: f32 = 2.2;

/// Whether a corner should receive line blending (vs being left sharp).
#[inline(always)]
fn should_blend(edge_info: u8, center: u32, pixels: &CornerPixels) -> bool {
    // Dominant edges always blend
    if read_edge(edge_info, 2) >= EDGE_DOMINANT { return true; }
    // Conflicting adjacent edges suppress blending
    if read_edge(edge_info, 1) != EDGE_NONE && !colors_equal(center, pixels.g) { return false; }
    if read_edge(edge_info, 3) != EDGE_NONE && !colors_equal(center, pixels.c) { return false; }
    // Continuous same-color band suppresses blending
    if !colors_equal(center, pixels.i)
        && colors_equal(pixels.g, pixels.h)
        && colors_equal(pixels.h, pixels.i)
        && colors_equal(pixels.i, pixels.f)
        && colors_equal(pixels.f, pixels.c)
    { return false; }
    true
}

/// The 8 relevant neighbor pixels for a corner (after rotation, these are
/// always relative to the bottom-right corner).
struct CornerPixels {
    b: u32, c: u32,     // top-center, top-right
    f: u32, g: u32,     // center-right, bottom-right (the edge pixel)
    h: u32, i: u32,     // bottom-center, bottom-right-far
    d: u32, _e: u32,    // far-right, far-right-bottom
}

/// Edge direction classification.
enum EdgeDir { Steep, Shallow, SteepAndShallow, Diagonal }

/// Classify the edge direction at a corner.
///
/// `center` = the pixel being scaled
/// `cp.f` = right neighbor, `cp.h` = bottom neighbor, `cp.g` = diagonal
/// `cp.c` = top-right, `cp.b` = top, `cp.d` = far-right
///
/// Shallow = edge runs mostly horizontal (f-g distance is small relative to h-c).
/// Steep = edge runs mostly vertical (h-c distance is small relative to f-g).
fn classify_edge(center: u32, cp: &CornerPixels) -> EdgeDir {
    let fg = dist(cp.f, cp.g);
    let hc = dist(cp.h, cp.c);

    let is_shallow = STEEP_THRESHOLD * fg <= hc
        && !colors_equal(center, cp.g) && !colors_equal(cp.d, cp.g);
    let is_steep = STEEP_THRESHOLD * hc <= fg
        && !colors_equal(center, cp.c) && !colors_equal(cp.b, cp.c);

    match (is_steep, is_shallow) {
        (true, true) => EdgeDir::SteepAndShallow,
        (true, false) => EdgeDir::Steep,
        (false, true) => EdgeDir::Shallow,
        (false, false) => EdgeDir::Diagonal,
    }
}

// ── Blend pattern application ───────────────────────────────────────────────

/// A single blend instruction: write `weight` of the edge color at (row, col)
/// within the NxN output block.
struct BlendOp { row: u8, col: u8, weight: f32 }

/// Apply a set of blend operations to an output block.
/// Coordinates are rotated by `rotation` (0-3 for 0°/90°/180°/270°).
fn apply_blend(
    out: &mut [u32], out_w: usize,
    base_x: usize, base_y: usize, n: usize,
    rotation: u8, center: u32, edge: u32,
    ops: &[BlendOp],
) {
    for op in ops {
        let (r, c) = rotate_coord(op.row as usize, op.col as usize, n, rotation);
        let idx = (base_y + r) * out_w + (base_x + c);
        if op.weight >= 1.0 {
            out[idx] = edge;
        } else {
            out[idx] = weighted_mix(center, edge, op.weight);
        }
    }
}

/// Map blend-table coordinates (relative to bottom-right corner) to the
/// actual output block position for each rotation.
/// rot 0 = BR: no flip. rot 1 = BL: flip col. rot 2 = TL: flip both. rot 3 = TR: flip row.
#[inline(always)]
fn rotate_coord(row: usize, col: usize, n: usize, rotation: u8) -> (usize, usize) {
    let m = n - 1;
    match rotation & 3 {
        0 => (row, col),
        1 => (row, m - col),
        2 => (m - row, m - col),
        _ => (m - row, col),
    }
}

/// Weighted blend between two XRGB pixels.
#[inline(always)]
fn weighted_mix(a: u32, b: u32, w: f32) -> u32 {
    let inv = 1.0 - w;
    let r = (((a >> 16) & 0xFF) as f32 * inv + ((b >> 16) & 0xFF) as f32 * w + 0.5) as u32;
    let g = (((a >> 8) & 0xFF) as f32 * inv + ((b >> 8) & 0xFF) as f32 * w + 0.5) as u32;
    let b_ = ((a & 0xFF) as f32 * inv + (b & 0xFF) as f32 * w + 0.5) as u32;
    (r.min(255) << 16) | (g.min(255) << 8) | b_.min(255)
}

// ── Per-scale blend tables ──────────────────────────────────────────────────
//
// Each table defines which output pixels to blend for a given edge direction.
// Coordinates are relative to the bottom-right corner (rotation 0); the
// apply_blend function handles the other 3 rotations.
//
// The blend weights are chosen to approximate anti-aliased edges at each
// scale factor. These specific values define the xBRZ visual style.

macro_rules! ops {
    ($(($r:expr, $c:expr, $w:expr)),+ $(,)?) => {
        &[$(BlendOp { row: $r, col: $c, weight: $w }),+]
    }
}

// Blend tables derived from the xBRZ algorithm specification.
// Each table encodes which sub-pixels get blended and at what weight.
// Coordinates are relative to the bottom-right corner (rotation 0).

fn shallow_ops(n: usize) -> &'static [BlendOp] {
    match n {
        2 => ops![(1,0, 0.25), (1,1, 0.75)],
        3 => ops![(2,0, 0.25), (1,2, 0.25), (2,1, 0.75), (2,2, 1.0)],
        4 => ops![(3,0, 0.25), (2,2, 0.25), (3,1, 0.75), (2,3, 0.75), (3,2, 1.0), (3,3, 1.0)],
        5 => ops![(4,0, 0.25), (3,2, 0.25), (2,4, 0.25), (4,1, 0.75), (3,3, 0.75), (4,2, 1.0), (4,3, 1.0), (4,4, 1.0), (3,4, 1.0)],
        _ => ops![(5,0, 0.25), (4,2, 0.25), (3,4, 0.25), (5,1, 0.75), (4,3, 0.75), (3,5, 0.75), (5,2, 1.0), (5,3, 1.0), (5,4, 1.0), (5,5, 1.0), (4,4, 1.0), (4,5, 1.0)],
    }
}

fn steep_ops(n: usize) -> &'static [BlendOp] {
    // Steep = transpose of shallow (swap row/col)
    match n {
        2 => ops![(0,1, 0.25), (1,1, 0.75)],
        3 => ops![(0,2, 0.25), (2,1, 0.25), (1,2, 0.75), (2,2, 1.0)],
        4 => ops![(0,3, 0.25), (2,2, 0.25), (1,3, 0.75), (3,2, 0.75), (2,3, 1.0), (3,3, 1.0)],
        5 => ops![(0,4, 0.25), (2,3, 0.25), (4,2, 0.25), (1,4, 0.75), (3,3, 0.75), (2,4, 1.0), (3,4, 1.0), (4,4, 1.0), (4,3, 1.0)],
        _ => ops![(0,5, 0.25), (2,4, 0.25), (4,3, 0.25), (1,5, 0.75), (3,4, 0.75), (5,3, 0.75), (2,5, 1.0), (3,5, 1.0), (4,5, 1.0), (5,5, 1.0), (4,4, 1.0), (5,4, 1.0)],
    }
}

fn steep_and_shallow_ops(n: usize) -> &'static [BlendOp] {
    match n {
        2 => ops![(0,1, 0.25), (1,0, 0.25), (1,1, 5.0/6.0)],
        3 => ops![(0,2, 0.25), (2,0, 0.25), (2,1, 0.75), (1,2, 0.75), (2,2, 1.0)],
        4 => ops![(0,3, 0.25), (3,0, 0.25), (3,1, 0.75), (1,3, 0.75), (2,2, 1.0/3.0), (3,2, 1.0), (2,3, 1.0), (3,3, 1.0)],
        5 => ops![(0,4, 0.25), (2,3, 0.25), (1,4, 0.75), (4,0, 0.25), (3,2, 0.25), (4,1, 0.75), (3,3, 2.0/3.0), (2,4, 1.0), (3,4, 1.0), (4,4, 1.0), (4,2, 1.0), (4,3, 1.0)],
        _ => ops![(0,5, 0.25), (2,4, 0.25), (1,5, 0.75), (3,4, 0.75), (5,0, 0.25), (4,2, 0.25), (5,1, 0.75), (4,3, 0.75), (2,5, 1.0), (3,5, 1.0), (4,5, 1.0), (5,5, 1.0), (4,4, 1.0), (5,4, 1.0), (5,2, 1.0), (5,3, 1.0)],
    }
}

fn diagonal_ops(n: usize) -> &'static [BlendOp] {
    match n {
        2 => ops![(1,1, 0.5)],
        3 => ops![(1,2, 1.0/8.0), (2,1, 1.0/8.0), (2,2, 7.0/8.0)],
        4 => ops![(3,2, 0.5), (2,3, 0.5), (3,3, 1.0)],
        5 => ops![(4,2, 1.0/8.0), (3,3, 1.0/8.0), (2,4, 1.0/8.0), (4,3, 7.0/8.0), (3,4, 7.0/8.0), (4,4, 1.0)],
        _ => ops![(5,3, 0.5), (4,4, 0.5), (3,5, 0.5), (4,5, 1.0), (5,4, 1.0), (5,5, 1.0)],
    }
}

fn corner_ops(n: usize) -> &'static [BlendOp] {
    // Corner weights approximate the area outside a quarter-circle inscribed
    // in the NxN output block. 2x: 1-π/4 ≈ 0.2146, then geometrically scaled.
    match n {
        2 => ops![(1,1, 0.21)],
        3 => ops![(2,2, 0.45)],
        4 => ops![(3,3, 0.68), (3,2, 0.09), (2,3, 0.09)],
        5 => ops![(4,4, 0.86), (4,3, 0.23), (3,4, 0.23)],
        _ => ops![(5,5, 0.97), (5,4, 0.42), (4,5, 0.42), (5,3, 0.06), (3,5, 0.06)],
    }
}

// ── Main scaling function ───────────────────────────────────────────────────

/// Scale an image using xBRZ at the specified scale factor (2x–6x).
pub fn scale(src: &[u32], w: usize, h: usize, mode: XbrzScale) -> Vec<u32> {
    let n = mode.factor() as usize;
    let ow = w * n;
    let oh = h * n;
    let mut out = vec![0u32; ow * oh];

    // Phase 1: analyze edges
    let edges = analyze_edges(src, w, h);

    // Phase 2: scale each pixel
    for py in 0..h {
        for px in 0..w {
            let center = get(src, w, h, px as isize, py as isize);
            let ox = px * n;
            let oy = py * n;

            // Fill entire output block with center color
            for r in 0..n {
                for c in 0..n {
                    out[(oy + r) * ow + (ox + c)] = center;
                }
            }

            let edge_info = edges[py * w + px];
            if edge_info == 0 { continue; } // no edges, block is uniform

            let ix = px as isize;
            let iy = py as isize;

            // Process each of 4 corners via rotation.
            // rot 0 = BR corner, rot 1 = BL, rot 2 = TL, rot 3 = TR.
            // Direction vectors point from center toward the corner being processed.
            const DIRS: [(isize, isize); 4] = [(1, 1), (-1, 1), (-1, -1), (1, -1)];

            for rotation in 0u8..4 {
                // Rotate edge info to bring current corner to position 2 (bottom-right)
                let rotated = {
                    let mut r = edge_info;
                    for _ in 0..rotation { r = rotate_edges(r); }
                    r
                };

                if read_edge(rotated, 2) == EDGE_NONE { continue; }

                // Gather neighbors using direction vectors.
                // (dx, dy) points toward the corner being processed.
                let (dx, dy) = DIRS[rotation as usize];
                let cp = CornerPixels {
                    f: get(src, w, h, ix + dx, iy),          // horizontal toward corner
                    h: get(src, w, h, ix, iy + dy),          // vertical toward corner
                    g: get(src, w, h, ix - dx, iy + dy),     // opposite-horizontal, same-vertical
                    i: get(src, w, h, ix + dx, iy + dy),     // the diagonal corner itself
                    c: get(src, w, h, ix + dx, iy - dy),     // horizontal toward, vertical away
                    b: get(src, w, h, ix, iy - dy),          // vertical away from corner
                    d: get(src, w, h, ix - dx, iy),          // horizontal away from corner
                    _e: get(src, w, h, ix - dx, iy - dy),    // opposite diagonal
                };

                // Blend target: the closer of the two orthogonal neighbors
                let target = if dist(center, cp.f) <= dist(center, cp.h) { cp.f } else { cp.h };

                if !should_blend(rotated, center, &cp) {
                    // Even without line blending, corners get a small AA blend
                    apply_blend(&mut out, ow, ox, oy, n, rotation, center, target, corner_ops(n));
                    continue;
                }

                let dir = classify_edge(center, &cp);
                let ops = match dir {
                    EdgeDir::Shallow => shallow_ops(n),
                    EdgeDir::Steep => steep_ops(n),
                    EdgeDir::SteepAndShallow => steep_and_shallow_ops(n),
                    EdgeDir::Diagonal => diagonal_ops(n),
                };
                apply_blend(&mut out, ow, ox, oy, n, rotation, center, target, ops);
            }
        }
    }
    out
}
