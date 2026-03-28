//! HQx (hq2x, hq3x, hq4x) pixel-art magnification filters.
//!
//! Implements Maxim Stepin's HQx algorithm using rotation symmetry: each scale
//! factor has a single kernel function that is applied once per quadrant with
//! a rotated view of the 3x3 neighborhood. Edge detection uses inline YUV
//! color distance with integer arithmetic.

use super::HqxScale;

// ── Color comparison ────────────────────────────────────────────────────────

/// Returns true if two colors are perceptually different in YUV space.
/// Computes YUV delta inline using integer math (no lookup table).
/// Thresholds: dY > 48, dU > 7, dV > 6 (matching Stepin's original,
/// scaled by 1000 to avoid floating point).
#[inline(always)]
fn colors_differ(a: u32, b: u32) -> bool {
    if a == b { return false; }
    let dr = ((a >> 16) & 0xFF) as i32 - ((b >> 16) & 0xFF) as i32;
    let dg = ((a >> 8) & 0xFF) as i32 - ((b >> 8) & 0xFF) as i32;
    let db = (a & 0xFF) as i32 - (b & 0xFF) as i32;
    let dy = (299 * dr + 587 * dg + 114 * db).unsigned_abs();
    let du = (-169 * dr - 331 * dg + 500 * db).unsigned_abs();
    let dv = (500 * dr - 419 * dg - 81 * db).unsigned_abs();
    dy > 48000 || du > 7000 || dv > 6000
}

// ── Pixel blending ──────────────────────────────────────────────────────────

#[inline(always)]
fn mix2(c1: u32, w1: u32, c2: u32, w2: u32, shift: u32) -> u32 {
    let hi = ((c1 & 0xFF00FF00) >> 8).wrapping_mul(w1)
        .wrapping_add(((c2 & 0xFF00FF00) >> 8).wrapping_mul(w2));
    let lo = (c1 & 0x00FF00FF).wrapping_mul(w1)
        .wrapping_add((c2 & 0x00FF00FF).wrapping_mul(w2));
    ((hi << (8 - shift)) & 0xFF00FF00) | ((lo >> shift) & 0x00FF00FF)
}

#[inline(always)]
fn mix3(c1: u32, w1: u32, c2: u32, w2: u32, c3: u32, w3: u32, shift: u32) -> u32 {
    let hi = ((c1 & 0xFF00FF00) >> 8).wrapping_mul(w1)
        .wrapping_add(((c2 & 0xFF00FF00) >> 8).wrapping_mul(w2))
        .wrapping_add(((c3 & 0xFF00FF00) >> 8).wrapping_mul(w3));
    let lo = (c1 & 0x00FF00FF).wrapping_mul(w1)
        .wrapping_add((c2 & 0x00FF00FF).wrapping_mul(w2))
        .wrapping_add((c3 & 0x00FF00FF).wrapping_mul(w3));
    ((hi << (8 - shift)) & 0xFF00FF00) | ((lo >> shift) & 0x00FF00FF)
}

// ── Rotatable 3x3 neighborhood ──────────────────────────────────────────────

struct CornerView {
    edge: u32,
    diag: u32, top: u32, left: u32, center: u32, right: u32, bot: u32,
}

const TRANSFORMS: [[usize; 9]; 4] = [
    [0,1,2,3,4,5,6,7,8], [2,1,0,5,4,3,8,7,6],
    [6,7,8,3,4,5,0,1,2], [8,7,6,5,4,3,2,1,0],
];

const ROTATIONS: [[usize; 9]; 4] = [
    [0,1,2,3,4,5,6,7,8], [2,5,8,1,4,7,0,3,6],
    [6,3,0,7,4,1,8,5,2], [8,7,6,5,4,3,2,1,0],
];

#[inline(always)]
fn remap_edge_pattern(pattern: u32, perm: &[usize; 9], reverse: bool) -> u32 {
    let mut out = 0u32;
    let indices = [0,1,2,3,5,6,7,8];
    for (bit_pos, &idx) in indices.iter().enumerate() {
        let src_idx = perm[idx];
        let src_bit = if src_idx > 4 { src_idx - 1 } else { src_idx };
        let actual_bit = if reverse { 7 - src_bit } else { src_bit };
        out |= ((pattern >> actual_bit) & 1) << bit_pos;
    }
    out
}

fn build_corner_view(pixels: &[u32; 9], pattern: u32, perm: &[usize; 9], reverse: bool) -> CornerView {
    CornerView {
        edge: remap_edge_pattern(pattern, perm, reverse),
        diag: pixels[perm[0]], top: pixels[perm[1]], left: pixels[perm[3]],
        center: pixels[perm[4]], right: pixels[perm[5]], bot: pixels[perm[7]],
    }
}

#[inline(always)]
fn edge_match(v: &CornerView, mask: u32, expected: u32) -> bool {
    (v.edge & mask) == expected
}

// ── Edge pattern constants ──────────────────────────────────────────────────

const EDGE_HORIZ: [(u32,u32); 2] = [(0xbf,0x37), (0xdb,0x13)];
const EDGE_VERT: [(u32,u32); 2] = [(0xdb,0x49), (0xef,0x6d)];
const EDGE_SHARP_DIAG: [(u32,u32); 3] = [(0x0b,0x0b), (0xfe,0x4a), (0xfe,0x1a)];
const EDGE_PARTIAL_DIAG: [(u32,u32); 13] = [
    (0x6f,0x2a),(0x5b,0x0a),(0xbf,0x3a),(0xdf,0x5a),(0x9f,0x8a),(0xcf,0x8a),
    (0xef,0x4e),(0x3f,0x0e),(0xfb,0x5a),(0xbb,0x8a),(0x7f,0x5a),(0xaf,0x8a),(0xeb,0x8a),
];

#[inline(always)]
fn any_match(v: &CornerView, pairs: &[(u32,u32)]) -> bool {
    pairs.iter().any(|&(m,e)| edge_match(v, m, e))
}

// ── hq2x ────────────────────────────────────────────────────────────────────

fn hq2x_corner(v: &CornerView) -> u32 {
    let CornerView { diag, top, left, center, right, bot, .. } = *v;
    if any_match(v, &EDGE_HORIZ) && colors_differ(top, right) { return mix2(center,3,left,1,2); }
    if any_match(v, &EDGE_VERT) && colors_differ(bot, left) { return mix2(center,3,top,1,2); }
    if any_match(v, &EDGE_SHARP_DIAG) && colors_differ(left, top) { return center; }
    if any_match(v, &EDGE_PARTIAL_DIAG) && colors_differ(left, top) { return mix2(center,3,diag,1,2); }
    if edge_match(v,0x0b,0x08) { return mix3(center,2,diag,1,top,1,2); }
    if edge_match(v,0x0b,0x02) { return mix3(center,2,diag,1,left,1,2); }
    if edge_match(v,0x2f,0x2f) { return mix3(center,14,left,1,top,1,4); }
    if any_match(v, &EDGE_HORIZ) { return mix3(center,5,top,2,left,1,3); }
    if any_match(v, &EDGE_VERT) { return mix3(center,5,left,2,top,1,3); }
    if edge_match(v,0x1b,0x03)||edge_match(v,0x4f,0x43)||edge_match(v,0x8b,0x83)||edge_match(v,0x6b,0x43) { return mix2(center,3,left,1,2); }
    if edge_match(v,0x4b,0x09)||edge_match(v,0x8b,0x89)||edge_match(v,0x1f,0x19)||edge_match(v,0x3b,0x19) { return mix2(center,3,top,1,2); }
    if edge_match(v,0x7e,0x2a)||edge_match(v,0xef,0xab)||edge_match(v,0xbf,0x8f)||edge_match(v,0x7e,0x0e) { return mix3(center,2,left,3,top,3,3); }
    if edge_match(v,0xfb,0x6a)||edge_match(v,0x6f,0x6e)||edge_match(v,0x3f,0x3e)||edge_match(v,0xfb,0xfa)||edge_match(v,0xdf,0xde)||edge_match(v,0xdf,0x1e) { return mix2(center,3,diag,1,2); }
    if edge_match(v,0x0a,0x00)||edge_match(v,0x4f,0x4b)||edge_match(v,0x9f,0x1b)||edge_match(v,0x2f,0x0b)||edge_match(v,0xbe,0x0a)||edge_match(v,0xee,0x0a)||edge_match(v,0x7e,0x0a)||edge_match(v,0xeb,0x4b)||edge_match(v,0x3b,0x1b) { return mix3(center,2,left,1,top,1,2); }
    mix3(center,6,left,1,top,1,3)
}

// ── hq3x ────────────────────────────────────────────────────────────────────

fn hq3x_corner_and_edge(v: &CornerView) -> (u32, u32) {
    let CornerView { diag, top, left, center, right, bot, .. } = *v;
    let corner = if any_match(v, &EDGE_VERT) && colors_differ(bot, left) { mix2(center,3,top,1,2) }
    else if any_match(v, &EDGE_HORIZ) && colors_differ(top, right) { mix2(center,3,left,1,2) }
    else if any_match(v, &EDGE_SHARP_DIAG) && colors_differ(left, top) { center }
    else if any_match(v, &EDGE_PARTIAL_DIAG) && colors_differ(left, top) { mix2(center,3,diag,1,2) }
    else if edge_match(v,0x4b,0x09)||edge_match(v,0x8b,0x89)||edge_match(v,0x1f,0x19)||edge_match(v,0x3b,0x19) { mix2(center,3,top,1,2) }
    else if edge_match(v,0x1b,0x03)||edge_match(v,0x4f,0x43)||edge_match(v,0x8b,0x83)||edge_match(v,0x6b,0x43) { mix2(center,3,left,1,2) }
    else if edge_match(v,0x7e,0x2a)||edge_match(v,0xef,0xab)||edge_match(v,0xbf,0x8f)||edge_match(v,0x7e,0x0e) { mix2(left,1,top,1,1) }
    else if edge_match(v,0x4f,0x4b)||edge_match(v,0x9f,0x1b)||edge_match(v,0x2f,0x0b)||edge_match(v,0xbe,0x0a)||edge_match(v,0xee,0x0a)||edge_match(v,0x7e,0x0a)||edge_match(v,0xeb,0x4b)||edge_match(v,0x3b,0x1b) { mix3(center,2,left,7,top,7,4) }
    else if edge_match(v,0x0b,0x08)||edge_match(v,0xf9,0x68)||edge_match(v,0xf3,0x62)||edge_match(v,0x6d,0x6c)||edge_match(v,0x67,0x66)||edge_match(v,0x3d,0x3c)||edge_match(v,0x37,0x36)||edge_match(v,0xf9,0xf8)||edge_match(v,0xdd,0xdc)||edge_match(v,0xf3,0xf2)||edge_match(v,0xd7,0xd6)||edge_match(v,0xdd,0x1c)||edge_match(v,0xd7,0x16)||edge_match(v,0x0b,0x02) { mix2(center,3,diag,1,2) }
    else { mix3(center,2,left,1,top,1,2) };

    let edge = if (edge_match(v,0xfe,0xde)||edge_match(v,0x9e,0x16)||edge_match(v,0xda,0x12)||edge_match(v,0x17,0x16)||edge_match(v,0x5b,0x12)||edge_match(v,0xbb,0x12)) && colors_differ(top, right) { center }
    else if (edge_match(v,0x0f,0x0b)||edge_match(v,0x5e,0x0a)||edge_match(v,0xfb,0x7b)||edge_match(v,0x3b,0x0b)||edge_match(v,0xbe,0x0a)||edge_match(v,0x7a,0x0a)) && colors_differ(left, top) { center }
    else if edge_match(v,0xbf,0x8f)||edge_match(v,0x7e,0x0e)||edge_match(v,0xbf,0x37)||edge_match(v,0xdb,0x13) { mix2(top,3,center,1,2) }
    else if edge_match(v,0x02,0x00)||edge_match(v,0x7c,0x28)||edge_match(v,0xed,0xa9)||edge_match(v,0xf5,0xb4)||edge_match(v,0xd9,0x90) { mix2(center,3,top,1,2) }
    else if edge_match(v,0x4f,0x4b)||edge_match(v,0xfb,0x7b)||edge_match(v,0xfe,0x7e)||edge_match(v,0x9f,0x1b)||edge_match(v,0x2f,0x0b)||edge_match(v,0xbe,0x0a)||edge_match(v,0x7e,0x0a)||edge_match(v,0xfb,0x4b)||edge_match(v,0xfb,0xdb)||edge_match(v,0xfe,0xde)||edge_match(v,0xfe,0x56)||edge_match(v,0x57,0x56)||edge_match(v,0x97,0x16)||edge_match(v,0x3f,0x1e)||edge_match(v,0xdb,0x12)||edge_match(v,0xbb,0x12) { mix2(center,7,top,1,3) }
    else { center };
    (corner, edge)
}

// ── hq4x ────────────────────────────────────────────────────────────────────

fn hq4x_quadrant(v: &CornerView) -> [u32; 4] {
    let CornerView { diag, top, left, center, right, bot, .. } = *v;
    let horiz_edge = any_match(v, &EDGE_HORIZ) && colors_differ(top, right);
    let vert_edge = any_match(v, &EDGE_VERT) && colors_differ(bot, left);
    let partial_diag = any_match(v, &EDGE_PARTIAL_DIAG) && colors_differ(left, top);
    let has_vert = any_match(v, &EDGE_VERT);
    let has_horiz = any_match(v, &EDGE_HORIZ);
    let left_edge = edge_match(v,0x1b,0x03)||edge_match(v,0x4f,0x43)||edge_match(v,0x8b,0x83)||edge_match(v,0x6b,0x43);
    let top_edge = edge_match(v,0x4b,0x09)||edge_match(v,0x8b,0x89)||edge_match(v,0x1f,0x19)||edge_match(v,0x3b,0x19);
    let near_diag = edge_match(v,0x0b,0x08)||edge_match(v,0xf9,0x68)||edge_match(v,0xf3,0x62)||edge_match(v,0x6d,0x6c)||edge_match(v,0x67,0x66)||edge_match(v,0x3d,0x3c)||edge_match(v,0x37,0x36)||edge_match(v,0xf9,0xf8)||edge_match(v,0xdd,0xdc)||edge_match(v,0xf3,0xf2)||edge_match(v,0xd7,0xd6)||edge_match(v,0xdd,0x1c)||edge_match(v,0xd7,0x16)||edge_match(v,0x0b,0x02);
    let sharp = any_match(v, &EDGE_SHARP_DIAG) && colors_differ(left, top);
    let cross = (edge_match(v,0x0f,0x0b)||edge_match(v,0x2b,0x0b)||edge_match(v,0xfe,0x4a)||edge_match(v,0xfe,0x1a)) && colors_differ(left, top);
    let all_diff = edge_match(v,0x2f,0x2f);
    let smooth = edge_match(v,0x0a,0x00);
    let inner_top = edge_match(v,0x0b,0x09);
    let diag_a = edge_match(v,0x7e,0x2a)||edge_match(v,0xef,0xab);
    let diag_b = edge_match(v,0xbf,0x8f)||edge_match(v,0x7e,0x0e);
    let gradient = edge_match(v,0x4f,0x4b)||edge_match(v,0x9f,0x1b)||edge_match(v,0x2f,0x0b)||edge_match(v,0xbe,0x0a)||edge_match(v,0xee,0x0a)||edge_match(v,0x7e,0x0a)||edge_match(v,0xeb,0x4b)||edge_match(v,0x3b,0x1b);
    let inner_left = edge_match(v,0x0b,0x03);

    let px00 = if horiz_edge { mix2(center,5,left,3,3) } else if vert_edge { mix2(center,5,top,3,3) } else if sharp { center } else if partial_diag { mix2(center,5,diag,3,3) } else if has_vert { mix2(center,3,left,1,2) } else if has_horiz { mix2(center,3,top,1,2) } else if left_edge { mix2(center,5,left,3,3) } else if top_edge { mix2(center,5,top,3,3) } else if edge_match(v,0x0f,0x0b)||edge_match(v,0x5e,0x0a)||edge_match(v,0x2b,0x0b)||edge_match(v,0xbe,0x0a)||edge_match(v,0x7a,0x0a)||edge_match(v,0xee,0x0a) { mix2(top,1,left,1,1) } else if near_diag { mix2(center,5,diag,3,3) } else { mix3(center,2,top,1,left,1,2) };

    let px01 = if horiz_edge { mix2(center,7,left,1,3) } else if cross { center } else if partial_diag { mix2(center,3,diag,1,2) } else if all_diff { center } else if smooth { mix3(center,5,top,2,left,1,3) } else if edge_match(v,0x0b,0x08) { mix3(center,5,top,2,diag,1,3) } else if inner_top { mix2(center,5,top,3,3) } else if has_horiz { mix2(top,3,center,1,2) } else if diag_a { mix3(top,2,center,1,left,1,2) } else if diag_b { mix2(top,5,left,3,3) } else if left_edge { mix2(center,7,left,1,3) } else if edge_match(v,0xf3,0x62)||edge_match(v,0x67,0x66)||edge_match(v,0x37,0x36)||edge_match(v,0xf3,0xf2)||edge_match(v,0xd7,0xd6)||edge_match(v,0xd7,0x16)||edge_match(v,0x0b,0x02) { mix2(center,3,diag,1,2) } else if gradient { mix2(top,1,center,1,1) } else { mix2(center,3,top,1,2) };

    let px10 = if vert_edge { mix2(center,7,top,1,3) } else if cross { center } else if partial_diag { mix2(center,3,diag,1,2) } else if all_diff { center } else if smooth { mix3(center,5,left,2,top,1,3) } else if edge_match(v,0x0b,0x02) { mix3(center,5,left,2,diag,1,3) } else if inner_left { mix2(center,5,left,3,3) } else if has_vert { mix2(left,3,center,1,2) } else if diag_b { mix3(left,2,center,1,top,1,2) } else if diag_a { mix2(left,5,top,3,3) } else if top_edge { mix2(center,7,top,1,3) } else if edge_match(v,0x0b,0x08)||edge_match(v,0xf9,0x68)||edge_match(v,0x6d,0x6c)||edge_match(v,0x3d,0x3c)||edge_match(v,0xf9,0xf8)||edge_match(v,0xdd,0xdc)||edge_match(v,0xdd,0x1c) { mix2(center,3,diag,1,2) } else if gradient { mix2(left,1,center,1,1) } else { mix2(center,3,left,1,2) };

    let px11 = if (edge_match(v,0x7f,0x2b)||edge_match(v,0xef,0xab)||edge_match(v,0xbf,0x8f)||edge_match(v,0x7f,0x0f)) && colors_differ(left, top) { center } else if partial_diag { mix2(center,7,diag,1,3) } else if inner_left { mix2(center,7,left,1,3) } else if inner_top { mix2(center,7,top,1,3) } else if smooth||edge_match(v,0x7e,0x2a)||edge_match(v,0xef,0xab)||edge_match(v,0xbf,0x8f)||edge_match(v,0x7e,0x0e) { mix3(center,6,left,1,top,1,3) } else if near_diag { mix2(center,7,diag,1,3) } else { center };

    [px00, px01, px10, px11]
}

// ── Neighborhood ────────────────────────────────────────────────────────────

fn sample_neighborhood(src: &[u32], w: usize, h: usize, x: usize, y: usize) -> [u32; 9] {
    let x0 = x.saturating_sub(1); let x1 = (x+1).min(w-1);
    let y0 = y.saturating_sub(1); let y1 = (y+1).min(h-1);
    [src[y0*w+x0], src[y0*w+x], src[y0*w+x1],
     src[y*w+x0],  src[y*w+x],  src[y*w+x1],
     src[y1*w+x0], src[y1*w+x], src[y1*w+x1]]
}

fn compute_edge_pattern(pixels: &[u32; 9]) -> u32 {
    let c = pixels[4];
    let mut pat = 0u32;
    for i in 0..9 {
        if i == 4 { continue; }
        let bit = if i > 4 { i - 1 } else { i };
        if colors_differ(c, pixels[i]) {
            pat |= 1 << bit;
        }
    }
    pat
}

// ── Public API ──────────────────────────────────────────────────────────────

pub fn hq2x(src: &[u32], w: usize, h: usize) -> Vec<u32> {

    let dw = w * 2;
    let mut dst = vec![0u32; dw * h * 2];
    for y in 0..h {
        for x in 0..w {
            let nb = sample_neighborhood(src, w, h, x, y);
            let pat = compute_edge_pattern(&nb);
            let dx = x * 2; let dy = y * 2;
            for (i, xform) in TRANSFORMS.iter().enumerate() {
                let view = build_corner_view(&nb, pat, xform, false);
                dst[(dy + i/2) * dw + dx + (i%2)] = hq2x_corner(&view);
            }
        }
    }
    dst
}

pub fn hq3x(src: &[u32], w: usize, h: usize) -> Vec<u32> {

    let dw = w * 3;
    let mut dst = vec![0u32; dw * h * 3];
    for y in 0..h {
        for x in 0..w {
            let nb = sample_neighborhood(src, w, h, x, y);
            let pat = compute_edge_pattern(&nb);
            let dx = x * 3; let dy = y * 3;
            let (c00,e01) = hq3x_corner_and_edge(&build_corner_view(&nb, pat, &ROTATIONS[0], false));
            let (c02,e12) = hq3x_corner_and_edge(&build_corner_view(&nb, pat, &ROTATIONS[1], true));
            let (c20,e10) = hq3x_corner_and_edge(&build_corner_view(&nb, pat, &ROTATIONS[2], true));
            let (c22,e21) = hq3x_corner_and_edge(&build_corner_view(&nb, pat, &ROTATIONS[3], false));
            dst[dy*dw+dx]=c00; dst[dy*dw+dx+1]=e01; dst[dy*dw+dx+2]=c02;
            dst[(dy+1)*dw+dx]=e10; dst[(dy+1)*dw+dx+1]=nb[4]; dst[(dy+1)*dw+dx+2]=e12;
            dst[(dy+2)*dw+dx]=c20; dst[(dy+2)*dw+dx+1]=e21; dst[(dy+2)*dw+dx+2]=c22;
        }
    }
    dst
}

pub fn hq4x(src: &[u32], w: usize, h: usize) -> Vec<u32> {

    let dw = w * 4;
    let mut dst = vec![0u32; dw * h * 4];
    for y in 0..h {
        for x in 0..w {
            let nb = sample_neighborhood(src, w, h, x, y);
            let pat = compute_edge_pattern(&nb);
            let dx = x * 4; let dy = y * 4;
            let tl = hq4x_quadrant(&build_corner_view(&nb, pat, &TRANSFORMS[0], false));
            let tr = hq4x_quadrant(&build_corner_view(&nb, pat, &TRANSFORMS[1], false));
            let bl = hq4x_quadrant(&build_corner_view(&nb, pat, &TRANSFORMS[2], false));
            let br = hq4x_quadrant(&build_corner_view(&nb, pat, &TRANSFORMS[3], false));
            dst[dy*dw+dx]=tl[0]; dst[dy*dw+dx+1]=tl[1]; dst[dy*dw+dx+2]=tr[1]; dst[dy*dw+dx+3]=tr[0];
            dst[(dy+1)*dw+dx]=tl[2]; dst[(dy+1)*dw+dx+1]=tl[3]; dst[(dy+1)*dw+dx+2]=tr[3]; dst[(dy+1)*dw+dx+3]=tr[2];
            dst[(dy+2)*dw+dx]=bl[2]; dst[(dy+2)*dw+dx+1]=bl[3]; dst[(dy+2)*dw+dx+2]=br[3]; dst[(dy+2)*dw+dx+3]=br[2];
            dst[(dy+3)*dw+dx]=bl[0]; dst[(dy+3)*dw+dx+1]=bl[1]; dst[(dy+3)*dw+dx+2]=br[1]; dst[(dy+3)*dw+dx+3]=br[0];
        }
    }
    dst
}

pub fn scale(src: &[u32], src_w: usize, src_h: usize, mode: HqxScale) -> Vec<u32> {
    match mode {
        HqxScale::Hq2x => hq2x(src, src_w, src_h),
        HqxScale::Hq3x => hq3x(src, src_w, src_h),
        HqxScale::Hq4x => hq4x(src, src_w, src_h),
    }
}
