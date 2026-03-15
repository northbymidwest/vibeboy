//! xBRZ — pixel-art scaling by Zenju (2x–6x).
//!
//! Two-phase algorithm:
//!  1. preProcessCorners: for every 2x2 source block (F,G,J,K), compute
//!     diagonal gradient sums and assign blend types to the four corners
//!     of the four participating pixels.
//!  2. scalePixel: for each source pixel, read the accumulated corner blend
//!     info and apply anti-aliased blending patterns to the output block.

use super::get;
use super::blend_argb;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XbrzScale { Xbrz2x, Xbrz3x, Xbrz4x, Xbrz5x, Xbrz6x }

impl XbrzScale {
    pub fn factor(self) -> u32 {
        match self {
            XbrzScale::Xbrz2x => 2, XbrzScale::Xbrz3x => 3,
            XbrzScale::Xbrz4x => 4, XbrzScale::Xbrz5x => 5,
            XbrzScale::Xbrz6x => 6,
        }
    }
}

const DOMINANT_RATIO: f32 = 3.6;
const LINE_DETECT_RATIO: f32 = 2.2;
const EQ_TOLERANCE: f32 = 30.0;

// ── Color distance (BT.2020 YCbCr) ─────────────────────────────────────────

/// YCbCr color distance using ITU-R BT.2020 conversion.
#[inline(always)]
fn dist(a: u32, b: u32) -> f32 {
    if a == b { return 0.0; }
    let dr = ((a >> 16) & 0xFF) as f32 - ((b >> 16) & 0xFF) as f32;
    let dg = ((a >> 8) & 0xFF) as f32 - ((b >> 8) & 0xFF) as f32;
    let db = (a & 0xFF) as f32 - (b & 0xFF) as f32;
    const K_R: f32 = 0.2627;
    const K_G: f32 = 0.6780;
    const K_B: f32 = 0.0593;
    const S_B: f32 = 0.5 / (1.0 - K_B);
    const S_R: f32 = 0.5 / (1.0 - K_R);
    let y = K_R * dr + K_G * dg + K_B * db;
    let cb = S_B * (db - y);
    let cr = S_R * (dr - y);
    // luminanceWeight = 1.0
    (y * y + cb * cb + cr * cr).sqrt()
}

#[inline(always)]
fn eq(a: u32, b: u32) -> bool { dist(a, b) < EQ_TOLERANCE }

// ── Blend info packed into a u8: 2 bits per corner ─────────────────────────
// Matches C++ layout: TL=[1:0], TR=[3:2], BR=[5:4], BL=[7:6]
const BL_NONE: u8 = 0;
const BL_NORMAL: u8 = 1;
const BL_DOMINANT: u8 = 2;

#[inline(always)] fn get_tl(b: u8) -> u8 { b & 3 }
#[inline(always)] fn get_tr(b: u8) -> u8 { (b >> 2) & 3 }
#[inline(always)] fn get_br(b: u8) -> u8 { (b >> 4) & 3 }
#[inline(always)] fn get_bl(b: u8) -> u8 { (b >> 6) & 3 }
#[inline(always)] fn set_tl(b: &mut u8, v: u8) { *b |= v & 3; }
#[inline(always)] fn set_tr(b: &mut u8, v: u8) { *b |= (v & 3) << 2; }
#[inline(always)] fn set_br(b: &mut u8, v: u8) { *b |= (v & 3) << 4; }
#[inline(always)] fn set_bl(b: &mut u8, v: u8) { *b |= (v & 3) << 6; }

// ── Phase 1: preProcessCorners ─────────────────────────────────────────────

fn pre_process_corners(src: &[u32], w: usize, h: usize, blend: &mut [u8]) {
    let iw = w as isize;
    let ih = h as isize;
    for y in 0..h as isize {
        for x in 0..w as isize {
            let f = get(src, w, h, x, y);
            let g = get(src, w, h, x+1, y);
            let j = get(src, w, h, x, y+1);
            let k = get(src, w, h, x+1, y+1);

            // Skip uniform blocks (exact pixel comparison)
            if (f == g && j == k) || (f == j && g == k) { continue; }

            let b = get(src, w, h, x, y-1);
            let c = get(src, w, h, x+1, y-1);
            let e = get(src, w, h, x-1, y);
            let hp = get(src, w, h, x+2, y);
            let i = get(src, w, h, x-1, y+1);
            let l = get(src, w, h, x+2, y+1);
            let n = get(src, w, h, x, y+2);
            let o = get(src, w, h, x+1, y+2);

            let jg = dist(i, f) + dist(f, c) + dist(n, k) + dist(k, hp) + 4.0 * dist(j, g);
            let fk = dist(e, j) + dist(j, o) + dist(b, g) + dist(g, l) + 4.0 * dist(f, k);

            if jg < fk {
                let bt = if DOMINANT_RATIO * jg < fk { BL_DOMINANT } else { BL_NORMAL };
                // Exact pixel comparison (not threshold-based eq)
                if f != g && f != j {
                    set_br(&mut blend[y as usize * w + x as usize], bt);
                }
                if k != j && k != g {
                    let kx = (x + 1).min(iw - 1) as usize;
                    let ky = (y + 1).min(ih - 1) as usize;
                    set_tl(&mut blend[ky * w + kx], bt);
                }
            } else if fk < jg {
                let bt = if DOMINANT_RATIO * fk < jg { BL_DOMINANT } else { BL_NORMAL };
                if j != f && j != k {
                    let jy = (y + 1).min(ih - 1) as usize;
                    set_tr(&mut blend[jy * w + x as usize], bt);
                }
                if g != f && g != k {
                    let gx = (x + 1).min(iw - 1) as usize;
                    set_bl(&mut blend[y as usize * w + gx], bt);
                }
            }
        }
    }
}

// ── Phase 2: scalePixel ────────────────────────────────────────────────────

#[inline(always)]
fn rot_idx(row: usize, col: usize, n: usize, rot: u8) -> usize {
    match rot {
        0 => row * n + col,
        1 => row * n + (n - 1 - col),
        2 => (n - 1 - row) * n + (n - 1 - col),
        3 => (n - 1 - row) * n + col,
        _ => unreachable!(),
    }
}

#[inline(always)]
fn bset(block: &mut [u32], n: usize, rot: u8, r: usize, c: usize, color: u32, alpha: f32) {
    let idx = rot_idx(r, c, n, rot);
    if alpha >= 1.0 {
        block[idx] = color;
    } else {
        block[idx] = blend_argb(block[idx], color, alpha);
    }
}

/// Rotate blend info so that the current corner maps to BottomRight position.
fn rotate_blend(bi: u8, rot: u8) -> u8 {
    // Rotate the 4 corner fields clockwise by `rot` positions
    match rot {
        0 => bi,
        1 => (get_tl(bi) << 6) | (get_bl(bi) << 4) | (get_tl(bi)) | (get_tr(bi) << 2),
        _ => {
            // General rotation: shift the 8-bit value
            let r = (rot * 2) & 7;
            (bi >> r) | (bi << (8 - r))
        }
    }
}

/// Check if line blending should be used for the bottom-right corner.
fn do_line_blend(bi: u8, e: u32, g: u32, h: u32, i: u32, f: u32, c: u32) -> bool {
    if get_br(bi) >= BL_DOMINANT { return true; }
    if get_tr(bi) != BL_NONE && !eq(e, g) { return false; }
    if get_bl(bi) != BL_NONE && !eq(e, c) { return false; }
    if !eq(e, i) && eq(g, h) && eq(h, i) && eq(i, f) && eq(f, c) { return false; }
    true
}

fn scale_pixel(
    src: &[u32], w: usize, h: usize,
    blend_info: &[u8], px: usize, py: usize,
    n: usize, block: &mut [u32],
) {
    let e = src[py * w + px];
    let bi = blend_info[py * w + px];
    for p in block.iter_mut() { *p = e; }
    if bi == 0 { return; }

    let ix = px as isize;
    let iy = py as isize;

    for rot in 0..4u8 {
        // Rotate blend info so current corner is at BottomRight
        let rbi = rotate_blend(bi, rot);
        if get_br(rbi) == BL_NONE { continue; }

        let (dx, dy) = match rot {
            0 => (1_isize, 1_isize),
            1 => (-1, 1),
            2 => (-1, -1),
            3 => (1, -1),
            _ => unreachable!(),
        };

        // 3x3 rotated kernel: a b c / d e f / g h i
        let f_px = get(src, w, h, ix + dx, iy);
        let h_px = get(src, w, h, ix, iy + dy);
        let c_px = get(src, w, h, ix + dx, iy - dy);
        let g_px = get(src, w, h, ix - dx, iy + dy);
        let i_px = get(src, w, h, ix + dx, iy + dy);

        // Blend target: more similar axis neighbor
        let target = if dist(e, f_px) <= dist(e, h_px) { f_px } else { h_px };

        let line = do_line_blend(rbi, e, g_px, h_px, i_px, f_px, c_px);

        if !line {
            blend_corner(block, n, rot, target);
        } else {
            let d_px = get(src, w, h, ix - dx, iy);
            let b_px = get(src, w, h, ix, iy - dy);
            let fg = dist(f_px, g_px);
            let hc = dist(h_px, c_px);
            let shallow = LINE_DETECT_RATIO * fg <= hc && !eq(e, g_px) && !eq(d_px, g_px);
            let steep   = LINE_DETECT_RATIO * hc <= fg && !eq(e, c_px) && !eq(b_px, c_px);

            match (steep, shallow) {
                (true, true)   => blend_steep_and_shallow(block, n, rot, target),
                (false, true)  => blend_shallow(block, n, rot, target),
                (true, false)  => blend_steep(block, n, rot, target),
                (false, false) => blend_diagonal(block, n, rot, target),
            }
        }
    }
}

// ── Blending patterns ──────────────────────────────────────────────────────
// All positions are (row, col) in the NxN output block, oriented so the
// blended corner is at (n-1, n-1) (bottom-right). Rotation handles the
// other three corners.

fn blend_corner(block: &mut [u32], n: usize, rot: u8, t: u32) {
    let m = n - 1;
    match n {
        2 => { bset(block, 2, rot, 1, 1, t, 0.21); }
        3 => { bset(block, 3, rot, 2, 2, t, 0.45); }
        4 => {
            bset(block, 4, rot, 3, 3, t, 0.68);
            bset(block, 4, rot, 3, 2, t, 0.09);
            bset(block, 4, rot, 2, 3, t, 0.09);
        }
        5 => {
            bset(block, 5, rot, 4, 4, t, 0.86);
            bset(block, 5, rot, 4, 3, t, 0.23);
            bset(block, 5, rot, 3, 4, t, 0.23);
        }
        _ => {
            bset(block, n, rot, m, m, t, 0.97);
            bset(block, n, rot, m-1, m, t, 0.42);
            bset(block, n, rot, m, m-1, t, 0.42);
            bset(block, n, rot, m, m-2, t, 0.06);
            bset(block, n, rot, m-2, m, t, 0.06);
        }
    }
}

fn blend_shallow(block: &mut [u32], n: usize, rot: u8, t: u32) {
    let m = n - 1;
    match n {
        2 => {
            bset(block, 2, rot, 1, 0, t, 0.25);
            bset(block, 2, rot, 1, 1, t, 0.75);
        }
        3 => {
            bset(block, 3, rot, 2, 0, t, 0.25);
            bset(block, 3, rot, 1, 2, t, 0.25);
            bset(block, 3, rot, 2, 1, t, 0.75);
            bset(block, 3, rot, 2, 2, t, 1.0);
        }
        4 => {
            bset(block, 4, rot, 3, 0, t, 0.25);
            bset(block, 4, rot, 2, 2, t, 0.25);
            bset(block, 4, rot, 3, 1, t, 0.75);
            bset(block, 4, rot, 2, 3, t, 0.75);
            bset(block, 4, rot, 3, 2, t, 1.0);
            bset(block, 4, rot, 3, 3, t, 1.0);
        }
        5 => {
            bset(block, 5, rot, 4, 0, t, 0.25);
            bset(block, 5, rot, 3, 2, t, 0.25);
            bset(block, 5, rot, 2, 4, t, 0.25);
            bset(block, 5, rot, 4, 1, t, 0.75);
            bset(block, 5, rot, 3, 3, t, 0.75);
            bset(block, 5, rot, 4, 2, t, 1.0);
            bset(block, 5, rot, 4, 3, t, 1.0);
            bset(block, 5, rot, 4, 4, t, 1.0);
            bset(block, 5, rot, 3, 4, t, 1.0);
        }
        _ => { // 6x
            bset(block, n, rot, m, 0, t, 0.25);
            bset(block, n, rot, m-1, 2, t, 0.25);
            bset(block, n, rot, m-2, 4, t, 0.25);
            bset(block, n, rot, m, 1, t, 0.75);
            bset(block, n, rot, m-1, 3, t, 0.75);
            bset(block, n, rot, m-2, m, t, 0.75);
            bset(block, n, rot, m, 2, t, 1.0);
            bset(block, n, rot, m, 3, t, 1.0);
            bset(block, n, rot, m, 4, t, 1.0);
            bset(block, n, rot, m, m, t, 1.0);
            bset(block, n, rot, m-1, m-1, t, 1.0);
            bset(block, n, rot, m-1, m, t, 1.0);
        }
    }
}

fn blend_steep(block: &mut [u32], n: usize, rot: u8, t: u32) {
    let m = n - 1;
    match n {
        2 => {
            bset(block, 2, rot, 0, 1, t, 0.25);
            bset(block, 2, rot, 1, 1, t, 0.75);
        }
        3 => {
            bset(block, 3, rot, 0, 2, t, 0.25);
            bset(block, 3, rot, 2, 1, t, 0.25);
            bset(block, 3, rot, 1, 2, t, 0.75);
            bset(block, 3, rot, 2, 2, t, 1.0);
        }
        4 => {
            bset(block, 4, rot, 0, 3, t, 0.25);
            bset(block, 4, rot, 2, 2, t, 0.25);
            bset(block, 4, rot, 1, 3, t, 0.75);
            bset(block, 4, rot, 3, 2, t, 0.75);
            bset(block, 4, rot, 2, 3, t, 1.0);
            bset(block, 4, rot, 3, 3, t, 1.0);
        }
        5 => {
            bset(block, 5, rot, 0, 4, t, 0.25);
            bset(block, 5, rot, 2, 3, t, 0.25);
            bset(block, 5, rot, 4, 1, t, 0.25);
            bset(block, 5, rot, 1, 4, t, 0.75);
            bset(block, 5, rot, 3, 3, t, 0.75);
            bset(block, 5, rot, 2, 4, t, 1.0);
            bset(block, 5, rot, 3, 4, t, 1.0);
            bset(block, 5, rot, 4, 4, t, 1.0);
            bset(block, 5, rot, 4, 3, t, 1.0);
        }
        _ => { // 6x
            bset(block, n, rot, 0, m, t, 0.25);
            bset(block, n, rot, 2, m-1, t, 0.25);
            bset(block, n, rot, 4, m-2, t, 0.25);
            bset(block, n, rot, 1, m, t, 0.75);
            bset(block, n, rot, 3, m-1, t, 0.75);
            bset(block, n, rot, m, m-2, t, 0.75);
            bset(block, n, rot, 2, m, t, 1.0);
            bset(block, n, rot, 3, m, t, 1.0);
            bset(block, n, rot, 4, m, t, 1.0);
            bset(block, n, rot, m, m, t, 1.0);
            bset(block, n, rot, m-1, m-1, t, 1.0);
            bset(block, n, rot, m, m-1, t, 1.0);
        }
    }
}

fn blend_steep_and_shallow(block: &mut [u32], n: usize, rot: u8, t: u32) {
    let m = n - 1;
    match n {
        2 => {
            bset(block, 2, rot, 1, 0, t, 0.25);
            bset(block, 2, rot, 0, 1, t, 0.25);
            bset(block, 2, rot, 1, 1, t, 5.0/6.0);
        }
        3 => {
            bset(block, 3, rot, 2, 0, t, 0.25);
            bset(block, 3, rot, 0, 2, t, 0.25);
            bset(block, 3, rot, 2, 1, t, 0.75);
            bset(block, 3, rot, 1, 2, t, 0.75);
            bset(block, 3, rot, 2, 2, t, 1.0);
        }
        4 => {
            bset(block, 4, rot, 3, 1, t, 0.75);
            bset(block, 4, rot, 1, 3, t, 0.75);
            bset(block, 4, rot, 3, 0, t, 0.25);
            bset(block, 4, rot, 0, 3, t, 0.25);
            bset(block, 4, rot, 2, 2, t, 1.0/3.0);
            bset(block, 4, rot, 3, 2, t, 1.0);
            bset(block, 4, rot, 2, 3, t, 1.0);
            bset(block, 4, rot, 3, 3, t, 1.0);
        }
        5 => {
            bset(block, 5, rot, 0, 4, t, 0.25);
            bset(block, 5, rot, 2, 3, t, 0.25);
            bset(block, 5, rot, 1, 4, t, 0.75);
            bset(block, 5, rot, 4, 0, t, 0.25);
            bset(block, 5, rot, 3, 2, t, 0.25);
            bset(block, 5, rot, 4, 1, t, 0.75);
            bset(block, 5, rot, 3, 3, t, 2.0/3.0);
            bset(block, 5, rot, 2, 4, t, 1.0);
            bset(block, 5, rot, 3, 4, t, 1.0);
            bset(block, 5, rot, 4, 4, t, 1.0);
            bset(block, 5, rot, 4, 2, t, 1.0);
            bset(block, 5, rot, 4, 3, t, 1.0);
        }
        _ => { // 6x
            bset(block, n, rot, 0, m, t, 0.25);
            bset(block, n, rot, 2, m-1, t, 0.25);
            bset(block, n, rot, 1, m, t, 0.75);
            bset(block, n, rot, 3, m-1, t, 0.75);
            bset(block, n, rot, m, 0, t, 0.25);
            bset(block, n, rot, m-1, 2, t, 0.25);
            bset(block, n, rot, m, 1, t, 0.75);
            bset(block, n, rot, m-1, 3, t, 0.75);
            bset(block, n, rot, 2, m, t, 1.0);
            bset(block, n, rot, 3, m, t, 1.0);
            bset(block, n, rot, 4, m, t, 1.0);
            bset(block, n, rot, m, m, t, 1.0);
            bset(block, n, rot, m-1, m-1, t, 1.0);
            bset(block, n, rot, m, m-1, t, 1.0);
            bset(block, n, rot, m, 2, t, 1.0);
            bset(block, n, rot, m, 3, t, 1.0);
        }
    }
}

fn blend_diagonal(block: &mut [u32], n: usize, rot: u8, t: u32) {
    let m = n - 1;
    match n {
        2 => {
            bset(block, 2, rot, 1, 1, t, 0.5);
        }
        3 => {
            bset(block, 3, rot, 1, 2, t, 1.0/8.0);
            bset(block, 3, rot, 2, 1, t, 1.0/8.0);
            bset(block, 3, rot, 2, 2, t, 7.0/8.0);
        }
        4 => {
            bset(block, 4, rot, m, m/2, t, 0.5);
            bset(block, 4, rot, m-1, m/2+1, t, 0.5);
            bset(block, 4, rot, m, m, t, 1.0);
        }
        5 => {
            bset(block, 5, rot, m, m/2, t, 1.0/8.0);
            bset(block, 5, rot, m-1, m/2+1, t, 1.0/8.0);
            bset(block, 5, rot, m-2, m/2+2, t, 1.0/8.0);
            bset(block, 5, rot, 4, 3, t, 7.0/8.0);
            bset(block, 5, rot, 3, 4, t, 7.0/8.0);
            bset(block, 5, rot, 4, 4, t, 1.0);
        }
        _ => { // 6x
            bset(block, n, rot, m, m/2, t, 0.5);
            bset(block, n, rot, m-1, m/2+1, t, 0.5);
            bset(block, n, rot, m-2, m/2+2, t, 0.5);
            bset(block, n, rot, m-1, m, t, 1.0);
            bset(block, n, rot, m, m, t, 1.0);
            bset(block, n, rot, m, m-1, t, 1.0);
        }
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

pub fn scale(src: &[u32], src_w: usize, src_h: usize, mode: XbrzScale) -> Vec<u32> {
    let n = mode.factor() as usize;
    let dst_w = src_w * n;
    let dst_h = src_h * n;
    let mut dst = vec![0u32; dst_w * dst_h];

    // Phase 1: preprocess all corners
    let mut blend_info = vec![0u8; src_w * src_h];
    pre_process_corners(src, src_w, src_h, &mut blend_info);

    // Phase 2: scale each pixel
    let mut block = vec![0u32; n * n];
    for y in 0..src_h {
        for x in 0..src_w {
            scale_pixel(src, src_w, src_h, &blend_info, x, y, n, &mut block);
            let dx = x * n;
            let dy = y * n;
            for r in 0..n {
                for c in 0..n {
                    dst[(dy + r) * dst_w + dx + c] = block[r * n + c];
                }
            }
        }
    }
    dst
}
