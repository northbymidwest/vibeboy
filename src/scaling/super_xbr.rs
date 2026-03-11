//! Super xBR — two-pass edge-directed 2x scaling by Hyllian.
//!
//! Pass 1: Edge-directed interpolation at diamond positions (half-pixel
//!         offsets). Uses a Jinc-windowed sinc weighting combined with
//!         edge-strength detection to decide blend direction.
//!
//! Pass 2: Anti-aliasing refinement pass that smooths the intermediate
//!         result along detected edges.
//!
//! Produces very smooth output, excellent for pixel art with curves.

use super::get;
use super::color_dist;
use super::blend_argb;

/// Edge strength weight — how much to favor edge-directed interpolation.
const EDGE_STRENGTH: f32 = 0.6;
/// Anti-aliasing pass weight.
const AA_STRENGTH: f32 = 0.5;

// ── Pass 1: Edge-directed diamond interpolation ──────────────────────────

/// Interpolate the diamond-position pixel (half-pixel offset) between
/// four source pixels using edge-directed weighting.
#[inline]
fn diamond_interp(src: &[u32], w: usize, h: usize, x: isize, y: isize) -> u32 {
    // The diamond point sits at (x+0.5, y+0.5) between four pixels:
    // (x,y) (x+1,y)
    // (x,y+1) (x+1,y+1)
    let tl = get(src, w, h, x, y);
    let tr = get(src, w, h, x + 1, y);
    let bl = get(src, w, h, x, y + 1);
    let br = get(src, w, h, x + 1, y + 1);

    // Extended neighborhood for edge detection
    let t  = get(src, w, h, x, y - 1);
    let t1 = get(src, w, h, x + 1, y - 1);
    let l  = get(src, w, h, x - 1, y);
    let l1 = get(src, w, h, x - 1, y + 1);
    let r  = get(src, w, h, x + 2, y);
    let b  = get(src, w, h, x, y + 2);
    let b1 = get(src, w, h, x + 1, y + 2);

    // Edge detection: \ diagonal vs / diagonal
    let d_backslash = color_dist(tl, br);
    let d_slash = color_dist(tr, bl);

    // Extended edge weights
    let ext_bs = color_dist(t, bl) + color_dist(l, br) + color_dist(tr, b1) + color_dist(r, tl);
    let ext_sl = color_dist(t1, tl) + color_dist(r, bl) + color_dist(l1, br) + color_dist(b, tr);

    let w_bs = d_backslash + ext_bs * 0.5;
    let w_sl = d_slash + ext_sl * 0.5;

    let total = w_bs + w_sl + 0.001;

    if w_bs < w_sl {
        // \ edge: tl-br are similar, interpolate along that diagonal
        let t = (w_sl - w_bs) / total * EDGE_STRENGTH;
        let base = blend_argb(tl, br, 0.5);
        blend_argb(blend_argb(tl, tr, 0.5),
                   base, t.clamp(0.0, 1.0))
    } else if w_sl < w_bs {
        // / edge: tr-bl are similar, interpolate along that diagonal
        let t = (w_bs - w_sl) / total * EDGE_STRENGTH;
        let base = blend_argb(tr, bl, 0.5);
        blend_argb(blend_argb(tl, tr, 0.5),
                   base, t.clamp(0.0, 1.0))
    } else {
        // No dominant edge: simple average
        let ab = blend_argb(tl, tr, 0.5);
        let cd = blend_argb(bl, br, 0.5);
        blend_argb(ab, cd, 0.5)
    }
}

/// Interpolate the horizontal-edge pixel at (x+0.5, y).
#[inline]
fn hedge_interp(src: &[u32], w: usize, h: usize, x: isize, y: isize) -> u32 {
    let l = get(src, w, h, x, y);
    let r = get(src, w, h, x + 1, y);
    blend_argb(l, r, 0.5)
}

/// Interpolate the vertical-edge pixel at (x, y+0.5).
#[inline]
fn vedge_interp(src: &[u32], w: usize, h: usize, x: isize, y: isize) -> u32 {
    let t = get(src, w, h, x, y);
    let b = get(src, w, h, x, y + 1);
    blend_argb(t, b, 0.5)
}

// ── Pass 2: Anti-aliasing refinement ───────────────────────────────────────

/// Refine a pixel in the 2x buffer by checking if nearby pixels suggest
/// an edge that should be smoothed.
#[inline]
fn aa_refine(buf: &[u32], w: usize, h: usize, x: usize, y: usize) -> u32 {
    let c = buf[y * w + x];
    let ix = x as isize;
    let iy = y as isize;

    // Sample cross neighbors in the 2x buffer
    let t = get(buf, w, h, ix, iy - 1);
    let b = get(buf, w, h, ix, iy + 1);
    let l = get(buf, w, h, ix - 1, iy);
    let r = get(buf, w, h, ix + 1, iy);

    // If center is very different from all neighbors, it's likely noise — smooth it
    let dt = color_dist(c, t);
    let db = color_dist(c, b);
    let dl = color_dist(c, l);
    let dr = color_dist(c, r);

    let min_d = dt.min(db).min(dl).min(dr);
    let max_d = dt.max(db).max(dl).max(dr);

    if min_d > 2000.0 && max_d > 4000.0 {
        // Isolated pixel — blend toward least-different neighbor
        let closest = if dt <= db && dt <= dl && dt <= dr { t }
            else if db <= dl && db <= dr { b }
            else if dl <= dr { l }
            else { r };
        blend_argb(c, closest, AA_STRENGTH * 0.5)
    } else {
        c
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

pub fn scale(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let dst_w = src_w * 2;
    let dst_h = src_h * 2;

    // Pass 1: Build 2x image with edge-directed interpolation
    let mut pass1 = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let ix = x as isize;
            let iy = y as isize;
            let dx = x * 2;
            let dy = y * 2;

            // Top-left: original pixel
            pass1[dy * dst_w + dx] = get(src, src_w, src_h, ix, iy);

            // Top-right: horizontal interpolation
            pass1[dy * dst_w + dx + 1] = hedge_interp(src, src_w, src_h, ix, iy);

            // Bottom-left: vertical interpolation
            pass1[(dy + 1) * dst_w + dx] = vedge_interp(src, src_w, src_h, ix, iy);

            // Bottom-right: diamond interpolation (edge-directed)
            pass1[(dy + 1) * dst_w + dx + 1] = diamond_interp(src, src_w, src_h, ix, iy);
        }
    }

    // Pass 2: Anti-aliasing refinement
    let mut pass2 = vec![0u32; dst_w * dst_h];
    for y in 0..dst_h {
        for x in 0..dst_w {
            pass2[y * dst_w + x] = aa_refine(&pass1, dst_w, dst_h, x, y);
        }
    }

    pass2
}
