//! xBR-Hybrid — combines xBR edge detection with bilinear smoothing.
//!
//! For each pixel, measures local color variance. In smooth-gradient regions
//! (low variance), uses bilinear interpolation for natural blending. In
//! high-contrast edges, uses standard xBR rules to preserve pixel-art detail.
//! The two are blended based on the variance ratio.

use super::get;
use super::color_dist;
use super::blend_argb;

/// Variance threshold below which bilinear dominates.
const LOW_VARIANCE: f32 = 500.0;
/// Variance threshold above which xBR dominates.
const HIGH_VARIANCE: f32 = 3000.0;

/// Compute local color variance in a 3x3 neighborhood.
#[inline]
fn local_variance(src: &[u32], w: usize, h: usize, x: isize, y: isize) -> f32 {
    let c = get(src, w, h, x, y);
    let mut total = 0.0_f32;
    let mut count = 0;
    for dy in -1..=1_isize {
        for dx in -1..=1_isize {
            if dx == 0 && dy == 0 { continue; }
            total += color_dist(c, get(src, w, h, x + dx, y + dy));
            count += 1;
        }
    }
    total / count as f32
}

/// Bilinear 2x upscale for a single pixel — returns [TL, TR, BL, BR].
#[inline]
fn bilinear_block(src: &[u32], w: usize, h: usize, x: isize, y: isize) -> [u32; 4] {
    let p = get(src, w, h, x, y);
    let r = get(src, w, h, x + 1, y);
    let d = get(src, w, h, x, y + 1);
    let dr = get(src, w, h, x + 1, y + 1);

    [
        p,
        blend_argb(p, r, 0.5),
        blend_argb(p, d, 0.5),
        blend_argb(blend_argb(p, r, 0.5), blend_argb(d, dr, 0.5), 0.5),
    ]
}

/// xBR 2x for a single pixel — returns [TL, TR, BL, BR].
/// Simplified: uses the 3x3 neighborhood for edge detection.
#[inline]
fn xbr_block(src: &[u32], w: usize, h: usize, x: isize, y: isize) -> [u32; 4] {
    let e = get(src, w, h, x, y);
    let b = get(src, w, h, x, y - 1);
    let d = get(src, w, h, x - 1, y);
    let f = get(src, w, h, x + 1, y);
    let h_px = get(src, w, h, x, y + 1);

    let a = get(src, w, h, x - 1, y - 1);
    let c = get(src, w, h, x + 1, y - 1);
    let g = get(src, w, h, x - 1, y + 1);
    let i = get(src, w, h, x + 1, y + 1);

    let mut out = [e; 4];

    // Bottom-right
    {
        let d_diag = color_dist(d, h_px) + color_dist(b, f) +
                     color_dist(e, i) * 2.0;
        let d_ortho = color_dist(e, f) + color_dist(e, h_px) +
                      color_dist(d, i) + color_dist(b, i);
        if d_diag < d_ortho && !colors_similar(e, i) {
            let t = if colors_similar(f, h_px) { 0.5 } else { 0.25 };
            let blend_c = if color_dist(e, f) < color_dist(e, h_px) { f } else { h_px };
            out[3] = blend_argb(e, blend_c, t);
        }
    }

    // Bottom-left
    {
        let d_diag = color_dist(f, h_px) + color_dist(b, d) +
                     color_dist(e, g) * 2.0;
        let d_ortho = color_dist(e, d) + color_dist(e, h_px) +
                      color_dist(f, g) + color_dist(b, g);
        if d_diag < d_ortho && !colors_similar(e, g) {
            let t = if colors_similar(d, h_px) { 0.5 } else { 0.25 };
            let blend_c = if color_dist(e, d) < color_dist(e, h_px) { d } else { h_px };
            out[2] = blend_argb(e, blend_c, t);
        }
    }

    // Top-right
    {
        let d_diag = color_dist(d, b) + color_dist(h_px, f) +
                     color_dist(e, c) * 2.0;
        let d_ortho = color_dist(e, f) + color_dist(e, b) +
                      color_dist(d, c) + color_dist(h_px, c);
        if d_diag < d_ortho && !colors_similar(e, c) {
            let t = if colors_similar(b, f) { 0.5 } else { 0.25 };
            let blend_c = if color_dist(e, f) < color_dist(e, b) { f } else { b };
            out[1] = blend_argb(e, blend_c, t);
        }
    }

    // Top-left
    {
        let d_diag = color_dist(f, b) + color_dist(h_px, d) +
                     color_dist(e, a) * 2.0;
        let d_ortho = color_dist(e, d) + color_dist(e, b) +
                      color_dist(f, a) + color_dist(h_px, a);
        if d_diag < d_ortho && !colors_similar(e, a) {
            let t = if colors_similar(d, b) { 0.5 } else { 0.25 };
            let blend_c = if color_dist(e, d) < color_dist(e, b) { d } else { b };
            out[0] = blend_argb(e, blend_c, t);
        }
    }

    out
}

#[inline(always)]
fn colors_similar(a: u32, b: u32) -> bool {
    color_dist(a, b) < 100.0
}

pub fn scale(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let dst_w = src_w * 2;
    let dst_h = src_h * 2;
    let mut dst = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let ix = x as isize;
            let iy = y as isize;

            let variance = local_variance(src, src_w, src_h, ix, iy);

            let block = if variance < LOW_VARIANCE {
                bilinear_block(src, src_w, src_h, ix, iy)
            } else if variance > HIGH_VARIANCE {
                xbr_block(src, src_w, src_h, ix, iy)
            } else {
                // Blend between bilinear and xBR
                let t = (variance - LOW_VARIANCE) / (HIGH_VARIANCE - LOW_VARIANCE);
                let bil = bilinear_block(src, src_w, src_h, ix, iy);
                let xbr = xbr_block(src, src_w, src_h, ix, iy);
                [
                    blend_argb(bil[0], xbr[0], t),
                    blend_argb(bil[1], xbr[1], t),
                    blend_argb(bil[2], xbr[2], t),
                    blend_argb(bil[3], xbr[3], t),
                ]
            };

            let dx = x * 2;
            let dy = y * 2;
            dst[dy * dst_w + dx] = block[0];
            dst[dy * dst_w + dx + 1] = block[1];
            dst[(dy + 1) * dst_w + dx] = block[2];
            dst[(dy + 1) * dst_w + dx + 1] = block[3];
        }
    }
    dst
}
