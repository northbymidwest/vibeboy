//! Bicubic interpolation — 2x smooth scaling.
//!
//! Uses Catmull-Rom spline interpolation over a 4x4 neighborhood for
//! each output sub-pixel. Produces sharper results than bilinear while
//! still being smooth.

use super::get;

/// Catmull-Rom cubic weight function.
/// t is the fractional distance (0.0, 0.25, 0.5, 0.75).
#[inline(always)]
fn cubic_weight(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

/// Extract one channel (0=B, 8=G, 16=R) from an ARGB pixel.
#[inline(always)]
fn ch(c: u32, shift: u32) -> f32 {
    ((c >> shift) & 0xFF) as f32
}

/// Bicubic sample at fractional position (fx, fy) relative to integer grid.
/// (ix, iy) is the integer position of the top-left of the 2x2 center.
#[inline(always)]
fn bicubic_sample(src: &[u32], w: usize, h: usize, ix: isize, iy: isize, fx: f32, fy: f32) -> u32 {
    let wx = cubic_weight(fx);
    let wy = cubic_weight(fy);

    let mut r = 0.0_f32;
    let mut g = 0.0_f32;
    let mut b = 0.0_f32;

    for j in 0..4_isize {
        let row_w = wy[j as usize];
        for i in 0..4_isize {
            let w_val = wx[i as usize] * row_w;
            let pixel = get(src, w, h, ix + i - 1, iy + j - 1);
            r += ch(pixel, 16) * w_val;
            g += ch(pixel, 8) * w_val;
            b += ch(pixel, 0) * w_val;
        }
    }

    let rc = r.round().clamp(0.0, 255.0) as u32;
    let gc = g.round().clamp(0.0, 255.0) as u32;
    let bc = b.round().clamp(0.0, 255.0) as u32;
    0xFF000000 | (rc << 16) | (gc << 8) | bc
}

/// Scale to arbitrary output dimensions using bicubic (Catmull-Rom) interpolation.
pub fn scale_to(src: &[u32], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<u32> {
    let mut dst = vec![0u32; dst_w * dst_h];
    let sx = src_w as f64 / dst_w as f64;
    let sy = src_h as f64 / dst_h as f64;

    for oy in 0..dst_h {
        let src_y = oy as f64 * sy;
        let iy = src_y as isize;
        let fy = (src_y - iy as f64) as f32;

        for ox in 0..dst_w {
            let src_x = ox as f64 * sx;
            let ix = src_x as isize;
            let fx = (src_x - ix as f64) as f32;

            dst[oy * dst_w + ox] = bicubic_sample(src, src_w, src_h, ix, iy, fx, fy);
        }
    }
    dst
}

/// Fixed 2x bicubic scale (fast path).
pub fn scale(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let dst_w = src_w * 2;
    let dst_h = src_h * 2;
    let mut dst = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let ix = x as isize;
            let iy = y as isize;

            let dx = x * 2;
            let dy = y * 2;
            // Sub-pixel offsets: (0,0), (0.5,0), (0,0.5), (0.5,0.5)
            dst[dy * dst_w + dx] = bicubic_sample(src, src_w, src_h, ix, iy, 0.0, 0.0);
            dst[dy * dst_w + dx + 1] = bicubic_sample(src, src_w, src_h, ix, iy, 0.5, 0.0);
            dst[(dy + 1) * dst_w + dx] = bicubic_sample(src, src_w, src_h, ix, iy, 0.0, 0.5);
            dst[(dy + 1) * dst_w + dx + 1] = bicubic_sample(src, src_w, src_h, ix, iy, 0.5, 0.5);
        }
    }
    dst
}
