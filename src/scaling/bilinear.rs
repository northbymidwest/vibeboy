//! Bilinear interpolation — 2x smooth scaling.
//!
//! Each source pixel is mapped to a 2x2 output block. The four output
//! sub-pixels are computed by bilinearly interpolating the source pixel
//! with its right, bottom, and bottom-right neighbors.

use super::get;

/// Blend two ARGB colors by 50/50.
#[inline(always)]
fn avg2(a: u32, b: u32) -> u32 {
    let mask = 0xFEFEFEFE_u32;
    ((a & mask) >> 1) + ((b & mask) >> 1) + (a & b & 0x01010101)
}

/// Blend four ARGB colors equally (average).
#[inline(always)]
fn avg4(a: u32, b: u32, c: u32, d: u32) -> u32 {
    avg2(avg2(a, b), avg2(c, d))
}

/// Scale to arbitrary output dimensions using bilinear interpolation.
pub fn scale_to(src: &[u32], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<u32> {
    let mut dst = vec![0u32; dst_w * dst_h];
    let sx = src_w as f64 / dst_w as f64;
    let sy = src_h as f64 / dst_h as f64;

    for oy in 0..dst_h {
        // Half-pixel offset centers interpolation on pixel boundaries
        let src_y = ((oy as f64 + 0.5) * sy - 0.5).max(0.0);
        let iy = src_y as isize;
        let fy = src_y - iy as f64;

        for ox in 0..dst_w {
            let src_x = ((ox as f64 + 0.5) * sx - 0.5).max(0.0);
            let ix = src_x as isize;
            let fx = src_x - ix as f64;

            let p00 = get(src, src_w, src_h, ix, iy);
            let p10 = get(src, src_w, src_h, ix + 1, iy);
            let p01 = get(src, src_w, src_h, ix, iy + 1);
            let p11 = get(src, src_w, src_h, ix + 1, iy + 1);

            let mut r = 0u32;
            let mut g = 0u32;
            let mut b = 0u32;
            for shift in [0u32, 8, 16] {
                let c00 = ((p00 >> shift) & 0xFF) as f64;
                let c10 = ((p10 >> shift) & 0xFF) as f64;
                let c01 = ((p01 >> shift) & 0xFF) as f64;
                let c11 = ((p11 >> shift) & 0xFF) as f64;
                let c = c00 * (1.0 - fx) * (1.0 - fy)
                      + c10 * fx * (1.0 - fy)
                      + c01 * (1.0 - fx) * fy
                      + c11 * fx * fy;
                let v = c.round().clamp(0.0, 255.0) as u32;
                match shift {
                    0  => b = v,
                    8  => g = v,
                    _  => r = v,
                }
            }
            dst[oy * dst_w + ox] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
    }
    dst
}

/// Fixed 2x bilinear scale (fast path).
pub fn scale(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let dst_w = src_w * 2;
    let dst_h = src_h * 2;
    let mut dst = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let ix = x as isize;
            let iy = y as isize;
            let p = get(src, src_w, src_h, ix, iy);
            let r = get(src, src_w, src_h, ix + 1, iy);
            let d = get(src, src_w, src_h, ix, iy + 1);
            let dr = get(src, src_w, src_h, ix + 1, iy + 1);

            let dx = x * 2;
            let dy = y * 2;
            dst[dy * dst_w + dx] = p;
            dst[dy * dst_w + dx + 1] = avg2(p, r);
            dst[(dy + 1) * dst_w + dx] = avg2(p, d);
            dst[(dy + 1) * dst_w + dx + 1] = avg4(p, r, d, dr);
        }
    }
    dst
}
