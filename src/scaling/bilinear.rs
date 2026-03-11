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
