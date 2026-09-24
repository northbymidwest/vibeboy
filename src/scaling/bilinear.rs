//! Bilinear interpolation to arbitrary output dimensions.
//!
//! Each output pixel is computed by bilinearly interpolating the four
//! nearest source pixels.

use super::get;

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
                    0 => b = v,
                    8 => g = v,
                    _ => r = v,
                }
            }
            dst[oy * dst_w + ox] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
    }
    dst
}
