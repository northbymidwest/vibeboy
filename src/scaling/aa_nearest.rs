//! Antialiased nearest-neighbor scaling.
//!
//! Standard nearest-neighbor picks the single source pixel closest to each
//! output sample.  This variant instead considers the *area* of the source
//! grid that each output pixel covers:
//!
//! - If an output pixel falls entirely within one source pixel, that source
//!   color is used verbatim (identical to nearest-neighbor).
//! - If an output pixel straddles a source-pixel boundary, the overlapping
//!   source pixels are blended proportionally to their area coverage.
//!
//! At integer scale factors every output pixel aligns perfectly with a single
//! source pixel, so the result is identical to plain nearest-neighbor.  At
//! non-integer scales the pixel boundaries are smoothly antialiased rather
//! than exhibiting the uneven "thick/thin" rows that regular nearest-neighbor
//! produces.

/// Scale `src` (dimensions `src_w` x `src_h`) to exactly `dst_w` x `dst_h`
/// using area-coverage antialiased nearest-neighbor.
///
/// Pixel format is ARGB8888 (0xAA_RR_GG_BB).
pub fn scale(
    src: &[u32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u32> {
    assert!(src.len() >= src_w * src_h);
    let mut dst = vec![0u32; dst_w * dst_h];

    // Size of one output pixel in source-pixel coordinates.
    let sx = src_w as f64 / dst_w as f64;
    let sy = src_h as f64 / dst_h as f64;

    for oy in 0..dst_h {
        // Source-space vertical span of this output pixel.
        let src_y0 = oy as f64 * sy;
        let src_y1 = src_y0 + sy;

        let iy0 = (src_y0 as usize).min(src_h - 1);
        let iy1 = ((src_y1.ceil() as usize).max(1) - 1).min(src_h - 1);

        for ox in 0..dst_w {
            // Source-space horizontal span of this output pixel.
            let src_x0 = ox as f64 * sx;
            let src_x1 = src_x0 + sx;

            let ix0 = (src_x0 as usize).min(src_w - 1);
            let ix1 = ((src_x1.ceil() as usize).max(1) - 1).min(src_w - 1);

            if ix0 == ix1 && iy0 == iy1 {
                // Output pixel falls entirely within one source pixel.
                dst[oy * dst_w + ox] = src[iy0 * src_w + ix0];
            } else {
                // Accumulate weighted RGBA across overlapping source pixels.
                let mut r_acc = 0.0_f64;
                let mut g_acc = 0.0_f64;
                let mut b_acc = 0.0_f64;
                let mut area_total = 0.0_f64;

                for iy in iy0..=iy1 {
                    // Vertical overlap of source row `iy` with the output pixel.
                    let top = (iy as f64).max(src_y0);
                    let bot = ((iy + 1) as f64).min(src_y1);
                    let h = bot - top;
                    if h <= 0.0 {
                        continue;
                    }

                    for ix in ix0..=ix1 {
                        // Horizontal overlap of source column `ix`.
                        let left = (ix as f64).max(src_x0);
                        let right = ((ix + 1) as f64).min(src_x1);
                        let w = right - left;
                        if w <= 0.0 {
                            continue;
                        }

                        let area = w * h;
                        let c = src[iy * src_w + ix];
                        r_acc += ((c >> 16) & 0xFF) as f64 * area;
                        g_acc += ((c >> 8) & 0xFF) as f64 * area;
                        b_acc += (c & 0xFF) as f64 * area;
                        area_total += area;
                    }
                }

                if area_total > 0.0 {
                    let inv = 1.0 / area_total;
                    let r = (r_acc * inv).round().min(255.0) as u32;
                    let g = (g_acc * inv).round().min(255.0) as u32;
                    let b = (b_acc * inv).round().min(255.0) as u32;
                    dst[oy * dst_w + ox] = 0xFF00_0000 | (r << 16) | (g << 8) | b;
                }
            }
        }
    }
    dst
}

/// Convenience wrapper: scale by an integer factor.
///
/// At integer factors the result is identical to standard nearest-neighbor,
/// but the function is provided for API consistency with the other filters.
pub fn scale_integer(src: &[u32], src_w: usize, src_h: usize, factor: usize) -> Vec<u32> {
    scale(src, src_w, src_h, src_w * factor, src_h * factor)
}
