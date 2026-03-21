//! LCD grid effect — simulates the subpixel structure of an LCD panel.
//!
//! Each source pixel is rendered as an NxN block (minimum 3x) with visible
//! RGB subpixel columns and inter-pixel gaps. The effect is resolution-
//! independent: the subpixel pattern tiles based on the scale factor.
//!
//! Layout per pixel at 4x scale:
//!   col 0: R subpixel (red channel amplified, others dimmed)
//!   col 1: G subpixel (green channel amplified, others dimmed)
//!   col 2: B subpixel (blue channel amplified, others dimmed)
//!   col 3: inter-pixel gap (all channels darkened)
//!   row 3: inter-pixel gap row (all channels darkened)
//!
//! At 3x scale, no gap column — just R, G, B columns with a dimmed
//! bottom row for the horizontal gap.

use super::get;

/// Subpixel bleed: how much of the other channels leak into each subpixel.
/// 0.0 = pure RGB separation, 0.15 = slight bleed for a softer look.
const BLEED: f32 = 0.12;

/// Brightness boost to compensate for the subpixel mask darkening the image.
const BOOST: f32 = 2.2;

/// Gap brightness (fraction of the pixel color visible in gap regions).
const GAP_BRIGHTNESS: f32 = 0.08;

/// Horizontal gap row brightness.
const HGAP_BRIGHTNESS: f32 = 0.12;

#[inline(always)]
fn apply_subpixel(r: f32, g: f32, b: f32, col_in_pixel: usize, cols_per_pixel: usize) -> (f32, f32, f32) {
    // Last column is the vertical gap (if we have room for it)
    if cols_per_pixel > 3 && col_in_pixel == cols_per_pixel - 1 {
        return (r * GAP_BRIGHTNESS, g * GAP_BRIGHTNESS, b * GAP_BRIGHTNESS);
    }

    // Map column position to subpixel phase (0=R, 1=G, 2=B)
    let phase = (col_in_pixel * 3) / cols_per_pixel.max(1);

    let (sr, sg, sb) = match phase {
        0 => (r * BOOST, g * BLEED * BOOST, b * BLEED * BOOST),
        1 => (r * BLEED * BOOST, g * BOOST, b * BLEED * BOOST),
        _ => (r * BLEED * BOOST, g * BLEED * BOOST, b * BOOST),
    };

    (sr.min(1.0), sg.min(1.0), sb.min(1.0))
}

pub fn scale(src: &[u32], src_w: usize, src_h: usize, factor: usize) -> Vec<u32> {
    let factor = factor.max(3); // minimum 3x for R/G/B columns
    let dst_w = src_w * factor;
    let dst_h = src_h * factor;
    let mut dst = vec![0u32; dst_w * dst_h];

    for sy in 0..src_h {
        for sx in 0..src_w {
            let c = get(src, src_w, src_h, sx as isize, sy as isize);
            let r = ((c >> 16) & 0xFF) as f32 / 255.0;
            let g = ((c >> 8) & 0xFF) as f32 / 255.0;
            let b = (c & 0xFF) as f32 / 255.0;

            for py in 0..factor {
                let is_hgap = py == factor - 1;

                for px in 0..factor {
                    let (mut sr, mut sg, mut sb) = apply_subpixel(r, g, b, px, factor);

                    // Horizontal gap row
                    if is_hgap {
                        sr *= HGAP_BRIGHTNESS / GAP_BRIGHTNESS.max(0.01);
                        sg *= HGAP_BRIGHTNESS / GAP_BRIGHTNESS.max(0.01);
                        sb *= HGAP_BRIGHTNESS / GAP_BRIGHTNESS.max(0.01);
                        // Clamp again for gap intersections
                        sr = sr.min(1.0) * GAP_BRIGHTNESS;
                        sg = sg.min(1.0) * GAP_BRIGHTNESS;
                        sb = sb.min(1.0) * GAP_BRIGHTNESS;
                    }

                    let ri = (sr * 255.0 + 0.5) as u32;
                    let gi = (sg * 255.0 + 0.5) as u32;
                    let bi = (sb * 255.0 + 0.5) as u32;

                    let dx = sx * factor + px;
                    let dy = sy * factor + py;
                    dst[dy * dst_w + dx] = 0xFF000000 | (ri.min(255) << 16) | (gi.min(255) << 8) | bi.min(255);
                }
            }
        }
    }

    dst
}
