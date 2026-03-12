//! EDI — Edge-Directed Interpolation (2x upscaling).
//!
//! A simplified edge-directed interpolation that detects local edge
//! orientation via gradient analysis and blends along the edge to
//! preserve sharpness. Uses a 5-tap directional filter for smooth
//! edges and snaps to nearest for pixel-art-like sharp transitions.
//!
//! This is a pixel-art-friendly variant that combines EDI smoothness
//! with sharp pixel preservation when color differences are large.

use super::get;
use super::color_dist;

/// Edge direction threshold — below this, region is considered flat.
const FLAT_THRESHOLD: f32 = 500.0;
/// Sharp edge threshold — above this, snap to nearest pixel.
const SHARP_THRESHOLD: f32 = 8000.0;

/// Detect edge direction and strength at a diamond point.
/// Returns (direction_angle_in_radians, strength).
#[inline]
fn edge_direction(src: &[u32], w: usize, h: usize, x: isize, y: isize) -> (f32, f32) {
    // Structure tensor from color distances in 3x3 neighborhood
    let mut gxx = 0.0f32;
    let mut gyy = 0.0f32;
    let mut gxy = 0.0f32;

    for ky in 0..2_isize {
        for kx in 0..2_isize {
            let cx = x + kx;
            let cy = y + ky;

            let c = get(src, w, h, cx, cy);
            let r = get(src, w, h, cx + 1, cy);
            let d = get(src, w, h, cx, cy + 1);

            let dx = color_dist(c, r).sqrt();
            let dy = color_dist(c, d).sqrt();

            gxx += dx * dx;
            gyy += dy * dy;
            gxy += dx * dy;
        }
    }

    let strength = gxx + gyy;
    if strength < 0.001 {
        return (0.0, 0.0);
    }

    // Dominant direction from structure tensor eigenanalysis
    let angle = 0.5 * (2.0 * gxy).atan2(gxx - gyy);
    (angle, strength)
}

/// Interpolate the diamond point at (x+0.5, y+0.5).
#[inline]
fn diamond_interp(src: &[u32], w: usize, h: usize, x: isize, y: isize) -> u32 {
    let tl = get(src, w, h, x, y);
    let tr = get(src, w, h, x + 1, y);
    let bl = get(src, w, h, x, y + 1);
    let br = get(src, w, h, x + 1, y + 1);

    // Check for sharp pixel-art edges — snap to nearest
    let d_main = color_dist(tl, br); // \ diagonal similarity
    let d_anti = color_dist(tr, bl); // / diagonal similarity
    let d_h1 = color_dist(tl, tr);
    let d_h2 = color_dist(bl, br);
    let d_v1 = color_dist(tl, bl);
    let d_v2 = color_dist(tr, br);

    let max_d = d_main.max(d_anti).max(d_h1).max(d_h2).max(d_v1).max(d_v2);

    if max_d < FLAT_THRESHOLD {
        // Flat region — simple average
        return avg4(tl, tr, bl, br);
    }

    if max_d > SHARP_THRESHOLD {
        // Very sharp edge — check if one diagonal is much stronger
        if d_main < d_anti * 0.25 {
            // \ diagonal pixels are similar — interpolate along \
            return super::blend_argb(tl, br, 0.5);
        }
        if d_anti < d_main * 0.25 {
            // / diagonal pixels are similar — interpolate along /
            return super::blend_argb(tr, bl, 0.5);
        }
        // Both diagonals sharp — use nearest-of-4
        return avg4(tl, tr, bl, br);
    }

    // Medium edge — use directional interpolation
    let (angle, _strength) = edge_direction(src, w, h, x, y);

    // Map angle to blend weights
    // angle ≈ 0 → horizontal edge → blend vertically (tl+tr vs bl+br)
    // angle ≈ π/2 → vertical edge → blend horizontally (tl+bl vs tr+br)
    // angle ≈ π/4 → \ diagonal → blend tl+br
    // angle ≈ -π/4 → / diagonal → blend tr+bl
    let cos2 = (2.0 * angle).cos(); // -1..1
    let sin2 = (2.0 * angle).sin(); // -1..1

    // Weight for \ diagonal vs / diagonal
    let w_main = (1.0 + sin2) * 0.5; // high when angle ≈ π/4
    let w_anti = (1.0 - sin2) * 0.5; // high when angle ≈ -π/4

    // Weight for horizontal vs vertical
    let w_horz = (1.0 + cos2) * 0.5; // high when angle ≈ 0
    let w_vert = (1.0 - cos2) * 0.5; // high when angle ≈ π/2

    // Blend candidates
    let c_main = super::blend_argb(tl, br, 0.5);
    let c_anti = super::blend_argb(tr, bl, 0.5);
    let c_horz = super::blend_argb(
        super::blend_argb(tl, tr, 0.5),
        super::blend_argb(bl, br, 0.5),
        0.5,
    );
    let c_vert = super::blend_argb(
        super::blend_argb(tl, bl, 0.5),
        super::blend_argb(tr, br, 0.5),
        0.5,
    );

    // Combine based on detected direction
    // Use the two strongest directions
    let mut candidates = [
        (w_main * 0.5, c_main),
        (w_anti * 0.5, c_anti),
        (w_horz, c_horz),
        (w_vert, c_vert),
    ];

    // Sort by weight descending
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Blend top two
    let total = candidates[0].0 + candidates[1].0;
    if total < 0.001 {
        return avg4(tl, tr, bl, br);
    }
    let t = candidates[1].0 / total;
    super::blend_argb(candidates[0].1, candidates[1].1, t)
}

/// Average 4 colors equally.
#[inline(always)]
fn avg4(a: u32, b: u32, c: u32, d: u32) -> u32 {
    let r = (((a >> 16) & 0xFF) + ((b >> 16) & 0xFF) + ((c >> 16) & 0xFF) + ((d >> 16) & 0xFF)) / 4;
    let g = (((a >> 8) & 0xFF) + ((b >> 8) & 0xFF) + ((c >> 8) & 0xFF) + ((d >> 8) & 0xFF)) / 4;
    let bl = ((a & 0xFF) + (b & 0xFF) + (c & 0xFF) + (d & 0xFF)) / 4;
    0xFF000000 | (r << 16) | (g << 8) | bl
}

/// Horizontal edge interpolation at (x+0.5, y) — direction-aware.
#[inline]
fn hedge_interp(src: &[u32], w: usize, h: usize, x: isize, y: isize) -> u32 {
    let left = get(src, w, h, x, y);
    let right = get(src, w, h, x + 1, y);
    let d = color_dist(left, right);

    if d > SHARP_THRESHOLD {
        // Sharp horizontal edge — check which side the vertical edge favors
        let above_l = get(src, w, h, x, y - 1);
        let above_r = get(src, w, h, x + 1, y - 1);
        let below_l = get(src, w, h, x, y + 1);
        let below_r = get(src, w, h, x + 1, y + 1);

        // If vertical continuity is stronger on one side, bias toward it
        let cont_l = color_dist(above_l, left) + color_dist(left, below_l);
        let cont_r = color_dist(above_r, right) + color_dist(right, below_r);

        if cont_l < cont_r * 0.5 {
            return left;
        }
        if cont_r < cont_l * 0.5 {
            return right;
        }
    }

    super::blend_argb(left, right, 0.5)
}

/// Vertical edge interpolation at (x, y+0.5) — direction-aware.
#[inline]
fn vedge_interp(src: &[u32], w: usize, h: usize, x: isize, y: isize) -> u32 {
    let top = get(src, w, h, x, y);
    let bot = get(src, w, h, x, y + 1);
    let d = color_dist(top, bot);

    if d > SHARP_THRESHOLD {
        let left_t = get(src, w, h, x - 1, y);
        let left_b = get(src, w, h, x - 1, y + 1);
        let right_t = get(src, w, h, x + 1, y);
        let right_b = get(src, w, h, x + 1, y + 1);

        let cont_t = color_dist(left_t, top) + color_dist(top, right_t);
        let cont_b = color_dist(left_b, bot) + color_dist(bot, right_b);

        if cont_t < cont_b * 0.5 {
            return top;
        }
        if cont_b < cont_t * 0.5 {
            return bot;
        }
    }

    super::blend_argb(top, bot, 0.5)
}

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

            dst[dy * dst_w + dx] = get(src, src_w, src_h, ix, iy);
            dst[dy * dst_w + dx + 1] = hedge_interp(src, src_w, src_h, ix, iy);
            dst[(dy + 1) * dst_w + dx] = vedge_interp(src, src_w, src_h, ix, iy);
            dst[(dy + 1) * dst_w + dx + 1] = diamond_interp(src, src_w, src_h, ix, iy);
        }
    }

    dst
}
