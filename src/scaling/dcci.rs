//! DCCI — Directional Cubic Convolution Interpolation (2x upscaling).
//!
//! Detects local edge direction using gradient analysis, then applies
//! 1D cubic convolution along the detected edge direction for sharp
//! edge preservation. Falls back to standard bicubic for flat regions.
//!
//! Reference: Zhou & Siu, "Image Interpolation via Directional Filtering" (2012),
//! adapted from the DCCI concept by Muresan & Parks.

use super::get;

/// Luma from ARGB pixel.
#[inline(always)]
fn luma(p: u32) -> f32 {
    let r = ((p >> 16) & 0xFF) as f32;
    let g = ((p >> 8) & 0xFF) as f32;
    let b = (p & 0xFF) as f32;
    0.299 * r + 0.587 * g + 0.114 * b
}

/// Blend two ARGB colors.
#[inline(always)]
fn blend2(a: u32, b: u32, t: f32) -> u32 {
    super::blend_argb(a, b, t)
}

/// Cubic convolution kernel weight (Catmull-Rom, a = -0.5).
#[inline(always)]
fn cubic_weight(d: f32) -> f32 {
    let d = d.abs();
    if d < 1.0 {
        (1.5 * d - 2.5) * d * d + 1.0
    } else if d < 2.0 {
        ((-0.5 * d + 2.5) * d - 4.0) * d + 2.0
    } else {
        0.0
    }
}

/// Blend four colors with cubic weights.
#[inline(always)]
fn cubic_blend4(c: [u32; 4], w: [f32; 4]) -> u32 {
    let sum: f32 = w.iter().sum();
    let inv = if sum.abs() > 0.001 { 1.0 / sum } else { 0.25 };
    let mut r = 0.0f32;
    let mut g = 0.0f32;
    let mut b = 0.0f32;
    for i in 0..4 {
        let wi = w[i] * inv;
        r += ((c[i] >> 16) & 0xFF) as f32 * wi;
        g += ((c[i] >> 8) & 0xFF) as f32 * wi;
        b += (c[i] & 0xFF) as f32 * wi;
    }
    let r = r.round().clamp(0.0, 255.0) as u32;
    let g = g.round().clamp(0.0, 255.0) as u32;
    let b = b.round().clamp(0.0, 255.0) as u32;
    0xFF000000 | (r << 16) | (g << 8) | b
}

/// Detect edge direction at a diamond point (x+0.5, y+0.5).
/// Returns: 0 = no edge (flat), 1 = horizontal, 2 = vertical,
///          3 = main diagonal (\), 4 = anti diagonal (/)
#[inline]
fn detect_direction(src: &[u32], w: usize, h: usize, x: isize, y: isize) -> u8 {
    // Compute gradients using Sobel-like operators on luma
    let p = |dx: isize, dy: isize| luma(get(src, w, h, x + dx, y + dy));

    // 3x3 centered on (x, y) to (x+1, y+1)
    let tl = p(0, 0);
    let tc = p(1, 0);
    let tr = p(2, 0);
    let ml = p(0, 1);
    let mr = p(2, 1);
    let bl = p(0, 2);
    let bc = p(1, 2);
    let br = p(2, 2);

    // Horizontal gradient (detects vertical edges)
    let gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
    // Vertical gradient (detects horizontal edges)
    let gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);

    let gx2 = gx * gx;
    let gy2 = gy * gy;
    let _gxy = gx * gy;

    let strength = (gx2 + gy2).sqrt();
    if strength < 10.0 {
        return 0; // Flat region
    }

    // Compute dominant direction from gradient angle
    let angle = gy.atan2(gx); // radians
    let deg = angle.to_degrees();

    // Edge direction is perpendicular to gradient
    // gradient angle -> edge direction:
    //   ~0° or ~180° → vertical edge → interpolate horizontally (1)
    //   ~90° or ~-90° → horizontal edge → interpolate vertically (2)
    //   ~45° → anti-diagonal edge → interpolate along anti-diagonal (4)
    //   ~-45° → main-diagonal edge → interpolate along main diagonal (3)
    let abs_deg = deg.abs();
    if abs_deg < 22.5 || abs_deg > 157.5 {
        1 // Vertical edge → horizontal interpolation
    } else if abs_deg > 67.5 && abs_deg < 112.5 {
        2 // Horizontal edge → vertical interpolation
    } else if (deg > 22.5 && deg < 67.5) || (deg < -112.5 && deg > -157.5) {
        4 // Anti-diagonal edge
    } else {
        3 // Main diagonal edge
    }
}

/// Interpolate the diamond point at (x+0.5, y+0.5) using direction-adapted cubic.
#[inline]
fn diamond_interp(src: &[u32], w: usize, h: usize, x: isize, y: isize) -> u32 {
    let dir = detect_direction(src, w, h, x, y);

    match dir {
        1 => {
            // Horizontal interpolation along edge (vertical edge detected)
            let c = [
                get(src, w, h, x - 1, y),
                get(src, w, h, x, y),
                get(src, w, h, x + 1, y),
                get(src, w, h, x + 2, y),
            ];
            let c2 = [
                get(src, w, h, x - 1, y + 1),
                get(src, w, h, x, y + 1),
                get(src, w, h, x + 1, y + 1),
                get(src, w, h, x + 2, y + 1),
            ];
            let wt = [cubic_weight(1.5), cubic_weight(0.5), cubic_weight(0.5), cubic_weight(1.5)];
            let top = cubic_blend4(c, wt);
            let bot = cubic_blend4(c2, wt);
            blend2(top, bot, 0.5)
        }
        2 => {
            // Vertical interpolation along edge (horizontal edge detected)
            let c = [
                get(src, w, h, x, y - 1),
                get(src, w, h, x, y),
                get(src, w, h, x, y + 1),
                get(src, w, h, x, y + 2),
            ];
            let c2 = [
                get(src, w, h, x + 1, y - 1),
                get(src, w, h, x + 1, y),
                get(src, w, h, x + 1, y + 1),
                get(src, w, h, x + 1, y + 2),
            ];
            let wt = [cubic_weight(1.5), cubic_weight(0.5), cubic_weight(0.5), cubic_weight(1.5)];
            let left = cubic_blend4(c, wt);
            let right = cubic_blend4(c2, wt);
            blend2(left, right, 0.5)
        }
        3 => {
            // Main diagonal (\) — interpolate along the diagonal
            let d0 = get(src, w, h, x - 1, y - 1);
            let d1 = get(src, w, h, x, y);
            let d2 = get(src, w, h, x + 1, y + 1);
            let d3 = get(src, w, h, x + 2, y + 2);
            let wt = [cubic_weight(1.5), cubic_weight(0.5), cubic_weight(0.5), cubic_weight(1.5)];
            cubic_blend4([d0, d1, d2, d3], wt)
        }
        4 => {
            // Anti-diagonal (/) — interpolate along the anti-diagonal
            let d0 = get(src, w, h, x + 2, y - 1);
            let d1 = get(src, w, h, x + 1, y);
            let d2 = get(src, w, h, x, y + 1);
            let d3 = get(src, w, h, x - 1, y + 2);
            let wt = [cubic_weight(1.5), cubic_weight(0.5), cubic_weight(0.5), cubic_weight(1.5)];
            cubic_blend4([d0, d1, d2, d3], wt)
        }
        _ => {
            // Flat region — standard bicubic (average of 4 neighbors)
            let tl = get(src, w, h, x, y);
            let tr = get(src, w, h, x + 1, y);
            let bl = get(src, w, h, x, y + 1);
            let br = get(src, w, h, x + 1, y + 1);
            let top = blend2(tl, tr, 0.5);
            let bot = blend2(bl, br, 0.5);
            blend2(top, bot, 0.5)
        }
    }
}

/// Horizontal edge interpolation at (x+0.5, y) using cubic.
#[inline]
fn hedge_interp(src: &[u32], w: usize, h: usize, x: isize, y: isize) -> u32 {
    let c = [
        get(src, w, h, x - 1, y),
        get(src, w, h, x, y),
        get(src, w, h, x + 1, y),
        get(src, w, h, x + 2, y),
    ];
    let wt = [cubic_weight(1.5), cubic_weight(0.5), cubic_weight(0.5), cubic_weight(1.5)];
    cubic_blend4(c, wt)
}

/// Vertical edge interpolation at (x, y+0.5) using cubic.
#[inline]
fn vedge_interp(src: &[u32], w: usize, h: usize, x: isize, y: isize) -> u32 {
    let c = [
        get(src, w, h, x, y - 1),
        get(src, w, h, x, y),
        get(src, w, h, x, y + 1),
        get(src, w, h, x, y + 2),
    ];
    let wt = [cubic_weight(1.5), cubic_weight(0.5), cubic_weight(0.5), cubic_weight(1.5)];
    cubic_blend4(c, wt)
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

            // Top-left: original pixel
            dst[dy * dst_w + dx] = get(src, src_w, src_h, ix, iy);

            // Top-right: horizontal cubic interpolation
            dst[dy * dst_w + dx + 1] = hedge_interp(src, src_w, src_h, ix, iy);

            // Bottom-left: vertical cubic interpolation
            dst[(dy + 1) * dst_w + dx] = vedge_interp(src, src_w, src_h, ix, iy);

            // Bottom-right: direction-adaptive diamond interpolation
            dst[(dy + 1) * dst_w + dx + 1] = diamond_interp(src, src_w, src_h, ix, iy);
        }
    }

    dst
}
