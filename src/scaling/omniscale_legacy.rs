//! OmniScale Legacy — diagonal-aware pixel-art upscaler with quantized fallback.
//!
//! Algorithm designed by Lior Halphon.
//!
//! Scales pixel art to arbitrary resolutions by detecting and preserving
//! diagonal edges in 2×2 source quads. The approach:
//!
//! 1. Map each output pixel to source coordinates (with −0.5 offset so
//!    interpolation is centered on pixel boundaries).
//! 2. Sample the surrounding 2×2 source texels.
//! 3. Test both diagonals for color similarity. If exactly one diagonal
//!    is present, the sub-pixel position along that diagonal picks the
//!    winning color. Ambiguous cases (both diagonals) are resolved by a
//!    4×4 neighborhood vote; ties average both results.
//! 4. When no diagonal is found, bilinear interpolation is performed and
//!    the result is quantized to whichever of the four source colors is
//!    closest in Manhattan RGB distance.

use super::get;

// ── Color utilities ────────────────────────────────────────────────────

/// Sum of absolute per-channel differences (ignoring alpha).
#[inline(always)]
fn channel_dist(a: u32, b: u32) -> u32 {
    let dr = (((a >> 16) & 0xFF) as i32 - ((b >> 16) & 0xFF) as i32).unsigned_abs();
    let dg = (((a >> 8) & 0xFF) as i32 - ((b >> 8) & 0xFF) as i32).unsigned_abs();
    let db = ((a & 0xFF) as i32 - (b & 0xFF) as i32).unsigned_abs();
    dr + dg + db
}

/// Perceptual similarity — threshold of 15 in Manhattan RGB.
#[inline(always)]
fn alike(a: u32, b: u32) -> bool {
    a == b || channel_dist(a, b) < 15
}

/// Fixed-point blend: `t` in 0..=256 (0 → full `a`, 256 → full `b`).
#[inline(always)]
fn blend(a: u32, b: u32, t: u32) -> u32 {
    if t == 0 { return a; }
    if t >= 256 { return b; }
    let s = 256 - t;
    let r = (((a >> 16) & 0xFF) * s + ((b >> 16) & 0xFF) * t) >> 8;
    let g = (((a >> 8) & 0xFF) * s + ((b >> 8) & 0xFF) * t) >> 8;
    let b_ch = ((a & 0xFF) * s + (b & 0xFF) * t) >> 8;
    0xFF00_0000 | (r << 16) | (g << 8) | b_ch
}

// ── Diagonal detection ─────────────────────────────────────────────────

/// Result of analyzing the two diagonals of a 2×2 quad.
///
/// Rather than storing a tag enum + four colors separately, we precompute
/// a function pointer (closure-like via enum dispatch) that maps any
/// sub-pixel coordinate directly to a color. This collapses classification
/// and evaluation into a single step.
enum QuadEval {
    /// All four texels alike — return this color everywhere.
    Flat(u32),
    /// Single diagonal detected. `axis` is 0 for `/` (rising), 1 for `\` (falling).
    /// The three relevant colors are stored in order: [near, far, diagonal].
    Edge { axis: u8, colors: [u32; 4] },
    /// Both diagonals present — evaluate both and average.
    Ambiguous([u32; 4]),
    /// No diagonal — quantized bilinear.
    Smooth([u32; 4]),
}

/// Analyze the 2×2 quad at (sx, sy) in the source image.
fn analyze(src: &[u32], w: usize, h: usize, sx: isize, sy: isize) -> QuadEval {
    let c = [
        get(src, w, h, sx, sy),         // 0: top-left
        get(src, w, h, sx + 1, sy),     // 1: top-right
        get(src, w, h, sx, sy + 1),     // 2: bottom-left
        get(src, w, h, sx + 1, sy + 1), // 3: bottom-right
    ];

    // `/` diagonal: top-right ≈ bottom-left
    let rising = alike(c[2], c[1]);
    // `\` diagonal: top-left ≈ bottom-right
    let falling = alike(c[0], c[3]);

    if rising && falling {
        // All four similar → flat fill
        if alike(c[0], c[2]) {
            return QuadEval::Flat(c[0]);
        }
        // Both diagonals present — use 4×4 neighborhood to break tie
        let mut bias: i32 = 0;
        for row in -1..3_isize {
            for col in -1..3_isize {
                let n = get(src, w, h, sx + col, sy + row);
                if alike(n, c[0]) { bias += 1; }
                if alike(n, c[2]) { bias -= 1; }
            }
        }
        return if bias < 0 {
            QuadEval::Edge { axis: 1, colors: c }
        } else if bias > 0 {
            QuadEval::Edge { axis: 0, colors: c }
        } else {
            QuadEval::Ambiguous(c)
        };
    }

    if rising {
        QuadEval::Edge { axis: 0, colors: c }
    } else if falling {
        QuadEval::Edge { axis: 1, colors: c }
    } else {
        QuadEval::Smooth(c)
    }
}

/// Evaluate a rising (`/`) diagonal at the given sub-pixel position.
/// Returns one of the three regions: top-left, bottom-right, or on-diagonal.
#[inline(always)]
fn pick_rising(c: &[u32; 4], fx: u32, fy: u32) -> u32 {
    let sum = fx + fy;
    if sum < 128 { c[0] } else if sum > 384 { c[3] } else { c[2] }
}

/// Evaluate a falling (`\`) diagonal.
#[inline(always)]
fn pick_falling(c: &[u32; 4], fx: u32, fy: u32) -> u32 {
    let diff = 256 - fx + fy;
    if diff < 128 { c[1] } else if diff > 384 { c[2] } else { c[0] }
}

/// Map a sub-pixel coordinate through a QuadEval to produce a color.
#[inline(always)]
fn resolve(qe: &QuadEval, fx: u32, fy: u32) -> u32 {
    match qe {
        QuadEval::Flat(c) => *c,

        QuadEval::Edge { axis: 0, colors } => pick_rising(colors, fx, fy),
        QuadEval::Edge { axis: _, colors } => pick_falling(colors, fx, fy),

        QuadEval::Ambiguous(c) => {
            blend(pick_falling(c, fx, fy), pick_rising(c, fx, fy), 128)
        }

        QuadEval::Smooth(c) => {
            // Bilinear blend, then snap to nearest source texel
            let top = blend(c[0], c[1], fx);
            let bot = blend(c[2], c[3], fx);
            let mixed = blend(top, bot, fy);

            // Find the closest of the four original colors
            let distances = [
                channel_dist(mixed, c[0]),
                channel_dist(mixed, c[1]),
                channel_dist(mixed, c[2]),
                channel_dist(mixed, c[3]),
            ];
            let min_d = distances[0].min(distances[1]).min(distances[2]).min(distances[3]);
            // First match wins (deterministic tie-breaking)
            if distances[0] == min_d { c[0] }
            else if distances[1] == min_d { c[1] }
            else if distances[2] == min_d { c[2] }
            else { c[3] }
        }
    }
}

// ── Public API ─────────────────────────────────────────────────────────

/// Scale to arbitrary output dimensions.
pub fn scale_to(
    src: &[u32], src_w: usize, src_h: usize,
    dst_w: usize, dst_h: usize,
) -> Vec<u32> {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return vec![0u32; dst_w * dst_h];
    }

    let mut out = vec![0u32; dst_w * dst_h];
    let rx = src_w as f32 / dst_w as f32;
    let ry = src_h as f32 / dst_h as f32;

    // Cache the last analyzed quad to skip re-analysis when consecutive
    // output pixels map to the same source position.
    let mut prev: (usize, usize) = (usize::MAX, usize::MAX);
    let mut qe = QuadEval::Flat(0);

    for dy in 0..dst_h {
        let my = (dy as f32 + 0.5) * ry - 0.5;
        let sy = (my.floor().max(0.0) as usize).min(src_h - 1);
        let fy = ((my - sy as f32).clamp(0.0, 1.0) * 256.0) as u32;

        for dx in 0..dst_w {
            let mx = (dx as f32 + 0.5) * rx - 0.5;
            let sx = (mx.floor().max(0.0) as usize).min(src_w - 1);
            let fx = ((mx - sx as f32).clamp(0.0, 1.0) * 256.0) as u32;

            let key = (sx, sy);
            if key != prev {
                qe = analyze(src, src_w, src_h, sx as isize, sy as isize);
                prev = key;
            }

            out[dy * dst_w + dx] = resolve(&qe, fx, fy);
        }
    }

    out
}

/// Scale by an integer factor.
pub fn scale(src: &[u32], src_w: usize, src_h: usize, factor: u32) -> Vec<u32> {
    scale_to(src, src_w, src_h, src_w * factor as usize, src_h * factor as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn test_image() -> (Vec<u32>, usize, usize) {
        let w = 32;
        let h = 32;
        let mut img = vec![0xFF000000u32; w * h];
        for y in 0..8 { for x in 0..8 { img[y*w+x] = 0xFF_FF0000; } }
        for y in 0..8 { for x in 8..16 { img[y*w+x] = 0xFF_00FF00; } }
        for y in 8..16 { for x in 0..8 { img[y*w+x] = 0xFF_0000FF; } }
        for y in 8..16 { for x in 8..16 { img[y*w+x] = if (x+y)%2==0 { 0xFF_FFFFFF } else { 0xFF_000000 }; } }
        for i in 0..16 { img[(16+i)*w+i] = 0xFF_FFFF00; }
        for y in 16..24 { for x in 16..32 { let v = (x-16)*16; img[y*w+x] = 0xFF000000 | (v as u32) << 16 | (v as u32) << 8 | v as u32; } }
        for y in 24..32 { for x in 0..16 { img[y*w+x] = 0xFF_808080 + (x as u32); } }
        (img, w, h)
    }

    fn hash_pixels(pixels: &[u32]) -> u64 {
        let mut h = DefaultHasher::new();
        pixels.hash(&mut h);
        h.finish()
    }

    #[test]
    fn legacy_reference_2x() {
        let (img, w, h) = test_image();
        let out = scale(&img, w, h, 2);
        assert_eq!(hash_pixels(&out), 7247171930574841914);
    }

    #[test]
    fn legacy_reference_3x() {
        let (img, w, h) = test_image();
        let out = scale(&img, w, h, 3);
        assert_eq!(hash_pixels(&out), 2402797872407896390);
    }

    #[test]
    fn legacy_reference_scale_to() {
        let (img, w, h) = test_image();
        let out = scale_to(&img, w, h, 100, 80);
        assert_eq!(hash_pixels(&out), 5645473771994077500);
    }
}
