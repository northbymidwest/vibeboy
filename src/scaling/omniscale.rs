//! OmniScale — resolution-independent pixel-art upscaler.
//!
//! Algorithm designed by Lior Halphon.
//!
//! Pattern-based edge interpolation generalized from HQnx principles to
//! arbitrary output resolutions. Each output pixel is mapped back to source
//! coordinates; the 3×3 source neighborhood is classified into an 8-bit
//! edge descriptor, then a table of geometric rules determines how to
//! interpolate between the center pixel and its neighbors.
//!
//! Quadrant symmetry reduces the problem: sub-pixel offsets are always
//! reflected into the top-left quarter, and the neighborhood window is
//! oriented to match. Anti-aliased transitions use a `pixel_size` metric
//! (the Euclidean length of one destination pixel in source space).

use super::get;

// ── Color channel helpers ──────────────────────────────────────────────

/// Unpack ARGB channels as i32 for arithmetic.
#[inline(always)]
fn rgb(c: u32) -> (i32, i32, i32) {
    (((c >> 16) & 0xFF) as i32, ((c >> 8) & 0xFF) as i32, (c & 0xFF) as i32)
}

/// Perceptual difference test in a YCbCr-like integer space.
///
/// The transform (scaled ×8 to stay in integers):
///   Y·8  = 2·(ΔR + ΔG + ΔB)    threshold 36
///   Cb·8 = 2·(ΔR − ΔB)         threshold  4
///   Cr·8 = −ΔR + 2·ΔG − ΔB     threshold 10
#[inline(always)]
fn colors_differ(a: u32, b: u32) -> bool {
    if a == b { return false; }
    let (ra, ga, ba) = rgb(a);
    let (rb, gb, bb) = rgb(b);
    let (dr, dg, db) = (ra - rb, ga - gb, ba - bb);
    (2 * (dr + dg + db)).unsigned_abs() > 36
        || (2 * (dr - db)).unsigned_abs() > 4
        || (-dr + 2 * dg - db).unsigned_abs() > 10
}

// ── Blending ───────────────────────────────────────────────────────────

/// Blend two ARGB colors. `t` in 0..=256 (0→a, 256→b).
#[inline(always)]
fn mix(a: u32, b: u32, t: u32) -> u32 {
    if t == 0 { return a; }
    if t >= 256 { return b; }
    let s = 256 - t;
    let r = (((a >> 16) & 0xFF) * s + ((b >> 16) & 0xFF) * t) >> 8;
    let g = (((a >> 8) & 0xFF) * s + ((b >> 8) & 0xFF) * t) >> 8;
    let bl = ((a & 0xFF) * s + (b & 0xFF) * t) >> 8;
    0xFF000000 | (r << 16) | (g << 8) | bl
}

/// Blend with a float factor in [0,1].
#[inline(always)]
fn mixf(a: u32, b: u32, f: f32) -> u32 {
    mix(a, b, (f.clamp(0.0, 1.0) * 256.0 + 0.5) as u32)
}

/// Three-way weighted blend (weights should sum to 1).
#[inline(always)]
fn mix3(a: u32, b: u32, c: u32, wa: f32, wb: f32, wc: f32) -> u32 {
    let r = (((a >> 16) & 0xFF) as f32 * wa + ((b >> 16) & 0xFF) as f32 * wb + ((c >> 16) & 0xFF) as f32 * wc) as u32;
    let g = (((a >> 8) & 0xFF) as f32 * wa + ((b >> 8) & 0xFF) as f32 * wb + ((c >> 8) & 0xFF) as f32 * wc) as u32;
    let bl = ((a & 0xFF) as f32 * wa + (b & 0xFF) as f32 * wb + (c & 0xFF) as f32 * wc) as u32;
    0xFF000000 | (r.min(255) << 16) | (g.min(255) << 8) | bl.min(255)
}

// ── Pattern-driven rule table ──────────────────────────────────────────
//
// Instead of a cascade of if-else checks, rules are encoded as entries in
// a static table. Each entry has a mask/value pair for the 8-bit pattern,
// an optional secondary condition, and an action enum describing what
// blend to perform.
//
// The dispatcher walks the table top-to-bottom; the first matching entry
// wins. This produces identical results to the original cascade but makes
// the structure data-driven.

/// What kind of blend a rule performs.
#[derive(Clone, Copy)]
enum Action {
    /// Blend center↔left neighbor along X: mix(w4, w3, 0.5−px)
    SlideX,
    /// Blend center↔top neighbor along Y: mix(w4, w1, 0.5−py)
    SlideY,
    /// Return center unmodified.
    Solid,
    /// Gentle corner: mix(w4, mix(w4, w0, 0.5−px), 0.5−py)
    GentleCorner,
    /// Asymmetric corner where top neighbor differs: 3-way mix fading to center.
    PartialCornerTop,
    /// Asymmetric corner where left neighbor differs: 3-way mix fading to center.
    PartialCornerLeft,
    /// 45° circular diagonal with radial AA band.
    CircularDiag,
    /// 2:1 slope diagonal, horizontal-dominant (px − 2·py).
    SteepH,
    /// 2:1 slope diagonal, vertical-dominant (py − 2·px).
    SteepV,
    /// Reverse 2:1 slope (px + 2·py, offset 1).
    ReverseH,
    /// Reverse 2:1 slope (py + 2·px, offset 1).
    ReverseV,
    /// Smooth linear corner falloff: mix(w4, w0, (1−px−py)/2).
    LinearCorner,
    /// AA diagonal at 45° with pixel_size transition band.
    AaDiag,
    /// Bilinear-ish: top neighbor matches left, use midpoint blend.
    BilinearMid,
    /// Bilinear-ish: all three (w0,w1,w3) similar, full 2D interpolation.
    BilinearFull,
}

/// Extra condition beyond the mask/value pattern match.
#[derive(Clone, Copy)]
enum Cond {
    None,
    /// w1 differs from w5 (top ≠ right).
    TopNeRight,
    /// w7 differs from w3 (bottom ≠ left).
    BotNeLeft,
    /// w3 differs from w1 (left ≠ top).
    LeftNeTop,
}

struct Rule {
    mask: u32,
    value: u32,
    cond: Cond,
    action: Action,
}

const RULES: &[Rule] = &[
    // ── Strong cardinal edges ──
    Rule { mask: 0xBF, value: 0x37, cond: Cond::TopNeRight, action: Action::SlideX },
    Rule { mask: 0xDB, value: 0x13, cond: Cond::TopNeRight, action: Action::SlideX },
    Rule { mask: 0xDB, value: 0x49, cond: Cond::BotNeLeft,  action: Action::SlideY },
    Rule { mask: 0xEF, value: 0x6D, cond: Cond::BotNeLeft,  action: Action::SlideY },
    // ── Isolated corner ──
    Rule { mask: 0x0B, value: 0x0B, cond: Cond::LeftNeTop, action: Action::Solid },
    Rule { mask: 0xFE, value: 0x4A, cond: Cond::LeftNeTop, action: Action::Solid },
    Rule { mask: 0xFE, value: 0x1A, cond: Cond::LeftNeTop, action: Action::Solid },
    // ── Gentle corners ──
    Rule { mask: 0x6F, value: 0x2A, cond: Cond::LeftNeTop, action: Action::GentleCorner },
    Rule { mask: 0x5B, value: 0x0A, cond: Cond::LeftNeTop, action: Action::GentleCorner },
    Rule { mask: 0xBF, value: 0x3A, cond: Cond::LeftNeTop, action: Action::GentleCorner },
    Rule { mask: 0xDF, value: 0x5A, cond: Cond::LeftNeTop, action: Action::GentleCorner },
    Rule { mask: 0x9F, value: 0x8A, cond: Cond::LeftNeTop, action: Action::GentleCorner },
    Rule { mask: 0xCF, value: 0x8A, cond: Cond::LeftNeTop, action: Action::GentleCorner },
    Rule { mask: 0xEF, value: 0x4E, cond: Cond::LeftNeTop, action: Action::GentleCorner },
    Rule { mask: 0x3F, value: 0x0E, cond: Cond::LeftNeTop, action: Action::GentleCorner },
    Rule { mask: 0xFB, value: 0x5A, cond: Cond::LeftNeTop, action: Action::GentleCorner },
    Rule { mask: 0xBB, value: 0x8A, cond: Cond::LeftNeTop, action: Action::GentleCorner },
    Rule { mask: 0x7F, value: 0x5A, cond: Cond::LeftNeTop, action: Action::GentleCorner },
    Rule { mask: 0xAF, value: 0x8A, cond: Cond::LeftNeTop, action: Action::GentleCorner },
    Rule { mask: 0xEB, value: 0x8A, cond: Cond::LeftNeTop, action: Action::GentleCorner },
    // ── Partial corners ──
    Rule { mask: 0x0B, value: 0x08, cond: Cond::None, action: Action::PartialCornerTop },
    Rule { mask: 0x0B, value: 0x02, cond: Cond::None, action: Action::PartialCornerLeft },
    // ── Full 45° diagonal ──
    Rule { mask: 0x2F, value: 0x2F, cond: Cond::None, action: Action::CircularDiag },
    // ── Steep 2:1 diagonals ──
    Rule { mask: 0xBF, value: 0x37, cond: Cond::None, action: Action::SteepH },
    Rule { mask: 0xDB, value: 0x13, cond: Cond::None, action: Action::SteepH },
    Rule { mask: 0xDB, value: 0x49, cond: Cond::None, action: Action::SteepV },
    Rule { mask: 0xEF, value: 0x6D, cond: Cond::None, action: Action::SteepV },
    // ── Reverse-slope diagonals ──
    Rule { mask: 0xBF, value: 0x8F, cond: Cond::None, action: Action::ReverseH },
    Rule { mask: 0x7E, value: 0x0E, cond: Cond::None, action: Action::ReverseH },
    Rule { mask: 0x7E, value: 0x2A, cond: Cond::None, action: Action::ReverseV },
    Rule { mask: 0xEF, value: 0xAB, cond: Cond::None, action: Action::ReverseV },
    // ── Cardinal slides ──
    Rule { mask: 0x1B, value: 0x03, cond: Cond::None, action: Action::SlideX },
    Rule { mask: 0x4F, value: 0x43, cond: Cond::None, action: Action::SlideX },
    Rule { mask: 0x8B, value: 0x83, cond: Cond::None, action: Action::SlideX },
    Rule { mask: 0x6B, value: 0x43, cond: Cond::None, action: Action::SlideX },
    Rule { mask: 0x4B, value: 0x09, cond: Cond::None, action: Action::SlideY },
    Rule { mask: 0x8B, value: 0x89, cond: Cond::None, action: Action::SlideY },
    Rule { mask: 0x1F, value: 0x19, cond: Cond::None, action: Action::SlideY },
    Rule { mask: 0x3B, value: 0x19, cond: Cond::None, action: Action::SlideY },
    // ── Smooth linear corners ──
    Rule { mask: 0xFB, value: 0x6A, cond: Cond::None, action: Action::LinearCorner },
    Rule { mask: 0x6F, value: 0x6E, cond: Cond::None, action: Action::LinearCorner },
    Rule { mask: 0x3F, value: 0x3E, cond: Cond::None, action: Action::LinearCorner },
    Rule { mask: 0xFB, value: 0xFA, cond: Cond::None, action: Action::LinearCorner },
    Rule { mask: 0xDF, value: 0xDE, cond: Cond::None, action: Action::LinearCorner },
    Rule { mask: 0xDF, value: 0x1E, cond: Cond::None, action: Action::LinearCorner },
    // ── AA diagonals ──
    Rule { mask: 0x4F, value: 0x4B, cond: Cond::None, action: Action::AaDiag },
    Rule { mask: 0x9F, value: 0x1B, cond: Cond::None, action: Action::AaDiag },
    Rule { mask: 0x2F, value: 0x0B, cond: Cond::None, action: Action::AaDiag },
    Rule { mask: 0xBE, value: 0x0A, cond: Cond::None, action: Action::AaDiag },
    Rule { mask: 0xEE, value: 0x0A, cond: Cond::None, action: Action::AaDiag },
    Rule { mask: 0x7E, value: 0x0A, cond: Cond::None, action: Action::AaDiag },
    Rule { mask: 0xEB, value: 0x4B, cond: Cond::None, action: Action::AaDiag },
    Rule { mask: 0x3B, value: 0x1B, cond: Cond::None, action: Action::AaDiag },
    // ── Bilinear fallbacks ──
    Rule { mask: 0x0B, value: 0x01, cond: Cond::None, action: Action::BilinearMid },
    Rule { mask: 0x0B, value: 0x00, cond: Cond::None, action: Action::BilinearFull },
];

/// Compute the corner blend used by several diagonal rules. When the
/// corner pixel differs from both cardinals, interpolate along the
/// diagonal; otherwise do a smooth three-way blend.
#[inline(always)]
fn corner_blend(w0: u32, w1: u32, w3: u32, px: f32, py: f32) -> u32 {
    if colors_differ(w0, w1) || colors_differ(w0, w3) {
        mixf(w1, w3, py - px + 0.5)
    } else {
        mixf(mixf(mix3(w1, w0, w3, 0.375, 0.25, 0.375), w3, py * 2.0), w1, px * 2.0)
    }
}

/// Anti-aliased transition along a line at signed distance `d` from the
/// boundary, with half-width `hw`. Returns the blend factor toward the
/// "outside" color (0 = fully inside, 1 = fully outside).
#[inline(always)]
fn aa_band(d: f32, hw: f32) -> f32 {
    (d + hw) / (2.0 * hw)
}

// ── Core evaluation ────────────────────────────────────────────────────

/// Evaluate a single output pixel. `w` holds the 3×3 neighborhood colors
/// (oriented so the sub-pixel offset falls in the top-left quarter),
/// `pat` is the 8-bit difference pattern, and (px,py) is the sub-pixel
/// position in [0, 0.5].
#[inline(always)]
fn evaluate(
    w: &[u32; 9], pat: u32,
    src: &[u32], sw: usize, sh: usize,
    cx: isize, cy: isize, ox: isize, oy: isize,
    px: f32, py: f32, pixel_size: f32,
) -> u32 {
    let center = w[4];

    // Walk the rule table looking for the first match.
    for rule in RULES {
        if (pat & rule.mask) != rule.value { continue; }

        let cond_ok = match rule.cond {
            Cond::None => true,
            Cond::TopNeRight => colors_differ(w[1], w[5]),
            Cond::BotNeLeft  => colors_differ(w[7], w[3]),
            Cond::LeftNeTop  => colors_differ(w[3], w[1]),
        };
        if !cond_ok { continue; }

        return match rule.action {
            Action::SlideX => mixf(center, w[3], 0.5 - px),
            Action::SlideY => mixf(center, w[1], 0.5 - py),
            Action::Solid  => center,

            Action::GentleCorner => {
                mixf(center, mixf(center, w[0], 0.5 - px), 0.5 - py)
            }

            Action::PartialCornerTop => {
                let tip = mix3(w[0], w[1], center, 0.375, 0.25, 0.375);
                let edge = mixf(center, w[1], 0.5);
                mixf(mixf(tip, edge, px * 2.0), center, py * 2.0)
            }
            Action::PartialCornerLeft => {
                let tip = mix3(w[0], w[3], center, 0.375, 0.25, 0.375);
                let edge = mixf(center, w[3], 0.5);
                mixf(mixf(tip, edge, py * 2.0), center, px * 2.0)
            }

            Action::CircularDiag => {
                let dx = 0.5 - px;
                let dy = 0.5 - py;
                let d2 = dx * dx + dy * dy;
                let inner = 0.5 - pixel_size / 2.0;
                if d2 < inner * inner { center }
                else {
                    let outer_color = corner_blend(w[0], w[1], w[3], px, py);
                    let outer = 0.5 + pixel_size / 2.0;
                    if d2 > outer * outer { outer_color }
                    else { mixf(center, outer_color, (d2.sqrt() - 0.5 + pixel_size / 2.0) / pixel_size) }
                }
            }

            Action::SteepH => {
                let d = px - 2.0 * py;
                let hw = pixel_size * 2.236 / 2.0;
                if d > hw { w[1] }
                else {
                    let base = mixf(w[3], center, px + 0.5);
                    if d < -hw { base }
                    else { mixf(base, w[1], aa_band(d, hw)) }
                }
            }
            Action::SteepV => {
                let d = py - 2.0 * px;
                let hw = pixel_size * 2.236 / 2.0;
                if d > hw { w[3] }
                else {
                    let base = mixf(w[1], center, px + 0.5);
                    if d < -hw { base }
                    else { mixf(base, w[3], aa_band(d, hw)) }
                }
            }

            Action::ReverseH => {
                let d = px + 2.0 * py;
                let hw = pixel_size * 2.236 / 2.0;
                if d > 1.0 + hw { center }
                else {
                    let edge_color = corner_blend(w[0], w[1], w[3], px, py);
                    if d < 1.0 - hw { edge_color }
                    else { mixf(edge_color, center, aa_band(d - 1.0, hw)) }
                }
            }
            Action::ReverseV => {
                let d = py + 2.0 * px;
                let hw = pixel_size * 2.236 / 2.0;
                if d > 1.0 + hw { center }
                else {
                    let edge_color = corner_blend(w[0], w[1], w[3], px, py);
                    if d < 1.0 - hw { edge_color }
                    else { mixf(edge_color, center, aa_band(d - 1.0, hw)) }
                }
            }

            Action::LinearCorner => {
                mixf(center, w[0], (1.0 - px - py) / 2.0)
            }

            Action::AaDiag => {
                let d = px + py;
                let hw = pixel_size / 2.0;
                if d > 0.5 + hw { center }
                else {
                    let edge_color = corner_blend(w[0], w[1], w[3], px, py);
                    if d < 0.5 - hw { edge_color }
                    else { mixf(edge_color, center, aa_band(d - 0.5, hw)) }
                }
            }

            Action::BilinearMid => {
                mixf(
                    mixf(center, w[3], 0.5 - px),
                    mixf(w[1], mixf(w[1], w[3], 0.5), 0.5 - px),
                    0.5 - py,
                )
            }
            Action::BilinearFull => {
                mixf(
                    mixf(center, w[3], 0.5 - px),
                    mixf(w[1], w[0], 0.5 - px),
                    0.5 - py,
                )
            }
        };
    }

    // ── Extended sampling fallback ──────────────────────────────────────
    //
    // No 3×3 rule matched. Sample 7 more pixels behind the corner to
    // determine whether we're on a true diagonal or in a flat region.
    let d = px + py;
    if d > 0.5 + pixel_size / 2.0 {
        return center;
    }

    // The 7 extended samples form an L-shape two rows/columns beyond the
    // 3×3 window, in the direction of the top-left corner.
    let extras = [
        get(src, sw, sh, cx - ox * 2, cy - oy * 2),
        get(src, sw, sh, cx - ox,     cy - oy * 2),
        get(src, sw, sh, cx,          cy - oy * 2),
        get(src, sw, sh, cx + ox,     cy - oy * 2),
        get(src, sw, sh, cx - ox * 2, cy - oy),
        get(src, sw, sh, cx - ox * 2, cy),
        get(src, sw, sh, cx - ox * 2, cy + oy),
    ];

    let mut extended = pat;
    for (i, &px_color) in extras.iter().enumerate() {
        if colors_differ(px_color, center) {
            extended |= 1 << (8 + i);
        }
    }

    // Majority vote: if more extended pixels match center than differ,
    // this is a diagonal edge — blend along it.
    if extended.count_ones() as i32 <= 7 {
        let edge = mixf(w[1], w[3], py - px + 0.5);
        if d < 0.5 - pixel_size / 2.0 { edge }
        else { mixf(edge, center, (d + pixel_size / 2.0 - 0.5) / pixel_size) }
    } else {
        center
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

    let mut dst = vec![0u32; dst_w * dst_h];
    let pixel_size = ((src_w as f32 / dst_w as f32).powi(2)
        + (src_h as f32 / dst_h as f32).powi(2)).sqrt();

    // Precompute which source pixels have a completely uniform 3×3 window.
    let mut uniform = vec![false; src_w * src_h];
    for y in 0..src_h {
        let ya = y.saturating_sub(1);
        let yb = (y + 1).min(src_h - 1);
        for x in 0..src_w {
            let c = src[y * src_w + x];
            let xa = x.saturating_sub(1);
            let xb = (x + 1).min(src_w - 1);
            uniform[y * src_w + x] =
                src[ya * src_w + xa] == c && src[ya * src_w + x] == c && src[ya * src_w + xb] == c &&
                src[y * src_w + xa] == c && src[y * src_w + xb] == c &&
                src[yb * src_w + xa] == c && src[yb * src_w + x] == c && src[yb * src_w + xb] == c;
        }
    }

    let rx = src_w as f32 / dst_w as f32;
    let ry = src_h as f32 / dst_h as f32;

    // Cache the neighborhood for runs of output pixels mapping to the
    // same source pixel + quadrant.
    let mut last: (usize, usize, isize, isize) = (usize::MAX, usize::MAX, 0, 0);
    let mut nb_colors = [0u32; 9];
    let mut nb_pat = 0u32;

    for dy in 0..dst_h {
        let syf = (dy as f32 + 0.5) * ry;
        let syi = (syf as usize).min(src_h - 1);
        let fy = syf - syi as f32;

        for dx in 0..dst_w {
            let sxf = (dx as f32 + 0.5) * rx;
            let sxi = (sxf as usize).min(src_w - 1);

            if uniform[syi * src_w + sxi] {
                dst[dy * dst_w + dx] = src[syi * src_w + sxi];
                continue;
            }

            let fx = sxf - sxi as f32;

            // Reflect into top-left quadrant
            let (px, ox) = if fx > 0.5 { (1.0 - fx, -1isize) } else { (fx, 1isize) };
            let (py, oy) = if fy > 0.5 { (1.0 - fy, -1isize) } else { (fy, 1isize) };

            let cx = sxi as isize;
            let cy = syi as isize;
            let key = (sxi, syi, ox, oy);

            if key != last {
                // Sample the oriented 3×3 window and build the pattern.
                let offsets: [(isize, isize); 9] = [
                    (-ox, -oy), (0, -oy), (ox, -oy),
                    (-ox,   0), (0,   0), (ox,   0),
                    (-ox,  oy), (0,  oy), (ox,  oy),
                ];
                for (i, &(dx, dy)) in offsets.iter().enumerate() {
                    nb_colors[i] = get(src, src_w, src_h, cx + dx, cy + dy);
                }
                let c = nb_colors[4];
                nb_pat = 0;
                for (i, &neighbor) in nb_colors.iter().enumerate() {
                    if i != 4 && colors_differ(neighbor, c) {
                        // Bits 0-3 map to indices 0-3, bits 4-7 map to indices 5-8
                        let bit = if i < 4 { i } else { i - 1 };
                        nb_pat |= 1 << bit;
                    }
                }
                last = key;
            }

            dst[dy * dst_w + dx] = evaluate(
                &nb_colors, nb_pat,
                src, src_w, src_h, cx, cy, ox, oy,
                px, py, pixel_size,
            );
        }
    }

    dst
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
        // Solid regions
        for y in 0..8 { for x in 0..8 { img[y*w+x] = 0xFF_FF0000; } }
        for y in 0..8 { for x in 8..16 { img[y*w+x] = 0xFF_00FF00; } }
        for y in 8..16 { for x in 0..8 { img[y*w+x] = 0xFF_0000FF; } }
        // Checkerboard
        for y in 8..16 { for x in 8..16 { img[y*w+x] = if (x+y)%2==0 { 0xFF_FFFFFF } else { 0xFF_000000 }; } }
        // Diagonal line
        for i in 0..16 { img[(16+i)*w+i] = 0xFF_FFFF00; }
        // Gradient
        for y in 16..24 { for x in 16..32 { let v = (x-16)*16; img[y*w+x] = 0xFF000000 | (v as u32) << 16 | (v as u32) << 8 | v as u32; } }
        // Subtle color differences
        for y in 24..32 { for x in 0..16 { img[y*w+x] = 0xFF_808080 + (x as u32); } }
        (img, w, h)
    }

    fn hash_pixels(pixels: &[u32]) -> u64 {
        let mut h = DefaultHasher::new();
        pixels.hash(&mut h);
        h.finish()
    }

    #[test]
    fn omniscale_reference_2x() {
        let (img, w, h) = test_image();
        let out = scale(&img, w, h, 2);
        assert_eq!(hash_pixels(&out), 11625170078773025937);
    }

    #[test]
    fn omniscale_reference_3x() {
        let (img, w, h) = test_image();
        let out = scale(&img, w, h, 3);
        assert_eq!(hash_pixels(&out), 802860077697398388);
    }

    #[test]
    fn omniscale_reference_scale_to() {
        let (img, w, h) = test_image();
        let out = scale_to(&img, w, h, 100, 80);
        assert_eq!(hash_pixels(&out), 3839519069517137725);
    }
}
