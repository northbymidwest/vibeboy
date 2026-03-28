//! Super xBR — edge-directed 2x scaling by Hyllian.
//!
//! The algorithm fills a 2x output grid in three stages:
//!
//! 1. **Diagonal stage**: Each source pixel's diagonal midpoint (the center
//!    of its 2x2 output quad) is interpolated using gradient analysis on the
//!    surrounding 4x4 source neighborhood.
//!
//! 2. **Cardinal stage**: Horizontal and vertical midpoints are computed from
//!    the partially-filled 2x buffer. A rotated sampling grid ensures only
//!    already-populated positions (source + diagonal) are read.
//!
//! 3. **Polish stage**: A light refinement sweep over the complete 2x buffer
//!    corrects residual staircase artifacts using localized gradient analysis.
//!
//! At each stage the same core logic applies: analyze a 4x4 neighborhood to
//! determine whether the local gradient runs diagonally or cardinally, then
//! blend between cubic-filtered candidates accordingly.

use super::get;
use super::channels;
use super::pack_channels;

// ── Neighborhood ────────────────────────────────────────────────────────────

/// A 4x4 pixel neighborhood laid out as four rows, top to bottom.
///
/// ```text
///   r0c0  r0c1  r0c2  r0c3
///   r1c0  r1c1  r1c2  r1c3
///   r2c0  r2c1  r2c2  r2c3
///   r3c0  r3c1  r3c2  r3c3
/// ```
///
/// The interpolation target sits at the center of the inner 2x2
/// (r1c1, r1c2, r2c1, r2c2).
struct Neighborhood {
    px: [u32; 16],
    lum: [f32; 16],
}

impl Neighborhood {
    /// Build from 16 pixel values, pre-computing luma for each.
    fn new(pixels: [u32; 16]) -> Self {
        let mut lum = [0.0f32; 16];
        for (i, &p) in pixels.iter().enumerate() {
            let r = ((p >> 16) & 0xFF) as f32;
            let g = ((p >> 8) & 0xFF) as f32;
            let b = (p & 0xFF) as f32;
            lum[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        }
        Self { px: pixels, lum }
    }

}

// Named position constants for the 4x4 grid.
const CTL: usize = 0;  const TL: usize = 1;  const TR: usize = 2;  const CTR: usize = 3;
const LT: usize = 4;   const ITL: usize = 5;  const ITR: usize = 6;  const RT: usize = 7;
const LB: usize = 8;   const IBL: usize = 9;  const IBR: usize = 10; const RB: usize = 11;
const CBL: usize = 12;  const BL: usize = 13;  const BR: usize = 14;  const CBR: usize = 15;

// ── Color math ──────────────────────────────────────────────────────────────

/// Apply a symmetric 4-tap cubic-like kernel to four RGB values.
/// `sharpness` controls the negative lobes (higher = sharper, with ringing risk).
#[inline]
fn cubic_filter(sharpness: f32, colors: [u32; 4]) -> [f32; 3] {
    let taps = [-sharpness, 0.5 + sharpness, 0.5 + sharpness, -sharpness];
    let mut out = [0.0f32; 3];
    for (i, &c) in colors.iter().enumerate() {
        let ch = channels(c);
        for k in 0..3 {
            out[k] += taps[i] * ch[k];
        }
    }
    out
}

/// Apply a symmetric 4-tap kernel to four *summed* RGB pairs.
/// Used for orthogonal candidates where each tap combines two pixels.
#[inline]
fn cubic_filter_paired(sharpness: f32, pairs: [(u32, u32); 4]) -> [f32; 3] {
    let taps = [-sharpness, 0.25 + sharpness, 0.25 + sharpness, -sharpness];
    let mut out = [0.0f32; 3];
    for (i, &(a, b)) in pairs.iter().enumerate() {
        let ca = channels(a);
        let cb = channels(b);
        for k in 0..3 {
            out[k] += taps[i] * (ca[k] + cb[k]);
        }
    }
    out
}

// ── Gradient analysis ───────────────────────────────────────────────────────

/// Six-term gradient profile controlling edge sensitivity.
///
/// The terms weight different spatial relationships in the neighborhood:
/// - `core`: weight for the primary pair straddling the potential edge
/// - `adjacent`: weight for pairs one step along the edge direction
/// - `cross_near` / `cross_far`: weights for pairs crossing the edge
///   (negative values reduce sensitivity to perpendicular variation)
/// - `parallel`: weight for pairs running parallel at distance
/// - `outer`: weight for the outermost corner pairs
struct GradientProfile {
    adjacent: f32,
    parallel: f32,
    cross_near: f32,
    core: f32,
    cross_far: f32,
    outer: f32,
}

/// Profiles for each processing stage.
const DIAGONAL_PROFILE: GradientProfile = GradientProfile {
    adjacent: 2.0, parallel: 1.0, cross_near: -1.0,
    core: 4.0, cross_far: -1.0, outer: 1.0,
};
const CARDINAL_PROFILE: GradientProfile = GradientProfile {
    adjacent: 1.0, parallel: 0.0, cross_near: 0.0,
    core: 0.0, cross_far: 0.0, outer: 0.0,
};
const POLISH_PROFILE: GradientProfile = GradientProfile {
    adjacent: 0.0, parallel: 0.0, cross_near: 0.0,
    core: 1.0, cross_far: 0.0, outer: 0.0,
};

/// Measure diagonal gradient strength from 14 luma values arranged around
/// the axis being tested. Higher values indicate more variation (stronger edge)
/// along this diagonal.
///
/// The 14 samples are organized into groups based on their spatial relationship
/// to the diagonal axis being measured.
#[inline]
fn diagonal_gradient(
    gp: &GradientProfile,
    // Outer corner pair
    outer_a: f32, outer_b: f32,
    // Near-axis triple (center flanked by two neighbors)
    axis_left: f32, axis_center: f32, axis_right: f32,
    // Core pair straddling the diagonal
    core_a: f32, core_b: f32,
    // Far-axis pair
    far_a: f32, far_b: f32,
    // Parallel support samples
    par_near_a: f32, par_near_b: f32, par_far_a: f32,
    // Cross-axis samples
    cross_a: f32, cross_b: f32,
) -> f32 {
    gp.adjacent * (ld(axis_center, axis_right) + ld(axis_center, axis_left)
                   + ld(par_near_b, par_near_a) + ld(par_near_b, par_far_a))
        + gp.parallel * (ld(far_a, far_b) + ld(core_a, core_b))
        + gp.cross_near * (ld(core_b, far_b) + ld(core_a, far_a))
        + gp.core * ld(core_b, far_a)
        + gp.cross_far * (ld(axis_left, axis_right) + ld(par_near_a, par_far_a))
        + gp.outer * (ld(outer_a, outer_b) + ld(cross_a, cross_b))
}

/// Measure cardinal (horizontal/vertical) gradient strength from 8 luma values.
#[inline]
fn cardinal_gradient(
    gp: &GradientProfile,
    inner: [f32; 4],
    extended: [f32; 4],
) -> f32 {
    gp.core * (ld(inner[0], inner[1]) + ld(inner[2], inner[3]))
        + gp.adjacent * (ld(inner[0], extended[0]) + ld(inner[1], extended[1])
                         + ld(inner[2], extended[2]) + ld(inner[3], extended[3]))
        + gp.cross_near * (ld(inner[0], extended[1]) + ld(inner[2], extended[3])
                           + ld(extended[0], inner[1]) + ld(extended[2], inner[3]))
}

/// Absolute luma difference.
#[inline(always)]
fn ld(a: f32, b: f32) -> f32 {
    (a - b).abs()
}

/// Hermite smoothstep for soft edge transitions.
#[inline(always)]
fn smooth_threshold(x: f32, limit: f32) -> f32 {
    let t = (x / limit).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ── Core interpolation ─────────────────────────────────────────────────────

/// Sharpness parameters for the two cubic filter types.
/// These control the negative lobe magnitude in the 4-tap kernels.
const DIAG_SHARPNESS: f32 = 1.29633 / 10.0;
const ORTHO_SHARPNESS: f32 = 1.75068 / 20.0;

/// Smoothstep threshold — controls how gradual the diagonal/cardinal
/// blending transition is. Higher = more aggressive diagonal preference.
const EDGE_THRESHOLD: f32 = 5.000001;

/// Interpolate the center of a 4x4 neighborhood using edge-directed blending.
fn edge_directed_interp(nb: &Neighborhood, gp: &GradientProfile) -> u32 {
    let l = &nb.lum;

    // ── Diagonal gradient analysis ──
    // Measure variation along the backslash (\) diagonal axis.
    let grad_bs = diagonal_gradient(
        gp,
        l[LT], l[TL],   // outer pair
        l[LB], l[ITL], l[TR],  // near-axis triple
        l[CBL], l[IBL], l[ITR], l[CTR],  // core + far
        l[BL], l[IBR], l[RT],  // parallel support
        l[BR], l[RB],   // cross pair
    );
    // Measure variation along the slash (/) diagonal axis.
    let grad_sl = diagonal_gradient(
        gp,
        l[TR], l[RT],   // outer pair
        l[TL], l[ITR], l[RB],  // near-axis triple
        l[CTL], l[ITL], l[IBR], l[CBR],  // core + far
        l[LT], l[IBL], l[BR],  // parallel support
        l[LB], l[BL],   // cross pair
    );

    // Positive = backslash has more variation → prefer slash interpolation.
    let diag_bias = grad_bs - grad_sl;
    let diag_confidence = smooth_threshold(diag_bias.abs(), EDGE_THRESHOLD);

    // ── Cardinal gradient analysis ──
    let grad_h = cardinal_gradient(
        gp,
        [l[ITR], l[IBR], l[ITL], l[IBL]],
        [l[TR], l[BR], l[TL], l[BL]],
    );
    let grad_v = cardinal_gradient(
        gp,
        [l[ITL], l[ITR], l[IBL], l[IBR]],
        [l[LT], l[RT], l[LB], l[RB]],
    );
    let cardinal_bias = grad_h - grad_v;

    // ── Build candidate colors ──

    // Diagonal candidates: 4-tap cubic along each diagonal through the center.
    // Slash (/) diagonal: corner_bl → inner_bl → inner_tr → corner_tr
    let slash_color = cubic_filter(
        DIAG_SHARPNESS,
        [nb.px[CBL], nb.px[IBL], nb.px[ITR], nb.px[CTR]],
    );
    // Backslash (\) diagonal: corner_tl → inner_tl → inner_br → corner_br
    let backslash_color = cubic_filter(
        DIAG_SHARPNESS,
        [nb.px[CTL], nb.px[ITL], nb.px[IBR], nb.px[CBR]],
    );

    // Cardinal candidates: 4-tap cubic on paired rows/columns.
    // Horizontal candidate (averaging left/right column pairs).
    let horiz_color = cubic_filter_paired(
        ORTHO_SHARPNESS,
        [
            (nb.px[LT], nb.px[LB]),
            (nb.px[ITL], nb.px[IBL]),
            (nb.px[ITR], nb.px[IBR]),
            (nb.px[RT], nb.px[RB]),
        ],
    );
    // Vertical candidate (averaging top/bottom row pairs).
    let vert_color = cubic_filter_paired(
        ORTHO_SHARPNESS,
        [
            (nb.px[TR], nb.px[TL]),
            (nb.px[ITR], nb.px[ITL]),
            (nb.px[IBR], nb.px[IBL]),
            (nb.px[BR], nb.px[BL]),
        ],
    );

    // ── Blend ──
    // Select the better diagonal candidate based on gradient direction.
    let diag_pick = if diag_bias >= 0.0 { backslash_color } else { slash_color };
    // Select the better cardinal candidate.
    let card_pick = if cardinal_bias >= 0.0 { vert_color } else { horiz_color };

    // Mix diagonal and cardinal: high confidence → favor diagonal,
    // low confidence → favor cardinal (smoother in flat regions).
    let mut result = [0.0f32; 3];
    for k in 0..3 {
        result[k] = card_pick[k] + diag_confidence * (diag_pick[k] - card_pick[k]);
    }

    // ── Anti-ringing ──
    // Clamp to the color range of the inner 2x2 to prevent overshoot
    // from the negative cubic taps.
    let c_itl = channels(nb.px[ITL]);
    let c_itr = channels(nb.px[ITR]);
    let c_ibl = channels(nb.px[IBL]);
    let c_ibr = channels(nb.px[IBR]);
    for k in 0..3 {
        let lo = c_itl[k].min(c_itr[k]).min(c_ibl[k]).min(c_ibr[k]);
        let hi = c_itl[k].max(c_itr[k]).max(c_ibl[k]).max(c_ibr[k]);
        result[k] = result[k].clamp(lo, hi);
    }

    pack_channels(result)
}

// ── Neighborhood construction helpers ───────────────────────────────────────

/// Build a 4x4 neighborhood from a source buffer at the given center position.
/// The center is the midpoint of the inner 2x2 starting at (cx, cy).
#[inline]
fn neighborhood_from(
    buf: &[u32], w: usize, h: usize,
    offsets: [(isize, isize); 16],
) -> Neighborhood {
    let mut px = [0u32; 16];
    for (i, &(ox, oy)) in offsets.iter().enumerate() {
        px[i] = get(buf, w, h, ox, oy);
    }
    Neighborhood::new(px)
}

// ── Public API ──────────────────────────────────────────────────────────────

pub fn scale(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let out_w = src_w * 2;
    let out_h = src_h * 2;
    let mut buf = vec![0u32; out_w * out_h];

    // Scatter source pixels onto the even-coordinate grid of the output.
    for y in 0..src_h {
        for x in 0..src_w {
            buf[y * 2 * out_w + x * 2] = src[y * src_w + x];
        }
    }

    // ── Stage 1: Diagonal midpoints ─────────────────────────────────────
    // For each source pixel (sx, sy), compute the diagonal midpoint at
    // output position (2*sx+1, 2*sy+1). The 4x4 neighborhood is sampled
    // directly from the source image.
    for sy in 0..src_h {
        for sx in 0..src_w {
            let x = sx as isize;
            let y = sy as isize;
            let offsets = [
                (x-1, y-1), (x, y-1), (x+1, y-1), (x+2, y-1),
                (x-1, y  ), (x, y  ), (x+1, y  ), (x+2, y  ),
                (x-1, y+1), (x, y+1), (x+1, y+1), (x+2, y+1),
                (x-1, y+2), (x, y+2), (x+1, y+2), (x+2, y+2),
            ];
            let nb = neighborhood_from(src, src_w, src_h, offsets);
            buf[(sy * 2 + 1) * out_w + sx * 2 + 1] =
                edge_directed_interp(&nb, &DIAGONAL_PROFILE);
        }
    }

    // ── Stage 2: Cardinal midpoints ─────────────────────────────────────
    // Horizontal and vertical midpoints are computed from the partially-filled
    // 2x buffer. The sampling grid is rotated 45° so that every sample lands
    // on an already-populated position (even,even = source or odd,odd = diagonal).

    // Horizontal midpoints at output (2*sx+1, 2*sy).
    for sy in 0..src_h {
        for sx in 0..src_w {
            let ox = (sx * 2 + 1) as isize;
            let oy = (sy * 2) as isize;
            let offsets = [
                (ox-3, oy  ), (ox-2, oy-1), (ox-1, oy-2), (ox,   oy-3),
                (ox-2, oy+1), (ox-1, oy  ), (ox,   oy-1), (ox+1, oy-2),
                (ox-1, oy+2), (ox,   oy+1), (ox+1, oy  ), (ox+2, oy-1),
                (ox,   oy+3), (ox+1, oy+2), (ox+2, oy+1), (ox+3, oy  ),
            ];
            let nb = neighborhood_from(&buf, out_w, out_h, offsets);
            buf[sy * 2 * out_w + sx * 2 + 1] =
                edge_directed_interp(&nb, &CARDINAL_PROFILE);
        }
    }

    // Vertical midpoints at output (2*sx, 2*sy+1).
    for sy in 0..src_h {
        for sx in 0..src_w {
            let ox = (sx * 2) as isize;
            let oy = (sy * 2 + 1) as isize;
            let offsets = [
                (ox,   oy-3), (ox-1, oy-2), (ox-2, oy-1), (ox-3, oy  ),
                (ox+1, oy-2), (ox,   oy-1), (ox-1, oy  ), (ox-2, oy+1),
                (ox+2, oy-1), (ox+1, oy  ), (ox,   oy+1), (ox-1, oy+2),
                (ox+3, oy  ), (ox+2, oy+1), (ox+1, oy+2), (ox,   oy+3),
            ];
            let nb = neighborhood_from(&buf, out_w, out_h, offsets);
            buf[(sy * 2 + 1) * out_w + sx * 2] =
                edge_directed_interp(&nb, &CARDINAL_PROFILE);
        }
    }

    // ── Stage 3: Polish ─────────────────────────────────────────────────
    // Light refinement sweep with minimal edge weights. Each pixel in the
    // 2x buffer is reconsidered against its immediate 4x4 neighborhood.
    let frozen = buf.clone();
    for y in 0..out_h {
        for x in 0..out_w {
            let ix = x as isize;
            let iy = y as isize;
            let offsets = [
                (ix-2, iy-2), (ix-1, iy-2), (ix, iy-2), (ix+1, iy-2),
                (ix-2, iy-1), (ix-1, iy-1), (ix, iy-1), (ix+1, iy-1),
                (ix-2, iy  ), (ix-1, iy  ), (ix, iy  ), (ix+1, iy  ),
                (ix-2, iy+1), (ix-1, iy+1), (ix, iy+1), (ix+1, iy+1),
            ];
            let nb = neighborhood_from(&frozen, out_w, out_h, offsets);
            buf[y * out_w + x] = edge_directed_interp(&nb, &POLISH_PROFILE);
        }
    }

    buf
}
