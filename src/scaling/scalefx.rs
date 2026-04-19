//! ScaleFX — 3x/9x pixel-art edge interpolation.
//!
//! Implementation of the ScaleFX algorithm by Sp00kyFox.
//! The algorithm detects diagonal edges at up to 6 slope levels and produces a 3x
//! scaled image containing only colors present in the original.
//!
//! Five-stage pipeline:
//!   1. Perceptual color distance metrics
//!   2. Corner interpolation weights
//!   3. Junction arbitration and edge classification
//!   4. Multi-level edge tracing with subpixel assignment
//!   5. 3x block assembly via neighbor lookup

use super::get;

// ---------------------------------------------------------------------------
// Perceptual color distance (Riemersma / compuphase)
// ---------------------------------------------------------------------------

/// Red-weighted perceptual distance between two packed XRGB pixels.
/// The weighting `(2+r̄, 4, 3-r̄)` approximates human color sensitivity.
#[inline(always)]
fn color_distance(p: u32, q: u32) -> f32 {
    let (pr, pg, pb) = unpack_norm(p);
    let (qr, qg, qb) = unpack_norm(q);
    let mean_r = 0.5 * (pr + qr);
    let rd = pr - qr;
    let gd = pg - qg;
    let bd = pb - qb;
    let weighted = (2.0 + mean_r) * rd * rd + 4.0 * gd * gd + (3.0 - mean_r) * bd * bd;
    weighted.max(0.0).sqrt() / 3.0
}

/// Unpack XRGB to normalized [0,1] RGB.
#[inline(always)]
fn unpack_norm(c: u32) -> (f32, f32, f32) {
    (
        ((c >> 16) & 0xFF) as f32 / 255.0,
        ((c >> 8) & 0xFF) as f32 / 255.0,
        (c & 0xFF) as f32 / 255.0,
    )
}

// ---------------------------------------------------------------------------
// Intermediate buffer access
// ---------------------------------------------------------------------------

#[inline(always)]
fn sample(buf: &[[f32; 4]], w: usize, h: usize, x: isize, y: isize) -> [f32; 4] {
    let cx = x.clamp(0, w as isize - 1) as usize;
    let cy = y.clamp(0, h as isize - 1) as usize;
    buf[cy * w + cx]
}

// ---------------------------------------------------------------------------
// Stage 1: Distance metrics
// ---------------------------------------------------------------------------

/// For every pixel, compute distances to its top-left, top, top-right, and
/// right neighbors. The distances encode how "different" each neighbor pair is.
///
/// The resulting 4-tuple per pixel serves as input to corner weight computation.
fn compute_metrics(src: &[u32], w: usize, h: usize) -> Vec<[f32; 4]> {
    let mut metrics = vec![[0.0f32; 4]; w * h];
    for py in 0..h {
        for px in 0..w {
            let ix = px as isize;
            let iy = py as isize;
            let center = get(src, w, h, ix, iy);
            metrics[py * w + px] = [
                color_distance(center, get(src, w, h, ix - 1, iy - 1)), // top-left
                color_distance(center, get(src, w, h, ix,     iy - 1)), // top
                color_distance(center, get(src, w, h, ix + 1, iy - 1)), // top-right
                color_distance(center, get(src, w, h, ix + 1, iy)),     // right
            ];
        }
    }
    metrics
}

// ---------------------------------------------------------------------------
// Stage 2: Corner interpolation weights
// ---------------------------------------------------------------------------

/// Threshold below which color differences suppress interpolation.
const CLARITY_THRESHOLD: f32 = 0.50;

/// Compute the interpolation weight for a single corner.
///
/// `edge_dist` is the distance across the diagonal being considered.
/// `primary` and `secondary` are distance pairs from adjacent edges used
/// to determine directional confidence and whether interpolation is warranted.
#[inline(always)]
fn corner_weight(edge_dist: f32, primary: [f32; 2], secondary: [f32; 2]) -> f32 {
    // Proximity: strong edges (high distance) get suppressed
    let proximity = (CLARITY_THRESHOLD - edge_dist).max(0.0) / CLARITY_THRESHOLD;

    // Directional confidence: prefer the direction with stronger contrast
    let bias = primary[0] - primary[1];
    let stronger_axis = if primary[0].min(secondary[0]) + primary[0]
                         > primary[1].min(secondary[1]) + primary[1]
    { bias } else { -bias };
    let direction = (1.0 - edge_dist + stronger_axis).clamp(0.0, 1.0);

    // Anti-aliasing gate: always interpolate (SFX_SAA=1 default)
    // The product of distances ensures both adjacent edges contribute
    proximity * direction * primary[0] * primary[1]
}

/// Compute the four corner weights for every pixel.
///
/// Each pixel has four corners (top-left, top-right, bottom-right, bottom-left).
/// The weight encodes how strongly each corner wants to be interpolated.
///
/// Metric layout: `[top-left, top, top-right, right]` distances.
/// We reference metrics from the 3×3 neighborhood to get all needed edge distances.
fn compute_corner_weights(metrics: &[[f32; 4]], w: usize, h: usize) -> Vec<[f32; 4]> {
    let mut weights = vec![[0.0f32; 4]; w * h];
    for py in 0..h {
        for px in 0..w {
            let ix = px as isize;
            let iy = py as isize;

            let tl = sample(metrics, w, h, ix - 1, iy - 1);
            let t  = sample(metrics, w, h, ix,     iy - 1);
            let l  = sample(metrics, w, h, ix - 1, iy);
            let c  = sample(metrics, w, h, ix,     iy);
            let r  = sample(metrics, w, h, ix + 1, iy);
            let bl = sample(metrics, w, h, ix - 1, iy + 1);
            let b  = sample(metrics, w, h, ix,     iy + 1);
            let br = sample(metrics, w, h, ix + 1, iy + 1);

            // Corner 0 (top-left): edge across D-E diagonal
            // corner_weight(D.topright, (D.right, E.top), (A.right, D.top))
            let w0 = corner_weight(l[2], [l[3], c[1]], [tl[3], l[1]]);
            // Corner 1 (top-right): edge across E-F diagonal
            let w1 = corner_weight(r[0], [c[3], c[1]], [t[3], r[1]]);
            // Corner 2 (bottom-right): edge across E-I diagonal
            let w2 = corner_weight(b[2], [c[3], b[1]], [b[3], br[1]]);
            // Corner 3 (bottom-left): edge across H-G diagonal
            let w3 = corner_weight(b[0], [l[3], b[1]], [bl[3], bl[1]]);

            weights[py * w + px] = [w0, w1, w2, w3];
        }
    }
    weights
}

// ---------------------------------------------------------------------------
// Stage 3: Junction arbitration and edge classification
// ---------------------------------------------------------------------------

// Branchless comparison helpers (matching GPU step-function semantics).
// Matching GPU step()-based semantics:
// Branchless comparison helpers matching GPU step()-based semantics.
// GPU's vec_ge/scalar_ge use 1-step(a,b) which is strict >, not >=.
#[inline(always)] fn gt(a: f32, b: f32) -> f32 { if a > b { 1.0 } else { 0.0 } }
#[inline(always)] fn lt(a: f32, b: f32) -> f32 { if a < b { 1.0 } else { 0.0 } }
#[inline(always)] fn leq(a: f32, b: f32) -> f32 { if a <= b { 1.0 } else { 0.0 } }
#[inline(always)] fn inv(x: f32) -> f32 { 1.0 - x }

/// Dominance of a corner at a junction: how much stronger it is than its neighbors.
/// Each junction has 4 contributing corners from 4 adjacent pixels.
#[inline(always)]
fn dominance(a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) -> [f32; 4] {
    [
        2.0 * a[1] - (a[0] + a[2]),
        2.0 * b[1] - (b[0] + b[2]),
        2.0 * c[1] - (c[0] + c[2]),
        2.0 * d[1] - (d[0] + d[2]),
    ]
}

/// Junction condition: whether the diagonal edge is strong enough in both
/// directions relative to the orthogonal edges.
#[inline(always)]
fn junction_condition(diagonal: [f32; 2], orth_a: [f32; 2], orth_b: [f32; 2]) -> f32 {
    if diagonal[0] >= orth_a[0].min(orth_a[1]).max(orth_b[0].min(orth_b[1]))
        && diagonal[1] >= orth_a[0].min(orth_b[1]).max(orth_b[0].min(orth_a[1]))
    { 1.0 } else { 0.0 }
}

/// Majority vote to resolve competing corners at a junction.
#[inline(always)]
fn majority_vote(dom: [f32; 4]) -> [f32; 4] {
    let mut result = [0.0f32; 4];
    for i in 0..4 {
        let next = dom[(i + 1) & 3];
        let prev = dom[(i + 3) & 3];
        let opposite = dom[(i + 2) & 3];
        let unopposed = leq(next, 0.0) * leq(prev, 0.0);
        let majority = gt(dom[i] + opposite, next + prev);
        result[i] = (gt(dom[i], 0.0) * (unopposed + majority)).min(1.0);
    }
    result
}

/// Resolve junctions, classify edges, and pack the result.
///
/// Reads both metrics (stage 1) and weights (stage 2). Produces a packed value
/// per pixel encoding four flags per corner: active, horizontal, vertical, orientation.
/// Packing: `(active + 2*horiz + 4*vert + 8*orient) / 15`
fn arbitrate_junctions(
    metrics: &[[f32; 4]],
    weights: &[[f32; 4]],
    w: usize, h: usize,
) -> Vec<[f32; 4]> {
    let mut result = vec![[0.0f32; 4]; w * h];
    for py in 0..h {
        for px in 0..w {
            let ix = px as isize;
            let iy = py as isize;

            // Metric neighborhood (3×3)
            let m_tl = sample(metrics, w, h, ix - 1, iy - 1);
            let m_t  = sample(metrics, w, h, ix,     iy - 1);
            let m_l  = sample(metrics, w, h, ix - 1, iy);
            let m_c  = sample(metrics, w, h, ix,     iy);
            let m_r  = sample(metrics, w, h, ix + 1, iy);
            let m_bl = sample(metrics, w, h, ix - 1, iy + 1);
            let m_b  = sample(metrics, w, h, ix,     iy + 1);
            let m_br = sample(metrics, w, h, ix + 1, iy + 1);

            // Weight neighborhood (3×3 + diagonals)
            let s_tl = sample(weights, w, h, ix - 1, iy - 1);
            let s_t  = sample(weights, w, h, ix,     iy - 1);
            let s_tr = sample(weights, w, h, ix + 1, iy - 1);
            let s_l  = sample(weights, w, h, ix - 1, iy);
            let s_c  = sample(weights, w, h, ix,     iy);
            let s_r  = sample(weights, w, h, ix + 1, iy);
            let s_bl = sample(weights, w, h, ix - 1, iy + 1);
            let s_b  = sample(weights, w, h, ix,     iy + 1);
            let s_br = sample(weights, w, h, ix + 1, iy + 1);

            // Four junctions around center pixel, each receiving one corner
            // from each of 4 adjacent pixels.
            // Junction 0 (top-left of center): corners tl.2, t.3, c.0, l.1
            let jstr0 = [s_tl[2], s_t[3], s_c[0], s_l[1]];
            let jdom0 = dominance(
                [s_tl[1], s_tl[2], s_tl[3]],
                [s_t[2], s_t[3], s_t[0]],
                [s_c[3], s_c[0], s_c[1]],
                [s_l[0], s_l[1], s_l[2]],
            );
            // Junction 1 (top-right): corners t.2, tr.3, r.0, c.1
            let jstr1 = [s_t[2], s_tr[3], s_r[0], s_c[1]];
            let jdom1 = dominance(
                [s_t[1], s_t[2], s_t[3]],
                [s_tr[2], s_tr[3], s_tr[0]],
                [s_r[3], s_r[0], s_r[1]],
                [s_c[0], s_c[1], s_c[2]],
            );
            // Junction 2 (bottom-right): corners c.2, r.3, br.0, b.1
            let jstr2 = [s_c[2], s_r[3], s_br[0], s_b[1]];
            let jdom2 = dominance(
                [s_c[1], s_c[2], s_c[3]],
                [s_r[2], s_r[3], s_r[0]],
                [s_br[3], s_br[0], s_br[1]],
                [s_b[0], s_b[1], s_b[2]],
            );
            // Junction 3 (bottom-left): corners l.2, c.3, b.0, bl.1
            let jstr3 = [s_l[2], s_c[3], s_b[0], s_bl[1]];
            let jdom3 = dominance(
                [s_l[1], s_l[2], s_l[3]],
                [s_c[2], s_c[3], s_c[0]],
                [s_b[3], s_b[0], s_b[1]],
                [s_bl[0], s_bl[1], s_bl[2]],
            );

            // Majority vote for each junction
            let v0 = majority_vote(jdom0);
            let v1 = majority_vote(jdom1);
            let v2 = majority_vote(jdom2);
            let v3 = majority_vote(jdom3);

            // Inject strength: a corner wins if it dominates, or if it has
            // strength and the two flanking corners are inactive.
            let inject = |vote: [f32; 4], stren: [f32; 4], ci: usize| -> f32 {
                let left = (ci + 1) & 3;
                let right = (ci + 3) & 3;
                let opposite = (ci + 2) & 3;
                (vote[ci] + inv(vote[left]) * inv(vote[right]) * gt(stren[ci], 0.0)
                    * (vote[opposite] + gt(stren[opposite] + stren[ci], stren[left] + stren[right])))
                .min(1.0)
            };

            let mut active = [
                inject(v0, jstr0, 2),
                inject(v1, jstr1, 3),
                inject(v2, jstr2, 0),
                inject(v3, jstr3, 1),
            ];

            // End-of-line suppression: isolated corners without support are killed
            let adj = [v0[2], v1[3], v2[0], v3[1]];
            let prev_active = active;
            for i in 0..4 {
                let wrap_prev = prev_active[(i + 3) & 3];
                let wrap_next = prev_active[(i + 1) & 3];
                active[i] = (active[i] * (adj[i] + inv(wrap_prev * wrap_next))).min(1.0);
            }

            // Edge classification using metric data
            let jcond = [
                junction_condition([m_l[2], m_c[0]], [m_l[3], m_c[1]], [m_tl[3], m_l[1]]),
                junction_condition([m_r[0], m_c[2]], [m_c[3], m_c[1]], [m_t[3], m_r[1]]),
                junction_condition([m_b[2], m_br[0]], [m_c[3], m_b[1]], [m_b[3], m_br[1]]),
                junction_condition([m_b[0], m_bl[2]], [m_l[3], m_b[1]], [m_bl[3], m_bl[1]]),
            ];

            let h_edge = [
                m_l[3].min(m_tl[3]),
                m_c[3].min(m_t[3]),
                m_c[3].min(m_b[3]),
                m_l[3].min(m_bl[3]),
            ];
            let v_edge = [
                m_c[1].min(m_l[1]),
                m_c[1].min(m_r[1]),
                m_b[1].min(m_br[1]),
                m_b[1].min(m_bl[1]),
            ];

            let h_extra = [m_l[3], m_c[3], m_c[3], m_l[3]];
            let v_extra = [m_c[1], m_c[1], m_b[1], m_b[1]];

            let mut packed = [0.0f32; 4];
            for i in 0..4 {
                let orient = gt(h_edge[i] + h_extra[i], v_edge[i] + v_extra[i]);
                let horiz = lt(h_edge[i], v_edge[i]) * jcond[i];
                let vert = gt(h_edge[i], v_edge[i]) * jcond[i];
                packed[i] = (active[i] + 2.0 * horiz + 4.0 * vert + 8.0 * orient) / 15.0;
            }

            result[py * w + px] = packed;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Stage 4: Multi-level edge tracing
// ---------------------------------------------------------------------------

/// Unpack the four boolean flags from a packed junction value.
struct JunctionFlags {
    corner: [bool; 4],
    horiz: [bool; 4],
    vert: [bool; 4],
    orient: [bool; 4],
}

fn decode_flags(val: [f32; 4]) -> JunctionFlags {
    let bit = |v: f32, scale: f32, offset: f32| -> bool {
        (v * scale + offset).floor() as i32 & 1 != 0
    };
    JunctionFlags {
        corner: [bit(val[0], 15.0, 0.5), bit(val[1], 15.0, 0.5), bit(val[2], 15.0, 0.5), bit(val[3], 15.0, 0.5)],
        horiz:  [bit(val[0], 7.5, 0.25), bit(val[1], 7.5, 0.25), bit(val[2], 7.5, 0.25), bit(val[3], 7.5, 0.25)],
        vert:   [bit(val[0], 3.75, 0.125), bit(val[1], 3.75, 0.125), bit(val[2], 3.75, 0.125), bit(val[3], 3.75, 0.125)],
        orient: [bit(val[0], 1.875, 0.0625), bit(val[1], 1.875, 0.0625), bit(val[2], 1.875, 0.0625), bit(val[3], 1.875, 0.0625)],
    }
}

/// Subpixel neighbor indices.
/// 0 = center, 1-4 = left/left2/right/right2, 5-8 = up/up2/down/down2.
const CENTER: f32 = 0.0;
const LEFT: f32 = 1.0;
const LEFT2: f32 = 2.0;
const RIGHT: f32 = 3.0;
const RIGHT2: f32 = 4.0;
const UP: f32 = 5.0;
const UP2: f32 = 6.0;
const DOWN: f32 = 7.0;
const DOWN2: f32 = 8.0;

/// Trace edges up to 6 levels and assign subpixel neighbor tags.
///
/// Each pixel gets 4 corner tags and 4 mid-edge tags. The tag indicates
/// which source pixel should fill that subpixel position in the 3× output.
fn trace_edges(junctions: &[[f32; 4]], w: usize, h: usize) -> Vec<[f32; 4]> {
    let mut tags = vec![[0.0f32; 4]; w * h];

    for py in 0..h {
        for px in 0..w {
            let ix = px as isize;
            let iy = py as isize;

            // Decode flags for center and all cardinal neighbors out to distance 3
            let e  = decode_flags(sample(junctions, w, h, ix, iy));
            let dl = decode_flags(sample(junctions, w, h, ix - 1, iy));
            let dl2 = decode_flags(sample(junctions, w, h, ix - 2, iy));
            let dl3 = decode_flags(sample(junctions, w, h, ix - 3, iy));
            let dr = decode_flags(sample(junctions, w, h, ix + 1, iy));
            let dr2 = decode_flags(sample(junctions, w, h, ix + 2, iy));
            let dr3 = decode_flags(sample(junctions, w, h, ix + 3, iy));
            let du = decode_flags(sample(junctions, w, h, ix, iy - 1));
            let du2 = decode_flags(sample(junctions, w, h, ix, iy - 2));
            let du3 = decode_flags(sample(junctions, w, h, ix, iy - 3));
            let dd = decode_flags(sample(junctions, w, h, ix, iy + 1));
            let dd2 = decode_flags(sample(junctions, w, h, ix, iy + 2));
            let dd3 = decode_flags(sample(junctions, w, h, ix, iy + 3));

            // Level 1: corner exists (SFX_SCN=1, so adjacent support always satisfied)
            let l1 = [
                e.corner[0],
                e.corner[1],
                e.corner[2],
                e.corner[3],
            ];

            // Level 2: mid-edge between two corners (horizontal or vertical)
            let l2 = [
                [e.corner[0] && e.horiz[1] && dl.corner[2], e.corner[1] && e.horiz[0] && dr.corner[3]],
                [e.corner[1] && e.vert[2] && du.corner[3],  e.corner[2] && e.vert[1] && dd.corner[0]],
                [e.corner[3] && e.horiz[2] && dl.corner[1], e.corner[2] && e.horiz[3] && dr.corner[0]],
                [e.corner[0] && e.vert[3] && du.corner[2],  e.corner[3] && e.vert[0] && dd.corner[1]],
            ];

            // Level 3: extended corner — level 2 confirmed by edge continuity
            let l3 = [
                [l2[0][1] && dl.horiz[1] && dl.horiz[0] && dr.horiz[2],
                 l2[3][1] && du.vert[3] && du.vert[0] && dd.vert[2]],
                [l2[0][0] && dr.horiz[0] && dr.horiz[1] && dl.horiz[3],
                 l2[1][1] && du.vert[2] && du.vert[1] && dd.vert[3]],
                [l2[2][0] && dr.horiz[3] && dr.horiz[2] && dl.horiz[0],
                 l2[1][0] && dd.vert[1] && dd.vert[2] && du.vert[0]],
                [l2[2][1] && dl.horiz[2] && dl.horiz[3] && dr.horiz[1],
                 l2[3][0] && dd.vert[0] && dd.vert[3] && du.vert[1]],
            ];

            // Level 4: 4-pixel run — corner + full edge chain + 2-away corner
            let l4 = [
                [dl.corner[0] && dl.horiz[1] && e.horiz[0] && e.horiz[1] && dr.horiz[0] && dr.horiz[1] && dl2.corner[2] && dl2.horiz[3],
                 du.corner[0] && du.vert[3] && e.vert[0] && e.vert[3] && dd.vert[0] && dd.vert[3] && du2.corner[2] && du2.vert[1]],
                [dr.corner[1] && dr.horiz[0] && e.horiz[1] && e.horiz[0] && dl.horiz[1] && dl.horiz[0] && dr2.corner[3] && dr2.horiz[2],
                 du.corner[1] && du.vert[2] && e.vert[1] && e.vert[2] && dd.vert[1] && dd.vert[2] && du2.corner[3] && du2.vert[0]],
                [dr.corner[2] && dr.horiz[3] && e.horiz[2] && e.horiz[3] && dl.horiz[2] && dl.horiz[3] && dr2.corner[0] && dr2.horiz[1],
                 dd.corner[2] && dd.vert[1] && e.vert[2] && e.vert[1] && du.vert[2] && du.vert[1] && dd2.corner[0] && dd2.vert[3]],
                [dl.corner[3] && dl.horiz[2] && e.horiz[3] && e.horiz[2] && dr.horiz[3] && dr.horiz[2] && dl2.corner[1] && dl2.horiz[0],
                 dd.corner[3] && dd.vert[0] && e.vert[3] && e.vert[0] && du.vert[3] && du.vert[0] && dd2.corner[1] && dd2.vert[2]],
            ];

            // Level 5: mid of 4-pixel run — level 4 extended with 3-pixel-out checks
            let l5 = [
                [l4[0][0] && dr2.horiz[0] && dr2.horiz[1] && dl3.horiz[2] && dl3.horiz[3],
                 l4[1][0] && dl2.horiz[1] && dl2.horiz[0] && dr3.horiz[3] && dr3.horiz[2]],
                [l4[1][1] && dd2.vert[1] && dd2.vert[2] && du3.vert[3] && du3.vert[0],
                 l4[2][1] && du2.vert[2] && du2.vert[1] && dd3.vert[0] && dd3.vert[3]],
                [l4[3][0] && dr2.horiz[3] && dr2.horiz[2] && dl3.horiz[1] && dl3.horiz[0],
                 l4[2][0] && dl2.horiz[2] && dl2.horiz[3] && dr3.horiz[0] && dr3.horiz[1]],
                [l4[0][1] && dd2.vert[0] && dd2.vert[3] && du3.vert[2] && du3.vert[1],
                 l4[3][1] && du2.vert[3] && du2.vert[0] && dd3.vert[1] && dd3.vert[2]],
            ];

            // Level 6: 6-pixel run
            let l6 = [
                [l5[0][1] && dl3.horiz[1] && dl3.horiz[0], l5[3][1] && du3.vert[3] && du3.vert[0]],
                [l5[0][0] && dr3.horiz[0] && dr3.horiz[1], l5[1][1] && du3.vert[2] && du3.vert[1]],
                [l5[2][0] && dr3.horiz[3] && dr3.horiz[2], l5[1][0] && dd3.vert[1] && dd3.vert[2]],
                [l5[2][1] && dl3.horiz[2] && dl3.horiz[3], l5[3][0] && dd3.vert[0] && dd3.vert[3]],
            ];

            // Assign corner subpixel tags based on highest detected level
            let corner_tag = |i: usize| -> f32 {
                let eo = e.orient;
                let lo = dl.orient;
                let ro = dr.orient;
                let uo = du.orient;
                let ddo = dd.orient;
                match i {
                    0 => if (l1[0] && eo[0]) || (l3[0][0] && eo[1]) || (l4[0][0] && lo[0]) || (l6[0][0] && ro[1]) { UP }
                         else if l1[0] || (l3[0][1] && !eo[3]) || (l4[0][1] && !uo[0]) || (l6[0][1] && !ddo[3]) { LEFT }
                         else if l3[0][0] { RIGHT } else if l3[0][1] { DOWN }
                         else if l4[0][0] { LEFT2 } else if l4[0][1] { UP2 }
                         else if l6[0][0] { RIGHT2 } else if l6[0][1] { DOWN2 } else { CENTER },
                    1 => if (l1[1] && eo[1]) || (l3[1][0] && eo[0]) || (l4[1][0] && ro[1]) || (l6[1][0] && lo[0]) { UP }
                         else if l1[1] || (l3[1][1] && !eo[2]) || (l4[1][1] && !uo[1]) || (l6[1][1] && !ddo[2]) { RIGHT }
                         else if l3[1][0] { LEFT } else if l3[1][1] { DOWN }
                         else if l4[1][0] { RIGHT2 } else if l4[1][1] { UP2 }
                         else if l6[1][0] { LEFT2 } else if l6[1][1] { DOWN2 } else { CENTER },
                    2 => if (l1[2] && eo[2]) || (l3[2][0] && eo[3]) || (l4[2][0] && ro[2]) || (l6[2][0] && lo[3]) { DOWN }
                         else if l1[2] || (l3[2][1] && !eo[1]) || (l4[2][1] && !ddo[2]) || (l6[2][1] && !uo[1]) { RIGHT }
                         else if l3[2][0] { LEFT } else if l3[2][1] { UP }
                         else if l4[2][0] { RIGHT2 } else if l4[2][1] { DOWN2 }
                         else if l6[2][0] { LEFT2 } else if l6[2][1] { UP2 } else { CENTER },
                    3 => if (l1[3] && eo[3]) || (l3[3][0] && eo[2]) || (l4[3][0] && lo[3]) || (l6[3][0] && ro[2]) { DOWN }
                         else if l1[3] || (l3[3][1] && !eo[0]) || (l4[3][1] && !ddo[3]) || (l6[3][1] && !uo[0]) { LEFT }
                         else if l3[3][0] { RIGHT } else if l3[3][1] { UP }
                         else if l4[3][0] { LEFT2 } else if l4[3][1] { DOWN2 }
                         else if l6[3][0] { RIGHT2 } else if l6[3][1] { UP2 } else { CENTER },
                    _ => CENTER,
                }
            };

            // Assign mid-edge subpixel tags
            let mid_tag = |i: usize| -> f32 {
                let eo = e.orient;
                let lo = dl.orient;
                let ro = dr.orient;
                let uo = du.orient;
                let ddo = dd.orient;
                match i {
                    0 => if (l2[0][0] && eo[0]) || (l2[0][1] && eo[1]) || (l5[0][0] && lo[0]) || (l5[0][1] && ro[1]) { UP }
                         else if l2[0][0] { LEFT } else if l2[0][1] { RIGHT }
                         else if l5[0][0] { LEFT2 } else if l5[0][1] { RIGHT2 }
                         else if e.corner[0] && dl.corner[2] && e.corner[1] && dr.corner[3] {
                             if eo[0] { if eo[1] { UP } else { RIGHT } } else { LEFT }
                         } else { CENTER },
                    1 => if (l2[1][0] && !eo[1]) || (l2[1][1] && !eo[2]) || (l5[1][0] && !uo[1]) || (l5[1][1] && !ddo[2]) { RIGHT }
                         else if l2[1][0] { UP } else if l2[1][1] { DOWN }
                         else if l5[1][0] { UP2 } else if l5[1][1] { DOWN2 }
                         else if e.corner[1] && du.corner[3] && e.corner[2] && dd.corner[0] {
                             if !eo[1] { if !eo[2] { RIGHT } else { DOWN } } else { UP }
                         } else { CENTER },
                    2 => if (l2[2][0] && eo[3]) || (l2[2][1] && eo[2]) || (l5[2][0] && lo[3]) || (l5[2][1] && ro[2]) { DOWN }
                         else if l2[2][0] { LEFT } else if l2[2][1] { RIGHT }
                         else if l5[2][0] { LEFT2 } else if l5[2][1] { RIGHT2 }
                         else if e.corner[2] && dr.corner[0] && e.corner[3] && dl.corner[1] {
                             if eo[2] { if eo[3] { DOWN } else { LEFT } } else { RIGHT }
                         } else { CENTER },
                    3 => if (l2[3][0] && !eo[0]) || (l2[3][1] && !eo[3]) || (l5[3][0] && !uo[0]) || (l5[3][1] && !ddo[3]) { LEFT }
                         else if l2[3][0] { UP } else if l2[3][1] { DOWN }
                         else if l5[3][0] { UP2 } else if l5[3][1] { DOWN2 }
                         else if e.corner[3] && dd.corner[1] && e.corner[0] && du.corner[2] {
                             if !eo[3] { if !eo[0] { LEFT } else { UP } } else { DOWN }
                         } else { CENTER },
                    _ => CENTER,
                }
            };

            let ct = [corner_tag(0), corner_tag(1), corner_tag(2), corner_tag(3)];
            let mt = [mid_tag(0), mid_tag(1), mid_tag(2), mid_tag(3)];

            tags[py * w + px] = [
                (ct[0] + 9.0 * mt[0]) / 80.0,
                (ct[1] + 9.0 * mt[1]) / 80.0,
                (ct[2] + 9.0 * mt[2]) / 80.0,
                (ct[3] + 9.0 * mt[3]) / 80.0,
            ];
        }
    }
    tags
}

// ---------------------------------------------------------------------------
// Stage 5: 3× block assembly
// ---------------------------------------------------------------------------

/// Decode a corner tag from the packed representation.
#[inline(always)]
fn unpack_corner_tag(v: f32) -> u8 {
    ((v * 80.0 + 0.5).floor() as i32 % 9) as u8
}

/// Decode a mid-edge tag from the packed representation.
#[inline(always)]
fn unpack_mid_tag(v: f32) -> u8 {
    ((v * 80.0 / 9.0 + 0.5 / 9.0).floor() as i32 % 9) as u8
}

/// Map a subpixel tag to a source pixel offset.
#[inline(always)]
fn neighbor_offset(tag: u8) -> (isize, isize) {
    match tag {
        1 => (-1,  0), // left
        2 => (-2,  0), // left ×2
        3 => ( 1,  0), // right
        4 => ( 2,  0), // right ×2
        5 => ( 0, -1), // up
        6 => ( 0, -2), // up ×2
        7 => ( 0,  1), // down
        8 => ( 0,  2), // down ×2
        _ => ( 0,  0), // center
    }
}

/// Assemble the 3× output image from edge tags and original pixels.
fn assemble_3x(edge_tags: &[[f32; 4]], src: &[u32], w: usize, h: usize) -> Vec<u32> {
    let ow = w * 3;
    let oh = h * 3;
    let mut out = vec![0u32; ow * oh];

    for sy in 0..h {
        for sx in 0..w {
            let packed = edge_tags[sy * w + sx];
            let ct = [
                unpack_corner_tag(packed[0]), unpack_corner_tag(packed[1]),
                unpack_corner_tag(packed[2]), unpack_corner_tag(packed[3]),
            ];
            let mt = [
                unpack_mid_tag(packed[0]), unpack_mid_tag(packed[1]),
                unpack_mid_tag(packed[2]), unpack_mid_tag(packed[3]),
            ];

            let ox = sx * 3;
            let oy = sy * 3;
            let cx = sx as isize;
            let cy = sy as isize;

            // 3×3 block: corners at even positions, mids at odd, center at (1,1)
            let block: [(usize, usize, u8); 9] = [
                (0, 0, ct[0]), (1, 0, mt[0]), (2, 0, ct[1]),
                (0, 1, mt[3]), (1, 1, 0),     (2, 1, mt[1]),
                (0, 2, ct[3]), (1, 2, mt[2]), (2, 2, ct[2]),
            ];

            for &(bx, by, tag) in &block {
                let (dx, dy) = neighbor_offset(tag);
                out[(oy + by) * ow + (ox + bx)] = get(src, w, h, cx + dx, cy + dy);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// ScaleFX 3× scaling.
pub fn scale(src: &[u32], w: usize, h: usize) -> Vec<u32> {
    let metrics = compute_metrics(src, w, h);
    let weights = compute_corner_weights(&metrics, w, h);
    let junctions = arbitrate_junctions(&metrics, &weights, w, h);
    let edge_tags = trace_edges(&junctions, w, h);
    assemble_3x(&edge_tags, src, w, h)
}

/// ScaleFX 9× scaling (two chained 3× passes).
pub fn scale9x(src: &[u32], w: usize, h: usize) -> Vec<u32> {
    let intermediate = scale(src, w, h);
    scale(&intermediate, w * 3, h * 3)
}
