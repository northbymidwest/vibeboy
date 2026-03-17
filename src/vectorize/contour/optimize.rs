//! Gradient descent optimizer, energy functions, corner detection, and B-spline fitting
//! (Paper Section 3.4).

use super::{NodeId, PathSegment};
use super::FxHashMap;
use crate::vectorize::voronoi::Point;
use std::collections::HashSet;

// --- Loop optimization (Paper Section 3.4) ---

const OPT_ITERATIONS: usize = 1;
const MAX_MOVE: f64 = 0.25;
const CURVATURE_INTERVALS: usize = 3;

/// Positional energy: (2.5 x ||delta||)^4 = 2.5^4 x ||delta||^4 ~ 39.06 x ||delta||^4.
///
/// The 2.5 scaling factor matches the reference implementation
/// (Depixelizing Pixel Art on GPUs, OptimizeEnergy.vert line 84).
/// The paper specifies ||delta||^4 without the scaling, but the reference
/// uses 2.5x which keeps nodes much closer to their original positions,
/// preventing over-smoothing of intentional pixel-art features.
const POSITIONAL_SCALE: f64 = 2.5;

/// Optimize boundary loop paths directly using gradient descent.
/// Works on full closed loops instead of short chains, so the optimizer
/// sees the complete contour shape for each color region.
/// Junction nodes (valence >= 3) are fixed.
pub(super) fn optimize_boundary_loops(
    all_loops: &[(Vec<NodeId>, u32)],
    positions: &mut FxHashMap<NodeId, Point>,
    junctions: &HashSet<NodeId>,
) {
    // Build a contiguous points array per loop for fast energy evaluation.
    // Map loop nodes to indices in the array; junction nodes are pinned.
    for (node_loop, _) in all_loops {
        let n = node_loop.len();
        if n < 4 { continue; }

        let mut pts: Vec<Point> = node_loop.iter()
            .map(|nd| *positions.get(nd).unwrap())
            .collect();
        let orig: Vec<Point> = pts.clone();
        // Paper Section 3.4, Figure 7: detect corners via x4 grid template matching.
        // Corner nodes are NOT pinned -- they can still move during optimization.
        // Only the B-spline spans touching corners are excluded from curvature energy.
        let corners = detect_corners_from_nodes(node_loop, true);

        let pinned: Vec<bool> = node_loop.iter()
            .map(|nd| junctions.contains(nd))
            .collect();

        for _iter in 0..OPT_ITERATIONS {
            for i in 0..n {
                if pinned[i] { continue; }

                let current = pts[i];
                let e0 = local_energy(&pts, &orig, &corners, i, n, true);
                if e0 < 1e-12 { continue; }

                // Analytic gradient
                let (gx, gy) = analytic_gradient(&pts, &orig, &corners, i, n);

                let grad_len = (gx * gx + gy * gy).sqrt();
                if grad_len < 1e-12 { continue; }

                let step = (e0 / grad_len).min(MAX_MOVE);
                let candidate = Point::new(
                    current.x - step * gx / grad_len,
                    current.y - step * gy / grad_len,
                );
                pts[i] = candidate;
                let e_new = local_energy(&pts, &orig, &corners, i, n, true);
                if e_new >= e0 {
                    pts[i] = current;
                }
            }
        }

        // Write back optimized positions
        for (i, nd) in node_loop.iter().enumerate() {
            positions.insert(*nd, pts[i]);
        }
    }
}

// --- B-spline fitting ---

/// Convert a closed loop of control points to quadratic B-spline segments.
pub(super) fn bspline_closed(ctrl: &[Point]) -> Vec<PathSegment> {
    let n = ctrl.len();
    if n < 3 {
        return line_segments(ctrl);
    }
    let mut segments = Vec::with_capacity(n);
    for i in 0..n {
        let p0 = ctrl[i];
        let p1 = ctrl[(i + 1) % n];
        let p2 = ctrl[(i + 2) % n];
        let q0 = Point::new((p0.x + p1.x) * 0.5, (p0.y + p1.y) * 0.5);
        let q1 = Point::new((p1.x + p2.x) * 0.5, (p1.y + p2.y) * 0.5);
        segments.push(PathSegment::QuadBezier(q0, p1, q1));
    }
    segments
}

/// Convert an open path of control points to quadratic B-spline segments.
pub(super) fn bspline_open(ctrl: &[Point]) -> Vec<PathSegment> {
    let n = ctrl.len();
    if n < 3 {
        return line_segments(ctrl);
    }

    let mut segments = Vec::new();

    let mid01 = Point::new(
        (ctrl[0].x + ctrl[1].x) * 0.5,
        (ctrl[0].y + ctrl[1].y) * 0.5,
    );
    segments.push(PathSegment::QuadBezier(ctrl[0], ctrl[0], mid01));

    for i in 0..n - 2 {
        let p0 = ctrl[i];
        let p1 = ctrl[i + 1];
        let p2 = ctrl[i + 2];
        let q0 = Point::new((p0.x + p1.x) * 0.5, (p0.y + p1.y) * 0.5);
        let q1 = Point::new((p1.x + p2.x) * 0.5, (p1.y + p2.y) * 0.5);
        segments.push(PathSegment::QuadBezier(q0, p1, q1));
    }

    let mid_last = Point::new(
        (ctrl[n - 2].x + ctrl[n - 1].x) * 0.5,
        (ctrl[n - 2].y + ctrl[n - 1].y) * 0.5,
    );
    segments.push(PathSegment::QuadBezier(mid_last, ctrl[n - 1], ctrl[n - 1]));

    segments
}

fn line_segments(pts: &[Point]) -> Vec<PathSegment> {
    pts.windows(2)
        .map(|w| PathSegment::Line(w[0], w[1]))
        .collect()
}

// --- Section 3.4: B-spline optimization ---

/// Detect corner patterns using Kopf-Lischinski template matching (Section 3.4, Figure 7).
///
/// On the x4 quantized grid, sharp features take on a finite set of patterns
/// (the paper's Figure 7, including all rotations and reflections). We detect
/// these by checking if the turn angle at each node >= 60 deg using exact integer
/// arithmetic on the x4 coordinates. This is equivalent to the paper's pattern
/// enumeration since all sharp patterns on the quantized grid have angles >= 60 deg.
///
/// Unlike the previous implementation which pinned corner nodes entirely,
/// the paper only excludes B-spline spans near corners from the curvature
/// integral -- corner nodes can still move during optimization.
pub(super) fn detect_corners_from_nodes(nodes: &[NodeId], is_closed: bool) -> Vec<bool> {
    let n = nodes.len();
    let mut is_corner = vec![false; n];

    if !is_closed {
        if n > 0 { is_corner[0] = true; }
        if n > 1 { is_corner[n - 1] = true; }
    }

    let range_start = if is_closed { 0 } else { 1 };
    let range_end = if is_closed { n } else { n - 1 };

    for i in range_start..range_end {
        let prev = if is_closed { nodes[(i + n - 1) % n] } else { nodes[i - 1] };
        let curr = nodes[i];
        let next = if is_closed { nodes[(i + 1) % n] } else { nodes[i + 1] };

        // Edge vectors in x4 integer coordinates
        let d1x = (curr.x4 - prev.x4) as i64;
        let d1y = (curr.y4 - prev.y4) as i64;
        let d2x = (next.x4 - curr.x4) as i64;
        let d2y = (next.y4 - curr.y4) as i64;

        let dot = d1x * d2x + d1y * d2y;

        if dot <= 0 {
            // Turn angle >= 90 deg -- always a corner
            is_corner[i] = true;
        } else {
            // Check if turn angle >= 60 deg using integer arithmetic:
            // cos(angle) <= 0.5  <->  4 * dot^2 <= |d1|^2 * |d2|^2
            let len1_sq = d1x * d1x + d1y * d1y;
            let len2_sq = d2x * d2x + d2y * d2y;
            if 4 * dot * dot <= len1_sq * len2_sq {
                is_corner[i] = true;
            }
        }
    }

    is_corner
}



#[inline(always)]
/// Analytic gradient of the total energy at node `idx`.
/// Returns (dE/dx, dE/dy).
///
/// Positional energy: E_pos = (s^2 * d^2)^2
///   grad(E_pos) = 4 * s^4 * d^2 * (p - p_orig)
///
/// Curvature energy per span (p0, p1, p2): E_curv ~ ||p0 - 2p1 + p2||^2
///   grad(E_curv) w.r.t. p1 (center) = 2*(4p1 - 2p0 - 2p2)
///   grad(E_curv) w.r.t. p0 (start)  = 2*(p0 - 2p1 + p2)
///   grad(E_curv) w.r.t. p2 (end)    = 2*(p0 - 2p1 + p2)
fn analytic_gradient(
    points: &[Point], orig: &[Point], corners: &[bool],
    idx: usize, n: usize,
) -> (f64, f64) {
    let p = points[idx];
    let o = orig[idx];

    // Positional gradient
    let dx = p.x - o.x;
    let dy = p.y - o.y;
    let d_sq = dx * dx + dy * dy;
    let s2 = POSITIONAL_SCALE * POSITIONAL_SCALE;
    let mut gx = 4.0 * s2 * s2 * d_sq * dx;
    let mut gy = 4.0 * s2 * s2 * d_sq * dy;

    // Curvature gradient: node participates in up to 3 spans
    for offset in 0..3i64 {
        let span_start = ((idx as i64 - 2 + offset) % n as i64 + n as i64) as usize % n;
        if span_start + 2 >= n { continue; }

        let i0 = span_start % n;
        let i1 = (span_start + 1) % n;
        let i2 = (span_start + 2) % n;

        if i1 >= n || i2 >= n { continue; }
        if corners[i0] || corners[i1] || corners[i2] { continue; }

        let p0 = points[i0];
        let p1 = points[i1];
        let p2 = points[i2];

        // Second difference: dd = p0 - 2*p1 + p2
        let ddx = p0.x - 2.0 * p1.x + p2.x;
        let ddy = p0.y - 2.0 * p1.y + p2.y;

        if i1 == idx {
            // This node is the center of the span
            gx += 2.0 * (4.0 * p1.x - 2.0 * p0.x - 2.0 * p2.x);
            gy += 2.0 * (4.0 * p1.y - 2.0 * p0.y - 2.0 * p2.y);
        } else {
            // This node is p0 or p2
            gx += 2.0 * ddx;
            gy += 2.0 * ddy;
        }
    }

    (gx, gy)
}

fn local_energy(
    points: &[Point], orig: &[Point], corners: &[bool],
    idx: usize, n: usize, is_closed: bool,
) -> f64 {
    curvature_energy(points, corners, idx, n, is_closed)
        + positional_energy(points, orig, idx)
}

#[inline(always)]
fn positional_energy(points: &[Point], orig: &[Point], idx: usize) -> f64 {
    let dx = points[idx].x - orig[idx].x;
    let dy = points[idx].y - orig[idx].y;
    let dist_sq = dx * dx + dy * dy;
    let scaled_dist_sq = POSITIONAL_SCALE * POSITIONAL_SCALE * dist_sq;
    scaled_dist_sq * scaled_dist_sq
}

#[inline(always)]
fn curvature_energy(
    points: &[Point], corners: &[bool], idx: usize, n: usize, is_closed: bool,
) -> f64 {
    let mut energy = 0.0;

    for offset in 0..3i64 {
        let span_start = ((idx as i64 - 2 + offset) % n as i64 + n as i64) as usize % n;

        if !is_closed && (span_start + 2 >= n) { continue; }

        let i0 = span_start % n;
        let i1 = if is_closed { (span_start + 1) % n } else { span_start + 1 };
        let i2 = if is_closed { (span_start + 2) % n } else { span_start + 2 };

        if i1 >= n || i2 >= n { continue; }

        if corners[i0] || corners[i1] || corners[i2] { continue; }

        energy += integrate_span_curvature(points[i0], points[i1], points[i2]);
    }

    energy
}

/// Integrate kappa^2 over one quadratic B-spline span.
///
/// NOTE: The paper (Equation 3) defines smoothness energy as integral(|kappa(s)|) ds
/// (absolute curvature integrated over arc length). We use integral(kappa^2) instead
/// because it penalizes curvature more aggressively per iteration,
/// producing visually smooth results with just 1 optimization pass.
#[inline(always)]
fn integrate_span_curvature(p0: Point, p1: Point, p2: Point) -> f64 {
    let ddx = p0.x - 2.0 * p1.x + p2.x;
    let ddy = p0.y - 2.0 * p1.y + p2.y;
    let cross_sq_factor = ddx * ddx + ddy * ddy;
    if cross_sq_factor < 1e-20 { return 0.0; }

    let dt = 1.0 / CURVATURE_INTERVALS as f64;
    let mut result = (curvature_sq_at(p0, p1, p2, 0.0, ddx, ddy)
        + curvature_sq_at(p0, p1, p2, 1.0, ddx, ddy)) * 0.5;
    for i in 1..CURVATURE_INTERVALS {
        result += curvature_sq_at(p0, p1, p2, i as f64 * dt, ddx, ddy);
    }
    result * dt
}

/// Compute kappa^2(t) = (d' x d'')^2 / |d'|^6 for one sample point.
#[inline(always)]
fn curvature_sq_at(p0: Point, p1: Point, p2: Point, t: f64, ddx: f64, ddy: f64) -> f64 {
    let dx = (t - 1.0) * p0.x + (1.0 - 2.0 * t) * p1.x + t * p2.x;
    let dy = (t - 1.0) * p0.y + (1.0 - 2.0 * t) * p1.y + t * p2.y;

    let numer = dx * ddy - dy * ddx;
    let denom_sq = dx * dx + dy * dy;
    let denom = denom_sq * denom_sq.sqrt();
    if denom < 1e-12 { 0.0 } else { (numer * numer) / (denom * denom) }
}
