//! 2D Newton-Raphson optimizer, corner detection, and B-spline fitting
//! (Paper Section 3.4).

use super::{NodeId, PathSegment};
use super::FxHashMap;
use crate::vectorize::voronoi::Point;
use std::collections::HashSet;

// --- Loop optimization (Paper Section 3.4) ---

const POSITIONAL_SCALE: f64 = 2.5;
const NEWTON_ITER: usize = 3;

/// Optimize boundary loop paths using 2D Newton-Raphson.
/// Works on full closed loops instead of short chains, so the optimizer
/// sees the complete contour shape for each color region.
/// Junction nodes (valence >= 3) are fixed.
pub(super) fn optimize_boundary_loops(
    all_loops: &[(Vec<NodeId>, u32)],
    positions: &mut FxHashMap<NodeId, Point>,
    junctions: &HashSet<NodeId>,
) {
    let s4 = POSITIONAL_SCALE * POSITIONAL_SCALE * POSITIONAL_SCALE * POSITIONAL_SCALE;

    for (node_loop, _) in all_loops {
        let n = node_loop.len();
        if n < 4 { continue; }

        let mut pts: Vec<Point> = node_loop.iter()
            .map(|nd| *positions.get(nd).unwrap())
            .collect();
        let orig: Vec<Point> = pts.clone();
        let corners = detect_corners_from_nodes(node_loop, true);

        let pinned: Vec<bool> = node_loop.iter()
            .map(|nd| junctions.contains(nd))
            .collect();

        for i in 0..n {
            if pinned[i] { continue; }

            let prev = (i + n - 1) % n;
            let next = (i + 1) % n;

            // Skip spans containing corner nodes
            if corners[i] || corners[prev] || corners[next] { continue; }

            let n0 = pts[prev];
            let n1 = pts[next];
            let p_orig = orig[i];
            let mut p = pts[i];

            // 2D Newton-Raphson: minimize E = |n0-2p+n1|² + (2.5·||p-p_orig||)⁴
            // Gradient: ∇E = 4(2p-n0-n1) + 4·s⁴·||d||²·d
            // Hessian:   H = 8I + 4·s⁴·(2d⊗d + ||d||²·I)
            for _ in 0..NEWTON_ITER {
                let dx = p.x - p_orig.x;
                let dy = p.y - p_orig.y;
                let d2 = dx * dx + dy * dy;

                let gx = 4.0 * (2.0 * p.x - n0.x - n1.x) + 4.0 * s4 * d2 * dx;
                let gy = 4.0 * (2.0 * p.y - n0.y - n1.y) + 4.0 * s4 * d2 * dy;

                let h00 = 8.0 + 4.0 * s4 * (2.0 * dx * dx + d2);
                let h11 = 8.0 + 4.0 * s4 * (2.0 * dy * dy + d2);
                let h01 = 8.0 * s4 * dx * dy;

                let det = h00 * h11 - h01 * h01;
                if det.abs() < 1e-20 { break; }

                let inv_det = 1.0 / det;
                p.x -= (h11 * gx - h01 * gy) * inv_det;
                p.y -= (-h01 * gx + h00 * gy) * inv_det;
            }

            pts[i] = p;
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
    pts.array_windows::<2>()
        .map(|&[a, b]| PathSegment::Line(a, b))
        .collect()
}

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

