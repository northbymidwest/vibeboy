//! Boundary loop tracing using the planar face algorithm.

use super::{NodeId, PathSegment, VOID_COLOR};
use super::{FxHashMap, fx_hashmap, fx_hashmap_cap};
use super::optimize::bspline_closed;
use crate::vectorize::voronoi::Point;
use std::collections::HashSet;

/// Compare angles of two direction vectors without atan2.
/// Uses half-plane + cross-product for a total ordering equivalent to atan2.
#[inline]
fn angle_cmp(adx: i32, ady: i32, bdx: i32, bdy: i32) -> std::cmp::Ordering {
    // Upper half-plane: y > 0, or y == 0 && x > 0
    let ha = ady > 0 || (ady == 0 && adx > 0);
    let hb = bdy > 0 || (bdy == 0 && bdx > 0);
    if ha != hb {
        return if ha { std::cmp::Ordering::Less } else { std::cmp::Ordering::Greater };
    }
    // Same half-plane: use cross product (positive = a before b in CCW order)
    let cross = (adx as i64) * (bdy as i64) - (ady as i64) * (bdx as i64);
    // Reverse because positive cross means a is CCW from b (smaller angle)
    cross.cmp(&0).reverse()
}

/// Trace ALL boundary loops globally, returning (nodes, color) for each loop.
/// Uses the planar face algorithm on directed edges with known right-side colors.
pub(super) fn trace_all_boundary_loops(
    directed_edges: &[(NodeId, NodeId, u32)],
) -> Vec<(Vec<NodeId>, u32)> {
    let n_edges = directed_edges.len();
    if n_edges == 0 { return Vec::new(); }

    // Build outgoing adjacency as a flat array, avoiding HashMap.
    // Step 1: collect (source_node, dest, dx, dy, edge_index) and sort by
    // (source, angle) so all outgoing edges from each node are contiguous.
    let mut out_edges: Vec<(NodeId, NodeId, i32, i32, u32)> = Vec::with_capacity(n_edges);
    for (i, &(a, b, _)) in directed_edges.iter().enumerate() {
        out_edges.push((a, b, b.x4 - a.x4, b.y4 - a.y4, i as u32));
    }
    out_edges.sort_unstable_by(|a, b| {
        a.0.cmp(&b.0).then_with(|| angle_cmp(a.2, a.3, b.2, b.3))
    });

    // Step 2: build a range index: for each node, [start..end) into out_edges.
    // Since out_edges is sorted by source node, we find group boundaries.
    let mut node_ranges: FxHashMap<NodeId, (u32, u32)> = fx_hashmap_cap(n_edges / 2);
    let mut i = 0;
    while i < out_edges.len() {
        let node = out_edges[i].0;
        let start = i;
        while i < out_edges.len() && out_edges[i].0 == node { i += 1; }
        node_ranges.insert(node, (start as u32, i as u32));
    }

    // Step 3: build next-edge index array using planar face algorithm.
    let mut next_idx: Vec<u32> = vec![u32::MAX; n_edges];

    for (i, &(p, c, _)) in directed_edges.iter().enumerate() {
        let rdx = p.x4 - c.x4;
        let rdy = p.y4 - c.y4;

        if let Some(&(start, end)) = node_ranges.get(&c) {
            let slice = &out_edges[start as usize..end as usize];
            let pos = slice.partition_point(|e|
                angle_cmp(e.2, e.3, rdx, rdy) == std::cmp::Ordering::Less
            );
            let prev_idx = if pos == 0 { slice.len() - 1 } else { pos - 1 };
            next_idx[i] = slice[prev_idx].4;
        }
    }

    // Trace loops using index-based traversal
    let mut used = vec![false; n_edges];
    let mut loops = Vec::new();

    for start in 0..n_edges {
        if used[start] { continue; }

        let mut nodes = Vec::new();
        let mut cur = start;
        let mut closed = false;

        loop {
            if used[cur] {
                if cur == start { closed = true; }
                break;
            }
            used[cur] = true;
            nodes.push(directed_edges[cur].0);
            let ni = next_idx[cur] as usize;
            if ni >= n_edges { break; }
            cur = ni;
        }

        if nodes.len() >= 3 && closed {
            let color = directed_edges[start].2;
            loops.push((nodes, color));
        }
    }

    loops
}



/// Split B-splines at junction nodes (valence >= 3 in visible edge graph).
/// At T-junctions with corrected positions, the endpoint is adjusted so the
/// ending curve meets the continuing curve smoothly.
pub(super) fn boundary_loop_to_segments(
    nodes: &[NodeId],
    optimized: &FxHashMap<NodeId, Point>,
    junctions: &HashSet<NodeId>,
    tjunc_corrected: &FxHashMap<NodeId, Point>,
) -> Vec<PathSegment> {
    let n = nodes.len();
    let points: Vec<Point> = nodes
        .iter()
        .map(|nd| optimized.get(nd).copied().unwrap_or_else(|| nd.to_point()))
        .collect();

    if n < 3 {
        let mut segs = Vec::new();
        for i in 0..n {
            segs.push(PathSegment::Line(points[i], points[(i + 1) % n]));
        }
        return segs;
    }

    // Always use bspline_closed for the full loop rather than splitting
    // at junction nodes. Splitting creates very short spans (2-3 control
    // points) around small features that can only produce straight lines.
    // The reference implementation fits B-splines to complete loops,
    // allowing small features to render as smooth curves.
    return bspline_closed(&points);
}
