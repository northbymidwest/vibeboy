//! Boundary edge building, chain construction, and T-junction merging (Paper Section 3.3).

use super::{NodeId, CellEdge, PathSegment, VOID_COLOR, YUV_VISIBLE_EDGES};
use super::cells::InlineCell;
use super::{FxHashMap, fx_hashmap, fx_hashmap_cap};
use crate::vectorize::voronoi::Point;
use std::collections::HashSet;

/// Pack a NodeId into a u32 for fast hashing. Coordinates are offset by +2
/// so negative border values (-1) become non-negative.
#[inline(always)]
fn pack_node(n: NodeId, stride: u32) -> u32 {
    (n.y4 + 2) as u32 * stride + (n.x4 + 2) as u32
}

/// Pack a canonical edge (a <= b) into a u64 key.
#[inline(always)]
fn pack_edge(a: u32, b: u32) -> u64 {
    (a as u64) << 32 | b as u64
}

/// Check if two colors are dissimilar enough to form a visible edge.
/// When YUV_VISIBLE_EDGES is true, uses the same YUV threshold as the
/// similarity graph (Paper Section 3.2). When false, any color difference
/// creates a visible edge (more robust for games with dithering/gradients).
/// VOID_COLOR edges (image border) are always visible.
#[inline(always)]
pub(super) fn is_visible_edge(left: u32, right: u32) -> bool {
    if left == right { return false; }
    if left == VOID_COLOR || right == VOID_COLOR { return true; }
    if YUV_VISIBLE_EDGES.load(std::sync::atomic::Ordering::Relaxed) {
        !crate::vectorize::graph::similar(left, right)
    } else {
        true
    }
}

/// Threshold for adaptive pipeline: above this, skip B-spline and sort.
const ADAPTIVE_EDGE_THRESHOLD: usize = 12000;

pub(super) fn build_directed_boundary_edges(
    pixels: &[u32], w: usize, h: usize, all_cells: &[InlineCell], adaptive: bool,
) -> (Vec<CellEdge>, Vec<(NodeId, NodeId, u32)>) {
    // Hash-merge approach: for each cell edge, look up or insert into a HashMap
    // keyed by canonical (na, nb). This is O(n) expected vs O(n log n) for sort.
    //
    // Each entry stores (left_color, right_color). When we see a half-edge:
    //   - is_forward (pa <= pb): the pixel's color goes to right_color
    //   - !is_forward (pa > pb): the pixel's color goes to left_color

    // Value: (left_color, right_color, na, nb)
    let estimated = w * h * 3;
    let mut edge_map: FxHashMap<u64, (u32, u32, NodeId, NodeId)> = fx_hashmap_cap(estimated);
    let stride = (4 * w + 4) as u32;

    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let color = pixels[row + x];
            let cell = &all_cells[row + x];
            let n = cell.len as usize;
            if n < 3 { continue; }

            for i in 0..n {
                let pa = cell.nodes[i];
                let pb = cell.nodes[if i + 1 < n { i + 1 } else { 0 }];

                let (key, is_forward) = if pa <= pb {
                    (pack_edge(pack_node(pa, stride), pack_node(pb, stride)), true)
                } else {
                    (pack_edge(pack_node(pb, stride), pack_node(pa, stride)), false)
                };
                let (na, nb) = if pa <= pb { (pa, pb) } else { (pb, pa) };

                let entry = edge_map.entry(key).or_insert((VOID_COLOR, VOID_COLOR, na, nb));
                if is_forward {
                    entry.1 = color; // right_color
                } else {
                    entry.0 = color; // left_color
                }
            }
        }
    }

    // Emit boundary edges from the map
    let mut boundary_count = 0usize;
    for &(left, right, _, _) in edge_map.values() {
        if is_visible_edge(left, right) { boundary_count += 1; }
    }

    let adaptive = adaptive && boundary_count > ADAPTIVE_EDGE_THRESHOLD;
    let mut directed = Vec::with_capacity(boundary_count * 2);
    let mut visible = if adaptive { Vec::new() } else { Vec::with_capacity(boundary_count) };

    for &(left, right, na, nb) in edge_map.values() {
        if !is_visible_edge(left, right) { continue; }
        directed.push((na, nb, right));
        directed.push((nb, na, left));
        if !adaptive {
            visible.push(CellEdge { a: na, b: nb, left_color: left, right_color: right });
        }
    }

    // Note: directed edges are NOT pre-sorted. trace_all_boundary_loops
    // does its own sort internally, so pre-sorting here was redundant.

    (visible, directed)
}


/// Chain visible edges through valence-2 nodes into paths.
/// Returns each chain with its canonical color pair (cpair).
/// Computes cpair valence inline from adjacency list (avoids separate HashMap).
pub(super) fn chain_visible_edges(edges: &[CellEdge]) -> Vec<(Vec<NodeId>, (u32, u32))> {
    // Build adjacency: node -> list of (neighbor, cpair, edge_index).
    // Use a single HashMap; compute cpair valence by counting inline.
    let mut adj: FxHashMap<NodeId, Vec<(NodeId, (u32, u32), usize)>> =
        fx_hashmap_cap(edges.len());

    for (ei, e) in edges.iter().enumerate() {
        let cpair = if e.left_color <= e.right_color {
            (e.left_color, e.right_color)
        } else {
            (e.right_color, e.left_color)
        };
        adj.entry(e.a).or_default().push((e.b, cpair, ei));
        adj.entry(e.b).or_default().push((e.a, cpair, ei));
    }

    // Inline cpair valence: count how many edges of a given cpair connect to a node.
    #[inline]
    fn cpair_valence(neighbors: &[(NodeId, (u32, u32), usize)], cpair: (u32, u32)) -> u8 {
        let mut v = 0u8;
        for &(_, cp, _) in neighbors {
            if cp == cpair { v += 1; }
        }
        v
    }

    let mut visited = vec![false; edges.len()];
    let mut chains: Vec<(Vec<NodeId>, (u32, u32))> = Vec::new();

    let all_nodes: Vec<NodeId> = adj.keys().copied().collect();

    // Start chains from endpoints (cpair valence != 2)
    for start_node in &all_nodes {
        if let Some(neighbors) = adj.get(start_node) {
            for &(next, cpair, ei) in neighbors {
                if visited[ei] { continue; }
                if cpair_valence(neighbors, cpair) == 2 { continue; }

                let mut chain = vec![*start_node];
                let mut next_node = next;
                let mut cur_ei = ei;

                loop {
                    if visited[cur_ei] { break; }
                    visited[cur_ei] = true;
                    chain.push(next_node);

                    let nbrs = adj.get(&next_node).unwrap();
                    if cpair_valence(nbrs, cpair) != 2 { break; }

                    let mut found_next = false;
                    for &(nn, cp, nei) in nbrs {
                        if cp != cpair { continue; }
                        if !visited[nei] {
                            next_node = nn;
                            cur_ei = nei;
                            found_next = true;
                            break;
                        }
                    }
                    if !found_next { break; }
                }

                if chain.len() >= 2 {
                    chains.push((chain, cpair));
                }
            }
        }
    }

    // Handle closed loops (all nodes are valence-2)
    for start_node in &all_nodes {
        if let Some(neighbors) = adj.get(start_node) {
            for &(next, cpair, ei) in neighbors {
                if visited[ei] { continue; }

                let mut chain = vec![*start_node];
                let mut next_node = next;
                let mut cur_ei = ei;

                loop {
                    if visited[cur_ei] { break; }
                    visited[cur_ei] = true;
                    chain.push(next_node);

                    let nbrs = adj.get(&next_node).unwrap();
                    let mut found_next = false;
                    for &(nn, cp, nei) in nbrs {
                        if cp != cpair { continue; }
                        if !visited[nei] {
                            next_node = nn;
                            cur_ei = nei;
                            found_next = true;
                            break;
                        }
                    }
                    if !found_next { break; }
                }

                if chain.len() >= 3 {
                    chains.push((chain, cpair));
                }
            }
        }
    }

    chains
}

// --- Section 3.3: T-junction merging ---

/// Check if a color pair represents a "shading edge" -- the two colors are
/// somewhat different (enough to be a visible edge) but not strongly dissimilar.
/// Uses Euclidean distance in YUV space <= 100/255, matching the reference
/// implementation (FullCellGraphConstruction.geom isContour).
#[inline]
fn is_shading_cpair(cpair: (u32, u32)) -> bool {
    let (a, b) = cpair;
    if a == b { return true; }
    if a == VOID_COLOR || b == VOID_COLOR { return false; }
    let dr = ((a >> 16) & 0xFF) as f64 - ((b >> 16) & 0xFF) as f64;
    let dg = ((a >> 8) & 0xFF) as f64 - ((b >> 8) & 0xFF) as f64;
    let db = (a & 0xFF) as f64 - (b & 0xFF) as f64;
    // YUV conversion (same coefficients as similarity graph)
    let dy = (0.299 * dr + 0.587 * dg + 0.114 * db) / 255.0;
    let du = (0.493 * (db / 255.0 - dy)) ;
    let dv = (0.877 * (dr / 255.0 - dy)) ;
    // Euclidean distance in YUV space <= 100/255
    let dist_sq = dy * dy + du * du + dv * dv;
    let threshold = 100.0 / 255.0;
    dist_sq <= threshold * threshold
}

/// At junction nodes (valence >= 3 in visible edge graph), merge chain pairs.
/// Paper Section 3.3 two-step heuristic:
///   1. Classify each edge as shading (similar colors) or contour (dissimilar).
///      If exactly 1 shading + 2 contour edges meet, connect the 2 contour edges.
///   2. Otherwise, connect the pair with the angle closest to 180 deg.
pub(super) fn merge_t_junctions(chains: &mut Vec<(Vec<NodeId>, (u32, u32))>) {
    // Build map: node -> list of (chain_index, is_start_endpoint)
    let mut endpoint_map: FxHashMap<NodeId, Vec<(usize, bool)>> = fx_hashmap();
    for (ci, (chain, _)) in chains.iter().enumerate() {
        if chain.len() < 2 { continue; }
        // Skip closed loops (first == last)
        if chain.first() == chain.last() { continue; }
        endpoint_map.entry(chain[0]).or_default().push((ci, true));
        endpoint_map.entry(chain[chain.len() - 1]).or_default().push((ci, false));
    }

    let mut merged = vec![false; chains.len()];

    for (_node, endpoints) in &endpoint_map {
        if endpoints.len() < 2 { continue; }

        // Collect valid (not yet merged) endpoints at this junction
        let mut active: Vec<(usize, bool)> = endpoints
            .iter()
            .filter(|(ci, _)| !merged[*ci])
            .copied()
            .collect();

        // Greedily merge pairs at this junction
        loop {
            if active.len() < 2 { break; }

            // Step 1: Shading/contour classification (Paper Section 3.3)
            // If exactly 3 endpoints with 1 shading + 2 contour, merge the contour pair.
            let merge_pair = if active.len() == 3 {
                let shading: Vec<usize> = (0..3)
                    .filter(|&i| is_shading_cpair(chains[active[i].0].1))
                    .collect();
                let contour: Vec<usize> = (0..3)
                    .filter(|&i| !is_shading_cpair(chains[active[i].0].1))
                    .collect();
                if shading.len() == 1 && contour.len() == 2 {
                    Some((contour[0], contour[1]))
                } else {
                    None
                }
            } else {
                None
            };

            // Step 2: Fall back to angle-based merging (straightest pair)
            let (idx_a, idx_b) = if let Some(pair) = merge_pair {
                pair
            } else {
                let mut best_cos = f64::NEG_INFINITY;
                let mut best_pair = (0usize, 1usize);

                for i in 0..active.len() {
                    for j in (i + 1)..active.len() {
                        let (ci_a, is_start_a) = active[i];
                        let (ci_b, is_start_b) = active[j];
                        if ci_a == ci_b { continue; }

                        let tan_a = chain_tangent_at_endpoint(&chains[ci_a].0, is_start_a);
                        let tan_b = chain_tangent_at_endpoint(&chains[ci_b].0, is_start_b);

                        let dot = tan_a.0 * tan_b.0 + tan_a.1 * tan_b.1;
                        let len_a = (tan_a.0 * tan_a.0 + tan_a.1 * tan_a.1).sqrt();
                        let len_b = (tan_b.0 * tan_b.0 + tan_b.1 * tan_b.1).sqrt();
                        if len_a < 1e-12 || len_b < 1e-12 { continue; }
                        let cos_angle = dot / (len_a * len_b);

                        // Most aligned = most negative cosine (closest to 180 deg)
                        if cos_angle < best_cos || best_cos == f64::NEG_INFINITY {
                            best_cos = cos_angle;
                            best_pair = (i, j);
                        }
                    }
                }
                best_pair
            };

            let (ci_a, is_start_a) = active[idx_a];
            let (ci_b, is_start_b) = active[idx_b];

            // Build merged chain: orient chain_a so junction is at end,
            // orient chain_b so junction is at start, then concatenate
            let mut new_chain = Vec::new();

            if is_start_a {
                for i in (0..chains[ci_a].0.len()).rev() {
                    new_chain.push(chains[ci_a].0[i]);
                }
            } else {
                new_chain.extend_from_slice(&chains[ci_a].0);
            }

            if is_start_b {
                for i in 1..chains[ci_b].0.len() {
                    new_chain.push(chains[ci_b].0[i]);
                }
            } else {
                for i in (0..chains[ci_b].0.len() - 1).rev() {
                    new_chain.push(chains[ci_b].0[i]);
                }
            }

            // Merged chain inherits cpair from chain_a (arbitrary; cpair is only
            // used for shading classification and this chain won't be re-classified)
            chains[ci_a].0 = new_chain;
            merged[ci_b] = true;

            active.remove(idx_b.max(idx_a));
            active.remove(idx_b.min(idx_a));
        }
    }

    // Remove merged chains
    let mut kept = Vec::with_capacity(chains.len());
    for (i, chain) in chains.drain(..).enumerate() {
        if !merged[i] {
            kept.push(chain);
        }
    }
    *chains = kept;
}

/// Get tangent vector at a chain endpoint, pointing AWAY from the endpoint.
fn chain_tangent_at_endpoint(chain: &[NodeId], is_start: bool) -> (f64, f64) {
    if chain.len() < 2 {
        return (0.0, 0.0);
    }
    if is_start {
        // Tangent at start: from node[0] toward node[1]
        let p0 = chain[0].to_point();
        let p1 = chain[1].to_point();
        (p1.x - p0.x, p1.y - p0.y)
    } else {
        // Tangent at end: from node[n-1] toward node[n-2]
        let n = chain.len();
        let p0 = chain[n - 1].to_point();
        let p1 = chain[n - 2].to_point();
        (p1.x - p0.x, p1.y - p0.y)
    }
}
