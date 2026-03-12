//! Kopf-Lischinski pixel-art vectorization: Sections 3.2–3.4.
//!
//! Pipeline:
//! 1. Build reshaped cell graph (generalized Voronoi diagram, Section 3.2)
//! 2. Extract visible edges between different-color cells (Section 3.3)
//! 3. Chain visible edges through valence-2 nodes into paths (Section 3.3)
//! 4. Merge chains at T-junctions with aligned tangents (Section 3.3)
//! 5. Optimize B-spline control points (Section 3.4)
//! 6. Flood-fill same-color regions, trace boundaries with smooth B-spline curves
//! 7. Emit region outlines as ColorPaths

use super::graph::SimilarityGraph;
use super::voronoi::Point;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

// --- Public types ---

/// A path segment for SVG output.
#[derive(Clone, Debug)]
pub enum PathSegment {
    Line(Point, Point),
    QuadBezier(Point, Point, Point),
}

/// A complete colored path ready for SVG serialization.
#[derive(Clone, Debug)]
pub struct ColorPath {
    pub color: u32,
    pub segments: Vec<PathSegment>,
}

// --- Section 3.2: Reshaped cell graph (Voronoi diagram) ---

/// Which pixel is this relative to a grid corner?
enum CornerRel {
    TL,
    TR,
    BL,
    BR,
}

/// Compute the vertices of a pixel's Voronoi cell at one of its corners.
fn corner_vertices(
    cx: i32, cy: i32, rel: CornerRel, graph: &SimilarityGraph, out: &mut Vec<Point>,
) {
    let w = graph.width as i32;
    let h = graph.height as i32;

    if cx <= 0 || cy <= 0 || cx >= w || cy >= h {
        out.push(Point::new(cx as f64, cy as f64));
        return;
    }

    let has_backslash = graph.edge((cx - 1) as usize, (cy - 1) as usize).down_right;
    let has_slash = graph.edge(cx as usize, (cy - 1) as usize).down_left;

    if has_backslash {
        let a = Point::new(cx as f64 - 0.25, cy as f64 + 0.25);
        let b = Point::new(cx as f64 + 0.25, cy as f64 - 0.25);
        match rel {
            CornerRel::TL => { out.push(b); out.push(a); }
            CornerRel::BR => { out.push(a); out.push(b); }
            CornerRel::TR => out.push(b),
            CornerRel::BL => out.push(a),
        }
    } else if has_slash {
        let c = Point::new(cx as f64 - 0.25, cy as f64 - 0.25);
        let d = Point::new(cx as f64 + 0.25, cy as f64 + 0.25);
        match rel {
            CornerRel::TR => { out.push(d); out.push(c); }
            CornerRel::BL => { out.push(c); out.push(d); }
            CornerRel::TL => out.push(c),
            CornerRel::BR => out.push(d),
        }
    } else {
        out.push(Point::new(cx as f64, cy as f64));
    }
}

/// Compute the full Voronoi cell polygon for pixel (px, py).
/// Returns vertices in CW order, 4–8 vertices.
fn pixel_cell(px: usize, py: usize, graph: &SimilarityGraph) -> Vec<Point> {
    let ipx = px as i32;
    let ipy = py as i32;
    let mut vertices = Vec::with_capacity(8);
    corner_vertices(ipx, ipy, CornerRel::BR, graph, &mut vertices);
    corner_vertices(ipx + 1, ipy, CornerRel::BL, graph, &mut vertices);
    corner_vertices(ipx + 1, ipy + 1, CornerRel::TL, graph, &mut vertices);
    corner_vertices(ipx, ipy + 1, CornerRel::TR, graph, &mut vertices);
    vertices
}

/// Render each pixel as its Voronoi cell polygon, grouped by color.
pub fn extract_cells(pixels: &[u32], graph: &SimilarityGraph) -> Vec<ColorPath> {
    let w = graph.width;
    let h = graph.height;

    let mut color_cells: BTreeMap<u32, Vec<Vec<Point>>> = BTreeMap::new();

    for y in 0..h {
        for x in 0..w {
            let color = pixels[y * w + x];
            let cell = pixel_cell(x, y, graph);
            color_cells.entry(color).or_default().push(cell);
        }
    }

    color_cells
        .into_iter()
        .map(|(color, cells)| {
            let mut segments = Vec::new();
            for cell in &cells {
                let n = cell.len();
                if n < 3 { continue; }
                for i in 0..n {
                    segments.push(PathSegment::Line(cell[i], cell[(i + 1) % n]));
                }
            }
            ColorPath { color, segments }
        })
        .collect()
}

// --- Section 3.3: Visible edge extraction and B-spline fitting ---

/// A node in the reshaped cell graph, identified by quantized coordinates.
/// We quantize to 1/4 pixel to handle the ±0.25 offsets.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
struct NodeId {
    x4: i32,
    y4: i32,
}

impl NodeId {
    fn from_point(p: &Point) -> Self {
        NodeId {
            x4: (p.x * 4.0).round() as i32,
            y4: (p.y * 4.0).round() as i32,
        }
    }

    fn to_point(self) -> Point {
        Point::new(self.x4 as f64 / 4.0, self.y4 as f64 / 4.0)
    }
}

/// An edge in the cell graph between two nodes, with colors on each side.
#[derive(Clone, Copy, Debug)]
struct CellEdge {
    a: NodeId,
    b: NodeId,
    left_color: u32,
    right_color: u32,
}

/// Extract all visible cell edges (different colors on each side).
fn extract_cell_edges(pixels: &[u32], graph: &SimilarityGraph) -> Vec<CellEdge> {
    let w = graph.width;
    let h = graph.height;

    let mut edge_map: HashMap<(NodeId, NodeId), (Option<u32>, Option<u32>)> = HashMap::new();

    for y in 0..h {
        for x in 0..w {
            let color = pixels[y * w + x];
            let cell = pixel_cell(x, y, graph);
            let n = cell.len();
            if n < 3 { continue; }

            for i in 0..n {
                let pa = NodeId::from_point(&cell[i]);
                let pb = NodeId::from_point(&cell[(i + 1) % n]);

                let (key, is_forward) = if pa <= pb {
                    ((pa, pb), true)
                } else {
                    ((pb, pa), false)
                };

                let entry = edge_map.entry(key).or_insert((None, None));
                if is_forward {
                    entry.1 = Some(color);
                } else {
                    entry.0 = Some(color);
                }
            }
        }
    }

    let mut visible = Vec::new();
    for ((a, b), (left, right)) in &edge_map {
        let lc = left.unwrap_or(0);
        let rc = right.unwrap_or(0);
        if lc != rc {
            visible.push(CellEdge {
                a: *a,
                b: *b,
                left_color: lc,
                right_color: rc,
            });
        }
    }

    visible
}

/// Chain visible edges through valence-2 nodes into paths.
fn chain_visible_edges(edges: &[CellEdge]) -> Vec<Vec<NodeId>> {
    let mut adj: HashMap<NodeId, Vec<(NodeId, (u32, u32))>> = HashMap::new();

    for e in edges {
        let cpair = if e.left_color <= e.right_color {
            (e.left_color, e.right_color)
        } else {
            (e.right_color, e.left_color)
        };
        adj.entry(e.a).or_default().push((e.b, cpair));
        adj.entry(e.b).or_default().push((e.a, cpair));
    }

    let mut visited_edges: HashSet<(NodeId, NodeId)> = HashSet::new();
    let mut chains: Vec<Vec<NodeId>> = Vec::new();

    let is_valence2 = |node: &NodeId, cpair: &(u32, u32)| -> bool {
        if let Some(neighbors) = adj.get(node) {
            neighbors.iter().filter(|(_, cp)| cp == cpair).count() == 2
        } else {
            false
        }
    };

    let all_nodes: Vec<NodeId> = adj.keys().copied().collect();

    // Start chains from endpoints (valence != 2)
    for start_node in &all_nodes {
        if let Some(neighbors) = adj.get(start_node) {
            for &(next, cpair) in neighbors {
                let edge_key = if *start_node <= next {
                    (*start_node, next)
                } else {
                    (next, *start_node)
                };
                if visited_edges.contains(&edge_key) { continue; }
                if is_valence2(start_node, &cpair) { continue; }

                let mut chain = vec![*start_node];
                let mut current = *start_node;
                let mut next_node = next;

                loop {
                    let ek = if current <= next_node {
                        (current, next_node)
                    } else {
                        (next_node, current)
                    };
                    if visited_edges.contains(&ek) { break; }
                    visited_edges.insert(ek);
                    chain.push(next_node);

                    if !is_valence2(&next_node, &cpair) { break; }

                    let neighbors_of_next = adj.get(&next_node).unwrap();
                    let mut found_next = false;
                    for &(nn, cp) in neighbors_of_next {
                        if cp != cpair { continue; }
                        let ek2 = if next_node <= nn {
                            (next_node, nn)
                        } else {
                            (nn, next_node)
                        };
                        if !visited_edges.contains(&ek2) {
                            current = next_node;
                            next_node = nn;
                            found_next = true;
                            break;
                        }
                    }
                    if !found_next { break; }
                }

                if chain.len() >= 2 {
                    chains.push(chain);
                }
            }
        }
    }

    // Handle closed loops (all nodes are valence-2)
    for start_node in &all_nodes {
        if let Some(neighbors) = adj.get(start_node) {
            for &(next, cpair) in neighbors {
                let edge_key = if *start_node <= next {
                    (*start_node, next)
                } else {
                    (next, *start_node)
                };
                if visited_edges.contains(&edge_key) { continue; }

                let mut chain = vec![*start_node];
                let mut current = *start_node;
                let mut next_node = next;

                loop {
                    let ek = if current <= next_node {
                        (current, next_node)
                    } else {
                        (next_node, current)
                    };
                    if visited_edges.contains(&ek) { break; }
                    visited_edges.insert(ek);
                    chain.push(next_node);

                    let neighbors_of_next = adj.get(&next_node).unwrap();
                    let mut found_next = false;
                    for &(nn, cp) in neighbors_of_next {
                        if cp != cpair { continue; }
                        let ek2 = if next_node <= nn {
                            (next_node, nn)
                        } else {
                            (nn, next_node)
                        };
                        if !visited_edges.contains(&ek2) {
                            current = next_node;
                            next_node = nn;
                            found_next = true;
                            break;
                        }
                    }
                    if !found_next { break; }
                }

                if chain.len() >= 3 {
                    chains.push(chain);
                }
            }
        }
    }

    chains
}

// --- Section 3.3: T-junction merging ---

/// At junction nodes (valence >= 3 in visible edge graph), find pairs of chain
/// endpoints with the most aligned tangent vectors and merge them.
/// Paper Section 3.3: merge if angle between tangents < 160°.
fn merge_t_junctions(chains: &mut Vec<Vec<NodeId>>) {
    // Build map: node → list of (chain_index, is_start_endpoint)
    let mut endpoint_map: HashMap<NodeId, Vec<(usize, bool)>> = HashMap::new();
    for (ci, chain) in chains.iter().enumerate() {
        if chain.len() < 2 { continue; }
        // Skip closed loops (first == last)
        if chain.first() == chain.last() { continue; }
        endpoint_map.entry(chain[0]).or_default().push((ci, true));
        endpoint_map.entry(chain[chain.len() - 1]).or_default().push((ci, false));
    }

    let mut merged: HashSet<usize> = HashSet::new();

    for (_node, endpoints) in &endpoint_map {
        if endpoints.len() < 2 { continue; }

        // Collect valid (not yet merged) endpoints at this junction
        let mut active: Vec<(usize, bool)> = endpoints
            .iter()
            .filter(|(ci, _)| !merged.contains(ci))
            .copied()
            .collect();

        // Greedily merge the most aligned pair
        loop {
            if active.len() < 2 { break; }

            let mut best_cos = f64::NEG_INFINITY;
            let mut best_pair = (0usize, 1usize);

            for i in 0..active.len() {
                for j in (i + 1)..active.len() {
                    let (ci_a, is_start_a) = active[i];
                    let (ci_b, is_start_b) = active[j];
                    if ci_a == ci_b { continue; }

                    // Get tangent vectors pointing AWAY from the junction node
                    let tan_a = chain_tangent_at_endpoint(&chains[ci_a], is_start_a);
                    let tan_b = chain_tangent_at_endpoint(&chains[ci_b], is_start_b);

                    // Tangents point away from junction; for merging, they should
                    // point in opposite directions, so we check cos of angle between them
                    // cos(180°) = -1.0. Paper: merge if angle < 160° → cos > -0.94
                    let dot = tan_a.0 * tan_b.0 + tan_a.1 * tan_b.1;
                    let len_a = (tan_a.0 * tan_a.0 + tan_a.1 * tan_a.1).sqrt();
                    let len_b = (tan_b.0 * tan_b.0 + tan_b.1 * tan_b.1).sqrt();
                    if len_a < 1e-12 || len_b < 1e-12 { continue; }
                    let cos_angle = dot / (len_a * len_b);

                    // Most aligned = most negative cosine (closest to 180°)
                    if cos_angle < best_cos || best_cos == f64::NEG_INFINITY {
                        best_cos = cos_angle;
                        best_pair = (i, j);
                    }
                }
            }

            // Only merge if angle > 160° (cos < -0.94)
            if best_cos > -0.94 { break; }

            let (idx_a, idx_b) = best_pair;
            let (ci_a, is_start_a) = active[idx_a];
            let (ci_b, is_start_b) = active[idx_b];

            // Build merged chain: orient chain_a so junction is at end,
            // orient chain_b so junction is at start, then concatenate
            let mut new_chain = Vec::new();

            if is_start_a {
                // Junction is at start of chain_a → reverse it so junction is at end
                for i in (0..chains[ci_a].len()).rev() {
                    new_chain.push(chains[ci_a][i]);
                }
            } else {
                // Junction is at end of chain_a → already correct
                new_chain.extend_from_slice(&chains[ci_a]);
            }

            // Skip the junction node (it's already the last element of new_chain)
            if is_start_b {
                // Junction is at start of chain_b → skip first element
                for i in 1..chains[ci_b].len() {
                    new_chain.push(chains[ci_b][i]);
                }
            } else {
                // Junction is at end of chain_b → reverse, skip first (was junction)
                for i in (0..chains[ci_b].len() - 1).rev() {
                    new_chain.push(chains[ci_b][i]);
                }
            }

            // Replace chain_a with merged, mark chain_b as merged
            chains[ci_a] = new_chain;
            merged.insert(ci_b);

            // Update active list: remove both, add new endpoint of merged chain
            let new_is_start_a = false; // junction was placed at end for chain_a part
            // The new chain's start is what was the far end of chain_a,
            // and its end is what was the far end of chain_b
            // We don't re-add to active since we consumed both endpoints at this junction
            active.remove(idx_b.max(idx_a));
            active.remove(idx_b.min(idx_a));
        }
    }

    // Remove merged chains
    let mut i = 0;
    while i < chains.len() {
        if merged.contains(&i) {
            chains.remove(i);
            // Update merged indices
            let mut new_merged = HashSet::new();
            for &m in &merged {
                if m > i {
                    new_merged.insert(m - 1);
                } else if m < i {
                    new_merged.insert(m);
                }
            }
            merged = new_merged;
        } else {
            i += 1;
        }
    }
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

// --- Optimized node positions ---

/// Build map from NodeId to optimized Point position.
/// Interior chain nodes get optimized positions; junction/corner nodes stay fixed.
fn build_optimized_positions(chains: &[Vec<NodeId>]) -> HashMap<NodeId, Point> {
    let mut positions: HashMap<NodeId, Point> = HashMap::new();

    for chain in chains {
        let is_closed = chain.len() > 2 && chain.first() == chain.last();
        let ctrl_nodes = if is_closed { &chain[..chain.len() - 1] } else { &chain[..] };

        let mut points: Vec<Point> = ctrl_nodes.iter().map(|n| n.to_point()).collect();
        if points.len() >= 4 {
            optimize_control_points(&mut points, is_closed);
        }

        for (i, node) in ctrl_nodes.iter().enumerate() {
            positions.insert(*node, points[i]);
        }
    }

    positions
}

// --- Region-based boundary tracing ---

/// Flood-fill pixels by color using 4-connectivity to find connected regions.
fn flood_fill_regions(pixels: &[u32], w: usize, h: usize) -> Vec<(u32, Vec<(usize, usize)>)> {
    let mut visited = vec![false; w * h];
    let mut regions = Vec::new();

    for y in 0..h {
        for x in 0..w {
            if visited[y * w + x] { continue; }
            let color = pixels[y * w + x];
            visited[y * w + x] = true;

            let mut region_pixels = vec![(x, y)];
            let mut queue = VecDeque::new();
            queue.push_back((x, y));

            while let Some((cx, cy)) = queue.pop_front() {
                for &(dx, dy) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let nx = cx as i32 + dx;
                    let ny = cy as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 { continue; }
                    let (nx, ny) = (nx as usize, ny as usize);
                    if visited[ny * w + nx] { continue; }
                    if pixels[ny * w + nx] != color { continue; }
                    visited[ny * w + nx] = true;
                    region_pixels.push((nx, ny));
                    queue.push_back((nx, ny));
                }
            }

            regions.push((color, region_pixels));
        }
    }

    regions
}

/// For a region, find all boundary cell edges (edges with only one side in the region).
/// Returns directed edges with the region on the RIGHT (from CW cell winding).
fn region_boundary_edges(
    region_pixels: &[(usize, usize)],
    graph: &SimilarityGraph,
) -> Vec<(NodeId, NodeId)> {
    // Collect all directed edges from region cells, count undirected occurrences
    let mut all_directed: Vec<(NodeId, NodeId)> = Vec::new();
    let mut undirected_count: HashMap<(NodeId, NodeId), usize> = HashMap::new();

    for &(px, py) in region_pixels {
        let cell = pixel_cell(px, py, graph);
        let n = cell.len();
        if n < 3 { continue; }

        for i in 0..n {
            let pa = NodeId::from_point(&cell[i]);
            let pb = NodeId::from_point(&cell[(i + 1) % n]);
            all_directed.push((pa, pb));
            let key = if pa <= pb { (pa, pb) } else { (pb, pa) };
            *undirected_count.entry(key).or_insert(0) += 1;
        }
    }

    // Boundary edges: undirected count == 1 (only one region cell uses this edge)
    all_directed
        .into_iter()
        .filter(|&(pa, pb)| {
            let key = if pa <= pb { (pa, pb) } else { (pb, pa) };
            undirected_count[&key] == 1
        })
        .collect()
}

/// Trace boundary edges into closed node loops using the planar face algorithm.
/// At each node, picks the rightmost turn (most CW) to keep region on the right.
fn trace_boundary_node_loops(boundary_edges: &[(NodeId, NodeId)]) -> Vec<Vec<NodeId>> {
    // Build outgoing edges sorted by angle at each node
    let mut outgoing: HashMap<NodeId, Vec<(NodeId, f64)>> = HashMap::new();
    for &(a, b) in boundary_edges {
        let ap = a.to_point();
        let bp = b.to_point();
        let angle = (bp.y - ap.y).atan2(bp.x - ap.x);
        outgoing.entry(a).or_default().push((b, angle));
    }
    // Sort by angle (ascending atan2 in y-down = CW visual order)
    for (_, edges) in outgoing.iter_mut() {
        edges.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    // Build next-edge map using planar face algorithm:
    // For incoming edge P→C, find the outgoing edge from C that is the rightmost
    // turn. This is the edge just BEFORE the reverse direction (C→P) in the
    // CW-sorted (ascending atan2) list.
    let mut next_map: HashMap<(NodeId, NodeId), (NodeId, NodeId)> = HashMap::new();

    for &(p, c) in boundary_edges {
        let pp = p.to_point();
        let cp = c.to_point();
        let reverse_angle = (pp.y - cp.y).atan2(pp.x - cp.x);

        if let Some(edges) = outgoing.get(&c) {
            // Find insertion point for reverse_angle in the sorted list
            let pos = edges.partition_point(|(_, a)| *a < reverse_angle);
            // Take the entry just before (wrapping around)
            let prev_idx = if pos == 0 { edges.len() - 1 } else { pos - 1 };
            let (next_node, _) = edges[prev_idx];
            next_map.insert((p, c), (c, next_node));
        }
    }

    // Trace loops by following the next-edge map
    let mut used: HashSet<(NodeId, NodeId)> = HashSet::new();
    let mut loops = Vec::new();

    for &start_edge in boundary_edges {
        if used.contains(&start_edge) { continue; }

        let mut nodes = Vec::new();
        let mut current = start_edge;
        let mut closed = false;

        loop {
            if used.contains(&current) {
                if current == start_edge { closed = true; }
                break;
            }
            used.insert(current);
            nodes.push(current.0);
            current = match next_map.get(&current) {
                Some(&next) => next,
                None => break,
            };
        }

        if nodes.len() >= 3 && closed {
            loops.push(nodes);
        }
    }

    loops
}

/// Convert a closed boundary node loop to smooth path segments.
/// Uses optimized positions and splits at corners for B-spline fitting.
fn boundary_loop_to_segments(
    nodes: &[NodeId],
    optimized: &HashMap<NodeId, Point>,
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

    let corners = detect_corners(&points, true);
    let corner_indices: Vec<usize> = (0..n).filter(|&i| corners[i]).collect();

    if corner_indices.is_empty() {
        // No corners: closed B-spline over entire loop
        return bspline_closed(&points);
    }

    if corner_indices.len() == 1 {
        // Single corner: open B-spline from corner around the full loop back to corner
        let c = corner_indices[0];
        let mut span_points = Vec::with_capacity(n + 1);
        for i in 0..=n {
            span_points.push(points[(c + i) % n]);
        }
        return bspline_open(&span_points);
    }

    // Split loop into spans between consecutive corners
    let mut segments = Vec::new();
    let num_corners = corner_indices.len();

    for ci in 0..num_corners {
        let start = corner_indices[ci];
        let end = corner_indices[(ci + 1) % num_corners];

        // Collect points from start to end (inclusive), wrapping around
        let mut span_points = Vec::new();
        let mut idx = start;
        loop {
            span_points.push(points[idx]);
            if idx == end { break; }
            idx = (idx + 1) % n;
        }

        if span_points.len() < 2 { continue; }

        if span_points.len() == 2 {
            segments.push(PathSegment::Line(span_points[0], span_points[1]));
        } else {
            segments.extend(bspline_open(&span_points));
        }
    }

    segments
}

// --- Main entry point ---

/// Vectorize with B-spline smoothing (Sections 3.2–3.4).
/// Region-based rendering: merges same-color cells into regions, traces
/// boundaries on the cell graph, emits smooth B-spline curves as region outlines.
pub fn extract_cells_smooth(pixels: &[u32], graph: &SimilarityGraph) -> Vec<ColorPath> {
    let w = graph.width;
    let h = graph.height;

    // Step 1: Extract visible edges and chain them
    let visible_edges = extract_cell_edges(pixels, graph);
    let mut chains = chain_visible_edges(&visible_edges);

    // Step 2: Merge T-junctions (Section 3.3)
    merge_t_junctions(&mut chains);

    // Step 3: Optimize control points and build position map
    let optimized = build_optimized_positions(&chains);

    // Step 4: Flood-fill regions
    let regions = flood_fill_regions(pixels, w, h);

    // Step 5: For each region, trace boundary and emit smooth paths
    let mut result: Vec<ColorPath> = Vec::new();

    for (color, region_pixels) in &regions {
        let boundary_edges = region_boundary_edges(region_pixels, graph);
        if boundary_edges.is_empty() { continue; }

        let loops = trace_boundary_node_loops(&boundary_edges);
        if loops.is_empty() { continue; }

        let mut all_segments = Vec::new();
        for node_loop in &loops {
            all_segments.extend(boundary_loop_to_segments(node_loop, &optimized));
        }

        if !all_segments.is_empty() {
            result.push(ColorPath {
                color: *color,
                segments: all_segments,
            });
        }
    }

    result
}

// --- B-spline fitting ---

/// Convert a closed loop of control points to quadratic B-spline segments.
fn bspline_closed(ctrl: &[Point]) -> Vec<PathSegment> {
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
fn bspline_open(ctrl: &[Point]) -> Vec<PathSegment> {
    let n = ctrl.len();
    if n < 3 {
        return line_segments(ctrl);
    }

    let mut segments = Vec::new();

    let mid01 = Point::new(
        (ctrl[0].x + ctrl[1].x) * 0.5,
        (ctrl[0].y + ctrl[1].y) * 0.5,
    );
    segments.push(PathSegment::Line(ctrl[0], mid01));

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
    segments.push(PathSegment::Line(mid_last, ctrl[n - 1]));

    segments
}

fn line_segments(pts: &[Point]) -> Vec<PathSegment> {
    pts.windows(2)
        .map(|w| PathSegment::Line(w[0], w[1]))
        .collect()
}

// --- Section 3.4: B-spline optimization ---

const OPT_ITERATIONS: usize = 50;
const POINT_GUESSES: usize = 40;
const GUESS_RADIUS: f64 = 0.25;
const CURVATURE_INTERVALS: usize = 20;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self { Self(seed.max(1)) }
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 & 0x000F_FFFF_FFFF_FFFF) as f64 / (0x0010_0000_0000_0000u64 as f64)
    }
}

fn detect_corners(points: &[Point], is_closed: bool) -> Vec<bool> {
    let n = points.len();
    let mut is_corner = vec![false; n];

    if !is_closed {
        if n > 0 { is_corner[0] = true; }
        if n > 1 { is_corner[n - 1] = true; }
    }

    let range_start = if is_closed { 0 } else { 1 };
    let range_end = if is_closed { n } else { n - 1 };

    for i in range_start..range_end {
        let prev = if is_closed {
            points[(i + n - 1) % n]
        } else {
            points[i - 1]
        };
        let curr = points[i];
        let next = if is_closed {
            points[(i + 1) % n]
        } else {
            points[i + 1]
        };

        let dx0 = curr.x - prev.x;
        let dy0 = curr.y - prev.y;
        let dx1 = next.x - curr.x;
        let dy1 = next.y - curr.y;

        let len0_sq = dx0 * dx0 + dy0 * dy0;
        let len1_sq = dx1 * dx1 + dy1 * dy1;
        if len0_sq < 1e-12 || len1_sq < 1e-12 { continue; }

        let dot = dx0 * dx1 + dy0 * dy1;
        let cos_angle = dot / (len0_sq.sqrt() * len1_sq.sqrt());

        if cos_angle < 0.5 {
            is_corner[i] = true;
        }
    }

    is_corner
}

fn optimize_control_points(points: &mut Vec<Point>, is_closed: bool) {
    let n = points.len();
    if n < 4 { return; }

    let orig: Vec<Point> = points.clone();
    let corners = detect_corners(&orig, is_closed);
    let mut rng = Rng::new(0xDEAD_BEEF_CAFE_1234);

    for _iter in 0..OPT_ITERATIONS {
        for i in 0..n {
            if corners[i] { continue; }

            let current = points[i];
            let mut best_energy = local_energy(points, &orig, &corners, i, n, is_closed);
            let mut best_point = current;

            for _ in 0..POINT_GUESSES {
                let r = rng.next_f64() * GUESS_RADIUS;
                let theta = rng.next_f64() * std::f64::consts::TAU;
                let candidate = Point::new(
                    current.x + r * theta.cos(),
                    current.y + r * theta.sin(),
                );
                points[i] = candidate;
                let energy = local_energy(points, &orig, &corners, i, n, is_closed);
                if energy < best_energy {
                    best_energy = energy;
                    best_point = candidate;
                }
            }

            points[i] = best_point;
        }
    }
}

fn local_energy(
    points: &[Point], orig: &[Point], corners: &[bool],
    idx: usize, n: usize, is_closed: bool,
) -> f64 {
    curvature_energy(points, corners, idx, n, is_closed)
        + positional_energy(points, orig, idx)
}

fn positional_energy(points: &[Point], orig: &[Point], idx: usize) -> f64 {
    let dx = points[idx].x - orig[idx].x;
    let dy = points[idx].y - orig[idx].y;
    let dist_sq = dx * dx + dy * dy;
    dist_sq * dist_sq
}

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

fn integrate_span_curvature(p0: Point, p1: Point, p2: Point) -> f64 {
    let dt = 1.0 / CURVATURE_INTERVALS as f64;
    let mut result = (curvature_sq_at(p0, p1, p2, 0.0)
        + curvature_sq_at(p0, p1, p2, 1.0)) / 2.0;
    for i in 1..CURVATURE_INTERVALS {
        result += curvature_sq_at(p0, p1, p2, i as f64 * dt);
    }
    result * dt
}

fn curvature_sq_at(p0: Point, p1: Point, p2: Point, t: f64) -> f64 {
    let dx = (t - 1.0) * p0.x + (1.0 - 2.0 * t) * p1.x + t * p2.x;
    let dy = (t - 1.0) * p0.y + (1.0 - 2.0 * t) * p1.y + t * p2.y;
    let ddx = p0.x - 2.0 * p1.x + p2.x;
    let ddy = p0.y - 2.0 * p1.y + p2.y;

    let numer = dx * ddy - dy * ddx;
    let denom_sq = dx * dx + dy * dy;
    let denom = denom_sq * denom_sq.sqrt();
    if denom < 1e-12 { 0.0 } else { (numer * numer) / (denom * denom) }
}
