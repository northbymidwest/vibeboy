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
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::AtomicBool;

/// When true, visible edges use YUV similarity threshold (Paper Section 3.2).
/// When false, any color difference creates a visible edge (default, more robust).
pub static YUV_VISIBLE_EDGES: AtomicBool = AtomicBool::new(false);

/// Sentinel color for the void outside the image. Must not collide with any
/// real pixel value. The PPU uses 0x00RRGGBB and the test_runner PNG loader
/// uses 0xFFRRGGBB, so 0x01000000 (alpha=1, RGB=0) is safe for both.
/// Previous value 0xFFFFFFFF collided with white (0xFFFFFFFF) in the
/// test_runner's 0xFFRRGGBB format.
const VOID_COLOR: u32 = 0x01000000;

// --- FxHash: fast non-cryptographic hasher for integer keys ---
// Replaces SipHash (default) which is ~3x slower for small integer keys.

const FXHASH_SEED: u64 = 0x517cc1b727220a95;

struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline] fn finish(&self) -> u64 { self.0 }
    #[inline] fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(FXHASH_SEED);
        }
    }
    #[inline] fn write_i32(&mut self, i: i32) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(FXHASH_SEED);
    }
    #[inline] fn write_u32(&mut self, i: u32) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(FXHASH_SEED);
    }
    #[inline] fn write_usize(&mut self, i: usize) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(FXHASH_SEED);
    }
    #[inline] fn write_u8(&mut self, i: u8) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(FXHASH_SEED);
    }
}

#[derive(Clone, Copy, Default)]
struct FxBuildHasher;
impl BuildHasher for FxBuildHasher {
    type Hasher = FxHasher;
    #[inline] fn build_hasher(&self) -> FxHasher { FxHasher(0) }
}

type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;

#[inline]
fn fx_hashmap<K, V>() -> FxHashMap<K, V> {
    HashMap::with_hasher(FxBuildHasher)
}

#[inline]
fn fx_hashmap_cap<K, V>(cap: usize) -> FxHashMap<K, V> {
    HashMap::with_capacity_and_hasher(cap, FxBuildHasher)
}

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

// --- Precomputed cells via template matching (Paper Section 3.2) ---
//
// The paper: "The shape of a Voronoi cell is fully determined by its local
// neighborhood in the similarity graph. The possible distinct shapes are easy
// to enumerate, enabling an extremely efficient algorithm, which walks in
// scanline order over the similarity graph, matches specific edge configurations
// in a 3×3 block at a time, and pastes together the corresponding cell templates."
//
// Each pixel's cell shape depends on the diagonal state at its 4 corners:
// none (0), backslash (1), or slash (2). With 4 corners × 3 states = 81
// templates, each mapping to a fixed cell polygon (4–8 vertices in ×4 coords).

/// A pre-computed Voronoi cell template: up to 8 vertex offsets in ×4 coordinates
/// relative to the pixel's top-left corner (4*px, 4*py).
#[derive(Clone, Copy)]
struct CellTemplate {
    offsets: [(i8, i8); 8],
    len: u8,
}

/// Build one cell template from the 4 corner diagonal states.
/// Corner states: 0 = no diagonal, 1 = backslash, 2 = slash.
/// Vertices are in CW order: TL(BR), TR(BL), BR(TL), BL(TR).
const fn build_cell_template(tl: u8, tr: u8, br: u8, bl: u8) -> CellTemplate {
    let mut offsets = [(0i8, 0i8); 8];
    let mut len = 0u8;

    // TL corner, pixel visits as CornerRel::BR
    if tl == 1 {        // backslash: a=(-1,1), b=(1,-1)
        offsets[len as usize] = (-1, 1); len += 1;
        offsets[len as usize] = (1, -1); len += 1;
    } else if tl == 2 { // slash: d=(1,1)
        offsets[len as usize] = (1, 1); len += 1;
    } else {             // none: (0,0)
        offsets[len as usize] = (0, 0); len += 1;
    }

    // TR corner, pixel visits as CornerRel::BL
    if tr == 1 {        // backslash: a=(3,1)
        offsets[len as usize] = (3, 1); len += 1;
    } else if tr == 2 { // slash: c=(3,-1), d=(5,1)
        offsets[len as usize] = (3, -1); len += 1;
        offsets[len as usize] = (5, 1); len += 1;
    } else {             // none: (4,0)
        offsets[len as usize] = (4, 0); len += 1;
    }

    // BR corner, pixel visits as CornerRel::TL
    if br == 1 {        // backslash: b=(5,3), a=(3,5)
        offsets[len as usize] = (5, 3); len += 1;
        offsets[len as usize] = (3, 5); len += 1;
    } else if br == 2 { // slash: c=(3,3)
        offsets[len as usize] = (3, 3); len += 1;
    } else {             // none: (4,4)
        offsets[len as usize] = (4, 4); len += 1;
    }

    // BL corner, pixel visits as CornerRel::TR
    if bl == 1 {        // backslash: b=(1,3)
        offsets[len as usize] = (1, 3); len += 1;
    } else if bl == 2 { // slash: d=(1,5), c=(-1,3)
        offsets[len as usize] = (1, 5); len += 1;
        offsets[len as usize] = (-1, 3); len += 1;
    } else {             // none: (0,4)
        offsets[len as usize] = (0, 4); len += 1;
    }

    CellTemplate { offsets, len }
}

/// All 81 Voronoi cell templates, indexed by:
///   tl_state * 27 + tr_state * 9 + br_state * 3 + bl_state
const fn compute_cell_templates() -> [CellTemplate; 81] {
    let mut templates = [CellTemplate { offsets: [(0, 0); 8], len: 0 }; 81];
    let mut i = 0u8;
    while i < 81 {
        let tl = i / 27;
        let tr = (i / 9) % 3;
        let br = (i / 3) % 3;
        let bl = i % 3;
        templates[i as usize] = build_cell_template(tl, tr, br, bl);
        i += 1;
    }
    templates
}

static CELL_TEMPLATES: [CellTemplate; 81] = compute_cell_templates();

/// Get diagonal state at grid corner (cx, cy): 0=none, 1=backslash, 2=slash.
/// Boundary corners (on image edge) always return 0.
#[inline(always)]
fn corner_diag_state(graph: &SimilarityGraph, cx: usize, cy: usize) -> usize {
    let w = graph.width;
    let h = graph.height;
    if cx == 0 || cy == 0 || cx >= w || cy >= h {
        return 0;
    }
    if graph.edge(cx - 1, cy - 1).down_right { return 1; }
    if graph.edge(cx, cy - 1).down_left { return 2; }
    0
}

/// Inline fixed-size cell: max 8 vertices per Voronoi cell, avoids heap allocation.
#[derive(Clone, Copy)]
struct InlineCell {
    nodes: [NodeId; 8],
    len: u8,
}

impl InlineCell {
    #[inline]
    fn as_slice(&self) -> &[NodeId] {
        &self.nodes[..self.len as usize]
    }
}

/// Precompute NodeId cell polygons for all pixels using template matching.
/// For each pixel, reads the diagonal state at its 4 corners, looks up the
/// corresponding cell template, and stamps out NodeId vertices directly
/// from ×4 integer offsets — no floating point or per-corner branching.
fn precompute_cells(w: usize, h: usize, graph: &SimilarityGraph) -> Vec<InlineCell> {
    let zero = NodeId { x4: 0, y4: 0 };
    let mut cells = vec![InlineCell { nodes: [zero; 8], len: 0 }; w * h];
    for y in 0..h {
        for x in 0..w {
            let tl = corner_diag_state(graph, x, y);
            let tr = corner_diag_state(graph, x + 1, y);
            let br = corner_diag_state(graph, x + 1, y + 1);
            let bl = corner_diag_state(graph, x, y + 1);

            let template = &CELL_TEMPLATES[tl * 27 + tr * 9 + br * 3 + bl];
            let cell = &mut cells[y * w + x];
            let base_x4 = (x * 4) as i32;
            let base_y4 = (y * 4) as i32;
            cell.len = template.len;
            for i in 0..template.len as usize {
                let (dx, dy) = template.offsets[i];
                cell.nodes[i] = NodeId {
                    x4: base_x4 + dx as i32,
                    y4: base_y4 + dy as i32,
                };
            }
        }
    }
    cells
}

/// Build directed boundary edges with right-side color from precomputed cells.
/// Each undirected edge between different colors produces two directed edges:
/// a→b with right_color and b→a with left_color.
///
/// Pack a NodeId into a u32 for fast hashing. Coordinates are offset by +2
/// so negative border values (-1) become non-negative.
#[inline(always)]
fn pack_node(n: NodeId, stride: u32) -> u32 {
    (n.y4 + 2) as u32 * stride + (n.x4 + 2) as u32
}

/// Pack a canonical edge (a ≤ b) into a u64 key.
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
fn is_visible_edge(left: u32, right: u32) -> bool {
    if left == right { return false; }
    if left == VOID_COLOR || right == VOID_COLOR { return true; }
    if YUV_VISIBLE_EDGES.load(std::sync::atomic::Ordering::Relaxed) {
        !super::graph::similar(left, right)
    } else {
        true
    }
}

/// Threshold for adaptive pipeline: above this, skip B-spline and sort.
const ADAPTIVE_EDGE_THRESHOLD: usize = 12000;

fn build_directed_boundary_edges(
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


/// Chain visible edges through valence-2 nodes into paths.
/// Returns each chain with its canonical color pair (cpair).
/// Computes cpair valence inline from adjacency list (avoids separate HashMap).
fn chain_visible_edges(edges: &[CellEdge]) -> Vec<(Vec<NodeId>, (u32, u32))> {
    // Build adjacency: node → list of (neighbor, cpair, edge_index).
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

/// Check if a color pair represents a "shading edge" — the two colors are
/// somewhat different (enough to be a visible edge) but not strongly dissimilar.
/// Uses Euclidean distance in YUV space ≤ 100/255, matching the reference
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
    // Euclidean distance in YUV space ≤ 100/255
    let dist_sq = dy * dy + du * du + dv * dv;
    let threshold = 100.0 / 255.0;
    dist_sq <= threshold * threshold
}

/// At junction nodes (valence >= 3 in visible edge graph), merge chain pairs.
/// Paper Section 3.3 two-step heuristic:
///   1. Classify each edge as shading (similar colors) or contour (dissimilar).
///      If exactly 1 shading + 2 contour edges meet, connect the 2 contour edges.
///   2. Otherwise, connect the pair with the angle closest to 180°.
fn merge_t_junctions(chains: &mut Vec<(Vec<NodeId>, (u32, u32))>) {
    // Build map: node → list of (chain_index, is_start_endpoint)
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

                        // Most aligned = most negative cosine (closest to 180°)
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
fn trace_all_boundary_loops(
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
fn boundary_loop_to_segments(
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

// --- Main entry point ---

/// Vectorize with B-spline smoothing (Sections 3.2–3.4).
/// Region-based rendering: merges same-color cells into regions, traces
/// boundaries on the cell graph, emits smooth B-spline curves as region outlines.
pub fn extract_cells_smooth(pixels: &[u32], graph: &SimilarityGraph, adaptive: bool) -> Vec<ColorPath> {
    let w = graph.width;
    let h = graph.height;
    let verbose = std::env::var("VECTORIZE_BENCH").is_ok();
    let t0 = std::time::Instant::now();

    let all_cells = precompute_cells(w, h, graph);
    let t1 = std::time::Instant::now();

    let (visible_edges, directed_edges) = build_directed_boundary_edges(pixels, w, h, &all_cells, adaptive);
    let t2 = std::time::Instant::now();

    // Adaptive pipeline: skip expensive B-spline optimization when boundary
    // complexity is high (noisy/dithered frames). The optimization provides
    // negligible visual benefit when boundaries are this dense.
    // visible_edges is empty when adaptive mode was triggered in build_directed_boundary_edges.
    let adaptive = visible_edges.is_empty();

    if adaptive {
        // Adaptive path: trace loops but skip chain+tjunc+optimize.
        // Use B-spline fitting on traced loops for smooth boundaries.
        let all_loops = trace_all_boundary_loops(&directed_edges);
        let optimized: FxHashMap<NodeId, Point> = fx_hashmap();
        let junctions: HashSet<NodeId> = HashSet::new();

        let mut color_loops: BTreeMap<u32, Vec<Vec<PathSegment>>> = BTreeMap::new();
        for (node_loop, color) in &all_loops {
            let no_tjunc: FxHashMap<NodeId, Point> = fx_hashmap();
            let segs = boundary_loop_to_segments(node_loop, &optimized, &junctions, &no_tjunc);
            if !segs.is_empty() {
                color_loops.entry(*color).or_default().push(segs);
            }
        }

        let mut result: Vec<ColorPath> = Vec::new();
        for (color, loop_segments) in color_loops {
            if color == VOID_COLOR { continue; }
            let mut all_segments = Vec::new();
            for segs in loop_segments {
                all_segments.extend(segs);
            }
            result.push(ColorPath { color, segments: all_segments });
        }
        return result;
    }

    // Build junction set and T-junction corrected positions.
    // At each T-junction (exactly 3 chains meeting), the two merged (contour)
    // chains define the continuing curve direction. The corrected position
    // is the B-spline evaluation at t=0.5 along that continuing curve:
    //   corrected = 0.125*prev_contour + 0.75*junction + 0.125*next_contour
    let (junctions, tjunc_corrected) = {
        let mut chains = chain_visible_edges(&visible_edges);
        merge_t_junctions(&mut chains);
        let mut j: HashSet<NodeId> = HashSet::new();
        let mut corrected: FxHashMap<NodeId, Point> = fx_hashmap();
        for (chain, _) in &chains {
            let is_closed = chain.len() > 2 && chain.first() == chain.last();
            if !is_closed && chain.len() >= 2 {
                // Start endpoint: junction with next node as contour direction
                let jn = chain[0];
                j.insert(jn);
                if chain.len() >= 3 {
                    // The merged chain continues through this junction.
                    // chain[1] is the next node in the contour direction.
                    // We need the node from the OTHER side — but we only have
                    // this chain's direction. Store it; if another chain also
                    // ends here, we'll combine both directions.
                    let jp = jn.to_point();
                    let np = chain[1].to_point();
                    corrected.entry(jn).and_modify(|existing| {
                        // Second contour direction found — compute correction
                        // existing holds the first direction's neighbor point
                        let first_neighbor = *existing;
                        *existing = Point::new(
                            0.125 * first_neighbor.x + 0.75 * jp.x + 0.125 * np.x,
                            0.125 * first_neighbor.y + 0.75 * jp.y + 0.125 * np.y,
                        );
                    }).or_insert(np); // Store first direction's neighbor
                }

                // End endpoint
                let jn = chain[chain.len() - 1];
                j.insert(jn);
                if chain.len() >= 3 {
                    let jp = jn.to_point();
                    let np = chain[chain.len() - 2].to_point();
                    corrected.entry(jn).and_modify(|existing| {
                        let first_neighbor = *existing;
                        *existing = Point::new(
                            0.125 * first_neighbor.x + 0.75 * jp.x + 0.125 * np.x,
                            0.125 * first_neighbor.y + 0.75 * jp.y + 0.125 * np.y,
                        );
                    }).or_insert(np);
                }
            }
        }
        // Filter out entries that only got one direction (stored raw neighbor, not corrected)
        // A properly corrected junction has both directions and the formula applied
        // We can detect this: if exactly 2 chains contributed, the entry was modified
        // via and_modify. If only 1 chain contributed, it's just a raw neighbor point.
        // Keep only entries where the correction was actually computed (2+ chains).
        // Unfortunately we can't distinguish. Let's use a separate counter.
        // Simpler approach: recompute. Collect all chain endpoints per junction,
        // then compute correction for junctions with exactly 2 chain endpoints.
        let mut junc_neighbors: FxHashMap<NodeId, Vec<Point>> = fx_hashmap();
        for (chain, _) in &chains {
            let is_closed = chain.len() > 2 && chain.first() == chain.last();
            if !is_closed && chain.len() >= 3 {
                let start = chain[0];
                junc_neighbors.entry(start).or_default().push(chain[1].to_point());
                let end = chain[chain.len() - 1];
                junc_neighbors.entry(end).or_default().push(chain[chain.len() - 2].to_point());
            }
        }
        let mut final_corrected: FxHashMap<NodeId, Point> = fx_hashmap();
        for (node, neighbors) in &junc_neighbors {
            if neighbors.len() == 2 {
                let jp = node.to_point();
                final_corrected.insert(*node, Point::new(
                    0.125 * neighbors[0].x + 0.75 * jp.x + 0.125 * neighbors[1].x,
                    0.125 * neighbors[0].y + 0.75 * jp.y + 0.125 * neighbors[1].y,
                ));
            }
        }
        (j, final_corrected)
    };
    let t3 = std::time::Instant::now();

    // Trace boundary loops, then optimize each loop directly.
    let all_loops = trace_all_boundary_loops(&directed_edges);
    let t4 = std::time::Instant::now();

    let mut node_positions: FxHashMap<NodeId, Point> = fx_hashmap();
    for (node_loop, _) in &all_loops {
        for nd in node_loop {
            node_positions.entry(*nd).or_insert_with(|| nd.to_point());
        }
    }

    optimize_boundary_loops(&all_loops, &mut node_positions, &junctions);
    let t5 = std::time::Instant::now();

    // Group loops by color and convert to segments
    let mut color_loops: BTreeMap<u32, Vec<Vec<PathSegment>>> = BTreeMap::new();
    for (node_loop, color) in &all_loops {
        let segs = boundary_loop_to_segments(node_loop, &node_positions, &junctions, &tjunc_corrected);
        if !segs.is_empty() {
            color_loops.entry(*color).or_default().push(segs);
        }
    }

    let mut result: Vec<ColorPath> = Vec::new();
    for (color, loop_segments) in color_loops {
        // Skip the void sentinel path (image perimeter loop)
        if color == VOID_COLOR { continue; }
        let mut all_segments = Vec::new();
        for segs in loop_segments {
            all_segments.extend(segs);
        }
        result.push(ColorPath {
            color,
            segments: all_segments,
        });
    }
    let t6 = std::time::Instant::now();

    if verbose {
        eprintln!("    cells:         {:>8.3}ms", (t1 - t0).as_secs_f64() * 1000.0);
        eprintln!("    boundary edges:{:>8.3}ms", (t2 - t1).as_secs_f64() * 1000.0);
        eprintln!("    chain+tjunc:   {:>8.3}ms", (t3 - t2).as_secs_f64() * 1000.0);
        eprintln!("    trace loops:   {:>8.3}ms", (t4 - t3).as_secs_f64() * 1000.0);
        eprintln!("    optimize:      {:>8.3}ms", (t5 - t4).as_secs_f64() * 1000.0);
        eprintln!("    bspline emit:  {:>8.3}ms", (t6 - t5).as_secs_f64() * 1000.0);
    }

    result
}

/// Extract boundary spans for edge-based rendering.
/// Each span is a B-spline segment shared by exactly two color regions,
/// eliminating gaps that occur with per-region path rendering.
///
/// Pipeline: cells → boundary edges → chains → optimize → B-spline fit.
///
/// Extract per-region ColorPaths using shared chain B-splines with winding fill.
///
/// Each boundary chain is shared between two regions. B-splines are fitted once
/// per chain, then each region's boundary loop is assembled from those shared
/// segments. Since both regions reference the same curve, boundaries are gap-free.
/// The scanline winding-number rasterizer then correctly fills each region.
pub fn extract_shared_edge_paths(
    pixels: &[u32], graph: &SimilarityGraph,
) -> (Vec<ColorPath>, u32) {
    extract_shared_edge_paths_inner(pixels, graph, false)
}

/// Dump CPU pipeline control points for visualization comparison with GPU.
/// Outputs to stdout: one line per boundary node with position and chain neighbors.
pub fn dump_cpu_control_points(pixels: &[u32], width: usize, height: usize) {
    let graph = super::graph::build(pixels, width, height);
    let w = graph.width;
    let h = graph.height;

    let all_cells = precompute_cells(w, h, &graph);
    let (visible_edges, directed_edges) = build_directed_boundary_edges(pixels, w, h, &all_cells, false);
    let chains = chain_visible_edges(&visible_edges);

    let mut junctions: HashSet<NodeId> = HashSet::new();
    for (chain, _) in &chains {
        let is_closed = chain.len() > 2 && chain.first() == chain.last();
        if !is_closed && chain.len() >= 2 {
            junctions.insert(chain[0]);
            junctions.insert(chain[chain.len() - 1]);
        }
    }

    let all_loops = trace_all_boundary_loops(&directed_edges);
    let mut node_positions: FxHashMap<NodeId, Point> = fx_hashmap();
    for (node_loop, _) in &all_loops {
        for nd in node_loop {
            node_positions.entry(*nd).or_insert_with(|| nd.to_point());
        }
    }
    optimize_boundary_loops(&all_loops, &mut node_positions, &junctions);

    // Build chain neighbor map: for each node, (prev, next) in chain
    let mut chain_nbrs: FxHashMap<NodeId, (Option<NodeId>, Option<NodeId>)> = fx_hashmap();
    for (chain, _) in &chains {
        let n = chain.len();
        for i in 0..n {
            let prev = if i > 0 { Some(chain[i - 1]) } else { None };
            let next = if i + 1 < n { Some(chain[i + 1]) } else { None };
            chain_nbrs.insert(chain[i], (prev, next));
        }
    }

    // Collect all unique boundary nodes
    let mut all_nodes: Vec<NodeId> = node_positions.keys().copied().collect();
    all_nodes.sort_by(|a, b| (a.y4, a.x4).cmp(&(b.y4, b.x4)));

    // Create a node-to-index map for the dump
    let mut node_idx: FxHashMap<NodeId, usize> = fx_hashmap();
    for (i, nd) in all_nodes.iter().enumerate() {
        node_idx.insert(*nd, i);
    }

    // Dump: idx px py 0 0 prev_idx next_idx flags
    for (i, nd) in all_nodes.iter().enumerate() {
        let p = node_positions.get(nd).copied().unwrap_or_else(|| nd.to_point());
        let (prev, next) = chain_nbrs.get(nd).copied().unwrap_or((None, None));
        let prev_i = prev.and_then(|n| node_idx.get(&n).copied()).map(|i| i as i32).unwrap_or(-1);
        let next_i = next.and_then(|n| node_idx.get(&n).copied()).map(|i| i as i32).unwrap_or(-1);
        let is_junction = junctions.contains(nd);
        let flag = if is_junction { 1u32 } else { 0u32 };
        println!("{} {:.4} {:.4} 0 0 {} {} {}", i, p.x, p.y, prev_i, next_i, flag);
    }
    eprintln!("CPU dump: {} boundary nodes, {} chains, {} junctions",
        all_nodes.len(), chains.len(), junctions.len());
}

/// Inner implementation with adaptive flag.
/// When adaptive=true and boundary count exceeds threshold, skips
/// chain building and optimization for faster (but less smooth) output.
pub fn extract_shared_edge_paths_inner(
    pixels: &[u32], graph: &SimilarityGraph, adaptive: bool,
) -> (Vec<ColorPath>, u32) {
    return extract_shared_edge_paths_gpu(pixels, graph, adaptive);
}

/// Inner implementation that builds shared-chain edge paths.
pub fn extract_shared_edge_paths_gpu(
    pixels: &[u32], graph: &SimilarityGraph, adaptive: bool,
) -> (Vec<ColorPath>, u32) {
    let w = graph.width;
    let h = graph.height;

    let all_cells = precompute_cells(w, h, graph);
    let (visible_edges, directed_edges) = build_directed_boundary_edges(pixels, w, h, &all_cells, adaptive);

    // Adaptive: when visible_edges is empty, boundary count exceeded threshold.
    // Skip chain building and optimization for faster output.
    let adaptive = adaptive && visible_edges.is_empty();

    if adaptive {
        let all_loops = trace_all_boundary_loops(&directed_edges);
        let mut color_loops: BTreeMap<u32, Vec<Vec<PathSegment>>> = BTreeMap::new();
        for (node_loop, color) in &all_loops {
            let points: Vec<Point> = node_loop.iter()
                .map(|nd| nd.to_point())
                .collect();
            if points.len() < 3 { continue; }
            let segs = bspline_closed(&points);
            if !segs.is_empty() {
                color_loops.entry(*color).or_default().push(segs);
            }
        }

        let mut result: Vec<ColorPath> = Vec::new();
        for (color, loop_segments) in color_loops {
            if color == VOID_COLOR { continue; }
            let mut all_segments = Vec::new();
            for segs in loop_segments {
                all_segments.extend(segs);
            }
            result.push(ColorPath { color, segments: all_segments });
        }
        let bg_color = detect_bg(pixels, w, h);
        return (result, bg_color);
    }

    // Build chains without T-junction merging.
    let chains = chain_visible_edges(&visible_edges);

    // Build junction set for optimization
    let mut junctions: HashSet<NodeId> = HashSet::new();
    for (chain, _) in &chains {
        let is_closed = chain.len() > 2 && chain.first() == chain.last();
        if !is_closed && chain.len() >= 2 {
            junctions.insert(chain[0]);
            junctions.insert(chain[chain.len() - 1]);
        }
    }

    // Optimize node positions
    let all_loops = trace_all_boundary_loops(&directed_edges);
    let mut node_positions: FxHashMap<NodeId, Point> = fx_hashmap();
    for (node_loop, _) in &all_loops {
        for nd in node_loop {
            node_positions.entry(*nd).or_insert_with(|| nd.to_point());
        }
    }

    optimize_boundary_loops(&all_loops, &mut node_positions, &junctions);

    // Fit B-splines to each chain
    let chain_segments: Vec<Vec<PathSegment>> = chains.iter().map(|(chain, _)| {
        let n = chain.len();
        if n < 2 { return Vec::new(); }
        let points: Vec<Point> = chain.iter()
            .map(|nd| node_positions.get(nd).copied().unwrap_or_else(|| nd.to_point()))
            .collect();
        let is_closed = n > 2 && chain.first() == chain.last();
        if is_closed {
            bspline_closed(&points[..n - 1])
        } else {
            bspline_open(&points)
        }
    }).collect();

    // Build edge→chain lookup: for each directed edge (a,b), store (chain_index, position_in_chain).
    // Position is the index of node `a` in the chain (so the edge is chain[pos]→chain[pos+1]).
    let mut edge_to_chain: FxHashMap<(NodeId, NodeId), (usize, usize, bool)> = fx_hashmap_cap(visible_edges.len() * 2);
    for (ci, (chain, _)) in chains.iter().enumerate() {
        let n = chain.len();
        for i in 0..n.saturating_sub(1) {
            // Forward: chain[i]→chain[i+1], position i
            edge_to_chain.entry((chain[i], chain[i + 1]))
                .or_insert((ci, i, true));
            // Reversed: chain[i+1]→chain[i], position i, reversed
            edge_to_chain.entry((chain[i + 1], chain[i]))
                .or_insert((ci, i, false));
        }
    }

    // Build a chain lookup by first node pair: given the first edge of a chain
    // traversal, find the chain and direction.
    // Key: (first_node, second_node) of the chain traversal
    // Value: (chain_index, forward: bool)
    let mut chain_by_entry: FxHashMap<(NodeId, NodeId), (usize, bool)> = fx_hashmap_cap(chains.len() * 2);
    for (ci, (chain, _)) in chains.iter().enumerate() {
        let n = chain.len();
        if n < 2 { continue; }
        let is_closed = n > 2 && chain.first() == chain.last();
        if is_closed { continue; } // closed chains aren't split at junctions
        // Forward entry: first two nodes
        chain_by_entry.insert((chain[0], chain[1]), (ci, true));
        // Reverse entry: last two nodes reversed
        chain_by_entry.insert((chain[n - 1], chain[n - 2]), (ci, false));
    }

    // For each boundary loop, assemble a ColorPath from shared chain segments.
    //
    // Strategy: walk the loop node by node. At each junction node, look up
    // the chain that starts with (junction, next_node). Emit that chain's
    // entire B-spline (forward or reversed). Skip ahead past all the chain's
    // interior nodes to the next junction.
    let mut color_loops: BTreeMap<u32, Vec<Vec<PathSegment>>> = BTreeMap::new();

    for (node_loop, color) in &all_loops {
        let n = node_loop.len();
        if n < 3 { continue; }

        // Rotate the loop to start at a junction node so we always enter
        // chains at their endpoints (where chain_by_entry can match).
        let rotation = node_loop.iter().position(|nd| junctions.contains(nd)).unwrap_or(0);
        let rotated: Vec<NodeId> = node_loop[rotation..].iter()
            .chain(node_loop[..rotation].iter())
            .copied().collect();
        let node_loop = &rotated;

        let mut segs = Vec::new();
        let mut i = 0;

        while i < n {
            let a = node_loop[i];
            let b = node_loop[(i + 1) % n];

            if let Some(&(ci, forward)) = chain_by_entry.get(&(a, b)) {
                let chain = &chains[ci].0;
                let c_segs = &chain_segments[ci];
                let chain_edges = chain.len() - 1; // number of edges in chain

                if forward {
                    // Emit all segments forward
                    for seg in c_segs.iter() {
                        segs.push(seg.clone());
                    }
                } else {
                    // Emit all segments reversed
                    for seg in c_segs.iter().rev() {
                        segs.push(reverse_segment(seg));
                    }
                }
                // Skip past the chain's edges in the loop
                i += chain_edges;
            } else {
                // This edge isn't the start of any chain — it might be a
                // single-edge chain (2 nodes) or a closed chain.
                // Look it up in the general edge→chain map.
                if let Some(&(ci, _pos, forward)) = edge_to_chain.get(&(a, b)) {
                    let c_segs = &chain_segments[ci];
                    let chain = &chains[ci].0;
                    let is_closed = chain.len() > 2 && chain.first() == chain.last();

                    if is_closed {
                        // Closed chain: the entire loop IS this chain
                        if forward {
                            for seg in c_segs.iter() {
                                segs.push(seg.clone());
                            }
                        } else {
                            for seg in c_segs.iter().rev() {
                                segs.push(reverse_segment(seg));
                            }
                        }
                        i += chain.len() - 1;
                    } else {
                        // Single edge or mid-chain entry — emit just this one
                        // segment as a fallback line
                        let pa = node_positions.get(&a).copied().unwrap_or_else(|| a.to_point());
                        let pb = node_positions.get(&b).copied().unwrap_or_else(|| b.to_point());
                        segs.push(PathSegment::Line(pa, pb));
                        i += 1;
                    }
                } else {
                    // Edge not in any chain — straight line fallback
                    let pa = node_positions.get(&a).copied().unwrap_or_else(|| a.to_point());
                    let pb = node_positions.get(&b).copied().unwrap_or_else(|| b.to_point());
                    segs.push(PathSegment::Line(pa, pb));
                    i += 1;
                }
            }
        }

        if !segs.is_empty() {
            color_loops.entry(*color).or_default().push(segs);
        }
    }

    let mut result: Vec<ColorPath> = Vec::new();
    for (color, loop_segments) in color_loops {
        if color == VOID_COLOR { continue; }
        let mut all_segments = Vec::new();
        for segs in loop_segments {
            all_segments.extend(segs);
        }
        result.push(ColorPath { color, segments: all_segments });
    }

    let bg_color = detect_bg(pixels, w, h);
    (result, bg_color)
}

fn reverse_segment(seg: &PathSegment) -> PathSegment {
    match seg {
        PathSegment::Line(a, b) => PathSegment::Line(*b, *a),
        PathSegment::QuadBezier(a, c, b) => PathSegment::QuadBezier(*b, *c, *a),
    }
}

/// Simple background color detection (most common edge color).
fn detect_bg(pixels: &[u32], w: usize, h: usize) -> u32 {
    let mut counts: FxHashMap<u32, u32> = fx_hashmap();
    for x in 0..w {
        *counts.entry(pixels[x]).or_insert(0) += 1;
        *counts.entry(pixels[(h - 1) * w + x]).or_insert(0) += 1;
    }
    for y in 1..h - 1 {
        *counts.entry(pixels[y * w]).or_insert(0) += 1;
        *counts.entry(pixels[y * w + w - 1]).or_insert(0) += 1;
    }
    counts.into_iter().max_by_key(|&(_, c)| c).map(|(color, _)| color).unwrap_or(0)
}


// --- Loop optimization (Paper Section 3.4) ---

/// Optimize boundary loop paths directly using gradient descent.
/// Works on full closed loops instead of short chains, so the optimizer
/// sees the complete contour shape for each color region.
/// Junction nodes (valence >= 3) are fixed.
fn optimize_boundary_loops(
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
        // Paper Section 3.4, Figure 7: detect corners via ×4 grid template matching.
        // Corner nodes are NOT pinned — they can still move during optimization.
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

const OPT_ITERATIONS: usize = 1;
const MAX_MOVE: f64 = 0.25;
const CURVATURE_INTERVALS: usize = 3;

/// Detect corner patterns using Kopf-Lischinski template matching (Section 3.4, Figure 7).
///
/// On the ×4 quantized grid, sharp features take on a finite set of patterns
/// (the paper's Figure 7, including all rotations and reflections). We detect
/// these by checking if the turn angle at each node ≥ 60° using exact integer
/// arithmetic on the ×4 coordinates. This is equivalent to the paper's pattern
/// enumeration since all sharp patterns on the quantized grid have angles ≥ 60°.
///
/// Unlike the previous implementation which pinned corner nodes entirely,
/// the paper only excludes B-spline spans near corners from the curvature
/// integral — corner nodes can still move during optimization.
fn detect_corners_from_nodes(nodes: &[NodeId], is_closed: bool) -> Vec<bool> {
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

        // Edge vectors in ×4 integer coordinates
        let d1x = (curr.x4 - prev.x4) as i64;
        let d1y = (curr.y4 - prev.y4) as i64;
        let d2x = (next.x4 - curr.x4) as i64;
        let d2y = (next.y4 - curr.y4) as i64;

        let dot = d1x * d2x + d1y * d2y;

        if dot <= 0 {
            // Turn angle ≥ 90° — always a corner
            is_corner[i] = true;
        } else {
            // Check if turn angle ≥ 60° using integer arithmetic:
            // cos(angle) ≤ 0.5  ↔  4 * dot² ≤ |d1|² * |d2|²
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
/// Returns (∂E/∂x, ∂E/∂y).
///
/// Positional energy: E_pos = (s² * d²)²
///   ∇E_pos = 4 * s⁴ * d² * (p - p_orig)
///
/// Curvature energy per span (p0, p1, p2): E_curv ≈ ||p0 - 2p1 + p2||²
///   ∇E_curv w.r.t. p1 (center) = 2*(4p1 - 2p0 - 2p2)
///   ∇E_curv w.r.t. p0 (start)  = 2*(p0 - 2p1 + p2)
///   ∇E_curv w.r.t. p2 (end)    = 2*(p0 - 2p1 + p2)
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
            // This node is the center of the span → ∂/∂p1 of ||dd||² = -4*dd
            gx += 2.0 * (4.0 * p1.x - 2.0 * p0.x - 2.0 * p2.x);
            gy += 2.0 * (4.0 * p1.y - 2.0 * p0.y - 2.0 * p2.y);
        } else {
            // This node is p0 or p2 → ∂/∂p0 or ∂/∂p2 of ||dd||² = 2*dd
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

/// Positional energy: (2.5 × ‖Δ‖)⁴ = 2.5⁴ × ‖Δ‖⁴ ≈ 39.06 × ‖Δ‖⁴.
///
/// The 2.5 scaling factor matches the reference implementation
/// (Depixelizing Pixel Art on GPUs, OptimizeEnergy.vert line 84).
/// The paper specifies ‖Δ‖⁴ without the scaling, but the reference
/// uses 2.5× which keeps nodes much closer to their original positions,
/// preventing over-smoothing of intentional pixel-art features.
const POSITIONAL_SCALE: f64 = 2.5;

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

/// Integrate κ² over one quadratic B-spline span.
///
/// NOTE: The paper (Equation 3) defines smoothness energy as ∫|κ(s)| ds
/// (absolute curvature integrated over arc length). We use ∫κ² instead
/// because it penalizes curvature more aggressively per iteration,
/// producing visually smooth results with just 1 optimization pass.
/// The paper's |κ| requires many stochastic iterations to converge
/// (~0.6s per their timing table), which is too slow for real-time use.
/// With GB's discrete palette colors, the difference in output quality
/// between κ² × 1 iteration and |κ| × many iterations is negligible.
///
/// For a quadratic Bezier with control points p0, p1, p2:
///   d'(t)  = (t-1)*p0 + (1-2t)*p1 + t*p2      (first derivative)
///   d''(t) = p0 - 2*p1 + p2                     (second derivative, constant)
///   κ(t)   = (d' × d'') / |d'|³                 (signed curvature)
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

/// Compute κ²(t) = (d' × d'')² / |d'|⁶ for one sample point.
#[inline(always)]
fn curvature_sq_at(p0: Point, p1: Point, p2: Point, t: f64, ddx: f64, ddy: f64) -> f64 {
    let dx = (t - 1.0) * p0.x + (1.0 - 2.0 * t) * p1.x + t * p2.x;
    let dy = (t - 1.0) * p0.y + (1.0 - 2.0 * t) * p1.y + t * p2.y;

    let numer = dx * ddy - dy * ddx;
    let denom_sq = dx * dx + dy * dy;
    let denom = denom_sq * denom_sq.sqrt();
    if denom < 1e-12 { 0.0 } else { (numer * numer) / (denom * denom) }
}
