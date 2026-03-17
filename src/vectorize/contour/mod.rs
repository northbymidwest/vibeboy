//! Kopf-Lischinski pixel-art vectorization: Sections 3.2-3.4.
//!
//! Pipeline:
//! 1. Build reshaped cell graph (generalized Voronoi diagram, Section 3.2)
//! 2. Extract visible edges between different-color cells (Section 3.3)
//! 3. Chain visible edges through valence-2 nodes into paths (Section 3.3)
//! 4. Merge chains at T-junctions with aligned tangents (Section 3.3)
//! 5. Optimize B-spline control points (Section 3.4)
//! 6. Flood-fill same-color regions, trace boundaries with smooth B-spline curves
//! 7. Emit region outlines as ColorPaths

mod cells;
mod edges;
mod loops;
mod optimize;

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

/// A node in the reshaped cell graph, identified by quantized coordinates.
/// We quantize to 1/4 pixel to handle the +/-0.25 offsets.
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


// --- Main entry point ---

/// Vectorize with B-spline smoothing (Sections 3.2-3.4).
/// Region-based rendering: merges same-color cells into regions, traces
/// boundaries on the cell graph, emits smooth B-spline curves as region outlines.
pub fn extract_cells_smooth(pixels: &[u32], graph: &SimilarityGraph, adaptive: bool) -> Vec<ColorPath> {
    let w = graph.width;
    let h = graph.height;
    let verbose = std::env::var("VECTORIZE_BENCH").is_ok();
    let t0 = std::time::Instant::now();

    let all_cells = cells::precompute_cells(w, h, graph);
    let t1 = std::time::Instant::now();

    let (visible_edges, directed_edges) = edges::build_directed_boundary_edges(pixels, w, h, &all_cells, adaptive);
    let t2 = std::time::Instant::now();

    // Adaptive pipeline: skip expensive B-spline optimization when boundary
    // complexity is high (noisy/dithered frames). The optimization provides
    // negligible visual benefit when boundaries are this dense.
    // visible_edges is empty when adaptive mode was triggered in build_directed_boundary_edges.
    let adaptive = visible_edges.is_empty();

    if adaptive {
        // Adaptive path: trace loops but skip chain+tjunc+optimize.
        // Use B-spline fitting on traced loops for smooth boundaries.
        let all_loops = loops::trace_all_boundary_loops(&directed_edges);
        let optimized: FxHashMap<NodeId, Point> = fx_hashmap();
        let junctions: HashSet<NodeId> = HashSet::new();

        let mut color_loops: BTreeMap<u32, Vec<Vec<PathSegment>>> = BTreeMap::new();
        for (node_loop, color) in &all_loops {
            let no_tjunc: FxHashMap<NodeId, Point> = fx_hashmap();
            let segs = loops::boundary_loop_to_segments(node_loop, &optimized, &junctions, &no_tjunc);
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
        let mut chains = edges::chain_visible_edges(&visible_edges);
        edges::merge_t_junctions(&mut chains);
        let mut j: HashSet<NodeId> = HashSet::new();
        let mut corrected: FxHashMap<NodeId, Point> = fx_hashmap();
        for (chain, _) in &chains {
            let is_closed = chain.len() > 2 && chain.first() == chain.last();
            if !is_closed && chain.len() >= 2 {
                // Start endpoint: junction with next node as contour direction
                let jn = chain[0];
                j.insert(jn);
                if chain.len() >= 3 {
                    let jp = jn.to_point();
                    let np = chain[1].to_point();
                    corrected.entry(jn).and_modify(|existing| {
                        let first_neighbor = *existing;
                        *existing = Point::new(
                            0.125 * first_neighbor.x + 0.75 * jp.x + 0.125 * np.x,
                            0.125 * first_neighbor.y + 0.75 * jp.y + 0.125 * np.y,
                        );
                    }).or_insert(np);
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
    let all_loops = loops::trace_all_boundary_loops(&directed_edges);
    let t4 = std::time::Instant::now();

    let mut node_positions: FxHashMap<NodeId, Point> = fx_hashmap();
    for (node_loop, _) in &all_loops {
        for nd in node_loop {
            node_positions.entry(*nd).or_insert_with(|| nd.to_point());
        }
    }

    optimize::optimize_boundary_loops(&all_loops, &mut node_positions, &junctions);
    let t5 = std::time::Instant::now();

    // Group loops by color and convert to segments
    let mut color_loops: BTreeMap<u32, Vec<Vec<PathSegment>>> = BTreeMap::new();
    for (node_loop, color) in &all_loops {
        let segs = loops::boundary_loop_to_segments(node_loop, &node_positions, &junctions, &tjunc_corrected);
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

/// Extract per-region ColorPaths using shared chain B-splines with winding fill.
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

    let all_cells = cells::precompute_cells(w, h, &graph);
    let (visible_edges, directed_edges) = edges::build_directed_boundary_edges(pixels, w, h, &all_cells, false);
    let chains = edges::chain_visible_edges(&visible_edges);

    let mut junctions: HashSet<NodeId> = HashSet::new();
    for (chain, _) in &chains {
        let is_closed = chain.len() > 2 && chain.first() == chain.last();
        if !is_closed && chain.len() >= 2 {
            junctions.insert(chain[0]);
            junctions.insert(chain[chain.len() - 1]);
        }
    }

    let all_loops = loops::trace_all_boundary_loops(&directed_edges);
    let mut node_positions: FxHashMap<NodeId, Point> = fx_hashmap();
    for (node_loop, _) in &all_loops {
        for nd in node_loop {
            node_positions.entry(*nd).or_insert_with(|| nd.to_point());
        }
    }
    optimize::optimize_boundary_loops(&all_loops, &mut node_positions, &junctions);

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

    let all_cells = cells::precompute_cells(w, h, graph);
    let (visible_edges, directed_edges) = edges::build_directed_boundary_edges(pixels, w, h, &all_cells, adaptive);

    // Adaptive: when visible_edges is empty, boundary count exceeded threshold.
    // Skip chain building and optimization for faster output.
    let adaptive = adaptive && visible_edges.is_empty();

    if adaptive {
        let all_loops = loops::trace_all_boundary_loops(&directed_edges);
        let mut color_loops: BTreeMap<u32, Vec<Vec<PathSegment>>> = BTreeMap::new();
        for (node_loop, color) in &all_loops {
            let points: Vec<Point> = node_loop.iter()
                .map(|nd| nd.to_point())
                .collect();
            if points.len() < 3 { continue; }
            let segs = optimize::bspline_closed(&points);
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
    let chains = edges::chain_visible_edges(&visible_edges);

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
    let all_loops = loops::trace_all_boundary_loops(&directed_edges);
    let mut node_positions: FxHashMap<NodeId, Point> = fx_hashmap();
    for (node_loop, _) in &all_loops {
        for nd in node_loop {
            node_positions.entry(*nd).or_insert_with(|| nd.to_point());
        }
    }

    optimize::optimize_boundary_loops(&all_loops, &mut node_positions, &junctions);

    // Fit B-splines to each chain
    let chain_segments: Vec<Vec<PathSegment>> = chains.iter().map(|(chain, _)| {
        let n = chain.len();
        if n < 2 { return Vec::new(); }
        let points: Vec<Point> = chain.iter()
            .map(|nd| node_positions.get(nd).copied().unwrap_or_else(|| nd.to_point()))
            .collect();
        let is_closed = n > 2 && chain.first() == chain.last();
        if is_closed {
            optimize::bspline_closed(&points[..n - 1])
        } else {
            optimize::bspline_open(&points)
        }
    }).collect();

    // Build edge->chain lookup
    let mut edge_to_chain: FxHashMap<(NodeId, NodeId), (usize, usize, bool)> = fx_hashmap_cap(visible_edges.len() * 2);
    for (ci, (chain, _)) in chains.iter().enumerate() {
        let n = chain.len();
        for i in 0..n.saturating_sub(1) {
            edge_to_chain.entry((chain[i], chain[i + 1]))
                .or_insert((ci, i, true));
            edge_to_chain.entry((chain[i + 1], chain[i]))
                .or_insert((ci, i, false));
        }
    }

    let mut chain_by_entry: FxHashMap<(NodeId, NodeId), (usize, bool)> = fx_hashmap_cap(chains.len() * 2);
    for (ci, (chain, _)) in chains.iter().enumerate() {
        let n = chain.len();
        if n < 2 { continue; }
        let is_closed = n > 2 && chain.first() == chain.last();
        if is_closed { continue; }
        chain_by_entry.insert((chain[0], chain[1]), (ci, true));
        chain_by_entry.insert((chain[n - 1], chain[n - 2]), (ci, false));
    }

    // For each boundary loop, assemble a ColorPath from shared chain segments.
    let mut color_loops: BTreeMap<u32, Vec<Vec<PathSegment>>> = BTreeMap::new();

    for (node_loop, color) in &all_loops {
        let n = node_loop.len();
        if n < 3 { continue; }

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
                let chain_edges = chain.len() - 1;

                if forward {
                    for seg in c_segs.iter() {
                        segs.push(seg.clone());
                    }
                } else {
                    for seg in c_segs.iter().rev() {
                        segs.push(reverse_segment(seg));
                    }
                }
                i += chain_edges;
            } else {
                if let Some(&(ci, _pos, forward)) = edge_to_chain.get(&(a, b)) {
                    let c_segs = &chain_segments[ci];
                    let chain = &chains[ci].0;
                    let is_closed = chain.len() > 2 && chain.first() == chain.last();

                    if is_closed {
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
                        let pa = node_positions.get(&a).copied().unwrap_or_else(|| a.to_point());
                        let pb = node_positions.get(&b).copied().unwrap_or_else(|| b.to_point());
                        segs.push(PathSegment::Line(pa, pb));
                        i += 1;
                    }
                } else {
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
