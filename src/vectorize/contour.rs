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

/// Inline buffer for pixel cell vertices (max 8).
struct CellBuf {
    pts: [Point; 8],
    len: usize,
}

impl CellBuf {
    #[inline]
    fn new() -> Self {
        CellBuf { pts: [Point::new(0.0, 0.0); 8], len: 0 }
    }
    #[inline]
    fn push(&mut self, p: Point) {
        self.pts[self.len] = p;
        self.len += 1;
    }
    #[inline]
    fn as_slice(&self) -> &[Point] {
        &self.pts[..self.len]
    }
}

/// Compute the full Voronoi cell polygon for pixel (px, py).
/// Returns vertices in CW order, 4–8 vertices. Uses inline buffer.
#[inline]
fn pixel_cell_inline(px: usize, py: usize, graph: &SimilarityGraph, buf: &mut CellBuf) {
    let ipx = px as i32;
    let ipy = py as i32;
    buf.len = 0;
    corner_vertices_inline(ipx, ipy, CornerRel::BR, graph, buf);
    corner_vertices_inline(ipx + 1, ipy, CornerRel::BL, graph, buf);
    corner_vertices_inline(ipx + 1, ipy + 1, CornerRel::TL, graph, buf);
    corner_vertices_inline(ipx, ipy + 1, CornerRel::TR, graph, buf);
}

/// Same as corner_vertices but uses CellBuf instead of Vec.
#[inline]
fn corner_vertices_inline(
    cx: i32, cy: i32, rel: CornerRel, graph: &SimilarityGraph, out: &mut CellBuf,
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
    let mut buf = CellBuf::new();
    pixel_cell_inline(px, py, graph, &mut buf);
    buf.as_slice().to_vec()
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
    // Sort-merge approach: collect all half-edges, sort by canonical key, merge
    // adjacent pairs. Cache-friendlier than HashMap for large images.
    let stride = (4 * w + 4) as u32;

    // Each pixel has 4-8 cell edges. Collect as (canonical_key, node_a, node_b, color, is_forward).
    let mut half_edges: Vec<(u64, NodeId, NodeId, u32, bool)> = Vec::with_capacity(w * h * 5);

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
                half_edges.push((key, na, nb, color, is_forward));
            }
        }
    }

    // Sort by canonical key — groups matching half-edges adjacent
    half_edges.sort_unstable_by_key(|e| e.0);

    // Merge adjacent pairs and emit boundary edges
    let mut boundary_count = 0usize;
    let mut i = 0;
    let len = half_edges.len();

    // First pass: count boundaries
    while i < len {
        let key = half_edges[i].0;
        let mut left = VOID_COLOR;
        let mut right = VOID_COLOR;
        let j = i;
        while i < len && half_edges[i].0 == key {
            if half_edges[i].4 { right = half_edges[i].3; }
            else { left = half_edges[i].3; }
            i += 1;
        }
        if is_visible_edge(left, right) { boundary_count += 1; }
    }

    let adaptive = adaptive && boundary_count > ADAPTIVE_EDGE_THRESHOLD;
    let mut directed = Vec::with_capacity(boundary_count * 2);
    let mut visible = if adaptive { Vec::new() } else { Vec::with_capacity(boundary_count) };

    // Second pass: emit edges
    i = 0;
    while i < len {
        let key = half_edges[i].0;
        let na = half_edges[i].1;
        let nb = half_edges[i].2;
        let mut left = VOID_COLOR;
        let mut right = VOID_COLOR;
        while i < len && half_edges[i].0 == key {
            if half_edges[i].4 { right = half_edges[i].3; }
            else { left = half_edges[i].3; }
            i += 1;
        }
        if !is_visible_edge(left, right) { continue; }
        directed.push((na, nb, right));
        directed.push((nb, na, left));
        if !adaptive {
            visible.push(CellEdge { a: na, b: nb, left_color: left, right_color: right });
        }
    }

    if !adaptive {
        // Already sorted by key from the sort above
        directed.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    }

    (visible, directed)
}

/// Find boundary edges for a region using precomputed cells.
fn region_boundary_edges_fast(
    region_pixels: &[(usize, usize)],
    w: usize,
    all_cells: &[InlineCell],
) -> Vec<(NodeId, NodeId)> {
    let mut all_directed: Vec<(NodeId, NodeId)> = Vec::new();
    let mut undirected_count: FxHashMap<(NodeId, NodeId), usize> = fx_hashmap();

    for &(px, py) in region_pixels {
        let cell = all_cells[py * w + px].as_slice();
        let n = cell.len();
        if n < 3 { continue; }

        for i in 0..n {
            let pa = cell[i];
            let pb = cell[(i + 1) % n];
            all_directed.push((pa, pb));
            let key = if pa <= pb { (pa, pb) } else { (pb, pa) };
            *undirected_count.entry(key).or_insert(0) += 1;
        }
    }

    all_directed
        .into_iter()
        .filter(|&(pa, pb)| {
            let key = if pa <= pb { (pa, pb) } else { (pb, pa) };
            undirected_count[&key] == 1
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

    let mut edge_map: FxHashMap<(NodeId, NodeId), (Option<u32>, Option<u32>)> = fx_hashmap();

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
        let lc = left.unwrap_or(VOID_COLOR);
        let rc = right.unwrap_or(VOID_COLOR);
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

// --- Optimized node positions ---

/// Build map from NodeId to optimized Point position, and collect junction nodes.
/// Junction nodes are endpoints of open chains (valence >= 3 in visible edge graph).
fn build_optimized_positions(chains: &[Vec<NodeId>])
    -> (FxHashMap<NodeId, Point>, HashSet<NodeId>)
{
    let mut positions: FxHashMap<NodeId, Point> = fx_hashmap();
    let mut junctions: HashSet<NodeId> = HashSet::new();

    for chain in chains {
        let is_closed = chain.len() > 2 && chain.first() == chain.last();
        let ctrl_nodes = if is_closed { &chain[..chain.len() - 1] } else { &chain[..] };

        if !is_closed && chain.len() >= 2 {
            junctions.insert(chain[0]);
            junctions.insert(chain[chain.len() - 1]);
        }

        let mut points: Vec<Point> = ctrl_nodes.iter().map(|n| n.to_point()).collect();
        if points.len() >= 4 {
            optimize_control_points(&mut points, is_closed);
        }

        for (i, node) in ctrl_nodes.iter().enumerate() {
            positions.insert(*node, points[i]);
        }
    }

    (positions, junctions)
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
    let mut undirected_count: FxHashMap<(NodeId, NodeId), usize> = fx_hashmap();

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
    // Build outgoing edges sorted by angle at each node.
    // Store (dest, dx, dy, edge_index) to avoid a separate edge_to_idx HashMap.
    let mut outgoing: FxHashMap<NodeId, Vec<(NodeId, i32, i32, u32)>> =
        fx_hashmap_cap(directed_edges.len());
    for (i, &(a, b, _)) in directed_edges.iter().enumerate() {
        outgoing.entry(a).or_default().push((b, b.x4 - a.x4, b.y4 - a.y4, i as u32));
    }
    for (_, edges) in outgoing.iter_mut() {
        edges.sort_unstable_by(|a, b| angle_cmp(a.1, a.2, b.1, b.2));
    }

    // Build next-edge index array using planar face algorithm.
    let n_edges = directed_edges.len();
    let mut next_idx: Vec<u32> = vec![u32::MAX; n_edges];

    for (i, &(p, c, _)) in directed_edges.iter().enumerate() {
        let rdx = p.x4 - c.x4;
        let rdy = p.y4 - c.y4;

        if let Some(edges) = outgoing.get(&c) {
            let pos = edges.partition_point(|e|
                angle_cmp(e.1, e.2, rdx, rdy) == std::cmp::Ordering::Less
            );
            let prev_idx = if pos == 0 { edges.len() - 1 } else { pos - 1 };
            next_idx[i] = edges[prev_idx].3;
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

/// Trace boundary edges into closed node loops using the planar face algorithm.
/// At each node, picks the rightmost turn (most CW) to keep region on the right.
fn trace_boundary_node_loops(boundary_edges: &[(NodeId, NodeId)]) -> Vec<Vec<NodeId>> {
    // Build outgoing edges sorted by angle at each node
    let mut outgoing: FxHashMap<NodeId, Vec<(NodeId, f64)>> = fx_hashmap();
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
    let mut next_map: FxHashMap<(NodeId, NodeId), (NodeId, NodeId)> = fx_hashmap();

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
/// Fast path: emit Line segments between consecutive grid-aligned nodes.
/// Used for noisy frames where B-spline smoothing adds no visible benefit.
fn boundary_loop_to_line_segments(nodes: &[NodeId]) -> Vec<PathSegment> {
    let n = nodes.len();
    if n < 2 { return Vec::new(); }
    let mut segs = Vec::with_capacity(n);
    for i in 0..n {
        let a = nodes[i].to_point();
        let b = nodes[(i + 1) % n].to_point();
        segs.push(PathSegment::Line(a, b));
    }
    segs
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

    let corner_indices: Vec<usize> = (0..n)
        .filter(|&i| junctions.contains(&nodes[i]))
        .collect();

    if corner_indices.is_empty() {
        return bspline_closed(&points);
    }

    // Helper: get the corrected position for a junction endpoint, or the
    // original position if no T-junction correction is available.
    let endpoint_pos = |idx: usize| -> Point {
        tjunc_corrected.get(&nodes[idx]).copied().unwrap_or(points[idx])
    };

    if corner_indices.len() == 1 {
        let c = corner_indices[0];
        let mut span_points = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let pt_idx = (c + i) % n;
            // Use corrected position at junction endpoints (first and last)
            if i == 0 || i == n {
                span_points.push(endpoint_pos(pt_idx));
            } else {
                span_points.push(points[pt_idx]);
            }
        }
        return bspline_open(&span_points);
    }

    let mut segments = Vec::new();
    let num_corners = corner_indices.len();

    for ci in 0..num_corners {
        let start = corner_indices[ci];
        let end = corner_indices[(ci + 1) % num_corners];

        let mut span_points = Vec::new();
        let mut idx = start;
        loop {
            // Use corrected position at junction endpoints
            if idx == start || idx == end {
                span_points.push(endpoint_pos(idx));
            } else {
                span_points.push(points[idx]);
            }
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

        let eps = GRADIENT_STEP;
        for _iter in 0..OPT_ITERATIONS {
            for i in 0..n {
                if pinned[i] { continue; }

                let current = pts[i];
                let e0 = local_energy(&pts, &orig, &corners, i, n, true);
                if e0 < 1e-12 { continue; }

                pts[i] = Point::new(current.x + eps, current.y);
                let ex_fwd = local_energy(&pts, &orig, &corners, i, n, true);
                pts[i] = Point::new(current.x, current.y + eps);
                let ey_fwd = local_energy(&pts, &orig, &corners, i, n, true);

                let inv_eps = 1.0 / eps;
                let gx = (ex_fwd - e0) * inv_eps;
                let gy = (ey_fwd - e0) * inv_eps;

                let grad_len = (gx * gx + gy * gy).sqrt();
                if grad_len < 1e-12 {
                    pts[i] = current;
                    continue;
                }

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

const OPT_ITERATIONS: usize = 1;
const GRADIENT_STEP: f64 = 0.01;
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

/// Float-based corner detection for standalone chain optimization (legacy path).
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

    // Check if chain is nearly straight — skip optimization if so
    let mut max_deviation = 0.0f64;
    if !is_closed && n >= 2 {
        let (ax, ay) = (points[0].x, points[0].y);
        let (bx, by) = (points[n - 1].x, points[n - 1].y);
        let dx = bx - ax;
        let dy = by - ay;
        let len_sq = dx * dx + dy * dy;
        if len_sq > 1e-12 {
            for p in points.iter() {
                let t = ((p.x - ax) * dx + (p.y - ay) * dy) / len_sq;
                let proj_x = ax + t * dx;
                let proj_y = ay + t * dy;
                let ex = p.x - proj_x;
                let ey = p.y - proj_y;
                max_deviation = max_deviation.max(ex * ex + ey * ey);
            }
        }
    }
    // Skip nearly-straight chains (deviation < 0.1 pixel)
    if !is_closed && max_deviation < 0.01 { return; }

    let eps = GRADIENT_STEP;

    for _iter in 0..OPT_ITERATIONS {
        for i in 0..n {
            if corners[i] { continue; }

            let current = points[i];
            let e0 = local_energy(points, &orig, &corners, i, n, is_closed);
            if e0 < 1e-12 { continue; }

            // Forward-difference gradient (3 evals instead of 5)
            points[i] = Point::new(current.x + eps, current.y);
            let ex_fwd = local_energy(points, &orig, &corners, i, n, is_closed);
            points[i] = Point::new(current.x, current.y + eps);
            let ey_fwd = local_energy(points, &orig, &corners, i, n, is_closed);

            let inv_eps = 1.0 / eps;
            let gx = (ex_fwd - e0) * inv_eps;
            let gy = (ey_fwd - e0) * inv_eps;

            let grad_len = (gx * gx + gy * gy).sqrt();
            if grad_len < 1e-12 {
                points[i] = current;
                continue;
            }

            // Normalized gradient step, clamped to MAX_MOVE
            let step = (e0 / grad_len).min(MAX_MOVE);
            let nx = current.x - step * gx / grad_len;
            let ny = current.y - step * gy / grad_len;

            // Accept if it improves energy
            let candidate = Point::new(nx, ny);
            points[i] = candidate;
            let e_new = local_energy(points, &orig, &corners, i, n, is_closed);
            if e_new >= e0 {
                points[i] = current; // revert
            }
        }
    }
}

#[inline(always)]
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
