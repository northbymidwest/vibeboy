//! SVG export for the vectorize-gpu pipeline.
//!
//! Builds filled vector regions from the GPU pipeline's resolved similarity graph
//! and optimized B-spline control points. Uses Voronoi cell boundary walking
//! (same as the CPU contour pipeline) to create a proper planar subdivision,
//! then traces faces with the planar face algorithm.
//!
//! Boundary curves use optimized positions with sharp corners at T-junctions
//! and crossings.

use std::collections::BTreeMap;
use vibeboy::scaling::vectorize::VectorizeData;

const IS_TJUNCTION: u32 = 32;
const IS_CROSSING: u32 = 64;
const SHARP_MASK: u32 = IS_TJUNCTION | IS_CROSSING;

/// Sentinel color for the void outside the image boundary.
const VOID_COLOR: u32 = 0x01000000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fmt(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        let s = format!("{:.4}", v);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

fn hex(c: u32) -> String {
    format!("#{:02X}{:02X}{:02X}", (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF)
}

/// Pack x4/y4 coordinates into a u64 node identifier.
fn pack_node(x4: i32, y4: i32) -> u64 {
    ((x4 as u64) << 32) | (y4 as u32 as u64)
}

/// Unpack a u64 node identifier into x4/y4 coordinates.
fn unpack_node(nid: u64) -> (i32, i32) {
    ((nid >> 32) as i32, nid as u32 as i32)
}

/// Cross-product angle comparison (exact integer, matches contour/loops.rs).
#[inline]
fn angle_cmp(adx: i64, ady: i64, bdx: i64, bdy: i64) -> std::cmp::Ordering {
    let ha = ady > 0 || (ady == 0 && adx > 0);
    let hb = bdy > 0 || (bdy == 0 && bdx > 0);
    if ha != hb {
        return if ha { std::cmp::Ordering::Less } else { std::cmp::Ordering::Greater };
    }
    (adx * bdy - ady * bdx).cmp(&0).reverse()
}

// ---------------------------------------------------------------------------
// Voronoi cell boundary → directed edges
// ---------------------------------------------------------------------------

/// Get diagonal state at grid corner (cx, cy) from the GPU resolved graph.
/// Returns 0=none, 1=backslash, 2=slash.
fn corner_diag(graph: &[u32], w: usize, h: usize, cx: usize, cy: usize) -> u8 {
    if cx == 0 || cy == 0 || cx >= w || cy >= h { return 0; }
    let stride = 2 * w + 1;
    let val = graph[2 * cy * stride + 2 * cx];
    let has_bs = (val & 1) != 0;
    let has_sl = (val & 2) != 0;
    if has_bs && !has_sl { 1 }
    else if has_sl && !has_bs { 2 }
    else { 0 }
}

/// Build a Voronoi cell polygon (in x4 coords) for pixel (px, py).
/// Returns nodes in CW order. Uses fixed-size array to avoid allocation.
fn pixel_cell(graph: &[u32], w: usize, h: usize, px: usize, py: usize) -> ([u64; 8], usize) {
    let tl = corner_diag(graph, w, h, px, py);
    let tr = corner_diag(graph, w, h, px + 1, py);
    let br = corner_diag(graph, w, h, px + 1, py + 1);
    let bl = corner_diag(graph, w, h, px, py + 1);

    let bx = (4 * px) as i32;
    let by = (4 * py) as i32;

    let mut nodes = [0u64; 8];
    let mut len = 0;
    let mut push = |x4: i32, y4: i32| { nodes[len] = pack_node(x4, y4); len += 1; };

    // TL corner (pixel visits as BR of the corner)
    match tl { 1 => { push(bx - 1, by + 1); push(bx + 1, by - 1); }
               2 => { push(bx + 1, by + 1); }
               _ => { push(bx, by); } }
    // TR corner (pixel visits as BL)
    match tr { 1 => { push(bx + 3, by + 1); }
               2 => { push(bx + 3, by - 1); push(bx + 5, by + 1); }
               _ => { push(bx + 4, by); } }
    // BR corner (pixel visits as TL)
    match br { 1 => { push(bx + 5, by + 3); push(bx + 3, by + 5); }
               2 => { push(bx + 3, by + 3); }
               _ => { push(bx + 4, by + 4); } }
    // BL corner (pixel visits as TR)
    match bl { 1 => { push(bx + 1, by + 3); }
               2 => { push(bx + 1, by + 5); push(bx - 1, by + 3); }
               _ => { push(bx, by + 4); } }

    (nodes, len)
}

struct DirEdge {
    from: u64,
    to: u64,
    color: u32,
}

/// Build directed boundary edges from Voronoi cells.
/// For each pixel, walk its cell polygon edges. Each half-edge gets the pixel's
/// color. Boundary edges (different colors on each side) produce directed pairs.
fn build_cell_edges(data: &VectorizeData, pixels: &[u32]) -> Vec<DirEdge> {
    let (w, h) = (data.img_w, data.img_h);

    // For each canonical edge (min→max), track (left_color, right_color).
    let mut edge_map: BTreeMap<(u64, u64), (u32, u32)> = BTreeMap::new();

    for y in 0..h {
        for x in 0..w {
            let color = pixels[y * w + x];
            let (cell, n) = pixel_cell(&data.graph, w, h, x, y);
            if n < 3 { continue; }

            for i in 0..n {
                let a = cell[i];
                let b = cell[(i + 1) % n];
                let (key, is_forward) = if a <= b { ((a, b), true) } else { ((b, a), false) };
                let entry = edge_map.entry(key).or_insert((u32::MAX, u32::MAX));
                if is_forward { entry.1 = color; } else { entry.0 = color; }
            }
        }
    }

    let mut edges = Vec::with_capacity(edge_map.len());
    for (&(a, b), &(left, right)) in &edge_map {
        let l = if left == u32::MAX { VOID_COLOR } else { left };
        let r = if right == u32::MAX { VOID_COLOR } else { right };
        if l == r { continue; }
        edges.push(DirEdge { from: a, to: b, color: r });
        edges.push(DirEdge { from: b, to: a, color: l });
    }

    edges
}

// ---------------------------------------------------------------------------
// Planar face tracing
// ---------------------------------------------------------------------------

/// Trace closed faces from directed boundary edges using the planar face algorithm.
/// Returns (loop of packed node IDs, fill color) for each face.
fn trace_faces(edges: &[DirEdge]) -> Vec<(Vec<u64>, u32)> {
    let n = edges.len();
    if n == 0 { return Vec::new(); }

    // Build outgoing adjacency sorted by angle.
    let mut out: Vec<(u64, i64, i64, usize)> = Vec::with_capacity(n);
    for (i, e) in edges.iter().enumerate() {
        let (fx, fy) = unpack_node(e.from);
        let (tx, ty) = unpack_node(e.to);
        out.push((e.from, (tx - fx) as i64, (ty - fy) as i64, i));
    }
    out.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| angle_cmp(a.1, a.2, b.1, b.2)));

    // Range index: for each node, [start..end) into out.
    let mut ranges: BTreeMap<u64, (usize, usize)> = BTreeMap::new();
    {
        let mut i = 0;
        while i < out.len() {
            let node = out[i].0;
            let start = i;
            while i < out.len() && out[i].0 == node { i += 1; }
            ranges.insert(node, (start, i));
        }
    }

    // Next-edge index: for each directed edge (p → c), find the next edge
    // leaving c that is immediately clockwise from the reverse direction.
    let mut next: Vec<usize> = vec![usize::MAX; n];
    for (i, e) in edges.iter().enumerate() {
        let (cx, cy) = unpack_node(e.to);
        let (px, py) = unpack_node(e.from);
        let rdx = (px - cx) as i64;
        let rdy = (py - cy) as i64;
        if let Some(&(s, end)) = ranges.get(&e.to) {
            let slice = &out[s..end];
            let p = slice.partition_point(|o|
                angle_cmp(o.1, o.2, rdx, rdy) == std::cmp::Ordering::Less
            );
            let prev = if p == 0 { slice.len() - 1 } else { p - 1 };
            next[i] = slice[prev].3;
        }
    }

    // Trace loops.
    let mut used = vec![false; n];
    let mut faces = Vec::new();
    for start in 0..n {
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
            nodes.push(edges[cur].from);
            let ni = next[cur];
            if ni >= n { break; }
            cur = ni;
        }
        if nodes.len() >= 3 && closed {
            faces.push((nodes, edges[start].color));
        }
    }
    faces
}

// ---------------------------------------------------------------------------
// Node → optimized position mapping
// ---------------------------------------------------------------------------

/// Build maps from x4 node → optimized position and → sharp flag.
/// Multiple CP slots may map to the same node; we take the first one found.
/// Nodes on the image perimeter are always marked sharp — boundary chains
/// terminate against the hard image edge, forming implicit T-junctions.
fn build_node_map(data: &VectorizeData) -> (BTreeMap<u64, (f64, f64)>, BTreeMap<u64, bool>) {
    let corners_w = data.img_w + 1;
    let num_cps = corners_w * (data.img_h + 1) * 2;
    let w4 = (data.img_w * 4) as i32;
    let h4 = (data.img_h * 4) as i32;

    let mut pos_map: BTreeMap<u64, (f64, f64)> = BTreeMap::new();
    let mut sharp_map: BTreeMap<u64, bool> = BTreeMap::new();

    for ci in 0..num_cps {
        if data.flags[ci] == 0 { continue; }
        let x4 = (data.orig_positions[ci * 2] * 4.0).round() as i32;
        let y4 = (data.orig_positions[ci * 2 + 1] * 4.0).round() as i32;
        let nid = pack_node(x4, y4);
        pos_map.entry(nid).or_insert((
            data.positions[ci * 2] as f64,
            data.positions[ci * 2 + 1] as f64,
        ));
        let on_border = x4 <= 0 || y4 <= 0 || x4 >= w4 || y4 >= h4;
        if on_border || data.flags[ci] & SHARP_MASK != 0 {
            sharp_map.insert(nid, true);
        }
    }

    (pos_map, sharp_map)
}

// ---------------------------------------------------------------------------
// Face → SVG path
// ---------------------------------------------------------------------------

/// Convert a traced face (closed loop of node IDs) to an SVG path `d` attribute.
///
/// Node types:
/// - **Grid-only** (no optimized position): straight line segment (L)
/// - **Smooth** (optimized, not sharp): B-spline control point (Q)
/// - **Sharp** (optimized, T-junction/crossing/border): adjacent curves still
///   approach smoothly via B-spline midpoints, but the vertex itself is a hard
///   corner connected by line segments (L through the sharp position)
fn face_to_svg_d(
    nodes: &[u64],
    pos_map: &BTreeMap<u64, (f64, f64)>,
    sharp_map: &BTreeMap<u64, bool>,
) -> String {
    let n = nodes.len();
    if n < 3 { return String::new(); }

    let get_pos = |nid: u64| -> (f64, f64) {
        pos_map.get(&nid).copied().unwrap_or_else(|| {
            let (x4, y4) = unpack_node(nid);
            (x4 as f64 / 4.0, y4 as f64 / 4.0)
        })
    };
    let is_optimized = |nid: u64| pos_map.contains_key(&nid);
    let is_sharp = |nid: u64| sharp_map.get(&nid).copied().unwrap_or(false);
    let is_grid = |nid: u64| !is_optimized(nid);
    let mid = |a: (f64, f64), b: (f64, f64)| ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5);

    let mut d = String::with_capacity(n * 32);

    // Determine start point: grid/sharp nodes start at their position,
    // smooth nodes start at the midpoint with the previous node.
    let first = get_pos(nodes[0]);
    let last = get_pos(nodes[n - 1]);
    let start = if is_grid(nodes[0]) || is_sharp(nodes[0]) {
        first
    } else {
        mid(last, first)
    };
    d.push_str(&format!("M{} {}", fmt(start.0), fmt(start.1)));

    for i in 0..n {
        let nid = nodes[i];
        let next_nid = nodes[(i + 1) % n];
        let p = get_pos(nid);
        let np = get_pos(next_nid);

        if is_grid(nid) {
            d.push_str(&format!("L{} {}", fmt(p.0), fmt(p.1)));
            if !is_grid(next_nid) && !is_sharp(next_nid) {
                let m = mid(p, np);
                d.push_str(&format!("L{} {}", fmt(m.0), fmt(m.1)));
            }
        } else if is_sharp(nid) {
            d.push_str(&format!("L{} {}", fmt(p.0), fmt(p.1)));
            if !is_grid(next_nid) && !is_sharp(next_nid) {
                let m = mid(p, np);
                d.push_str(&format!("L{} {}", fmt(m.0), fmt(m.1)));
            }
        } else {
            let end = mid(p, np);
            d.push_str(&format!("Q{} {} {} {}", fmt(p.0), fmt(p.1), fmt(end.0), fmt(end.1)));
        }
    }
    d.push('Z');
    d
}

// ---------------------------------------------------------------------------
// Background color detection
// ---------------------------------------------------------------------------

fn detect_bg(pixels: &[u32], w: usize, h: usize) -> u32 {
    let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
    for x in 0..w {
        *counts.entry(pixels[x]).or_default() += 1;
        *counts.entry(pixels[(h - 1) * w + x]).or_default() += 1;
    }
    for y in 1..h - 1 {
        *counts.entry(pixels[y * w]).or_default() += 1;
        *counts.entry(pixels[y * w + w - 1]).or_default() += 1;
    }
    counts.into_iter().max_by_key(|&(_, n)| n).map(|(c, _)| c).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Render vectorize-gpu pipeline output as a filled-region SVG document.
pub fn render_svg(data: &VectorizeData, pixels: &[u32]) -> String {
    let (w, h) = (data.img_w, data.img_h);
    let bg = detect_bg(pixels, w, h);

    let edges = build_cell_edges(data, pixels);
    let faces = trace_faces(&edges);
    let (pos_map, sharp_map) = build_node_map(data);

    let mut doc = svg::Document::new()
        .set("viewBox", (0, 0, w, h))
        .set("width", w * 4)
        .set("height", h * 4)
        .set("shape-rendering", "geometricPrecision");

    doc = doc.add(
        svg::node::element::Rectangle::new()
            .set("width", w)
            .set("height", h)
            .set("fill", hex(bg)),
    );

    // Group face paths by color for compact SVG output.
    let mut by_color: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    for (nodes, color) in &faces {
        if *color == bg || *color == VOID_COLOR { continue; }
        let d = face_to_svg_d(nodes, &pos_map, &sharp_map);
        if !d.is_empty() {
            by_color.entry(*color).or_default().push(d);
        }
    }

    for (color, ds) in &by_color {
        let mut combined = String::new();
        for d in ds {
            if !combined.is_empty() { combined.push(' '); }
            combined.push_str(d);
        }
        doc = doc.add(
            svg::node::element::Path::new()
                .set("fill", hex(*color))
                .set("fill-rule", "nonzero")
                .set("stroke", "none")
                .set("d", combined),
        );
    }

    doc.to_string()
}
