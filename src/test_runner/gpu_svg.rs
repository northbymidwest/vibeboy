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
use vibeboy::scaling::vectorize_faces::{
    VOID_COLOR, build_cell_edges, trace_faces, build_node_map, unpack_node,
};

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
    for face in &faces {
        if face.color == bg || face.color == VOID_COLOR { continue; }
        let d = face_to_svg_d(&face.nodes, &pos_map, &sharp_map);
        if !d.is_empty() {
            by_color.entry(face.color).or_default().push(d);
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
