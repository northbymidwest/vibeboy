//! Face extraction from the vectorize cell graph.
//!
//! Traces closed Voronoi cell faces from the resolved similarity graph
//! and maps face nodes to optimized B-spline control point positions.
//! Used by both the SVG exporter and the scanline rasterizer.

use std::collections::BTreeMap;

use super::vectorize::VectorizeData;

const IS_TJUNCTION: u32 = 32;
const IS_CROSSING: u32 = 64;
pub const SHARP_MASK: u32 = IS_TJUNCTION | IS_CROSSING;

/// Sentinel color for the void outside the image boundary.
pub const VOID_COLOR: u32 = 0x01000000;

/// A closed face traced from the Voronoi cell boundary graph.
pub struct Face {
    pub nodes: Vec<u64>,
    pub color: u32,
}

/// Pack x4/y4 coordinates into a u64 node identifier.
pub fn pack_node(x4: i32, y4: i32) -> u64 {
    ((x4 as u64) << 32) | (y4 as u32 as u64)
}

/// Unpack a u64 node identifier into x4/y4 coordinates.
pub fn unpack_node(nid: u64) -> (i32, i32) {
    ((nid >> 32) as i32, nid as u32 as i32)
}

/// Cross-product angle comparison (exact integer, matches contour/loops.rs).
#[inline]
pub fn angle_cmp(adx: i64, ady: i64, bdx: i64, bdy: i64) -> std::cmp::Ordering {
    let ha = ady > 0 || (ady == 0 && adx > 0);
    let hb = bdy > 0 || (bdy == 0 && bdx > 0);
    if ha != hb {
        return if ha { std::cmp::Ordering::Less } else { std::cmp::Ordering::Greater };
    }
    (adx * bdy - ady * bdx).cmp(&0).reverse()
}

/// Get diagonal state at grid corner (cx, cy) from the GPU resolved graph.
/// Returns 0=none, 1=backslash, 2=slash.
pub fn corner_diag(graph: &[u32], w: usize, h: usize, cx: usize, cy: usize) -> u8 {
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
pub fn pixel_cell(graph: &[u32], w: usize, h: usize, px: usize, py: usize) -> ([u64; 8], usize) {
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

/// A directed half-edge in the Voronoi cell boundary graph.
pub struct DirEdge {
    pub from: u64,
    pub to: u64,
    pub color: u32,
}

/// Build directed boundary edges from Voronoi cells.
/// For each pixel, walk its cell polygon edges. Each half-edge gets the pixel's
/// color. Boundary edges (different colors on each side) produce directed pairs.
pub fn build_cell_edges(data: &VectorizeData, pixels: &[u32]) -> Vec<DirEdge> {
    let (w, h) = (data.img_w, data.img_h);

    // For each canonical edge (min->max), track (left_color, right_color).
    // Sentinel for "no color assigned yet" -- must not collide with any real
    // pixel color. Real pixels always have 0xFF alpha, so 0x00000000 is safe.
    const NO_COLOR: u32 = 0;
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
                let entry = edge_map.entry(key).or_insert((NO_COLOR, NO_COLOR));
                if is_forward { entry.1 = color; } else { entry.0 = color; }
            }
        }
    }

    let mut edges = Vec::with_capacity(edge_map.len());
    for (&(a, b), &(left, right)) in &edge_map {
        let l = if left == NO_COLOR { VOID_COLOR } else { left };
        let r = if right == NO_COLOR { VOID_COLOR } else { right };
        if l == r { continue; }
        edges.push(DirEdge { from: a, to: b, color: r });
        edges.push(DirEdge { from: b, to: a, color: l });
    }

    edges
}

/// Trace closed faces from directed boundary edges using the planar face algorithm.
/// Returns a `Face` for each closed loop found.
pub fn trace_faces(edges: &[DirEdge]) -> Vec<Face> {
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

    // Next-edge index: for each directed edge (p -> c), find the next edge
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
            faces.push(Face { nodes, color: edges[start].color });
        }
    }
    faces
}

/// Build maps from x4 node -> optimized position and -> sharp flag.
/// Used by the SVG exporter (which still uses the sharp mask approach).
/// Multiple CP slots may map to the same node; we take the first one found.
pub fn build_node_map(data: &VectorizeData) -> (BTreeMap<u64, (f64, f64)>, BTreeMap<u64, bool>) {
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
// Scanline rasterizer: per-CP winding edge construction
// ---------------------------------------------------------------------------

/// A directed line segment for winding-rule scanline fill.
pub struct WindingEdge {
    pub x_at_ymin: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub dx_per_dy: f32,
    pub color: u32,
    pub winding: i8,
}

/// Build winding edges directly from the CP chain data.
///
/// Each active CP defines a B-spline span using the same (pp, pos, np) triplet
/// as the existing nearest-curve rasterizer, with the same degenerate fallback
/// for chain endpoints (missing neighbor → use self).
///
/// Each CP's span separates two colors (from get_edge_colors). color_right is
/// always the CW (right) side of the chain exit direction. The span is flattened
/// into line segments and two winding edges are emitted per segment: one for
/// color_right (+1 if downward, -1 if upward) and one for color_left (opposite).
///
/// Border edges are added for pixels on the image boundary to close regions
/// that touch the image edge.
pub fn build_winding_edges(
    data: &VectorizeData,
    pixels: &[u32],
    scale_factor: f32,
    out_h: usize,
) -> Vec<WindingEdge> {
    let corners_w = data.img_w + 1;
    let num_cps = corners_w * (data.img_h + 1) * 2;
    let img_w = data.img_w;
    let img_h = data.img_h;

    let get_px_color = |px: i32, py: i32| -> u32 {
        let px = px.clamp(0, img_w as i32 - 1) as usize;
        let py = py.clamp(0, img_h as i32 - 1) as usize;
        pixels[py * img_w + px]
    };
    let get_edge_colors = |icx: i32, icy: i32, dir: i32| -> (u32, u32) {
        match dir {
            0 => (get_px_color(icx - 1, icy - 1), get_px_color(icx, icy - 1)),
            1 => (get_px_color(icx, icy - 1), get_px_color(icx, icy)),
            2 => (get_px_color(icx, icy), get_px_color(icx - 1, icy)),
            3 => (get_px_color(icx - 1, icy), get_px_color(icx - 1, icy - 1)),
            _ => (0, 0),
        }
    };

    let mut edges = Vec::new();

    // For each active CP, flatten its B-spline span into winding edge pairs.
    // Include CPs with only prev_dir OR only next_dir (T-junction stems,
    // chain endpoints) — they have degenerate B-spline spans that are still
    // part of color boundaries.
    for ci in 0..num_cps {
        let flag = data.flags[ci];
        if flag == 0 { continue; }

        let prev_ci = data.neighbors[ci * 4];
        let next_ci = data.neighbors[ci * 4 + 1];
        if prev_ci < 0 && next_ci < 0 { continue; }

        let prev_dir = data.neighbors[ci * 4 + 2];
        let next_dir = data.neighbors[ci * 4 + 3];

        let icx = (ci / 2 % corners_w) as i32;
        let icy = (ci / 2 / corners_w) as i32;

        // Color resolution matching cell_rasterizer.slang resolve_color():
        // - next_dir: colors are SWAPPED (color_left = nr, color_right = nl)
        // - prev_dir: colors are NOT swapped (color_left = pl, color_right = pr)
        let (resolve_left, resolve_right) = if next_dir >= 0 {
            let (l, r) = get_edge_colors(icx, icy, next_dir);
            (r, l) // swap for next_dir
        } else if prev_dir >= 0 {
            get_edge_colors(icx, icy, prev_dir) // no swap for prev_dir
        } else {
            continue;
        };
        if resolve_left == resolve_right { continue; }

        // B-spline triplet — same as existing rasterizer (vectorize.rs line 1819)
        let pos = (data.positions[ci * 2], data.positions[ci * 2 + 1]);
        let pp = if prev_ci >= 0 {
            (data.positions[prev_ci as usize * 2], data.positions[prev_ci as usize * 2 + 1])
        } else { pos };
        let np = if next_ci >= 0 {
            (data.positions[next_ci as usize * 2], data.positions[next_ci as usize * 2 + 1])
        } else { pos };

        // Adaptive subdivision
        let chord_mid_x = (pp.0 + np.0) * 0.25 + pos.0 * 0.5;
        let chord_mid_y = (pp.1 + np.1) * 0.25 + pos.1 * 0.5;
        let dev = ((pos.0 - chord_mid_x).powi(2) + (pos.1 - chord_mid_y).powi(2)).sqrt();
        let subdiv = (dev * scale_factor * 2.0).ceil().clamp(1.0, 16.0) as usize;

        let mut prev_pt = beval(pp, pos, np, 0.0, scale_factor);

        for s in 1..=subdiv {
            let t = s as f32 / subdiv as f32;
            let cur_pt = beval(pp, pos, np, t, scale_factor);

            let dy = cur_pt.1 - prev_pt.1;
            let dx = cur_pt.0 - prev_pt.0;
            // Skip near-horizontal segments: they barely cross scanlines
            // and their extreme dx_per_dy produces wrong x-positions.
            if dy.abs() > 1e-6 && dy.abs() > dx.abs() * 0.01 {
                let (ymin, ymax, x_at_ymin) = if prev_pt.1 < cur_pt.1 {
                    (prev_pt.1, cur_pt.1, prev_pt.0)
                } else {
                    (cur_pt.1, prev_pt.1, cur_pt.0)
                };
                let dx_per_dy = (cur_pt.0 - prev_pt.0) / dy;

                // For downward edge: sweep crosses left→right, exiting
                // resolve_left's region and entering resolve_right's.
                let wr: i8 = if dy > 0.0 { 1 } else { -1 };

                edges.push(WindingEdge {
                    x_at_ymin, y_min: ymin, y_max: ymax, dx_per_dy,
                    color: resolve_right, winding: wr,
                });
                edges.push(WindingEdge {
                    x_at_ymin, y_min: ymin, y_max: ymax, dx_per_dy,
                    color: resolve_left, winding: -wr,
                });
            }

            prev_pt = cur_pt;
        }
    }

    // Border edges: close regions touching image boundaries.
    let out_w_f = img_w as f32 * scale_factor;
    for py in 0..img_h {
        let y0 = py as f32 * scale_factor;
        let y1 = (py + 1) as f32 * scale_factor;

        // Left border (x=0): downward edge at left boundary.
        // This is a vertical boundary at x=0. The pixel at column 0 is to the
        // RIGHT of this edge (CW side for a downward edge) → enters that color.
        // Left border (x=0): downward edge. Left-column pixel is to the
        // screen-right of this edge → it enters (+1).
        let lc = get_px_color(0, py as i32);
        edges.push(WindingEdge {
            x_at_ymin: 0.0, y_min: y0, y_max: y1, dx_per_dy: 0.0,
            color: lc, winding: 1,
        });
        // Right border (x=out_w): right-column pixel is to the screen-left
        // of this edge → it exits (-1).
        let rc = get_px_color(img_w as i32 - 1, py as i32);
        edges.push(WindingEdge {
            x_at_ymin: out_w_f, y_min: y0, y_max: y1, dx_per_dy: 0.0,
            color: rc, winding: -1,
        });
    }

    edges
}

#[inline]
fn beval(pp: (f32, f32), pos: (f32, f32), np: (f32, f32), t: f32, scale: f32) -> (f32, f32) {
    let u = 1.0 - t;
    (
        (0.5 * u * u * pp.0 + (u * t + 0.5) * pos.0 + 0.5 * t * t * np.0) * scale,
        (0.5 * u * u * pp.1 + (u * t + 0.5) * pos.1 + 0.5 * t * t * np.1) * scale,
    )
}
