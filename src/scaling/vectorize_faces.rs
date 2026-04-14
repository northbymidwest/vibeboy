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
// Scanline rasterizer support: node→CP mapping and face flattening
// ---------------------------------------------------------------------------

/// Map from x4 node → list of CP indices at that node.
/// At regular corners there's one CP. At T-junctions/crossings, two.
pub fn build_node_cp_map(data: &VectorizeData) -> BTreeMap<u64, Vec<usize>> {
    let corners_w = data.img_w + 1;
    let num_cps = corners_w * (data.img_h + 1) * 2;

    let mut map: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for ci in 0..num_cps {
        if data.flags[ci] == 0 { continue; }
        let x4 = (data.orig_positions[ci * 2] * 4.0).round() as i32;
        let y4 = (data.orig_positions[ci * 2 + 1] * 4.0).round() as i32;
        map.entry(pack_node(x4, y4)).or_default().push(ci);
    }
    map
}

/// Convert an x4 vector (from one face node to an adjacent face node)
/// into the edge direction code (0-3) at the destination grid corner.
///
/// The face arrives from the direction of the vector. The edge direction
/// code represents which edge of the grid corner was crossed:
///   0=north, 1=east, 2=south, 3=west
///
/// Vector pointing east (+x) means the face came from the west → crossed west edge → dir 3.
/// Returns None if the vector is zero or doesn't map cleanly to a direction.
fn x4_vector_to_entry_dir(dx4: i32, dy4: i32) -> Option<i32> {
    if dx4 == 0 && dy4 == 0 { return None; }
    // The entry direction is the OPPOSITE of the vector direction.
    // Vector pointing east → came from west → entry edge = west = 3
    if dx4.abs() >= dy4.abs() {
        if dx4 > 0 { Some(3) } else { Some(1) } // came from west / east
    } else {
        if dy4 > 0 { Some(0) } else { Some(2) } // came from north / south
    }
}

/// Same but for exit direction (vector FROM current TO next node).
/// Vector pointing east → exits via east edge → dir 1.
fn x4_vector_to_exit_dir(dx4: i32, dy4: i32) -> Option<i32> {
    if dx4 == 0 && dy4 == 0 { return None; }
    if dx4.abs() >= dy4.abs() {
        if dx4 > 0 { Some(1) } else { Some(3) } // exits east / west
    } else {
        if dy4 > 0 { Some(2) } else { Some(0) } // exits south / north
    }
}

/// Resolve which CP at a T-junction/crossing node matches the face traversal.
/// Uses exact direction code matching from the face's entry/exit x4 vectors.
fn resolve_cp_at_node(
    nid: u64,
    prev_nid: u64,
    next_nid: u64,
    node_cp_map: &BTreeMap<u64, Vec<usize>>,
    data: &VectorizeData,
) -> Option<usize> {
    let cps = node_cp_map.get(&nid)?;
    if cps.len() == 1 { return Some(cps[0]); }

    // Compute entry/exit edge directions from face node x4 vectors
    let (cx4, cy4) = unpack_node(nid);
    let (px4, py4) = unpack_node(prev_nid);
    let (nx4, ny4) = unpack_node(next_nid);

    let entry_dir = x4_vector_to_entry_dir(cx4 - px4, cy4 - py4);
    let exit_dir = x4_vector_to_exit_dir(nx4 - cx4, ny4 - cy4);

    // Find CP whose chain directions match. Check both forward and reverse traversal.
    for &ci in cps {
        let cp_prev_dir = data.neighbors[ci * 4 + 2]; // prev_dir
        let cp_next_dir = data.neighbors[ci * 4 + 3]; // next_dir

        // Forward: face enters via prev_dir, exits via next_dir
        if let (Some(ed), Some(xd)) = (entry_dir, exit_dir) {
            if cp_prev_dir == ed && cp_next_dir == xd { return Some(ci); }
        }
        // Partial match: just entry or just exit
        if let Some(ed) = entry_dir {
            if cp_prev_dir == ed { return Some(ci); }
        }
        if let Some(xd) = exit_dir {
            if cp_next_dir == xd { return Some(ci); }
        }
    }

    // Fallback: first CP
    Some(cps[0])
}

/// A directed line segment for winding-rule scanline fill.
pub struct WindingEdge {
    pub x_at_ymin: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub dx_per_dy: f32,
    pub color: u32,
    pub winding: i8,
}

/// Flatten a Voronoi-traced face into directed line segments for scanline fill.
///
/// For each node in the face, resolves the corresponding CP using exact
/// direction matching at T-junctions. Evaluates beval(prev, self, next)
/// using the chain's actual neighbors for smooth curves everywhere.
/// Nodes without CPs (diagonal intermediate points) use grid positions.
pub fn flatten_face_cp(
    face: &Face,
    node_cp_map: &BTreeMap<u64, Vec<usize>>,
    data: &VectorizeData,
    scale_factor: f32,
    edges_out: &mut Vec<WindingEdge>,
) {
    let n = face.nodes.len();
    if n < 3 { return; }

    let cp_pos = |ci: usize| -> (f32, f32) {
        (data.positions[ci * 2], data.positions[ci * 2 + 1])
    };

    let mut pts: Vec<(f32, f32)> = Vec::with_capacity(n * 4);

    for i in 0..n {
        let nid = face.nodes[i];
        let prev_nid = face.nodes[(i + n - 1) % n];
        let next_nid = face.nodes[(i + 1) % n];

        let ci = resolve_cp_at_node(nid, prev_nid, next_nid, node_cp_map, data);

        if let Some(ci) = ci {
            // Evaluate beval(pp, pos, np) with degenerate fallback — same as
            // the existing nearest-curve rasterizer (vectorize.rs line 1819).
            let pos = cp_pos(ci);
            let prev_ci = data.neighbors[ci * 4];
            let next_ci = data.neighbors[ci * 4 + 1];
            let pp = if prev_ci >= 0 { cp_pos(prev_ci as usize) } else { pos };
            let np = if next_ci >= 0 { cp_pos(next_ci as usize) } else { pos };

            // Adaptive subdivision
            let chord_mid_x = (pp.0 + np.0) * 0.25 + pos.0 * 0.5;
            let chord_mid_y = (pp.1 + np.1) * 0.25 + pos.1 * 0.5;
            let dev = ((pos.0 - chord_mid_x).powi(2) + (pos.1 - chord_mid_y).powi(2)).sqrt();
            let subdiv = (dev * scale_factor * 2.0).ceil().clamp(1.0, 16.0) as usize;

            for s in 0..=subdiv {
                let t = s as f32 / subdiv as f32;
                let u = 1.0 - t;
                let x = (0.5 * u * u * pp.0 + (u * t + 0.5) * pos.0 + 0.5 * t * t * np.0) * scale_factor;
                let y = (0.5 * u * u * pp.1 + (u * t + 0.5) * pos.1 + 0.5 * t * t * np.1) * scale_factor;
                pts.push((x, y));
            }
        } else {
            // No CP at this node (diagonal intermediate) — grid position
            let (x4, y4) = unpack_node(nid);
            pts.push((x4 as f32 / 4.0 * scale_factor, y4 as f32 / 4.0 * scale_factor));
        }
    }

    // Convert points to winding edges
    let color = face.color;
    let np = pts.len();
    for i in 0..np {
        let (x0, y0) = pts[i];
        let (x1, y1) = pts[(i + 1) % np];
        let dy = y1 - y0;
        if dy.abs() < 1e-6 { continue; }

        let winding: i8 = if dy > 0.0 { 1 } else { -1 };
        let (ymin, ymax, x_at_ymin) = if y0 < y1 { (y0, y1, x0) } else { (y1, y0, x1) };
        let dx_per_dy = (x1 - x0) / dy;

        edges_out.push(WindingEdge {
            x_at_ymin, y_min: ymin, y_max: ymax, dx_per_dy,
            color, winding,
        });
    }
}
