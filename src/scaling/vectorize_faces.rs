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

/// A boundary edge for scanline color-sweep fill.
/// Each edge separates two color regions. When a scanline crosses this edge
/// from left to right, the active color transitions from `screen_left` to
/// `screen_right`.
pub struct ScanEdge {
    pub x_at_ymin: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub dx_per_dy: f32,
    /// Color on the screen-right side of this edge.
    pub screen_right: u32,
    /// Color on the screen-left side of this edge.
    pub screen_left: u32,
    /// Source CP index (for diagnostics).
    pub ci: usize,
    /// CP flag bits (for identifying T-junctions).
    pub cp_flag: u32,
    /// Grid corner + direction info (diagnostics).
    pub diag_icx: i32,
    pub diag_icy: i32,
    pub diag_prev_dir: i32,
    pub diag_next_dir: i32,
    pub diag_prev_ci: i32,
    pub diag_next_ci: i32,
    /// Which color source was used: 0=prev, 1=next
    pub diag_color_src: u8,
}

/// Build scan edges directly from the CP chain data.
///
/// Each active CP defines a B-spline span using the same (pp, pos, np) triplet
/// as the existing nearest-curve rasterizer, with the same degenerate fallback
/// for chain endpoints (missing neighbor → use self).
///
/// Color resolution matches cell_rasterizer.slang resolve_color():
/// - For next_dir: colors are SWAPPED (left=nr, right=nl from get_edge_colors)
/// - For prev_dir: colors are NOT swapped (left=pl, right=pr)
///
/// For downward segments, left=screen-left, right=screen-right (from tangent
/// normal analysis). For upward segments, the screen relationship flips.
pub fn build_scan_edges(
    data: &VectorizeData,
    pixels: &[u32],
    scale_factor: f32,
) -> Vec<ScanEdge> {
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
        // The GPU rasterizer picks colors based on t along the span:
        //   t < 0.5 → prev_dir colors (left=pl, right=pr), ref_t=0.0
        //   t >= 0.5 → next_dir colors (left=nr, right=nl SWAPPED), ref_t=1.0
        // With fallback to the other direction if the preferred one is invalid.
        //
        // Chain endpoints (prev_ci < 0 or next_ci < 0) use only the valid side.
        let prev_colors = if prev_dir >= 0 {
            let (pl, pr) = get_edge_colors(icx, icy, prev_dir);
            if pl != pr { Some((pl, pr)) } else { None }
        } else { None };
        let next_colors = if next_dir >= 0 {
            let (nl, nr) = get_edge_colors(icx, icy, next_dir);
            if nl != nr { Some((nr, nl)) } else { None } // swap for next_dir
        } else { None };

        if prev_colors.is_none() && next_colors.is_none() { continue; }

        // B-spline triplets — matching GPU cell_rasterizer.slang exactly:
        //   pp = (prev_ci >= 0) ? read_pos(prev_ci) : cp;
        //   np = (next_ci >= 0) ? read_pos(next_ci) : cp;
        let pos = (data.positions[ci * 2], data.positions[ci * 2 + 1]);
        let pp = if prev_ci >= 0 {
            (data.positions[prev_ci as usize * 2], data.positions[prev_ci as usize * 2 + 1])
        } else { pos };
        let np = if next_ci >= 0 {
            (data.positions[next_ci as usize * 2], data.positions[next_ci as usize * 2 + 1])
        } else { pos };

        let orig_pos = (data.orig_positions[ci * 2], data.orig_positions[ci * 2 + 1]);
        let orig_pp = if prev_ci >= 0 {
            (data.orig_positions[prev_ci as usize * 2], data.orig_positions[prev_ci as usize * 2 + 1])
        } else { orig_pos };
        let orig_np = if next_ci >= 0 {
            (data.orig_positions[next_ci as usize * 2], data.orig_positions[next_ci as usize * 2 + 1])
        } else { orig_pos };

        // Adaptive subdivision. For junction CPs, force t=0.5 as a
        // mandatory sample point so all segments meeting at the junction
        // terminate at exactly the same point (beval at t=0.5).
        let chord_mid_x = (pp.0 + np.0) * 0.25 + pos.0 * 0.5;
        let chord_mid_y = (pp.1 + np.1) * 0.25 + pos.1 * 0.5;
        let dev = ((pos.0 - chord_mid_x).powi(2) + (pos.1 - chord_mid_y).powi(2)).sqrt();
        let subdiv = (dev * scale_factor * 2.0).ceil().clamp(1.0, 16.0) as usize;
        let is_junction = flag & SHARP_MASK != 0;

        // Build t-value list with mandatory t=0.5 for junctions
        let mut t_values: Vec<f32> = Vec::with_capacity(subdiv + 2);
        for s in 1..=subdiv {
            t_values.push(s as f32 / subdiv as f32);
        }
        if is_junction && !t_values.contains(&0.5) {
            t_values.push(0.5);
            t_values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        }

        let mut prev_pt = beval(pp, pos, np, 0.0, scale_factor);
        let mut t_prev_val = 0.0f32;

        for &t in &t_values {
            let cur_pt = beval(pp, pos, np, t, scale_factor);
            let t_prev = t_prev_val;

            let dy = cur_pt.1 - prev_pt.1;
            let mid_t = (t_prev + t) * 0.5;

            // For T-junction/crossing CPs, clip segments that overshoot
            // past the junction point — matching the GPU's dot-product
            // clipping in resolve_color for chain endpoints.
            if dy.abs() > 1e-6 && flag & SHARP_MASK != 0 {
                let u = 1.0 - mid_t;
                let curve_x = 0.5 * u * u * pp.0 + (u * mid_t + 0.5) * pos.0 + 0.5 * mid_t * mid_t * np.0;
                let curve_y = 0.5 * u * u * pp.1 + (u * mid_t + 0.5) * pos.1 + 0.5 * mid_t * mid_t * np.1;
                let (toward_x, toward_y) = if mid_t < 0.5 {
                    (pp.0 - pos.0, pp.1 - pos.1)
                } else {
                    (np.0 - pos.0, np.1 - pos.1)
                };
                let offset_x = curve_x - pos.0;
                let offset_y = curve_y - pos.1;
                if toward_x * offset_x + toward_y * offset_y < 0.0 {
                    prev_pt = cur_pt;
                    continue;
                }
            }

            if dy.abs() > 1e-6 {
                let (ymin, ymax, x_at_ymin) = if prev_pt.1 < cur_pt.1 {
                    (prev_pt.1, cur_pt.1, prev_pt.0)
                } else {
                    (cur_pt.1, prev_pt.1, cur_pt.0)
                };
                let dx_per_dy = (cur_pt.0 - prev_pt.0) / dy;

                // Color resolution matching GPU resolve_color():
                // - Chain start (prev_ci < 0): next_dir colors, ref_t=1.0
                // - Chain end (next_ci < 0): prev_dir colors, ref_t=0.0
                // - Both present: use SPATIAL position relative to junction
                //   point (pos) to determine prev vs next. The GPU uses
                //   projected t per-pixel; we approximate by checking which
                //   side of pos the segment midpoint falls on.
                let (resolve_left, resolve_right, ref_t, color_src) = if prev_ci < 0 {
                    let c = next_colors.unwrap_or_else(|| prev_colors.unwrap());
                    (c.0, c.1, 1.0f32, 1u8)
                } else if next_ci < 0 {
                    let c = prev_colors.unwrap_or_else(|| next_colors.unwrap());
                    (c.0, c.1, 0.0f32, 0u8)
                } else {
                    // Spatial test: is this segment on the prev or next side
                    // of the junction? dot(segment_mid - junction, pp - pos)
                    // > 0 means segment is on the prev side.
                    let seg_mid = ((prev_pt.0 + cur_pt.0) * 0.5,
                                   (prev_pt.1 + cur_pt.1) * 0.5);
                    let junc = (pos.0 * scale_factor, pos.1 * scale_factor);
                    let toward_prev = (pp.0 - pos.0, pp.1 - pos.1);
                    let offset = (seg_mid.0 - junc.0, seg_mid.1 - junc.1);
                    let on_prev_side = toward_prev.0 * offset.0
                        + toward_prev.1 * offset.1 > 0.0;
                    if on_prev_side {
                        if let Some(c) = prev_colors { (c.0, c.1, 0.0f32, 0u8) }
                        else { let c = next_colors.unwrap(); (c.0, c.1, 1.0, 1) }
                    } else {
                        if let Some(c) = next_colors { (c.0, c.1, 1.0f32, 1u8) }
                        else { let c = prev_colors.unwrap(); (c.0, c.1, 0.0, 0) }
                    }
                };

                // Screen-side determination matching GPU resolve_color():
                // orig tangent at ref_t, opt tangent at actual t.
                let opt_tan = beval_deriv(pp, pos, np, mid_t);
                let orig_tan = beval_deriv(orig_pp, orig_pos, orig_np, ref_t);
                let normals_agree = opt_tan.0 * orig_tan.0 + opt_tan.1 * orig_tan.1 > 0.0;

                let (screen_right, screen_left) =
                    if (dy > 0.0) == normals_agree {
                        (resolve_right, resolve_left)
                    } else {
                        (resolve_left, resolve_right)
                    };

                edges.push(ScanEdge {
                    x_at_ymin, y_min: ymin, y_max: ymax, dx_per_dy,
                    screen_right, screen_left,
                    ci, cp_flag: flag,
                    diag_icx: icx, diag_icy: icy,
                    diag_prev_dir: prev_dir, diag_next_dir: next_dir,
                    diag_prev_ci: prev_ci, diag_next_ci: next_ci,
                    diag_color_src: color_src,
                });
            }

            prev_pt = cur_pt;
            t_prev_val = t;
        }
    }

    edges
}

#[inline]
fn beval_deriv(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), t: f32) -> (f32, f32) {
    ((t - 1.0) * p0.0 + (1.0 - 2.0 * t) * p1.0 + t * p2.0,
     (t - 1.0) * p0.1 + (1.0 - 2.0 * t) * p1.1 + t * p2.1)
}

#[inline]
fn beval(pp: (f32, f32), pos: (f32, f32), np: (f32, f32), t: f32, scale: f32) -> (f32, f32) {
    let u = 1.0 - t;
    (
        (0.5 * u * u * pp.0 + (u * t + 0.5) * pos.0 + 0.5 * t * t * np.0) * scale,
        (0.5 * u * u * pp.1 + (u * t + 0.5) * pos.1 + 0.5 * t * t * np.1) * scale,
    )
}
