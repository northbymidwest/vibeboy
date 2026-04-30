//! SVG export for the vectorize-gpu pipeline.
//!
//! Builds filled vector regions from the GPU pipeline's resolved similarity graph
//! and optimized B-spline control points. Uses Voronoi cell boundary walking
//! to create a planar subdivision, then traces faces with the planar face
//! algorithm.
//!
//! At each face-loop node we consult the CP graph: if some CP at this grid
//! corner has chain neighbors whose grid positions match the face's previous
//! and next loop nodes (in either direction), the face follows that chain
//! smoothly through the junction and renders as a single quadratic Bézier
//! using that CP as control. Otherwise the node is a *kink* — a T-junction
//! stem terminus, a chain endpoint at the image border, a face transition
//! between two chains at a T-junction, or two chains crossing at the same
//! corner — and the curve must land on the junction position exactly.
//!
//! At a kink with an *interior* partial chain (a chain at this node whose
//! neighbor matches the face's prev_loop or next_loop), the boundary is
//! rendered as the de Casteljau split of that chain's full Q at t=0.5 — a
//! half-Q from `mid(neighbor, this_cp)` to `kink_pos`. This matches the
//! rasterizer's clamped-stem and crossing-chain rendering exactly. With an
//! *endpoint* partial chain (a 2-CP chain whose far end is itself a chain
//! endpoint), the full B-spline segment at this CP is emitted as a single Q.
//!
//! The kink position varies by junction type (see [`NodeMap::kink_pos`]):
//! T-junction stems' snap-corrected position, chain endpoints' own position,
//! and the crossings' chains-intersection point computed as the standard
//! B-spline blend `(prev + 6·this + next) / 8` at t=0.5.

use std::collections::{btree_map, BTreeMap};
use svg::node::element::path::{Command, Data, Position};
use vibeboy::scaling::vectorize::VectorizeData;

/// Sentinel color for the void outside the image boundary.
const VOID_COLOR: u32 = 0x01000000;

// CP flag bits — must match `src/scaling/vectorize.rs`.
const IS_TJUNCTION: u32 = 32;
const IS_CROSSING: u32 = 64;
const IS_ENDPOINT: u32 = 128;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn hex(c: u32) -> String {
    format!("#{:02X}{:02X}{:02X}", (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF)
}

fn cmd_move(p: (f64, f64)) -> Command {
    Command::Move(Position::Absolute, p.into())
}
fn cmd_line(p: (f64, f64)) -> Command {
    Command::Line(Position::Absolute, p.into())
}
fn cmd_quad(c: (f64, f64), p: (f64, f64)) -> Command {
    Command::QuadraticCurve(Position::Absolute, (c.0, c.1, p.0, p.1).into())
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
    // Sentinel for "no color assigned yet" — must not collide with any real
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
// Node → optimized position + chain neighbors
// ---------------------------------------------------------------------------

/// One CP's chain at a face-loop node. The position is the CP's optimized
/// position — this is what becomes the Q control when a face follows this
/// chain through the node.
#[derive(Clone)]
struct ChainNeighbors {
    prev: Option<u64>,
    next: Option<u64>,
    pos: (f64, f64),
}

struct NodeMap {
    /// Multiple CPs may share a corner (T-junction slot 0/1, crossing slot 0/1).
    /// Each entry's neighbors define one chain through this corner.
    chains: BTreeMap<u64, Vec<ChainNeighbors>>,
    /// Position used at *kinks* (boundary lands here exactly, no single chain
    /// matches the face traversal). Chosen so the path's incoming/outgoing
    /// half-Qs converge on the same point the rasterizer's curves do:
    ///   * T-junction: the stem CP's position (snap-corrected to lie on the
    ///     through curve at t=0.5).
    ///   * Crossing: B(t = crossing_t) — the geometric intersection of the
    ///     two chains, computed from the optimizer-final positions.
    ///   * Plain chain endpoint (border/isolated): the CP's own position.
    kink_pos: BTreeMap<u64, (f64, f64)>,
    /// Bezier parameter at which the B-spline span at this kink's CP passes
    /// through `kink_pos`. Used by the de Casteljau split that emits the
    /// half-Beziers on either side of the kink:
    ///   * T-junction stems / regular kinks: 0.5 (kp = B(0.5) by construction).
    ///   * Chain endpoints: 0.5 — actually irrelevant since the endpoint-partner
    ///     branch emits the full Bezier without subdivision.
    ///   * Crossings: `crossing_t[ci]` — generally not 0.5.
    /// Default 0.5 for any kink not explicitly populated.
    kink_t: BTreeMap<u64, f64>,
}

/// Compute the face-loop node ID where the CP at `ci` would appear, based on
/// the corner's diagonal state. Smooth diagonal CPs land on offset vertices;
/// everything else (axis-aligned smooth, T-junctions, crossings, even ones
/// with non-integer corrected orig positions from diagonal through-pairs)
/// lands on the integer grid corner because pixel cells push the corner
/// whenever `corner_diag` returns 0 (no single diagonal).
fn cp_loop_node(data: &VectorizeData, ci: usize, cw: usize) -> u64 {
    let cx = (ci / 2) % cw;
    let cy = (ci / 2) / cw;
    let slot = ci % 2;
    let diag = corner_diag(&data.graph, data.img_w, data.img_h, cx, cy);
    let cx4 = 4 * cx as i32;
    let cy4 = 4 * cy as i32;
    let junction = data.flags[ci] & (IS_TJUNCTION | IS_CROSSING);
    let (dx, dy) = match (diag, slot, junction) {
        // Diagonal corner with a smooth (non-junction) chain CP: pixel cells
        // push the offset for this slot. T-junctions and crossings sit at
        // the integer corner regardless of diag (their corrected positions
        // are handled by `pos`, not by offset selection).
        (1, 0, 0) => (-1, 1),
        (1, 1, 0) => (1, -1),
        (2, 0, 0) => (-1, -1),
        (2, 1, 0) => (1, 1),
        _ => (0, 0),
    };
    pack_node(cx4 + dx, cy4 + dy)
}

fn build_node_map(data: &VectorizeData) -> NodeMap {
    let cw = data.img_w + 1;
    let num_cps = cw * (data.img_h + 1) * 2;
    let mut chains: BTreeMap<u64, Vec<ChainNeighbors>> = BTreeMap::new();
    let mut kink_pos: BTreeMap<u64, (f64, f64)> = BTreeMap::new();
    let mut kink_t: BTreeMap<u64, f64> = BTreeMap::new();

    let neighbor_node = |nci: i32| -> Option<u64> {
        if nci < 0 { return None; }
        Some(cp_loop_node(data, nci as usize, cw))
    };

    for ci in 0..num_cps {
        if data.flags[ci] == 0 { continue; }
        let nid = cp_loop_node(data, ci, cw);
        let pos = (
            data.positions[ci * 2] as f64,
            data.positions[ci * 2 + 1] as f64,
        );
        chains.entry(nid).or_default().push(ChainNeighbors {
            prev: neighbor_node(data.neighbors[ci * 4]),
            next: neighbor_node(data.neighbors[ci * 4 + 1]),
            pos,
        });

        let is_endpoint = data.flags[ci] & IS_ENDPOINT != 0;
        let is_through = data.flags[ci] & IS_TJUNCTION != 0;
        let is_crossing = data.flags[ci] & IS_CROSSING != 0;
        let kp_value = if is_crossing {
            // Crossing kp = the geometric intersection of this CP's curve
            // and its partner's. Under the new pipeline crossings stay at
            // the optimizer's position, and `crossing_t[ci]` holds the
            // exact parameter on this slot's curve at which the two
            // curves cross. Evaluate B(t) with ghost-extended endpoints
            // (matching the rasterizer's clamped Bezier rendering) so the
            // SVG kink lands on the same point the wedge AA anchors at.
            // Slot 0 (N-S) and slot 1 (E-W) both produce the same
            // geometric point within FP epsilon, so override order
            // doesn't matter.
            let prev_idx = data.neighbors[ci * 4];
            let next_idx = data.neighbors[ci * 4 + 1];
            if prev_idx >= 0 && next_idx >= 0 {
                let prev_real = (
                    data.positions[prev_idx as usize * 2] as f64,
                    data.positions[prev_idx as usize * 2 + 1] as f64,
                );
                let next_real = (
                    data.positions[next_idx as usize * 2] as f64,
                    data.positions[next_idx as usize * 2 + 1] as f64,
                );
                let prev_is_end =
                    data.flags[prev_idx as usize] & IS_ENDPOINT != 0;
                let next_is_end =
                    data.flags[next_idx as usize] & IS_ENDPOINT != 0;
                let pp = if prev_is_end {
                    (2.0 * prev_real.0 - pos.0, 2.0 * prev_real.1 - pos.1)
                } else {
                    prev_real
                };
                let np = if next_is_end {
                    (2.0 * next_real.0 - pos.0, 2.0 * next_real.1 - pos.1)
                } else {
                    next_real
                };
                let t = data.crossing_t[ci] as f64;
                let u = 1.0 - t;
                // B(t) = 0.5·u²·pp + (u·t + 0.5)·cp + 0.5·t²·np
                let w_prev = 0.5 * u * u;
                let w_curr = u * t + 0.5;
                let w_next = 0.5 * t * t;
                (
                    w_prev * pp.0 + w_curr * pos.0 + w_next * np.0,
                    w_prev * pp.1 + w_curr * pos.1 + w_next * np.1,
                )
            } else {
                pos
            }
        } else {
            pos
        };
        // Crossings: also record the Bezier parameter τ at which this
        // chain's B-spline span passes through kp. The path emitter
        // de-Casteljau-splits the equivalent Bezier at τ to get the
        // half-Beziers on either side of the kink. For crossings τ is
        // generally not 0.5; for everything else 0.5 is the default and
        // we don't write to the map.
        if is_crossing {
            kink_t.insert(nid, data.crossing_t[ci] as f64);
        }
        // Override priority: prefer the IS_ENDPOINT (stem at T-junctions, on
        // the through curve) or IS_CROSSING (chain intersection point) over a
        // through CP's position that may have been inserted first by slot 0.
        // Among multiple endpoint/crossing CPs at the same node they all
        // converge on the same point, so subsequent overrides are no-ops.
        match kink_pos.entry(nid) {
            btree_map::Entry::Occupied(mut e) => {
                if (is_endpoint && !is_through) || is_crossing {
                    e.insert(kp_value);
                }
            }
            btree_map::Entry::Vacant(e) => {
                e.insert(kp_value);
            }
        }
    }

    NodeMap { chains, kink_pos, kink_t }
}

/// Returns the matching chain at `nid` (whose neighbors match the face's
/// `(prev_loop, next_loop)` in either direction) — meaning the face follows
/// that chain smoothly through this node. `None` means the face kinks here
/// and the curve must land on the junction position.
fn match_chain<'a>(
    nid: u64,
    prev_loop: u64,
    next_loop: u64,
    chains: &'a BTreeMap<u64, Vec<ChainNeighbors>>,
) -> Option<&'a ChainNeighbors> {
    chains.get(&nid)?.iter().find(|ch| {
        (ch.prev == Some(prev_loop) && ch.next == Some(next_loop))
            || (ch.prev == Some(next_loop) && ch.next == Some(prev_loop))
    })
}

// ---------------------------------------------------------------------------
// Face → SVG path
// ---------------------------------------------------------------------------

/// Append a traced face (closed loop of node IDs) to `data` as a closed
/// sub-path (`M ... Z`). Skips faces with fewer than 3 vertices.
///
/// Per-node behaviors:
/// - **Grid-only** (no optimized position): line segment through the grid
///   corner.
/// - **Smooth** (some chain at this CP has neighbors matching the face's
///   `prev`/`next`): quadratic Bézier with that chain's CP as control,
///   running between midpoints with the same chain's adjacent CP positions
///   on each side. This reproduces the rasterizer's uniform-knot B-spline
///   blend exactly.
/// - **Kink** (no chain fully matches but some interior chain partially
///   matches — face enters or leaves along that chain at this junction):
///   emit a half-Q (de Casteljau split of that chain's Q at t=0.5) so the
///   boundary lands on the through curve at `kink_pos`. T-junction stem
///   sides need this — they follow the through curve up to the corrected
///   T=0.5 point, then turn onto the stem.
/// - **Plain kink** (no partial match — chain endpoint at border, stem
///   terminus from the stem chain side, etc.): clamped end already lands on
///   `kink_pos` from the prev iteration, so this iteration emits nothing
///   unless the next iteration is also a plain kink, in which case we move
///   the pen with an L.
#[allow(unused_assignments)] // `pen`'s last write inside the loop is intentional.
fn append_face_path(nodes: &[u64], map: &NodeMap, data: &mut Data) {
    let n = nodes.len();
    if n < 3 { return; }

    let is_optimized = |nid: u64| map.chains.contains_key(&nid);
    let mid = |a: (f64, f64), b: (f64, f64)| ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5);
    let lerp = |a: (f64, f64), b: (f64, f64), t: f64| {
        (a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1))
    };
    // Per-kink Bezier parameter τ at which the B-spline span at the
    // kink CP passes through `kink_pos`. Defaults to 0.5; populated to
    // `crossing_t[ci]` for IS_CROSSING CPs in build_node_map. Drives
    // the de Casteljau split below.
    let kink_t = |nid: u64| -> f64 { map.kink_t.get(&nid).copied().unwrap_or(0.5) };
    let grid_pos = |nid: u64| -> (f64, f64) {
        let (x4, y4) = unpack_node(nid);
        (x4 as f64 / 4.0, y4 as f64 / 4.0)
    };

    // Find the chain at `next_nid` that's on the same chain as `this` (one of
    // its neighbors points back to `this_nid`). If found and the partner is an
    // *interior* CP (both neighbors valid) the chain continues smoothly to a
    // midpoint with the partner. If the partner is itself an endpoint, the
    // partner's position IS the clamped chain end. `None` means no chain
    // continues from `this` toward `next_nid` (the chain ends at this side).
    let chain_partner = |this_nid: u64, next_nid: u64| -> Option<&ChainNeighbors> {
        map.chains.get(&next_nid)?.iter().find(|p| {
            p.prev == Some(this_nid) || p.next == Some(this_nid)
        })
    };

    // For a smooth iteration emitting Q ending toward `next_nid`: the natural
    // endpoint is the midpoint with the chain partner (interior continuation)
    // or the partner's own position (clamped end at endpoint).
    let smooth_end = |this: &ChainNeighbors, this_nid: u64, next_nid: u64| -> (f64, f64) {
        if let Some(p) = chain_partner(this_nid, next_nid) {
            if p.prev.is_some() && p.next.is_some() {
                mid(this.pos, p.pos)
            } else {
                p.pos
            }
        } else {
            map.kink_pos
                .get(&next_nid)
                .copied()
                .unwrap_or_else(|| grid_pos(next_nid))
        }
    };

    // At a kink iteration, look for an interior chain at this node whose
    // neighbor matches `prev_loop` (face arriving along that chain) or
    // `next_loop` (face leaving along it). When found, the boundary follows
    // half of that chain's Q from `mid(neighbor.pos, this_chain.pos)` through
    // the kink point — a clamped Bézier matching the through curve.
    let partial_chain = |nid: u64, neighbor_loop: u64| -> Option<&ChainNeighbors> {
        map.chains.get(&nid)?.iter().find(|ch| {
            ch.prev.is_some()
                && ch.next.is_some()
                && (ch.prev == Some(neighbor_loop) || ch.next == Some(neighbor_loop))
        })
    };

    // Per iteration: full chain match (smooth) or `None`.
    let chain_match: Vec<Option<ChainNeighbors>> = (0..n)
        .map(|i| {
            let nid = nodes[i];
            let prev_loop = nodes[(i + n - 1) % n];
            let next_loop = nodes[(i + 1) % n];
            match_chain(nid, prev_loop, next_loop, &map.chains).cloned()
        })
        .collect();

    let is_kink = |i: usize| is_optimized(nodes[i]) && chain_match[i].is_none();

    // Pen position after iteration i (= start of iteration i+1). This is
    // also what the path's M command should be (= last iteration's end) so
    // the closing Z snaps cleanly.
    let pen_after = |i: usize| -> (f64, f64) {
        let next_idx = (i + 1) % n;
        let nid = nodes[i];
        let next_nid = nodes[next_idx];
        if let Some(ch) = &chain_match[i] {
            // Smooth: Q ended at smooth_end(this -> next).
            smooth_end(ch, nid, next_nid)
        } else if is_kink(i) {
            // Kink: pen-end depends on which branch of the kink emission
            // fires (mirrors the actual loop body below).
            //  * partial_next interior partner → mid(part, partner) (also
            //    the ending of a crossing-kink's single smooth Q)
            //  * partial_next endpoint partner → partner.pos
            //  * no partial_next, next is kink → next's kink_pos (the L
            //    kink→kink branch moves the pen there)
            //  * otherwise → this kink_pos
            let next_loop = next_nid;
            if let Some(part) = partial_chain(nid, next_loop) {
                if let Some(p) = chain_partner(nid, next_loop) {
                    let partner_interior = p.prev.is_some() && p.next.is_some();
                    return if partner_interior { mid(part.pos, p.pos) } else { p.pos };
                }
            }
            if is_kink(next_idx) {
                return map
                    .kink_pos
                    .get(&next_nid)
                    .copied()
                    .unwrap_or_else(|| grid_pos(next_nid));
            }
            map.kink_pos.get(&nid).copied().unwrap_or_else(|| grid_pos(nid))
        } else {
            // Grid: L moves to next-side midpoint when next is smooth, to
            // next's kink_pos when next is a kink, or to the grid corner.
            if is_optimized(next_nid) {
                if is_kink(next_idx) {
                    map.kink_pos
                        .get(&next_nid)
                        .copied()
                        .unwrap_or_else(|| grid_pos(next_nid))
                } else if let Some(ch_next) = &chain_match[next_idx] {
                    mid(grid_pos(nid), ch_next.pos)
                } else {
                    grid_pos(nid)
                }
            } else {
                grid_pos(nid)
            }
        }
    };

    let m_pos = pen_after(n - 1);
    let mut pen = m_pos;
    data.append(cmd_move(m_pos));

    for i in 0..n {
        let next_idx = (i + 1) % n;
        let nid = nodes[i];
        let next_nid = nodes[next_idx];

        if let Some(ch) = &chain_match[i] {
            // Smooth iteration: Q with the chain's CP as control.
            let end = smooth_end(ch, nid, next_nid);
            data.append(cmd_quad(ch.pos, end));
            pen = end;
        } else if is_kink(i) {
            let prev_loop = nodes[(i + n - 1) % n];
            let kp = map.kink_pos.get(&nid).copied().unwrap_or_else(|| grid_pos(nid));
            // τ for the de Casteljau split below. 0.5 for ordinary kinks
            // (T-junction stems, plain endpoints) where kp = B(0.5) of the
            // span at this CP. For IS_CROSSING kinks, τ = crossing_t so
            // the half-Bezier endpoints land on the geometric intersection
            // (which is generally not at t = 0.5).
            let t = kink_t(nid);

            // Q on the prev side: face arrived at this kink along an interior
            // chain. The shape depends on the chain partner at prev_loop.
            //   * Interior partner — prev iteration emitted a smooth Q ending
            //     at mid(partner.pos, this_chain.pos); render the second half
            //     of the equivalent Bezier (de Casteljau split at τ) to land
            //     on kp. Sub-Bezier control: lerp(pen, part.pos, τ).
            //   * Endpoint partner — pen comes in at partner.pos (the chain's
            //     clamped end). Render the full B-spline segment AT this CP
            //     with prev-side ghost expansion as a single Q from pen
            //     through control = part.pos to kp.
            if let Some(part) = partial_chain(nid, prev_loop) {
                let partner_interior = chain_partner(nid, prev_loop)
                    .is_some_and(|p| p.prev.is_some() && p.next.is_some());
                let control = if partner_interior { lerp(pen, part.pos, t) } else { part.pos };
                data.append(cmd_quad(control, kp));
                pen = kp;
            }

            // Q on the next side: face leaves this kink along an interior
            // chain. Mirror of the prev side.
            //   * Interior partner — emit the first-half de Casteljau Q from
            //     kp to mid(this_chain.pos, partner.pos). Sub-Bezier control:
            //     lerp(part.pos, end, τ).
            //   * Endpoint partner — the chain ends at next_loop with a
            //     clamped Bézier landing on partner.pos. Emit a single Q with
            //     control = part.pos and end = partner.pos.
            if let Some(part) = partial_chain(nid, next_nid) {
                if let Some(partner) = chain_partner(nid, next_nid) {
                    let partner_interior = partner.prev.is_some() && partner.next.is_some();
                    if partner_interior {
                        if (pen.0 - kp.0).abs() > 1e-9 || (pen.1 - kp.1).abs() > 1e-9 {
                            data.append(cmd_line(kp));
                        }
                        let end = mid(part.pos, partner.pos);
                        let control = lerp(part.pos, end, t);
                        data.append(cmd_quad(control, end));
                        pen = end;
                    } else {
                        // Pen sits where the previous chain (on which the
                        // face arrived) clamped — partner_for_prev.pos. The
                        // through chain's last segment AT this CP runs from
                        // mid(prev_through, this_pos) through control =
                        // this_pos (= part.pos) to partner.pos via ghost
                        // expansion. We approximate by starting from pen so
                        // the SVG remains continuous.
                        data.append(cmd_quad(part.pos, partner.pos));
                        pen = partner.pos;
                    }
                }
            } else if !is_kink(next_idx) && !is_optimized(next_nid) {
                // Pen stays at kp; next iter handles its own start.
            } else if is_kink(next_idx) {
                // Plain kink chained to another plain kink: line over.
                let next_kp = map
                    .kink_pos
                    .get(&next_nid)
                    .copied()
                    .unwrap_or_else(|| grid_pos(next_nid));
                if (pen.0 - next_kp.0).abs() > 1e-9 || (pen.1 - next_kp.1).abs() > 1e-9 {
                    data.append(cmd_line(next_kp));
                    pen = next_kp;
                }
            }
        } else {
            // Grid: L through the grid corner; if the next is a smooth chain,
            // also L to the midpoint so the next Q starts there.
            let p = grid_pos(nid);
            data.append(cmd_line(p));
            pen = p;
            if is_optimized(next_nid) {
                let target = if is_kink(next_idx) {
                    map.kink_pos
                        .get(&next_nid)
                        .copied()
                        .unwrap_or_else(|| grid_pos(next_nid))
                } else if let Some(ch_next) = &chain_match[next_idx] {
                    mid(p, ch_next.pos)
                } else {
                    p
                };
                data.append(cmd_line(target));
                pen = target;
            }
        }
    }
    data.append(Command::Close);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Render vectorize-gpu pipeline output as a filled-region SVG document.
pub fn render_svg(data: &VectorizeData, pixels: &[u32]) -> String {
    let (w, h) = (data.img_w, data.img_h);

    let edges = build_cell_edges(data, pixels);
    let faces = trace_faces(&edges);
    let map = build_node_map(data);

    let mut doc = svg::Document::new()
        .set("viewBox", (0, 0, w, h))
        .set("width", w * 4)
        .set("height", h * 4)
        .set("shape-rendering", "geometricPrecision");

    // Group face paths by color into one `Data` builder per color, so each
    // fill becomes a single `<path>` element in the document. Every color
    // region — including what would visually look like the background — is
    // emitted explicitly so adjacent regions share a path-vs-path boundary
    // and the SVG renderer's AA stays consistent across all seams.
    let mut by_color: BTreeMap<u32, Data> = BTreeMap::new();
    for (nodes, color) in &faces {
        if *color == VOID_COLOR { continue; }
        let entry = by_color.entry(*color).or_default();
        append_face_path(nodes, &map, entry);
    }

    for (color, data) in by_color {
        doc = doc.add(
            svg::node::element::Path::new()
                .set("fill", hex(color))
                .set("fill-rule", "nonzero")
                .set("stroke", "none")
                .set("d", data),
        );
    }

    doc.to_string()
}
