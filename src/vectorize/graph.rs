//! Similarity graph construction for the Kopf-Lischinski algorithm.
//!
//! Builds a graph where nodes are pixels and edges connect similar neighbors
//! (4-connected and diagonal). For ambiguous 2x2 diagonal crossings, applies
//! heuristics: curves length, sparse pixels, and island size.

use std::collections::HashSet;

/// Edge connectivity for a pixel. Each bool indicates whether the pixel is
/// "similar" (connected) to that neighbor.
#[derive(Clone, Copy, Default)]
pub struct PixelEdges {
    pub right: bool,
    pub down: bool,
    pub down_right: bool,
    pub down_left: bool,
}

/// The similarity graph: for each pixel (x, y), stores which neighbors it connects to.
/// Only stores edges going right, down, down-right, down-left to avoid duplication.
pub struct SimilarityGraph {
    pub width: usize,
    pub height: usize,
    pub edges: Vec<PixelEdges>,
}

impl SimilarityGraph {
    pub fn edge(&self, x: usize, y: usize) -> &PixelEdges {
        &self.edges[y * self.width + x]
    }
}

/// Per-channel YUV similarity check matching the reference implementation.
/// Uses PAL YUV color space with independent per-channel thresholds:
///   |ΔY| ≤ 48, |ΔU| ≤ 7, |ΔV| ≤ 6
/// Computed with fixed-point integer math (×1000) to avoid f64.
/// Branchless: combines all conditions to avoid branch misprediction.
#[inline(always)]
fn similar(a: u32, b: u32) -> bool {
    if a == b {
        return true;
    }
    let dr = ((a >> 16) & 0xFF) as i32 - ((b >> 16) & 0xFF) as i32;
    let dg = ((a >> 8) & 0xFF) as i32 - ((b >> 8) & 0xFF) as i32;
    let db = (a & 0xFF) as i32 - (b & 0xFF) as i32;

    let dy1000 = 299 * dr + 587 * dg + 114 * db;
    let du_num = 492i64 * (db as i64 * 1000 - dy1000 as i64);
    let dv_num = 877i64 * (dr as i64 * 1000 - dy1000 as i64);

    (dy1000.abs() <= 48000) & (du_num.abs() <= 7_000_000) & (dv_num.abs() <= 6_000_000)
}

#[inline]
fn px(pixels: &[u32], w: usize, x: usize, y: usize) -> u32 {
    pixels[y * w + x]
}

/// Build the initial similarity graph with all similar edges.
/// Uses direct row-pointer indexing to avoid repeated multiply in px() lookups.
pub fn build(pixels: &[u32], width: usize, height: usize) -> SimilarityGraph {
    let mut edges = vec![PixelEdges::default(); width * height];

    for y in 0..height {
        let row = y * width;
        let next_row = row + width; // only valid when y + 1 < height
        let has_next_row = y + 1 < height;

        for x in 0..width {
            let c = pixels[row + x];
            let idx = row + x;

            if x + 1 < width {
                edges[idx].right = similar(c, pixels[row + x + 1]);
            }
            if has_next_row {
                edges[idx].down = similar(c, pixels[next_row + x]);
                if x + 1 < width {
                    edges[idx].down_right = similar(c, pixels[next_row + x + 1]);
                }
                if x > 0 {
                    edges[idx].down_left = similar(c, pixels[next_row + x - 1]);
                }
            }
        }
    }

    // Resolve ambiguous diagonal crossings in 2x2 blocks
    resolve_crossings(pixels, width, height, &mut edges);

    SimilarityGraph { width, height, edges }
}

/// For each 2x2 block where both diagonals are connected, resolve them.
/// Matches reference: two-pass approach — weight all ambiguous pairs on the
/// same graph state, then resolve by removing min-weight diagonal.
fn resolve_crossings(_pixels: &[u32], w: usize, h: usize, edges: &mut [PixelEdges]) {
    // Pass 1: Identify crossings
    let mut remove_both = Vec::new();
    let mut ambiguous = Vec::new();

    for y in 0..h - 1 {
        for x in 0..w - 1 {
            let has_main = edges[y * w + x].down_right;
            let has_anti = edges[y * w + (x + 1)].down_left;

            if !has_main || !has_anti {
                continue;
            }

            // Check if ALL four cardinal edges exist in this 2x2 block
            // Paper: "If a 2×2 block is fully connected, it is part of a
            // continuously shaded region. The two diagonal connections can
            // be safely removed."
            let has_top = edges[y * w + x].right;
            let has_bottom = edges[(y + 1) * w + x].right;
            let has_left = edges[y * w + x].down;
            let has_right = edges[y * w + (x + 1)].down;

            if has_top && has_bottom && has_left && has_right {
                // Fully connected block → remove both diagonals
                remove_both.push((x, y));
            } else {
                // Pure diagonal crossing (exactly 2 edges) → apply heuristics
                ambiguous.push((x, y));
            }
        }
    }

    // Remove fully-connected-block diagonals
    for (x, y) in remove_both {
        edges[y * w + x].down_right = false;
        edges[y * w + (x + 1)].down_left = false;
    }

    // Fast path for noisy images: skip expensive heuristics (curve_length uses
    // HashSet-based BFS per crossing). Conservatively remove both diagonals.
    // Threshold: >2% of pixels have ambiguous crossings.
    if ambiguous.len() > w * h / 50 {
        for (x, y) in ambiguous {
            edges[y * w + x].down_right = false;
            edges[y * w + (x + 1)].down_left = false;
        }
        return;
    }

    // Pass 2: For each ambiguous pair, each heuristic VOTES for one side.
    // The vote weight is the DIFFERENCE between the two sides' raw scores.
    // Paper: "choose to keep the connection that has aggregated the most weight"
    let votes: Vec<(i32, i32)> = ambiguous
        .iter()
        .map(|&(x, y)| {
            let mut main_vote = 0i32;
            let mut anti_vote = 0i32;

            // Curves heuristic: longer curve wins, weight = length difference
            let main_curve = curve_length(edges, w, h, x, y, x + 1, y + 1);
            let anti_curve = curve_length(edges, w, h, x + 1, y, x, y + 1);
            if main_curve > anti_curve {
                main_vote += main_curve - anti_curve;
            } else if anti_curve > main_curve {
                anti_vote += anti_curve - main_curve;
            }

            // Sparse heuristic: smaller component wins, weight = size difference
            let main_size = component_size(edges, w, h, x, y, x + 1, y + 1);
            let anti_size = component_size(edges, w, h, x + 1, y, x, y + 1);
            if main_size < anti_size {
                main_vote += anti_size - main_size;
            } else if anti_size < main_size {
                anti_vote += main_size - anti_size;
            }

            // Islands heuristic: valence-1 endpoint → fixed weight 5 to keep
            let main_island = has_valence1_endpoint(edges, w, h, x, y, x + 1, y + 1);
            let anti_island = has_valence1_endpoint(edges, w, h, x + 1, y, x, y + 1);
            if main_island && !anti_island {
                main_vote += 5;
            } else if anti_island && !main_island {
                anti_vote += 5;
            }

            (main_vote, anti_vote)
        })
        .collect();

    // Pass 3: Resolve — keep higher-voted diagonal (tie → remove both)
    for (i, &(x, y)) in ambiguous.iter().enumerate() {
        let (main_v, anti_v) = votes[i];
        if main_v > anti_v {
            // Main wins → remove anti
            edges[y * w + (x + 1)].down_left = false;
        } else if anti_v > main_v {
            // Anti wins → remove main
            edges[y * w + x].down_right = false;
        } else {
            // Tie: remove both
            edges[y * w + x].down_right = false;
            edges[y * w + (x + 1)].down_left = false;
        }
    }
}

/// Curve heuristic: count edges in the curve containing this diagonal edge.
/// A curve is a sequence of edges connecting only valence-2 nodes.
/// Traces through valence-2 nodes from both endpoints. Returns length (min 1).
fn curve_length(
    edges: &[PixelEdges],
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> i32 {
    let mut seen_edges: HashSet<((usize, usize), (usize, usize))> = HashSet::new();
    seen_edges.insert(sorted_edge((x0, y0), (x1, y1)));
    let mut stack = vec![(x0, y0), (x1, y1)];

    while let Some(node) = stack.pop() {
        let (nbuf, ncount) = pixel_neighbors(edges, w, h, node.0, node.1);
        if ncount != 2 {
            continue;
        }

        for i in 0..ncount {
            let (nx, ny) = nbuf[i];
            let key = sorted_edge(node, (nx, ny));
            if !seen_edges.contains(&key) {
                seen_edges.insert(key);
                stack.push((nx, ny));
            }
        }
    }

    seen_edges.len() as i32
}

fn sorted_edge(
    a: (usize, usize),
    b: (usize, usize),
) -> ((usize, usize), (usize, usize)) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Sparse heuristic: BFS from both endpoints within an 8×8 window.
/// Returns connected component size (smaller = sparser = should be kept).
/// Uses a u64 bitmask (8×8 grid) instead of HashSet for zero-allocation visited tracking.
fn component_size(
    edges: &[PixelEdges],
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> i32 {
    let origin_x = x0.min(x1) as i32 - 3;
    let origin_y = y0.min(y1) as i32 - 3;

    #[inline(always)]
    fn bit_idx(x: i32, y: i32, ox: i32, oy: i32) -> Option<u32> {
        let lx = x - ox;
        let ly = y - oy;
        if lx >= 0 && lx < 8 && ly >= 0 && ly < 8 {
            Some((ly * 8 + lx) as u32)
        } else {
            None
        }
    }

    let mut visited: u64 = 0;
    let mut stack = [(0u8, 0u8); 64];
    let mut sp = 0;
    let mut count = 0i32;

    // Seed both endpoints
    for &(sx, sy) in &[(x0, y0), (x1, y1)] {
        if let Some(bit) = bit_idx(sx as i32, sy as i32, origin_x, origin_y) {
            let mask = 1u64 << bit;
            if visited & mask == 0 {
                visited |= mask;
                count += 1;
                stack[sp] = (sx as u8, sy as u8);
                sp += 1;
            }
        }
    }

    while sp > 0 {
        sp -= 1;
        let (cx, cy) = (stack[sp].0 as usize, stack[sp].1 as usize);
        let (nbuf, ncount) = pixel_neighbors(edges, w, h, cx, cy);
        for i in 0..ncount {
            let (nx, ny) = nbuf[i];
            if let Some(bit) = bit_idx(nx as i32, ny as i32, origin_x, origin_y) {
                let mask = 1u64 << bit;
                if visited & mask == 0 {
                    visited |= mask;
                    count += 1;
                    if sp < 64 {
                        stack[sp] = (nx as u8, ny as u8);
                        sp += 1;
                    }
                }
            }
        }
    }

    count
}

/// Island heuristic: true if either endpoint has valence 1
/// (cutting this edge would create a single disconnected pixel).
fn has_valence1_endpoint(
    edges: &[PixelEdges],
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> bool {
    full_valence(edges, w, h, x0, y0) == 1 || full_valence(edges, w, h, x1, y1) == 1
}

/// Get all connected neighbor pixels of (x, y).
/// Returns (array, count) to avoid heap allocation in hot loops.
#[inline]
fn pixel_neighbors(
    edges: &[PixelEdges],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
) -> ([(usize, usize); 8], usize) {
    let mut buf = [(0usize, 0usize); 8];
    let mut n = 0;
    let e = edges[y * w + x];
    if e.right && x + 1 < w {
        buf[n] = (x + 1, y); n += 1;
    }
    if e.down && y + 1 < h {
        buf[n] = (x, y + 1); n += 1;
    }
    if e.down_right && x + 1 < w && y + 1 < h {
        buf[n] = (x + 1, y + 1); n += 1;
    }
    if e.down_left && x > 0 && y + 1 < h {
        buf[n] = (x - 1, y + 1); n += 1;
    }
    if x > 0 && edges[y * w + (x - 1)].right {
        buf[n] = (x - 1, y); n += 1;
    }
    if y > 0 && edges[(y - 1) * w + x].down {
        buf[n] = (x, y - 1); n += 1;
    }
    if x > 0 && y > 0 && edges[(y - 1) * w + (x - 1)].down_right {
        buf[n] = (x - 1, y - 1); n += 1;
    }
    if x + 1 < w && y > 0 && edges[(y - 1) * w + (x + 1)].down_left {
        buf[n] = (x + 1, y - 1); n += 1;
    }
    (buf, n)
}

/// Count all edges connecting to pixel (x, y).
fn full_valence(edges: &[PixelEdges], w: usize, h: usize, x: usize, y: usize) -> u32 {
    let mut v = 0u32;
    let e = edges[y * w + x];
    if e.right && x + 1 < w {
        v += 1;
    }
    if e.down && y + 1 < h {
        v += 1;
    }
    if e.down_right && x + 1 < w && y + 1 < h {
        v += 1;
    }
    if e.down_left && x > 0 && y + 1 < h {
        v += 1;
    }
    if x > 0 && edges[y * w + (x - 1)].right {
        v += 1;
    }
    if y > 0 && edges[(y - 1) * w + x].down {
        v += 1;
    }
    if x > 0 && y > 0 && edges[(y - 1) * w + (x - 1)].down_right {
        v += 1;
    }
    if x + 1 < w && y > 0 && edges[(y - 1) * w + (x + 1)].down_left {
        v += 1;
    }
    v
}
