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
#[inline]
fn similar(a: u32, b: u32) -> bool {
    if a == b {
        return true;
    }
    let ar = ((a >> 16) & 0xFF) as f64;
    let ag = ((a >> 8) & 0xFF) as f64;
    let ab = (a & 0xFF) as f64;
    let br = ((b >> 16) & 0xFF) as f64;
    let bg = ((b >> 8) & 0xFF) as f64;
    let bb = (b & 0xFF) as f64;

    let dr = ar - br;
    let dg = ag - bg;
    let db = ab - bb;

    let dy = 0.299 * dr + 0.587 * dg + 0.114 * db;
    let du = 0.492 * (db - dy);
    let dv = 0.877 * (dr - dy);

    dy.abs() <= 48.0 && du.abs() <= 7.0 && dv.abs() <= 6.0
}

#[inline]
fn px(pixels: &[u32], w: usize, x: usize, y: usize) -> u32 {
    pixels[y * w + x]
}

/// Build the initial similarity graph with all similar edges.
pub fn build(pixels: &[u32], width: usize, height: usize) -> SimilarityGraph {
    let mut edges = vec![PixelEdges::default(); width * height];

    for y in 0..height {
        for x in 0..width {
            let c = px(pixels, width, x, y);
            if x + 1 < width {
                edges[y * width + x].right = similar(c, px(pixels, width, x + 1, y));
            }
            if y + 1 < height {
                edges[y * width + x].down = similar(c, px(pixels, width, x, y + 1));
            }
            if x + 1 < width && y + 1 < height {
                edges[y * width + x].down_right = similar(c, px(pixels, width, x + 1, y + 1));
            }
            if x > 0 && y + 1 < height {
                edges[y * width + x].down_left = similar(c, px(pixels, width, x - 1, y + 1));
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
        let neighbors = pixel_neighbors(edges, w, h, node.0, node.1);
        if neighbors.len() != 2 {
            continue;
        }

        for &(nx, ny) in &neighbors {
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
fn component_size(
    edges: &[PixelEdges],
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> i32 {
    let min_x = x0.min(x1) as i32;
    let min_y = y0.min(y1) as i32;

    let mut visited = HashSet::new();
    visited.insert((x0, y0));
    visited.insert((x1, y1));
    let mut stack = vec![(x0, y0), (x1, y1)];

    while let Some((cx, cy)) = stack.pop() {
        for &(nx, ny) in &pixel_neighbors(edges, w, h, cx, cy) {
            if visited.contains(&(nx, ny)) {
                continue;
            }
            let dx = nx as i32 - min_x;
            let dy = ny as i32 - min_y;
            if dx < -3 || dx > 4 || dy < -3 || dy > 4 {
                continue;
            }
            visited.insert((nx, ny));
            stack.push((nx, ny));
        }
    }

    visited.len() as i32
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
fn pixel_neighbors(
    edges: &[PixelEdges],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
) -> Vec<(usize, usize)> {
    let mut neighbors = Vec::with_capacity(8);
    let e = edges[y * w + x];
    if e.right && x + 1 < w {
        neighbors.push((x + 1, y));
    }
    if e.down && y + 1 < h {
        neighbors.push((x, y + 1));
    }
    if e.down_right && x + 1 < w && y + 1 < h {
        neighbors.push((x + 1, y + 1));
    }
    if e.down_left && x > 0 && y + 1 < h {
        neighbors.push((x - 1, y + 1));
    }
    if x > 0 && edges[y * w + (x - 1)].right {
        neighbors.push((x - 1, y));
    }
    if y > 0 && edges[(y - 1) * w + x].down {
        neighbors.push((x, y - 1));
    }
    if x > 0 && y > 0 && edges[(y - 1) * w + (x - 1)].down_right {
        neighbors.push((x - 1, y - 1));
    }
    if x + 1 < w && y > 0 && edges[(y - 1) * w + (x + 1)].down_left {
        neighbors.push((x + 1, y - 1));
    }
    neighbors
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
