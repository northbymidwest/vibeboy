//! Similarity graph construction for the Kopf-Lischinski algorithm.
//!
//! Builds a graph where nodes are pixels and edges connect similar neighbors
//! (4-connected and diagonal). For ambiguous 2x2 diagonal crossings, applies
//! heuristics: curves length, sparse pixels, and island size.

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

/// YCbCr-weighted color distance (same weighting as xBR scaler).
#[inline]
fn color_dist(a: u32, b: u32) -> f32 {
    if a == b {
        return 0.0;
    }
    let ar = ((a >> 16) & 0xFF) as f32;
    let ag = ((a >> 8) & 0xFF) as f32;
    let ab = (a & 0xFF) as f32;
    let br = ((b >> 16) & 0xFF) as f32;
    let bg = ((b >> 8) & 0xFF) as f32;
    let bb = (b & 0xFF) as f32;

    let dr = ar - br;
    let dg = ag - bg;
    let db = ab - bb;

    let dy = 0.299 * dr + 0.587 * dg + 0.114 * db;
    let dcb = -0.169 * dr - 0.331 * dg + 0.500 * db;
    let dcr = 0.500 * dr - 0.419 * dg - 0.081 * db;

    48.0 * dy * dy + 7.0 * dcb * dcb + 6.0 * dcr * dcr
}

/// Threshold for considering two colors "similar".
const SIMILARITY_THRESHOLD: f32 = 100.0;

#[inline]
fn similar(a: u32, b: u32) -> bool {
    color_dist(a, b) < SIMILARITY_THRESHOLD
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

/// For each 2x2 block where both diagonals are connected, remove one
/// using Kopf-Lischinski heuristics.
fn resolve_crossings(pixels: &[u32], w: usize, h: usize, edges: &mut [PixelEdges]) {
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            // Check if both diagonals of the 2x2 block (x,y)-(x+1,y+1) are connected
            let has_main = edges[y * w + x].down_right; // (x,y) -> (x+1,y+1)
            let has_anti = edges[y * w + (x + 1)].down_left; // (x+1,y) -> (x,y+1)

            if has_main && has_anti {
                // Both diagonals present — ambiguous crossing, must remove one
                let score = crossing_heuristic(pixels, w, h, edges, x, y);
                if score > 0 {
                    // Keep main diagonal, remove anti
                    edges[y * w + (x + 1)].down_left = false;
                } else if score < 0 {
                    // Keep anti diagonal, remove main
                    edges[y * w + x].down_right = false;
                } else {
                    // Tie: remove both (valence rule from paper)
                    edges[y * w + x].down_right = false;
                    edges[y * w + (x + 1)].down_left = false;
                }
            }
        }
    }
}

/// Heuristic score for a 2x2 crossing. Positive favors main diagonal,
/// negative favors anti diagonal.
fn crossing_heuristic(pixels: &[u32], w: usize, h: usize, edges: &[PixelEdges], x: usize, y: usize) -> i32 {
    let tl = px(pixels, w, x, y);
    let tr = px(pixels, w, x + 1, y);
    let bl = px(pixels, w, x, y + 1);
    let br = px(pixels, w, x + 1, y + 1);

    let mut score: i32 = 0;

    // Heuristic 1: Curves — prefer the diagonal whose endpoints have higher
    // valence (more connections), as it likely continues a longer curve.
    let val_main = valence(edges, w, h, x, y) + valence(edges, w, h, x + 1, y + 1);
    let val_anti = valence(edges, w, h, x + 1, y) + valence(edges, w, h, x, y + 1);
    score += (val_main as i32) - (val_anti as i32);

    // Heuristic 2: Sparse pixels — prefer connecting the color that has
    // fewer neighbors of the same color (the "island" heuristic).
    let island_main = island_size_approx(pixels, w, h, x, y) + island_size_approx(pixels, w, h, x + 1, y + 1);
    let island_anti = island_size_approx(pixels, w, h, x + 1, y) + island_size_approx(pixels, w, h, x, y + 1);
    // Prefer connecting the sparser (smaller island) pair
    if island_main < island_anti {
        score += 1;
    } else if island_anti < island_main {
        score -= 1;
    }

    // Heuristic 3: Color distance — if one diagonal pair is much more similar
    let dist_main = color_dist(tl, br);
    let dist_anti = color_dist(tr, bl);
    if dist_main < dist_anti * 0.5 {
        score += 1;
    } else if dist_anti < dist_main * 0.5 {
        score -= 1;
    }

    score
}

/// Count how many edges connect to pixel (x, y).
fn valence(edges: &[PixelEdges], w: usize, _h: usize, x: usize, y: usize) -> u32 {
    let mut v = 0u32;
    let e = edges[y * w + x];
    if e.right { v += 1; }
    if e.down { v += 1; }
    if e.down_right { v += 1; }
    if e.down_left { v += 1; }
    // Incoming edges
    if x > 0 && edges[y * w + (x - 1)].right { v += 1; }
    if y > 0 && edges[(y - 1) * w + x].down { v += 1; }
    if x > 0 && y > 0 && edges[(y - 1) * w + (x - 1)].down_right { v += 1; }
    if x + 1 < w && y > 0 && edges[(y - 1) * w + (x + 1)].down_left { v += 1; }
    v
}

/// Approximate island size: count 4-connected same-color pixels in a small neighborhood.
fn island_size_approx(pixels: &[u32], w: usize, h: usize, x: usize, y: usize) -> u32 {
    let c = px(pixels, w, x, y);
    let mut count = 0u32;
    for dy in -2i32..=2 {
        for dx in -2i32..=2 {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                if px(pixels, w, nx as usize, ny as usize) == c {
                    count += 1;
                }
            }
        }
    }
    count
}
