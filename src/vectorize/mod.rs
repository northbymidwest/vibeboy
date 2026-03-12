//! Kopf-Lischinski pixel-art vectorization.
//!
//! Converts a pixel-art frame buffer into smooth SVG vector graphics.
//! Pipeline:
//! 0. Detect and collapse nearest-neighbor upscaling (find native pixel grid)
//! 1. Build similarity graph (connect similar adjacent pixels)
//! 2. Resolve ambiguous diagonal crossings with heuristics
//! 3. Build grid graph, deform at diagonals, collapse valence-2 nodes
//! 4. Extract boundary contours from deformed grid
//! 5. Fit quadratic B-splines to contour points
//! 6. Serialize to SVG

pub mod contour;
pub mod graph;
#[cfg(feature = "gpu")]
pub mod gpu_rasterize;
pub mod rasterize;
pub mod svg;
pub mod voronoi;

/// Caches vectorized paths between frames to skip re-vectorization when
/// the source pixel buffer hasn't changed.
pub struct VectorizeCache {
    prev_pixels: Vec<u32>,
    cached_paths: Vec<contour::ColorPath>,
    cached_bg_color: u32,
}

impl VectorizeCache {
    pub fn new() -> Self {
        Self {
            prev_pixels: Vec::new(),
            cached_paths: Vec::new(),
            cached_bg_color: 0,
        }
    }

    /// Returns cached (paths, bg_color) if pixels unchanged, otherwise
    /// runs vectorization, updates cache, and returns new results.
    pub fn get_paths(&mut self, pixels: &[u32], width: usize, height: usize)
        -> (&[contour::ColorPath], u32)
    {
        if self.prev_pixels.len() == pixels.len() && self.prev_pixels == pixels {
            return (&self.cached_paths, self.cached_bg_color);
        }
        let (paths, bg_color) = vectorize_core(pixels, width, height);
        self.prev_pixels.clear();
        self.prev_pixels.extend_from_slice(pixels);
        self.cached_paths = paths;
        self.cached_bg_color = bg_color;
        (&self.cached_paths, self.cached_bg_color)
    }
}

/// Vectorize a pixel buffer to an SVG string.
///
/// `pixels` is a flat array of ARGB u32 values (0x00RRGGBB).
/// Returns a complete SVG document as a string.
pub fn vectorize_to_svg(pixels: &[u32], width: usize, height: usize) -> String {
    let (paths, w, h, bg_color) = vectorize_paths(pixels, width, height);
    svg::render_svg(&paths, w, h, bg_color)
}

/// Vectorize a pixel buffer and rasterize at the given integer scale.
/// Output is exactly (width*scale) x (height*scale).
/// Returns (pixels, output_width, output_height).
pub fn vectorize_to_raster(
    pixels: &[u32], width: usize, height: usize, scale: usize,
) -> (Vec<u32>, usize, usize) {
    let (paths, bg_color) = vectorize_core(pixels, width, height);
    let out_w = width * scale;
    let out_h = height * scale;
    let buf = rasterize::rasterize(&paths, width, height, bg_color, scale);
    (buf, out_w, out_h)
}

/// Vectorize a pixel buffer and rasterize at a floating-point scale factor.
/// Uses a single uniform scale so the aspect ratio is always preserved.
/// Returns (pixels, output_width, output_height).
pub fn vectorize_to_raster_scaled(
    pixels: &[u32], width: usize, height: usize, scale: f64,
) -> (Vec<u32>, usize, usize) {
    let (paths, bg_color) = vectorize_core(pixels, width, height);
    rasterize::rasterize_scaled(&paths, width, height, bg_color, scale)
}

/// Quantize colors to merge near-identical shades (e.g. #555555 vs #565656).
/// PPU output can have ±1-2 per channel jitter; without quantization the
/// vectorizer creates separate cells for each unique value, producing
/// hundreds of spurious boundaries that show as white outlines.
fn quantize_pixels(pixels: &[u32]) -> Vec<u32> {
    pixels.iter().map(|&c| {
        // Round each channel to nearest multiple of 4.
        // Max shift ±2 per channel — imperceptible, but merges jitter variants.
        let r = (((c >> 16) & 0xFF) + 2).min(255) & !3;
        let g = (((c >> 8) & 0xFF) + 2).min(255) & !3;
        let b = ((c & 0xFF) + 2).min(255) & !3;
        (r << 16) | (g << 8) | b
    }).collect()
}

/// Core vectorization: graph → contour → paths. No upscale collapse.
pub fn vectorize_core(
    pixels: &[u32], width: usize, height: usize,
) -> (Vec<contour::ColorPath>, u32) {
    let qpixels = quantize_pixels(pixels);
    let graph = graph::build(&qpixels, width, height);
    let paths = contour::extract_cells_smooth(&qpixels, &graph);
    let (bg_color, _) = detect_background_color(&qpixels, width, height);
    (paths, bg_color)
}

/// Full vectorization pipeline with upscale collapse detection.
/// Returns (paths, native_width, native_height, bg_color).
fn vectorize_paths(
    pixels: &[u32], width: usize, height: usize,
) -> (Vec<contour::ColorPath>, usize, usize, u32) {
    // Step 0: Detect and collapse nearest-neighbor upscaling
    let (native_pixels, nw, nh) = detect_and_collapse(pixels, width, height);
    let (px, w, h) = if !native_pixels.is_empty() {
        (native_pixels.as_slice(), nw, nh)
    } else {
        (pixels, width, height)
    };

    let (paths, bg_color) = vectorize_core(px, w, h);
    (paths, w, h, bg_color)
}

/// Detect background color: most common color along the image edges.
/// Returns the most common edge color (used for buffer init and optional path skipping).
/// Also returns whether the color is a strong background (covers ≥20% of pixels),
/// meaning we can safely skip rendering paths with that color.
fn detect_background_color(pixels: &[u32], width: usize, height: usize) -> (u32, bool) {
    let mut edge_counts = std::collections::HashMap::new();
    for x in 0..width {
        *edge_counts.entry(pixels[x]).or_insert(0u32) += 1;
        *edge_counts.entry(pixels[(height - 1) * width + x]).or_insert(0u32) += 1;
    }
    for y in 1..height - 1 {
        *edge_counts.entry(pixels[y * width]).or_insert(0u32) += 1;
        *edge_counts.entry(pixels[y * width + width - 1]).or_insert(0u32) += 1;
    }
    let candidate = edge_counts.into_iter().max_by_key(|&(_, c)| c).map(|(color, _)| color).unwrap_or(0);

    // Strong background: covers ≥20% of all pixels — safe to skip rendering
    let total = pixels.len();
    let coverage = pixels.iter().filter(|&&p| p == candidate).count();
    let is_strong = coverage * 5 >= total;
    (candidate, is_strong)
}

/// Detect nearest-neighbor upscaling and collapse to native pixel resolution.
fn detect_and_collapse(pixels: &[u32], width: usize, height: usize) -> (Vec<u32>, usize, usize) {
    let scale = detect_pixel_scale(pixels, width, height);
    if scale < 2 {
        return (Vec::new(), width, height);
    }

    let nw = ((width as f64) / (scale as f64)).round() as usize;
    let nh = ((height as f64) / (scale as f64)).round() as usize;
    if nw == 0 || nh == 0 {
        return (Vec::new(), width, height);
    }

    let mut native = Vec::with_capacity(nw * nh);
    for ny in 0..nh {
        for nx in 0..nw {
            let sx = ((nx as f64 + 0.5) * (width as f64) / (nw as f64)) as usize;
            let sy = ((ny as f64 + 0.5) * (height as f64) / (nh as f64)) as usize;
            let sx = sx.min(width - 1);
            let sy = sy.min(height - 1);
            native.push(pixels[sy * width + sx]);
        }
    }

    eprintln!("Detected ~{}x upscaling: {}x{} → {}x{}", scale, width, height, nw, nh);
    (native, nw, nh)
}

fn detect_pixel_scale(pixels: &[u32], width: usize, height: usize) -> usize {
    let bg = pixels[0];

    let mut run_counts = [0u32; 64];
    let sample_rows = height.min(64);
    for row_idx in 0..sample_rows {
        let y = row_idx * height / sample_rows;
        let row_start = y * width;
        let mut x = 0;
        while x < width {
            let color = pixels[row_start + x];
            let start = x;
            while x < width && pixels[row_start + x] == color { x += 1; }
            let run_len = x - start;
            if color != bg && run_len < 64 { run_counts[run_len] += 1; }
        }
    }

    let sample_cols = width.min(64);
    for col_idx in 0..sample_cols {
        let x = col_idx * width / sample_cols;
        let mut y = 0;
        while y < height {
            let color = pixels[y * width + x];
            let start = y;
            while y < height && pixels[y * width + x] == color { y += 1; }
            let run_len = y - start;
            if color != bg && run_len < 64 { run_counts[run_len] += 1; }
        }
    }

    let mut best_scale = 0;
    let mut best_count = 0;
    for s in 2..32 {
        let count = run_counts[s];
        if count > best_count { best_count = count; best_scale = s; }
    }

    if best_count < 4 || best_scale < 2 { return 0; }

    let mut total_runs = 0u32;
    let mut matching_runs = 0u32;
    for len in 2..64 {
        let count = run_counts[len];
        if count == 0 { continue; }
        total_runs += count;
        let remainder = len % best_scale;
        if remainder <= 1 || remainder >= best_scale - 1 { matching_runs += count; }
    }

    if total_runs == 0 || matching_runs * 100 / total_runs < 70 { return 0; }

    // Verify: sample scale×scale blocks and check they are uniform.
    // This catches false positives on native pixel art with coincidental run patterns.
    let nw = width / best_scale;
    let nh = height / best_scale;
    if nw < 2 || nh < 2 { return 0; }

    let mut uniform_blocks = 0u32;
    let mut total_blocks = 0u32;
    for by in 0..nh {
        for bx in 0..nw {
            let base_color = pixels[by * best_scale * width + bx * best_scale];
            let mut uniform = true;
            'block: for dy in 0..best_scale {
                for dx in 0..best_scale {
                    let sy = by * best_scale + dy;
                    let sx = bx * best_scale + dx;
                    if sy < height && sx < width && pixels[sy * width + sx] != base_color {
                        uniform = false;
                        break 'block;
                    }
                }
            }
            total_blocks += 1;
            if uniform { uniform_blocks += 1; }
        }
    }

    if total_blocks > 0 && uniform_blocks * 100 / total_blocks >= 90 { best_scale } else { 0 }
}
