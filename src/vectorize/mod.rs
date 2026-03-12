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
pub mod rasterize;
pub mod svg;
pub mod voronoi;

/// Vectorize a pixel buffer to an SVG string.
///
/// `pixels` is a flat array of ARGB u32 values (0x00RRGGBB).
/// Returns a complete SVG document as a string.
pub fn vectorize_to_svg(pixels: &[u32], width: usize, height: usize) -> String {
    let (paths, w, h, bg_color) = vectorize_paths(pixels, width, height);
    svg::render_svg(&paths, w, h, bg_color)
}

/// Vectorize a pixel buffer and rasterize at the given scale.
/// Returns (pixels, output_width, output_height).
pub fn vectorize_to_raster(
    pixels: &[u32], width: usize, height: usize, scale: usize,
) -> (Vec<u32>, usize, usize) {
    let (paths, w, h, bg_color) = vectorize_paths(pixels, width, height);
    let out_w = w * scale;
    let out_h = h * scale;
    let buf = rasterize::rasterize(&paths, w, h, bg_color, scale);
    (buf, out_w, out_h)
}

/// Shared vectorization pipeline: returns (paths, width, height, bg_color).
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

    // Step 1: Build similarity graph with diagonal crossing resolution
    let graph = graph::build(px, w, h);

    // Step 2: Build reshaped cell graph (Voronoi diagram per Section 3.2)
    let paths = contour::extract_cells_smooth(px, &graph);

    let bg_color = detect_background_color(px, w, h);
    (paths, w, h, bg_color)
}

/// Detect background color: most common color along the image edges,
/// but only if it also covers a significant portion of the total image.
/// Returns a sentinel (0xFFFFFFFF) if no clear background is found.
fn detect_background_color(pixels: &[u32], width: usize, height: usize) -> u32 {
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

    // Verify: the candidate must cover at least 20% of all pixels to be a true background.
    // This prevents dark outlines or small border elements from being misidentified.
    let total = pixels.len();
    let coverage = pixels.iter().filter(|&&p| p == candidate).count();
    if coverage * 5 >= total {
        candidate
    } else {
        // No dominant background — use a sentinel that won't match any real color
        0xFFFFFFFF
    }
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
