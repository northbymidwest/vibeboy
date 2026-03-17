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

/// Caches vectorized paths and rasterized output between frames to skip
/// re-vectorization and re-rasterization when the source pixel buffer
/// and scale haven't changed.
pub struct VectorizeLegacyCache {
    adaptive: bool,
    prev_pixels: Vec<u32>,
    cached_paths: Vec<contour::ColorPath>,
    cached_bg_color: u32,
    cached_raster: Vec<u32>,
    cached_raster_w: usize,
    cached_raster_h: usize,
    cached_scale: f64,
}

impl VectorizeLegacyCache {
    pub fn new(adaptive: bool) -> Self {
        Self {
            adaptive,
            prev_pixels: Vec::new(),
            cached_paths: Vec::new(),
            cached_bg_color: 0,
            cached_raster: Vec::new(),
            cached_raster_w: 0,
            cached_raster_h: 0,
            cached_scale: 0.0,
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
        let (paths, bg_color) = vectorize_core_inner(pixels, width, height, self.adaptive);
        self.prev_pixels.clear();
        self.prev_pixels.extend_from_slice(pixels);
        self.cached_paths = paths;
        self.cached_bg_color = bg_color;
        // Invalidate raster cache when paths change
        self.cached_scale = 0.0;
        (&self.cached_paths, self.cached_bg_color)
    }

    /// Vectorize and rasterize in one call, caching both stages.
    /// Returns (pixels, width, height).
    pub fn rasterize(&mut self, pixels: &[u32], width: usize, height: usize, scale: f64)
        -> (&[u32], usize, usize)
    {
        let (_paths, _bg_color) = self.get_paths(pixels, width, height);
        if scale == self.cached_scale && !self.cached_raster.is_empty() {
            return (&self.cached_raster, self.cached_raster_w, self.cached_raster_h);
        }
        let bg = self.cached_bg_color;
        let (buf, w, h) = rasterize::rasterize_scaled(
            &self.cached_paths, width, height, bg, scale,
        );
        self.cached_raster = buf;
        self.cached_raster_w = w;
        self.cached_raster_h = h;
        self.cached_scale = scale;
        (&self.cached_raster, w, h)
    }
}

/// Cache for the shared-chain vectorization pipeline.
/// Caches paths and rasterized output between frames.
pub struct VectorizeCache {
    adaptive: bool,
    prev_pixels: Vec<u32>,
    cached_paths: Vec<contour::ColorPath>,
    cached_bg_color: u32,
    cached_raster: Vec<u32>,
    cached_raster_w: usize,
    cached_raster_h: usize,
    cached_scale: f64,
}

impl VectorizeCache {
    pub fn new(adaptive: bool) -> Self {
        Self {
            adaptive,
            prev_pixels: Vec::new(),
            cached_paths: Vec::new(),
            cached_bg_color: 0,
            cached_raster: Vec::new(),
            cached_raster_w: 0,
            cached_raster_h: 0,
            cached_scale: 0.0,
        }
    }

    /// Returns cached (paths, bg_color) if pixels unchanged, otherwise
    /// runs shared-edge vectorization and updates cache.
    pub fn get_paths(&mut self, pixels: &[u32], width: usize, height: usize)
        -> (&[contour::ColorPath], u32)
    {
        if self.prev_pixels.len() == pixels.len() && self.prev_pixels == pixels {
            return (&self.cached_paths, self.cached_bg_color);
        }
        let graph = graph::build(pixels, width, height);
        let (paths, bg_color) = contour::extract_shared_edge_paths_inner(
            pixels, &graph, self.adaptive,
        );
        self.prev_pixels.clear();
        self.prev_pixels.extend_from_slice(pixels);
        self.cached_paths = paths;
        self.cached_bg_color = bg_color;
        self.cached_scale = 0.0;
        (&self.cached_paths, self.cached_bg_color)
    }

    /// Like get_paths but uses GPU device for optimization when available.
    #[cfg(feature = "sdl3-gpu-shaders")]
    pub fn get_paths_gpu(&mut self, pixels: &[u32], width: usize, height: usize,
        gpu: Option<&crate::scaling::gpu_pipelines::GpuOptRefs>,
    ) -> (&[contour::ColorPath], u32) {
        if self.prev_pixels.len() == pixels.len() && self.prev_pixels == pixels {
            return (&self.cached_paths, self.cached_bg_color);
        }
        let graph = graph::build(pixels, width, height);
        let (paths, bg_color) = contour::extract_shared_edge_paths_gpu(
            pixels, &graph, self.adaptive, gpu,
        );
        self.prev_pixels.clear();
        self.prev_pixels.extend_from_slice(pixels);
        self.cached_paths = paths;
        self.cached_bg_color = bg_color;
        self.cached_scale = 0.0;
        (&self.cached_paths, self.cached_bg_color)
    }

    /// Vectorize and rasterize in one call, caching both stages.
    pub fn rasterize(&mut self, pixels: &[u32], width: usize, height: usize, scale: f64)
        -> (&[u32], usize, usize)
    {
        let (_paths, _bg_color) = self.get_paths(pixels, width, height);
        if scale == self.cached_scale && !self.cached_raster.is_empty() {
            return (&self.cached_raster, self.cached_raster_w, self.cached_raster_h);
        }
        let bg = self.cached_bg_color;
        let (buf, w, h) = rasterize::rasterize_scaled(
            &self.cached_paths, width, height, bg, scale,
        );
        self.cached_raster = buf;
        self.cached_raster_w = w;
        self.cached_raster_h = h;
        self.cached_scale = scale;
        (&self.cached_raster, w, h)
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
/// Detects nearest-neighbor upscaling and collapses to native resolution first,
/// then scales output relative to native dimensions.
/// Returns (pixels, output_width, output_height).
pub fn vectorize_to_raster(
    pixels: &[u32], width: usize, height: usize, scale: usize,
) -> (Vec<u32>, usize, usize) {
    let (native_pixels, nw, nh) = detect_and_collapse(pixels, width, height);
    let (px, w, h) = if !native_pixels.is_empty() {
        (native_pixels.as_slice(), nw, nh)
    } else {
        (pixels, width, height)
    };
    let (paths, bg_color) = vectorize_core(px, w, h);
    let out_w = w * scale;
    let out_h = h * scale;
    let buf = rasterize::rasterize(&paths, w, h, bg_color, scale);
    (buf, out_w, out_h)
}

/// Vectorize a pixel buffer and rasterize at a floating-point scale factor.
/// Detects nearest-neighbor upscaling and collapses to native resolution first.
/// Uses a single uniform scale so the aspect ratio is always preserved.
/// Returns (pixels, output_width, output_height).
pub fn vectorize_to_raster_scaled(
    pixels: &[u32], width: usize, height: usize, scale: f64,
) -> (Vec<u32>, usize, usize) {
    let (native_pixels, nw, nh) = detect_and_collapse(pixels, width, height);
    let (px, w, h) = if !native_pixels.is_empty() {
        (native_pixels.as_slice(), nw, nh)
    } else {
        (pixels, width, height)
    };
    let (paths, bg_color) = vectorize_core(px, w, h);
    rasterize::rasterize_scaled(&paths, w, h, bg_color, scale)
}

/// Vectorize and rasterize using shared-chain rendering (gap-free).
/// Each boundary chain is shared between two regions, eliminating gaps.
/// Returns (pixels, output_width, output_height).
pub fn vectorize_to_raster_shared(
    pixels: &[u32], width: usize, height: usize, scale: usize,
) -> (Vec<u32>, usize, usize) {
    let (native_pixels, nw, nh) = detect_and_collapse(pixels, width, height);
    let (px, w, h) = if !native_pixels.is_empty() {
        (native_pixels.as_slice(), nw, nh)
    } else {
        (pixels, width, height)
    };
    let graph = graph::build(px, w, h);
    let (paths, bg_color) = contour::extract_shared_edge_paths(px, &graph);
    rasterize::rasterize_scaled(&paths, w, h, bg_color, scale as f64)
}

/// Core vectorization: graph → contour → paths. No upscale collapse.
pub fn vectorize_core(
    pixels: &[u32], width: usize, height: usize,
) -> (Vec<contour::ColorPath>, u32) {
    vectorize_core_inner(pixels, width, height, false)
}

/// Core vectorization pipeline.
/// When `adaptive` is true, allows the contour extractor to skip B-spline
/// optimization on complex frames (boundary edges > threshold).
fn vectorize_core_inner(
    pixels: &[u32], width: usize, height: usize, adaptive: bool,
) -> (Vec<contour::ColorPath>, u32) {
    let verbose = std::env::var("VECTORIZE_BENCH").is_ok();
    let t0 = std::time::Instant::now();

    let graph = graph::build(pixels, width, height);
    let t1 = std::time::Instant::now();

    let paths = contour::extract_cells_smooth(pixels, &graph, adaptive);
    let t2 = std::time::Instant::now();

    let (bg_color, _) = detect_background_color(pixels, width, height);
    let t3 = std::time::Instant::now();

    if verbose {
        eprintln!("  graph build:     {:>8.3}ms", (t1 - t0).as_secs_f64() * 1000.0);
        eprintln!("  contour extract: {:>8.3}ms", (t2 - t1).as_secs_f64() * 1000.0);
        eprintln!("  bg detect:       {:>8.3}ms", (t3 - t2).as_secs_f64() * 1000.0);
        eprintln!("  total pipeline:  {:>8.3}ms", (t3 - t0).as_secs_f64() * 1000.0);
    }

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
