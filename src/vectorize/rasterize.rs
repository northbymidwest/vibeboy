//! Rasterize vector paths (ColorPath) to a pixel buffer using scanline rendering
//! with 2x2 supersampling for anti-aliased edges.

use super::contour::{ColorPath, PathSegment};

/// A line segment in output pixel space with precomputed fields.
struct Edge {
    x0: f64,
    y0: f64,
    dx_per_dy: f64,
    y_min: f64,
    y_max: f64,
    dir: i32,
}

impl Edge {
    #[inline(always)]
    fn new(x0: f64, y0: f64, x1: f64, y1: f64) -> Self {
        let dy = y1 - y0;
        Edge {
            x0, y0,
            dx_per_dy: (x1 - x0) / dy,
            y_min: y0.min(y1),
            y_max: y0.max(y1),
            dir: if dy > 0.0 { 1 } else { -1 },
        }
    }

    /// X intersection at scanline y. Precomputed dx/dy reduces to 1 sub + 1 fma.
    #[inline(always)]
    fn intersect_x(&self, sy: f64) -> f64 {
        self.x0 + (sy - self.y0) * self.dx_per_dy
    }
}

/// Flatten a quadratic Bezier into line edges via recursive subdivision.
#[inline(always)]
fn flatten_quad(
    x0: f64, y0: f64, cx: f64, cy: f64, x1: f64, y1: f64,
    tol_sq: f64, edges: &mut Vec<Edge>,
) {
    let mx = (x0 + x1) * 0.5;
    let my = (y0 + y1) * 0.5;
    let dx = cx - mx;
    let dy = cy - my;
    if dx * dx + dy * dy <= tol_sq {
        if (y0 - y1).abs() > 1e-10 {
            edges.push(Edge::new(x0, y0, x1, y1));
        }
        return;
    }
    let mx01 = (x0 + cx) * 0.5;
    let my01 = (y0 + cy) * 0.5;
    let mx12 = (cx + x1) * 0.5;
    let my12 = (cy + y1) * 0.5;
    let midx = (mx01 + mx12) * 0.5;
    let midy = (my01 + my12) * 0.5;
    flatten_quad(x0, y0, mx01, my01, midx, midy, tol_sq, edges);
    flatten_quad(midx, midy, mx12, my12, x1, y1, tol_sq, edges);
}

/// Extract edges from path segments, scaled to output space.
fn extract_edges(segments: &[PathSegment], sx: f64, sy: f64, tol_sq: f64, edges: &mut Vec<Edge>) {
    edges.clear();
    for seg in segments {
        match seg {
            PathSegment::Line(a, b) => {
                let y0 = a.y * sy;
                let y1 = b.y * sy;
                if (y0 - y1).abs() > 1e-10 {
                    edges.push(Edge::new(a.x * sx, y0, b.x * sx, y1));
                }
            }
            PathSegment::QuadBezier(start, ctrl, end) => {
                flatten_quad(
                    start.x * sx, start.y * sy,
                    ctrl.x * sx, ctrl.y * sy,
                    end.x * sx, end.y * sy,
                    tol_sq, edges,
                );
            }
        }
    }
}

/// Rasterize vector paths to an ARGB pixel buffer.
/// Returns Vec<u32> of size (width*scale) * (height*scale).
pub fn rasterize(
    paths: &[ColorPath],
    width: usize,
    height: usize,
    bg_color: u32,
    scale: usize,
) -> Vec<u32> {
    let (buf, _, _) = rasterize_scaled(paths, width, height, bg_color, scale as f64);
    buf
}

/// Rasterize vector paths at a floating-point scale factor.
/// Output dimensions are `(width * scale).round()` x `(height * scale).round()`.
pub fn rasterize_scaled(
    paths: &[ColorPath],
    width: usize,
    height: usize,
    bg_color: u32,
    scale: f64,
) -> (Vec<u32>, usize, usize) {
    let out_w = (width as f64 * scale).round() as usize;
    let out_h = (height as f64 * scale).round() as usize;
    let mut buffer = vec![bg_color; out_w * out_h];
    let sx = scale;
    let sy = scale;
    let tol_sq = 0.25;

    let mut edges = Vec::new();
    let mut sorted: Vec<usize> = Vec::new();
    let mut coverage = vec![0u8; out_w];

    // Bucket sort: O(n) distribution into per-row buckets, then flatten.
    // Replaces O(n log n) comparison sort for edge y_min ordering.
    let mut bucket_heads: Vec<u32> = Vec::new();
    let mut bucket_next: Vec<u32> = Vec::new();

    for path in paths {
        if path.segments.is_empty() {
            continue;
        }
        if path.color == bg_color {
            continue;
        }

        extract_edges(&path.segments, sx, sy, tol_sq, &mut edges);
        if edges.is_empty() {
            continue;
        }
        // Linked-list bucket sort: O(n) with zero per-bucket allocation.
        bucket_heads.clear();
        bucket_heads.resize(out_h, u32::MAX);
        bucket_next.clear();
        bucket_next.resize(edges.len(), u32::MAX);
        for (i, e) in edges.iter().enumerate() {
            let row = (e.y_min.floor() as usize).min(out_h - 1);
            bucket_next[i] = bucket_heads[row];
            bucket_heads[row] = i as u32;
        }
        sorted.clear();
        sorted.reserve(edges.len());
        for row in 0..out_h {
            let mut idx = bucket_heads[row];
            while idx != u32::MAX {
                sorted.push(idx as usize);
                idx = bucket_next[idx as usize];
            }
        }

        coverage[..out_w].fill(0);

        rasterize_path(
            &edges,
            &sorted,
            path.color,
            &mut buffer,
            out_w,
            out_h,
            &mut coverage,
        );
    }

    (buffer, out_w, out_h)
}

/// Rasterize a single path's edges with 2x2 supersampling and nonzero winding.
fn rasterize_path(
    edges: &[Edge],
    sorted: &[usize],
    fill_color: u32,
    buffer: &mut [u32],
    out_w: usize,
    out_h: usize,
    coverage: &mut [u8],
) {
    let mut dirty_min = out_w;
    let mut dirty_max = 0usize;

    // Y bounding box
    let mut y_min_f = f64::MAX;
    let mut y_max_f = f64::MIN;
    for e in edges {
        if e.y_min < y_min_f { y_min_f = e.y_min; }
        if e.y_max > y_max_f { y_max_f = e.y_max; }
    }
    let py_start = (y_min_f.floor() as usize).min(out_h);
    let py_end = (y_max_f.ceil() as usize).min(out_h);

    let mut isects: [Vec<(f64, i32)>; 2] = [Vec::new(), Vec::new()];
    let mut scan_start = 0usize;

    for py in py_start..py_end {
        let y_top = py as f64;
        let y_bot = y_top + 1.0;

        // Reset dirty coverage
        if dirty_min <= dirty_max {
            for c in &mut coverage[dirty_min..=dirty_max.min(out_w - 1)] {
                *c = 0;
            }
            dirty_min = out_w;
            dirty_max = 0;
        }

        // Advance past edges above this row
        while scan_start < sorted.len() {
            if edges[sorted[scan_start]].y_max <= y_top {
                scan_start += 1;
            } else {
                break;
            }
        }

        // Collect intersections for both sub-scanlines
        for buf in isects.iter_mut() { buf.clear(); }

        for i in scan_start..sorted.len() {
            let e = &edges[sorted[i]];
            if e.y_min >= y_bot { break; }

            let sy0 = y_top + 0.25;
            let sy1 = y_top + 0.75;
            if sy0 >= e.y_min && sy0 < e.y_max {
                isects[0].push((e.intersect_x(sy0), e.dir));
            }
            if sy1 >= e.y_min && sy1 < e.y_max {
                isects[1].push((e.intersect_x(sy1), e.dir));
            }
        }

        // Process each sub-scanline with nonzero winding rule
        for si in 0..2 {
            let isect = &mut isects[si];
            if isect.is_empty() { continue; }

            isect.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

            let mut winding = 0i32;
            let mut i = 0;
            while i < isect.len() {
                winding += isect[i].1;

                if winding != 0 {
                    let x_enter = isect[i].0;
                    let mut j = i + 1;
                    while j < isect.len() {
                        winding += isect[j].1;
                        if winding == 0 { break; }
                        j += 1;
                    }

                    let x_exit = if j < isect.len() {
                        isect[j].0
                    } else {
                        break;
                    };

                    let px_start = (x_enter.max(0.0) as usize).min(out_w);
                    let px_end = ((x_exit.ceil() as usize).min(out_w)).max(px_start);

                    if px_start < dirty_min { dirty_min = px_start; }
                    if px_end > 0 && px_end - 1 > dirty_max { dirty_max = px_end - 1; }

                    // Fixed-point coverage: multiply x coords by 256 for integer math.
                    // Sample points at px+0.25 and px+0.75 become 256*px+64 and 256*px+192.
                    let enter_fp = (x_enter * 256.0).round() as i64;
                    let exit_fp = (x_exit * 256.0).round() as i64;

                    for px in px_start..px_end {
                        let base = (px as i64) * 256;
                        // Sample at px + 0.25 (base + 64)
                        if base + 64 >= enter_fp && base + 64 < exit_fp { coverage[px] += 1; }
                        // Sample at px + 0.75 (base + 192)
                        if base + 192 >= enter_fp && base + 192 < exit_fp { coverage[px] += 1; }
                    }

                    i = j + 1;
                } else {
                    i += 1;
                }
            }
        }

        // Write pixels — batch-fill contiguous full-coverage runs
        if dirty_min <= dirty_max {
            let row_start = py * out_w;
            let end = dirty_max.min(out_w - 1);
            let mut px = dirty_min;
            while px <= end {
                let cov = coverage[px];
                if cov == 0 { px += 1; continue; }
                if cov >= 4 {
                    // Find contiguous run of full coverage for batch fill
                    let run_start = px;
                    while px <= end && coverage[px] >= 4 {
                        px += 1;
                    }
                    buffer[row_start + run_start..row_start + px].fill(fill_color);
                } else {
                    buffer[row_start + px] = blend4(buffer[row_start + px], fill_color, cov);
                    px += 1;
                }
            }
        }
    }
}

/// GPU-ready edge data for compute shader upload.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct GpuEdge {
    pub x0: f32,
    pub y0: f32,
    pub dx_per_dy: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub dir: i32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Per-path metadata for GPU compute shader.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct GpuPathMeta {
    pub color: u32,
    pub edge_start: u32,
    pub edge_count: u32,
    pub _pad: u32,
}

/// Flatten all paths into GPU-ready edge and path metadata arrays.
/// Skips background-colored paths. Returns (edges, path_metas).
pub fn prepare_gpu_edges(
    paths: &[ColorPath], bg_color: u32, scale: f64,
) -> (Vec<GpuEdge>, Vec<GpuPathMeta>) {
    let sx = scale;
    let sy = scale;
    let tol_sq = 0.25;
    let mut cpu_edges = Vec::new();
    let mut gpu_edges = Vec::new();
    let mut metas = Vec::new();

    for path in paths {
        if path.segments.is_empty() || path.color == bg_color {
            continue;
        }
        extract_edges(&path.segments, sx, sy, tol_sq, &mut cpu_edges);
        if cpu_edges.is_empty() {
            continue;
        }
        let start = gpu_edges.len() as u32;
        for e in &cpu_edges {
            gpu_edges.push(GpuEdge {
                x0: e.x0 as f32,
                y0: e.y0 as f32,
                dx_per_dy: e.dx_per_dy as f32,
                y_min: e.y_min as f32,
                y_max: e.y_max as f32,
                dir: e.dir,
                _pad0: 0,
                _pad1: 0,
            });
        }
        metas.push(GpuPathMeta {
            color: path.color,
            edge_start: start,
            edge_count: gpu_edges.len() as u32 - start,
            _pad: 0,
        });
    }
    (gpu_edges, metas)
}

/// GPU-ready edge with embedded path color, for row-indexed compute shader.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct GpuEdgeV2 {
    pub x0: f32,
    pub y0: f32,
    pub dx_per_dy: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub dir: i32,
    pub color: u32,
    pub _pad: u32,
}

/// Per-row range into the sorted edge index array.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct GpuRowRange {
    pub start: u32,
    pub count: u32,
}

/// Build row-indexed GPU edge data for the compute rasterizer.
/// Each edge carries its path color. Edges are bucketed into per-row index
/// arrays so the GPU only tests edges overlapping each pixel's scanline.
/// Returns (edges, row_ranges, edge_indices, out_w, out_h).
pub fn prepare_gpu_edges_v2(
    paths: &[ColorPath], bg_color: u32, scale: f64,
    src_w: usize, src_h: usize,
) -> (Vec<GpuEdgeV2>, Vec<GpuRowRange>, Vec<u32>, u32, u32) {
    let sx = scale;
    let sy = scale;
    let tol_sq = 0.25;
    let out_w = (src_w as f64 * scale).round() as u32;
    let out_h = (src_h as f64 * scale).round() as u32;
    let mut cpu_edges = Vec::new();
    let mut all_edges = Vec::new();

    for path in paths {
        if path.segments.is_empty() || path.color == bg_color {
            continue;
        }
        extract_edges(&path.segments, sx, sy, tol_sq, &mut cpu_edges);
        for e in &cpu_edges {
            all_edges.push(GpuEdgeV2 {
                x0: e.x0 as f32,
                y0: e.y0 as f32,
                dx_per_dy: e.dx_per_dy as f32,
                y_min: e.y_min as f32,
                y_max: e.y_max as f32,
                dir: e.dir,
                color: path.color,
                _pad: 0,
            });
        }
    }

    // Build per-row index: for each row, collect indices of edges whose
    // y_min..y_max range overlaps [row, row+1).
    let num_rows = out_h as usize;
    let mut row_buckets: Vec<Vec<u32>> = vec![Vec::new(); num_rows];
    for (i, e) in all_edges.iter().enumerate() {
        let row_start = (e.y_min.floor() as usize).min(num_rows.saturating_sub(1));
        let row_end = (e.y_max.ceil() as usize).min(num_rows);
        for row in row_start..row_end {
            row_buckets[row].push(i as u32);
        }
    }

    // Flatten buckets into a contiguous index array + per-row ranges
    let mut edge_indices = Vec::new();
    let mut row_ranges = Vec::with_capacity(num_rows);
    for bucket in &row_buckets {
        let start = edge_indices.len() as u32;
        edge_indices.extend_from_slice(bucket);
        row_ranges.push(GpuRowRange {
            start,
            count: bucket.len() as u32,
        });
    }

    (all_edges, row_ranges, edge_indices, out_w, out_h)
}

/// Blend two ARGB colors with 2x2 coverage (0..4).
#[inline(always)]
fn blend4(bg: u32, fg: u32, coverage: u8) -> u32 {
    let alpha = coverage as u32;
    let inv = 4 - alpha;

    let bg_r = (bg >> 16) & 0xFF;
    let bg_g = (bg >> 8) & 0xFF;
    let bg_b = bg & 0xFF;

    let fg_r = (fg >> 16) & 0xFF;
    let fg_g = (fg >> 8) & 0xFF;
    let fg_b = fg & 0xFF;

    let r = (bg_r * inv + fg_r * alpha) >> 2;
    let g = (bg_g * inv + fg_g * alpha) >> 2;
    let b = (bg_b * inv + fg_b * alpha) >> 2;

    0xFF000000 | (r << 16) | (g << 8) | b
}

// --- Gaussian diffusion rasterizer (Paper Section 3.5) ---
//
// "We place truncated Gaussian influence functions (σ = 1, radius 2 pixels)
// at the cell centroids and set their support to zero outside the region
// visible from the cell centroid. The final color at a point is computed as
// the weighted average of all pixel colors according to their respective
// influence."
//
// Region boundaries follow the smooth Voronoi cell contours (not the pixel
// grid). Each cell polygon is scanline-filled to an ownership map at the
// output resolution, giving smooth diagonal boundaries at deformed corners.

use super::graph::SimilarityGraph;

/// Rasterize using Gaussian color diffusion (Paper Section 3.5).
///
/// Each pixel centroid emits its color with a truncated Gaussian (σ=1, r=2).
/// Color propagation is blocked by contour lines (visible edges between
/// Voronoi cells). Region boundaries follow the smooth cell geometry, not
/// the pixel grid.
pub fn rasterize_diffusion(
    pixels: &[u32],
    width: usize,
    height: usize,
    scale: usize,
) -> (Vec<u32>, usize, usize) {
    let out_w = width * scale;
    let out_h = height * scale;

    // Build resolved similarity graph (with crossing resolution)
    let graph = super::graph::build(pixels, width, height);

    // Flood-fill source pixels along graph edges to find contour-bounded regions.
    let src_regions = build_graph_regions(width, height, &graph);

    // Scanline-fill each Voronoi cell polygon at output resolution to build
    // a smooth ownership map. Each output pixel is assigned to the source
    // pixel whose deformed Voronoi cell contains it.
    let ownership = build_voronoi_ownership(width, height, &graph, scale);

    let inv_scale = 1.0 / scale as f64;
    let sigma_sq_2 = 2.0; // 2 * σ² with σ = 1
    let radius = 2.0f64;
    let r_sq = radius * radius;

    let mut buffer = vec![0u32; out_w * out_h];

    for oy in 0..out_h {
        let sy = (oy as f64 + 0.5) * inv_scale;
        let min_py = ((sy - radius).floor() as i32).max(0) as usize;
        let max_py = ((sy + radius).ceil() as i32).min(height as i32 - 1) as usize;

        for ox in 0..out_w {
            let sx = (ox as f64 + 0.5) * inv_scale;

            // Region of this output pixel via Voronoi ownership
            let owner = ownership[oy * out_w + ox] as usize;
            let my_region = src_regions[owner];

            let min_px = ((sx - radius).floor() as i32).max(0) as usize;
            let max_px = ((sx + radius).ceil() as i32).min(width as i32 - 1) as usize;

            let mut tr = 0.0f64;
            let mut tg = 0.0f64;
            let mut tb = 0.0f64;
            let mut tw = 0.0f64;

            for py in min_py..=max_py {
                for px in min_px..=max_px {
                    if src_regions[py * width + px] != my_region { continue; }

                    let dx = sx - (px as f64 + 0.5);
                    let dy = sy - (py as f64 + 0.5);
                    let d_sq = dx * dx + dy * dy;
                    if d_sq > r_sq { continue; }

                    let w = (-d_sq / sigma_sq_2).exp();
                    let color = pixels[py * width + px];
                    tr += w * ((color >> 16) & 0xFF) as f64;
                    tg += w * ((color >> 8) & 0xFF) as f64;
                    tb += w * (color & 0xFF) as f64;
                    tw += w;
                }
            }

            if tw > 0.0 {
                let inv_tw = 1.0 / tw;
                let r = (tr * inv_tw).round().min(255.0) as u32;
                let g = (tg * inv_tw).round().min(255.0) as u32;
                let b = (tb * inv_tw).round().min(255.0) as u32;
                buffer[oy * out_w + ox] = (r << 16) | (g << 8) | b;
            }
        }
    }

    (buffer, out_w, out_h)
}

/// Build region labels by flood-filling along resolved similarity graph edges.
/// Two pixels are in the same region if connected (directly or transitively)
/// through graph edges — i.e., no visible edge (contour line) separates them.
pub fn build_graph_regions(w: usize, h: usize, graph: &SimilarityGraph) -> Vec<u32> {
    let mut regions = vec![u32::MAX; w * h];
    let mut region_id = 0u32;

    for start_y in 0..h {
        for start_x in 0..w {
            if regions[start_y * w + start_x] != u32::MAX { continue; }

            regions[start_y * w + start_x] = region_id;
            let mut stack = vec![(start_x, start_y)];

            while let Some((cx, cy)) = stack.pop() {
                let e = graph.edge(cx, cy);

                // Right
                if e.right && cx + 1 < w && regions[cy * w + cx + 1] == u32::MAX {
                    regions[cy * w + cx + 1] = region_id;
                    stack.push((cx + 1, cy));
                }
                // Down
                if e.down && cy + 1 < h && regions[(cy + 1) * w + cx] == u32::MAX {
                    regions[(cy + 1) * w + cx] = region_id;
                    stack.push((cx, cy + 1));
                }
                // Down-right
                if e.down_right && cx + 1 < w && cy + 1 < h
                    && regions[(cy + 1) * w + cx + 1] == u32::MAX
                {
                    regions[(cy + 1) * w + cx + 1] = region_id;
                    stack.push((cx + 1, cy + 1));
                }
                // Down-left
                if e.down_left && cx > 0 && cy + 1 < h
                    && regions[(cy + 1) * w + cx - 1] == u32::MAX
                {
                    regions[(cy + 1) * w + cx - 1] = region_id;
                    stack.push((cx - 1, cy + 1));
                }
                // Left (reverse of neighbor's right)
                if cx > 0 && graph.edge(cx - 1, cy).right
                    && regions[cy * w + cx - 1] == u32::MAX
                {
                    regions[cy * w + cx - 1] = region_id;
                    stack.push((cx - 1, cy));
                }
                // Up (reverse of neighbor's down)
                if cy > 0 && graph.edge(cx, cy - 1).down
                    && regions[(cy - 1) * w + cx] == u32::MAX
                {
                    regions[(cy - 1) * w + cx] = region_id;
                    stack.push((cx, cy - 1));
                }
                // Up-right (reverse of neighbor's down-left)
                if cx + 1 < w && cy > 0 && graph.edge(cx + 1, cy - 1).down_left
                    && regions[(cy - 1) * w + cx + 1] == u32::MAX
                {
                    regions[(cy - 1) * w + cx + 1] = region_id;
                    stack.push((cx + 1, cy - 1));
                }
                // Up-left (reverse of neighbor's down-right)
                if cx > 0 && cy > 0 && graph.edge(cx - 1, cy - 1).down_right
                    && regions[(cy - 1) * w + cx - 1] == u32::MAX
                {
                    regions[(cy - 1) * w + cx - 1] = region_id;
                    stack.push((cx - 1, cy - 1));
                }
            }

            region_id += 1;
        }
    }

    regions
}

/// Get diagonal state at grid corner (cx, cy): 0=none, 1=backslash, 2=slash.
#[inline(always)]
fn corner_diag(graph: &SimilarityGraph, cx: usize, cy: usize) -> u8 {
    let w = graph.width;
    let h = graph.height;
    if cx == 0 || cy == 0 || cx >= w || cy >= h { return 0; }
    if graph.edge(cx - 1, cy - 1).down_right { return 1; }
    if graph.edge(cx, cy - 1).down_left { return 2; }
    0
}

/// Compute Voronoi cell vertices for pixel (px, py) in source-space coordinates.
/// Returns up to 8 vertices in CW order.
fn cell_vertices_f64(px: usize, py: usize, graph: &SimilarityGraph) -> [(f64, f64); 8] {
    let mut verts = [(0.0, 0.0); 8];
    let mut n = 0usize;
    let bx = px as f64;
    let by = py as f64;

    // TL corner (rel=BR)
    match corner_diag(graph, px, py) {
        1 => { verts[n] = (bx - 0.25, by + 0.25); n += 1;
               verts[n] = (bx + 0.25, by - 0.25); n += 1; }
        2 => { verts[n] = (bx + 0.25, by + 0.25); n += 1; }
        _ => { verts[n] = (bx, by); n += 1; }
    }
    // TR corner (rel=BL)
    match corner_diag(graph, px + 1, py) {
        1 => { verts[n] = (bx + 0.75, by + 0.25); n += 1; }
        2 => { verts[n] = (bx + 0.75, by - 0.25); n += 1;
               verts[n] = (bx + 1.25, by + 0.25); n += 1; }
        _ => { verts[n] = (bx + 1.0, by); n += 1; }
    }
    // BR corner (rel=TL)
    match corner_diag(graph, px + 1, py + 1) {
        1 => { verts[n] = (bx + 1.25, by + 0.75); n += 1;
               verts[n] = (bx + 0.75, by + 1.25); n += 1; }
        2 => { verts[n] = (bx + 0.75, by + 0.75); n += 1; }
        _ => { verts[n] = (bx + 1.0, by + 1.0); n += 1; }
    }
    // BL corner (rel=TR)
    match corner_diag(graph, px, py + 1) {
        1 => { verts[n] = (bx + 0.25, by + 0.75); n += 1; }
        2 => { verts[n] = (bx + 0.25, by + 1.25); n += 1;
               verts[n] = (bx - 0.25, by + 0.75); n += 1; }
        _ => { verts[n] = (bx, by + 1.0); n += 1; }
    }

    // Store count in unused slots (we know n <= 8)
    // Return full array; caller uses cell_vertex_count to get n
    verts
}

/// Count of vertices for a cell (same logic as cell_vertices_f64).
#[inline]
fn cell_vertex_count(px: usize, py: usize, graph: &SimilarityGraph) -> usize {
    let mut n = 0;
    for &(cx, cy) in &[(px, py), (px + 1, py), (px + 1, py + 1), (px, py + 1)] {
        let d = corner_diag(graph, cx, cy);
        n += if d == 1 || d == 2 { if (cx == px && cy == py && d == 1)
            || (cx == px + 1 && cy == py && d == 2)
            || (cx == px + 1 && cy == py + 1 && d == 1)
            || (cx == px && cy == py + 1 && d == 2) { 2 } else { 1 }
        } else { 1 };
    }
    n
}

/// Build Voronoi ownership map at output resolution by scanline-filling
/// each cell polygon. Each output pixel is assigned to the source pixel
/// whose deformed Voronoi cell contains it.
/// Check if point (px, py) is inside convex polygon with `nv` vertices.
/// Uses cross-product winding test.
#[inline]
fn point_in_convex_poly(verts: &[(f64, f64); 8], nv: usize, px: f64, py: f64) -> bool {
    if nv < 3 { return false; }
    let mut sign = 0i32;
    for i in 0..nv {
        let (x0, y0) = verts[i];
        let (x1, y1) = verts[(i + 1) % nv];
        let cross = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0);
        let s = if cross > 1e-10 { 1 } else if cross < -1e-10 { -1 } else { 0 };
        if s != 0 {
            if sign == 0 { sign = s; }
            else if sign != s { return false; }
        }
    }
    true
}

/// Count vertices for a cell (matches cell_vertices_f64 output length).
#[inline]
fn cell_nv(graph: &SimilarityGraph, px: usize, py: usize) -> usize {
    let corners = [
        (corner_diag(graph, px, py), true),       // TL: 2 verts if backslash
        (corner_diag(graph, px + 1, py), false),   // TR: 2 verts if slash
        (corner_diag(graph, px + 1, py + 1), true),// BR: 2 verts if backslash
        (corner_diag(graph, px, py + 1), false),   // BL: 2 verts if slash
    ];
    corners.iter().map(|&(d, is_bs_double)| {
        if d == 1 && is_bs_double { 2 }
        else if d == 2 && !is_bs_double { 2 }
        else { 1 }
    }).sum()
}

pub fn build_voronoi_ownership(
    w: usize, h: usize, graph: &SimilarityGraph, scale: usize,
) -> Vec<u32> {
    let out_w = w * scale;
    let out_h = h * scale;
    let inv_scale = 1.0 / scale as f64;
    let mut ownership = vec![0u32; out_w * out_h];

    for oy in 0..out_h {
        let sy = (oy as f64 + 0.5) * inv_scale;
        for ox in 0..out_w {
            let sx = (ox as f64 + 0.5) * inv_scale;

            // Home source pixel (default owner)
            let home_x = (sx.floor() as usize).min(w - 1);
            let home_y = (sy.floor() as usize).min(h - 1);
            let home_id = (home_y * w + home_x) as u32;

            // For most pixels (interior of cell), home is correct.
            // Only pixels near diagonal corners need refinement.
            // Check if any of the 4 corners of the home cell have diagonals.
            let tl = corner_diag(graph, home_x, home_y);
            let tr = corner_diag(graph, home_x + 1, home_y);
            let br = corner_diag(graph, home_x + 1, home_y + 1);
            let bl = corner_diag(graph, home_x, home_y + 1);

            if tl == 0 && tr == 0 && br == 0 && bl == 0 {
                // No diagonals — cell is a perfect square, home is correct
                ownership[oy * out_w + ox] = home_id;
                continue;
            }

            // Diagonal exists at some corner — test against home cell and neighbors
            let home_verts = cell_vertices_f64(home_x, home_y, graph);
            let home_nv = cell_nv(graph, home_x, home_y);

            if point_in_convex_poly(&home_verts, home_nv, sx, sy) {
                ownership[oy * out_w + ox] = home_id;
                continue;
            }

            // Point is outside home cell due to diagonal deformation.
            // Check the 8 neighbors for the correct owner.
            let mut found = false;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 { continue; }
                    let nx = home_x as i32 + dx;
                    let ny = home_y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 { continue; }
                    let (nx, ny) = (nx as usize, ny as usize);
                    let verts = cell_vertices_f64(nx, ny, graph);
                    let nv = cell_nv(graph, nx, ny);
                    if point_in_convex_poly(&verts, nv, sx, sy) {
                        ownership[oy * out_w + ox] = (ny * w + nx) as u32;
                        found = true;
                        break;
                    }
                }
                if found { break; }
            }

            // Fallback (shouldn't happen with correct cells)
            if !found {
                ownership[oy * out_w + ox] = home_id;
            }
        }
    }

    ownership
}

/// Rasterize vector paths with NO anti-aliasing (single center-point sample).
/// Produces hard region boundaries suitable for region-map usage in the
/// spline-diffusion pipeline. Blended colors would break region matching.
fn rasterize_noaa(
    paths: &[ColorPath],
    width: usize,
    height: usize,
    bg_color: u32,
    scale: usize,
) -> Vec<u32> {
    let out_w = width * scale;
    let out_h = height * scale;
    let mut buffer = vec![bg_color; out_w * out_h];
    let sx = scale as f64;
    let sy = scale as f64;
    let tol_sq = 0.25;

    let mut edges = Vec::new();

    for path in paths {
        if path.segments.is_empty() || path.color == bg_color { continue; }

        extract_edges(&path.segments, sx, sy, tol_sq, &mut edges);
        if edges.is_empty() { continue; }

        // Sort edges by y_min for scanline traversal
        edges.sort_unstable_by(|a, b| a.y_min.total_cmp(&b.y_min));

        let fill_color = path.color;
        let mut scan_start = 0usize;

        for py in 0..out_h {
            let scan_y = py as f64 + 0.5;

            while scan_start < edges.len() && edges[scan_start].y_max <= scan_y {
                scan_start += 1;
            }

            // Collect x intersections with nonzero winding
            let mut isects: Vec<(f64, i32)> = Vec::new();
            for i in scan_start..edges.len() {
                let e = &edges[i];
                if e.y_min >= scan_y + 1.0 { break; }
                if scan_y >= e.y_min && scan_y < e.y_max {
                    isects.push((e.intersect_x(scan_y), e.dir));
                }
            }
            if isects.is_empty() { continue; }

            isects.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

            let mut winding = 0i32;
            let mut i = 0;
            while i < isects.len() {
                winding += isects[i].1;
                if winding != 0 {
                    let x_enter = isects[i].0;
                    let mut j = i + 1;
                    while j < isects.len() {
                        winding += isects[j].1;
                        if winding == 0 { break; }
                        j += 1;
                    }
                    let x_exit = if j < isects.len() { isects[j].0 } else { break };

                    let px_start = (x_enter.ceil() as usize).min(out_w);
                    let px_end = (x_exit.floor() as usize + 1).min(out_w);
                    for px in px_start..px_end {
                        let cx = px as f64 + 0.5;
                        if cx >= x_enter && cx < x_exit {
                            buffer[py * out_w + px] = fill_color;
                        }
                    }
                    i = j + 1;
                } else {
                    i += 1;
                }
            }
        }
    }

    buffer
}

/// Flood-fill connected components on a color buffer at output resolution.
/// Two 4-connected output pixels with the same color are in the same region.
fn flood_fill_output_regions(colors: &[u32], w: usize, h: usize) -> Vec<u32> {
    let mut ids = vec![u32::MAX; w * h];
    let mut region_id = 0u32;

    for start in 0..w * h {
        if ids[start] != u32::MAX { continue; }
        let color = colors[start];
        ids[start] = region_id;
        let mut stack = vec![start];

        while let Some(idx) = stack.pop() {
            let x = idx % w;
            let y = idx / w;
            for &(dx, dy) in &[(1i32, 0), (-1, 0), (0, 1), (0, -1)] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 { continue; }
                let ni = ny as usize * w + nx as usize;
                if ids[ni] != u32::MAX { continue; }
                if colors[ni] == color {
                    ids[ni] = region_id;
                    stack.push(ni);
                }
            }
        }

        region_id += 1;
    }

    ids
}

/// Flatten all B-spline contour paths into line segments at source resolution.
/// Returns (x0, y0, x1, y1) segments in source-pixel coordinates.
fn flatten_contour_segments(paths: &[ColorPath]) -> Vec<[f64; 4]> {
    let mut segs = Vec::new();
    let tol_sq = 0.01; // tighter tolerance at source resolution

    for path in paths {
        for seg in &path.segments {
            match seg {
                PathSegment::Line(a, b) => {
                    let dx = b.x - a.x;
                    let dy = b.y - a.y;
                    if dx * dx + dy * dy > 1e-12 {
                        segs.push([a.x, a.y, b.x, b.y]);
                    }
                }
                PathSegment::QuadBezier(start, ctrl, end) => {
                    flatten_quad_to_segs(
                        start.x, start.y, ctrl.x, ctrl.y, end.x, end.y,
                        tol_sq, &mut segs,
                    );
                }
            }
        }
    }
    segs
}

/// Recursively flatten a quadratic Bezier into line segments.
fn flatten_quad_to_segs(
    x0: f64, y0: f64, cx: f64, cy: f64, x1: f64, y1: f64,
    tol_sq: f64, out: &mut Vec<[f64; 4]>,
) {
    let mx = (x0 + x1) * 0.5;
    let my = (y0 + y1) * 0.5;
    let dx = cx - mx;
    let dy = cy - my;
    if dx * dx + dy * dy <= tol_sq {
        let d = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0);
        if d > 1e-12 {
            out.push([x0, y0, x1, y1]);
        }
        return;
    }
    let mx01 = (x0 + cx) * 0.5;
    let my01 = (y0 + cy) * 0.5;
    let mx12 = (cx + x1) * 0.5;
    let my12 = (cy + y1) * 0.5;
    let midx = (mx01 + mx12) * 0.5;
    let midy = (my01 + my12) * 0.5;
    flatten_quad_to_segs(x0, y0, mx01, my01, midx, midy, tol_sq, out);
    flatten_quad_to_segs(midx, midy, mx12, my12, x1, y1, tol_sq, out);
}

/// Spatial grid: for each source pixel cell, store indices of contour segments
/// that overlap or pass near it (within 0.5 pixel margin).
fn build_segment_grid(segs: &[[f64; 4]], w: usize, h: usize) -> Vec<Vec<u32>> {
    let mut grid = vec![Vec::new(); w * h];
    for (i, s) in segs.iter().enumerate() {
        let x_min = s[0].min(s[2]) - 0.5;
        let x_max = s[0].max(s[2]) + 0.5;
        let y_min = s[1].min(s[3]) - 0.5;
        let y_max = s[1].max(s[3]) + 0.5;
        let gx0 = (x_min.floor() as i32).max(0) as usize;
        let gx1 = (x_max.ceil() as i32).min(w as i32) as usize;
        let gy0 = (y_min.floor() as i32).max(0) as usize;
        let gy1 = (y_max.ceil() as i32).min(h as i32) as usize;
        for gy in gy0..gy1 {
            for gx in gx0..gx1 {
                grid[gy * w + gx].push(i as u32);
            }
        }
    }
    grid
}

/// Count how many contour segments the line from (ax, ay) to (bx, by) crosses.
/// Odd count = blocked (different side of contour). Even = visible (same side).
/// Uses the spatial grid to test only nearby segments.
fn ray_crossing_count(
    ax: f64, ay: f64, bx: f64, by: f64,
    segs: &[[f64; 4]], grid: &[Vec<u32>], grid_w: usize,
) -> u32 {
    let x_min = ax.min(bx) - 0.1;
    let x_max = ax.max(bx) + 0.1;
    let y_min = ay.min(by) - 0.1;
    let y_max = ay.max(by) + 0.1;
    let gx0 = (x_min.floor() as usize).min(grid_w.saturating_sub(1));
    let gx1 = (x_max.ceil() as usize).min(grid_w);
    let gy0 = y_min.floor() as usize;
    let gy1 = y_max.ceil() as usize;
    let grid_h = grid.len() / grid_w;
    let gy1 = gy1.min(grid_h);

    let mut tested = [u32::MAX; 128];
    let mut n_tested = 0usize;
    let mut crossings = 0u32;

    for gy in gy0..gy1 {
        for gx in gx0..gx1 {
            for &si in &grid[gy * grid_w + gx] {
                if n_tested < 128 && tested[..n_tested].contains(&si) { continue; }
                if n_tested < 128 { tested[n_tested] = si; n_tested += 1; }

                let s = &segs[si as usize];
                if segments_intersect(ax, ay, bx, by, s[0], s[1], s[2], s[3]) {
                    crossings += 1;
                }
            }
        }
    }
    crossings
}

/// Test if two line segments intersect (strict interior intersection only).
#[inline]
fn segments_intersect(
    p1x: f64, p1y: f64, p2x: f64, p2y: f64,
    p3x: f64, p3y: f64, p4x: f64, p4y: f64,
) -> bool {
    let d1x = p2x - p1x;
    let d1y = p2y - p1y;
    let d2x = p4x - p3x;
    let d2y = p4y - p3y;

    let cross = d1x * d2y - d1y * d2x;
    if cross.abs() < 1e-12 { return false; }

    let inv_cross = 1.0 / cross;
    let dx = p3x - p1x;
    let dy = p3y - p1y;
    let t = (dx * d2y - dy * d2x) * inv_cross;
    let u = (dx * d1y - dy * d1x) * inv_cross;

    // Strict interior: exclude endpoints to avoid self-intersection at corners
    t > 0.01 && t < 0.99 && u > 0.01 && u < 0.99
}

/// Snap a color to the nearest palette color by RGB Euclidean distance.
fn snap_to_nearest(palette: &[u32], color: u32) -> u32 {
    let cr = ((color >> 16) & 0xFF) as i32;
    let cg = ((color >> 8) & 0xFF) as i32;
    let cb = (color & 0xFF) as i32;
    let mut best = color;
    let mut best_dist = i32::MAX;
    for &p in palette {
        let pr = ((p >> 16) & 0xFF) as i32;
        let pg = ((p >> 8) & 0xFF) as i32;
        let pb = (p & 0xFF) as i32;
        let d = (cr - pr) * (cr - pr) + (cg - pg) * (cg - pg) + (cb - pb) * (cb - pb);
        if d < best_dist {
            best_dist = d;
            best = p;
        }
    }
    best
}

// --- Spline-bounded Gaussian diffusion rasterizer (Paper Section 3.5) ---
//
// Combines the smooth B-spline contour boundaries from the vectorization
// pipeline with Gaussian color diffusion from the paper's rendering method.
//
// 1. Scanline-rasterize the vector paths to a flat-color buffer (smooth spline boundaries)
// 2. For each output pixel, Gaussian-blend nearby source pixel centroids that
//    fall within the same spline-bounded region (same rasterized color)
//
// This gives smooth spline edges AND soft Gaussian anti-aliasing, matching
// the paper's described rendering approach.

/// Rasterize using Gaussian diffusion with B-spline contour boundaries.
///
/// First rasterizes the vectorized paths via the scanline rasterizer to establish
/// smooth region boundaries. Then applies Gaussian blending (σ=1, r=2) within
/// those spline-bounded regions. Each source pixel centroid contributes to output
/// pixels that share its region (same rasterized color at the centroid's position).
pub fn rasterize_spline_diffusion(
    paths: &[ColorPath],
    pixels: &[u32],
    width: usize,
    height: usize,
    bg_color: u32,
    scale: usize,
) -> (Vec<u32>, usize, usize) {
    let out_w = width * scale;
    let out_h = height * scale;

    // Step 1: Scanline-rasterize with AA, snap to palette for region boundaries,
    // then flood-fill connected components for region IDs.
    let aa_buf = rasterize(paths, width, height, bg_color, scale);
    let palette: Vec<u32> = paths.iter().map(|p| p.color).chain(std::iter::once(bg_color)).collect();
    let color_buf: Vec<u32> = aa_buf.iter().map(|&c| snap_to_nearest(&palette, c)).collect();
    let region_ids = flood_fill_output_regions(&color_buf, out_w, out_h);

    // Step 2: Map source centroids to regions. Each source pixel's snapped color
    // determines which output region it belongs to.
    let mut src_region = vec![0u32; width * height];
    for py in 0..height {
        for px in 0..width {
            // Find a point in the centroid's output cell with the right color
            let src_color = snap_to_nearest(&palette, pixels[py * width + px]);
            let base_ox = px * scale;
            let base_oy = py * scale;
            let mut assigned = false;
            // Scan in spiral from center outward
            let cx = (base_ox + scale / 2).min(out_w - 1);
            let cy = (base_oy + scale / 2).min(out_h - 1);
            if color_buf[cy * out_w + cx] == src_color {
                src_region[py * width + px] = region_ids[cy * out_w + cx];
                assigned = true;
            }
            if !assigned {
                for dy in 0..scale {
                    for dx in 0..scale {
                        let ox = (base_ox + dx).min(out_w - 1);
                        let oy = (base_oy + dy).min(out_h - 1);
                        if color_buf[oy * out_w + ox] == src_color {
                            src_region[py * width + px] = region_ids[oy * out_w + ox];
                            assigned = true;
                            break;
                        }
                    }
                    if assigned { break; }
                }
            }
            if !assigned {
                src_region[py * width + px] = region_ids[cy * out_w + cx];
            }
        }
    }

    // Step 3: Gaussian diffusion within connected regions.
    let inv_scale = 1.0 / scale as f64;
    let sigma_sq_2 = 2.0;
    let radius = 2.0f64;
    let r_sq = radius * radius;

    let mut buffer = vec![0u32; out_w * out_h];

    for oy in 0..out_h {
        let sy = (oy as f64 + 0.5) * inv_scale;
        let min_py = ((sy - radius).floor() as i32).max(0) as usize;
        let max_py = ((sy + radius).ceil() as i32).min(height as i32 - 1) as usize;

        for ox in 0..out_w {
            let sx = (ox as f64 + 0.5) * inv_scale;
            let my_region = region_ids[oy * out_w + ox];

            let min_px = ((sx - radius).floor() as i32).max(0) as usize;
            let max_px = ((sx + radius).ceil() as i32).min(width as i32 - 1) as usize;

            let mut tr = 0.0f64;
            let mut tg = 0.0f64;
            let mut tb = 0.0f64;
            let mut tw = 0.0f64;

            for py in min_py..=max_py {
                for px in min_px..=max_px {
                    if src_region[py * width + px] != my_region {
                        continue;
                    }

                    let dx = sx - (px as f64 + 0.5);
                    let dy = sy - (py as f64 + 0.5);
                    let d_sq = dx * dx + dy * dy;
                    if d_sq > r_sq { continue; }

                    let w = (-d_sq / sigma_sq_2).exp();
                    let color = pixels[py * width + px];
                    tr += w * ((color >> 16) & 0xFF) as f64;
                    tg += w * ((color >> 8) & 0xFF) as f64;
                    tb += w * (color & 0xFF) as f64;
                    tw += w;
                }
            }

            if tw > 0.0 {
                let inv_tw = 1.0 / tw;
                let r = (tr * inv_tw).round().min(255.0) as u32;
                let g = (tg * inv_tw).round().min(255.0) as u32;
                let b = (tb * inv_tw).round().min(255.0) as u32;
                buffer[oy * out_w + ox] = (r << 16) | (g << 8) | b;
            } else {
                buffer[oy * out_w + ox] = color_buf[oy * out_w + ox];
            }
        }
    }

    (buffer, out_w, out_h)
}
