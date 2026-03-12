//! Rasterize vector paths (ColorPath) to a pixel buffer using scanline rendering
//! with 2x2 supersampling for anti-aliased edges.

use super::contour::{ColorPath, PathSegment};

/// A line segment in output pixel space with precomputed fields.
struct Edge {
    x0: f64,
    y0: f64,
    x1: f64,
    inv_dy: f64,
    y_min: f64,
    y_max: f64,
    dir: i32,
}

impl Edge {
    #[inline(always)]
    fn new(x0: f64, y0: f64, x1: f64, y1: f64) -> Self {
        let dy = y1 - y0;
        Edge {
            x0, y0, x1,
            inv_dy: 1.0 / dy,
            y_min: y0.min(y1),
            y_max: y0.max(y1),
            dir: if dy > 0.0 { 1 } else { -1 },
        }
    }

    #[inline(always)]
    fn intersect_x(&self, sy: f64) -> f64 {
        let t = (sy - self.y0) * self.inv_dy;
        self.x0 + t * (self.x1 - self.x0)
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

        sorted.clear();
        sorted.extend(0..edges.len());
        sorted.sort_unstable_by(|&a, &b| {
            edges[a].y_min.total_cmp(&edges[b].y_min)
        });

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

        // Write pixels
        if dirty_min <= dirty_max {
            let row_start = py * out_w;
            let end = dirty_max.min(out_w - 1);
            for px in dirty_min..=end {
                let cov = coverage[px];
                if cov == 0 { continue; }
                if cov >= 4 {
                    buffer[row_start + px] = fill_color;
                } else {
                    buffer[row_start + px] = blend4(buffer[row_start + px], fill_color, cov);
                }
            }
        }
    }
}

/// GPU-ready edge data for compute shader upload.
#[cfg(feature = "gpu")]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuEdge {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub inv_dy: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub dir: i32,
    pub _pad: u32,
}

/// Per-path metadata for GPU compute shader.
#[cfg(feature = "gpu")]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuPathMeta {
    pub color: u32,
    pub edge_start: u32,
    pub edge_count: u32,
    pub _pad: u32,
}

/// Flatten all paths into GPU-ready edge and path metadata arrays.
/// Skips background-colored paths. Returns (edges, path_metas).
#[cfg(feature = "gpu")]
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
                x1: e.x1 as f32,
                inv_dy: e.inv_dy as f32,
                y_min: e.y_min as f32,
                y_max: e.y_max as f32,
                dir: e.dir,
                _pad: 0,
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
