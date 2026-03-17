//! GPU edge preparation for compute shader upload.

use super::super::contour::{ColorPath, PathSegment};
use super::scanline::{Edge, extract_edges};

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

    // Build per-row index
    let num_rows = out_h as usize;
    let mut row_buckets: Vec<Vec<u32>> = vec![Vec::new(); num_rows];
    for (i, e) in all_edges.iter().enumerate() {
        let row_start = (e.y_min.floor() as usize).min(num_rows.saturating_sub(1));
        let row_end = (e.y_max.ceil() as usize).min(num_rows);
        for row in row_start..row_end {
            row_buckets[row].push(i as u32);
        }
    }

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
