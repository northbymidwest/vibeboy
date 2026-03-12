//! Rasterize vector paths (ColorPath) to a pixel buffer using scanline rendering
//! with 4x4 supersampling for anti-aliased edges.

use super::contour::{ColorPath, PathSegment};

/// A line segment in output pixel space.
struct Edge {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
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
    let out_w = width * scale;
    let out_h = height * scale;
    let mut buffer = vec![bg_color; out_w * out_h];
    let scale_f = scale as f64;
    let tolerance = 0.25 / scale_f;

    for path in paths {
        if path.color == bg_color || path.segments.is_empty() {
            continue;
        }

        // Extract and flatten all edges for this path, scaled to output space
        let edges = extract_edges(path, scale_f, tolerance);
        if edges.is_empty() {
            continue;
        }

        // Sort edges by y_min for active edge tracking
        let mut sorted_indices: Vec<usize> = (0..edges.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            let ya = edges[a].y0.min(edges[a].y1);
            let yb = edges[b].y0.min(edges[b].y1);
            ya.partial_cmp(&yb).unwrap()
        });

        // Rasterize with 4x4 supersampling
        rasterize_path_supersampled(
            &edges,
            &sorted_indices,
            path.color,
            bg_color,
            &mut buffer,
            out_w,
            out_h,
        );
    }

    buffer
}

/// Flatten path segments into line edges in output pixel space.
fn extract_edges(path: &ColorPath, scale: f64, tolerance: f64) -> Vec<Edge> {
    let mut edges = Vec::new();

    for seg in &path.segments {
        match seg {
            PathSegment::Line(a, b) => {
                let e = Edge {
                    x0: a.x * scale,
                    y0: a.y * scale,
                    x1: b.x * scale,
                    y1: b.y * scale,
                };
                // Skip degenerate edges
                if (e.y0 - e.y1).abs() > 1e-10 {
                    edges.push(e);
                }
            }
            PathSegment::QuadBezier(start, ctrl, end) => {
                flatten_quad(
                    start.x * scale,
                    start.y * scale,
                    ctrl.x * scale,
                    ctrl.y * scale,
                    end.x * scale,
                    end.y * scale,
                    tolerance * scale,
                    &mut edges,
                );
            }
        }
    }

    edges
}

/// Recursively flatten a quadratic Bezier curve into line segments via de Casteljau.
fn flatten_quad(
    x0: f64, y0: f64,
    cx: f64, cy: f64,
    x1: f64, y1: f64,
    tolerance: f64,
    edges: &mut Vec<Edge>,
) {
    // Check flatness: distance from control point to line (x0,y0)-(x1,y1)
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len_sq = dx * dx + dy * dy;
    let flatness = if len_sq < 1e-12 {
        let dcx = cx - x0;
        let dcy = cy - y0;
        (dcx * dcx + dcy * dcy).sqrt()
    } else {
        let t = ((cx - x0) * dx + (cy - y0) * dy) / len_sq;
        let px = x0 + t * dx;
        let py = y0 + t * dy;
        let ex = cx - px;
        let ey = cy - py;
        (ex * ex + ey * ey).sqrt()
    };

    if flatness <= tolerance {
        // Flat enough — emit as a line
        if (y0 - y1).abs() > 1e-10 {
            edges.push(Edge { x0, y0, x1, y1 });
        }
        return;
    }

    // Subdivide at t=0.5
    let mx01 = (x0 + cx) * 0.5;
    let my01 = (y0 + cy) * 0.5;
    let mx12 = (cx + x1) * 0.5;
    let my12 = (cy + y1) * 0.5;
    let midx = (mx01 + mx12) * 0.5;
    let midy = (my01 + my12) * 0.5;

    flatten_quad(x0, y0, mx01, my01, midx, midy, tolerance, edges);
    flatten_quad(midx, midy, mx12, my12, x1, y1, tolerance, edges);
}

/// Rasterize a single path's edges with 4x4 supersampling.
fn rasterize_path_supersampled(
    edges: &[Edge],
    sorted_indices: &[usize],
    fill_color: u32,
    _bg_color: u32,
    buffer: &mut [u32],
    out_w: usize,
    out_h: usize,
) {
    let sub_offsets: [f64; 4] = [0.125, 0.375, 0.625, 0.875];

    // Coverage buffer for one row of pixels
    let mut coverage = vec![0u8; out_w];

    // Track where sorted_indices scanning starts
    let mut scan_start = 0usize;

    for py in 0..out_h {
        let y_top = py as f64;
        let y_bot = y_top + 1.0;

        // Reset coverage
        for c in coverage.iter_mut() {
            *c = 0;
        }

        // Advance scan_start past edges entirely above this row
        while scan_start < sorted_indices.len() {
            let e = &edges[sorted_indices[scan_start]];
            if e.y0.max(e.y1) <= y_top {
                scan_start += 1;
            } else {
                break;
            }
        }

        // Process 4 sub-scanlines
        for &sub_off in &sub_offsets {
            let sy = y_top + sub_off;

            // Collect x-intersections for this sub-scanline
            let mut intersections: Vec<(f64, i32)> = Vec::new();

            for i in scan_start..sorted_indices.len() {
                let e = &edges[sorted_indices[i]];
                let e_ymin = e.y0.min(e.y1);
                let e_ymax = e.y0.max(e.y1);

                // Early exit: if edge starts below this row, all remaining do too
                if e_ymin >= y_bot {
                    break;
                }

                // Does this edge cross the sub-scanline?
                if sy < e_ymin || sy >= e_ymax {
                    continue;
                }

                // Compute x intersection
                let t = (sy - e.y0) / (e.y1 - e.y0);
                let ix = e.x0 + t * (e.x1 - e.x0);

                // Winding direction: +1 if edge goes down, -1 if up
                let dir = if e.y1 > e.y0 { 1i32 } else { -1i32 };
                intersections.push((ix, dir));
            }

            // Sort intersections by x
            intersections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Apply nonzero winding rule
            let mut winding = 0i32;
            let mut i = 0;
            while i < intersections.len() {
                let (x_enter, dir) = intersections[i];
                winding += dir;

                if winding != 0 {
                    // Find where winding returns to zero
                    let mut j = i + 1;
                    while j < intersections.len() {
                        winding += intersections[j].1;
                        if winding == 0 {
                            break;
                        }
                        j += 1;
                    }

                    let x_exit = if j < intersections.len() {
                        intersections[j].0
                    } else {
                        // Shouldn't happen with proper closed paths, but be safe
                        break;
                    };

                    // Mark pixels in [x_enter, x_exit] as covered for this sub-scanline
                    let px_start = (x_enter.max(0.0) as usize).min(out_w);
                    let px_end = ((x_exit.ceil() as usize).min(out_w)).max(px_start);

                    for px in px_start..px_end {
                        // Check sub-pixel coverage within this pixel
                        let pixel_left = px as f64;
                        // How many of 4 horizontal sub-pixels are inside?
                        for &sx_off in &sub_offsets {
                            let sx = pixel_left + sx_off;
                            if sx >= x_enter && sx < x_exit {
                                coverage[px] += 1;
                            }
                        }
                    }

                    i = j + 1;
                } else {
                    i += 1;
                }
            }
        }

        // Write pixels from coverage
        let row_start = py * out_w;
        for px in 0..out_w {
            let cov = coverage[px];
            if cov == 0 {
                continue;
            }

            if cov >= 16 {
                buffer[row_start + px] = fill_color;
            } else {
                // Blend fill over existing pixel
                let existing = buffer[row_start + px];
                buffer[row_start + px] = blend(existing, fill_color, cov);
            }
        }
    }
}

/// Blend two ARGB colors with coverage (0..16).
fn blend(bg: u32, fg: u32, coverage: u8) -> u32 {
    let alpha = coverage as u32;
    let inv = 16 - alpha;

    let bg_r = (bg >> 16) & 0xFF;
    let bg_g = (bg >> 8) & 0xFF;
    let bg_b = bg & 0xFF;

    let fg_r = (fg >> 16) & 0xFF;
    let fg_g = (fg >> 8) & 0xFF;
    let fg_b = fg & 0xFF;

    let r = (bg_r * inv + fg_r * alpha) / 16;
    let g = (bg_g * inv + fg_g * alpha) / 16;
    let b = (bg_b * inv + fg_b * alpha) / 16;

    0xFF000000 | (r << 16) | (g << 8) | b
}
