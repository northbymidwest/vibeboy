//! Scanline rasterizer with analytical anti-aliasing.
//!
//! Processes ALL paths simultaneously per scanline row. At each pixel,
//! edge crossings determine which color region is active. Boundary pixels
//! (where an edge crosses within the pixel) get analytically blended
//! between the two adjacent region colors based on the crossing position.
//!
//! Two Y samples per pixel row (at y+0.25 and y+0.75) provide vertical AA.
//! Each sample uses analytical X coverage for horizontal AA. The two samples
//! are averaged to produce the final pixel color.
//!
//! This eliminates seam artifacts from the painter's algorithm: adjacent
//! regions blend directly (red↔blue) without background color bleeding.

use super::super::contour::{ColorPath, PathSegment};

/// A line segment in output pixel space with precomputed fields.
pub(super) struct Edge {
    pub(super) x0: f64,
    pub(super) y0: f64,
    pub(super) dx_per_dy: f64,
    pub(super) y_min: f64,
    pub(super) y_max: f64,
    pub(super) dir: i32,
}

impl Edge {
    #[inline(always)]
    pub(super) fn new(x0: f64, y0: f64, x1: f64, y1: f64) -> Self {
        let dy = y1 - y0;
        Edge {
            x0, y0,
            dx_per_dy: (x1 - x0) / dy,
            y_min: y0.min(y1),
            y_max: y0.max(y1),
            dir: if dy > 0.0 { 1 } else { -1 },
        }
    }

    #[inline(always)]
    pub(super) fn intersect_x(&self, sy: f64) -> f64 {
        self.x0 + (sy - self.y0) * self.dx_per_dy
    }
}

/// Flatten a quadratic Bezier into line edges via recursive subdivision.
#[inline(always)]
pub(super) fn flatten_quad(
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

/// Edge with color tag for global multi-path rasterization.
struct ColorEdge {
    x0: f64,
    y0: f64,
    dx_per_dy: f64,
    y_min: f64,
    y_max: f64,
    dir: i32,
    color: u32,
}

impl ColorEdge {
    #[inline(always)]
    fn intersect_x(&self, sy: f64) -> f64 {
        self.x0 + (sy - self.y0) * self.dx_per_dy
    }
}

/// Extract edges from path segments, scaled to output space.
pub(super) fn extract_edges(segments: &[PathSegment], sx: f64, sy: f64, tol_sq: f64, edges: &mut Vec<Edge>) {
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

/// Rasterize vector paths at a floating-point scale factor with analytical AA.
///
/// All paths are processed simultaneously per scanline. Edge crossings from
/// every path are sorted by X, and the active color tracks which region we're
/// inside. Boundary pixels blend the two adjacent colors analytically.
/// Two Y samples per pixel row provide vertical anti-aliasing.
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

    // Build global edge list from all paths, tagged with color.
    let mut all_edges: Vec<ColorEdge> = Vec::new();
    let mut tmp_edges: Vec<Edge> = Vec::new();

    for path in paths {
        if path.segments.is_empty() { continue; }
        extract_edges(&path.segments, sx, sy, tol_sq, &mut tmp_edges);
        for e in &tmp_edges {
            all_edges.push(ColorEdge {
                x0: e.x0, y0: e.y0, dx_per_dy: e.dx_per_dy,
                y_min: e.y_min, y_max: e.y_max, dir: e.dir,
                color: path.color,
            });
        }
    }

    if all_edges.is_empty() {
        return (buffer, out_w, out_h);
    }

    // Bucket sort edges by starting row for scanline sweep.
    let mut bucket_heads = vec![u32::MAX; out_h];
    let mut bucket_next = vec![u32::MAX; all_edges.len()];
    for (i, e) in all_edges.iter().enumerate() {
        let row = (e.y_min.floor() as usize).min(out_h - 1);
        bucket_next[i] = bucket_heads[row];
        bucket_heads[row] = i as u32;
    }
    let mut sorted: Vec<usize> = Vec::with_capacity(all_edges.len());
    for row in 0..out_h {
        let mut idx = bucket_heads[row];
        while idx != u32::MAX {
            sorted.push(idx as usize);
            idx = bucket_next[idx as usize];
        }
    }

    let mut crossings: Vec<(f64, u32, i32)> = Vec::new();
    let mut scan_start = 0usize;
    let mut winding_state: Vec<(u32, i32)> = Vec::new();

    // Two row buffers for the two Y samples
    let mut row_a = vec![bg_color; out_w];
    let mut row_b = vec![bg_color; out_w];

    for py in 0..out_h {
        let y_top = py as f64;
        let y_bot = y_top + 1.0;

        // Advance past edges fully above this row
        while scan_start < sorted.len() {
            if all_edges[sorted[scan_start]].y_max <= y_top {
                scan_start += 1;
            } else {
                break;
            }
        }

        // Render two sub-scanlines at y+0.25 and y+0.75 for vertical AA
        let y_samples = [y_top + 0.25, y_top + 0.75];

        for (si, &y_sample) in y_samples.iter().enumerate() {
            let row_buf = if si == 0 { &mut row_a } else { &mut row_b };
            row_buf.fill(bg_color);

            // Collect crossings at this Y from all active edges
            crossings.clear();
            for i in scan_start..sorted.len() {
                let e = &all_edges[sorted[i]];
                if e.y_min >= y_bot { break; }
                if y_sample >= e.y_min && y_sample < e.y_max {
                    crossings.push((e.intersect_x(y_sample), e.color, e.dir));
                }
            }

            if crossings.is_empty() {
                // No crossings — entire row is bg_color (already filled)
                continue;
            }

            crossings.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

            // Sweep left to right with winding state
            let mut active_color = bg_color;
            let mut last_px = 0usize;
            winding_state.clear();

            for &(x_cross, edge_color, dir) in &crossings {
                // Update winding for this color
                let mut found = false;
                for (c, w) in winding_state.iter_mut() {
                    if *c == edge_color {
                        *w += dir;
                        found = true;
                        break;
                    }
                }
                if !found {
                    winding_state.push((edge_color, dir));
                }

                // Active color: region with nonzero winding, or keep current
                // (to prevent seams at sub-pixel gaps between adjacent regions)
                let new_color = winding_state.iter()
                    .find(|(_, w)| *w != 0)
                    .map(|(c, _)| *c)
                    .unwrap_or(active_color);

                if new_color == active_color { continue; }

                // Off-screen left: just update color
                if x_cross < 0.0 {
                    active_color = new_color;
                    continue;
                }
                let cross_px = x_cross.floor() as usize;
                let solid_end = cross_px.min(out_w);

                // Fill solid pixels before crossing
                if last_px < solid_end {
                    row_buf[last_px..solid_end].fill(active_color);
                }

                // Blend boundary pixel with analytical X coverage
                if cross_px < out_w {
                    let frac = (x_cross - cross_px as f64).clamp(0.0, 1.0) as f32;
                    row_buf[cross_px] = lerp_color(active_color, new_color, frac);
                    last_px = cross_px + 1;
                } else {
                    last_px = solid_end;
                }

                active_color = new_color;
            }

            // Fill remaining pixels
            if last_px < out_w {
                row_buf[last_px..out_w].fill(active_color);
            }
        }

        // Average the two Y samples into the output buffer
        let row_start = py * out_w;
        for px in 0..out_w {
            let a = row_a[px];
            let b = row_b[px];
            if a == b {
                buffer[row_start + px] = a;
            } else {
                buffer[row_start + px] = avg_color(a, b);
            }
        }
    }

    (buffer, out_w, out_h)
}

/// Linearly interpolate two ARGB colors.
/// `t` = 0.0 → color_a, `t` = 1.0 → color_b.
#[inline(always)]
fn lerp_color(a: u32, b: u32, t: f32) -> u32 {
    let inv = 1.0 - t;
    let r = ((a >> 16) & 0xFF) as f32 * inv + ((b >> 16) & 0xFF) as f32 * t;
    let g = ((a >> 8) & 0xFF) as f32 * inv + ((b >> 8) & 0xFF) as f32 * t;
    let bl = (a & 0xFF) as f32 * inv + (b & 0xFF) as f32 * t;
    0xFF000000 | ((r as u32) << 16) | ((g as u32) << 8) | (bl as u32)
}

/// Average two ARGB colors (50/50 blend).
#[inline(always)]
fn avg_color(a: u32, b: u32) -> u32 {
    // Byte-parallel average: (a & b) + ((a ^ b) >> 1) with mask to prevent carry leak
    let mask = 0x00FEFEFE;
    let avg = (a & b & mask) + (((a ^ b) & mask) >> 1);
    0xFF000000 | avg
}

/// Blend two ARGB colors with 2x2 coverage (0..4). Kept for other rasterizers.
#[inline(always)]
pub(super) fn blend4(bg: u32, fg: u32, coverage: u8) -> u32 {
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
