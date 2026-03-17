//! Spline-bounded Gaussian diffusion rasterizer (Paper Section 3.5).
//!
//! Combines smooth B-spline contour boundaries from the vectorization
//! pipeline with Gaussian color diffusion.

use super::super::contour::ColorPath;
use super::scanline::rasterize;

/// Flood-fill connected components on a color buffer at output resolution.
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

/// Rasterize using Gaussian diffusion with B-spline contour boundaries.
///
/// First rasterizes the vectorized paths via the scanline rasterizer to establish
/// smooth region boundaries. Then applies Gaussian blending within those
/// spline-bounded regions.
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

    // Step 2: Map source centroids to regions.
    let mut src_region = vec![0u32; width * height];
    for py in 0..height {
        for px in 0..width {
            let src_color = snap_to_nearest(&palette, pixels[py * width + px]);
            let base_ox = px * scale;
            let base_oy = py * scale;
            let mut assigned = false;
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
    let gauss_k = 2.5;
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

                    let w = (-d_sq * gauss_k).exp();
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
