//! Voronoi diffusion rasterizer (Paper Section 3.5).
//!
//! Places truncated Gaussian influence functions at cell centroids and
//! blends colors within contour-bounded regions.

use super::super::graph::SimilarityGraph;

/// Rasterize using Gaussian color diffusion (Paper Section 3.5).
///
/// Each pixel centroid emits its color with a truncated Gaussian (sigma=1, r=2).
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
    let graph = super::super::graph::build(pixels, width, height);

    // Flood-fill source pixels along graph edges to find contour-bounded regions.
    let src_regions = build_graph_regions(width, height, &graph);

    // Scanline-fill each Voronoi cell polygon at output resolution to build
    // a smooth ownership map.
    let ownership = build_voronoi_ownership(width, height, &graph, scale);

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
            }
        }
    }

    (buffer, out_w, out_h)
}

/// Build region labels by flood-filling along resolved similarity graph edges.
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

                if e.right && cx + 1 < w && regions[cy * w + cx + 1] == u32::MAX {
                    regions[cy * w + cx + 1] = region_id;
                    stack.push((cx + 1, cy));
                }
                if e.down && cy + 1 < h && regions[(cy + 1) * w + cx] == u32::MAX {
                    regions[(cy + 1) * w + cx] = region_id;
                    stack.push((cx, cy + 1));
                }
                if e.down_right && cx + 1 < w && cy + 1 < h
                    && regions[(cy + 1) * w + cx + 1] == u32::MAX
                {
                    regions[(cy + 1) * w + cx + 1] = region_id;
                    stack.push((cx + 1, cy + 1));
                }
                if e.down_left && cx > 0 && cy + 1 < h
                    && regions[(cy + 1) * w + cx - 1] == u32::MAX
                {
                    regions[(cy + 1) * w + cx - 1] = region_id;
                    stack.push((cx - 1, cy + 1));
                }
                if cx > 0 && graph.edge(cx - 1, cy).right
                    && regions[cy * w + cx - 1] == u32::MAX
                {
                    regions[cy * w + cx - 1] = region_id;
                    stack.push((cx - 1, cy));
                }
                if cy > 0 && graph.edge(cx, cy - 1).down
                    && regions[(cy - 1) * w + cx] == u32::MAX
                {
                    regions[(cy - 1) * w + cx] = region_id;
                    stack.push((cx, cy - 1));
                }
                if cx + 1 < w && cy > 0 && graph.edge(cx + 1, cy - 1).down_left
                    && regions[(cy - 1) * w + cx + 1] == u32::MAX
                {
                    regions[(cy - 1) * w + cx + 1] = region_id;
                    stack.push((cx + 1, cy - 1));
                }
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
fn cell_vertices_f64(px: usize, py: usize, graph: &SimilarityGraph) -> [(f64, f64); 8] {
    let mut verts = [(0.0, 0.0); 8];
    let mut n = 0usize;
    let bx = px as f64;
    let by = py as f64;

    match corner_diag(graph, px, py) {
        1 => { verts[n] = (bx - 0.25, by + 0.25); n += 1;
               verts[n] = (bx + 0.25, by - 0.25); n += 1; }
        2 => { verts[n] = (bx + 0.25, by + 0.25); n += 1; }
        _ => { verts[n] = (bx, by); n += 1; }
    }
    match corner_diag(graph, px + 1, py) {
        1 => { verts[n] = (bx + 0.75, by + 0.25); n += 1; }
        2 => { verts[n] = (bx + 0.75, by - 0.25); n += 1;
               verts[n] = (bx + 1.25, by + 0.25); n += 1; }
        _ => { verts[n] = (bx + 1.0, by); n += 1; }
    }
    match corner_diag(graph, px + 1, py + 1) {
        1 => { verts[n] = (bx + 1.25, by + 0.75); n += 1;
               verts[n] = (bx + 0.75, by + 1.25); n += 1; }
        2 => { verts[n] = (bx + 0.75, by + 0.75); n += 1; }
        _ => { verts[n] = (bx + 1.0, by + 1.0); n += 1; }
    }
    match corner_diag(graph, px, py + 1) {
        1 => { verts[n] = (bx + 0.25, by + 0.75); n += 1; }
        2 => { verts[n] = (bx + 0.25, by + 1.25); n += 1;
               verts[n] = (bx - 0.25, by + 0.75); n += 1; }
        _ => { verts[n] = (bx, by + 1.0); n += 1; }
    }

    verts
}

/// Check if point (px, py) is inside convex polygon with `nv` vertices.
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

/// Count vertices for a cell.
#[inline]
fn cell_nv(graph: &SimilarityGraph, px: usize, py: usize) -> usize {
    let corners = [
        (corner_diag(graph, px, py), true),
        (corner_diag(graph, px + 1, py), false),
        (corner_diag(graph, px + 1, py + 1), true),
        (corner_diag(graph, px, py + 1), false),
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

            let home_x = (sx.floor() as usize).min(w - 1);
            let home_y = (sy.floor() as usize).min(h - 1);
            let home_id = (home_y * w + home_x) as u32;

            let tl = corner_diag(graph, home_x, home_y);
            let tr = corner_diag(graph, home_x + 1, home_y);
            let br = corner_diag(graph, home_x + 1, home_y + 1);
            let bl = corner_diag(graph, home_x, home_y + 1);

            if tl == 0 && tr == 0 && br == 0 && bl == 0 {
                ownership[oy * out_w + ox] = home_id;
                continue;
            }

            let home_verts = cell_vertices_f64(home_x, home_y, graph);
            let home_nv = cell_nv(graph, home_x, home_y);

            if point_in_convex_poly(&home_verts, home_nv, sx, sy) {
                ownership[oy * out_w + ox] = home_id;
                continue;
            }

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

            if !found {
                ownership[oy * out_w + ox] = home_id;
            }
        }
    }

    ownership
}
