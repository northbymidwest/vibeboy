//! Scanline edge construction for the vectorize pipeline.
//!
//! Builds directed scan edges from B-spline control point chains for
//! color-sweep scanline fill. Each edge carries screen-left/right colors
//! determined by the same resolve_color logic as the GPU cell rasterizer.

use super::vectorize::VectorizeData;

const IS_TJUNCTION: u32 = 32;
const IS_CROSSING: u32 = 64;
const SHARP_MASK: u32 = IS_TJUNCTION | IS_CROSSING;

/// A boundary edge for scanline color-sweep fill.
/// Each edge separates two color regions. When a scanline crosses this edge
/// from left to right, the active color transitions from `screen_left` to
/// `screen_right`.
pub struct ScanEdge {
    pub x_at_ymin: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub dx_per_dy: f32,
    /// Color on the screen-right side of this edge.
    pub screen_right: u32,
    /// Color on the screen-left side of this edge.
    pub screen_left: u32,
    /// Source CP index (for diagnostics).
    pub ci: usize,
    /// CP flag bits (for identifying T-junctions).
    pub cp_flag: u32,
    /// Grid corner + direction info (diagnostics).
    pub diag_icx: i32,
    pub diag_icy: i32,
    pub diag_prev_dir: i32,
    pub diag_next_dir: i32,
    pub diag_prev_ci: i32,
    pub diag_next_ci: i32,
    /// Which color source was used: 0=prev, 1=next
    pub diag_color_src: u8,
}

/// Build scan edges directly from the CP chain data.
///
/// Each active CP defines a B-spline span using the same (pp, pos, np) triplet
/// as the existing nearest-curve rasterizer, with the same degenerate fallback
/// for chain endpoints (missing neighbor → use self).
///
/// Color resolution matches cell_rasterizer.slang resolve_color():
/// - For next_dir: colors are SWAPPED (left=nr, right=nl from get_edge_colors)
/// - For prev_dir: colors are NOT swapped (left=pl, right=pr)
///
/// For downward segments, left=screen-left, right=screen-right (from tangent
/// normal analysis). For upward segments, the screen relationship flips.
pub fn build_scan_edges(
    data: &VectorizeData,
    pixels: &[u32],
    scale_factor: f32,
) -> Vec<ScanEdge> {
    let corners_w = data.img_w + 1;
    let num_cps = corners_w * (data.img_h + 1) * 2;
    let img_w = data.img_w;
    let img_h = data.img_h;

    let get_px_color = |px: i32, py: i32| -> u32 {
        let px = px.clamp(0, img_w as i32 - 1) as usize;
        let py = py.clamp(0, img_h as i32 - 1) as usize;
        pixels[py * img_w + px]
    };
    let get_edge_colors = |icx: i32, icy: i32, dir: i32| -> (u32, u32) {
        match dir {
            0 => (get_px_color(icx - 1, icy - 1), get_px_color(icx, icy - 1)),
            1 => (get_px_color(icx, icy - 1), get_px_color(icx, icy)),
            2 => (get_px_color(icx, icy), get_px_color(icx - 1, icy)),
            3 => (get_px_color(icx - 1, icy), get_px_color(icx - 1, icy - 1)),
            _ => (0, 0),
        }
    };

    let mut edges = Vec::new();

    for ci in 0..num_cps {
        let flag = data.flags[ci];
        if flag == 0 { continue; }

        let prev_ci = data.neighbors[ci * 4];
        let next_ci = data.neighbors[ci * 4 + 1];
        if prev_ci < 0 && next_ci < 0 { continue; }

        let prev_dir = data.neighbors[ci * 4 + 2];
        let next_dir = data.neighbors[ci * 4 + 3];

        let icx = (ci / 2 % corners_w) as i32;
        let icy = (ci / 2 / corners_w) as i32;

        let prev_colors = if prev_dir >= 0 {
            let (pl, pr) = get_edge_colors(icx, icy, prev_dir);
            if pl != pr { Some((pl, pr)) } else { None }
        } else { None };
        let next_colors = if next_dir >= 0 {
            let (nl, nr) = get_edge_colors(icx, icy, next_dir);
            if nl != nr { Some((nr, nl)) } else { None }
        } else { None };

        if prev_colors.is_none() && next_colors.is_none() { continue; }

        let pos = (data.positions[ci * 2], data.positions[ci * 2 + 1]);
        let pp = if prev_ci >= 0 {
            (data.positions[prev_ci as usize * 2], data.positions[prev_ci as usize * 2 + 1])
        } else { pos };
        let np = if next_ci >= 0 {
            (data.positions[next_ci as usize * 2], data.positions[next_ci as usize * 2 + 1])
        } else { pos };

        let orig_pos = (data.orig_positions[ci * 2], data.orig_positions[ci * 2 + 1]);
        let orig_pp = if prev_ci >= 0 {
            (data.orig_positions[prev_ci as usize * 2], data.orig_positions[prev_ci as usize * 2 + 1])
        } else { orig_pos };
        let orig_np = if next_ci >= 0 {
            (data.orig_positions[next_ci as usize * 2], data.orig_positions[next_ci as usize * 2 + 1])
        } else { orig_pos };

        // Adaptive subdivision. For junction CPs, force t=0.5 as a
        // mandatory sample so all segments at a junction terminate at
        // the same point (beval at t=0.5).
        let chord_mid_x = (pp.0 + np.0) * 0.25 + pos.0 * 0.5;
        let chord_mid_y = (pp.1 + np.1) * 0.25 + pos.1 * 0.5;
        let dev = ((pos.0 - chord_mid_x).powi(2) + (pos.1 - chord_mid_y).powi(2)).sqrt();
        let subdiv = (dev * scale_factor * 2.0).ceil().clamp(1.0, 16.0) as usize;
        let is_junction = flag & SHARP_MASK != 0;

        let mut t_values: Vec<f32> = Vec::with_capacity(subdiv + 2);
        for s in 1..=subdiv {
            t_values.push(s as f32 / subdiv as f32);
        }
        if is_junction && !t_values.contains(&0.5) {
            t_values.push(0.5);
            t_values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        }

        let mut prev_pt = beval(pp, pos, np, 0.0, scale_factor);
        let mut t_prev_val = 0.0f32;

        for &t in &t_values {
            let cur_pt = beval(pp, pos, np, t, scale_factor);
            let t_prev = t_prev_val;

            let dy = cur_pt.1 - prev_pt.1;
            let mid_t = (t_prev + t) * 0.5;

            // For chain endpoints at junctions, clip segments that overshoot
            // past the junction point. Only for endpoints (prev_ci<0 or
            // next_ci<0), NOT through-chain CPs which pass through junctions.
            if dy.abs() > 1e-6 && is_junction && (prev_ci < 0 || next_ci < 0) {
                let u = 1.0 - mid_t;
                let curve_x = 0.5 * u * u * pp.0 + (u * mid_t + 0.5) * pos.0 + 0.5 * mid_t * mid_t * np.0;
                let curve_y = 0.5 * u * u * pp.1 + (u * mid_t + 0.5) * pos.1 + 0.5 * mid_t * mid_t * np.1;
                let (toward_x, toward_y) = if mid_t < 0.5 {
                    (pp.0 - pos.0, pp.1 - pos.1)
                } else {
                    (np.0 - pos.0, np.1 - pos.1)
                };
                let offset_x = curve_x - pos.0;
                let offset_y = curve_y - pos.1;
                if toward_x * offset_x + toward_y * offset_y < 0.0 {
                    prev_pt = cur_pt;
                    t_prev_val = t;
                    continue;
                }
            }

            if dy.abs() > 1e-6 {
                let (ymin, ymax, x_at_ymin) = if prev_pt.1 < cur_pt.1 {
                    (prev_pt.1, cur_pt.1, prev_pt.0)
                } else {
                    (cur_pt.1, prev_pt.1, cur_pt.0)
                };
                let dx_per_dy = (cur_pt.0 - prev_pt.0) / dy;

                let (resolve_left, resolve_right, ref_t, color_src) = if prev_ci < 0 {
                    let c = next_colors.unwrap_or_else(|| prev_colors.unwrap());
                    (c.0, c.1, 1.0f32, 1u8)
                } else if next_ci < 0 {
                    let c = prev_colors.unwrap_or_else(|| next_colors.unwrap());
                    (c.0, c.1, 0.0f32, 0u8)
                } else {
                    let seg_mid = ((prev_pt.0 + cur_pt.0) * 0.5,
                                   (prev_pt.1 + cur_pt.1) * 0.5);
                    // For crossings (inverse B-spline), the curve passes
                    // through beval@0.5, not pos. Use the actual junction.
                    let junc_src = if flag & IS_CROSSING != 0 {
                        (0.125 * pp.0 + 0.75 * pos.0 + 0.125 * np.0,
                         0.125 * pp.1 + 0.75 * pos.1 + 0.125 * np.1)
                    } else { pos };
                    let junc = (junc_src.0 * scale_factor, junc_src.1 * scale_factor);
                    let toward_prev = (pp.0 - pos.0, pp.1 - pos.1);
                    let offset = (seg_mid.0 - junc.0, seg_mid.1 - junc.1);
                    let on_prev_side = toward_prev.0 * offset.0
                        + toward_prev.1 * offset.1 > 0.0;
                    if on_prev_side {
                        if let Some(c) = prev_colors { (c.0, c.1, 0.0f32, 0u8) }
                        else { let c = next_colors.unwrap(); (c.0, c.1, 1.0, 1) }
                    } else {
                        if let Some(c) = next_colors { (c.0, c.1, 1.0f32, 1u8) }
                        else { let c = prev_colors.unwrap(); (c.0, c.1, 0.0, 0) }
                    }
                };

                let opt_tan = beval_deriv(pp, pos, np, mid_t);
                let orig_tan = beval_deriv(orig_pp, orig_pos, orig_np, ref_t);
                let normals_agree = opt_tan.0 * orig_tan.0 + opt_tan.1 * orig_tan.1 > 0.0;

                let (screen_right, screen_left) =
                    if (dy > 0.0) == normals_agree {
                        (resolve_right, resolve_left)
                    } else {
                        (resolve_left, resolve_right)
                    };

                edges.push(ScanEdge {
                    x_at_ymin, y_min: ymin, y_max: ymax, dx_per_dy,
                    screen_right, screen_left,
                    ci, cp_flag: flag,
                    diag_icx: icx, diag_icy: icy,
                    diag_prev_dir: prev_dir, diag_next_dir: next_dir,
                    diag_prev_ci: prev_ci, diag_next_ci: next_ci,
                    diag_color_src: color_src,
                });
            }

            prev_pt = cur_pt;
            t_prev_val = t;
        }
    }

    edges
}

#[inline]
fn beval_deriv(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), t: f32) -> (f32, f32) {
    ((t - 1.0) * p0.0 + (1.0 - 2.0 * t) * p1.0 + t * p2.0,
     (t - 1.0) * p0.1 + (1.0 - 2.0 * t) * p1.1 + t * p2.1)
}

#[inline]
fn beval(pp: (f32, f32), pos: (f32, f32), np: (f32, f32), t: f32, scale: f32) -> (f32, f32) {
    let u = 1.0 - t;
    (
        (0.5 * u * u * pp.0 + (u * t + 0.5) * pos.0 + 0.5 * t * t * np.0) * scale,
        (0.5 * u * u * pp.1 + (u * t + 0.5) * pos.1 + 0.5 * t * t * np.1) * scale,
    )
}
