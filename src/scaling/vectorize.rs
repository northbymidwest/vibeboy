//! CPU implementation of the 6-stage GPU vectorize pipeline.
//!
//! This is a line-for-line faithful translation of the GPU compute shaders:
//!   1. similarity_graph.comp  -> build_similarity_graph()
//!   2. resolve_crossings.comp -> resolve_crossings()
//!   3. cell_graph.comp        -> build_cell_graph()
//!   4. update_tjunction.comp  -> update_tjunctions()
//!   5. optimize_energy.comp   -> optimize_energy()
//!   6. cell_rasterizer.comp   -> rasterize()
//!
//! Output is pixel-identical to the GPU pipeline.

const IS_CORNER: u32 = 16;
const IS_TJUNCTION: u32 = 32;
const IS_CROSSING: u32 = 64;
/// Chain endpoint (prev or next neighbor is -1). Set by write_cp/write_cp_full.
/// The rasterizer uses clamped Bezier boundaries when an adjacent CP carries
/// this bit, so the final span ends exactly at the endpoint position with a
/// real quadratic curve instead of a degenerate straight tail.
pub const IS_ENDPOINT: u32 = 128;

const NEWTON_ITER: i32 = 3;

// Direction bitmask encoding (matches reference)
const DIR_NW: u32 = 1;
const DIR_W: u32 = 2;
const DIR_SW: u32 = 4;
const DIR_S: u32 = 8;
const DIR_SE: u32 = 16;
const DIR_E: u32 = 32;
const DIR_NE: u32 = 64;
const DIR_N: u32 = 128;

/// Intermediate output from the vectorize-gpu pipeline (stages 1-5).
/// Contains optimized B-spline control points with connectivity and flags.
///
/// `crossing_t[ci]` holds the curve-curve intersection parameter for IS_CROSSING
/// CPs (t on the curve owned by `ci`). Non-crossing entries are 0.5 (the
/// natural midpoint). The rasterizer uses this as `t_branch` — the t at which
/// resolution switches from prev-edge to next-edge colors and the wedge AA
/// junction point lives.
pub struct VectorizeData {
    pub positions: Vec<f32>,
    pub orig_positions: Vec<f32>,
    pub neighbors: Vec<i32>,
    pub flags: Vec<u32>,
    pub crossing_t: Vec<f32>,
    pub graph: Vec<u32>,  // resolved similarity graph (2*w+1 × 2*h+1)
    pub img_w: usize,
    pub img_h: usize,
}

/// Run stages 1-5 of the vectorize-gpu pipeline without rasterizing.
/// Returns intermediate CP data for SVG export or other consumers.
pub fn vectorize(src: &[u32], src_w: usize, src_h: usize) -> VectorizeData {
    let graph = build_similarity_graph(src, src_w, src_h);
    let graph = resolve_crossings(&graph, src_w, src_h);
    let (positions, neighbors, flags) = build_cell_graph(&graph, src_w, src_h);

    let corners_w = src_w + 1;
    let corners_h = src_h + 1;
    let num_cps = corners_w * corners_h * 2;

    let orig_positions = positions.clone();
    let positions = optimize_energy(&positions, &orig_positions, &neighbors, &flags, num_cps);
    let mut positions = positions;
    let mut crossing_t = vec![0.5f32; num_cps];
    update_tjunctions(&mut positions, &mut crossing_t, &neighbors, &flags, num_cps);

    VectorizeData {
        positions,
        orig_positions,
        neighbors,
        flags,
        crossing_t,
        graph,
        img_w: src_w,
        img_h: src_h,
    }
}

/// Public entry point: runs all 6 GPU pipeline stages on CPU.
pub fn scale(src: &[u32], src_w: usize, src_h: usize, scale_factor: f32) -> Vec<u32> {
    let out_w = (src_w as f32 * scale_factor).ceil() as usize;
    let out_h = (src_h as f32 * scale_factor).ceil() as usize;

    let data = vectorize(src, src_w, src_h);

    rasterize(
        src,
        &data.positions,
        &data.orig_positions,
        &data.flags,
        &data.neighbors,
        &data.crossing_t,
        data.img_w,
        data.img_h,
        out_w,
        out_h,
        scale_factor,
    )
}

// ============================================================================
// Stage 1: Build similarity graph
// ============================================================================

fn build_similarity_graph(pixels: &[u32], img_w: usize, img_h: usize) -> Vec<u32> {
    let graph_stride = 2 * img_w + 1;
    let graph_h = 2 * img_h + 1;
    let mut graph = vec![0u32; graph_stride * graph_h];

    for y in 0..img_h {
        for x in 0..img_w {
            let c = pixels[y * img_w + x];

            // Store pixel color at odd coordinate
            graph[(2 * y + 1) * graph_stride + (2 * x + 1)] = c;

            // Horizontal edge to right neighbor
            if x + 1 < img_w {
                graph[(2 * y + 1) * graph_stride + (2 * x + 2)] =
                    if c == pixels[y * img_w + x + 1] { 1 } else { 0 };
            }

            // Vertical edge to down neighbor
            if y + 1 < img_h {
                graph[(2 * y + 2) * graph_stride + (2 * x + 1)] =
                    if c == pixels[(y + 1) * img_w + x] { 1 } else { 0 };
            }

            // Diagonal corner
            if x + 1 < img_w && y + 1 < img_h {
                let mut flags = 0u32;
                // Down-right: (x,y) -> (x+1,y+1)
                if c == pixels[(y + 1) * img_w + x + 1] {
                    flags |= 1;
                }
                // Down-left: (x+1,y) -> (x,y+1)
                if pixels[y * img_w + x + 1] == pixels[(y + 1) * img_w + x] {
                    flags |= 2;
                }
                graph[(2 * y + 2) * graph_stride + (2 * x + 2)] = flags;
            }

            // Border zeros
            if x == 0 {
                graph[(2 * y + 1) * graph_stride] = 0;
            }
            if y == 0 {
                graph[2 * x + 1] = 0;
            }
        }
    }

    graph
}

// ============================================================================
// Stage 2: Resolve crossings
// ============================================================================

fn resolve_crossings(graph_in: &[u32], img_w: usize, img_h: usize) -> Vec<u32> {
    let graph_stride = 2 * img_w + 1;
    let graph_height = 2 * img_h + 1;
    let mut graph_out = graph_in.to_vec();

    let g = |gx: usize, gy: usize| -> u32 { graph_in[gy * graph_stride + gx] };

    // Compute 8-connected valence bitmask for pixel (x,y)
    let valence_mask = |x: usize, y: usize| -> u32 {
        let mut v = 0u32;
        let gx = 2 * x + 1;
        let gy = 2 * y + 1;
        let gw = graph_stride;
        let gh = graph_height;

        // Cardinal
        if gx >= 2 && g(gx - 1, gy) != 0 {
            v |= DIR_W;
        }
        if gx + 1 < gw && g(gx + 1, gy) != 0 {
            v |= DIR_E;
        }
        if gy >= 2 && g(gx, gy - 1) != 0 {
            v |= DIR_N;
        }
        if gy + 1 < gh && g(gx, gy + 1) != 0 {
            v |= DIR_S;
        }
        // Diagonal
        if gx >= 2 && gy >= 2 && (g(gx - 1, gy - 1) & 1) != 0 {
            v |= DIR_NW;
        }
        if gx + 1 < gw && gy >= 2 && (g(gx + 1, gy - 1) & 2) != 0 {
            v |= DIR_NE;
        }
        if gx >= 2 && gy + 1 < gh && (g(gx - 1, gy + 1) & 2) != 0 {
            v |= DIR_SW;
        }
        if gx + 1 < gw && gy + 1 < gh && (g(gx + 1, gy + 1) & 1) != 0 {
            v |= DIR_SE;
        }
        v
    };

    let valence_count = |mask: u32| -> u32 { mask.count_ones() };

    // Walk a chain of valence-2 nodes using XOR direction tracking.
    let walk_chain = |sx: usize, sy: usize, fx: usize, fy: usize| -> u32 {
        let mut cx = sx;
        let mut cy = sy;
        let mut length = 1u32;

        let idx = fx as i32 - sx as i32;
        let idy = fy as i32 - sy as i32;
        let mut pred_dir = 0u32;
        if idx == 1 && idy == 0 {
            pred_dir = DIR_E;
        }
        if idx == -1 && idy == 0 {
            pred_dir = DIR_W;
        }
        if idx == 0 && idy == 1 {
            pred_dir = DIR_S;
        }
        if idx == 0 && idy == -1 {
            pred_dir = DIR_N;
        }
        if idx == 1 && idy == 1 {
            pred_dir = DIR_SE;
        }
        if idx == -1 && idy == -1 {
            pred_dir = DIR_NW;
        }
        if idx == 1 && idy == -1 {
            pred_dir = DIR_NE;
        }
        if idx == -1 && idy == 1 {
            pred_dir = DIR_SW;
        }

        for _ in 0..200 {
            let mask = valence_mask(cx, cy);
            if valence_count(mask) != 2 {
                break;
            }

            let next_dir = mask ^ pred_dir;

            let (dx, dy, reverse) = match next_dir {
                DIR_N => (0i32, -1i32, DIR_S),
                DIR_NE => (1, -1, DIR_SW),
                DIR_E => (1, 0, DIR_W),
                DIR_SE => (1, 1, DIR_NW),
                DIR_S => (0, 1, DIR_N),
                DIR_SW => (-1, 1, DIR_NE),
                DIR_W => (-1, 0, DIR_E),
                DIR_NW => (-1, -1, DIR_SE),
                _ => return length,
            };

            cx = (cx as i32 + dx) as usize;
            cy = (cy as i32 + dy) as usize;
            pred_dir = reverse;
            length += 1;
        }
        length
    };

    // Simultaneous A/B component labeling in 8x8 window
    let sparse_pixel_sizes = |bx: usize, by: usize| -> (u32, u32) {
        let mut labels = [0i32; 64];
        labels[3 * 8 + 3] = 1;
        labels[3 * 8 + 4] = 2;
        labels[4 * 8 + 3] = 2;
        labels[4 * 8 + 4] = 1;

        let mut size_a = 0u32;
        let mut size_b = 0u32;

        let dcol: [i32; 8] = [-1, -1, -1, 0, 1, 1, 1, 0];
        let drow: [i32; 8] = [-1, 0, 1, 1, 1, 0, -1, -1];
        let dirs: [u32; 8] = [DIR_NW, DIR_W, DIR_SW, DIR_S, DIR_SE, DIR_E, DIR_NE, DIR_N];

        let mut q_col = [0u32; 64];
        let mut q_row = [0u32; 64];
        let mut head = 0usize;
        let mut tail = 0usize;

        q_col[tail] = 3;
        q_row[tail] = 3;
        tail += 1;
        q_col[tail] = 4;
        q_row[tail] = 3;
        tail += 1;
        q_col[tail] = 4;
        q_row[tail] = 4;
        tail += 1;
        q_col[tail] = 3;
        q_row[tail] = 4;
        tail += 1;

        while head < tail {
            let col = q_col[head] as i32;
            let row = q_row[head] as i32;
            head += 1;
            let label = labels[row as usize * 8 + col as usize];

            let px = bx as i32 + col - 3;
            let py = by as i32 + row - 3;
            if px < 0 || py < 0 || px >= img_w as i32 || py >= img_h as i32 {
                continue;
            }

            let vmask = valence_mask(px as usize, py as usize);

            for d in 0..8 {
                if (vmask & dirs[d]) == 0 {
                    continue;
                }
                let nc = col + dcol[d];
                let nr = row + drow[d];
                if nc < 0 || nc >= 8 || nr < 0 || nr >= 8 {
                    continue;
                }
                if labels[nr as usize * 8 + nc as usize] != 0 {
                    continue;
                }

                let npx = bx as i32 + nc - 3;
                let npy = by as i32 + nr - 3;
                if npx < 0 || npy < 0 || npx >= img_w as i32 || npy >= img_h as i32 {
                    continue;
                }

                labels[nr as usize * 8 + nc as usize] = label;
                if label == 1 {
                    size_a += 1;
                } else {
                    size_b += 1;
                }

                if tail < 64 {
                    q_col[tail] = nc as u32;
                    q_row[tail] = nr as u32;
                    tail += 1;
                }
            }
        }

        (size_a, size_b)
    };

    for by in 0..img_h.saturating_sub(1) {
        for bx in 0..img_w.saturating_sub(1) {
            let corner_gx = 2 * bx + 2;
            let corner_gy = 2 * by + 2;
            let flags = g(corner_gx, corner_gy);

            if flags == 0 {
                continue;
            }

            // Fully connected check
            let top = g(corner_gx, corner_gy - 1);
            let bottom = g(corner_gx, corner_gy + 1);
            let left = g(corner_gx - 1, corner_gy);
            let right = g(corner_gx + 1, corner_gy);
            let fully_connected = top != 0 && bottom != 0 && left != 0 && right != 0;

            // Single diagonal in fully-connected block: remove it
            if flags == 1 || flags == 2 {
                if fully_connected {
                    graph_out[corner_gy * graph_stride + corner_gx] = 0;
                }
                continue;
            }

            // flags == 3: crossing (both diagonals)
            if fully_connected {
                graph_out[corner_gy * graph_stride + corner_gx] = 0;
                continue;
            }

            let mut main_vote: i32 = 0;
            let mut anti_vote: i32 = 0;

            // Heuristic 1: Curve length
            let main_len =
                walk_chain(bx, by, bx + 1, by + 1) + walk_chain(bx + 1, by + 1, bx, by);
            let anti_len =
                walk_chain(bx + 1, by, bx, by + 1) + walk_chain(bx, by + 1, bx + 1, by);
            if main_len > anti_len {
                main_vote += (main_len - anti_len) as i32;
            } else if anti_len > main_len {
                anti_vote += (anti_len - main_len) as i32;
            }

            // Heuristic 2: Sparse pixels
            let (main_size, anti_size) = sparse_pixel_sizes(bx, by);
            if main_size < anti_size {
                main_vote += (anti_size - main_size) as i32;
            } else if anti_size < main_size {
                anti_vote += (main_size - anti_size) as i32;
            }

            // Heuristic 3: Islands (valence-1 endpoint)
            let v_main0 = valence_count(valence_mask(bx, by));
            let v_main1 = valence_count(valence_mask(bx + 1, by + 1));
            let v_anti0 = valence_count(valence_mask(bx + 1, by));
            let v_anti1 = valence_count(valence_mask(bx, by + 1));

            if v_main0 == 1 {
                main_vote += 5;
            } else if v_main1 == 1 {
                main_vote += 5;
            } else if v_anti0 == 1 {
                anti_vote += 5;
            } else if v_anti1 == 1 {
                anti_vote += 5;
            }

            // Resolve
            if main_vote > anti_vote {
                graph_out[corner_gy * graph_stride + corner_gx] = 1; // keep main
            } else if anti_vote > main_vote {
                graph_out[corner_gy * graph_stride + corner_gx] = 2; // keep anti
            } else {
                graph_out[corner_gy * graph_stride + corner_gx] = 0; // tie: remove both (per paper)
            }
        }
    }

    graph_out
}

// ============================================================================
// Stage 3: Build cell graph
// ============================================================================

fn build_cell_graph(
    graph: &[u32],
    img_w: usize,
    img_h: usize,
) -> (Vec<f32>, Vec<i32>, Vec<u32>) {
    let graph_stride = 2 * img_w + 1;
    let corners_w = img_w + 1;
    let corners_h = img_h + 1;
    let num_cps = corners_w * corners_h * 2;

    let mut positions = vec![0.0f32; num_cps * 2];
    let mut neighbors = vec![-1i32; num_cps * 4];
    let mut flags = vec![0u32; num_cps];

    let g = |gx: i32, gy: i32| -> u32 {
        let gx = gx.clamp(0, graph_stride as i32 - 1) as usize;
        let gy = gy.clamp(0, (2 * img_h) as i32) as usize;
        graph[gy * graph_stride + gx]
    };

    // Read pixel color from the graph (stored at odd coordinates).
    let px_color = |px: i32, py: i32| -> u32 {
        let px = px.clamp(0, img_w as i32 - 1);
        let py = py.clamp(0, img_h as i32 - 1);
        g(2 * px + 1, 2 * py + 1)
    };

    // Get the two pixel colors separated by a boundary edge at corner (cx,cy) in direction dir.
    let bnd_colors = |cx: i32, cy: i32, dir: i32| -> (u32, u32) {
        match dir {
            0 => (px_color(cx - 1, cy - 1), px_color(cx, cy - 1)),
            1 => (px_color(cx, cy - 1), px_color(cx, cy)),
            2 => (px_color(cx, cy), px_color(cx - 1, cy)),
            3 => (px_color(cx - 1, cy), px_color(cx - 1, cy - 1)),
            _ => (0, 0),
        }
    };

    // Classify an edge as shading (similar colors) per paper Section 3.3.
    // YUV Euclidean distance <= 100/255.
    let is_shading_edge = |ca: u32, cb: u32| -> bool {
        if ca == cb { return true; }
        let dr = ((ca >> 16) & 0xFF) as f32 - ((cb >> 16) & 0xFF) as f32;
        let dg = ((ca >> 8) & 0xFF) as f32 - ((cb >> 8) & 0xFF) as f32;
        let db = (ca & 0xFF) as f32 - (cb & 0xFF) as f32;
        let dy = (0.299 * dr + 0.587 * dg + 0.114 * db) / 255.0;
        let du = 0.493 * (db / 255.0 - dy);
        let dv = 0.877 * (dr / 255.0 - dy);
        let threshold = 100.0 / 255.0;
        (dy * dy + du * du + dv * dv) <= threshold * threshold
    };

    // Select the through-pair at a 3-way T-junction (paper Section 3.3).
    let select_tjunction_pair = |cx: i32, cy: i32, d0: i32, d1: i32, d2: i32| -> (i32, i32) {
        // Step 1: Shading/contour classification.
        let (ca0, cb0) = bnd_colors(cx, cy, d0);
        let (ca1, cb1) = bnd_colors(cx, cy, d1);
        let (ca2, cb2) = bnd_colors(cx, cy, d2);
        let s0 = is_shading_edge(ca0, cb0);
        let s1 = is_shading_edge(ca1, cb1);
        let s2 = is_shading_edge(ca2, cb2);
        let shading_count = s0 as i32 + s1 as i32 + s2 as i32;

        if shading_count == 1 {
            // 1 shading + 2 contour: connect the 2 contour edges.
            if s0 { return (d1, d2); }
            if s1 { return (d0, d2); }
            return (d0, d1);
        }

        // Step 2: Angle-based fallback — connect the pair closest to 180 degrees.
        // At grid corners, opposite pairs (N-S=0^2, E-W=1^3) are 180 degrees;
        // all other pairs are 90 degrees. With 3 of 4 cardinal directions,
        // there is always exactly one opposite pair.
        if (d0 ^ d2) == 2 { (d0, d2) }
        else if (d0 ^ d1) == 2 { (d0, d1) }
        else { (d1, d2) }
    };

    let cp_base = |cx: i32, cy: i32| -> i32 {
        if cx < 0 || cy < 0 || cx >= corners_w as i32 || cy >= corners_h as i32 {
            return -1;
        }
        (cy * corners_w as i32 + cx) * 2
    };

    // Check if two adjacent pixels are similar using the graph.
    let similar = |px0: i32, py0: i32, px1: i32, py1: i32| -> bool {
        let dx = px1 - px0;
        let dy = py1 - py0;
        let gx = 2 * px0 + 1;
        let gy = 2 * py0 + 1;

        if dx == 1 && dy == 0 {
            return g(gx + 1, gy) != 0;
        }
        if dx == -1 && dy == 0 {
            return g(gx - 1, gy) != 0;
        }
        if dx == 0 && dy == 1 {
            return g(gx, gy + 1) != 0;
        }
        if dx == 0 && dy == -1 {
            return g(gx, gy - 1) != 0;
        }
        false
    };

    // Get the correct CP index at a neighbor corner, accounting for diagonal splits.
    let nbr_cp_idx = |cx: i32, cy: i32, from_dir: i32| -> i32 {
        let base = cp_base(cx, cy);
        if base < 0 {
            return -1;
        }

        // Border corners never have diagonals
        if cx <= 0 || cy <= 0 || cx >= img_w as i32 || cy >= img_h as i32 {
            return base;
        }

        let diag = g(2 * cx, 2 * cy);
        let is_ullr = (diag & 1) != 0 && (diag & 2) == 0;
        let is_llur = (diag & 2) != 0 && (diag & 1) == 0;

        if !is_ullr && !is_llur {
            // No resolved diagonal: check for 3/4-way junction needing slot routing.
            // Slot 0 = main pair, slot 1 = stem (3-way) or second pair (4-way).
            let t_bnd_n = g(2 * cx, 2 * cy - 1) == 0;
            let t_bnd_e = g(2 * cx + 1, 2 * cy) == 0;
            let t_bnd_s = g(2 * cx, 2 * cy + 1) == 0;
            let t_bnd_w = g(2 * cx - 1, 2 * cy) == 0;
            let t_count = t_bnd_n as u32 + t_bnd_e as u32 + t_bnd_s as u32 + t_bnd_w as u32;
            if t_count == 4 {
                // Valence-4 crossing: slot 0 = N-S, slot 1 = E-W
                let target_side = from_dir ^ 2;
                if target_side != 0 && target_side != 2 {
                    return base + 1;
                }
            } else if t_count == 3 {
                // Collect the 3 boundary directions and select through-pair
                // using paper's shading/contour + angle heuristic.
                let mut dirs = [0i32; 3];
                let mut di = 0;
                if t_bnd_n { dirs[di] = 0; di += 1; }
                if t_bnd_e { dirs[di] = 1; di += 1; }
                if t_bnd_s { dirs[di] = 2; di += 1; }
                if t_bnd_w { dirs[di] = 3; }
                let (pair0, pair1) = select_tjunction_pair(cx, cy, dirs[0], dirs[1], dirs[2]);
                let target_side = from_dir ^ 2;
                if target_side != pair0 && target_side != pair1 {
                    return base + 1;
                }
            }
            return base;
        }

        let slot;
        if from_dir == 0 {
            slot = if is_llur { 1 } else { 0 };
        } else if from_dir == 1 {
            slot = 0;
        } else if from_dir == 2 {
            slot = if is_ullr { 1 } else { 0 };
        } else {
            slot = 1;
        }

        base + slot
    };

    // Compute the actual CP position at a neighbor grid corner
    let neighbor_cp_pos = |cx: i32, cy: i32, from_dir: i32| -> (f32, f32) {
        let base_x = cx as f32;
        let base_y = cy as f32;

        if cx <= 0 || cy <= 0 || cx >= img_w as i32 || cy >= img_h as i32 {
            return (base_x, base_y);
        }

        let diag = g(2 * cx, 2 * cy);
        let is_ullr = (diag & 1) != 0 && (diag & 2) == 0;
        let is_llur = (diag & 2) != 0 && (diag & 1) == 0;

        if !is_ullr && !is_llur {
            return (base_x, base_y);
        }

        let slot;
        if from_dir == 0 {
            slot = if is_llur { 1 } else { 0 };
        } else if from_dir == 2 {
            slot = if is_ullr { 1 } else { 0 };
        } else if from_dir == 3 {
            slot = 1;
        } else {
            slot = 0;
        }

        if is_ullr {
            if slot == 0 {
                (base_x - 0.25, base_y + 0.25)
            } else {
                (base_x + 0.25, base_y - 0.25)
            }
        } else {
            if slot == 0 {
                (base_x - 0.25, base_y - 0.25)
            } else {
                (base_x + 0.25, base_y + 0.25)
            }
        }
    };

    // Corner detection
    let check_for_corner = |v1: (f32, f32), v2: (f32, f32)| -> bool {
        let len1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt();
        let len2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt();
        if len1 < 1e-6 || len2 < 1e-6 {
            return false;
        }
        let dp = (v1.0 * v2.0 + v1.1 * v2.1) / (len1 * len2);
        if dp > -0.01 && dp < 0.01 {
            return true;
        }
        if dp > -0.72 && dp < -0.69 {
            return true;
        }
        if dp > -0.33 && dp < -0.30 {
            return true;
        }
        false
    };

    // Write CP with prev_dir/next_dir stored in neighbors[2]/[3]
    let write_cp_full = |positions: &mut [f32],
                         neighbors_buf: &mut [i32],
                         flags_buf: &mut [u32],
                         idx: i32,
                         pos: (f32, f32),
                         prev: i32,
                         next: i32,
                         flag: u32,
                         prev_dir: i32,
                         next_dir: i32,
                         _icx: i32,
                         _icy: i32| {
        let i = idx as usize;
        positions[i * 2] = pos.0;
        positions[i * 2 + 1] = pos.1;
        neighbors_buf[i * 4] = prev;
        neighbors_buf[i * 4 + 1] = next;
        neighbors_buf[i * 4 + 2] = prev_dir;
        neighbors_buf[i * 4 + 3] = next_dir;
        // Only mark genuinely-active CPs as endpoints. Default-init slots
        // have flag=0 and prev=next=-1; without the flag!=0 guard they'd
        // get IS_ENDPOINT, overloading the bit's meaning.
        let flag = if flag != 0 && (prev < 0 || next < 0) { flag | IS_ENDPOINT } else { flag };
        flags_buf[i] = flag;
    };

    // Write CP without direction info
    let write_cp = |positions: &mut [f32],
                    neighbors_buf: &mut [i32],
                    flags_buf: &mut [u32],
                    idx: i32,
                    pos: (f32, f32),
                    prev: i32,
                    next: i32,
                    flag: u32| {
        let i = idx as usize;
        // Only mark genuinely-active CPs as endpoints. Default-init slots
        // have flag=0 and prev=next=-1; without the flag!=0 guard they'd
        // get IS_ENDPOINT, overloading the bit's meaning.
        let flag = if flag != 0 && (prev < 0 || next < 0) { flag | IS_ENDPOINT } else { flag };
        positions[i * 2] = pos.0;
        positions[i * 2 + 1] = pos.1;
        neighbors_buf[i * 4] = prev;
        neighbors_buf[i * 4 + 1] = next;
        neighbors_buf[i * 4 + 2] = -1;
        neighbors_buf[i * 4 + 3] = -1;
        flags_buf[i] = flag;
    };

    for cy in 0..corners_h {
        for cx in 0..corners_w {
            let base = cp_base(cx as i32, cy as i32);
            if base < 0 {
                continue;
            }

            // Default: both slots inactive
            write_cp(
                &mut positions,
                &mut neighbors,
                &mut flags,

                base,
                (0.0, 0.0),
                -1,
                -1,
                0,
            );
            write_cp(
                &mut positions,
                &mut neighbors,
                &mut flags,

                base + 1,
                (0.0, 0.0),
                -1,
                -1,
                0,
            );

            let is_interior = cx > 0 && cy > 0 && cx < img_w && cy < img_h;

            if !is_interior {
                // Border CPs: pinned CP that connects back to whichever
                // interior CP references us. Read both slots at the
                // interior neighbor corner and find the one whose prev
                // or next points to our base index.
                let mut has_boundary = false;
                if cy == 0 && cx > 0 && cx < img_w {
                    has_boundary = !similar(cx as i32 - 1, 0, cx as i32, 0);
                }
                if cy == img_h && cx > 0 && cx < img_w {
                    has_boundary = has_boundary || !similar(cx as i32 - 1, img_h as i32 - 1, cx as i32, img_h as i32 - 1);
                }
                if cx == 0 && cy > 0 && cy < img_h {
                    has_boundary = has_boundary || !similar(0, cy as i32 - 1, 0, cy as i32);
                }
                if cx == img_w && cy > 0 && cy < img_h {
                    has_boundary = has_boundary || !similar(img_w as i32 - 1, cy as i32 - 1, img_w as i32 - 1, cy as i32);
                }
                if has_boundary {
                    // Find interior CP via nbr_cp_idx (graph-only, no
                    // race condition, mirrors cell_graph.slang).
                    let mut nbr = -1i32;
                    let mut our_dir = -1i32;
                    // from_dir = direction at OUR (border) corner.
                    // nbr_cp_idx maps it to the opposite side at the neighbor.
                    if cy == img_h {
                        nbr = nbr_cp_idx(cx as i32, cy as i32 - 1, 0); our_dir = 0;
                    } else if cy == 0 {
                        nbr = nbr_cp_idx(cx as i32, cy as i32 + 1, 2); our_dir = 2;
                    } else if cx == img_w {
                        nbr = nbr_cp_idx(cx as i32 - 1, cy as i32, 3); our_dir = 3;
                    } else if cx == 0 {
                        nbr = nbr_cp_idx(cx as i32 + 1, cy as i32, 1); our_dir = 1;
                    }
                    if nbr >= 0 {
                        write_cp_full(&mut positions, &mut neighbors, &mut flags,
                            base, (cx as f32, cy as f32), -1, nbr, 1, -1, our_dir,
                            cx as i32, cy as i32);
                    } else {
                        write_cp(&mut positions, &mut neighbors, &mut flags,
                            base, (cx as f32, cy as f32), -1, -1, 1);
                    }
                }
                continue;
            }

            // Interior corner
            let icx = cx as i32;
            let icy = cy as i32;

            // Diagonal state at this corner
            let diag = g(2 * icx, 2 * icy);
            let has_main = (diag & 1) != 0;
            let has_anti = (diag & 2) != 0;

            // Boundary edges
            let bnd_n = !similar(icx - 1, icy - 1, icx, icy - 1);
            let bnd_e = !similar(icx, icy - 1, icx, icy);
            let bnd_s = !similar(icx - 1, icy, icx, icy);
            let bnd_w = !similar(icx - 1, icy - 1, icx - 1, icy);

            let bnd_count = bnd_n as u32 + bnd_e as u32 + bnd_s as u32 + bnd_w as u32;

            if bnd_count == 0 {
                continue;
            }

            // Handle diagonal crossings: backslash
            if has_main && !has_anti {
                let cp0_has_s = bnd_s;
                let cp0_has_w = bnd_w;
                let cp1_has_n = bnd_n;
                let cp1_has_e = bnd_e;

                let cp0_alive = cp0_has_s || cp0_has_w;
                let cp1_alive = cp1_has_n || cp1_has_e;

                if cp0_alive {
                    let p0 = (cx as f32 - 0.25, cy as f32 + 0.25);
                    let prev0 = if cp0_has_s {
                        nbr_cp_idx(icx, icy + 1, 2)
                    } else {
                        -1
                    };
                    let next0 = if cp0_has_w {
                        nbr_cp_idx(icx - 1, icy, 3)
                    } else {
                        -1
                    };
                    let cp0_junction = !(cp0_has_s && cp0_has_w);
                    let mut cp0_flag = if cp0_junction { 1u32 } else { 2u32 };
                    if !cp0_junction {
                        let s_pos = neighbor_cp_pos(icx, icy + 1, 2);
                        let w_pos = neighbor_cp_pos(icx - 1, icy, 3);
                        if check_for_corner(
                            (s_pos.0 - p0.0, s_pos.1 - p0.1),
                            (w_pos.0 - p0.0, w_pos.1 - p0.1),
                        ) {
                            cp0_flag |= IS_CORNER;
                        }
                    }
                    let d0_prev = if cp0_has_s { 2 } else { -1 };
                    let d0_next = if cp0_has_w { 3 } else { -1 };
                    write_cp_full(
                        &mut positions,
                        &mut neighbors,
                        &mut flags,
        
                        base,
                        p0,
                        prev0,
                        next0,
                        cp0_flag,
                        d0_prev,
                        d0_next,
                        icx,
                        icy,
                    );
                }

                if cp1_alive {
                    let p1 = (cx as f32 + 0.25, cy as f32 - 0.25);
                    let prev1 = if cp1_has_n {
                        nbr_cp_idx(icx, icy - 1, 0)
                    } else {
                        -1
                    };
                    let next1 = if cp1_has_e {
                        nbr_cp_idx(icx + 1, icy, 1)
                    } else {
                        -1
                    };
                    let cp1_junction = !(cp1_has_n && cp1_has_e);
                    let mut cp1_flag = if cp1_junction { 1u32 } else { 2u32 };
                    if !cp1_junction {
                        let n_pos = neighbor_cp_pos(icx, icy - 1, 0);
                        let e_pos = neighbor_cp_pos(icx + 1, icy, 1);
                        if check_for_corner(
                            (n_pos.0 - p1.0, n_pos.1 - p1.1),
                            (e_pos.0 - p1.0, e_pos.1 - p1.1),
                        ) {
                            cp1_flag |= IS_CORNER;
                        }
                    }
                    let d1_prev = if cp1_has_n { 0 } else { -1 };
                    let d1_next = if cp1_has_e { 1 } else { -1 };
                    write_cp_full(
                        &mut positions,
                        &mut neighbors,
                        &mut flags,
        
                        base + 1,
                        p1,
                        prev1,
                        next1,
                        cp1_flag,
                        d1_prev,
                        d1_next,
                        icx,
                        icy,
                    );
                }
                continue;
            }

            // Handle diagonal crossings: slash
            if has_anti && !has_main {
                let cp0_has_n = bnd_n;
                let cp0_has_w = bnd_w;
                let cp1_has_s = bnd_s;
                let cp1_has_e = bnd_e;

                let cp0_alive = cp0_has_n || cp0_has_w;
                let cp1_alive = cp1_has_s || cp1_has_e;

                if cp0_alive {
                    let p0 = (cx as f32 - 0.25, cy as f32 - 0.25);
                    let prev0 = if cp0_has_n {
                        nbr_cp_idx(icx, icy - 1, 0)
                    } else {
                        -1
                    };
                    let next0 = if cp0_has_w {
                        nbr_cp_idx(icx - 1, icy, 3)
                    } else {
                        -1
                    };
                    let cp0_junction = !(cp0_has_n && cp0_has_w);
                    let mut cp0_flag = if cp0_junction { 1u32 } else { 2u32 };
                    if !cp0_junction {
                        let n_pos = neighbor_cp_pos(icx, icy - 1, 0);
                        let w_pos = neighbor_cp_pos(icx - 1, icy, 3);
                        if check_for_corner(
                            (n_pos.0 - p0.0, n_pos.1 - p0.1),
                            (w_pos.0 - p0.0, w_pos.1 - p0.1),
                        ) {
                            cp0_flag |= IS_CORNER;
                        }
                    }
                    let d0_prev = if cp0_has_n { 0 } else { -1 };
                    let d0_next = if cp0_has_w { 3 } else { -1 };
                    write_cp_full(
                        &mut positions,
                        &mut neighbors,
                        &mut flags,
        
                        base,
                        p0,
                        prev0,
                        next0,
                        cp0_flag,
                        d0_prev,
                        d0_next,
                        icx,
                        icy,
                    );
                }

                if cp1_alive {
                    let p1 = (cx as f32 + 0.25, cy as f32 + 0.25);
                    let prev1 = if cp1_has_s {
                        nbr_cp_idx(icx, icy + 1, 2)
                    } else {
                        -1
                    };
                    let next1 = if cp1_has_e {
                        nbr_cp_idx(icx + 1, icy, 1)
                    } else {
                        -1
                    };
                    let cp1_junction = !(cp1_has_s && cp1_has_e);
                    let mut cp1_flag = if cp1_junction { 1u32 } else { 2u32 };
                    if !cp1_junction {
                        let s_pos = neighbor_cp_pos(icx, icy + 1, 2);
                        let e_pos = neighbor_cp_pos(icx + 1, icy, 1);
                        if check_for_corner(
                            (s_pos.0 - p1.0, s_pos.1 - p1.1),
                            (e_pos.0 - p1.0, e_pos.1 - p1.1),
                        ) {
                            cp1_flag |= IS_CORNER;
                        }
                    }
                    let d1_prev = if cp1_has_s { 2 } else { -1 };
                    let d1_next = if cp1_has_e { 1 } else { -1 };
                    write_cp_full(
                        &mut positions,
                        &mut neighbors,
                        &mut flags,
        
                        base + 1,
                        p1,
                        prev1,
                        next1,
                        cp1_flag,
                        d1_prev,
                        d1_next,
                        icx,
                        icy,
                    );
                }
                continue;
            }

            // No diagonal (or both kept): single CP
            let mut pos = (cx as f32, cy as f32);

            let n_idx = if bnd_n { nbr_cp_idx(icx, icy - 1, 0) } else { -1 };
            let e_idx = if bnd_e { nbr_cp_idx(icx + 1, icy, 1) } else { -1 };
            let s_idx = if bnd_s { nbr_cp_idx(icx, icy + 1, 2) } else { -1 };
            let w_idx = if bnd_w { nbr_cp_idx(icx - 1, icy, 3) } else { -1 };

            if bnd_count == 2 {
                // Regular chain node
                let mut prev = -1i32;
                let mut next = -1i32;
                let mut prev_dir = -1i32;
                let mut next_dir = -1i32;

                if bnd_n {
                    if prev < 0 {
                        prev = n_idx;
                        prev_dir = 0;
                    } else {
                        next = n_idx;
                        next_dir = 0;
                    }
                }
                if bnd_e {
                    if prev < 0 {
                        prev = e_idx;
                        prev_dir = 1;
                    } else {
                        next = e_idx;
                        next_dir = 1;
                    }
                }
                if bnd_s {
                    if prev < 0 {
                        prev = s_idx;
                        prev_dir = 2;
                    } else {
                        next = s_idx;
                        next_dir = 2;
                    }
                }
                if bnd_w {
                    if prev < 0 {
                        prev = w_idx;
                        prev_dir = 3;
                    } else {
                        next = w_idx;
                        next_dir = 3;
                    }
                }

                let mut flag = 2u32;
                if prev_dir >= 0 && next_dir >= 0 {
                    let prev_cx =
                        icx + if prev_dir == 1 { 1 } else if prev_dir == 3 { -1 } else { 0 };
                    let prev_cy =
                        icy + if prev_dir == 2 { 1 } else if prev_dir == 0 { -1 } else { 0 };
                    let next_cx =
                        icx + if next_dir == 1 { 1 } else if next_dir == 3 { -1 } else { 0 };
                    let next_cy =
                        icy + if next_dir == 2 { 1 } else if next_dir == 0 { -1 } else { 0 };
                    let prev_pos = neighbor_cp_pos(prev_cx, prev_cy, prev_dir);
                    let next_pos = neighbor_cp_pos(next_cx, next_cy, next_dir);
                    if check_for_corner(
                        (prev_pos.0 - pos.0, prev_pos.1 - pos.1),
                        (next_pos.0 - pos.0, next_pos.1 - pos.1),
                    ) {
                        flag |= IS_CORNER;
                    }
                }
                write_cp_full(
                    &mut positions,
                    &mut neighbors,
                    &mut flags,
    
                    base,
                    pos,
                    prev,
                    next,
                    flag,
                    prev_dir,
                    next_dir,
                    icx,
                    icy,
                );
            } else if bnd_count == 1 {
                // Valence-1 endpoint -- pinned junction
                let (nbr, nbr_dir);
                if bnd_n {
                    nbr = n_idx;
                    nbr_dir = 0;
                } else if bnd_e {
                    nbr = e_idx;
                    nbr_dir = 1;
                } else if bnd_s {
                    nbr = s_idx;
                    nbr_dir = 2;
                } else {
                    nbr = w_idx;
                    nbr_dir = 3;
                }
                write_cp_full(
                    &mut positions,
                    &mut neighbors,
                    &mut flags,
    
                    base,
                    pos,
                    nbr,
                    -1,
                    1,
                    nbr_dir,
                    -1,
                    icx,
                    icy,
                );
            } else if bnd_count == 3 {
                // Paper Section 3.3: select through-pair via shading/contour + angle heuristic.
                let mut dirs = [0i32; 3];
                let mut di = 0;
                if bnd_n { dirs[di] = 0; di += 1; }
                if bnd_e { dirs[di] = 1; di += 1; }
                if bnd_s { dirs[di] = 2; di += 1; }
                if bnd_w { dirs[di] = 3; }
                let (t_prev_dir, t_next_dir) = select_tjunction_pair(icx, icy, dirs[0], dirs[1], dirs[2]);

                let idx_arr = [n_idx, e_idx, s_idx, w_idx];
                let prev = idx_arr[t_prev_dir as usize];
                let next = idx_arr[t_next_dir as usize];

                // T-junction position correction
                if prev >= 0 && next >= 0 {
                    let prev_slot = prev / 2;
                    let prev_cx = prev_slot % corners_w as i32;
                    let prev_cy = prev_slot / corners_w as i32;
                    let next_slot = next / 2;
                    let next_cx = next_slot % corners_w as i32;
                    let next_cy = next_slot / corners_w as i32;
                    pos = (
                        0.125 * prev_cx as f32 + 0.75 * cx as f32 + 0.125 * next_cx as f32,
                        0.125 * prev_cy as f32 + 0.75 * cy as f32 + 0.125 * next_cy as f32,
                    );
                }

                write_cp_full(
                    &mut positions,
                    &mut neighbors,
                    &mut flags,
    
                    base,
                    pos,
                    prev,
                    next,
                    2 | IS_TJUNCTION,
                    t_prev_dir,
                    t_next_dir,
                    icx,
                    icy,
                );

                // Create endpoint CP at slot 1 for the 3rd (dropped) boundary direction
                let mut stem_idx = -1i32;
                let mut stem_dir = -1i32;
                if bnd_n && t_prev_dir != 0 && t_next_dir != 0 {
                    stem_idx = n_idx;
                    stem_dir = 0;
                } else if bnd_e && t_prev_dir != 1 && t_next_dir != 1 {
                    stem_idx = e_idx;
                    stem_dir = 1;
                } else if bnd_s && t_prev_dir != 2 && t_next_dir != 2 {
                    stem_idx = s_idx;
                    stem_dir = 2;
                } else if bnd_w && t_prev_dir != 3 && t_next_dir != 3 {
                    stem_idx = w_idx;
                    stem_dir = 3;
                }
                if stem_idx >= 0 {
                    write_cp_full(
                        &mut positions,
                        &mut neighbors,
                        &mut flags,
        
                        base + 1,
                        pos,
                        stem_idx,
                        -1,
                        1,
                        stem_dir,
                        -1,
                        icx,
                        icy,
                    );
                }
            } else {
                // Valence 4 cross-junction: inverse-correct both CPs so their
                // B-spline curves pass through the grid corner at t=0.5.
                write_cp_full(
                    &mut positions,
                    &mut neighbors,
                    &mut flags,
                    base,
                    pos,
                    n_idx,
                    s_idx,
                    2 | IS_CROSSING,
                    0,
                    2,
                    icx,
                    icy,
                );
                write_cp_full(
                    &mut positions,
                    &mut neighbors,
                    &mut flags,
                    base + 1,
                    pos,
                    e_idx,
                    w_idx,
                    2 | IS_CROSSING,
                    1,
                    3,
                    icx,
                    icy,
                );
            }
        }
    }

    // Fix border CP connectivity: interior CPs already reference border
    // CPs as neighbors, but border CPs don't link back. Make the connection
    // reciprocal so border CPs produce proper chain-endpoint B-spline edges
    // that extend to the image border. Each linked border CP still has its
    // other neighbor at -1, so it remains a chain endpoint and IS_ENDPOINT
    // stays correct after the link.
    // Mask IS_ENDPOINT before testing for "plain" CPs: every isolated border
    // CP carries that bit, so a bare `flags[pi] == 1` check would skip the
    // candidates this fixup is supposed to rescue.
    for i in 0..num_cps {
        if flags[i] == 0 { continue; }
        let prev = neighbors[i * 4];
        let next = neighbors[i * 4 + 1];
        if prev >= 0 {
            let pi = prev as usize;
            if (flags[pi] & !IS_ENDPOINT) == 1
                && neighbors[pi * 4] < 0
                && neighbors[pi * 4 + 1] < 0
            {
                neighbors[pi * 4 + 1] = i as i32;
                let d = neighbors[i * 4 + 2];
                if d >= 0 { neighbors[pi * 4 + 3] = (d + 2) % 4; }
            }
        }
        if next >= 0 {
            let ni = next as usize;
            if (flags[ni] & !IS_ENDPOINT) == 1
                && neighbors[ni * 4] < 0
                && neighbors[ni * 4 + 1] < 0
            {
                neighbors[ni * 4] = i as i32;
                let d = neighbors[i * 4 + 3];
                if d >= 0 { neighbors[ni * 4 + 2] = (d + 2) % 4; }
            }
        }
    }

    (positions, neighbors, flags)
}

// ============================================================================
// Stage 4: Optimize energy (2D Newton-Raphson)
// ============================================================================

fn optimize_energy(
    positions: &[f32],
    orig_positions: &[f32],
    neighbors: &[i32],
    flags: &[u32],
    num_cps: usize,
) -> Vec<f32> {
    let positional_scale: f32 = 2.5;
    let s4 = positional_scale * positional_scale * positional_scale * positional_scale;

    let read_pos = |buf: &[f32], i: usize| -> (f32, f32) {
        (buf[i * 2], buf[i * 2 + 1])
    };

    let read_neighbor_pos = |buf: &[f32], i: usize| -> (f32, f32) {
        if (flags[i] & IS_TJUNCTION) != 0 {
            let ci = i + 1;
            if ci < num_cps {
                let p = (buf[ci * 2], buf[ci * 2 + 1]);
                if p.0 != 0.0 || p.1 != 0.0 {
                    return p;
                }
            }
        }
        (buf[i * 2], buf[i * 2 + 1])
    };

    let optimize_one_pass = |pos_in: &[f32], pos_out: &mut [f32]| {
        for i in 0..num_cps {
            let mut p = read_pos(pos_in, i);
            pos_out[i * 2] = p.0;
            pos_out[i * 2 + 1] = p.1;

            let f = flags[i];

            // Pinned nodes don't move
            if (f & 1) != 0 {
                continue;
            }

            let prev_idx = neighbors[i * 4];
            let next_idx = neighbors[i * 4 + 1];
            if prev_idx < 0 || next_idx < 0 {
                continue;
            }

            // Crossings: only process from slot 0 (even index) to avoid double-processing.
            // The even slot writes the result to both slots AFTER optimization.
            // Revert the initial copy that clobbered the even slot's write.
            let is_crossing = (f & IS_CROSSING) != 0;
            if is_crossing && (i & 1) != 0 {
                let even = i - 1;
                pos_out[i * 2] = pos_out[even * 2];
                pos_out[i * 2 + 1] = pos_out[even * 2 + 1];
                continue;
            }

            let n0 = read_neighbor_pos(pos_in, prev_idx as usize);
            let n1 = read_neighbor_pos(pos_in, next_idx as usize);
            let p_orig = read_pos(orig_positions, i);

            // For crossings, gather the other pair's neighbors
            let (n2, n3) = if is_crossing && i + 1 < num_cps {
                let other_prev = neighbors[(i + 1) * 4];
                let other_next = neighbors[(i + 1) * 4 + 1];
                let n2 = if other_prev >= 0 { read_neighbor_pos(pos_in, other_prev as usize) } else { (0.0, 0.0) };
                let n3 = if other_next >= 0 { read_neighbor_pos(pos_in, other_next as usize) } else { (0.0, 0.0) };
                (n2, n3)
            } else {
                ((0.0, 0.0), (0.0, 0.0))
            };

            // Corners: exclude curvature energy, keep positional energy only.
            let exclude_curvature = (f & IS_CORNER) != 0;

            for _ in 0..NEWTON_ITER {
                let dx = p.0 - p_orig.0;
                let dy = p.1 - p_orig.1;
                let d2 = dx * dx + dy * dy;

                let (gx, gy) = if exclude_curvature {
                    (4.0 * s4 * d2 * dx, 4.0 * s4 * d2 * dy)
                } else {
                    let mut cx = 4.0 * (2.0 * p.0 - n0.0 - n1.0);
                    let mut cy = 4.0 * (2.0 * p.1 - n0.1 - n1.1);
                    if is_crossing {
                        cx += 4.0 * (2.0 * p.0 - n2.0 - n3.0);
                        cy += 4.0 * (2.0 * p.1 - n2.1 - n3.1);
                    }
                    (cx + 4.0 * s4 * d2 * dx, cy + 4.0 * s4 * d2 * dy)
                };

                let curv_h: f32 = if exclude_curvature { 0.0 } else if is_crossing { 16.0 } else { 8.0 };
                let h00 = curv_h + 4.0 * s4 * (2.0 * dx * dx + d2);
                let h11 = curv_h + 4.0 * s4 * (2.0 * dy * dy + d2);
                let h01 = 8.0 * s4 * dx * dy;

                let det = h00 * h11 - h01 * h01;
                if det.abs() < 1e-20 {
                    break;
                }

                let inv_det = 1.0 / det;
                p.0 -= (h11 * gx - h01 * gy) * inv_det;
                p.1 -= (-h01 * gx + h00 * gy) * inv_det;
            }

            pos_out[i * 2] = p.0;
            pos_out[i * 2 + 1] = p.1;
            // Write same position to both crossing slots
            if is_crossing && i + 1 < num_cps {
                pos_out[(i + 1) * 2] = p.0;
                pos_out[(i + 1) * 2 + 1] = p.1;
            }
        }
    };

    // Pass 1: positions -> opt_out
    let mut opt_out = vec![0.0f32; num_cps * 2];
    optimize_one_pass(positions, &mut opt_out);

    // Pass 2: opt_out -> final
    let mut final_pos = vec![0.0f32; num_cps * 2];
    optimize_one_pass(&opt_out, &mut final_pos);

    final_pos
}

// ============================================================================
// Stage 5a: B-spline curve intersection (used by Stage 5b crossing cache)
// ============================================================================

/// Real roots of c4·t⁴ + c3·t³ + c2·t² + c1·t + c0 = 0 in [0, 1].
/// Ferrari's method via the resolvent cubic. Returns at most 4 roots.
fn solve_quartic_in_unit(c4: f32, c3: f32, c2: f32, c1: f32, c0: f32) -> [Option<f32>; 4] {
    let mut out: [Option<f32>; 4] = [None; 4];

    // Degeneracy detection — relative to the other coefficients, not an
    // absolute threshold. Two B-spline spans whose second-derivatives are
    // parallel (e.g. both ~diagonal) produce u2 ≈ 0, hence c4 ≈ 1e-16
    // while c0..c2 sit at ~1e-2. An absolute `1e-20` cutoff misses this.
    // 1e-6 because dividing by a coefficient that small still amplifies
    // noise enough to swamp legitimate roots.
    let scale = c0.abs().max(c1.abs()).max(c2.abs()).max(c3.abs()).max(c4.abs());
    let eps = 1e-6 * scale.max(1.0);
    if c4.abs() < eps {
        return solve_cubic_in_unit(c3, c2, c1, c0);
    }

    // Normalize: t⁴ + a·t³ + b·t² + c·t + d = 0
    let a = c3 / c4;
    let b = c2 / c4;
    let c = c1 / c4;
    let d = c0 / c4;

    // Depress via t = u - a/4. Result: u⁴ + p·u² + q·u + r = 0
    let a2 = a * a;
    let p = b - 3.0 * a2 / 8.0;
    let q = c - a * b / 2.0 + a2 * a / 8.0;
    let r = d - a * c / 4.0 + a2 * b / 16.0 - 3.0 * a2 * a2 / 256.0;
    let shift = -a / 4.0;

    // Biquadratic special case: u⁴ + p·u² + r = 0.
    if q.abs() < 1e-12 {
        let disc = p * p - 4.0 * r;
        if disc < 0.0 {
            return out;
        }
        let sq = disc.sqrt();
        let u2_a = (-p + sq) / 2.0;
        let u2_b = (-p - sq) / 2.0;
        let mut idx = 0;
        for u2 in [u2_a, u2_b] {
            if u2 >= 0.0 {
                let u = u2.sqrt();
                for u_signed in [u, -u] {
                    let t = u_signed + shift;
                    if (0.0..=1.0).contains(&t) && idx < 4 {
                        out[idx] = Some(t);
                        idx += 1;
                    }
                }
            }
        }
        return out;
    }

    // Resolvent cubic: 8·y³ + 8·p·y² + (2·p² − 8·r)·y − q² = 0.
    // Pick any real root y0; use it to factor the quartic into two quadratics.
    let cy = solve_cubic_real_root(8.0, 8.0 * p, 2.0 * p * p - 8.0 * r, -q * q);

    // Two quadratics: u² ± √(2y) · u + (p/2 + y ∓ q/(2√(2y))) = 0
    let two_y = 2.0 * cy;
    if two_y < 0.0 {
        return out;
    }
    let m = two_y.sqrt();
    if m.abs() < 1e-12 {
        return out;
    }
    let q_over_2m = q / (2.0 * m);
    let half_p_plus_y = p / 2.0 + cy;

    let mut idx = 0;
    for (sign_m, sign_q) in [(1.0_f32, -1.0_f32), (-1.0, 1.0)] {
        // u² + (sign_m · m) · u + (half_p_plus_y + sign_q · q_over_2m) = 0
        let bb = sign_m * m;
        let cc = half_p_plus_y + sign_q * q_over_2m;
        let disc = bb * bb - 4.0 * cc;
        if disc < 0.0 {
            continue;
        }
        let sq = disc.sqrt();
        for u_root in [(-bb + sq) / 2.0, (-bb - sq) / 2.0] {
            let t = u_root + shift;
            if (0.0..=1.0).contains(&t) && idx < 4 {
                out[idx] = Some(t);
                idx += 1;
            }
        }
    }
    out
}

/// Solve c3·t³ + c2·t² + c1·t + c0 = 0 in [0,1]; returns at most 3 roots.
fn solve_cubic_in_unit(c3: f32, c2: f32, c1: f32, c0: f32) -> [Option<f32>; 4] {
    let mut out = [None; 4];
    let scale = c0.abs().max(c1.abs()).max(c2.abs()).max(c3.abs());
    let eps = 1e-6 * scale.max(1.0);
    if c3.abs() < eps {
        // Quadratic
        if c2.abs() < eps {
            if c1.abs() > eps {
                let t = -c0 / c1;
                if (0.0..=1.0).contains(&t) {
                    out[0] = Some(t);
                }
            }
            return out;
        }
        let disc = c1 * c1 - 4.0 * c2 * c0;
        if disc < 0.0 {
            return out;
        }
        let sq = disc.sqrt();
        let mut idx = 0;
        for t in [(-c1 + sq) / (2.0 * c2), (-c1 - sq) / (2.0 * c2)] {
            if (0.0..=1.0).contains(&t) {
                out[idx] = Some(t);
                idx += 1;
            }
        }
        return out;
    }

    // Cardano. Normalize to t³ + a·t² + b·t + c = 0, depress with t = u - a/3.
    let a = c2 / c3;
    let b = c1 / c3;
    let cc = c0 / c3;
    let p = b - a * a / 3.0;
    let q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + cc;
    let shift = -a / 3.0;
    let disc = q * q / 4.0 + p * p * p / 27.0;

    if disc > 0.0 {
        let sq = disc.sqrt();
        let u1 = (-q / 2.0 + sq).cbrt();
        let u2 = (-q / 2.0 - sq).cbrt();
        let t = u1 + u2 + shift;
        if (0.0..=1.0).contains(&t) {
            out[0] = Some(t);
        }
    } else {
        // disc <= 0: three real roots via trigonometric form.
        let r = (-p / 3.0).max(0.0).sqrt();
        let cos_arg = if r > 1e-20 { (-q / 2.0) / (r * r * r) } else { 0.0 };
        let theta = cos_arg.clamp(-1.0, 1.0).acos();
        for k in 0..3 {
            let t = 2.0 * r * ((theta + 2.0 * std::f32::consts::PI * k as f32) / 3.0).cos() + shift;
            if (0.0..=1.0).contains(&t) {
                out[k] = Some(t);
            }
        }
    }
    out
}

/// Returns one real root of c3·y³ + c2·y² + c1·y + c0 = 0. Used by the
/// quartic resolvent — we only need a single real root, which is guaranteed
/// to exist for any real cubic.
fn solve_cubic_real_root(c3: f32, c2: f32, c1: f32, c0: f32) -> f32 {
    let a = c2 / c3;
    let b = c1 / c3;
    let cc = c0 / c3;
    let p = b - a * a / 3.0;
    let q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + cc;
    let shift = -a / 3.0;
    let disc = q * q / 4.0 + p * p * p / 27.0;

    if disc >= 0.0 {
        let sq = disc.sqrt();
        let u1 = (-q / 2.0 + sq).cbrt();
        let u2 = (-q / 2.0 - sq).cbrt();
        u1 + u2 + shift
    } else {
        let r = (-p / 3.0).sqrt();
        let theta = ((-q / 2.0) / (r * r * r)).clamp(-1.0, 1.0).acos();
        2.0 * r * (theta / 3.0).cos() + shift
    }
}

/// Result of intersecting two quadratic uniform B-spline spans.
#[derive(Clone, Copy, Debug)]
pub struct CurveIntersect {
    pub p: (f32, f32),
    pub t_a: f32,
    pub t_b: f32,
}

/// Quadratic uniform B-spline span coefficients: B(t) = a·t² + b·t + c.
/// Standard formula B(t) = 0.5(1-t)²·P0 + ((1-t)t + 0.5)·P1 + 0.5·t²·P2 expands to:
///   a = 0.5·P0 − P1 + 0.5·P2
///   b = −P0 + P1
///   c = 0.5·P0 + 0.5·P1
#[inline]
fn bspline_poly(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32))
    -> ((f32, f32), (f32, f32), (f32, f32))
{
    let a = (0.5 * p0.0 - p1.0 + 0.5 * p2.0, 0.5 * p0.1 - p1.1 + 0.5 * p2.1);
    let b = (-p0.0 + p1.0, -p0.1 + p1.1);
    let c = (0.5 * p0.0 + 0.5 * p1.0, 0.5 * p0.1 + 0.5 * p1.1);
    (a, b, c)
}

/// Intersect two quadratic uniform B-spline spans. The pipeline guarantees
/// they cross within `t,s ∈ [0,1]` at a 4-way crossing, so we pick the
/// in-unit root closest to (0.5, 0.5).
pub fn intersect_quadratic_bsplines(
    a_p0: (f32, f32), a_p1: (f32, f32), a_p2: (f32, f32),
    b_p0: (f32, f32), b_p1: (f32, f32), b_p2: (f32, f32),
) -> CurveIntersect {
    let (aa, ba, ca) = bspline_poly(a_p0, a_p1, a_p2);
    let (ab, bb, cb) = bspline_poly(b_p0, b_p1, b_p2);

    // f(t) = aa.x·t² + ba.x·t + ca.x  matched against  g(s) = ab.x·s² + bb.x·s + cb.x  (same for y).
    // Eliminate s via the resultant of two quadratics in s. With
    //   A = -ab.x, B = -bb.x, C = f(t) - cb.x
    //   D = -ab.y, E = -bb.y, F = h(t) - cb.y
    // Res(p,q) = (AF - CD)² - (AE - BD)·(BF - CE)
    // U(t) = AF − CD = ab.y·f(t) − ab.x·h(t) + ab.x·cb.y − ab.y·cb.x   (deg 2)
    // V(t) = BF − CE = bb.y·f(t) − bb.x·h(t) + bb.x·cb.y − bb.y·cb.x   (deg 2)
    // M    = AE − BD = ab.x·bb.y − bb.x·ab.y                           (constant)
    // R(t) = U(t)² − M·V(t) = quartic in t.
    let m = ab.0 * bb.1 - bb.0 * ab.1;

    let u2 = ab.1 * aa.0 - ab.0 * aa.1;
    let u1 = ab.1 * ba.0 - ab.0 * ba.1;
    let u0 = ab.1 * ca.0 - ab.0 * ca.1 + ab.0 * cb.1 - ab.1 * cb.0;

    let v2 = bb.1 * aa.0 - bb.0 * aa.1;
    let v1 = bb.1 * ba.0 - bb.0 * ba.1;
    let v0 = bb.1 * ca.0 - bb.0 * ca.1 + bb.0 * cb.1 - bb.1 * cb.0;

    // U² − M·V coefficients:
    let r4 = u2 * u2;
    let r3 = 2.0 * u1 * u2;
    let r2 = u1 * u1 + 2.0 * u0 * u2 - m * v2;
    let r1 = 2.0 * u0 * u1 - m * v1;
    let r0 = u0 * u0 - m * v0;

    let roots = solve_quartic_in_unit(r4, r3, r2, r1, r0);

    // Pick the root closest to t=0.5 (the optimizer keeps crossings near the grid corner).
    let mut best_t = 0.5f32;
    let mut best_dist = f32::INFINITY;
    for r in roots.iter().flatten() {
        let d = (*r - 0.5).abs();
        if d < best_dist {
            best_dist = d;
            best_t = *r;
        }
    }
    let t = best_t;

    // Intersection point from curve A.
    let p = (
        aa.0 * t * t + ba.0 * t + ca.0,
        aa.1 * t * t + ba.1 * t + ca.1,
    );

    // Find s on curve B such that B_b(s) = p. Two quadratics (one per axis);
    // pick the [0,1] root closest to 0.5 from whichever axis has a non-degenerate
    // leading coefficient.
    let s = solve_b_param_for_point(ab, bb, cb, p);

    CurveIntersect { p, t_a: t, t_b: s }
}

/// Given B(s) = a·s² + b·s + c (component-wise) and a target point `target`,
/// find s ∈ [0,1] satisfying both axes (any consistent root). Picks the
/// in-range root closest to 0.5 from the better-conditioned axis.
fn solve_b_param_for_point(
    a: (f32, f32), b: (f32, f32), c: (f32, f32), target: (f32, f32),
) -> f32 {
    fn roots_axis(aa: f32, bb: f32, cc: f32, t: f32) -> [Option<f32>; 2] {
        // aa·s² + bb·s + (cc − t) = 0
        let mut out = [None; 2];
        let kk = cc - t;
        if aa.abs() < 1e-12 {
            if bb.abs() > 1e-12 {
                let s = -kk / bb;
                if (0.0..=1.0).contains(&s) {
                    out[0] = Some(s);
                }
            }
            return out;
        }
        let disc = bb * bb - 4.0 * aa * kk;
        if disc < 0.0 {
            return out;
        }
        let sq = disc.sqrt();
        let mut idx = 0;
        for s in [(-bb + sq) / (2.0 * aa), (-bb - sq) / (2.0 * aa)] {
            if (0.0..=1.0).contains(&s) {
                out[idx] = Some(s);
                idx += 1;
            }
        }
        out
    }

    // Prefer the axis with the larger leading coefficient (better conditioning).
    let prefer_x = a.0.abs() >= a.1.abs();
    let primary = if prefer_x {
        roots_axis(a.0, b.0, c.0, target.0)
    } else {
        roots_axis(a.1, b.1, c.1, target.1)
    };

    let mut best = 0.5f32;
    let mut best_dist = f32::INFINITY;
    for s in primary.iter().flatten() {
        let d = (*s - 0.5).abs();
        if d < best_dist {
            best_dist = d;
            best = *s;
        }
    }
    best
}

// ============================================================================
// Stage 5: Update T-junctions
// ============================================================================

fn update_tjunctions(
    positions: &mut [f32],
    crossing_t: &mut [f32],
    neighbors: &[i32],
    flags: &[u32],
    num_cps: usize,
) {
    // Two jobs, ordered so each sees the other's input in its final form:
    //   - Phase 1 (IS_TJUNCTION): stem-snap onto the rendered through-curve
    //     via the ghost-aware algebraic B(0.5) formula. 3 passes to converge
    //     (contraction ≈ 0.17/pass). Writes only stem CP positions.
    //   - Phase 2 (IS_CROSSING, slot 0 only): solve the curve-curve
    //     intersection of the N-S and E-W spans, write `(t_ns, t_ew)` into
    //     `crossing_t`. Must run *after* Phase 1 because a crossing's
    //     N/S/E/W neighbor can be a stem CP that gets repositioned by Phase 1
    //     (e.g., when an adjacent T-junction's through curve clamps onto the
    //     crossing). Reading stale stem positions would shift the
    //     intersection enough to flip wedge classification on near-boundary
    //     pixels.

    let read_pos = |positions: &[f32], ci: usize| -> (f32, f32) {
        (positions[ci * 2], positions[ci * 2 + 1])
    };
    let is_end = |idx: i32| -> bool {
        idx >= 0 && (flags[idx as usize] & IS_ENDPOINT) != 0
    };

    // Phase 1: T-junction stem snap. 3 passes for convergence.
    for _ in 0..3 {
        for i in 0..num_cps {
            let f = flags[i];
            if (f & IS_TJUNCTION) == 0 { continue; }
            let prev_idx = neighbors[i * 4];
            let next_idx = neighbors[i * 4 + 1];
            if prev_idx < 0 || next_idx < 0 { continue; }

            let prev_pos = read_pos(positions, prev_idx as usize);
            let next_pos = read_pos(positions, next_idx as usize);
            let prev_is_end = is_end(prev_idx);
            let next_is_end = is_end(next_idx);
            let through = read_pos(positions, i);
            let stem = i ^ 1;
            if stem < num_cps && (flags[stem] & !IS_ENDPOINT) == 1 {
                let (sp, st, sn) = match (prev_is_end, next_is_end) {
                    (false, false) => (0.125, 0.75,  0.125),
                    (true,  false) => (0.25,  0.625, 0.125),
                    (false, true ) => (0.125, 0.625, 0.25),
                    (true,  true ) => (0.25,  0.5,   0.25),
                };
                positions[stem * 2]     = sp * prev_pos.0 + st * through.0 + sn * next_pos.0;
                positions[stem * 2 + 1] = sp * prev_pos.1 + st * through.1 + sn * next_pos.1;
            }
        }
    }

    // Phase 2: write t values for every crossing. Runs after Phase 1 so any
    // stem-CP neighbor reads see the snapped position.
    for i in 0..num_cps {
        if (flags[i] & IS_CROSSING) == 0 || (i & 1) != 0 { continue; }
        let other = i + 1;
        if other >= num_cps { continue; }

        let n_idx = neighbors[i * 4];
        let s_idx = neighbors[i * 4 + 1];
        let e_idx = neighbors[other * 4];
        let w_idx = neighbors[other * 4 + 1];
        if n_idx < 0 || s_idx < 0 || e_idx < 0 || w_idx < 0 { continue; }

        let cp_a = read_pos(positions, i);
        let cp_b = read_pos(positions, other);
        let ghost = |np: (f32, f32), is_endpoint: bool, cp: (f32, f32)| -> (f32, f32) {
            if is_endpoint { (2.0 * np.0 - cp.0, 2.0 * np.1 - cp.1) } else { np }
        };
        let n_in = ghost(read_pos(positions, n_idx as usize), is_end(n_idx), cp_a);
        let s_in = ghost(read_pos(positions, s_idx as usize), is_end(s_idx), cp_a);
        let e_in = ghost(read_pos(positions, e_idx as usize), is_end(e_idx), cp_b);
        let w_in = ghost(read_pos(positions, w_idx as usize), is_end(w_idx), cp_b);

        let r = intersect_quadratic_bsplines(n_in, cp_a, s_in, e_in, cp_b, w_in);
        crossing_t[i]     = r.t_a; // t on N-S curve (slot 0)
        crossing_t[other] = r.t_b; // t on E-W curve (slot 1)
    }
}


// ============================================================================
// Stage 6: Rasterize
// ============================================================================

#[inline(always)]
fn beval(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), t: f32) -> (f32, f32) {
    let u = 1.0 - t;
    (
        0.5 * u * u * p0.0 + (u * t + 0.5) * p1.0 + 0.5 * t * t * p2.0,
        0.5 * u * u * p0.1 + (u * t + 0.5) * p1.1 + 0.5 * t * t * p2.1,
    )
}

#[inline(always)]
fn beval_deriv(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), t: f32) -> (f32, f32) {
    (
        (t - 1.0) * p0.0 + (1.0 - 2.0 * t) * p1.0 + t * p2.0,
        (t - 1.0) * p0.1 + (1.0 - 2.0 * t) * p1.1 + t * p2.1,
    )
}

/// Find closest parameter t on a quadratic B-spline span to point pt.
/// Closed-form closest-point on a line segment from a0 to a1. Returns
/// `(t, d²)` matching `closest_on_span_poly`'s signature so callers can
/// dispatch on `CpData::is_line` and merge results into the same hit array.
#[inline(always)]
fn closest_on_segment(
    a0x: f32, a0y: f32, a1x: f32, a1y: f32,
    ptx: f32, pty: f32,
) -> (f32, f32) {
    let vx = a1x - a0x;
    let vy = a1y - a0y;
    let vv = vx * vx + vy * vy;
    let t = if vv > 0.0 {
        (((ptx - a0x) * vx + (pty - a0y) * vy) / vv).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let cx = a0x + t * vx;
    let cy = a0y + t * vy;
    let dx = ptx - cx;
    let dy = pty - cy;
    (t, dx * dx + dy * dy)
}

/// Uses precomputed polynomial coefficients: B(t) = a*t² + b*t + c.
///
/// Exact cubic solver: D(t) = |B(t)-pt|² is degree 4, D'(t) is cubic.
/// Solves D'(t)=0 analytically, evaluates D at roots + endpoints, picks min.
/// No iterative Newton, no coarse sweep, no degenerate endpoint traps.
#[inline(always)]
fn closest_on_span_poly(
    ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32,
    ptx: f32, pty: f32,
) -> (f32, f32) {
    // Shift: let dx(t) = ax*t² + bx*t + (cx-ptx), dy(t) similarly
    let ex = cx - ptx;
    let ey = cy - pty;

    // D'(t)/2 = (ax*t²+bx*t+ex)(2ax*t+bx) + (ay*t²+by*t+ey)(2ay*t+by)
    // Expand to c3*t³ + c2*t² + c1*t + c0:
    //   c3 = 2(ax²+ay²)
    //   c2 = 3(ax*bx+ay*by)
    //   c1 = 2(ax*ex+ay*ey) + bx²+by²
    //   c0 = bx*ex+by*ey
    let c3 = 2.0 * (ax * ax + ay * ay);
    let c2 = 3.0 * (ax * bx + ay * by);
    let c1 = 2.0 * (ax * ex + ay * ey) + bx * bx + by * by;
    let c0 = bx * ex + by * ey;

    // Evaluate D(t) = |B(t)-pt|² via Horner
    let eval_d2 = |t: f32| -> f32 {
        let dx = (ax * t + bx) * t + ex;
        let dy = (ay * t + by) * t + ey;
        dx * dx + dy * dy
    };

    // Start with endpoints
    let d0 = eval_d2(0.0);
    let d1 = eval_d2(1.0);
    let (mut best_t, mut best_d2) = if d0 <= d1 { (0.0f32, d0) } else { (1.0f32, d1) };

    if c3.abs() < 1e-12 {
        // Degenerate: quadratic or linear D'
        if c2.abs() > 1e-12 {
            // Quadratic: c2*t² + c1*t + c0 = 0.
            // Use the numerically stable formula (Numerical Recipes 5.6):
            //   q = -0.5 * (c1 + sgn(c1) * sqrt(disc))
            //   roots: q/c2  and  c0/q
            // This avoids catastrophic cancellation when c2 is small
            // (near-linear D') and the standard `(-c1 ± sq)/(2c2)` form
            // subtracts two nearly-equal numbers.
            let disc = c1 * c1 - 4.0 * c2 * c0;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                let q = -0.5 * (c1 + c1.signum() * sq);
                let t1 = q / c2;
                let t2 = c0 / q;
                for t in [t1, t2] {
                    if t > 0.0 && t < 1.0 {
                        let d = eval_d2(t);
                        if d < best_d2 { best_d2 = d; best_t = t; }
                    }
                }
            }
        } else if c1.abs() > 1e-12 {
            // Linear: c1*t + c0 = 0
            let t = -c0 / c1;
            if t > 0.0 && t < 1.0 {
                let d = eval_d2(t);
                if d < best_d2 { best_d2 = d; best_t = t; }
            }
        }
    } else {
        // Full cubic: c3*t³ + c2*t² + c1*t + c0 = 0
        // Depressed cubic via substitution t = u - c2/(3*c3)
        let inv3a = 1.0 / (3.0 * c3);
        let shift = -c2 * inv3a;
        let p = (3.0 * c3 * c1 - c2 * c2) / (3.0 * c3 * c3);
        let q = (2.0 * c2 * c2 * c2 - 9.0 * c3 * c2 * c1 + 27.0 * c3 * c3 * c0)
            / (27.0 * c3 * c3 * c3);

        let disc = q * q / 4.0 + p * p * p / 27.0;

        if disc > 1e-12 {
            // One real root
            let sq = disc.sqrt();
            let u = (-q / 2.0 + sq).cbrt() + (-q / 2.0 - sq).cbrt();
            let t = u + shift;
            if t > 0.0 && t < 1.0 {
                let d = eval_d2(t);
                if d < best_d2 { best_d2 = d; best_t = t; }
            }
        } else {
            // Three real roots (trigonometric method)
            let r = (-p * p * p / 27.0).max(0.0).sqrt();
            let phi = if r.abs() < 1e-15 { 0.0 } else { (-q / (2.0 * r)).clamp(-1.0, 1.0).acos() };
            let cube_r = r.cbrt() * 2.0;
            for k in 0..3 {
                let angle = (phi + std::f32::consts::TAU * k as f32) / 3.0;
                let t = cube_r * angle.cos() + shift;
                if t > 0.0 && t < 1.0 {
                    let d = eval_d2(t);
                    if d < best_d2 { best_d2 = d; best_t = t; }
                }
            }
        }
    }

    (best_t, best_d2)
}

/// CP data for rasterization (replaces SharedCP in the shader).
/// Linear approximation of a curve at the AA hit's closest_t. Used by the
/// multi-curve wedge AA path to partition a pixel by multiple curves; each
/// wedge's color is then resampled via classify_at, so we only need the
/// geometry here, not the curve's own pos/neg side codes.
struct AaLine {
    cpt: (f32, f32),
    /// Unit normal. Side test: `(pt - cpt) · normal`.
    normal: (f32, f32),
}

struct CpData {
    /// Cell-graph CP index (= corner_index*2 + slot). Stored so the
    /// rasterizer can find slot-1 at the same corner via flags/neighbors.
    ci: i32,
    pos: (f32, f32),
    orig_pos: (f32, f32),
    prev_pos: (f32, f32),
    next_pos: (f32, f32),
    orig_prev: (f32, f32),
    orig_next: (f32, f32),
    /// Precomputed polynomial coefficients: B(t) = a*t² + b*t + c
    /// B'(t) = 2*a*t + b, B''(t) = 2*a
    poly_ax: f32, poly_ay: f32,
    poly_bx: f32, poly_by: f32,
    poly_cx: f32, poly_cy: f32,
    /// Branch threshold for prev_dir vs next_dir in resolve_from_cp. For
    /// non-clamped (interior) spans this is 0.5; for clamped Bezier spans
    /// (Q0=prev_endpoint or Q2=next_endpoint), the parameterization is
    /// shifted, so t_branch is the t value at which the clamped curve
    /// reaches the same physical "before/after sc" boundary that the
    /// equivalent interior B-spline reaches at t=0.5. Pixels with closest-
    /// point t < t_branch classify as prev side; t ≥ t_branch as next side.
    t_branch: f32,
    prev_dir: i32,
    next_dir: i32,
    icx: i32,
    icy: i32,
    prev_ci: i32,
    next_ci: i32,
    /// True for 2-CP chains (degenerate stem with both ends being endpoint
    /// markers). The span is geometrically a straight line, so the rasterizer
    /// uses a closed-form line-segment distance instead of the cubic solver,
    /// avoiding the float-noise vs degeneracy-threshold trap.
    is_line: bool,
    /// CP flags from the cell graph (IS_TJUNCTION, IS_CROSSING, ...).
    /// Used by the rasterizer to gate wedge AA — slot 1 stem CPs at
    /// T-junctions get filtered out of all_cps because their span is
    /// owned by the interior chain neighbor, so a same-corner check
    /// alone can't identify T-junctions; we need the IS_TJUNCTION flag
    /// on the through-curve CP instead.
    flag: u32,
}

#[inline(always)]
fn get_px_color(pixels: &[u32], img_w: usize, img_h: usize, px: i32, py: i32) -> u32 {
    let px = px.clamp(0, img_w as i32 - 1) as usize;
    let py = py.clamp(0, img_h as i32 - 1) as usize;
    pixels[py * img_w + px]
}

#[inline(always)]
fn get_edge_colors(
    pixels: &[u32], img_w: usize, img_h: usize,
    icx: i32, icy: i32, dir: i32,
) -> (u32, u32) {
    match dir {
        0 => (get_px_color(pixels, img_w, img_h, icx - 1, icy - 1),
              get_px_color(pixels, img_w, img_h, icx, icy - 1)),
        1 => (get_px_color(pixels, img_w, img_h, icx, icy - 1),
              get_px_color(pixels, img_w, img_h, icx, icy)),
        2 => (get_px_color(pixels, img_w, img_h, icx, icy),
              get_px_color(pixels, img_w, img_h, icx - 1, icy)),
        3 => (get_px_color(pixels, img_w, img_h, icx - 1, icy),
              get_px_color(pixels, img_w, img_h, icx - 1, icy - 1)),
        _ => (0, 0),
    }
}

/// sRGB decode: u32 ARGB → (r, g, b) in linear space.
#[inline(always)]
fn srgb_decode(c: u32) -> (f32, f32, f32) {
    let r = ((c >> 16) & 0xFF) as f32 / 255.0;
    let g = ((c >> 8) & 0xFF) as f32 / 255.0;
    let b = (c & 0xFF) as f32 / 255.0;
    (r.powf(2.2), g.powf(2.2), b.powf(2.2))
}

/// sRGB encode: linear (r, g, b) → ARGB u32 with alpha=0xFF.
#[inline(always)]
fn srgb_encode_argb(r: f32, g: f32, b: f32) -> u32 {
    let r_o = (r.powf(1.0 / 2.2) * 255.0).round().clamp(0.0, 255.0) as u32;
    let g_o = (g.powf(1.0 / 2.2) * 255.0).round().clamp(0.0, 255.0) as u32;
    let b_o = (b.powf(1.0 / 2.2) * 255.0).round().clamp(0.0, 255.0) as u32;
    0xFF000000 | (r_o << 16) | (g_o << 8) | b_o
}

/// Exact pixel coverage by a tangent line. `d_perp_pixel` is the line's
/// signed perpendicular distance from pixel center in pixel-side units.
/// Returns area on the line's positive side, in [0, 1].
///   |d_p| <= (a-b)/2     → linear regime (line crosses parallel edges)
///   (a-b)/2 < |d_p| < (a+b)/2 → corner-cut quadratic
///   |d_p| >= (a+b)/2     → saturated
#[inline(always)]
fn line_coverage_pos(normal: (f32, f32), d_perp_pixel: f32) -> f32 {
    let nx = normal.0.abs();
    let ny = normal.1.abs();
    let a = nx.max(ny);
    let b = nx.min(ny);
    let half_ext = (a + b) * 0.5;
    let lin_ext = (a - b) * 0.5;
    if d_perp_pixel >= half_ext { 1.0 }
    else if d_perp_pixel <= -half_ext { 0.0 }
    else if d_perp_pixel.abs() <= lin_ext { 0.5 + d_perp_pixel / a }
    else if d_perp_pixel > 0.0 {
        let t = half_ext - d_perp_pixel;
        1.0 - 0.5 * t * t / (a * b)
    } else {
        let t = half_ext + d_perp_pixel;
        0.5 * t * t / (a * b)
    }
}

/// Build an AaLine + the curve's pos/neg side colors at parameter `t`.
/// `cpt` is the closest point on the curve (sc) at `t`; `normal` is the
/// unit perpendicular to the tangent. Colors come from the t_branch
/// segment containing `t` via `resolve_lut_segment`. Returns None when
/// the tangent is too short (singular point on the curve) or the curve
/// has no usable edge color discontinuity. Used by single-curve and
/// dual-curve AA — wedge AA uses `build_junction_aa_line` instead since
/// its cpt is an external junction point and colors are resolved per
/// segment by the LUT.
fn build_aa_line(
    sc: &CpData, t: f32,
    pixels: &[u32], img_w: usize, img_h: usize,
) -> Option<(AaLine, u32, u32)> {
    let tang = beval_deriv(sc.prev_pos, sc.pos, sc.next_pos, t);
    let tl = (tang.0 * tang.0 + tang.1 * tang.1).sqrt();
    if tl < 1e-4 { return None; }
    let normal = (-tang.1 / tl, tang.0 / tl);
    let target_seg1 = t >= sc.t_branch;
    let (pos, neg) =
        resolve_lut_segment(sc, normal, pixels, img_w, img_h, target_seg1)?;
    let cpt = beval(sc.prev_pos, sc.pos, sc.next_pos, t);
    Some((AaLine { cpt, normal }, pos, neg))
}

/// Junction-anchored wedge AA line: cpt = junction position J (instead of
/// the curve's closest_t to the pixel), tangent evaluated at `t_eval`.
/// Returns the unit tangent alongside the line so the caller can align
/// line_b's normal to line_a's tangent direction.
fn build_junction_aa_line(
    sc: &CpData, t_eval: f32, j: (f32, f32),
    pixels: &[u32], img_w: usize, img_h: usize,
) -> Option<(AaLine, (f32, f32))> {
    let tang = beval_deriv(sc.prev_pos, sc.pos, sc.next_pos, t_eval);
    let tl = (tang.0 * tang.0 + tang.1 * tang.1).sqrt();
    if tl < 1e-4 { return None; }
    let tang_unit = (tang.0 / tl, tang.1 / tl);
    let normal = (-tang_unit.1, tang_unit.0);

    let prev_split = sc.prev_dir >= 0 && {
        let (l, r) = get_edge_colors(pixels, img_w, img_h, sc.icx, sc.icy, sc.prev_dir);
        l != r
    };
    let next_split = sc.next_dir >= 0 && {
        let (l, r) = get_edge_colors(pixels, img_w, img_h, sc.icx, sc.icy, sc.next_dir);
        l != r
    };
    if !prev_split && !next_split { return None; }

    Some((AaLine { cpt: j, normal }, tang_unit))
}

/// Resolve one segment of the wedge-AA color LUT for `sc_a` (the through
/// curve at a T-junction). Returns (pos_color, neg_color) aligned to
/// `line_a_normal`. See "Wedge AA model" comment in rasterize() for context.
fn resolve_lut_segment(
    sc_a: &CpData,
    line_a_normal: (f32, f32),
    pixels: &[u32], img_w: usize, img_h: usize,
    target_is_seg1: bool,
) -> Option<(u32, u32)> {
    let (mut pl, mut pr) = (0u32, 0u32);
    let (mut nl, mut nr) = (0u32, 0u32);
    if sc_a.prev_dir >= 0 {
        let (l, r) = get_edge_colors(pixels, img_w, img_h, sc_a.icx, sc_a.icy, sc_a.prev_dir);
        pl = l; pr = r;
    }
    if sc_a.next_dir >= 0 {
        let (l, r) = get_edge_colors(pixels, img_w, img_h, sc_a.icx, sc_a.icy, sc_a.next_dir);
        nl = l; nr = r;
    }
    let prev_valid = pl != pr;
    let next_valid = nl != nr;
    let (color_left, color_right, ref_t);
    if sc_a.prev_ci < 0 {
        if next_valid { color_left = nr; color_right = nl; ref_t = 1.0; }
        else { return None; }
    } else if sc_a.next_ci < 0 {
        if prev_valid { color_left = pl; color_right = pr; ref_t = 0.0; }
        else { return None; }
    } else if !target_is_seg1 {
        if prev_valid { color_left = pl; color_right = pr; ref_t = 0.0; }
        else if next_valid { color_left = nr; color_right = nl; ref_t = 1.0; }
        else { return None; }
    } else {
        if next_valid { color_left = nr; color_right = nl; ref_t = 1.0; }
        else if prev_valid { color_left = pl; color_right = pr; ref_t = 0.0; }
        else { return None; }
    }
    let ot = beval_deriv(sc_a.orig_prev, sc_a.orig_pos, sc_a.orig_next, ref_t);
    let on = (-ot.1, ot.0);
    let flip = line_a_normal.0 * on.0 + line_a_normal.1 * on.1 < 0.0;
    let pos = if flip { color_right } else { color_left };
    let neg = if flip { color_left } else { color_right };
    Some((pos, neg))
}

fn rasterize(
    pixels: &[u32],
    positions: &[f32],
    orig_positions: &[f32],
    flags: &[u32],
    cp_neighbors: &[i32],
    crossing_t: &[f32],
    img_w: usize,
    img_h: usize,
    out_w: usize,
    out_h: usize,
    scale_factor: f32,
) -> Vec<u32> {
    let corners_w = img_w + 1;
    let num_cps = corners_w * (img_h + 1) * 2;

    let read_pos_f = |idx: i32| -> (f32, f32) {
        if idx < 0 {
            return (-1e10, -1e10);
        }
        let i = idx as usize;
        (positions[i * 2], positions[i * 2 + 1])
    };

    let read_orig_f = |idx: i32| -> (f32, f32) {
        if idx < 0 {
            return (-1e10, -1e10);
        }
        let i = idx as usize;
        (orig_positions[i * 2], orig_positions[i * 2 + 1])
    };

    // Build list of active CPs
    let mut all_cps: Vec<CpData> = Vec::new();
    for ci in 0..num_cps {
        let flag = flags[ci];
        if flag == 0 {
            continue;
        }

        let prev_ci = cp_neighbors[ci * 4];
        let next_ci = cp_neighbors[ci * 4 + 1];
        // Clamped model: endpoint CPs don't own a span; their interior
        // neighbor's span extends to reach them via knot multiplicity 3.
        // Exception: 2-CP chains have no interior neighbor, render here.
        if prev_ci < 0 && next_ci < 0 {
            continue;
        }
        let i_am_endpoint = prev_ci < 0 || next_ci < 0;
        let mut two_cp_chain = false;
        if i_am_endpoint {
            let other = if prev_ci < 0 { next_ci } else { prev_ci };
            let other_is_end = (flags[other as usize] & IS_ENDPOINT) != 0;
            if !other_is_end {
                continue;
            }
            // 2-CP chain: both endpoints are markers with no interior CP. The
            // span has to render here, but only ONCE — the lower-indexed
            // endpoint owns it. Without this guard both endpoints render the
            // same physical span with opposite-bowing ghost curvatures and
            // pollute wedge-AA classifier lines at T-junctions.
            if (ci as i32) > other {
                continue;
            }
            two_cp_chain = true;
        }

        let cp_real = read_pos_f(ci as i32);
        // 2-CP-chain fallback: if either neighbor is missing, use the CP
        // itself for that side (degenerate, straight-line span).
        let prev_pos = if prev_ci >= 0 { read_pos_f(prev_ci) } else { cp_real };
        let next_pos = if next_ci >= 0 { read_pos_f(next_ci) } else { cp_real };

        // If a neighbor is an endpoint, replace its position with a virtual
        // ghost = 2*real - cp so that bspline_eval(...,t=0|1) lands exactly
        // on the real endpoint position. For interior neighbors, ghost = real.
        let prev_is_end = prev_ci >= 0 && (flags[prev_ci as usize] & IS_ENDPOINT) != 0;
        let next_is_end = next_ci >= 0 && (flags[next_ci as usize] & IS_ENDPOINT) != 0;

        // For 2-CP chains, the span runs between two endpoint markers with no
        // interior CP. Compute its anchors once (in pos and orig space) and
        // share across the bspline-eval-friendly p0/p1/p2, the polynomial
        // coefficients, and the orig overrides.
        let (a0_pos, a1_pos, a0_orig, a1_orig) = if two_cp_chain {
            let other = if prev_ci < 0 { next_ci } else { prev_ci };
            let other_pos = read_pos_f(other);
            let other_orig = read_orig_f(other);
            let this_orig = read_orig_f(ci as i32);
            let a0_pos = if prev_ci < 0 { cp_real } else { other_pos };
            let a1_pos = if next_ci < 0 { cp_real } else { other_pos };
            let a0_orig = if prev_ci < 0 { this_orig } else { other_orig };
            let a1_orig = if next_ci < 0 { this_orig } else { other_orig };
            (a0_pos, a1_pos, a0_orig, a1_orig)
        } else {
            ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        };

        let (cp, pp, np) = if two_cp_chain {
            // Render the span as a straight line: pick p0=(3·a0-a1)/2,
            // p1=midpoint, p2=(3·a1-a0)/2 so bspline_eval gives
            // B(t) = lerp(a0, a1, t) and the polynomial t² coefficient is 0.
            // a0 = prev side, a1 = next side.
            let p0 = (1.5 * a0_pos.0 - 0.5 * a1_pos.0, 1.5 * a0_pos.1 - 0.5 * a1_pos.1);
            let p1 = (0.5 * (a0_pos.0 + a1_pos.0), 0.5 * (a0_pos.1 + a1_pos.1));
            let p2 = (1.5 * a1_pos.0 - 0.5 * a0_pos.0, 1.5 * a1_pos.1 - 0.5 * a0_pos.1);
            (p1, p0, p2)
        } else {
            let cp = cp_real;
            let pp = if prev_is_end {
                (2.0 * prev_pos.0 - cp.0, 2.0 * prev_pos.1 - cp.1)
            } else {
                prev_pos
            };
            let np = if next_is_end {
                (2.0 * next_pos.0 - cp.0, 2.0 * next_pos.1 - cp.1)
            } else {
                next_pos
            };
            (cp, pp, np)
        };

        // Polynomial form: B(t) = a*t² + b*t + c (uniform B-spline, virtual
        // ghosts make the eval equivalent to the clamped Bezier).
        // a = 0.5*(p0 - 2*p1 + p2), b = p1 - p0, c = 0.5*(p0 + p1).
        // For 2-CP chains, set coefficients directly so the t² term is
        // *exactly* zero. Computing it via 0.5*(pp - 2*cp + np) leaves ~1e-6
        // of float noise, which makes c3 = 2|a|² ≈ 4e-12 — just above
        // closest_on_span_poly's 1e-12 degenerate-detection threshold —
        // sending the solver down the full-cubic numerical-disaster path.
        let (poly_ax, poly_ay, poly_bx, poly_by, poly_cx, poly_cy) = if two_cp_chain {
            (
                0.0, 0.0,
                a1_pos.0 - a0_pos.0, a1_pos.1 - a0_pos.1,
                a0_pos.0, a0_pos.1,
            )
        } else {
            (
                0.5 * (pp.0 - 2.0 * cp.0 + np.0),
                0.5 * (pp.1 - 2.0 * cp.1 + np.1),
                cp.0 - pp.0,
                cp.1 - pp.1,
                0.5 * (pp.0 + cp.0),
                0.5 * (pp.1 + cp.1),
            )
        };

        // Compute t_branch: see CpData::t_branch doc comment. For non-clamped
        // spans this is 0.5; for clamped spans we find the clamped curve's t
        // at the position the equivalent interior B-spline would reach at
        // t=0.5 (the natural before/after-sc pivot in physical space).
        // Crossings override the default with the actual curve-curve
        // intersection parameter from `crossing_t`, so the rasterizer's
        // J = beval(...t_branch) lands on the geometric intersection, not
        // the integer grid corner.
        let t_branch = if (flag & IS_CROSSING) != 0 {
            crossing_t[ci]
        } else if two_cp_chain {
            // Straight line: interior_mid = cp = midpoint, closest t is 0.5.
            0.5
        } else if prev_is_end || next_is_end {
            // Use ghost-adjusted positions (= rendered B(0.5)) as interior_mid.
            // This makes the bifurcation point coincide with the algebraic
            // ghost-aware stem snap and the ghost-aware crossing correction.
            let interior_mid_x = 0.125 * pp.0 + 0.75 * cp.0 + 0.125 * np.0;
            let interior_mid_y = 0.125 * pp.1 + 0.75 * cp.1 + 0.125 * np.1;
            closest_on_span_poly(
                poly_ax, poly_ay, poly_bx, poly_by, poly_cx, poly_cy,
                interior_mid_x, interior_mid_y,
            )
            .0
        } else {
            0.5
        };

        // For 2-CP chains, mirror the straight-line construction in orig
        // space too so beval(orig_prev, orig_pos, orig_next) gives the same
        // line in original cell-graph coordinates that the AA flip-detection
        // in the rasterizer expects.
        let (orig_pos_out, orig_prev_out, orig_next_out) = if two_cp_chain {
            let q0 = (1.5 * a0_orig.0 - 0.5 * a1_orig.0, 1.5 * a0_orig.1 - 0.5 * a1_orig.1);
            let q1 = (0.5 * (a0_orig.0 + a1_orig.0), 0.5 * (a0_orig.1 + a1_orig.1));
            let q2 = (1.5 * a1_orig.0 - 0.5 * a0_orig.0, 1.5 * a1_orig.1 - 0.5 * a0_orig.1);
            (q1, q0, q2)
        } else {
            (
                read_orig_f(ci as i32),
                if prev_ci >= 0 { read_orig_f(prev_ci) } else { read_orig_f(ci as i32) },
                if next_ci >= 0 { read_orig_f(next_ci) } else { read_orig_f(ci as i32) },
            )
        };

        all_cps.push(CpData {
            ci: ci as i32,
            pos: cp,
            orig_pos: orig_pos_out,
            prev_pos: pp,
            next_pos: np,
            orig_prev: orig_prev_out,
            orig_next: orig_next_out,
            poly_ax, poly_ay, poly_bx, poly_by, poly_cx, poly_cy,
            prev_dir: cp_neighbors[ci * 4 + 2],
            next_dir: cp_neighbors[ci * 4 + 3],
            icx: (ci / 2 % corners_w) as i32,
            icy: (ci / 2 / corners_w) as i32,
            prev_ci,
            next_ci,
            t_branch,
            is_line: two_cp_chain,
            flag: flags[ci],
        });
    }

    // Reverse map: cell-graph CP index → all_cps array position, or -1 for
    // CPs filtered out by the topology pass. The 4-cell-corner walk in
    // rasterize_rows looks up CPs by their ci directly (slot 0 and slot 1
    // at each of the cell's 4 corners) instead of iterating a per-cell
    // bbox-overlap list.
    let mut ci_to_all_cps: Vec<i32> = vec![-1; num_cps];
    for (i, sc) in all_cps.iter().enumerate() {
        ci_to_all_cps[sc.ci as usize] = i as i32;
    }

    let mut output = vec![0u32; out_w * out_h];

    let resolve_from_cp =
        |pt: (f32, f32), sc: &CpData, t: f32| -> Option<u32> {
            // Endpoint defer: when this CP's curve has its closest point
            // exactly at an endpoint that's structurally co-located with
            // another CP's curve (degenerate stem at t=1, clamped Bezier
            // extension at t=0), the local edges describe a different region.
            // Defer to the next-closest candidate so the local CP wins. The
            // returned None falls through to the fallback color (the source
            // pixel underneath), which is correct for genuine
            // outside-the-curve-extent pixels.
            //
            // The t == 0.0 / t == 1.0 exact compares are sound because
            // closest_on_segment / closest_on_span_poly clamp t to exactly
            // 0 or 1 at the endpoints — they don't return 1e-7 noise. Don't
            // relax this to an epsilon: a curve whose closest point is
            // *just* inside (t=0.001) shouldn't defer.
            let prev_extends =
                sc.prev_ci < 0 || (flags[sc.prev_ci as usize] & IS_ENDPOINT) != 0;
            let next_extends =
                sc.next_ci < 0 || (flags[sc.next_ci as usize] & IS_ENDPOINT) != 0;
            if (prev_extends && t == 0.0) || (next_extends && t == 1.0) {
                return None;
            }
            let (mut pl, mut pr) = (0u32, 0u32);
            let (mut nl, mut nr) = (0u32, 0u32);
            if sc.prev_dir >= 0 { let (l, r) = get_edge_colors(pixels, img_w, img_h, sc.icx, sc.icy, sc.prev_dir); pl = l; pr = r; }
            if sc.next_dir >= 0 { let (l, r) = get_edge_colors(pixels, img_w, img_h, sc.icx, sc.icy, sc.next_dir); nl = l; nr = r; }

            let prev_valid = pl != pr;
            let next_valid = nl != nr;

            let (color_left, color_right, ref_t);

            if sc.prev_ci < 0 {
                if next_valid {
                    color_left = nr;
                    color_right = nl;
                    ref_t = 1.0;
                } else {
                    return None;
                }
            } else if sc.next_ci < 0 {
                if prev_valid {
                    color_left = pl;
                    color_right = pr;
                    ref_t = 0.0;
                } else {
                    return None;
                }
            } else if t < sc.t_branch {
                if prev_valid {
                    color_left = pl;
                    color_right = pr;
                    ref_t = 0.0;
                } else if next_valid {
                    color_left = nr;
                    color_right = nl;
                    ref_t = 1.0;
                } else {
                    return None;
                }
            } else {
                if next_valid {
                    color_left = nr;
                    color_right = nl;
                    ref_t = 1.0;
                } else if prev_valid {
                    color_left = pl;
                    color_right = pr;
                    ref_t = 0.0;
                } else {
                    return None;
                }
            }

            let orig_tangent = beval_deriv(sc.orig_prev, sc.orig_pos, sc.orig_next, ref_t);
            let tl2 = (orig_tangent.0 * orig_tangent.0 + orig_tangent.1 * orig_tangent.1).sqrt();
            if tl2 < 1e-4 {
                return None;
            }
            let orig_tangent = (orig_tangent.0 / tl2, orig_tangent.1 / tl2);
            let orig_normal = (-orig_tangent.1, orig_tangent.0);

            let cpt = beval(sc.prev_pos, sc.pos, sc.next_pos, t);
            let opt_tangent = beval_deriv(sc.prev_pos, sc.pos, sc.next_pos, t);
            let otl = (opt_tangent.0 * opt_tangent.0 + opt_tangent.1 * opt_tangent.1).sqrt();
            if otl < 1e-4 {
                return None;
            }
            let opt_tangent = (opt_tangent.0 / otl, opt_tangent.1 / otl);
            let opt_normal = (-opt_tangent.1, opt_tangent.0);

            let normals_agree =
                opt_normal.0 * orig_normal.0 + opt_normal.1 * orig_normal.1 > 0.0;
            let side = (pt.0 - cpt.0) * opt_normal.0 + (pt.1 - cpt.1) * opt_normal.1;

            if normals_agree {
                Some(if side > 0.0 { color_left } else { color_right })
            } else {
                Some(if side > 0.0 { color_right } else { color_left })
            }
        };

    let inv_scale = 1.0 / scale_factor;
    // AA threshold: a pixel's half-diagonal² in source units is
    // 0.5/scale². Beyond that, the curve passes outside the pixel's
    // bounds and the analytical pixel-coverage formula saturates to
    // a solid color anyway, so AA work would be wasted. Apply this
    // tight bound to both single-curve AA and the wedge AA gate so
    // each fires only on pixels the curve(s) actually pass through.
    let aa_threshold = 0.5 / (scale_factor * scale_factor);

    // Rasterize row range into output chunk
    let rasterize_rows = |chunk: &mut [u32], start: usize,
        all_cps: &[CpData], ci_to_all_cps: &[i32]|
    {
        // Per-fragment span test: insert into a top-3 hit array if close
        // enough. `cp_i` is the all_cps array position (matches the slot
        // returned by the wedge-AA dual-curve hit lookup).
        let mut try_cp = |cp_i: u32, center: (f32, f32),
                          hit_d2: &mut [f32; 3], hit_t: &mut [f32; 3],
                          hit_idx: &mut [u32; 3], num_hits: &mut usize| {
            let sc = &all_cps[cp_i as usize];
            // Quick screen: 3-sample distance² reject.
            let b0 = (sc.poly_cx, sc.poly_cy);
            let bm = (0.125 * sc.prev_pos.0 + 0.75 * sc.pos.0 + 0.125 * sc.next_pos.0,
                       0.125 * sc.prev_pos.1 + 0.75 * sc.pos.1 + 0.125 * sc.next_pos.1);
            let b1 = (0.5 * (sc.pos.0 + sc.next_pos.0), 0.5 * (sc.pos.1 + sc.next_pos.1));
            let qd0 = (center.0-b0.0)*(center.0-b0.0) + (center.1-b0.1)*(center.1-b0.1);
            let qdm = (center.0-bm.0)*(center.0-bm.0) + (center.1-bm.1)*(center.1-bm.1);
            let qd1 = (center.0-b1.0)*(center.0-b1.0) + (center.1-b1.1)*(center.1-b1.1);
            let quick_d2 = qd0.min(qdm).min(qd1);
            if quick_d2 > 2.0 { return; }

            let result = if sc.is_line {
                let (a0x, a0y) = (0.5 * (sc.prev_pos.0 + sc.pos.0),
                                   0.5 * (sc.prev_pos.1 + sc.pos.1));
                let (a1x, a1y) = (0.5 * (sc.pos.0 + sc.next_pos.0),
                                   0.5 * (sc.pos.1 + sc.next_pos.1));
                closest_on_segment(a0x, a0y, a1x, a1y, center.0, center.1)
            } else {
                closest_on_span_poly(
                    sc.poly_ax, sc.poly_ay, sc.poly_bx, sc.poly_by,
                    sc.poly_cx, sc.poly_cy, center.0, center.1,
                )
            };
            let span_best_t = result.0;
            let span_best_d2 = result.1;
            if span_best_d2 >= 1.0 { return; }

            // Insertion sort via explicit if/else over constant indices.
            // Mirrors the GPU rasterizer's structure for consistency.
            if span_best_d2 < hit_d2[0] {
                hit_d2[2] = hit_d2[1]; hit_t[2] = hit_t[1]; hit_idx[2] = hit_idx[1];
                hit_d2[1] = hit_d2[0]; hit_t[1] = hit_t[0]; hit_idx[1] = hit_idx[0];
                hit_d2[0] = span_best_d2; hit_t[0] = span_best_t; hit_idx[0] = cp_i;
                if *num_hits < 3 { *num_hits += 1; }
            } else if span_best_d2 < hit_d2[1] {
                hit_d2[2] = hit_d2[1]; hit_t[2] = hit_t[1]; hit_idx[2] = hit_idx[1];
                hit_d2[1] = span_best_d2; hit_t[1] = span_best_t; hit_idx[1] = cp_i;
                if *num_hits < 3 { *num_hits += 1; }
            } else if span_best_d2 < hit_d2[2] {
                hit_d2[2] = span_best_d2; hit_t[2] = span_best_t; hit_idx[2] = cp_i;
                if *num_hits < 3 { *num_hits += 1; }
            }
        };

        let chunk_rows = chunk.len() / out_w;
        for local_y in 0..chunk_rows {
            let opy = start + local_y;
            let center_y = (opy as f32 + 0.5) * inv_scale;
            let fb_y = (center_y.floor() as i32).clamp(0, img_h as i32 - 1) as usize;

            for opx in 0..out_w {
                let center_x = (opx as f32 + 0.5) * inv_scale;
                let center = (center_x, center_y);
                let fb_x = (center_x.floor() as i32).clamp(0, img_w as i32 - 1) as usize;
                let fallback = pixels[fb_y * img_w + fb_x];

                // Top-3 hit array (avoids Vec allocation per pixel).
                let mut hit_d2 = [1e10f32; 3];
                let mut hit_t = [0.0f32; 3];
                let mut hit_idx = [0u32; 3];
                let mut num_hits = 0usize;

                // 4-cell-corner walk: a quadratic B-spline bows ≤0.5 units
                // from chord, so spans centered further than the cell's own
                // corners can't reach into the cell. Test slot 0 + slot 1 at
                // each cell corner via the ci → all_cps reverse map.
                let cell_x = (center_x.floor() as i32).clamp(0, img_w as i32 - 1) as usize;
                let cell_y = (center_y.floor() as i32).clamp(0, img_h as i32 - 1) as usize;
                for cdy in 0..2 {
                    for cdx in 0..2 {
                        let cx = cell_x + cdx;
                        let cy = cell_y + cdy;
                        if cx >= corners_w || cy >= img_h + 1 { continue; }
                        let slot0_ci = (cy * corners_w + cx) * 2;
                        // slot 0
                        let m0 = ci_to_all_cps[slot0_ci];
                        if m0 >= 0 {
                            try_cp(m0 as u32, center, &mut hit_d2, &mut hit_t,
                                   &mut hit_idx, &mut num_hits);
                        }
                        // slot 1
                        let m1 = ci_to_all_cps[slot0_ci + 1];
                        if m1 >= 0 {
                            try_cp(m1 as u32, center, &mut hit_d2, &mut hit_t,
                                   &mut hit_idx, &mut num_hits);
                        }
                        // T-junction stem fan-out: at this cell corner if
                        // slot 0 is a T-junction through-CP, the stem-bottom
                        // CP can sit at offset (+2, +1) etc. — outside the
                        // 4-corner walk. Look it up via slot 1's chain link.
                        if (flags[slot0_ci] & IS_TJUNCTION) != 0 {
                            let slot1_ci = slot0_ci + 1;
                            let prev_ci = cp_neighbors[slot1_ci * 4];
                            let next_ci = cp_neighbors[slot1_ci * 4 + 1];
                            let sci = if prev_ci >= 0 { prev_ci } else { next_ci };
                            if sci >= 0 {
                                let s_cx = (sci as usize / 2) % corners_w;
                                let s_cy = (sci as usize / 2) / corners_w;
                                // Skip when stem-bottom is already a cell corner.
                                let already = s_cx >= cell_x && s_cx <= cell_x + 1
                                           && s_cy >= cell_y && s_cy <= cell_y + 1;
                                if !already {
                                    let mf = ci_to_all_cps[sci as usize];
                                    if mf >= 0 {
                                        try_cp(mf as u32, center, &mut hit_d2, &mut hit_t,
                                               &mut hit_idx, &mut num_hits);
                                    }
                                }
                            }
                        }
                    }
                }

                let best_idx = if num_hits > 0 { hit_idx[0] as i32 } else { -1 };
                let best_t = if num_hits > 0 { hit_t[0] } else { 0.0 };
                let best_d2 = if num_hits > 0 { hit_d2[0] } else { 1e10 };

                // Resolve color from hits. Track which hit actually resolved to
                // a valid color so the AA path uses the SAME CP that produced
                // center_color, not just the closest one (which may have
                // returned None via endpoint-defer in resolve_from_cp).
                let mut center_color = fallback;
                let mut resolved_h: i32 = -1;
                // Try each hit in order — unrolled across constant indices
                // for consistency with the GPU rasterizer.
                if num_hits >= 1 && resolved_h < 0 {
                    if let Some(c) = resolve_from_cp(center, &all_cps[hit_idx[0] as usize], hit_t[0]) {
                        center_color = c;
                        resolved_h = 0;
                    }
                }
                if num_hits >= 2 && resolved_h < 0 {
                    if let Some(c) = resolve_from_cp(center, &all_cps[hit_idx[1] as usize], hit_t[1]) {
                        center_color = c;
                        resolved_h = 1;
                    }
                }
                if num_hits >= 3 && resolved_h < 0 {
                    if let Some(c) = resolve_from_cp(center, &all_cps[hit_idx[2] as usize], hit_t[2]) {
                        center_color = c;
                        resolved_h = 2;
                    }
                }

                // For AA, use the hit that actually resolved to a valid color
                // (so endpoint-deferred hits don't drive the AA blend).
                // resolved_h ∈ {0, 1, 2}; explicit selection avoids dynamic indexing.
                let (aa_idx, aa_t, aa_d2) = match resolved_h {
                    0 => (hit_idx[0] as i32, hit_t[0], hit_d2[0]),
                    1 => (hit_idx[1] as i32, hit_t[1], hit_d2[1]),
                    2 => (hit_idx[2] as i32, hit_t[2], hit_d2[2]),
                    _ => (best_idx, best_t, best_d2),
                };
                let need_aa = aa_idx >= 0 && aa_d2 < aa_threshold;

                // Wedge AA needs the closest curve to be the actual through
                // curve at a junction AND the second-closest to be the
                // matching stem render curve. Identify this structurally:
                //
                // At a T-junction corner, slot 0 holds the through curve;
                // slot 1 holds the original stem CP (filtered from all_cps —
                // its render span is owned by an adjacent corner via clamped
                // Bezier). Slot 1's surviving non-(-1) neighbor IS the stem
                // render CP. So: candidate A is the through-curve CP at a
                // junction iff
                //   (a) A is IS_TJUNCTION-flagged, AND
                //   (b) slot 1 at A's corner has B's ci as one of its
                //       neighbors (i.e., B is the stem render CP).
                //
                // Crossings (valence-4): both hits are IS_CROSSING at the
                // same corner — partner identification is direct.
                // slot1_neighbor_match returns Some(t_b_eval) if sc_b is the
                // stem-side curve for sc_a's junction, where t_b_eval is the
                // parameter value on sc_b's curve at which it passes through
                // (or terminates at) the junction. None if no structural
                // junction relationship.
                //
                // Three structural cases, all handled without closest_t:
                //   (a) sc_b IS slot 1 of sc_a's corner — 2-CP-chain stem.
                //       Cell graph creates slot 1 with next_ci=-1, so the
                //       2-CP construction puts the junction-side endpoint
                //       at a1 = slot1.pos, hit at t=1.
                //   (b) sc_b is slot 1's neighbor — slot 1 was filtered, sc_b
                //       is the chain CP at the adjacent corner that owns the
                //       stem span via clamped Bezier toward slot 1's pos.
                //       sc_b's curve ghost-extends through slot 1's pos at
                //       whichever of sc_b's prev/next is IS_ENDPOINT-flagged
                //       (t=0 for prev side, t=1 for next side).
                let slot1_neighbor_t = |sc_a: &CpData, sc_b: &CpData| -> Option<f32> {
                    let slot1_ci = ((sc_a.icy as usize * corners_w + sc_a.icx as usize) * 2 + 1) as i32;
                    if (slot1_ci as usize) >= flags.len() { return None; }
                    // Case (a)
                    if sc_b.ci == slot1_ci { return Some(1.0); }
                    // Case (b)
                    let slot1_prev = cp_neighbors[(slot1_ci as usize) * 4];
                    let slot1_next = cp_neighbors[(slot1_ci as usize) * 4 + 1];
                    if slot1_prev != sc_b.ci && slot1_next != sc_b.ci { return None; }
                    let prev_is_end = sc_b.prev_ci >= 0
                        && (flags[sc_b.prev_ci as usize] & IS_ENDPOINT) != 0;
                    let next_is_end = sc_b.next_ci >= 0
                        && (flags[sc_b.next_ci as usize] & IS_ENDPOINT) != 0;
                    if prev_is_end { Some(0.0) }
                    else if next_is_end { Some(1.0) }
                    else { None }  // shouldn't happen for a real slot-1 neighbor
                };
                let mut through_h: Option<usize> = None;
                let mut stem_h: Option<usize> = None;
                let mut t_b_eval: f32 = 0.0;
                if num_hits >= 2 && hit_d2[1] < aa_threshold {
                    let cp0 = &all_cps[hit_idx[0] as usize];
                    let cp1 = &all_cps[hit_idx[1] as usize];
                    let cp0_t = if (cp0.flag & IS_TJUNCTION) != 0 {
                        slot1_neighbor_t(cp0, cp1)
                    } else {
                        None
                    };
                    let cp1_t = if (cp1.flag & IS_TJUNCTION) != 0 {
                        slot1_neighbor_t(cp1, cp0)
                    } else {
                        None
                    };
                    let both_x_same =
                        (cp0.flag & IS_CROSSING) != 0
                        && (cp1.flag & IS_CROSSING) != 0
                        && cp0.icx == cp1.icx
                        && cp0.icy == cp1.icy;
                    if let Some(t) = cp0_t {
                        through_h = Some(0);
                        stem_h = Some(1);
                        t_b_eval = t;
                    } else if let Some(t) = cp1_t {
                        through_h = Some(1);
                        stem_h = Some(0);
                        t_b_eval = t;
                    } else if both_x_same {
                        // Crossings: sc_b's curve passes through J at t_branch.
                        through_h = Some(0);
                        stem_h = Some(1);
                        t_b_eval = all_cps[hit_idx[1] as usize].t_branch;
                    }
                }
                // The wedge AA model assumes the 3-color region (the wedge
                // emanating from the junction) overlaps the pixel. That's
                // true only when the junction itself — the geometric meeting
                // point of the through and stem curves, which equals the
                // through CP's pos post-correction — lies inside the pixel
                // square. When two curves pass near a pixel but their junction
                // lies outside it, the local geometry is 2-color and the
                // wedge LUT mixes a third color that isn't actually present
                // there. Fall back to single-curve AA in that case.
                // TODO: real handling for multi-curve pixels where the
                // junction is outside (e.g. two parallel boundaries cutting
                // the same pixel). For now they get single-curve AA, which
                // is at least free of the spurious-third-color artifact.
                if let Some(ah) = through_h {
                    let sc_a = if ah == 0 { &all_cps[hit_idx[0] as usize] }
                                          else { &all_cps[hit_idx[1] as usize] };
                    let j = beval(sc_a.prev_pos, sc_a.pos, sc_a.next_pos, sc_a.t_branch);
                    let pix_half = inv_scale * 0.5;
                    let in_pixel = (j.0 - center.0).abs() <= pix_half
                                && (j.1 - center.1).abs() <= pix_half;
                    if !in_pixel {
                        through_h = None;
                        stem_h = None;
                    }
                }
                // line_a = through curve, line_b = stem render curve.
                //
                // Junction-anchored: both lines pass through the actual
                // junction J = sc_a.pos (= geometric meeting point of the
                // through and stem curves post-correction). Tangents are
                // evaluated at the t value where each curve passes nearest J:
                //   - line_a (through): t = sc_a.t_branch (curve passes
                //     through its own pos at t_branch by construction).
                //   - line_b (stem): if sc_b.pos coincides with J (2-CP-chain
                //     stem at slot 1 of junction corner, or crossing partner
                //     at same corner), use sc_b.t_branch. Otherwise (regular
                //     stem at adjacent corner), use closest_t on sc_b's curve
                //     to J.
                // This makes the 4-wedge partition geometrically meaningful
                // (both lines literally cross at J) and lets sb-sign directly
                // separate seg0 (pre-junction) from seg1 (post-junction)
                // along line_a, eliminating the b_separates / center_in_seg1
                // fallback.
                let mut line_a: Option<AaLine> = None;
                let mut line_b: Option<AaLine> = None;
                let mut line_a_hit: (u32, f32) = (0, 0.0);
                if let (Some(ah), Some(bh)) = (through_h, stem_h) {
                    // ah, bh ∈ {0, 1}; explicit selection.
                    let sc_a = if ah == 0 { &all_cps[hit_idx[0] as usize] }
                                          else { &all_cps[hit_idx[1] as usize] };
                    let sc_b = if bh == 0 { &all_cps[hit_idx[0] as usize] }
                                          else { &all_cps[hit_idx[1] as usize] };
                    // J = beval(sc_a, t_branch) — point on the rendered through
                    // curve at the junction parameter (NOT sc_a.pos, which is
                    // a polynomial control point and may differ from the curve
                    // for non-ghost-extended CPs).
                    let j = beval(sc_a.prev_pos, sc_a.pos, sc_a.next_pos, sc_a.t_branch);
                    let la = build_junction_aa_line(sc_a, sc_a.t_branch, j, pixels, img_w, img_h);
                    // t_b_eval was determined structurally by the gate above.
                    let lb = build_junction_aa_line(sc_b, t_b_eval, j, pixels, img_w, img_h);
                    if let (Some((la_line, tang_a_unit)), Some((mut lb_line, _))) = (la, lb) {
                        // Align line_b.normal so sb > 0 means "in tang_a
                        // direction" along the through curve, i.e., post-
                        // junction (seg 1). If lb's normal opposes tang_a,
                        // flip it so sb-sign tracks pre/post consistently.
                        if lb_line.normal.0 * tang_a_unit.0
                           + lb_line.normal.1 * tang_a_unit.1 < 0.0 {
                            lb_line.normal = (-lb_line.normal.0, -lb_line.normal.1);
                        }
                        line_a = Some(la_line);
                        line_b = Some(lb_line);
                        line_a_hit = if ah == 0 { (hit_idx[0], hit_t[0]) }
                                                else { (hit_idx[1], hit_t[1]) };
                    }
                }

                if !need_aa {
                    chunk[local_y * out_w + opx] = pack_color(center_color);
                } else if let (Some(line_a), Some(line_b)) = (line_a.as_ref(), line_b.as_ref()) {
                    // Wedge AA model
                    // ==============
                    // line_a: through-curve tangent at the junction. Carries up
                    //   to 4 colors via t_branch — seg 0 (t < t_branch) uses
                    //   prev edge colors, seg 1 (t >= t_branch) uses next.
                    //   sa-sign picks pos vs neg side of line_a within the
                    //   chosen segment.
                    // line_b: stem-curve tangent at the junction. Aligned so
                    //   sb > 0 lies in the tang_a direction along line_a =
                    //   post-junction (seg 1).
                    //
                    // Boundary-sweep area accumulation: walk the pixel
                    // boundary, find line_a/line_b crossings on each edge,
                    // accumulate signed-triangle areas (J as origin) into 4
                    // wedge buckets. The sub-segment midpoint sign-tests give
                    // the (sa, sb) wedge directly — no centroid pass.
                    let (a_idx, _a_t) = line_a_hit;
                    let sc_a = &all_cps[a_idx as usize];

                    // Through-curve color LUT (seg0 / seg1 × pos / neg).
                    let uniform_pn = (center_color, center_color);
                    let (pos_seg0, neg_seg0) =
                        resolve_lut_segment(sc_a, line_a.normal, pixels, img_w, img_h, false)
                            .unwrap_or(uniform_pn);
                    let (pos_seg1, neg_seg1) =
                        resolve_lut_segment(sc_a, line_a.normal, pixels, img_w, img_h, true)
                            .unwrap_or(uniform_pn);
                    // LUT indexed by (sa>0) | ((sb>0) << 1):
                    //   0 = sa<0,sb<0 = neg_seg0
                    //   1 = sa>0,sb<0 = pos_seg0
                    //   2 = sa<0,sb>0 = neg_seg1
                    //   3 = sa>0,sb>0 = pos_seg1
                    let lut: [(f32, f32, f32); 4] = [
                        srgb_decode(neg_seg0),
                        srgb_decode(pos_seg0),
                        srgb_decode(neg_seg1),
                        srgb_decode(pos_seg1),
                    ];

                    let pix_h = inv_scale * 0.5;
                    let corners: [(f32, f32); 4] = [
                        (center.0 - pix_h, center.1 - pix_h),
                        (center.0 + pix_h, center.1 - pix_h),
                        (center.0 + pix_h, center.1 + pix_h),
                        (center.0 - pix_h, center.1 + pix_h),
                    ];
                    let j = line_a.cpt;
                    let sa_at = |p: (f32, f32)| -> f32 {
                        (p.0 - line_a.cpt.0) * line_a.normal.0
                            + (p.1 - line_a.cpt.1) * line_a.normal.1
                    };
                    let sb_at = |p: (f32, f32)| -> f32 {
                        (p.0 - line_b.cpt.0) * line_b.normal.0
                            + (p.1 - line_b.cpt.1) * line_b.normal.1
                    };

                    let mut sum_r = 0.0f32;
                    let mut sum_g = 0.0f32;
                    let mut sum_b = 0.0f32;
                    let mut total_area = 0.0f32;

                    let accumulate = |u: (f32, f32), v: (f32, f32),
                                          sa_m: f32, sb_m: f32,
                                          sum_r: &mut f32, sum_g: &mut f32,
                                          sum_b: &mut f32, total: &mut f32| {
                        let ux = u.0 - j.0;
                        let uy = u.1 - j.1;
                        let vx = v.0 - j.0;
                        let vy = v.1 - j.1;
                        // J is interior, walk is consistent; abs covers
                        // either Y-up or Y-down convention.
                        let area = 0.5 * (ux * vy - uy * vx).abs();
                        if area < 1e-9 { return; }
                        let idx = (if sa_m > 0.0 { 1 } else { 0 })
                            | (if sb_m > 0.0 { 2 } else { 0 });
                        let (rl, gl, bl) = lut[idx];
                        *sum_r += rl * area;
                        *sum_g += gl * area;
                        *sum_b += bl * area;
                        *total += area;
                    };

                    for i in 0..4 {
                        let p = corners[i];
                        let q = corners[(i + 1) & 3];
                        let sa_p = sa_at(p);
                        let sa_q = sa_at(q);
                        let sb_p = sb_at(p);
                        let sb_q = sb_at(q);

                        // Up to 2 crossings per edge (one per line, if signs flip).
                        let mut splits: [f32; 2] = [0.0; 2];
                        let mut nsplit = 0usize;
                        if sa_p * sa_q < 0.0 {
                            splits[nsplit] = sa_p / (sa_p - sa_q);
                            nsplit += 1;
                        }
                        if sb_p * sb_q < 0.0 {
                            splits[nsplit] = sb_p / (sb_p - sb_q);
                            nsplit += 1;
                        }
                        if nsplit == 2 && splits[0] > splits[1] {
                            splits.swap(0, 1);
                        }

                        let edge_pt = |t: f32| -> (f32, f32) {
                            (p.0 + t * (q.0 - p.0), p.1 + t * (q.1 - p.1))
                        };

                        // Walk sub-segments [0, splits[0]], [splits[0], splits[1]], [splits[1], 1].
                        let mut t_prev = 0.0f32;
                        let mut u = p;
                        for k in 0..nsplit {
                            let t_curr = splits[k];
                            if t_curr > t_prev + 1e-9 {
                                let v = edge_pt(t_curr);
                                let m = edge_pt(0.5 * (t_prev + t_curr));
                                accumulate(u, v, sa_at(m), sb_at(m),
                                    &mut sum_r, &mut sum_g, &mut sum_b, &mut total_area);
                                u = v;
                            }
                            t_prev = t_curr;
                        }
                        if t_prev < 1.0 - 1e-9 {
                            let m = edge_pt(0.5 * (t_prev + 1.0));
                            accumulate(u, q, sa_at(m), sb_at(m),
                                &mut sum_r, &mut sum_g, &mut sum_b, &mut total_area);
                        }
                    }

                    chunk[local_y * out_w + opx] = if total_area > 0.0 {
                        let inv = 1.0 / total_area;
                        srgb_encode_argb(sum_r * inv, sum_g * inv, sum_b * inv)
                    } else {
                        pack_color(center_color)
                    };
                } else if let Some(packed) = ({
                    // Dual-curve AA: two distinct-chain curves passing through
                    // the pixel without an actual junction inside it.
                    //
                    // line_a: tangent at sc_a's closest_t (= aa_idx hit).
                    // line_b: tangent at sc_b's closest_t (= first non-chain
                    //   neighbor hit within aa_threshold).
                    // K = intersection of the two lines (may be outside pixel).
                    //
                    // Boundary-sweep with K as origin: each line leg of a
                    // wedge polygon is on a ray through K, so its cross-
                    // product contribution is zero. Pixel-edge sub-segments
                    // alone reproduce 2·area(wedge) when summed signed (no
                    // abs per segment — the closed-loop math gives the
                    // correct sign for K outside the pixel).
                    //
                    // Returns Some(packed_color) on success, None for any
                    // degeneracy (no valid second curve, parallel lines,
                    // missing edge colors).
                    if aa_idx < 0 { None }
                    else {
                        let sc_a = &all_cps[aa_idx as usize];
                        // Pick sc_b: first hit not on sc_a's chain, within
                        // threshold. Hits are sorted ascending by d2, so we
                        // can break once d2 exceeds threshold.
                        // Pick first non-chain hit, unrolled across constant
                        // indices for consistency with the GPU rasterizer.
                        let mut bh: Option<usize> = None;
                        macro_rules! try_sec {
                            ($s:expr, $hi:expr, $ht:expr, $hd:expr) => {
                                if bh.is_none() && $s < num_hits
                                    && $hi as i32 != aa_idx
                                    && $hd < aa_threshold {
                                    let cp_b = &all_cps[$hi as usize];
                                    let same_chain = sc_a.ci == cp_b.ci
                                        || sc_a.ci == cp_b.prev_ci
                                        || sc_a.ci == cp_b.next_ci
                                        || cp_b.ci == sc_a.prev_ci
                                        || cp_b.ci == sc_a.next_ci;
                                    if !same_chain { bh = Some($s); }
                                }
                            };
                        }
                        try_sec!(0, hit_idx[0], hit_t[0], hit_d2[0]);
                        try_sec!(1, hit_idx[1], hit_t[1], hit_d2[1]);
                        try_sec!(2, hit_idx[2], hit_t[2], hit_d2[2]);
                        bh.and_then(|bh| {
                            // bh ∈ {0, 1, 2}; explicit selection.
                            let (b_idx, bt) = match bh {
                                0 => (hit_idx[0], hit_t[0]),
                                1 => (hit_idx[1], hit_t[1]),
                                _ => (hit_idx[2], hit_t[2]),
                            };
                            let sc_b = &all_cps[b_idx as usize];

                            let (la, a_pos, a_neg) =
                                build_aa_line(sc_a, aa_t, pixels, img_w, img_h)?;
                            let (lb, b_pos, b_neg) =
                                build_aa_line(sc_b, bt, pixels, img_w, img_h)?;
                            if a_pos == a_neg && b_pos == b_neg { return None; }

                            // Inner-side: which side of curve A faces curve B?
                            // dot(cpt_b - cpt_a, na) > 0 → B is on A's pos side.
                            let delta_ba = (lb.cpt.0 - la.cpt.0, lb.cpt.1 - la.cpt.1);
                            let inner_a_pos = delta_ba.0 * la.normal.0 + delta_ba.1 * la.normal.1 > 0.0;
                            let inner_b_pos = -delta_ba.0 * lb.normal.0 + -delta_ba.1 * lb.normal.1 > 0.0;
                            let a_inner = if inner_a_pos { a_pos } else { a_neg };
                            let a_outer = if inner_a_pos { a_neg } else { a_pos };
                            let b_inner = if inner_b_pos { b_pos } else { b_neg };
                            let b_outer = if inner_b_pos { b_neg } else { b_pos };

                            // Dual-curve assumes the two curves border a
                            // shared middle region, so A_inner == B_inner.
                            // When they disagree, the local geometry has 3
                            // distinct colors (near-junction missed by the
                            // structural detector — e.g. two chains terminating
                            // at the same point just outside the pixel).
                            // Defer to single-curve AA.
                            if a_inner != b_inner { return None; }

                            // 3-stripe formula. Lines that don't cross in the
                            // pixel partition it into A_outer | middle | B_outer
                            // (A_outer ∩ B_outer is empty, so the areas sum to
                            // the pixel). Each outer fraction is a single-line
                            // coverage — closed-form, no boundary sweep.
                            let d_a = ((center.0 - la.cpt.0) * la.normal.0
                                     + (center.1 - la.cpt.1) * la.normal.1) / inv_scale;
                            let d_b = ((center.0 - lb.cpt.0) * lb.normal.0
                                     + (center.1 - lb.cpt.1) * lb.normal.1) / inv_scale;
                            let frac_a_pos = line_coverage_pos(la.normal, d_a);
                            let frac_b_pos = line_coverage_pos(lb.normal, d_b);
                            let frac_a_outer = if inner_a_pos { 1.0 - frac_a_pos } else { frac_a_pos };
                            let frac_b_outer = if inner_b_pos { 1.0 - frac_b_pos } else { frac_b_pos };

                            // Lines crossing inside the pixel → A_outer and
                            // B_outer regions overlap, fractions sum > 1.
                            // Defer to single-curve AA.
                            if frac_a_outer + frac_b_outer > 1.0 + 1e-4 {
                                return None;
                            }
                            let frac_middle = (1.0 - frac_a_outer - frac_b_outer).max(0.0);

                            // Blend in linear light.
                            let (a_or, a_og, a_ob) = srgb_decode(a_outer);
                            let (b_or, b_og, b_ob) = srgb_decode(b_outer);
                            let (m_r, m_g, m_b) = srgb_decode(a_inner);
                            let r = a_or * frac_a_outer + b_or * frac_b_outer + m_r * frac_middle;
                            let g = a_og * frac_a_outer + b_og * frac_b_outer + m_g * frac_middle;
                            let b = a_ob * frac_a_outer + b_ob * frac_b_outer + m_b * frac_middle;
                            Some(srgb_encode_argb(r, g, b))
                        })
                    }
                }) {
                    chunk[local_y * out_w + opx] = packed;
                } else {
                    // Single-curve AA fallback: tangent line + pixel coverage
                    // formula. build_aa_line bundles geometry + color
                    // resolution; line_coverage_pos gives the closed-form
                    // area on the line's positive side.
                    let sc = &all_cps[aa_idx as usize];
                    match build_aa_line(sc, aa_t, pixels, img_w, img_h) {
                        Some((line, pos_side, neg_side)) if pos_side != neg_side => {
                            let d_p = ((center.0 - line.cpt.0) * line.normal.0
                                     + (center.1 - line.cpt.1) * line.normal.1) / inv_scale;
                            let frac = line_coverage_pos(line.normal, d_p);
                            let (r0, g0, b0) = srgb_decode(pos_side);
                            let (r1, g1, b1) = srgb_decode(neg_side);
                            let inv_frac = 1.0 - frac;
                            chunk[local_y * out_w + opx] = srgb_encode_argb(
                                frac * r0 + inv_frac * r1,
                                frac * g0 + inv_frac * g1,
                                frac * b0 + inv_frac * b1,
                            );
                        }
                        _ => chunk[local_y * out_w + opx] = pack_color(center_color),
                    }
                }
            } // opx
        } // local_y
    };

    // Parallel on native, sequential on wasm
    #[cfg(not(target_arch = "wasm32"))]
    {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let rows_per_thread = out_h.div_ceil(num_threads);
        std::thread::scope(|scope| {
            let chunks: Vec<&mut [u32]> = output.chunks_mut(out_w * rows_per_thread).collect();
            let all_cps = &all_cps;
            let ci_to_all_cps = &ci_to_all_cps;
            let handles: Vec<_> = chunks.into_iter().enumerate().map(|(ci, chunk)| {
                let start = ci * rows_per_thread;
                scope.spawn(move || {
                    rasterize_rows(chunk, start, all_cps, ci_to_all_cps);
                })
            }).collect();
            for h in handles { h.join().unwrap(); }
        });
    }
    #[cfg(target_arch = "wasm32")]
    {
        rasterize_rows(&mut output, 0, &all_cps, &ci_to_all_cps);
    }

    output
}

/// Pack ARGB color (shader outputs RGB components divided by 255, then stored;
/// here we just ensure alpha is 0xFF).
#[inline(always)]
fn pack_color(c: u32) -> u32 {
    0xFF000000 | (c & 0x00FFFFFF)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect_in_unit(roots: [Option<f32>; 4]) -> Vec<f32> {
        let mut v: Vec<f32> = roots.into_iter().flatten().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    }

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn assert_root(quartic_roots: &[f32], expected: f32, eps: f32) {
        assert!(
            quartic_roots.iter().any(|r| approx_eq(*r, expected, eps)),
            "expected root {expected} not found in {quartic_roots:?}",
        );
    }

    // ---------- quartic solver ----------

    #[test]
    fn quartic_biquadratic_two_unit_roots() {
        // (t² − 0.25)(t² − 0.49) = 0  →  roots ±0.5, ±0.7. In [0,1]: 0.5, 0.7.
        // Expanded: t⁴ − 0.74·t² + 0.1225 = 0
        let roots = collect_in_unit(solve_quartic_in_unit(1.0, 0.0, -0.74, 0.0, 0.1225));
        assert_eq!(roots.len(), 2);
        assert_root(&roots, 0.5, 1e-4);
        assert_root(&roots, 0.7, 1e-4);
    }

    #[test]
    fn quartic_four_distinct_unit_roots() {
        // (t − 0.1)(t − 0.4)(t − 0.6)(t − 0.9) = 0
        // (t² − 0.5t + 0.04)(t² − 1.5t + 0.54)
        // = t⁴ − 2.0·t³ + 1.33·t² − 0.33·t + 0.0216
        let roots = collect_in_unit(solve_quartic_in_unit(1.0, -2.0, 1.33, -0.33, 0.0216));
        assert_eq!(roots.len(), 4);
        for expected in [0.1, 0.4, 0.6, 0.9] {
            assert_root(&roots, expected, 5e-3);
        }
    }

    #[test]
    fn quartic_no_real_roots_in_unit() {
        // (t² + 1)² = 0 has only complex roots.
        let roots = collect_in_unit(solve_quartic_in_unit(1.0, 0.0, 2.0, 0.0, 1.0));
        assert!(roots.is_empty());
    }

    #[test]
    fn quartic_single_unit_root_with_others_outside() {
        // (t − 0.3)(t + 1)(t² + 4) = 0  →  only 0.3 in [0,1].
        // = (t² + 0.7t − 0.3)(t² + 4) = t⁴ + 0.7t³ + 3.7t² + 2.8t − 1.2
        let roots = collect_in_unit(solve_quartic_in_unit(1.0, 0.7, 3.7, 2.8, -1.2));
        assert_eq!(roots.len(), 1);
        assert_root(&roots, 0.3, 1e-4);
    }

    #[test]
    fn quartic_degenerate_cubic() {
        // (t − 0.4)(t − 0.6)(t − 2) = t³ − 3·t² + 2.24·t − 0.48
        let roots = collect_in_unit(solve_quartic_in_unit(0.0, 1.0, -3.0, 2.24, -0.48));
        assert_eq!(roots.len(), 2);
        assert_root(&roots, 0.4, 1e-4);
        assert_root(&roots, 0.6, 1e-4);
    }

    // ---------- B-spline span intersection ----------

    /// Helper: evaluate a quadratic uniform B-spline span at t.
    fn beval2(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), t: f32) -> (f32, f32) {
        let u = 1.0 - t;
        (
            0.5 * u * u * p0.0 + (u * t + 0.5) * p1.0 + 0.5 * t * t * p2.0,
            0.5 * u * u * p0.1 + (u * t + 0.5) * p1.1 + 0.5 * t * t * p2.1,
        )
    }

    #[test]
    fn intersect_axis_aligned_through_corner() {
        // Curve A: vertical (P0=top, P1=mid, P2=bot), all at x=0.
        // Curve B: horizontal, all at y=0.
        // Intersection at origin. Both span centers (P1) are at origin so curves
        // are straight; intersection should land at (0,0) with t=s=0.5.
        let r = intersect_quadratic_bsplines(
            (0.0, -1.0), (0.0, 0.0), (0.0, 1.0),
            (-1.0, 0.0), (0.0, 0.0), (1.0, 0.0),
        );
        assert!(approx_eq(r.p.0, 0.0, 1e-4) && approx_eq(r.p.1, 0.0, 1e-4),
            "p = {:?}", r.p);
        assert!(approx_eq(r.t_a, 0.5, 1e-4));
        assert!(approx_eq(r.t_b, 0.5, 1e-4));
    }

    #[test]
    fn intersect_offset_cps_recovers_actual_curves() {
        // Two curves with off-center mid-CPs (the realistic case after the
        // optimizer moves things). Intersection point and t-values should be
        // such that B_a(t_a) == B_b(t_b) within tolerance.
        let a0 = (-1.0, -1.2);
        let a1 = ( 0.1,  0.05);
        let a2 = ( 0.9,  1.4);
        let b0 = (-1.3,  1.0);
        let b1 = (-0.05, 0.1);
        let b2 = ( 1.2, -0.9);

        let r = intersect_quadratic_bsplines(a0, a1, a2, b0, b1, b2);

        let pa = beval2(a0, a1, a2, r.t_a);
        let pb = beval2(b0, b1, b2, r.t_b);
        // Both evaluated points should equal r.p and each other.
        assert!(approx_eq(pa.0, r.p.0, 1e-3), "pa.x={} r.p.x={}", pa.0, r.p.0);
        assert!(approx_eq(pa.1, r.p.1, 1e-3), "pa.y={} r.p.y={}", pa.1, r.p.1);
        assert!(approx_eq(pb.0, r.p.0, 1e-3), "pb.x={} r.p.x={}", pb.0, r.p.0);
        assert!(approx_eq(pb.1, r.p.1, 1e-3), "pb.y={} r.p.y={}", pb.1, r.p.1);
        assert!((0.0..=1.0).contains(&r.t_a));
        assert!((0.0..=1.0).contains(&r.t_b));
    }

    #[test]
    fn intersect_curved_x_pattern() {
        // X-shape with curvature: NW-SE curve and SW-NE curve, mid-CPs offset
        // off the corner. Verify the returned (p, t_a, t_b) satisfy both curves.
        let a0 = (-2.0, -2.0);
        let a1 = ( 0.0,  0.3);
        let a2 = ( 2.0,  2.0);
        let b0 = (-2.0,  2.0);
        let b1 = (-0.2,  0.0);
        let b2 = ( 2.0, -2.0);

        let r = intersect_quadratic_bsplines(a0, a1, a2, b0, b1, b2);

        let pa = beval2(a0, a1, a2, r.t_a);
        let pb = beval2(b0, b1, b2, r.t_b);
        let dx = pa.0 - pb.0;
        let dy = pa.1 - pb.1;
        assert!(dx * dx + dy * dy < 1e-4,
            "B_a(t_a)={pa:?}  B_b(t_b)={pb:?}  delta=({dx},{dy})");
    }
}
