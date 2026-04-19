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
pub struct VectorizeData {
    pub positions: Vec<f32>,
    pub orig_positions: Vec<f32>,
    pub neighbors: Vec<i32>,
    pub flags: Vec<u32>,
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
    update_tjunctions(&mut positions, &neighbors, &flags, num_cps);

    VectorizeData {
        positions,
        orig_positions,
        neighbors,
        flags,
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
    // that extend to the image border.
    for i in 0..num_cps {
        if flags[i] == 0 { continue; }
        let prev = neighbors[i * 4];
        let next = neighbors[i * 4 + 1];
        if prev >= 0 {
            let pi = prev as usize;
            if flags[pi] == 1 && neighbors[pi * 4] < 0 && neighbors[pi * 4 + 1] < 0 {
                neighbors[pi * 4 + 1] = i as i32;
                let d = neighbors[i * 4 + 2];
                if d >= 0 { neighbors[pi * 4 + 3] = (d + 2) % 4; }
            }
        }
        if next >= 0 {
            let ni = next as usize;
            if flags[ni] == 1 && neighbors[ni * 4] < 0 && neighbors[ni * 4 + 1] < 0 {
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

            // Skip non-crossing nodes adjacent to crossings (crossing
            // position depends on all 4 neighbors; optimizing one side
            // without the other would break symmetry). Crossing nodes
            // themselves must NOT be skipped — they handle their own
            // combined-curvature optimization.
            if (f & IS_CROSSING) == 0 {
                if (flags[prev_idx as usize] & IS_CROSSING) != 0 {
                    continue;
                }
                if (flags[next_idx as usize] & IS_CROSSING) != 0 {
                    continue;
                }
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
// Stage 5: Update T-junctions
// ============================================================================

fn update_tjunctions(positions: &mut [f32], neighbors: &[i32], flags: &[u32], num_cps: usize) {
    // Unified iterative solver matching the GPU shader (update_tjunction.slang).
    // Save original positions, then iterate 3 passes. Each pass processes ALL
    // junction CPs in one loop:
    //   - Crossings: inverse B-spline correction (each slot independently —
    //     different pp/np require different corrected positions for both curves
    //     to pass through the same grid_pos at t=0.5)
    //   - T-junctions: forward B-spline correction + stem snapping
    // Both use orig for the center weight and latest neighbor positions.
    // 3 passes converge to sub-pixel accuracy (contraction ≈ 0.17/pass).
    let orig = positions.to_vec();
    for _ in 0..3 {
        for i in 0..num_cps {
            let f = flags[i];
            let prev_idx = neighbors[i * 4];
            let next_idx = neighbors[i * 4 + 1];
            if prev_idx < 0 || next_idx < 0 { continue; }

            let prev_pos = (positions[prev_idx as usize * 2], positions[prev_idx as usize * 2 + 1]);
            let next_pos = (positions[next_idx as usize * 2], positions[next_idx as usize * 2 + 1]);
            let grid_pos = (orig[i * 2], orig[i * 2 + 1]);

            if (f & IS_CROSSING) != 0 {
                let corrected = (
                    (grid_pos.0 - 0.125 * prev_pos.0 - 0.125 * next_pos.0) / 0.75,
                    (grid_pos.1 - 0.125 * prev_pos.1 - 0.125 * next_pos.1) / 0.75,
                );
                positions[i * 2] = corrected.0;
                positions[i * 2 + 1] = corrected.1;
            } else if (f & IS_TJUNCTION) != 0 {
                let corrected = (
                    0.125 * prev_pos.0 + 0.75 * grid_pos.0 + 0.125 * next_pos.0,
                    0.125 * prev_pos.1 + 0.75 * grid_pos.1 + 0.125 * next_pos.1,
                );
                positions[i * 2] = corrected.0;
                positions[i * 2 + 1] = corrected.1;

                let stem = i ^ 1;
                if stem < num_cps && flags[stem] == 1 {
                    positions[stem * 2] = 0.125 * prev_pos.0 + 0.75 * corrected.0 + 0.125 * next_pos.0;
                    positions[stem * 2 + 1] = 0.125 * prev_pos.1 + 0.75 * corrected.1 + 0.125 * next_pos.1;
                }
            }
        }
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
            // Quadratic: c2*t² + c1*t + c0 = 0
            let disc = c1 * c1 - 4.0 * c2 * c0;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                let inv = 0.5 / c2;
                for t in [(-c1 + sq) * inv, (-c1 - sq) * inv] {
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
            let r = (-p * p * p / 27.0).sqrt();
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
struct CpData {
    pos: (f32, f32),
    orig_pos: (f32, f32),
    prev_pos: (f32, f32),
    next_pos: (f32, f32),
    orig_prev: (f32, f32),
    orig_next: (f32, f32),
    bbox_min: (f32, f32),
    bbox_max: (f32, f32),
    /// Precomputed polynomial coefficients: B(t) = a*t² + b*t + c
    /// B'(t) = 2*a*t + b, B''(t) = 2*a
    poly_ax: f32, poly_ay: f32,
    poly_bx: f32, poly_by: f32,
    poly_cx: f32, poly_cy: f32,
    prev_dir: i32,
    next_dir: i32,
    icx: i32,
    icy: i32,
    prev_ci: i32,
    next_ci: i32,
}

fn rasterize(
    pixels: &[u32],
    positions: &[f32],
    orig_positions: &[f32],
    flags: &[u32],
    cp_neighbors: &[i32],
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
        if prev_ci < 0 && next_ci < 0 {
            continue;
        }

        let cp = read_pos_f(ci as i32);
        let pp = if prev_ci >= 0 { read_pos_f(prev_ci) } else { cp };
        let np = if next_ci >= 0 { read_pos_f(next_ci) } else { cp };

        let bbox_min = (pp.0.min(cp.0).min(np.0), pp.1.min(cp.1).min(np.1));
        let bbox_max = (pp.0.max(cp.0).max(np.0), pp.1.max(cp.1).max(np.1));
        // Polynomial form: B(t) = a*t² + b*t + c
        // a = 0.5*(p0 - 2*p1 + p2), b = p1 - p0, c = 0.5*(p0 + p1)
        let poly_ax = 0.5 * (pp.0 - 2.0 * cp.0 + np.0);
        let poly_ay = 0.5 * (pp.1 - 2.0 * cp.1 + np.1);
        let poly_bx = cp.0 - pp.0;
        let poly_by = cp.1 - pp.1;
        let poly_cx = 0.5 * (pp.0 + cp.0);
        let poly_cy = 0.5 * (pp.1 + cp.1);

        all_cps.push(CpData {
            pos: cp,
            orig_pos: read_orig_f(ci as i32),
            prev_pos: pp,
            next_pos: np,
            orig_prev: if prev_ci >= 0 {
                read_orig_f(prev_ci)
            } else {
                read_orig_f(ci as i32)
            },
            orig_next: if next_ci >= 0 {
                read_orig_f(next_ci)
            } else {
                read_orig_f(ci as i32)
            },
            bbox_min,
            bbox_max,
            poly_ax, poly_ay, poly_bx, poly_by, poly_cx, poly_cy,
            prev_dir: cp_neighbors[ci * 4 + 2],
            next_dir: cp_neighbors[ci * 4 + 3],
            icx: (ci / 2 % corners_w) as i32,
            icy: (ci / 2 / corners_w) as i32,
            prev_ci,
            next_ci,
        });
    }

    // Build spatial grid for fast CP lookup.
    // 1×1 source-pixel cells for tighter spatial queries.
    // Use a flat packed array instead of Vec<Vec<>> to avoid pointer chasing.
    let grid_cell = 1.0f32;
    let grid_w = ((img_w as f32 / grid_cell).ceil() as usize).max(1);
    let grid_h = ((img_h as f32 / grid_cell).ceil() as usize).max(1);
    let grid_total = grid_w * grid_h;

    // Pass 1: count CPs per cell
    let mut cell_count = vec![0u16; grid_total];
    for sc in &all_cps {
        let min_gx = ((sc.bbox_min.0 - 1.0) / grid_cell).floor().max(0.0) as usize;
        let min_gy = ((sc.bbox_min.1 - 1.0) / grid_cell).floor().max(0.0) as usize;
        let max_gx = ((sc.bbox_max.0 + 1.0) / grid_cell).floor() as usize;
        let max_gy = ((sc.bbox_max.1 + 1.0) / grid_cell).floor() as usize;
        for gy in min_gy..=max_gy.min(grid_h - 1) {
            for gx in min_gx..=max_gx.min(grid_w - 1) {
                cell_count[gy * grid_w + gx] += 1;
            }
        }
    }

    // Pass 2: compute offsets (prefix sum)
    let mut cell_offset = vec![0u32; grid_total + 1];
    for i in 0..grid_total {
        cell_offset[i + 1] = cell_offset[i] + cell_count[i] as u32;
    }
    let total_entries = cell_offset[grid_total] as usize;

    // Pass 3: fill flat array
    let mut grid_data = vec![0u16; total_entries];
    let mut fill_pos = cell_offset[..grid_total].to_vec();
    for (i, sc) in all_cps.iter().enumerate() {
        let min_gx = ((sc.bbox_min.0 - 1.0) / grid_cell).floor().max(0.0) as usize;
        let min_gy = ((sc.bbox_min.1 - 1.0) / grid_cell).floor().max(0.0) as usize;
        let max_gx = ((sc.bbox_max.0 + 1.0) / grid_cell).floor() as usize;
        let max_gy = ((sc.bbox_max.1 + 1.0) / grid_cell).floor() as usize;
        for gy in min_gy..=max_gy.min(grid_h - 1) {
            for gx in min_gx..=max_gx.min(grid_w - 1) {
                let ci = gy * grid_w + gx;
                let pos = fill_pos[ci] as usize;
                grid_data[pos] = i as u16;
                fill_pos[ci] += 1;
            }
        }
    }

    let mut output = vec![0u32; out_w * out_h];

    // Compute edge colors on-the-fly from pixel buffer
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

    let resolve_from_cp =
        |pt: (f32, f32), sc: &CpData, t: f32| -> Option<u32> {
            let (mut pl, mut pr) = (0u32, 0u32);
            let (mut nl, mut nr) = (0u32, 0u32);
            if sc.prev_dir >= 0 { let (l, r) = get_edge_colors(sc.icx, sc.icy, sc.prev_dir); pl = l; pr = r; }
            if sc.next_dir >= 0 { let (l, r) = get_edge_colors(sc.icx, sc.icy, sc.next_dir); nl = l; nr = r; }

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
            } else if t < 0.5 {
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
    let aa_threshold = 2.0 / (scale_factor * scale_factor);

    // Rasterize row range into output chunk
    let rasterize_rows = |chunk: &mut [u32], start: usize,
        all_cps: &[CpData], grid_data: &[u16], cell_offset: &[u32]|
    {
                let chunk_rows = chunk.len() / out_w;
                for local_y in 0..chunk_rows {
                    let opy = start + local_y;
                    let center_y = (opy as f32 + 0.5) * inv_scale;
                    let fb_y = (center_y.floor() as i32).clamp(0, img_h as i32 - 1) as usize;
                    let gy = (center_y.floor() as usize).min(grid_h - 1);
                    let gy_offset = gy * grid_w;

                    for opx in 0..out_w {
            let center_x = (opx as f32 + 0.5) * inv_scale;
            let center = (center_x, center_y);
            let fb_x = (center_x.floor() as i32).clamp(0, img_w as i32 - 1) as usize;
            let fallback = pixels[fb_y * img_w + fb_x];

            let gx = (center_x.floor() as usize).min(grid_w - 1);
            let cell_idx = gy_offset + gx;
            let cell_start = cell_offset[cell_idx] as usize;
            let cell_end = cell_offset[cell_idx + 1] as usize;

            // Fixed-size top-4 hit array (avoids Vec allocation per pixel)
            let mut hit_d2 = [1e10f32; 4];
            let mut hit_t = [0.0f32; 4];
            let mut hit_idx = [0u32; 4];
            let mut num_hits = 0usize;

            for ci in cell_start..cell_end {
                let cp_i = grid_data[ci];
                let sc = &all_cps[cp_i as usize];

                // Quick screen: 3-sample reject (matches GPU bspline_quick_screen)
                let b0 = (sc.poly_cx, sc.poly_cy); // beval at t=0
                let bm = (0.125 * sc.prev_pos.0 + 0.75 * sc.pos.0 + 0.125 * sc.next_pos.0,
                           0.125 * sc.prev_pos.1 + 0.75 * sc.pos.1 + 0.125 * sc.next_pos.1);
                let b1 = (0.5 * (sc.pos.0 + sc.next_pos.0), 0.5 * (sc.pos.1 + sc.next_pos.1));
                let qd0 = (center.0-b0.0)*(center.0-b0.0) + (center.1-b0.1)*(center.1-b0.1);
                let qdm = (center.0-bm.0)*(center.0-bm.0) + (center.1-bm.1)*(center.1-bm.1);
                let qd1 = (center.0-b1.0)*(center.0-b1.0) + (center.1-b1.1)*(center.1-b1.1);
                let quick_d2 = qd0.min(qdm).min(qd1);
                if quick_d2 > 2.0 { continue; }

                // Exact cubic closest-point solve
                let result = closest_on_span_poly(
                    sc.poly_ax, sc.poly_ay, sc.poly_bx, sc.poly_by,
                    sc.poly_cx, sc.poly_cy, center.0, center.1,
                );
                let span_best_t = result.0;
                let span_best_d2 = result.1;

                if span_best_d2 < 1.0 {
                    // Insert sorted into top 4
                    for h in 0..4 {
                        if span_best_d2 < hit_d2[h] {
                            // Shift down
                            for j in (h + 1..4).rev() {
                                hit_d2[j] = hit_d2[j - 1];
                                hit_t[j] = hit_t[j - 1];
                                hit_idx[j] = hit_idx[j - 1];
                            }
                            hit_d2[h] = span_best_d2;
                            hit_t[h] = span_best_t;
                            hit_idx[h] = cp_i as u32;
                            if num_hits < 4 { num_hits += 1; }
                            break;
                        }
                    }
                }
            }

            let best_idx = if num_hits > 0 { hit_idx[0] as i32 } else { -1 };
            let best_t = if num_hits > 0 { hit_t[0] } else { 0.0 };
            let best_d2 = if num_hits > 0 { hit_d2[0] } else { 1e10 };

            // Resolve color from hits
            let mut center_color = fallback;
            for h in 0..num_hits {
                if let Some(c) = resolve_from_cp(center, &all_cps[hit_idx[h] as usize], hit_t[h]) {
                    center_color = c;
                    break;
                }
            }

            let need_aa = best_idx >= 0 && best_d2 < aa_threshold;

            if !need_aa {
                chunk[local_y * out_w + opx] = pack_color(center_color);
            } else {
                let sc = &all_cps[best_idx as usize];
                let cpt = beval(sc.prev_pos, sc.pos, sc.next_pos, best_t);
                let tang = beval_deriv(sc.prev_pos, sc.pos, sc.next_pos, best_t);
                let tl = (tang.0 * tang.0 + tang.1 * tang.1).sqrt();

                if tl < 1e-4 {
                    chunk[local_y * out_w + opx] = pack_color(center_color);
                } else {
                    let normal = (-tang.1 / tl, tang.0 / tl);
                    let d = (center.0 - cpt.0) * normal.0 + (center.1 - cpt.1) * normal.1;

                    // Compute edge colors on-the-fly
                    let (aa_pl, aa_pr) = if sc.prev_dir >= 0 { get_edge_colors(sc.icx, sc.icy, sc.prev_dir) } else { (0, 0) };
                    let (aa_nl, aa_nr) = if sc.next_dir >= 0 { get_edge_colors(sc.icx, sc.icy, sc.next_dir) } else { (0, 0) };

                    // Resolve both colors from edge colors
                    let mut color_left = 0u32;
                    let mut color_right = 0u32;
                    let mut have_edge = false;
                    let mut ref_t = 0.0f32;

                    if best_t < 0.5 && aa_pl != aa_pr {
                        color_left = aa_pl;
                        color_right = aa_pr;
                        ref_t = 0.0;
                        have_edge = true;
                    } else if best_t >= 0.5 && aa_nl != aa_nr {
                        color_left = aa_nr;
                        color_right = aa_nl;
                        ref_t = 1.0;
                        have_edge = true;
                    } else if aa_nl != aa_nr {
                        color_left = aa_nr;
                        color_right = aa_nl;
                        ref_t = 1.0;
                        have_edge = true;
                    } else if aa_pl != aa_pr {
                        color_left = aa_pl;
                        color_right = aa_pr;
                        ref_t = 0.0;
                        have_edge = true;
                    }

                    if !have_edge {
                        chunk[local_y * out_w + opx] = pack_color(center_color);
                    } else {
                        let orig_tang =
                            beval_deriv(sc.orig_prev, sc.orig_pos, sc.orig_next, ref_t);
                        let orig_norm = (-orig_tang.1, orig_tang.0);
                        let flip =
                            normal.0 * orig_norm.0 + normal.1 * orig_norm.1 < 0.0;
                        let pos_side = if flip { color_right } else { color_left };
                        let neg_side = if flip { color_left } else { color_right };

                        if pos_side == neg_side {
                            chunk[local_y * out_w + opx] = pack_color(center_color);
                        } else {
                            let pw = inv_scale;
                            let proj_w = pw * normal.0.abs().max(normal.1.abs());
                            let frac = (0.5 + d / proj_w).clamp(0.0, 1.0);

                            // Blend in linear light (sRGB decode → blend → sRGB encode)
                            let r0 = (((pos_side >> 16) & 0xFF) as f32 / 255.0).powf(2.2);
                            let g0 = (((pos_side >> 8) & 0xFF) as f32 / 255.0).powf(2.2);
                            let b0 = ((pos_side & 0xFF) as f32 / 255.0).powf(2.2);
                            let r1 = (((neg_side >> 16) & 0xFF) as f32 / 255.0).powf(2.2);
                            let g1 = (((neg_side >> 8) & 0xFF) as f32 / 255.0).powf(2.2);
                            let b1 = ((neg_side & 0xFF) as f32 / 255.0).powf(2.2);
                            let ri = ((frac * r0 + (1.0 - frac) * r1).powf(1.0 / 2.2) * 255.0).round().clamp(0.0, 255.0) as u32;
                            let gi = ((frac * g0 + (1.0 - frac) * g1).powf(1.0 / 2.2) * 255.0).round().clamp(0.0, 255.0) as u32;
                            let bi = ((frac * b0 + (1.0 - frac) * b1).powf(1.0 / 2.2) * 255.0).round().clamp(0.0, 255.0) as u32;
                            chunk[local_y * out_w + opx] = 0xFF000000 | (ri << 16) | (gi << 8) | bi;
                        }
                    }
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
            let grid_data = &grid_data;
            let cell_offset = &cell_offset;
            let handles: Vec<_> = chunks.into_iter().enumerate().map(|(ci, chunk)| {
                let start = ci * rows_per_thread;
                scope.spawn(move || {
                    rasterize_rows(chunk, start, all_cps, grid_data, cell_offset);
                })
            }).collect();
            for h in handles { h.join().unwrap(); }
        });
    }
    #[cfg(target_arch = "wasm32")]
    {
        rasterize_rows(&mut output, 0, &all_cps, &grid_data, &cell_offset);
    }

    output
}

/// Pack ARGB color (shader outputs RGB components divided by 255, then stored;
/// here we just ensure alpha is 0xFF).
#[inline(always)]
fn pack_color(c: u32) -> u32 {
    0xFF000000 | (c & 0x00FFFFFF)
}
