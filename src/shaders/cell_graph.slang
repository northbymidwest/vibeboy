// Build cell graph: generate B-spline control points at grid corners.
//
// Each thread processes one grid corner (cx, cy) where
// 0 <= cx <= img_w and 0 <= cy <= img_h.

import modules.vectorize;

struct Uniforms {
    uint img_w;
    uint img_h;
    uint graph_stride; // 2*img_w + 1
    uint corners_w;    // img_w + 1
};
[[vk::binding(0, 2)]]
#ifdef DXIL_TARGET
ConstantBuffer<Uniforms> u : register(b0, space2);
#else
ConstantBuffer<Uniforms> u;
#endif

[[vk::binding(0, 0)]]
#ifdef DXIL_TARGET
StructuredBuffer<uint> graph : register(t0, space0);
#else
StructuredBuffer<uint> graph;
#endif
[[vk::binding(0, 1)]]
#ifdef DXIL_TARGET
RWStructuredBuffer<float> positions : register(u0, space1);
#else
RWStructuredBuffer<float> positions;
#endif
[[vk::binding(1, 1)]]
#ifdef DXIL_TARGET
RWStructuredBuffer<int> neighbors : register(u1, space1);
#else
RWStructuredBuffer<int> neighbors;
#endif
[[vk::binding(2, 1)]]
#ifdef DXIL_TARGET
RWStructuredBuffer<uint> flags : register(u2, space1);
#else
RWStructuredBuffer<uint> flags;
#endif

uint g(int gx, int gy) {
    gx = clamp(gx, 0, int(u.graph_stride) - 1);
    gy = clamp(gy, 0, int(2u * u.img_h));
    return graph[gy * int(u.graph_stride) + gx];
}

// Read pixel color from the graph (stored at odd coordinates).
uint px_color(int px, int py) {
    px = clamp(px, 0, int(u.img_w) - 1);
    py = clamp(py, 0, int(u.img_h) - 1);
    return g(2 * px + 1, 2 * py + 1);
}

// Get the two pixel colors separated by a boundary edge at corner (cx,cy) in direction dir.
void bnd_colors(int cx, int cy, int dir, out uint ca, out uint cb) {
    switch (dir) {
        case 0: ca = px_color(cx - 1, cy - 1); cb = px_color(cx, cy - 1); break; // N
        case 1: ca = px_color(cx, cy - 1);      cb = px_color(cx, cy);     break; // E
        case 2: ca = px_color(cx, cy);           cb = px_color(cx - 1, cy); break; // S
        case 3: ca = px_color(cx - 1, cy);       cb = px_color(cx - 1, cy - 1); break; // W
        default: ca = 0u; cb = 0u; break;
    }
}

// Classify an edge as shading (similar colors) per paper Section 3.3.
// YUV Euclidean distance <= 100/255.
bool is_shading_edge(uint ca, uint cb) {
    if (ca == cb) return true;
    float dr = float(int((ca >> 16u) & 0xFFu) - int((cb >> 16u) & 0xFFu));
    float dg = float(int((ca >> 8u) & 0xFFu) - int((cb >> 8u) & 0xFFu));
    float db = float(int(ca & 0xFFu) - int(cb & 0xFFu));
    float dy = (0.299 * dr + 0.587 * dg + 0.114 * db) / 255.0;
    float du = 0.493 * (db / 255.0 - dy);
    float dv = 0.877 * (dr / 255.0 - dy);
    float threshold = 100.0 / 255.0;
    return (dy * dy + du * du + dv * dv) <= threshold * threshold;
}

// Select the through-pair at a 3-way T-junction (paper Section 3.3).
// dirs must be a 3-element array of the boundary directions present.
// Returns the two through-pair directions in pair0/pair1.
void select_tjunction_pair(int cx, int cy, int d0, int d1, int d2,
                           out int pair0, out int pair1) {
    // Step 1: Shading/contour classification.
    uint ca0, cb0, ca1, cb1, ca2, cb2;
    bnd_colors(cx, cy, d0, ca0, cb0);
    bnd_colors(cx, cy, d1, ca1, cb1);
    bnd_colors(cx, cy, d2, ca2, cb2);
    bool s0 = is_shading_edge(ca0, cb0);
    bool s1 = is_shading_edge(ca1, cb1);
    bool s2 = is_shading_edge(ca2, cb2);
    int shading_count = (s0 ? 1 : 0) + (s1 ? 1 : 0) + (s2 ? 1 : 0);

    if (shading_count == 1) {
        // 1 shading + 2 contour: connect the 2 contour edges.
        if (s0) { pair0 = d1; pair1 = d2; }
        else if (s1) { pair0 = d0; pair1 = d2; }
        else { pair0 = d0; pair1 = d1; }
        return;
    }

    // Step 2: Angle-based fallback — connect the pair closest to 180 degrees.
    // At grid corners, opposite pairs (N-S=0^2, E-W=1^3) are 180 degrees;
    // all other pairs are 90 degrees. With 3 of 4 cardinal directions,
    // there is always exactly one opposite pair.
    if ((d0 ^ d2) == 2) { pair0 = d0; pair1 = d2; } // d0-d2 opposite
    else if ((d0 ^ d1) == 2) { pair0 = d0; pair1 = d1; } // d0-d1 opposite
    else { pair0 = d1; pair1 = d2; } // d1-d2 opposite
}

// Compute the actual CP position at a neighbor grid corner, accounting for
// which diagonal slot we'd connect to (from_dir determines the slot).
float2 neighbor_cp_pos(int cx, int cy, int from_dir) {
    float2 base_pos = float2(float(cx), float(cy));

    if (cx <= 0 || cy <= 0 || cx >= int(u.img_w) || cy >= int(u.img_h))
        return base_pos;

    uint diag = g(2 * cx, 2 * cy);
    bool is_ullr = (diag & 1u) != 0u && (diag & 2u) == 0u;
    bool is_llur = (diag & 2u) != 0u && (diag & 1u) == 0u;

    if (!is_ullr && !is_llur) return base_pos;

    int slot = 0;
    if (from_dir == 0) { if (is_llur) slot = 1; }
    else if (from_dir == 2) { if (is_ullr) slot = 1; }
    else if (from_dir == 3) { slot = 1; }

    if (is_ullr) {
        return (slot == 0) ? base_pos + float2(-0.25, 0.25)
                           : base_pos + float2(0.25, -0.25);
    } else {
        return (slot == 0) ? base_pos + float2(-0.25, -0.25)
                           : base_pos + float2(0.25, 0.25);
    }
}

// Corner detection from Kopf-Lischinski reference implementation.
bool checkForCorner(float2 v1, float2 v2) {
    float len1 = length(v1);
    float len2 = length(v2);
    if (len1 < 1e-6 || len2 < 1e-6) return false;
    float dp = dot(v1, v2) / (len1 * len2);
    if (dp > -0.01 && dp < 0.01) return true;
    if (dp > -0.72 && dp < -0.69) return true;
    if (dp > -0.33 && dp < -0.30) return true;
    return false;
}

int cp_base(int cx, int cy) {
    if (cx < 0 || cy < 0 || cx >= int(u.corners_w) || cy >= int(u.img_h + 1u))
        return -1;
    return (cy * int(u.corners_w) + cx) * 2;
}

int nbr_cp_idx(int cx, int cy, int from_dir) {
    int base = cp_base(cx, cy);
    if (base < 0) return -1;

    if (cx <= 0 || cy <= 0 || cx >= int(u.img_w) || cy >= int(u.img_h))
        return base;

    uint diag = g(2 * cx, 2 * cy);
    bool is_ullr = (diag & 1u) != 0u && (diag & 2u) == 0u;
    bool is_llur = (diag & 2u) != 0u && (diag & 1u) == 0u;

    if (!is_ullr && !is_llur) {
        // No resolved diagonal: check for 3/4-way junction needing slot routing.
        // Slot 0 = main pair, slot 1 = stem (3-way) or second pair (4-way).
        bool t_bnd_n = g(2 * cx, 2 * cy - 1) == 0u;
        bool t_bnd_e = g(2 * cx + 1, 2 * cy) == 0u;
        bool t_bnd_s = g(2 * cx, 2 * cy + 1) == 0u;
        bool t_bnd_w = g(2 * cx - 1, 2 * cy) == 0u;
        uint t_count = (t_bnd_n ? 1u : 0u) + (t_bnd_e ? 1u : 0u)
                     + (t_bnd_s ? 1u : 0u) + (t_bnd_w ? 1u : 0u);
        if (t_count >= 3u) {
            // Collect the 3 boundary directions and select through-pair
            // using paper's shading/contour + angle heuristic.
            int dirs[3]; int di = 0;
            if (t_bnd_n) dirs[di++] = 0;
            if (t_bnd_e) dirs[di++] = 1;
            if (t_bnd_s) dirs[di++] = 2;
            if (t_bnd_w) dirs[di++] = 3;
            int pair0, pair1;
            select_tjunction_pair(cx, cy, dirs[0], dirs[1], dirs[2], pair0, pair1);
            // Map from_dir to the boundary direction at the target corner
            int target_side = from_dir ^ 2;
            if (target_side != pair0 && target_side != pair1) return base + 1;
        }
        return base;
    }

    int slot = 0;
    if (from_dir == 0) {
        if (is_llur) slot = 1;
    } else if (from_dir == 1) {
        // slot stays 0
    } else if (from_dir == 2) {
        if (is_ullr) slot = 1;
    } else { // from_dir == 3
        slot = 1;
    }

    return base + slot;
}

void write_cp_full(int idx, float2 pos, int prev, int next, uint flag,
                   int prev_dir, int next_dir, int icx, int icy) {
    positions[idx * 2] = pos.x;
    positions[idx * 2 + 1] = pos.y;
    neighbors[idx * 4] = prev;
    neighbors[idx * 4 + 1] = next;
    neighbors[idx * 4 + 2] = prev_dir;
    neighbors[idx * 4 + 3] = next_dir;
    flags[idx] = flag;
}

void write_cp(int idx, float2 pos, int prev, int next, uint flag) {
    positions[idx * 2] = pos.x;
    positions[idx * 2 + 1] = pos.y;
    neighbors[idx * 4] = prev;
    neighbors[idx * 4 + 1] = next;
    neighbors[idx * 4 + 2] = -1;
    neighbors[idx * 4 + 3] = -1;
    flags[idx] = flag;
}

bool similar(int px0, int py0, int px1, int py1) {
    int dx = px1 - px0;
    int dy = py1 - py0;
    int gx = 2 * px0 + 1;
    int gy = 2 * py0 + 1;

    if (dx == 1 && dy == 0) return g(gx + 1, gy) != 0u;
    if (dx == -1 && dy == 0) return g(gx - 1, gy) != 0u;
    if (dx == 0 && dy == 1) return g(gx, gy + 1) != 0u;
    if (dx == 0 && dy == -1) return g(gx, gy - 1) != 0u;
    return false;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint cx = tid.x;
    uint cy = tid.y;
    if (cx > u.img_w || cy > u.img_h) return;

    int base = cp_base(int(cx), int(cy));
    if (base < 0) return;

    // Default: both slots inactive
    write_cp(base, float2(0.0, 0.0), -1, -1, 0u);
    write_cp(base + 1, float2(0.0, 0.0), -1, -1, 0u);

    bool is_interior = cx > 0u && cy > 0u && cx < u.img_w && cy < u.img_h;

    if (!is_interior) {
        bool has_boundary = false;
        if (cy == 0u && cx > 0u && cx < u.img_w) {
            has_boundary = !similar(int(cx) - 1, 0, int(cx), 0);
        }
        if (cy == u.img_h && cx > 0u && cx < u.img_w) {
            has_boundary = has_boundary || !similar(int(cx) - 1, int(u.img_h) - 1, int(cx), int(u.img_h) - 1);
        }
        if (cx == 0u && cy > 0u && cy < u.img_h) {
            has_boundary = has_boundary || !similar(0, int(cy) - 1, 0, int(cy));
        }
        if (cx == u.img_w && cy > 0u && cy < u.img_h) {
            has_boundary = has_boundary || !similar(int(u.img_w) - 1, int(cy) - 1, int(u.img_w) - 1, int(cy));
        }
        if (has_boundary) {
            write_cp(base, float2(float(cx), float(cy)), -1, -1, 1u);
        }
        return;
    }

    // Interior corner: 4 pixels meet here
    uint diag = g(int(2u * cx), int(2u * cy));
    bool has_main = (diag & 1u) != 0u;
    bool has_anti = (diag & 2u) != 0u;

    bool bnd_n = !similar(int(cx) - 1, int(cy) - 1, int(cx), int(cy) - 1);
    bool bnd_e = !similar(int(cx), int(cy) - 1, int(cx), int(cy));
    bool bnd_s = !similar(int(cx) - 1, int(cy), int(cx), int(cy));
    bool bnd_w = !similar(int(cx) - 1, int(cy) - 1, int(cx) - 1, int(cy));

    uint bnd_count = (bnd_n ? 1u : 0u) + (bnd_e ? 1u : 0u)
                   + (bnd_s ? 1u : 0u) + (bnd_w ? 1u : 0u);

    if (bnd_count == 0u) return;

    // Handle diagonal crossings: split into 2 CPs
    if (has_main && !has_anti) {
        bool cp0_has_s = bnd_s;
        bool cp0_has_w = bnd_w;
        bool cp1_has_n = bnd_n;
        bool cp1_has_e = bnd_e;

        bool cp0_alive = cp0_has_s || cp0_has_w;
        bool cp1_alive = cp1_has_n || cp1_has_e;

        if (cp0_alive) {
            float2 p0 = float2(float(cx) - 0.25, float(cy) + 0.25);
            int prev0 = cp0_has_s ? nbr_cp_idx(int(cx), int(cy) + 1, 2) : -1;
            int next0 = cp0_has_w ? nbr_cp_idx(int(cx) - 1, int(cy), 3) : -1;
            bool cp0_junction = !(cp0_has_s && cp0_has_w);
            uint cp0_flag = cp0_junction ? 1u : 2u;
            if (!cp0_junction) {
                float2 s_pos = neighbor_cp_pos(int(cx), int(cy) + 1, 2);
                float2 w_pos = neighbor_cp_pos(int(cx) - 1, int(cy), 3);
                if (checkForCorner(s_pos - p0, w_pos - p0)) cp0_flag |= IS_CORNER;
            }
            int d0_prev = cp0_has_s ? 2 : -1;
            int d0_next = cp0_has_w ? 3 : -1;
            write_cp_full(base, p0, prev0, next0, cp0_flag, d0_prev, d0_next, int(cx), int(cy));
        }

        if (cp1_alive) {
            float2 p1 = float2(float(cx) + 0.25, float(cy) - 0.25);
            int prev1 = cp1_has_n ? nbr_cp_idx(int(cx), int(cy) - 1, 0) : -1;
            int next1 = cp1_has_e ? nbr_cp_idx(int(cx) + 1, int(cy), 1) : -1;
            bool cp1_junction = !(cp1_has_n && cp1_has_e);
            uint cp1_flag = cp1_junction ? 1u : 2u;
            if (!cp1_junction) {
                float2 n_pos = neighbor_cp_pos(int(cx), int(cy) - 1, 0);
                float2 e_pos = neighbor_cp_pos(int(cx) + 1, int(cy), 1);
                if (checkForCorner(n_pos - p1, e_pos - p1)) cp1_flag |= IS_CORNER;
            }
            int d1_prev = cp1_has_n ? 0 : -1;
            int d1_next = cp1_has_e ? 1 : -1;
            write_cp_full(base + 1, p1, prev1, next1, cp1_flag, d1_prev, d1_next, int(cx), int(cy));
        }
        return;
    }

    if (has_anti && !has_main) {
        bool cp0_has_n = bnd_n;
        bool cp0_has_w = bnd_w;
        bool cp1_has_s = bnd_s;
        bool cp1_has_e = bnd_e;

        bool cp0_alive = cp0_has_n || cp0_has_w;
        bool cp1_alive = cp1_has_s || cp1_has_e;

        if (cp0_alive) {
            float2 p0 = float2(float(cx) - 0.25, float(cy) - 0.25);
            int prev0 = cp0_has_n ? nbr_cp_idx(int(cx), int(cy) - 1, 0) : -1;
            int next0 = cp0_has_w ? nbr_cp_idx(int(cx) - 1, int(cy), 3) : -1;
            bool cp0_junction = !(cp0_has_n && cp0_has_w);
            uint cp0_flag = cp0_junction ? 1u : 2u;
            if (!cp0_junction) {
                float2 n_pos = neighbor_cp_pos(int(cx), int(cy) - 1, 0);
                float2 w_pos = neighbor_cp_pos(int(cx) - 1, int(cy), 3);
                if (checkForCorner(n_pos - p0, w_pos - p0)) cp0_flag |= IS_CORNER;
            }
            int d0_prev = cp0_has_n ? 0 : -1;
            int d0_next = cp0_has_w ? 3 : -1;
            write_cp_full(base, p0, prev0, next0, cp0_flag, d0_prev, d0_next, int(cx), int(cy));
        }

        if (cp1_alive) {
            float2 p1 = float2(float(cx) + 0.25, float(cy) + 0.25);
            int prev1 = cp1_has_s ? nbr_cp_idx(int(cx), int(cy) + 1, 2) : -1;
            int next1 = cp1_has_e ? nbr_cp_idx(int(cx) + 1, int(cy), 1) : -1;
            bool cp1_junction = !(cp1_has_s && cp1_has_e);
            uint cp1_flag = cp1_junction ? 1u : 2u;
            if (!cp1_junction) {
                float2 s_pos = neighbor_cp_pos(int(cx), int(cy) + 1, 2);
                float2 e_pos = neighbor_cp_pos(int(cx) + 1, int(cy), 1);
                if (checkForCorner(s_pos - p1, e_pos - p1)) cp1_flag |= IS_CORNER;
            }
            int d1_prev = cp1_has_s ? 2 : -1;
            int d1_next = cp1_has_e ? 1 : -1;
            write_cp_full(base + 1, p1, prev1, next1, cp1_flag, d1_prev, d1_next, int(cx), int(cy));
        }
        return;
    }

    // No diagonal (or both diagonals kept): single CP
    float2 pos = float2(float(cx), float(cy));

    int n_idx = bnd_n ? nbr_cp_idx(int(cx), int(cy) - 1, 0) : -1;
    int e_idx = bnd_e ? nbr_cp_idx(int(cx) + 1, int(cy), 1) : -1;
    int s_idx = bnd_s ? nbr_cp_idx(int(cx), int(cy) + 1, 2) : -1;
    int w_idx = bnd_w ? nbr_cp_idx(int(cx) - 1, int(cy), 3) : -1;

    if (bnd_count == 2u) {
        int prev = -1, next = -1;
        int prev_dir = -1, next_dir = -1;
        if (bnd_n) { if (prev < 0) { prev = n_idx; prev_dir = 0; } else { next = n_idx; next_dir = 0; } }
        if (bnd_e) { if (prev < 0) { prev = e_idx; prev_dir = 1; } else { next = e_idx; next_dir = 1; } }
        if (bnd_s) { if (prev < 0) { prev = s_idx; prev_dir = 2; } else { next = s_idx; next_dir = 2; } }
        if (bnd_w) { if (prev < 0) { prev = w_idx; prev_dir = 3; } else { next = w_idx; next_dir = 3; } }
        uint flag = 2u;
        if (prev_dir >= 0 && next_dir >= 0) {
            int prev_cx = int(cx) + ((prev_dir == 1) ? 1 : (prev_dir == 3) ? -1 : 0);
            int prev_cy = int(cy) + ((prev_dir == 2) ? 1 : (prev_dir == 0) ? -1 : 0);
            int next_cx = int(cx) + ((next_dir == 1) ? 1 : (next_dir == 3) ? -1 : 0);
            int next_cy = int(cy) + ((next_dir == 2) ? 1 : (next_dir == 0) ? -1 : 0);
            float2 prev_pos = neighbor_cp_pos(prev_cx, prev_cy, prev_dir);
            float2 next_pos = neighbor_cp_pos(next_cx, next_cy, next_dir);
            if (checkForCorner(prev_pos - pos, next_pos - pos))
                flag |= IS_CORNER;
        }
        write_cp_full(base, pos, prev, next, flag, prev_dir, next_dir, int(cx), int(cy));

    } else if (bnd_count == 1u) {
        int nbr = -1; int nbr_dir = -1;
        if (bnd_n) { nbr = n_idx; nbr_dir = 0; }
        else if (bnd_e) { nbr = e_idx; nbr_dir = 1; }
        else if (bnd_s) { nbr = s_idx; nbr_dir = 2; }
        else { nbr = w_idx; nbr_dir = 3; }
        write_cp_full(base, pos, nbr, -1, 1u, nbr_dir, -1, int(cx), int(cy));

    } else if (bnd_count == 3u) {
        // Paper Section 3.3: select through-pair via shading/contour + angle heuristic.
        int dirs[3]; int di = 0;
        if (bnd_n) dirs[di++] = 0;
        if (bnd_e) dirs[di++] = 1;
        if (bnd_s) dirs[di++] = 2;
        if (bnd_w) dirs[di++] = 3;
        int t_prev_dir, t_next_dir;
        select_tjunction_pair(int(cx), int(cy), dirs[0], dirs[1], dirs[2],
                              t_prev_dir, t_next_dir);

        int idx_arr[4] = { n_idx, e_idx, s_idx, w_idx };
        int prev = idx_arr[t_prev_dir];
        int next = idx_arr[t_next_dir];

        if (prev >= 0 && next >= 0) {
            int prev_slot = prev / 2;
            int prev_cx = prev_slot % int(u.corners_w);
            int prev_cy = prev_slot / int(u.corners_w);
            int next_slot = next / 2;
            int next_cx = next_slot % int(u.corners_w);
            int next_cy = next_slot / int(u.corners_w);
            pos = float2(
                0.125 * float(prev_cx) + 0.75 * float(cx) + 0.125 * float(next_cx),
                0.125 * float(prev_cy) + 0.75 * float(cy) + 0.125 * float(next_cy)
            );
        }

        write_cp_full(base, pos, prev, next, 2u | IS_TJUNCTION, t_prev_dir, t_next_dir, int(cx), int(cy));

        int stem_idx = -1;
        int stem_dir = -1;
        if (bnd_n && t_prev_dir != 0 && t_next_dir != 0) { stem_idx = n_idx; stem_dir = 0; }
        else if (bnd_e && t_prev_dir != 1 && t_next_dir != 1) { stem_idx = e_idx; stem_dir = 1; }
        else if (bnd_s && t_prev_dir != 2 && t_next_dir != 2) { stem_idx = s_idx; stem_dir = 2; }
        else if (bnd_w && t_prev_dir != 3 && t_next_dir != 3) { stem_idx = w_idx; stem_dir = 3; }
        if (stem_idx >= 0) {
            write_cp_full(base + 1, pos, stem_idx, -1, 1u, stem_dir, -1, int(cx), int(cy));
        }

    } else {
        // Valence 4 cross-junction: inverse-correct both CPs so their
        // B-spline curves pass through the grid corner at t=0.5.
        // IS_CORNER prevents optimizer movement, IS_CROSSING triggers
        // inverse correction in update_tjunction.
        write_cp_full(base, pos, n_idx, s_idx, 2u | IS_CROSSING, 0, 2, int(cx), int(cy));
        write_cp_full(base + 1, pos, e_idx, w_idx, 2u | IS_CROSSING, 1, 3, int(cx), int(cy));
    }
}
