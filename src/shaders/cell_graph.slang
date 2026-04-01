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
[[vk::binding(0, 2)]] ConstantBuffer<Uniforms> u : register(b0, space2);

[[vk::binding(0, 0)]] StructuredBuffer<uint> graph : register(t0, space0);
[[vk::binding(0, 1)]] RWStructuredBuffer<float> positions : register(u0, space1);
[[vk::binding(1, 1)]] RWStructuredBuffer<int> neighbors : register(u1, space1);
[[vk::binding(2, 1)]] RWStructuredBuffer<uint> flags : register(u2, space1);

uint g(int gx, int gy) {
    gx = clamp(gx, 0, int(u.graph_stride) - 1);
    gy = clamp(gy, 0, int(2u * u.img_h));
    return graph[gy * int(u.graph_stride) + gx];
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

    if (!is_ullr && !is_llur) return base;

    int slot = 0;
    if (from_dir == 0) {
        if (is_llur) slot = 1;
    } else if (from_dir == 1) {
        // slot stays 0
    } else if (from_dir == 2) {
        if (is_ullr) slot = 1;
    } else { // from_dir == 3
        slot = 1;
        if (!is_ullr && !is_llur) slot = 0;
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
        int prev = -1, next = -1;
        int t_prev_dir = -1, t_next_dir = -1;

        if (bnd_n && bnd_s) {
            prev = n_idx; next = s_idx; t_prev_dir = 0; t_next_dir = 2;
        } else if (bnd_e && bnd_w) {
            prev = e_idx; next = w_idx; t_prev_dir = 1; t_next_dir = 3;
        } else if (bnd_n && bnd_e) {
            prev = n_idx; next = e_idx; t_prev_dir = 0; t_next_dir = 1;
        } else if (bnd_n && bnd_w) {
            prev = n_idx; next = w_idx; t_prev_dir = 0; t_next_dir = 3;
        } else if (bnd_s && bnd_e) {
            prev = s_idx; next = e_idx; t_prev_dir = 2; t_next_dir = 1;
        } else {
            prev = s_idx; next = w_idx; t_prev_dir = 2; t_next_dir = 3;
        }

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
        // Valence 4 cross-junction: pinned, pick N-S as spline pair
        write_cp_full(base, pos, n_idx, s_idx, 1u, 0, 2, int(cx), int(cy));
    }
}
