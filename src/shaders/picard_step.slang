// Per-CP inner Newton (Picard step). Pass A of the 3-pass outer iter
// (Picard → IFT correction → gradient correction) for the unbiased
// optimizer. Mirrors the CPU `optimize_one_pass` in
// src/scaling/vectorize.rs::optimize_energy.
//
// Energy = E_curvature + E_positional
//   E_curvature: sum of θ²/2 over the (up to) three quadratic-spline
//     segments touching p_i, where θ is the segment's turning angle.
//     Plus the second chain at crossings (slot i+1) and the stem
//     segment via the ghost (chain rule at T-junctions).
//   E_positional: (positional_scale · ‖p − p_orig‖)⁴.
//
// Output: p_new = starting_p + f, where f is the inner-Newton delta
// for CP i alone. No Anderson, no prev_state — the host orchestrates
// outer iteration via separate IFT/gradient passes.

import modules.vectorize;

struct Uniforms {
    uint num_nodes;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
[[vk::binding(0, 2)]]
#ifdef DXIL_TARGET
ConstantBuffer<Uniforms> u : register(b0, space2);
#else
ConstantBuffer<Uniforms> u;
#endif

[[vk::binding(0, 0)]]
#ifdef DXIL_TARGET
StructuredBuffer<float> pos_in : register(t0, space0);
#else
StructuredBuffer<float> pos_in;
#endif
[[vk::binding(1, 0)]]
#ifdef DXIL_TARGET
StructuredBuffer<float> orig_pos : register(t1, space0);
#else
StructuredBuffer<float> orig_pos;
#endif
[[vk::binding(2, 0)]]
#ifdef DXIL_TARGET
StructuredBuffer<int> neighbor_data : register(t2, space0);
#else
StructuredBuffer<int> neighbor_data;
#endif
[[vk::binding(3, 0)]]
#ifdef DXIL_TARGET
StructuredBuffer<uint> node_flags : register(t3, space0);
#else
StructuredBuffer<uint> node_flags;
#endif
[[vk::binding(0, 1)]]
#ifdef DXIL_TARGET
RWStructuredBuffer<float> pos_out : register(u0, space1);
#else
RWStructuredBuffer<float> pos_out;
#endif

// 1 inner Newton iter per outer pass. With clamped-step Picard +
// gradient correction as the outer loop, a single Newton iter is
// enough: a smaller per-iter Picard step leaves more headroom for
// the gradient correction to debias toward ∇E = 0 rather than
// overshooting the Gauss-Seidel fixed point. CPU sweep with
// OUTER=2 outer iters: NEWTON=1 N=2 e=289.95 vs NEWTON=2 N=2 e=282.04
// (+2.8%); at N=10 NEWTON=1 e=271.55 vs NEWTON=2 e=271.87 (NEWTON=1
// converges to a slightly lower final energy). Matches NEWTON_ITER
// in vectorscale's optimize-energy.slang and CPU default.
static const int NEWTON_ITER = 1;

float2 read_pos_raw(uint i) {
    return float2(pos_in[i * 2u], pos_in[i * 2u + 1u]);
}

float2 read_orig(uint i) {
    return float2(orig_pos[i * 2u], orig_pos[i * 2u + 1u]);
}

// T-junction ghost CP (junction-pinned, immediately follows a T-junction
// in CP indexing): synthesize the snapped position on-the-fly from T's
// through-pair instead of reading the stored (stale-during-optimization)
// value. Matches what update_tjunctions Phase 1 will write
// post-optimization. Weights depend on whether the through-pair neighbors
// are clamped boundary endpoints (multiplicity-3 knot adjustment).
float2 read_neighbor_pos(uint i) {
    if (i > 0u && (node_flags[i] & 1u) != 0u && (node_flags[i - 1u] & IS_TJUNCTION) != 0u) {
        uint t = i - 1u;
        int t_prev = neighbor_data[t * 4u];
        int t_next = neighbor_data[t * 4u + 1u];
        if (t_prev >= 0 && t_next >= 0) {
            bool prev_is_end = (node_flags[uint(t_prev)] & IS_ENDPOINT) != 0u;
            bool next_is_end = (node_flags[uint(t_next)] & IS_ENDPOINT) != 0u;
            float sp, st, sn;
            if (prev_is_end && next_is_end)      { sp = 0.25;  st = 0.5;   sn = 0.25; }
            else if (prev_is_end)                { sp = 0.25;  st = 0.625; sn = 0.125; }
            else if (next_is_end)                { sp = 0.125; st = 0.625; sn = 0.25; }
            else                                 { sp = 0.125; st = 0.75;  sn = 0.125; }
            float2 pp = float2(pos_in[uint(t_prev) * 2u], pos_in[uint(t_prev) * 2u + 1u]);
            float2 pt = float2(pos_in[t * 2u], pos_in[t * 2u + 1u]);
            float2 pn = float2(pos_in[uint(t_next) * 2u], pos_in[uint(t_next) * 2u + 1u]);
            return sp * pp + st * pt + sn * pn;
        }
    }
    return float2(pos_in[i * 2u], pos_in[i * 2u + 1u]);
}

// Walk to a grandneighbor: from node `via`, return its neighbor that is
// not `back`. Only IS_CORNER blocks (paper's per-pattern exclusion).
// T-junctions are walkable — slots 0/1 hold the through-pair so the
// "non-back" neighbor is the next node along the through-chain.
// Endpoints are valence-1 by definition, so the walk dies at -1
// naturally; no explicit block needed.
int grandneighbor_idx(int via, int back) {
    if (via < 0) return -1;
    if ((node_flags[uint(via)] & IS_CORNER) != 0u) return -1;
    int a = neighbor_data[uint(via) * 4u];
    int b = neighbor_data[uint(via) * 4u + 1u];
    return (a == back) ? b : a;
}

// Add one segment's contribution to (g, H). v_a and v_b are the two
// difference vectors of the quadratic-spline segment; (s_va, s_vb) are
// the chain-rule coefficients of p in v_a and v_b respectively.
// No energy term: with NEWTON_ITER=1 and no line search, the energy
// value is never read — the host's outer iteration handles convergence.
// Matches the structure of vectorscale's optimize-energy.slang.
void add_segment(float2 va, float2 vb, float s_va, float s_vb,
                 inout float2 g,
                 inout float h00, inout float h01, inout float h11) {
    float cross_v = va.x * vb.y - va.y * vb.x;
    float dot_v = va.x * vb.x + va.y * vb.y;
    float theta = atan2(cross_v, dot_v);
    float inv_lva2 = 1.0 / max(va.x * va.x + va.y * va.y, 1e-20);
    float inv_lvb2 = 1.0 / max(vb.x * vb.x + vb.y * vb.y, 1e-20);
    // dθ/dp = s_va · (−perp(va)/|va|²) + s_vb · (+perp(vb)/|vb|²)
    // −perp((x,y)) = (y, −x); +perp((x,y)) = (−y, x)
    float dtheta_dpx = s_va * (va.y * inv_lva2) + s_vb * (-vb.y * inv_lvb2);
    float dtheta_dpy = s_va * (-va.x * inv_lva2) + s_vb * (vb.x * inv_lvb2);
    g.x += theta * dtheta_dpx;
    g.y += theta * dtheta_dpy;
    h00 += dtheta_dpx * dtheta_dpx;
    h01 += dtheta_dpx * dtheta_dpy;
    h11 += dtheta_dpy * dtheta_dpy;
}

// Add a B-spline segment in mode {0=full, 1=left half, 2=right half, 3=skip}.
// For half-segments substitute m=(v_a+v_b)/2 for v_a (right half) or v_b
// (left half), with chain-rule coefficients halved accordingly. Reuses
// the same θ kernel — gradient/Hessian fall out via the chain rule.
void add_segment_mode(int mode, float2 va, float2 vb, float s_va, float s_vb,
                      inout float2 g,
                      inout float h00, inout float h01, inout float h11) {
    if (mode == 3) return;
    float2 va_eff = va;
    float2 vb_eff = vb;
    float s_va_eff = s_va;
    float s_vb_eff = s_vb;
    if (mode == 1) {
        vb_eff = 0.5 * (va + vb);
        s_vb_eff = 0.5 * (s_va + s_vb);
    } else if (mode == 2) {
        va_eff = 0.5 * (va + vb);
        s_va_eff = 0.5 * (s_va + s_vb);
    }
    add_segment(va_eff, vb_eff, s_va_eff, s_vb_eff, g, h00, h01, h11);
}

void ghost_weights(bool prev_is_end, bool next_is_end,
                    out float st, out float sp, out float sn) {
    if (prev_is_end && next_is_end)      { st = 0.5;   sp = 0.25;  sn = 0.25; }
    else if (prev_is_end)                { st = 0.625; sp = 0.25;  sn = 0.125; }
    else if (next_is_end)                { st = 0.625; sp = 0.125; sn = 0.25; }
    else                                 { st = 0.75;  sp = 0.125; sn = 0.125; }
}

// Returns (s_pos, so_pos) for the stem CP attached to the T-junction at
// index t, or returns false if there's no valid stem.
bool stem_endpoints(uint t, out float2 s_pos, out float2 so_pos) {
    s_pos = float2(0, 0);
    so_pos = float2(0, 0);
    if (t + 1u >= u.num_nodes) return false;
    int s_idx = neighbor_data[(t + 1u) * 4u];
    if (s_idx < 0) return false;
    uint s = uint(s_idx);
    int sa = neighbor_data[s * 4u];
    int sb = neighbor_data[s * 4u + 1u];
    int so = (sa == int(t + 1u)) ? sb : sa;
    if (so < 0) return false;
    s_pos = float2(pos_in[s * 2u], pos_in[s * 2u + 1u]);
    so_pos = read_neighbor_pos(uint(so));
    return true;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= u.num_nodes) return;

    float2 starting_p = read_pos_raw(i);
    float2 p = starting_p;
    pos_out[i * 2u] = p.x;
    pos_out[i * 2u + 1u] = p.y;

    uint flags = node_flags[i];

    // Valence-1 junction pin (cell_graph sets bit 0 for these).
    if ((flags & 1u) != 0u) return;

    int prev_idx = neighbor_data[i * 4u];
    int next_idx = neighbor_data[i * 4u + 1u];
    if (prev_idx < 0 || next_idx < 0) return;

    bool is_crossing = (flags & IS_CROSSING) != 0u;
    // Crossings: only process the even slot; slot i+1 is the second chain.
    if (is_crossing && (i & 1u) != 0u) {
        uint even = i - 1u;
        pos_out[i * 2u] = pos_out[even * 2u];
        pos_out[i * 2u + 1u] = pos_out[even * 2u + 1u];
        return;
    }

    // ---- Slot 0 (this index): one chain, three triplets ----
    float2 n0 = read_neighbor_pos(uint(prev_idx));
    float2 n1 = read_neighbor_pos(uint(next_idx));
    float2 p_orig = read_orig(i);

    uint flags_prev = node_flags[uint(prev_idx)];
    uint flags_next = node_flags[uint(next_idx)];

    // Half-segment corner exclusion. The paper drops the curvature
    // integral on the half of each curve segment between a corner and its
    // non-corner neighbor — i.e. from the midpoint adjacent to the corner
    // up to the corner itself. Per discrete B-spline segment centered at
    // CP_j with neighbors L=CP_{j-1}, R=CP_{j+1}:
    //   center_corner            → drop the whole segment
    //   left+right both corners  → drop the whole segment
    //   only left corner         → use only the right half
    //   only right corner        → use only the left half
    // Half-segments reuse the standard θ kernel by substituting
    // m=(v_a+v_b)/2 for v_a (right half) or v_b (left half).
    bool self_corner = (flags & IS_CORNER) != 0u;
    bool prev_corner = (flags_prev & IS_CORNER) != 0u;
    bool next_corner = (flags_next & IS_CORNER) != 0u;

    int gprev_idx = grandneighbor_idx(prev_idx, int(i));
    int gnext_idx = grandneighbor_idx(next_idx, int(i));
    bool gprev_corner = gprev_idx >= 0 && (node_flags[uint(gprev_idx)] & IS_CORNER) != 0u;
    bool gnext_corner = gnext_idx >= 0 && (node_flags[uint(gnext_idx)] & IS_CORNER) != 0u;
    float2 nn0 = (gprev_idx >= 0) ? read_neighbor_pos(uint(gprev_idx)) : float2(0, 0);
    float2 nn1 = (gnext_idx >= 0) ? read_neighbor_pos(uint(gnext_idx)) : float2(0, 0);

    // SegMode = 0 full | 1 left-half | 2 right-half | 3 skip.
    // Bit-packed `state` keeps modes in a single uint:
    //   bits 0-1: mode_self
    //   bits 2-3: mode_e_prev
    //   bits 4-5: mode_e_next
    // Crossing/stem activity are NOT packed here — they're computed
    // inline inside their respective IS_CROSSING / IS_TJUNCTION branches
    // in the Newton step so their working state stays local.
    int mode_self_v   = self_corner ? 3 : ((prev_corner && next_corner) ? 3 : (prev_corner ? 2 : (next_corner ? 1 : 0)));
    int mode_e_prev_v = (gprev_idx < 0) ? 3
                      : (prev_corner ? 3
                        : ((gprev_corner && self_corner) ? 3
                          : (gprev_corner ? 2 : (self_corner ? 1 : 0))));
    int mode_e_next_v = (gnext_idx < 0) ? 3
                      : (next_corner ? 3
                        : ((self_corner && gnext_corner) ? 3
                          : (self_corner ? 2 : (gnext_corner ? 1 : 0))));
    uint state = uint(mode_self_v)
               | (uint(mode_e_prev_v) << 2)
               | (uint(mode_e_next_v) << 4);

    // ---- Pre-compute p-independent segment partials ----
    // For segments θ_{i-1} and θ_{i+1}, va or vb is fixed across the
    // Newton step. Compute once.
    float2 va_prev_const = n0 - nn0;  // va of segment θ_{i-1} when active
    float2 vb_next_const = nn1 - n1;  // vb of segment θ_{i+1} when active

    // Positional energy weight: (positional_scale·d)⁴ with positional_scale = 2.5.
    // Hardcoded to mirror src/scaling/vectorize.rs::optimize_energy.
    const float s4 = 2.5 * 2.5 * 2.5 * 2.5;

    // ---- Single Newton step at starting_p (NEWTON_ITER=1) ----
    // Stem coupling and crossing data are computed INLINE inside this
    // block rather than hoisted before it: with N=1 there is no reuse
    // to amortize, and inline computation lets the compiler kill the
    // intermediates outside their if-block — reducing register pressure
    // for both stem/crossing and non-stem/non-crossing CPs (the common
    // path no longer holds those slots).
    float2 g = float2(0, 0);
    float h00 = 0.0;
    float h01 = 0.0;
    float h11 = 0.0;

    int m_self   = int(state & 3u);
    int m_e_prev = int((state >> 2) & 3u);
    int m_e_next = int((state >> 4) & 3u);

    // Three main segments touching p.
    if (m_e_prev != 3) {
        add_segment_mode(m_e_prev, va_prev_const, p - n0, 0.0, 1.0,
                         g, h00, h01, h11);
    }
    if (m_self != 3) {
        add_segment_mode(m_self, p - n0, n1 - p, 1.0, -1.0,
                         g, h00, h01, h11);
    }
    if (m_e_next != 3) {
        add_segment_mode(m_e_next, n1 - p, vb_next_const, -1.0, 0.0,
                         g, h00, h01, h11);
    }

    // ---- Crossing's second chain, inline (un-hoisted) ----
    if (is_crossing) {
        uint other = i + 1u;
        int c_prev_idx = neighbor_data[other * 4u];
        int c_next_idx = neighbor_data[other * 4u + 1u];
        bool other_not_corner = (node_flags[other] & IS_CORNER) == 0u;
        bool c_inc_prev = (c_prev_idx >= 0)
            && ((node_flags[uint(c_prev_idx)] & IS_CORNER) == 0u);
        bool c_inc_next = (c_next_idx >= 0)
            && ((node_flags[uint(c_next_idx)] & IS_CORNER) == 0u);
        float2 c_n0 = (c_prev_idx >= 0) ? read_neighbor_pos(uint(c_prev_idx)) : float2(0, 0);
        float2 c_n1 = (c_next_idx >= 0) ? read_neighbor_pos(uint(c_next_idx)) : float2(0, 0);

        if (c_inc_prev) {
            int c_gprev = grandneighbor_idx(c_prev_idx, int(other));
            if (c_gprev >= 0) {
                float2 c_va_prev_const = c_n0 - read_neighbor_pos(uint(c_gprev));
                add_segment(c_va_prev_const, p - c_n0, 0.0, 1.0,
                            g, h00, h01, h11);
            }
        }
        if (other_not_corner) {
            add_segment(p - c_n0, c_n1 - p, 1.0, -1.0, g, h00, h01, h11);
        }
        if (c_inc_next) {
            int c_gnext = grandneighbor_idx(c_next_idx, int(other));
            if (c_gnext >= 0) {
                float2 c_vb_next_const = read_neighbor_pos(uint(c_gnext)) - c_n1;
                add_segment(c_n1 - p, c_vb_next_const, -1.0, 0.0,
                            g, h00, h01, h11);
            }
        }
    }

    // ---- Stem coupling, inline (un-hoisted) ----
    // Case A: i is a T-junction itself.
    if ((flags & IS_TJUNCTION) != 0u) {
        float2 s_pos, so_pos;
        if (stem_endpoints(i, s_pos, so_pos)) {
            bool prev_is_end = (flags_prev & IS_ENDPOINT) != 0u;
            bool next_is_end = (flags_next & IS_ENDPOINT) != 0u;
            float st, sp, sn;
            ghost_weights(prev_is_end, next_is_end, st, sp, sn);
            float2 va = s_pos - so_pos;
            float2 vb = sp * n0 + sn * n1 - s_pos + st * p;
            add_segment(va, vb, 0.0, st, g, h00, h01, h11);
        }
    }
    // Case B[0]: prev is a T-junction with i in its through-pair.
    if ((flags_prev & IS_TJUNCTION) != 0u) {
        int j_prev = neighbor_data[uint(prev_idx) * 4u];
        int j_next = neighbor_data[uint(prev_idx) * 4u + 1u];
        if (j_prev >= 0 && j_next >= 0) {
            bool jpe = (node_flags[uint(j_prev)] & IS_ENDPOINT) != 0u;
            bool jne = (node_flags[uint(j_next)] & IS_ENDPOINT) != 0u;
            float st, sp, sn;
            ghost_weights(jpe, jne, st, sp, sn);
            float p_coef = 0.0;
            float2 ghost_const = float2(0, 0);
            if (j_prev == int(i) || j_next == int(i)) {
                float2 p_t = float2(pos_in[uint(prev_idx) * 2u], pos_in[uint(prev_idx) * 2u + 1u]);
                if (j_prev == int(i)) {
                    p_coef = sp;
                    ghost_const = st * p_t + sn * read_neighbor_pos(uint(j_next));
                } else {
                    p_coef = sn;
                    ghost_const = sp * read_neighbor_pos(uint(j_prev)) + st * p_t;
                }
            }
            if (p_coef != 0.0) {
                float2 s_pos, so_pos;
                if (stem_endpoints(uint(prev_idx), s_pos, so_pos)) {
                    float2 va = s_pos - so_pos;
                    float2 vb = ghost_const - s_pos + p_coef * p;
                    add_segment(va, vb, 0.0, p_coef, g, h00, h01, h11);
                }
            }
        }
    }
    // Case B[1]: next is a T-junction with i in its through-pair.
    if ((flags_next & IS_TJUNCTION) != 0u) {
        int j_prev = neighbor_data[uint(next_idx) * 4u];
        int j_next = neighbor_data[uint(next_idx) * 4u + 1u];
        if (j_prev >= 0 && j_next >= 0) {
            bool jpe = (node_flags[uint(j_prev)] & IS_ENDPOINT) != 0u;
            bool jne = (node_flags[uint(j_next)] & IS_ENDPOINT) != 0u;
            float st, sp, sn;
            ghost_weights(jpe, jne, st, sp, sn);
            float p_coef = 0.0;
            float2 ghost_const = float2(0, 0);
            if (j_prev == int(i) || j_next == int(i)) {
                float2 p_t = float2(pos_in[uint(next_idx) * 2u], pos_in[uint(next_idx) * 2u + 1u]);
                if (j_prev == int(i)) {
                    p_coef = sp;
                    ghost_const = st * p_t + sn * read_neighbor_pos(uint(j_next));
                } else {
                    p_coef = sn;
                    ghost_const = sp * read_neighbor_pos(uint(j_prev)) + st * p_t;
                }
            }
            if (p_coef != 0.0) {
                float2 s_pos, so_pos;
                if (stem_endpoints(uint(next_idx), s_pos, so_pos)) {
                    float2 va = s_pos - so_pos;
                    float2 vb = ghost_const - s_pos + p_coef * p;
                    add_segment(va, vb, 0.0, p_coef, g, h00, h01, h11);
                }
            }
        }
    }

    // Positional energy gradient + Hessian (no energy term needed).
    float2 d = p - p_orig;
    float d2 = dot(d, d);
    g.x += 4.0 * s4 * d2 * d.x;
    g.y += 4.0 * s4 * d2 * d.y;
    h00 += 4.0 * s4 * (2.0 * d.x * d.x + d2);
    h11 += 4.0 * s4 * (2.0 * d.y * d.y + d2);
    h01 += 8.0 * s4 * d.x * d.y;

    // Newton step with branchless rank-1 fallback. When |det| < ε we
    // fall back to the trace-normalized pseudo-inverse (rank-1 Hessian,
    // e.g. corner CP with only positional energy non-zero). Both
    // branches always evaluate; the select picks the right result.
    float det = h00 * h11 - h01 * h01;
    bool low_rank = abs(det) < 1e-6;
    float trace = max(h00 + h11, 1e-20);
    float inv_trace = 1.0 / trace;
    float inv_det = 1.0 / (low_rank ? 1.0 : det);
    float dx_lr = -g.x * inv_trace;
    float dy_lr = -g.y * inv_trace;
    float dx_fr = -(h11 * g.x - h01 * g.y) * inv_det;
    float dy_fr = -(-h01 * g.x + h00 * g.y) * inv_det;
    float dx = low_rank ? dx_lr : dx_fr;
    float dy = low_rank ? dy_lr : dy_fr;

    // Magnitude cap (trust region; replaces line search). CPU sweep
    // showed CAP=0.10 with no line search converges to a slightly
    // LOWER final energy than line-search at every N — the cap is a
    // more conservative trust region than first-accept line search,
    // never overshoots. Matches vectorscale's optimize-energy.slang.
    const float MAX_STEP = 0.10;
    float step_len2 = dx * dx + dy * dy;
    if (step_len2 > MAX_STEP * MAX_STEP) {
        float s = MAX_STEP / sqrt(step_len2);
        dx *= s;
        dy *= s;
    }
    p = p + float2(dx, dy);

    // Plain Picard step: write the inner-Newton output directly. Outer
    // iteration (IFT correction, gradient correction) runs in separate
    // dispatches against this output.
    pos_out[i * 2u] = p.x;
    pos_out[i * 2u + 1u] = p.y;
    if (is_crossing) {
        pos_out[(i + 1u) * 2u] = p.x;
        pos_out[(i + 1u) * 2u + 1u] = p.y;
    }
}
