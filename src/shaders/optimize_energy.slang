// 2D Newton optimizer for B-spline control points. Mirrors the CPU
// implementation in src/scaling/vectorize.rs::optimize_energy.
//
// Energy = E_curvature + E_positional
//   E_curvature: sum of θ²/2 over the (up to) three quadratic-spline
//     segments touching p_i, where θ is the segment's turning angle =
//     atan2(v_a × v_b, v_a · v_b). For a uniform quadratic B-spline this
//     equals ∫|κ|ds along the segment (Gauss-Bonnet for plane curves with
//     constant-sign curvature). Plus the same on the second chain at
//     crossings (slot i+1) and on the stem segment via the ghost
//     (chain-rule coupling at T-junctions and their through-pair).
//   E_positional: (positional_scale · ‖p − p_orig‖)⁴.
//
// Optimizer: plain Newton with Gauss-Newton Hessian (always PSD),
// determinant guard against singular H, and a step-magnitude cap to
// prevent overshoot in long straight chains.
//
// Outer iteration: orchestrated by the host. N dispatches with under-
// relaxation blending the result with the input each pass.

import modules.vectorize;

struct Uniforms {
    uint num_nodes;
    // 0 on the first outer pass, 1 thereafter. We can't infer this from
    // prev_state alone: a genuine "no history yet" pass and a "pass-1
    // line-search rejected, prev_f=0 legitimately" pass both have
    // |prev_f|² == 0, but require different handling. The first wants a
    // Picard step (just `starting_p + f`); the second wants the Anderson
    // formula, which with dp=0 and df=f gives γ=1 and cancels back to
    // starting_p — exactly the behavior the CPU reference produces.
    uint have_history;
    uint _pad0;
    uint _pad1;
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
// Per-CP Anderson(2) acceleration history. Each CP owns 4 floats:
//   prev_state[i*4 + 0..1] = previous outer-pass input (p_in at pass k-1)
//   prev_state[i*4 + 2..3] = previous fixed-point residual (f at pass k-1)
//                            where f = G(p_in) - p_in for the inner solver G.
// Single read + single write per CP per pass; no race (thread i touches
// only its own slot). Anderson computes per-CP optimal mix γ via the
// 1-D least-squares between the current and previous residuals — no
// global reduction. Whether to apply the formula or fall back to a
// Picard step is decided by the `have_history` uniform (0 on the first
// pass, 1 thereafter), not by inspecting prev_state — see the Uniforms
// struct comment.
[[vk::binding(1, 1)]]
#ifdef DXIL_TARGET
RWStructuredBuffer<float> prev_state : register(u1, space1);
#else
RWStructuredBuffer<float> prev_state;
#endif

static const int NEWTON_ITER = 10;

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

// Add one segment's contribution to e/g/h. v_a and v_b are the two
// difference vectors of the quadratic-spline segment; (s_va, s_vb) are
// the chain-rule coefficients of p in v_a and v_b respectively.
void add_segment(float2 va, float2 vb, float s_va, float s_vb,
                 inout float e, inout float2 g,
                 inout float h00, inout float h01, inout float h11) {
    float cross_v = va.x * vb.y - va.y * vb.x;
    float dot_v = va.x * vb.x + va.y * vb.y;
    float theta = atan2(cross_v, dot_v);
    e += 0.5 * theta * theta;
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
                      inout float e, inout float2 g,
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
    add_segment(va_eff, vb_eff, s_va_eff, s_vb_eff, e, g, h00, h01, h11);
}

// Energy-only segment add. Used inside the line-search backtracks
// where the gradient and Hessian are discarded. Skipping the chain
// rule + Hessian update cuts ~70% of the ALU per call vs the full
// add_segment.
void add_e_full(float2 va, float2 vb, inout float e) {
    float cross_v = va.x * vb.y - va.y * vb.x;
    float dot_v = va.x * vb.x + va.y * vb.y;
    float theta = atan2(cross_v, dot_v);
    e += 0.5 * theta * theta;
}

void add_e_mode(int mode, float2 va, float2 vb, inout float e) {
    if (mode == 3) return;
    if (mode == 1) vb = 0.5 * (va + vb);
    else if (mode == 2) va = 0.5 * (va + vb);
    add_e_full(va, vb, e);
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
    // Bit-packed `state` keeps modes + crossing + stem flags in a
    // single uint so the inner Newton loop only holds one register
    // for all of these instead of ~9 individual scalars:
    //   bits 0-1: mode_self
    //   bits 2-3: mode_e_prev
    //   bits 4-5: mode_e_next
    //   bit  6:   c_inc_self
    //   bit  7:   c_inc_e_prev
    //   bit  8:   c_inc_e_next
    //   bit  9:   stem_a_active
    //   bit 10:   stem_b0_active
    //   bit 11:   stem_b1_active
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
    // Newton iteration. Compute once outside the loop.
    float2 va_prev_const = n0 - nn0;  // va of segment θ_{i-1} when active
    float2 vb_next_const = nn1 - n1;  // vb of segment θ_{i+1} when active

    // ---- Slot 1 (crossing's other chain) ----
    float2 c_n0 = float2(0, 0);
    float2 c_n1 = float2(0, 0);
    float2 c_va_prev_const = float2(0, 0); // = c_n0 - c_nn0
    float2 c_vb_next_const = float2(0, 0); // = c_nn1 - c_n1
    if (is_crossing) {
        uint other = i + 1u;
        int c_prev_idx = neighbor_data[other * 4u];
        int c_next_idx = neighbor_data[other * 4u + 1u];
        if (c_prev_idx >= 0) c_n0 = read_neighbor_pos(uint(c_prev_idx));
        if (c_next_idx >= 0) c_n1 = read_neighbor_pos(uint(c_next_idx));

        uint other_flags = node_flags[other];
        if ((other_flags & IS_CORNER) == 0u) state |= 1u << 6;

        bool c_inc_prev = (c_prev_idx >= 0)
            && ((node_flags[uint(c_prev_idx)] & IS_CORNER) == 0u);
        bool c_inc_next = (c_next_idx >= 0)
            && ((node_flags[uint(c_next_idx)] & IS_CORNER) == 0u);

        int c_gprev = c_inc_prev ? grandneighbor_idx(c_prev_idx, int(other)) : -1;
        int c_gnext = c_inc_next ? grandneighbor_idx(c_next_idx, int(other)) : -1;
        if (c_gprev >= 0) {
            c_va_prev_const = c_n0 - read_neighbor_pos(uint(c_gprev));
            state |= 1u << 7;
        }
        if (c_gnext >= 0) {
            c_vb_next_const = read_neighbor_pos(uint(c_gnext)) - c_n1;
            state |= 1u << 8;
        }
    }

    // ---- Hoist stem coupling out of the Newton loop ----
    // Stem case A and B contributions are linear in p:
    //   add_segment(va_const, vb_const + p_coef * p, 0, weight)
    // The va, vb_const, p_coef, and weight are all p-independent and
    // can be pre-baked once per CP. Inside the inner loop each stem
    // segment becomes 2 vec2 adds + the standard θ kernel — no more
    // stem_endpoints / read_neighbor_pos / Case-B-loop work per iter.
    // Stem activity flags packed in state bits 9-11 (set when each
    // case below populates its slot).
    float2 stem_a_va = float2(0, 0);
    float2 stem_a_vb_const = float2(0, 0);
    float stem_a_p_coef = 0.0;
    float2 stem_b0_va = float2(0, 0);
    float2 stem_b0_vb_const = float2(0, 0);
    float stem_b0_p_coef = 0.0;
    float2 stem_b1_va = float2(0, 0);
    float2 stem_b1_vb_const = float2(0, 0);
    float stem_b1_p_coef = 0.0;

    // Case A: i is a T-junction. ghost = sp*pp + st*p + sn*pn.
    // pp/pn are read_neighbor_pos(prev/next) which we already cached
    // as n0/n1 — reuse instead of re-fetching.
    if ((flags & IS_TJUNCTION) != 0u) {
        float2 s_pos, so_pos;
        if (stem_endpoints(i, s_pos, so_pos)) {
            bool prev_is_end = (flags_prev & IS_ENDPOINT) != 0u;
            bool next_is_end = (flags_next & IS_ENDPOINT) != 0u;
            float st, sp, sn;
            ghost_weights(prev_is_end, next_is_end, st, sp, sn);
            state |= 1u << 9;
            stem_a_va       = s_pos - so_pos;
            stem_a_vb_const = sp * n0 + sn * n1 - s_pos;
            stem_a_p_coef   = st;
        }
    }

    // Case B[0/1]: prev/next is a T-junction with i in its through-pair.
    // Unrolled into two explicit blocks (was a 2-iter loop) so each
    // slot's fetches happen exactly once.
    if ((flags_prev & IS_TJUNCTION) != 0u) {
        int j_prev = neighbor_data[uint(prev_idx) * 4u];
        int j_next = neighbor_data[uint(prev_idx) * 4u + 1u];
        if (j_prev >= 0 && j_next >= 0) {
            bool jpe = (node_flags[uint(j_prev)] & IS_ENDPOINT) != 0u;
            bool jne = (node_flags[uint(j_next)] & IS_ENDPOINT) != 0u;
            float st, sp, sn;
            ghost_weights(jpe, jne, st, sp, sn);
            float2 p_t = float2(pos_in[uint(prev_idx) * 2u], pos_in[uint(prev_idx) * 2u + 1u]);
            float p_coef = 0.0;
            float2 ghost_const = float2(0, 0);
            if (j_prev == int(i)) {
                p_coef = sp;
                ghost_const = st * p_t + sn * read_neighbor_pos(uint(j_next));
            } else if (j_next == int(i)) {
                p_coef = sn;
                ghost_const = sp * read_neighbor_pos(uint(j_prev)) + st * p_t;
            }
            if (p_coef != 0.0) {
                float2 s_pos, so_pos;
                if (stem_endpoints(uint(prev_idx), s_pos, so_pos)) {
                    state |= 1u << 10;
                    stem_b0_va       = s_pos - so_pos;
                    stem_b0_vb_const = ghost_const - s_pos;
                    stem_b0_p_coef   = p_coef;
                }
            }
        }
    }
    if ((flags_next & IS_TJUNCTION) != 0u) {
        int j_prev = neighbor_data[uint(next_idx) * 4u];
        int j_next = neighbor_data[uint(next_idx) * 4u + 1u];
        if (j_prev >= 0 && j_next >= 0) {
            bool jpe = (node_flags[uint(j_prev)] & IS_ENDPOINT) != 0u;
            bool jne = (node_flags[uint(j_next)] & IS_ENDPOINT) != 0u;
            float st, sp, sn;
            ghost_weights(jpe, jne, st, sp, sn);
            float2 p_t = float2(pos_in[uint(next_idx) * 2u], pos_in[uint(next_idx) * 2u + 1u]);
            float p_coef = 0.0;
            float2 ghost_const = float2(0, 0);
            if (j_prev == int(i)) {
                p_coef = sp;
                ghost_const = st * p_t + sn * read_neighbor_pos(uint(j_next));
            } else if (j_next == int(i)) {
                p_coef = sn;
                ghost_const = sp * read_neighbor_pos(uint(j_prev)) + st * p_t;
            }
            if (p_coef != 0.0) {
                float2 s_pos, so_pos;
                if (stem_endpoints(uint(next_idx), s_pos, so_pos)) {
                    state |= 1u << 11;
                    stem_b1_va       = s_pos - so_pos;
                    stem_b1_vb_const = ghost_const - s_pos;
                    stem_b1_p_coef   = p_coef;
                }
            }
        }
    }

    // Positional energy weight: (positional_scale·d)⁴ with positional_scale = 2.5.
    // Hardcoded to mirror src/scaling/vectorize.rs::optimize_energy.
    const float s4 = 2.5 * 2.5 * 2.5 * 2.5;

    for (int iter = 0; iter < NEWTON_ITER; iter++) {
        float e_total = 0.0;
        float2 g = float2(0, 0);
        float h00 = 0.0;
        float h01 = 0.0;
        float h11 = 0.0;

        // Modes packed in state bits 0-5; extract once per iter.
        int m_self   = int(state & 3u);
        int m_e_prev = int((state >> 2) & 3u);
        int m_e_next = int((state >> 4) & 3u);

        // Three main segments touching p.
        if (m_e_prev != 3) {
            add_segment_mode(m_e_prev, va_prev_const, p - n0, 0.0, 1.0,
                             e_total, g, h00, h01, h11);
        }
        if (m_self != 3) {
            add_segment_mode(m_self, p - n0, n1 - p, 1.0, -1.0,
                             e_total, g, h00, h01, h11);
        }
        if (m_e_next != 3) {
            add_segment_mode(m_e_next, n1 - p, vb_next_const, -1.0, 0.0,
                             e_total, g, h00, h01, h11);
        }
        // Crossing's second chain (full segments only). State bits 6/7/8.
        if ((state & (1u << 7)) != 0u)
            add_segment(c_va_prev_const, p - c_n0, 0.0, 1.0,
                        e_total, g, h00, h01, h11);
        if ((state & (1u << 6)) != 0u)
            add_segment(p - c_n0, c_n1 - p, 1.0, -1.0,
                        e_total, g, h00, h01, h11);
        if ((state & (1u << 8)) != 0u)
            add_segment(c_n1 - p, c_vb_next_const, -1.0, 0.0,
                        e_total, g, h00, h01, h11);
        // Stem segments (state bits 9/10/11; pre-baked outside the loop).
        if ((state & (1u << 9)) != 0u) {
            add_segment(stem_a_va, stem_a_vb_const + stem_a_p_coef * p,
                        0.0, stem_a_p_coef, e_total, g, h00, h01, h11);
        }
        if ((state & (1u << 10)) != 0u) {
            add_segment(stem_b0_va, stem_b0_vb_const + stem_b0_p_coef * p,
                        0.0, stem_b0_p_coef, e_total, g, h00, h01, h11);
        }
        if ((state & (1u << 11)) != 0u) {
            add_segment(stem_b1_va, stem_b1_vb_const + stem_b1_p_coef * p,
                        0.0, stem_b1_p_coef, e_total, g, h00, h01, h11);
        }

        // Positional energy ‖p − p_orig‖⁴.
        float2 d = p - p_orig;
        float d2 = dot(d, d);
        e_total += s4 * d2 * d2;
        g.x += 4.0 * s4 * d2 * d.x;
        g.y += 4.0 * s4 * d2 * d.y;
        h00 += 4.0 * s4 * (2.0 * d.x * d.x + d2);
        h11 += 4.0 * s4 * (2.0 * d.y * d.y + d2);
        h01 += 8.0 * s4 * d.x * d.y;

        float det = h00 * h11 - h01 * h01;
        // Step direction. Two cases:
        //   det >= ε: full-rank Newton solve δ = −H⁻¹·g
        //   det <  ε: Gauss-Newton pseudo-inverse step. The H here is
        //             rank-1 (single contributing segment, e.g. corner CP
        //             with only positional energy nonzero), with H = J·Jᵀ,
        //             g = θ·Jᵀ, |J|² = h00+h11. Min-norm LS: δ = −g/|J|².
        float dx, dy;
        if (abs(det) < 1e-6) {
            float trace = h00 + h11;
            if (trace < 1e-20) break;
            float inv_trace = 1.0 / trace;
            dx = -g.x * inv_trace;
            dy = -g.y * inv_trace;
        } else {
            float inv_det = 1.0 / det;
            dx = -(h11 * g.x - h01 * g.y) * inv_det;
            dy = -(-h01 * g.x + h00 * g.y) * inv_det;
        }

        // Backtracking line search using the pre-baked stem/crossing
        // partials. Trial uses ENERGY-ONLY evaluation (skip gradient
        // and Hessian) since they're discarded — saves ~70% of the
        // per-trial ALU. Same accept criterion (e_new < e_total) as
        // the CPU reference.
        bool accepted = false;
        float scale_factor = 1.0;
        for (int bt = 0; bt < 6; bt++) {
            float2 p_trial = p + scale_factor * float2(dx, dy);
            float e_new = 0.0;

            if (m_e_prev != 3) add_e_mode(m_e_prev, va_prev_const, p_trial - n0, e_new);
            if (m_self != 3)   add_e_mode(m_self, p_trial - n0, n1 - p_trial, e_new);
            if (m_e_next != 3) add_e_mode(m_e_next, n1 - p_trial, vb_next_const, e_new);
            if ((state & (1u << 7)) != 0u) add_e_full(c_va_prev_const, p_trial - c_n0, e_new);
            if ((state & (1u << 6)) != 0u) add_e_full(p_trial - c_n0, c_n1 - p_trial, e_new);
            if ((state & (1u << 8)) != 0u) add_e_full(c_n1 - p_trial, c_vb_next_const, e_new);
            if ((state & (1u << 9)) != 0u) {
                add_e_full(stem_a_va, stem_a_vb_const + stem_a_p_coef * p_trial, e_new);
            }
            if ((state & (1u << 10)) != 0u) {
                add_e_full(stem_b0_va, stem_b0_vb_const + stem_b0_p_coef * p_trial, e_new);
            }
            if ((state & (1u << 11)) != 0u) {
                add_e_full(stem_b1_va, stem_b1_vb_const + stem_b1_p_coef * p_trial, e_new);
            }
            float2 d_t = p_trial - p_orig;
            float d2_t = dot(d_t, d_t);
            e_new += s4 * d2_t * d2_t;

            if (e_new < e_total) {
                p = p_trial;
                accepted = true;
                break;
            }
            scale_factor *= 0.5;
        }
        if (!accepted) break;
    }

    // Per-CP Anderson(2) acceleration. Treat the inner Newton-CG solve
    // as a fixed-point map G; let f = G(p_in) − p_in be the residual.
    // With one step of history (prev_p, prev_f), the optimal mix on a
    // single CP collapses to the 1-D least-squares
    //     γ = (Δf · f) / (Δf · Δf),  Δp = p_in − prev_p,  Δf = f − prev_f
    // and the accelerated update is
    //     p_out = (p_in + f) − γ · (Δp + Δf).
    // Same dispatch profile as Polyak — single kernel per outer pass —
    // but converges instead of wandering, since the mix is optimal per
    // CP rather than a fixed momentum coefficient.
    float2 f = p - starting_p;
    float2 p_new;
    if (u.have_history == 0u) {
        // First pass: take the Picard step and seed prev_state so the
        // Anderson formula has valid history on the next pass.
        p_new = starting_p + f;
    } else {
        float2 prev_p_i = float2(prev_state[i * 4u + 0u], prev_state[i * 4u + 1u]);
        float2 prev_f_i = float2(prev_state[i * 4u + 2u], prev_state[i * 4u + 3u]);
        float2 dp = starting_p - prev_p_i;
        float2 df = f - prev_f_i;
        float df_dot_df = df.x * df.x + df.y * df.y;
        float gamma = (df_dot_df > 1e-12) ? (df.x * f.x + df.y * f.y) / df_dot_df : 0.0;
        // Safeguarded Anderson: clamp γ to [0, 1.5]. Negative γ would
        // anti-accelerate; |γ| > 1.5 means extrapolating > 50% past the
        // Picard step — empirically overshoots at noisy CPs (corners,
        // T-junctions where the inner Newton's local model is poorly
        // conditioned), producing visible artifacts in low-iter-count runs.
        gamma = clamp(gamma, 0.0, 1.5);
        p_new = (starting_p + f) - gamma * (dp + df);
    }
    prev_state[i * 4u + 0u] = starting_p.x;
    prev_state[i * 4u + 1u] = starting_p.y;
    prev_state[i * 4u + 2u] = f.x;
    prev_state[i * 4u + 3u] = f.y;
    pos_out[i * 2u] = p_new.x;
    pos_out[i * 2u + 1u] = p_new.y;
    if (is_crossing) {
        prev_state[(i + 1u) * 4u + 0u] = starting_p.x;
        prev_state[(i + 1u) * 4u + 1u] = starting_p.y;
        prev_state[(i + 1u) * 4u + 2u] = f.x;
        prev_state[(i + 1u) * 4u + 3u] = f.y;
        pos_out[(i + 1u) * 2u] = p_new.x;
        pos_out[(i + 1u) * 2u + 1u] = p_new.y;
    }
}
