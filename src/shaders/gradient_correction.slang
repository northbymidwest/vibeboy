// Global gradient correction step. Pass C of the 3-pass outer iter
// (Picard → IFT → gradient). Mirrors the CPU `compute_g`-then-step
// in src/scaling/vectorize.rs::optimize_energy.
//
// For each CP i, computes ∂E/∂p_i from the 3 quadratic-spline segments
// touching i, plus the T-junction stem segment (when i is the chain CP
// or i is T_prev/T_next of an active T-junction), plus the positional
// anchor term. Then takes a clipped step:
//   p_out[i] = p_in[i] − η · clip(∇E_i, max_step)
//
// This step debiases the IFT iteration's fixed point: the IFT cross-
// Hessian correction has a small but persistent error in long-
// wavelength chain modes, producing a slowly-converging standing wave
// along near-straight edges. The gradient correction makes the
// iteration's fixed point coincide with ∇E = 0 (the true local
// minimum) rather than the biased IFT fixed point.
//
// GPU-friendly: same 5-CP neighborhood as picard_step.slang, no global
// reductions, no scalar uniforms threaded between dispatches.

import modules.vectorize;

struct Uniforms {
    uint num_nodes;
    float eta;
    float max_step;
    uint _pad0;
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

float2 read_raw(uint i) {
    return float2(pos_in[i * 2u], pos_in[i * 2u + 1u]);
}

float2 read_ghost(uint i) {
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
            float2 pp = read_raw(uint(t_prev));
            float2 pt = read_raw(t);
            float2 pn = read_raw(uint(t_next));
            return sp * pp + st * pt + sn * pn;
        }
    }
    return read_raw(i);
}

int grandneighbor_idx(int via, int back) {
    if (via < 0) return -1;
    if ((node_flags[uint(via)] & IS_CORNER) != 0u) return -1;
    int a = neighbor_data[uint(via) * 4u];
    int b = neighbor_data[uint(via) * 4u + 1u];
    return (a == back) ? b : a;
}

bool is_stem_shadow(int k) {
    if (k <= 0) return false;
    uint uk = uint(k);
    if (uk >= u.num_nodes) return false;
    if ((node_flags[uk] & 1u) == 0u) return false;
    return (node_flags[uk - 1u] & IS_TJUNCTION) != 0u;
}

// Segment θ-Jacobian at endpoint i, evaluated at (pa, pb, pc) where
// pb is the segment's center. `which`: 0 = ∂/∂a, 1 = ∂/∂b, 2 = ∂/∂c.
// `mode`: 0=full, 1=left half (vb→m), 2=right half (va→m).
// Returns (theta, j_at_which) — the chain rule of θ at the requested
// endpoint slot.
void seg_grad(float2 pa, float2 pb, float2 pc, int mode,
              out float theta,
              out float2 j_a, out float2 j_b, out float2 j_c) {
    float2 va = pb - pa;
    float2 vb = pc - pb;
    float inv_la2 = 1.0 / max(va.x * va.x + va.y * va.y, 1e-20);
    float inv_lb2 = 1.0 / max(vb.x * vb.x + vb.y * vb.y, 1e-20);
    float2 dva = float2( va.y * inv_la2, -va.x * inv_la2);
    float2 dvb = float2(-vb.y * inv_lb2,  vb.x * inv_lb2);
    if (mode == 0) {
        float cross_v = va.x * vb.y - va.y * vb.x;
        float dot_v = va.x * vb.x + va.y * vb.y;
        theta = atan2(cross_v, dot_v);
        j_a = -dva;
        j_b = dva - dvb;
        j_c = dvb;
    } else if (mode == 1) {
        // Left half: substitute vb → m = (va + vb) / 2.
        float2 m = 0.5 * (va + vb);
        float cross_v = va.x * m.y - va.y * m.x;
        float dot_v = va.x * m.x + va.y * m.y;
        theta = atan2(cross_v, dot_v);
        float inv_lm2 = 1.0 / max(m.x * m.x + m.y * m.y, 1e-20);
        float2 dm = float2(-m.y * inv_lm2, m.x * inv_lm2);
        j_a = -dva - 0.5 * dm;
        j_b = dva;
        j_c = 0.5 * dm;
    } else {
        // mode 2: right half — substitute va → m.
        float2 m = 0.5 * (va + vb);
        float cross_v = m.x * vb.y - m.y * vb.x;
        float dot_v = m.x * vb.x + m.y * vb.y;
        theta = atan2(cross_v, dot_v);
        float inv_lm2 = 1.0 / max(m.x * m.x + m.y * m.y, 1e-20);
        float2 dm = float2(-m.y * inv_lm2, m.x * inv_lm2);
        j_a = 0.5 * dm;
        j_b = -dvb;
        j_c = -0.5 * dm + dvb;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint ci = tid.x;
    if (ci >= u.num_nodes) return;

    // Default: pass through.
    float2 p_self = read_raw(ci);
    pos_out[ci * 2u]      = p_self.x;
    pos_out[ci * 2u + 1u] = p_self.y;

    uint f = node_flags[ci];
    // Pinned (bit 0) or completely-unused CPs: skip gradient step.
    if (f == 0u || (f & 1u) != 0u) return;

    // Accumulate ∇E for the 3 segments touching ci that aren't skipped
    // for half-segment corner exclusion.
    float2 g = float2(0, 0);

    int prev_idx = neighbor_data[ci * 4u];
    int next_idx = neighbor_data[ci * 4u + 1u];
    bool self_corner = (f & IS_CORNER) != 0u;

    // Segment centered at self: pa = read_ghost(prev), pb = p_self, pc = read_ghost(next).
    if (!self_corner && prev_idx >= 0 && next_idx >= 0
        && !is_stem_shadow(prev_idx) && !is_stem_shadow(next_idx)) {
        bool prev_corner = (node_flags[uint(prev_idx)] & IS_CORNER) != 0u;
        bool next_corner = (node_flags[uint(next_idx)] & IS_CORNER) != 0u;
        int mode = (prev_corner && next_corner) ? -1
                 : (prev_corner ? 2 : (next_corner ? 1 : 0));
        if (mode >= 0) {
            float2 pa = read_ghost(uint(prev_idx));
            float2 pc = read_ghost(uint(next_idx));
            float theta;
            float2 j_a, j_b, j_c;
            seg_grad(pa, p_self, pc, mode, theta, j_a, j_b, j_c);
            g += theta * j_b;
        }
    }

    // Segment centered at prev: ci is the "c" endpoint. CPs (gprev_of_prev, prev, ci).
    if (prev_idx >= 0) {
        uint pi = uint(prev_idx);
        uint fp = node_flags[pi];
        if ((fp & (1u | IS_CORNER)) == 0u) {
            int gp = grandneighbor_idx(prev_idx, int(ci));
            if (gp >= 0 && !is_stem_shadow(gp) && !is_stem_shadow(int(ci))) {
                bool gp_corner = (node_flags[uint(gp)] & IS_CORNER) != 0u;
                int mode = (gp_corner && self_corner) ? -1
                         : (gp_corner ? 2 : (self_corner ? 1 : 0));
                if (mode >= 0) {
                    float2 pa = read_ghost(uint(gp));
                    float2 pb = read_ghost(pi);
                    float theta;
                    float2 j_a, j_b, j_c;
                    seg_grad(pa, pb, p_self, mode, theta, j_a, j_b, j_c);
                    g += theta * j_c;
                }
            }
        }
    }

    // Segment centered at next: ci is the "a" endpoint. CPs (ci, next, gnext_of_next).
    if (next_idx >= 0) {
        uint ni = uint(next_idx);
        uint fn = node_flags[ni];
        if ((fn & (1u | IS_CORNER)) == 0u) {
            int gn = grandneighbor_idx(next_idx, int(ci));
            if (gn >= 0 && !is_stem_shadow(gn) && !is_stem_shadow(int(ci))) {
                bool gn_corner = (node_flags[uint(gn)] & IS_CORNER) != 0u;
                int mode = (self_corner && gn_corner) ? -1
                         : (self_corner ? 2 : (gn_corner ? 1 : 0));
                if (mode >= 0) {
                    float2 pb = read_ghost(ni);
                    float2 pc = read_ghost(uint(gn));
                    float theta;
                    float2 j_a, j_b, j_c;
                    seg_grad(p_self, pb, pc, mode, theta, j_a, j_b, j_c);
                    g += theta * j_a;
                }
            }
        }
    }

    // T-junction stem segment: routes through the ghost.
    // Case A: ci is a T-junction (its own stem segment contributes to T,
    // T_prev, T_next through chain rule). We are computing ∇E at ci = T:
    //   ghost = sp·p_tp + st·p_t + sn·p_tn
    //   ∂E_stem/∂T = st · J_a · θ
    if ((f & IS_TJUNCTION) != 0u && prev_idx >= 0 && next_idx >= 0
        && ci + 1u < u.num_nodes) {
        int sn_idx = neighbor_data[(ci + 1u) * 4u];
        if (sn_idx >= 0) {
            uint s = uint(sn_idx);
            int sa = neighbor_data[s * 4u];
            int sb = neighbor_data[s * 4u + 1u];
            int so_i = (sa == int(ci + 1u)) ? sb : sa;
            if (so_i >= 0) {
                bool pe = (node_flags[uint(prev_idx)] & IS_ENDPOINT) != 0u;
                bool ne = (node_flags[uint(next_idx)] & IS_ENDPOINT) != 0u;
                float sp, st, sn;
                if (pe && ne)      { sp = 0.25;  st = 0.5;   sn = 0.25; }
                else if (pe)       { sp = 0.25;  st = 0.625; sn = 0.125; }
                else if (ne)       { sp = 0.125; st = 0.625; sn = 0.25; }
                else               { sp = 0.125; st = 0.75;  sn = 0.125; }
                float2 p_tp = read_raw(uint(prev_idx));
                float2 p_tn = read_raw(uint(next_idx));
                float2 ghost = sp * p_tp + st * p_self + sn * p_tn;
                float2 p_s = read_raw(s);
                float2 p_so = read_raw(uint(so_i));
                float theta;
                float2 j_a, j_b, j_c;
                seg_grad(ghost, p_s, p_so, 0, theta, j_a, j_b, j_c);
                g += theta * st * j_a;
            }
        }
    }
    // Case B: ci is T_prev or T_next of an adjacent T-junction (via
    // immediate neighbor). Then ∂E_stem/∂ci = (sp or sn) · J_a · θ.
    // We check both neighbors (prev and next).
    int nbrs[2] = { prev_idx, next_idx };
    for (int k = 0; k < 2; k++) {
        int t_idx = nbrs[k];
        if (t_idx < 0) continue;
        uint t = uint(t_idx);
        if ((node_flags[t] & IS_TJUNCTION) == 0u) continue;
        int t_prev = neighbor_data[t * 4u];
        int t_next = neighbor_data[t * 4u + 1u];
        if (t_prev < 0 || t_next < 0) continue;
        if (t + 1u >= u.num_nodes) continue;
        int sn_idx = neighbor_data[(t + 1u) * 4u];
        if (sn_idx < 0) continue;
        uint s = uint(sn_idx);
        int sa = neighbor_data[s * 4u];
        int sb = neighbor_data[s * 4u + 1u];
        int so_i = (sa == int(t + 1u)) ? sb : sa;
        if (so_i < 0) continue;
        bool pe = (node_flags[uint(t_prev)] & IS_ENDPOINT) != 0u;
        bool ne = (node_flags[uint(t_next)] & IS_ENDPOINT) != 0u;
        float sp, st, sn;
        if (pe && ne)      { sp = 0.25;  st = 0.5;   sn = 0.25; }
        else if (pe)       { sp = 0.25;  st = 0.625; sn = 0.125; }
        else if (ne)       { sp = 0.125; st = 0.625; sn = 0.25; }
        else               { sp = 0.125; st = 0.75;  sn = 0.125; }
        // Pick the routing weight for ci's role on T.
        float w = 0.0;
        if (t_prev == int(ci)) w = sp;
        else if (t_next == int(ci)) w = sn;
        else continue;
        float2 p_tp = read_raw(uint(t_prev));
        float2 p_t  = read_raw(t);
        float2 p_tn = read_raw(uint(t_next));
        float2 ghost = sp * p_tp + st * p_t + sn * p_tn;
        float2 p_s = read_raw(s);
        float2 p_so = read_raw(uint(so_i));
        float theta;
        float2 j_a, j_b, j_c;
        seg_grad(ghost, p_s, p_so, 0, theta, j_a, j_b, j_c);
        g += theta * w * j_a;
    }
    // Case C: ci is the chain CP s of an adjacent T-junction (∂E_stem/∂s = J_b).
    if (ci > 0u && (f & 1u) == 0u) {
        // Walk both neighbors and check if they are the "+1 slot" of a T-junction.
        for (int k = 0; k < 2; k++) {
            int n_idx = (k == 0) ? prev_idx : next_idx;
            if (n_idx <= 0) continue;
            uint n_u = uint(n_idx);
            if ((node_flags[n_u] & 1u) == 0u) continue;
            if ((node_flags[n_u - 1u] & IS_TJUNCTION) == 0u) continue;
            // n_u is a stem shadow whose T-junction is at n_u - 1.
            uint t = n_u - 1u;
            int t_prev = neighbor_data[t * 4u];
            int t_next = neighbor_data[t * 4u + 1u];
            if (t_prev < 0 || t_next < 0) continue;
            // The "s" CP for this T-junction is neighbor_data[(t+1)*4 + 0].
            int s_idx = neighbor_data[(t + 1u) * 4u];
            if (s_idx != int(ci)) continue; // We aren't s for this T.
            int sa = neighbor_data[ci * 4u];
            int sb = neighbor_data[ci * 4u + 1u];
            int so_i = (sa == int(t + 1u)) ? sb : sa;
            if (so_i < 0) continue;
            bool pe = (node_flags[uint(t_prev)] & IS_ENDPOINT) != 0u;
            bool ne = (node_flags[uint(t_next)] & IS_ENDPOINT) != 0u;
            float sp, st, sn;
            if (pe && ne)      { sp = 0.25;  st = 0.5;   sn = 0.25; }
            else if (pe)       { sp = 0.25;  st = 0.625; sn = 0.125; }
            else if (ne)       { sp = 0.125; st = 0.625; sn = 0.25; }
            else               { sp = 0.125; st = 0.75;  sn = 0.125; }
            float2 p_tp = read_raw(uint(t_prev));
            float2 p_t  = read_raw(t);
            float2 p_tn = read_raw(uint(t_next));
            float2 ghost = sp * p_tp + st * p_t + sn * p_tn;
            float2 p_so = read_raw(uint(so_i));
            float theta;
            float2 j_a, j_b, j_c;
            seg_grad(ghost, p_self, p_so, 0, theta, j_a, j_b, j_c);
            g += theta * j_b;
        }
    }

    // Positional energy gradient: 4·s4·d²·d.
    const float s4 = 2.5 * 2.5 * 2.5 * 2.5;
    float2 p_orig = float2(orig_pos[ci * 2u], orig_pos[ci * 2u + 1u]);
    float2 d = p_self - p_orig;
    float d2 = d.x * d.x + d.y * d.y;
    g += 4.0 * s4 * d2 * d;

    // Step: -η · ∇E, clipped to max_step per CP.
    float dx = -u.eta * g.x;
    float dy = -u.eta * g.y;
    float len2 = dx * dx + dy * dy;
    float ms = u.max_step;
    if (len2 > ms * ms) {
        float s = ms / sqrt(len2);
        dx *= s;
        dy *= s;
    }
    pos_out[ci * 2u]      = p_self.x + dx;
    pos_out[ci * 2u + 1u] = p_self.y + dy;
}
