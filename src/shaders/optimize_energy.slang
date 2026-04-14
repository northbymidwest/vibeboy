// 2D Newton-Raphson optimizer for B-spline control points.
//
// Energy function: E = E_curvature + E_positional
//   E_curvature = |p0 - 2*p1 + p2|^2  (simple kappa^2 approximation)
//   E_positional = (2.5 * distance(p, p_orig))^4

import modules.vectorize;

struct Uniforms {
    uint num_nodes;
    float gradient_step;  // unused (kept for ABI compatibility)
    float max_move;       // unused
    float positional_scale; // 2.5
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

static const int NEWTON_ITER = 3;

float2 read_pos(uint i) {
    return float2(pos_in[i * 2u], pos_in[i * 2u + 1u]);
}

float2 read_neighbor_pos(uint i) {
    if ((node_flags[i] & IS_TJUNCTION) != 0u) {
        uint ci = i + 1u;
        float2 p = float2(pos_in[ci * 2u], pos_in[ci * 2u + 1u]);
        if (p.x != 0.0 || p.y != 0.0) return p;
    }
    return float2(pos_in[i * 2u], pos_in[i * 2u + 1u]);
}

float2 read_orig(uint i) {
    return float2(orig_pos[i * 2u], orig_pos[i * 2u + 1u]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= u.num_nodes) return;

    float2 p = read_pos(i);
    pos_out[i * 2u] = p.x;
    pos_out[i * 2u + 1u] = p.y;

    uint flags = node_flags[i];

    // Pinned nodes don't move
    if ((flags & 1u) != 0u) return;

    int prev_idx = neighbor_data[i * 4u];
    int next_idx = neighbor_data[i * 4u + 1u];
    if (prev_idx < 0 || next_idx < 0) return;

    // Skip non-crossing nodes adjacent to crossings (crossing position
    // depends on all 4 neighbors; optimizing one side without the other
    // would break symmetry). Crossing nodes themselves must not be
    // skipped — they handle their own combined-curvature optimization.
    if ((flags & IS_CROSSING) == 0u) {
        if ((node_flags[uint(prev_idx)] & IS_CROSSING) != 0u) return;
        if ((node_flags[uint(next_idx)] & IS_CROSSING) != 0u) return;
    }

    float2 n0 = read_neighbor_pos(uint(prev_idx));
    float2 n1 = read_neighbor_pos(uint(next_idx));
    float2 p_orig = read_orig(i);

    float s4 = u.positional_scale * u.positional_scale;
    s4 *= s4; // 2.5^4 = 39.0625

    // Crossings: optimize with combined curvature from both chains.
    // Slot 0 has N-S neighbors, slot 1 (at i+1 or i-1) has E-W neighbors.
    // Combine all 4 neighbors' curvature into one gradient. Write same
    // position to both slots so they stay coincident.
    bool is_crossing = (flags & IS_CROSSING) != 0u;
    // Only process crossings from the even slot (slot 0) to avoid double-processing.
    if (is_crossing && (i & 1u) != 0u) return;

    // Corners: exclude curvature energy, keep positional energy only.
    bool exclude_curvature = (flags & IS_CORNER) != 0u;

    // For crossings, gather the other pair's neighbors too.
    float2 n2 = float2(0, 0);
    float2 n3 = float2(0, 0);
    if (is_crossing) {
        uint other = i + 1u;
        int other_prev = neighbor_data[other * 4u];
        int other_next = neighbor_data[other * 4u + 1u];
        if (other_prev >= 0) n2 = read_neighbor_pos(uint(other_prev));
        if (other_next >= 0) n3 = read_neighbor_pos(uint(other_next));
    }

    for (int iter = 0; iter < NEWTON_ITER; iter++) {
        float2 d = p - p_orig;
        float d2 = dot(d, d);

        // Gradient: curvature (kappa²) + positional ((2.5*dist)⁴)
        float2 grad_curv = 4.0 * (2.0 * p - n0 - n1);
        if (is_crossing) {
            // Add curvature from the second chain (E-W)
            grad_curv += 4.0 * (2.0 * p - n2 - n3);
        }
        float2 grad_pos = 4.0 * s4 * d2 * d;
        float2 grad = exclude_curvature ? grad_pos : (grad_curv + grad_pos);

        // Hessian
        float curv_h = exclude_curvature ? 0.0 : (is_crossing ? 16.0 : 8.0);
        float h00 = curv_h + 4.0 * s4 * (2.0 * d.x * d.x + d2);
        float h11 = curv_h + 4.0 * s4 * (2.0 * d.y * d.y + d2);
        float h01 = 8.0 * s4 * d.x * d.y;

        float det = h00 * h11 - h01 * h01;
        if (abs(det) < 1e-20) break;

        p -= float2(h11 * grad.x - h01 * grad.y,
                     -h01 * grad.x + h00 * grad.y) / det;
    }

    pos_out[i * 2u] = p.x;
    pos_out[i * 2u + 1u] = p.y;
    // Write same position to both crossing slots
    if (is_crossing) {
        pos_out[(i + 1u) * 2u] = p.x;
        pos_out[(i + 1u) * 2u + 1u] = p.y;
    }
}
