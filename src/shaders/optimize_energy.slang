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

    // Skip optimization entirely for spans containing pinned nodes
    // (geometric corners and crossing nodes whose positions are inverse-corrected)
    static const uint PINNED_MASK = IS_CORNER | IS_CROSSING;
    if ((flags & PINNED_MASK) != 0u) return;
    if ((node_flags[uint(prev_idx)] & PINNED_MASK) != 0u) return;
    if ((node_flags[uint(next_idx)] & PINNED_MASK) != 0u) return;

    float2 n0 = read_neighbor_pos(uint(prev_idx));
    float2 n1 = read_neighbor_pos(uint(next_idx));
    float2 p_orig = read_orig(i);

    float s4 = u.positional_scale * u.positional_scale;
    s4 *= s4; // 2.5^4 = 39.0625

    for (int iter = 0; iter < NEWTON_ITER; iter++) {
        float2 d = p - p_orig;
        float d2 = dot(d, d);

        float2 grad = 4.0 * (2.0 * p - n0 - n1) + 4.0 * s4 * d2 * d;

        float h00 = 8.0 + 4.0 * s4 * (2.0 * d.x * d.x + d2);
        float h11 = 8.0 + 4.0 * s4 * (2.0 * d.y * d.y + d2);
        float h01 = 8.0 * s4 * d.x * d.y;

        float det = h00 * h11 - h01 * h01;
        if (abs(det) < 1e-20) break;

        p -= float2(h11 * grad.x - h01 * grad.y,
                     -h01 * grad.x + h00 * grad.y) / det;
    }

    pos_out[i * 2u] = p.x;
    pos_out[i * 2u + 1u] = p.y;
}
