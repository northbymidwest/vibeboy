// Post-optimization junction position correction.
//
// Crossings: ghost-aware inverse B-spline correction so the *rendered*
// curve passes through the grid corner at t=0.5. Coefficients depend on
// whether prev/next is a chain endpoint (which causes the rasterizer
// to ghost-adjust that side of the clamped Bezier).
//
// T-junctions: through-CP is left at the optimizer's value (not
// overwritten). The stem CP is snapped onto the rendered through-curve
// via the ghost-aware algebraic B(0.5) formula.
//
// Dispatch a few times for convergence when junction CPs are neighbors
// of other junction CPs.

import modules.vectorize;

struct Uniforms {
    uint num_nodes;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
[[vk::binding(0, 1)]]
#ifdef DXIL_TARGET
ConstantBuffer<Uniforms> u : register(b0, space2);
#else
ConstantBuffer<Uniforms> u;
#endif

[[vk::binding(1, 0)]]
#ifdef DXIL_TARGET
StructuredBuffer<int> neighbor_data : register(t0, space0);
#else
StructuredBuffer<int> neighbor_data;
#endif
[[vk::binding(2, 0)]]
#ifdef DXIL_TARGET
StructuredBuffer<uint> node_flags : register(t1, space0);
#else
StructuredBuffer<uint> node_flags;
#endif
[[vk::binding(3, 0)]]
#ifdef DXIL_TARGET
StructuredBuffer<float> orig_positions : register(t2, space0);
#else
StructuredBuffer<float> orig_positions;
#endif
[[vk::binding(0, 0)]]
#ifdef DXIL_TARGET
RWStructuredBuffer<float> positions : register(u0, space1);
#else
RWStructuredBuffer<float> positions;
#endif

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= u.num_nodes) return;

    uint flags = node_flags[i];

    int prev_idx = neighbor_data[i * 4u];
    int next_idx = neighbor_data[i * 4u + 1u];
    if (prev_idx < 0 || next_idx < 0) return;

    bool prev_is_end = (node_flags[uint(prev_idx)] & IS_ENDPOINT) != 0u;
    bool next_is_end = (node_flags[uint(next_idx)] & IS_ENDPOINT) != 0u;
    float2 prev_pos = float2(positions[uint(prev_idx) * 2u], positions[uint(prev_idx) * 2u + 1u]);
    float2 next_pos = float2(positions[uint(next_idx) * 2u], positions[uint(next_idx) * 2u + 1u]);

    if ((flags & IS_CROSSING) != 0u) {
        // Ghost-aware inverse correction: solve B_rendered(0.5) = grid for cp.
        float a, bp, bn;
        if (!prev_is_end && !next_is_end) { a = 0.75; bp = 0.125; bn = 0.125; }
        else if (prev_is_end && !next_is_end) { a = 0.625; bp = 0.25; bn = 0.125; }
        else if (!prev_is_end && next_is_end) { a = 0.625; bp = 0.125; bn = 0.25; }
        else { a = 0.5; bp = 0.25; bn = 0.25; }

        float2 grid_pos = float2(orig_positions[i * 2u], orig_positions[i * 2u + 1u]);
        float2 corrected = (grid_pos - bp * prev_pos - bn * next_pos) / a;
        positions[i * 2u] = corrected.x;
        positions[i * 2u + 1u] = corrected.y;
        return;
    }

    if ((flags & IS_TJUNCTION) == 0u) return;

    // Through-CP stays at the optimizer's position. Snap the stem onto
    // the rendered through-curve via the ghost-aware algebraic B(0.5).
    uint stem = i ^ 1u;
    if (stem >= u.num_nodes) return;
    if ((node_flags[stem] & ~IS_ENDPOINT) != 1u) return;

    float2 through = float2(positions[i * 2u], positions[i * 2u + 1u]);
    float sp, st, sn;
    if (!prev_is_end && !next_is_end) { sp = 0.125; st = 0.75;  sn = 0.125; }
    else if (prev_is_end && !next_is_end) { sp = 0.25;  st = 0.625; sn = 0.125; }
    else if (!prev_is_end && next_is_end) { sp = 0.125; st = 0.625; sn = 0.25; }
    else { sp = 0.25;  st = 0.5;   sn = 0.25; }

    float2 on_curve = sp * prev_pos + st * through + sn * next_pos;
    positions[stem * 2u] = on_curve.x;
    positions[stem * 2u + 1u] = on_curve.y;
}
