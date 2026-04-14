// Post-optimization junction position correction.
//
// Crossings: inverse B-spline correction so the curve passes through
// the grid corner at t=0.5. Each crossing slot is corrected independently
// (different pp/np → different corrected positions, but both curves pass
// through the same grid_pos).
//
// T-junctions: forward B-spline correction + stem snapping.
//
// Uses orig_positions for the center weight (pre-correction optimizer
// output). Dispatch 3 times for convergence when junction CPs are
// neighbors of other junction CPs.

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

    // Valence-4 crossing: inverse B-spline correction.
    // BOTH slots are processed independently — each has its own pp/np,
    // so each needs its own corrected position for its curve to pass
    // through grid_pos at t=0.5.
    if ((flags & IS_CROSSING) != 0u) {
        int prev_idx = neighbor_data[i * 4u];
        int next_idx = neighbor_data[i * 4u + 1u];
        if (prev_idx < 0 || next_idx < 0) return;

        float2 prev_pos = float2(positions[uint(prev_idx) * 2u], positions[uint(prev_idx) * 2u + 1u]);
        float2 grid_pos = float2(orig_positions[i * 2u], orig_positions[i * 2u + 1u]);
        float2 next_pos = float2(positions[uint(next_idx) * 2u], positions[uint(next_idx) * 2u + 1u]);
        float2 corrected = (grid_pos - 0.125 * prev_pos - 0.125 * next_pos) / 0.75;
        positions[i * 2u] = corrected.x;
        positions[i * 2u + 1u] = corrected.y;
        return;
    }

    if ((flags & IS_TJUNCTION) == 0u) return;

    int prev_idx = neighbor_data[i * 4u];
    int next_idx = neighbor_data[i * 4u + 1u];
    if (prev_idx < 0 || next_idx < 0) return;

    // Neighbors: latest positions (may be updated by other threads)
    float2 prev_pos = float2(positions[uint(prev_idx) * 2u], positions[uint(prev_idx) * 2u + 1u]);
    float2 next_pos = float2(positions[uint(next_idx) * 2u], positions[uint(next_idx) * 2u + 1u]);
    // Center: ORIGINAL optimizer output (not the corrected value)
    float2 curr_orig = float2(orig_positions[i * 2u], orig_positions[i * 2u + 1u]);

    float2 corrected = 0.125 * prev_pos + 0.75 * curr_orig + 0.125 * next_pos;
    positions[i * 2u] = corrected.x;
    positions[i * 2u + 1u] = corrected.y;

    // Snap stem CP at the other slot onto the through curve.
    uint stem = i ^ 1u;
    if (stem < u.num_nodes && node_flags[stem] == 1u) {
        float2 on_curve = 0.125 * prev_pos + 0.75 * corrected + 0.125 * next_pos;
        positions[stem * 2u] = on_curve.x;
        positions[stem * 2u + 1u] = on_curve.y;
    }
}
