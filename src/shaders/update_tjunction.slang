// Post-optimization T-junction position correction.
//
// Matches reference: UpdateCorrectedPositions.vert
// For each CP with IS_TJUNCTION flag, recompute position as:
//   pos = 0.125 * prev_pos + 0.75 * pos + 0.125 * next_pos

import modules.vectorize;

struct Uniforms {
    uint num_nodes;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
[[vk::binding(0, 1)]] ConstantBuffer<Uniforms> u;

[[vk::binding(1, 0)]] StructuredBuffer<int> neighbor_data;
[[vk::binding(2, 0)]] StructuredBuffer<uint> node_flags;
[[vk::binding(0, 0)]] RWStructuredBuffer<float> positions;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= u.num_nodes) return;

    uint flags = node_flags[i];
    if ((flags & IS_TJUNCTION) == 0u) return;

    int prev_idx = neighbor_data[i * 4u];
    int next_idx = neighbor_data[i * 4u + 1u];
    if (prev_idx < 0 || next_idx < 0) return;

    float2 prev_pos = float2(positions[uint(prev_idx) * 2u], positions[uint(prev_idx) * 2u + 1u]);
    float2 curr_pos = float2(positions[i * 2u], positions[i * 2u + 1u]);
    float2 next_pos = float2(positions[uint(next_idx) * 2u], positions[uint(next_idx) * 2u + 1u]);

    float2 corrected = 0.125 * prev_pos + 0.75 * curr_pos + 0.125 * next_pos;
    positions[i * 2u] = corrected.x;
    positions[i * 2u + 1u] = corrected.y;

    // Update the stem CP at the other slot (i ^ 1) so it terminates
    // exactly ON the smooth curve.
    uint stem = i ^ 1u;
    if (stem < u.num_nodes && node_flags[stem] == 1u) {
        float2 on_curve = 0.125 * prev_pos + 0.75 * corrected + 0.125 * next_pos;
        positions[stem * 2u] = on_curve.x;
        positions[stem * 2u + 1u] = on_curve.y;
    }
}
