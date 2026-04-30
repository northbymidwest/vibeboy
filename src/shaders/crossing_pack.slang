// Stage 5b: At each 4-way crossing, solve the curve-curve intersection of
// the N-S and E-W spans and write (t_NS, t_EW) into the crossing_t buffer.
//
// Runs after update_tjunction so neighbor stem positions are finalized
// (a crossing's N/S/E/W neighbor can be a T-junction stem CP whose position
// gets repositioned by the snap; reading stale stem positions would shift
// the intersection enough to flip wedge classification on near-boundary
// pixels).
//
// One thread per CP. Slot 1 of the crossing pair early-exits since we write
// both t values from slot 0; non-crossings early-exit too.

import modules.vectorize;
import modules.intersect;

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
StructuredBuffer<float> positions : register(t2, space0);
#else
StructuredBuffer<float> positions;
#endif
[[vk::binding(0, 0)]]
#ifdef DXIL_TARGET
RWStructuredBuffer<float> crossing_t : register(u0, space1);
#else
RWStructuredBuffer<float> crossing_t;
#endif

float2 read_pos(uint ci) {
    return float2(positions[ci * 2u], positions[ci * 2u + 1u]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= u.num_nodes) return;

    // Slot 1 of a crossing — slot 0 writes both entries below; just exit so
    // we don't race with the slot-0 thread.
    bool is_crossing = (node_flags[i] & IS_CROSSING) != 0u;
    if (is_crossing && (i & 1u) != 0u) return;

    // Non-crossings: emit the natural-midpoint default (0.5) so the
    // rasterizer's `crossing_t[ci]` read returns the right value when
    // some other code path mistakenly treats a non-crossing as one. Cheap
    // initialization of the buffer per frame.
    if (!is_crossing) {
        crossing_t[i] = 0.5;
        return;
    }

    // Slot 0 of a crossing: solve the curve-curve intersection.
    uint other = i + 1u;
    if (other >= u.num_nodes) return;

    int n_idx = neighbor_data[i * 4u];
    int s_idx = neighbor_data[i * 4u + 1u];
    int e_idx = neighbor_data[other * 4u];
    int w_idx = neighbor_data[other * 4u + 1u];
    if (n_idx < 0 || s_idx < 0 || e_idx < 0 || w_idx < 0) return;

    float2 cp_a = read_pos(i);
    float2 cp_b = read_pos(other);

    float2 n_pos = read_pos(uint(n_idx));
    float2 s_pos = read_pos(uint(s_idx));
    float2 e_pos = read_pos(uint(e_idx));
    float2 w_pos = read_pos(uint(w_idx));

    bool n_is_end = (node_flags[uint(n_idx)] & IS_ENDPOINT) != 0u;
    bool s_is_end = (node_flags[uint(s_idx)] & IS_ENDPOINT) != 0u;
    bool e_is_end = (node_flags[uint(e_idx)] & IS_ENDPOINT) != 0u;
    bool w_is_end = (node_flags[uint(w_idx)] & IS_ENDPOINT) != 0u;

    // Ghost-extend endpoint neighbors so the curve B(t) we solve against
    // matches what the rasterizer renders (clamped Bezier via knot
    // multiplicity 3 at endpoints).
    float2 n_in = n_is_end ? (2.0 * n_pos - cp_a) : n_pos;
    float2 s_in = s_is_end ? (2.0 * s_pos - cp_a) : s_pos;
    float2 e_in = e_is_end ? (2.0 * e_pos - cp_b) : e_pos;
    float2 w_in = w_is_end ? (2.0 * w_pos - cp_b) : w_pos;

    float2 t = bspline_intersect(n_in, cp_a, s_in, e_in, cp_b, w_in);
    crossing_t[i]     = t.x;  // t on N-S curve (slot 0)
    crossing_t[other] = t.y;  // t on E-W curve (slot 1)
}
