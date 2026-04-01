// Scanline rasterizer variant that writes flat colors to a storage buffer.
// Used as pass 1 of the spline-diffusion pipeline.
//
// NO anti-aliasing: single center-point sample per pixel, producing hard
// region boundaries.

import modules.vectorize;

struct Uniforms {
    uint out_w;
    uint out_h;
    uint num_edges;
    uint bg_color;
};
[[vk::binding(0, 2)]] ConstantBuffer<Uniforms> u : register(b0, space2);

[[vk::binding(0, 0)]] StructuredBuffer<float> edge_data : register(t0, space0);
[[vk::binding(1, 0)]] StructuredBuffer<uint> row_data : register(t1, space0);
[[vk::binding(2, 0)]] StructuredBuffer<uint> edge_indices : register(t2, space0);
[[vk::binding(0, 1)]] RWStructuredBuffer<uint> out_colors : register(u0, space1);

float edge_x0(uint i)        { return edge_data[i * 8u + 0u]; }
float edge_y0(uint i)        { return edge_data[i * 8u + 1u]; }
float edge_dx_per_dy(uint i) { return edge_data[i * 8u + 2u]; }
float edge_y_min(uint i)     { return edge_data[i * 8u + 3u]; }
float edge_y_max(uint i)     { return edge_data[i * 8u + 4u]; }
int   edge_dir(uint i)       { return reinterpret<int>(edge_data[i * 8u + 5u]); }
uint  edge_color(uint i)     { return reinterpret<uint>(edge_data[i * 8u + 6u]); }

uint row_start(uint r) { return row_data[r * 2u]; }
uint row_count(uint r) { return row_data[r * 2u + 1u]; }

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint px = tid.x;
    uint py = tid.y;
    if (px >= u.out_w || py >= u.out_h) return;

    uint color = u.bg_color;

    // Single center-point sample: no AA blending, hard region boundaries
    float sx = float(px) + 0.5;
    float sy = float(py) + 0.5;

    uint r_start = row_start(py);
    uint r_count = row_count(py);

    uint prev_color = VOID_COLOR;
    int winding = 0;

    for (uint ri = 0u; ri < r_count; ri++) {
        uint ei = edge_indices[r_start + ri];

        float e_x0 = edge_x0(ei);
        float e_y0 = edge_y0(ei);
        float e_dx = edge_dx_per_dy(ei);
        float e_ymin = edge_y_min(ei);
        float e_ymax = edge_y_max(ei);
        int e_dir = edge_dir(ei);
        uint e_color = edge_color(ei);

        // New path: apply previous path's winding
        if (e_color != prev_color && prev_color != VOID_COLOR) {
            if (winding != 0) {
                color = prev_color;
            }
            winding = 0;
        }
        prev_color = e_color;

        if (sy >= e_ymin && sy < e_ymax) {
            float ix = e_x0 + (sy - e_y0) * e_dx;
            if (sx < ix) winding += e_dir;
        }
    }

    // Final path
    if (prev_color != VOID_COLOR && winding != 0) {
        color = prev_color;
    }

    out_colors[py * u.out_w + px] = color;
}
