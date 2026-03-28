// Scanline rasterizer with 2x2 supersampling, nonzero winding.

import modules.color;
import modules.vectorize;

struct Uniforms {
    uint out_w;
    uint out_h;
    uint num_edges;
    uint bg_color;
};
[[vk::binding(0, 2)]] ConstantBuffer<Uniforms> u;

[[vk::binding(0, 0)]] StructuredBuffer<float> edge_data;
[[vk::binding(1, 0)]] StructuredBuffer<uint> row_data;
[[vk::binding(2, 0)]] StructuredBuffer<uint> edge_indices;
[[vk::image_format("rgba8")]]
[[vk::binding(0, 1)]] WTexture2D<float4> output_img;

// Edge field accessors (32 bytes per edge = 8 floats)
float edge_x0(uint i)        { return edge_data[i * 8u + 0u]; }
float edge_y0(uint i)        { return edge_data[i * 8u + 1u]; }
float edge_dx_per_dy(uint i) { return edge_data[i * 8u + 2u]; }
float edge_y_min(uint i)     { return edge_data[i * 8u + 3u]; }
float edge_y_max(uint i)     { return edge_data[i * 8u + 4u]; }
int   edge_dir(uint i)       { return reinterpret<int>(edge_data[i * 8u + 5u]); }
uint  edge_color(uint i)     { return reinterpret<uint>(edge_data[i * 8u + 6u]); }

// Row range accessors (8 bytes per row = 2 uints)
uint row_start(uint r) { return row_data[r * 2u]; }
uint row_count(uint r) { return row_data[r * 2u + 1u]; }

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint px = tid.x;
    uint py = tid.y;
    if (px >= u.out_w || py >= u.out_h) {
        return;
    }

    uint color = u.bg_color;
    float x_left = float(px);

    // 2x2 supersampling sample points
    float sy0 = float(py) + 0.25;
    float sy1 = float(py) + 0.75;
    float sx0 = x_left + 0.25;
    float sx1 = x_left + 0.75;

    uint r_start = row_start(py);
    uint r_count = row_count(py);

    uint prev_color = VOID_COLOR;
    int w00 = 0, w10 = 0, w01 = 0, w11 = 0;

    for (uint ri = 0u; ri < r_count; ri++) {
        uint ei = edge_indices[r_start + ri];

        float e_x0 = edge_x0(ei);
        float e_y0 = edge_y0(ei);
        float e_dx = edge_dx_per_dy(ei);
        float e_ymin = edge_y_min(ei);
        float e_ymax = edge_y_max(ei);
        int e_dir = edge_dir(ei);
        uint e_color = edge_color(ei);

        if (e_color != prev_color && prev_color != VOID_COLOR) {
            uint coverage = 0u;
            if (w00 != 0) coverage++;
            if (w10 != 0) coverage++;
            if (w01 != 0) coverage++;
            if (w11 != 0) coverage++;
            if (coverage == 4u) {
                color = prev_color;
            } else if (coverage > 0u) {
                color = blend_over(color, prev_color, coverage, 4u);
            }
            w00 = 0; w10 = 0; w01 = 0; w11 = 0;
        }
        prev_color = e_color;

        // Row 0 (sy = y + 0.25)
        if (sy0 >= e_ymin && sy0 < e_ymax) {
            float ix = e_x0 + (sy0 - e_y0) * e_dx;
            if (sx0 < ix) w00 += e_dir;
            if (sx1 < ix) w10 += e_dir;
        }

        // Row 1 (sy = y + 0.75)
        if (sy1 >= e_ymin && sy1 < e_ymax) {
            float ix = e_x0 + (sy1 - e_y0) * e_dx;
            if (sx0 < ix) w01 += e_dir;
            if (sx1 < ix) w11 += e_dir;
        }
    }

    // Composite final path
    if (prev_color != VOID_COLOR) {
        uint coverage = 0u;
        if (w00 != 0) coverage++;
        if (w10 != 0) coverage++;
        if (w01 != 0) coverage++;
        if (w11 != 0) coverage++;
        if (coverage == 4u) {
            color = prev_color;
        } else if (coverage > 0u) {
            color = blend_over(color, prev_color, coverage, 4u);
        }
    }

    output_img.Store(int2(tid.xy), (color).unpack_rgba());
}
