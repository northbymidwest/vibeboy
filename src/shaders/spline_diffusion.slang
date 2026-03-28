// Gaussian diffusion pass (pass 2 of spline-diffusion pipeline).
// Blends source pixel colors within spline-bounded regions using
// 2x2 supersampling for smooth boundary transitions.

import modules.color;

struct Uniforms {
    uint out_w;
    uint out_h;
    uint src_w;
    uint src_h;
    float inv_scale;
    float gauss_k;
    float radius;
    uint scale_int;
};
[[vk::binding(0, 2)]] ConstantBuffer<Uniforms> u;

[[vk::binding(0, 0)]] StructuredBuffer<uint> src_pixels;
[[vk::binding(1, 0)]] StructuredBuffer<uint> region_colors;
[[vk::image_format("rgba8")]]
[[vk::binding(0, 1)]] WTexture2D<float4> output_img;

// Gaussian-blend nearby source pixel centroids within the same spline-bounded
// region as the sample point. Returns (r, g, b, total_weight) in 0-255 scale.
float4 gaussian_sample(float sx, float sy, uint sample_region) {
    float r_sq = u.radius * u.radius;
    int min_px = max(int(floor(sx - u.radius)), 0);
    int max_px = min(int(ceil(sx + u.radius)), int(u.src_w) - 1);
    int min_py = max(int(floor(sy - u.radius)), 0);
    int max_py = min(int(ceil(sy + u.radius)), int(u.src_h) - 1);

    float tr = 0.0, tg = 0.0, tb = 0.0, tw = 0.0;
    uint half_scale = u.scale_int / 2u;

    for (int py = min_py; py <= max_py; py++) {
        for (int px = min_px; px <= max_px; px++) {
            uint cx_out = min(uint(px) * u.scale_int + half_scale, u.out_w - 1u);
            uint cy_out = min(uint(py) * u.scale_int + half_scale, u.out_h - 1u);
            if (region_colors[cy_out * u.out_w + cx_out] != sample_region) continue;

            float dx = sx - (float(px) + 0.5);
            float dy = sy - (float(py) + 0.5);
            float d_sq = dx * dx + dy * dy;
            if (d_sq > r_sq) continue;

            float w = exp(-d_sq * u.gauss_k);
            uint c = src_pixels[uint(py) * u.src_w + uint(px)];
            tr += w * float(c.channel(16u));
            tg += w * float(c.channel(8u));
            tb += w * float(c.channel(0u));
            tw += w;
        }
    }

    return float4(tr, tg, tb, tw);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint ox = tid.x;
    uint oy = tid.y;
    if (ox >= u.out_w || oy >= u.out_h) return;

    float3 accum = float3(0.0, 0.0, 0.0);

    for (int si = 0; si < 4; si++) {
        float sub_x = float(ox) + ((si & 1) != 0 ? 0.75 : 0.25);
        float sub_y = float(oy) + ((si & 2) != 0 ? 0.75 : 0.25);

        uint rx = min(uint(sub_x), u.out_w - 1u);
        uint ry = min(uint(sub_y), u.out_h - 1u);
        uint sample_region = region_colors[ry * u.out_w + rx];

        float sx = sub_x * u.inv_scale;
        float sy = sub_y * u.inv_scale;

        float4 g = gaussian_sample(sx, sy, sample_region);

        if (g.w > 0.0) {
            float inv_w = 1.0 / g.w;
            accum += float3(
                clamp(g.x * inv_w / 255.0, 0.0, 1.0),
                clamp(g.y * inv_w / 255.0, 0.0, 1.0),
                clamp(g.z * inv_w / 255.0, 0.0, 1.0)
            );
        } else {
            accum += (sample_region).unpack_rgb();
        }
    }

    float4 rgba = float4(accum * 0.25, 1.0);
    output_img.Store(int2(ox, oy), rgba);
}
