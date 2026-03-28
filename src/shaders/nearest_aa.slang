// Antialiased nearest-neighbor scaling compute shader.

import modules.color;
import modules.pixel_source;

struct Uniforms {
    uint src_w;
    uint src_h;
    uint out_w;
    uint out_h;
};
[[vk::binding(0, 2)]] ConstantBuffer<Uniforms> u;

[[vk::binding(0, 0)]] StructuredBuffer<uint> pixels;
[[vk::image_format("rgba8")]]
[[vk::binding(0, 1)]] WTexture2D<float4> output_img;

float3 fetch_rgb(int2 coord) { return px_rgb(pixels, int(u.src_w), int(u.src_h), coord); }

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint opx = tid.x;
    uint opy = tid.y;
    if (opx >= u.out_w || opy >= u.out_h) return;

    float2 pixel = float2(float(u.src_w), float(u.src_h)) / float2(float(u.out_w), float(u.out_h));
    float2 uv = (float2(opx, opy) + 0.5) / float2(float(u.out_w), float(u.out_h));
    float2 src0 = uv * float2(float(u.src_w), float(u.src_h)) - pixel * 0.5;
    float2 src1 = src0 + pixel;

    float2 src_max = float2(float(u.src_w) - 1.0, float(u.src_h) - 1.0);
    int2 i0 = int2(clamp(floor(src0), float2(0.0), src_max));
    int2 i1 = int2(clamp(floor(src1 - float2(1e-5)), float2(0.0), src_max));

    if (all(i0 == i1)) {
        output_img.Store(int2(opx, opy), float4(fetch_rgb(i0), 1.0));
    } else {
        float3 acc = float3(0.0);
        float area_total = 0.0;

        for (int iy = i0.y; iy <= i1.y; iy++) {
            float top = max(float(iy), src0.y);
            float bot = min(float(iy + 1), src1.y);
            float h = bot - top;
            if (h <= 0.0) continue;

            for (int ix = i0.x; ix <= i1.x; ix++) {
                float left = max(float(ix), src0.x);
                float right = min(float(ix + 1), src1.x);
                float w = right - left;
                if (w <= 0.0) continue;

                float area = w * h;
                acc += fetch_rgb(int2(ix, iy)) * area;
                area_total += area;
            }
        }

        output_img.Store(int2(opx, opy), float4(acc / max(area_total, 1e-5), 1.0));
    }
}
