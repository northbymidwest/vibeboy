// Bilinear scaling compute shader.

import modules.color;
import modules.pixel_source;

struct Uniforms {
    uint src_w;
    uint src_h;
    uint out_w;
    uint out_h;
};
[[vk::binding(0, 2)]]
#ifdef DXIL_TARGET
ConstantBuffer<Uniforms> u : register(b0, space2);
#else
ConstantBuffer<Uniforms> u;
#endif

[[vk::binding(0, 0)]]
#ifdef DXIL_TARGET
StructuredBuffer<uint> pixels : register(t0, space0);
#else
StructuredBuffer<uint> pixels;
#endif
[[vk::image_format("rgba8")]]
[[vk::binding(0, 1)]]
#ifdef DXIL_TARGET
WTexture2D<float4> output_img : register(u0, space1);
#else
WTexture2D<float4> output_img;
#endif

float3 fetch_rgb(int2 coord) { return px_rgb(pixels, int(u.src_w), int(u.src_h), coord); }

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    int2 pos = int2(tid.xy);
    if (pos.x >= int(u.out_w) || pos.y >= int(u.out_h)) return;

    float sx = (float(pos.x) + 0.5) * float(u.src_w) / float(u.out_w) - 0.5;
    float sy = (float(pos.y) + 0.5) * float(u.src_h) / float(u.out_h) - 0.5;

    int x0 = int(floor(sx));
    int y0 = int(floor(sy));
    float fx = sx - float(x0);
    float fy = sy - float(y0);

    float3 c00 = fetch_rgb(int2(x0,     y0));
    float3 c10 = fetch_rgb(int2(x0 + 1, y0));
    float3 c01 = fetch_rgb(int2(x0,     y0 + 1));
    float3 c11 = fetch_rgb(int2(x0 + 1, y0 + 1));

    float3 color = lerp(lerp(c00, c10, fx), lerp(c01, c11, fx), fy);

    output_img.Store(pos, float4(color, 1.0));
}
