// Bicubic (Catmull-Rom) interpolation compute shader.

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

float4 cubic_weight(float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    return float4(
        -0.5 * t3 + t2 - 0.5 * t,
         1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
         0.5 * t3 - 0.5 * t2
    );
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint opx = tid.x;
    uint opy = tid.y;
    if (opx >= u.out_w || opy >= u.out_h) return;

    float2 src_coord = (float2(opx, opy) + 0.5) * float2(float(u.src_w) / float(u.out_w), float(u.src_h) / float(u.out_h)) - 0.5;
    int2 src_i = int2(floor(src_coord));
    float2 f = src_coord - float2(src_i);

    float4 wx = cubic_weight(f.x);
    float4 wy = cubic_weight(f.y);

    float3 color = float3(0.0);
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            float w = wx[i] * wy[j];
            color += fetch_rgb(src_i + int2(i - 1, j - 1)) * w;
        }
    }

    output_img.Store(int2(opx, opy), float4(clamp(color, 0.0, 1.0), 1.0));
}
