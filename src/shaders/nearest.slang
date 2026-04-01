// Nearest-neighbor scaling compute shader.

import modules.color;

struct Uniforms {
    uint src_w;
    uint src_h;
    uint out_w;
    uint out_h;
};
[[vk::binding(0, 2)]] ConstantBuffer<Uniforms> u : register(b0, space2);

[[vk::binding(0, 0)]] StructuredBuffer<uint> pixels : register(t0, space0);
[[vk::image_format("rgba8")]]
[[vk::binding(0, 1)]] WTexture2D<float4> output_img : register(u0, space1);

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    int2 pos = int2(tid.xy);
    if (pos.x >= int(u.out_w) || pos.y >= int(u.out_h)) return;

    int sx = clamp(int(float(pos.x) * float(u.src_w) / float(u.out_w)), 0, int(u.src_w) - 1);
    int sy = clamp(int(float(pos.y) * float(u.src_h) / float(u.out_h)), 0, int(u.src_h) - 1);
    uint c = pixels[sy * int(u.src_w) + sx];
    output_img.Store(pos, (c).unpack_rgba());
}
