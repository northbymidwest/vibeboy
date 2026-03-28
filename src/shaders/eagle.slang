// Eagle -- 2x pixel-art scaling (compute shader).
// Comparisons use packed u32 colors for exact integer matching.

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

uint fetch(int2 coord) { return px_raw(pixels, int(u.src_w), int(u.src_h), coord); }

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint opx = tid.x;
    uint opy = tid.y;
    if (opx >= u.out_w || opy >= u.out_h) return;

    // Integer mapping: fixed 2x scale
    int2 out_pos = int2(opx, opy);
    int2 src_i = clamp(out_pos / 2, int2(0), int2(int(u.src_w) - 1, int(u.src_h) - 1));
    int2 sub = out_pos - src_i * 2;

    uint p = fetch(src_i);
    uint s = fetch(src_i + int2(-1, -1));
    uint t = fetch(src_i + int2( 0, -1));
    uint uu = fetch(src_i + int2( 1, -1));
    uint v = fetch(src_i + int2(-1,  0));
    uint w = fetch(src_i + int2( 1,  0));
    uint x = fetch(src_i + int2(-1,  1));
    uint y = fetch(src_i + int2( 0,  1));
    uint z = fetch(src_i + int2( 1,  1));

    uint result;
    if      (sub.x == 0 && sub.y == 0) result = (s==t && s==v) ? s : p;
    else if (sub.x == 1 && sub.y == 0) result = (t==uu && uu==w) ? uu : p;
    else if (sub.x == 0 && sub.y == 1) result = (v==x && x==y) ? x : p;
    else                               result = (w==z && z==y) ? z : p;

    output_img.Store(int2(opx, opy), (result).unpack_rgba());
}
