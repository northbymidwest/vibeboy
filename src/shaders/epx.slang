// EPX / Scale2x / Scale4x compute shader -- pixel-art scaling by Eric Johnston.
// Comparisons use packed u32 colors for exact integer matching.

import modules.color;
import modules.pixel_source;

struct Uniforms {
    uint src_w;
    uint src_h;
    uint out_w;
    uint out_h;
    uint iscale;  // 2 or 4
    uint _pad0;
    uint _pad1;
    uint _pad2;
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

uint fetch(int2 coord) { return px_raw(pixels, int(u.src_w), int(u.src_h), coord); }

uint epx(uint p, uint a, uint b, uint c, uint d, int sx, int sy) {
    if (sx == 0 && sy == 0) return (c==a && c!=d && a!=b) ? a : p;
    if (sx == 1 && sy == 0) return (a==b && a!=c && b!=d) ? b : p;
    if (sx == 0 && sy == 1) return (d==c && d!=b && c!=a) ? c : p;
    return (b==d && b!=a && d!=c) ? d : p;
}

uint epx_at(int2 pos) {
    int2 src = pos / 2;
    int2 sub = pos - src * 2;
    uint p = fetch(src);
    uint a = fetch(src + int2(0, -1));
    uint b = fetch(src + int2(1, 0));
    uint c = fetch(src + int2(-1, 0));
    uint d = fetch(src + int2(0, 1));
    return epx(p, a, b, c, d, sub.x, sub.y);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint opx = tid.x;
    uint opy = tid.y;
    if (opx >= u.out_w || opy >= u.out_h) return;

    // Integer mapping: output pixel to source pixel + sub-pixel position
    int isc = int(u.iscale);
    int2 out_pos = int2(opx, opy);
    int2 src_i = clamp(out_pos / isc, int2(0), int2(int(u.src_w) - 1, int(u.src_h) - 1));
    int2 sub = out_pos - src_i * isc;

    uint result;
    if (u.iscale == 2u) {
        uint p = fetch(src_i);
        uint a = fetch(src_i + int2(0, -1));
        uint b = fetch(src_i + int2(1, 0));
        uint c = fetch(src_i + int2(-1, 0));
        uint d = fetch(src_i + int2(0, 1));
        result = epx(p, a, b, c, d, sub.x, sub.y);
    } else {
        int2 mid_pos = src_i * 2 + sub / 2;
        int2 sub4 = sub - (sub / 2) * 2;
        uint p = epx_at(mid_pos);
        uint a = epx_at(mid_pos + int2(0, -1));
        uint b = epx_at(mid_pos + int2(1, 0));
        uint c = epx_at(mid_pos + int2(-1, 0));
        uint d = epx_at(mid_pos + int2(0, 1));
        result = epx(p, a, b, c, d, sub4.x, sub4.y);
    }

    output_img.Store(int2(opx, opy), (result).unpack_rgba());
}
