// Scale3x -- 3x pixel-art scaling (compute shader).
// Fixed 3x scale: output must be exactly src_w*3 x src_h*3.
// Comparisons use packed u32 colors for exact integer matching.

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

uint fetch(int2 coord) { return px_raw(pixels, int(u.src_w), int(u.src_h), coord); }

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint opx = tid.x;
    uint opy = tid.y;
    if (opx >= u.out_w || opy >= u.out_h) return;

    int sx = int(opx) / 3;
    int sy = int(opy) / 3;
    int sub_x = int(opx) % 3;
    int sub_y = int(opy) % 3;
    int idx = sub_y * 3 + sub_x;

    int2 p = int2(sx, sy);
    uint a = fetch(p + int2(-1, -1));
    uint b = fetch(p + int2( 0, -1));
    uint c = fetch(p + int2( 1, -1));
    uint d = fetch(p + int2(-1,  0));
    uint e = fetch(p);
    uint f = fetch(p + int2( 1,  0));
    uint g = fetch(p + int2(-1,  1));
    uint h = fetch(p + int2( 0,  1));
    uint i = fetch(p + int2( 1,  1));

    uint result = e;

    if (b != h && d != f) {
        if      (idx == 0) result = (d == b) ? d : e;
        else if (idx == 1) result = ((d == b && e != c) || (b == f && e != a)) ? b : e;
        else if (idx == 2) result = (b == f) ? f : e;
        else if (idx == 3) result = ((d == b && e != g) || (d == h && e != a)) ? d : e;
        // idx == 4: center, stays e
        else if (idx == 5) result = ((b == f && e != i) || (h == f && e != c)) ? f : e;
        else if (idx == 6) result = (d == h) ? d : e;
        else if (idx == 7) result = ((d == h && e != i) || (h == f && e != g)) ? h : e;
        else               result = (h == f) ? f : e;
    }

    output_img.Store(int2(opx, opy), (result).unpack_rgba());
}
