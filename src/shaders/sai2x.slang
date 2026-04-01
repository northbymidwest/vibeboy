// 2xSaI (Scale and Interpolation) compute shader by Kreed.
// Each workgroup thread handles one output pixel (2x scaled).

import modules.color;
import modules.pixel_source;
import modules.sai_blend;

struct Uniforms {
    uint src_w;
    uint src_h;
    uint out_w;
    uint out_h;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    uint _pad3;
};
[[vk::binding(0, 2)]] ConstantBuffer<Uniforms> u : register(b0, space2);

[[vk::binding(0, 0)]] StructuredBuffer<uint> pixels : register(t0, space0);
[[vk::image_format("rgba8")]]
[[vk::binding(0, 1)]] WTexture2D<float4> output_img : register(u0, space1);

uint fetch(int2 coord) { return px_raw(pixels, int(u.src_w), int(u.src_h), coord); }

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint opx = tid.x;
    uint opy = tid.y;
    if (opx >= u.out_w || opy >= u.out_h) return;

    int2 out_pos = int2(opx, opy);
    int2 src_i = out_pos / 2;
    int2 sub = out_pos - src_i * 2;

    // Sample 4x4 neighborhood
    uint i = fetch(src_i + int2(-1, -1));
    uint e = fetch(src_i + int2( 0, -1));
    uint f = fetch(src_i + int2( 1, -1));
    uint j = fetch(src_i + int2( 2, -1));
    uint g = fetch(src_i + int2(-1,  0));
    uint a = fetch(src_i + int2( 0,  0));
    uint b = fetch(src_i + int2( 1,  0));
    uint k = fetch(src_i + int2( 2,  0));
    uint h = fetch(src_i + int2(-1,  1));
    uint c = fetch(src_i + int2( 0,  1));
    uint d = fetch(src_i + int2( 1,  1));
    uint l = fetch(src_i + int2( 2,  1));
    uint m = fetch(src_i + int2(-1,  2));
    // n = fetch(src_i + int2( 0,  2)); // unused
    uint o = fetch(src_i + int2( 1,  2));
    // p16 = fetch(src_i + int2( 2,  2)); // unused

    // Compute all four output pixels
    uint product, product1, product2;

    if (a == d && b != c) {
        product = ((a == e && b == l) || (a == c && a == f && b != e && b == j)) ? a : interp(a, b);
        product1 = ((a == g && c == o) || (a == b && a == h && g != c && c == m)) ? a : interp(a, c);
        product2 = a;
    } else if (b == c && a != d) {
        product = ((b == f && a == h) || (b == e && b == d && a != f && a == i)) ? b : interp(a, b);
        product1 = ((c == h && a == f) || (c == g && c == d && a != h && a == i)) ? c : interp(a, c);
        product2 = b;
    } else if (a == d && b == c) {
        if (a == b) {
            product = a;
            product1 = a;
            product2 = a;
        } else {
            int r = 0;
            product = interp(a, b);
            product1 = interp(a, c);

            if (a == g || a == e) r += 1;
            if (a == i || a == m) r += 1;
            if (b == k || b == f) r -= 1;
            if (b == j || b == l) r -= 1;

            product2 = (r > 0) ? a : (r < 0) ? b : qinterp(a, b, c, d);
        }
    } else {
        product2 = qinterp(a, b, c, d);
        product = (a == c && a == f && b != e && b == j) ? a
                : (b == e && b == d && a != f && a == i) ? b
                : interp(a, b);
        product1 = (a == b && a == h && g != c && c == m) ? a
                 : (c == g && c == d && a != h && a == i) ? c
                 : interp(a, c);
    }

    // Select output pixel based on sub-pixel position
    uint result;
    if (sub.x == 0 && sub.y == 0) result = a;
    else if (sub.x == 1 && sub.y == 0) result = product;
    else if (sub.x == 0 && sub.y == 1) result = product1;
    else result = product2;

    output_img.Store(int2(opx, opy), (result).unpack_rgba());
}
