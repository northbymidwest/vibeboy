// Super 2xSaI compute shader by Kreed -- smoother variant that interpolates all four output pixels.

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
    // n unused
    uint o = fetch(src_i + int2( 1,  2));
    // p16 unused

    // Bottom-right output pixel (product3)
    uint product3;
    if (b == c && a != d) {
        product3 = b;
    } else if (a == d && b != c) {
        product3 = a;
    } else if (a == d && b == c) {
        if (a == b) {
            product3 = a;
        } else {
            int r = 0;
            if (a == e || a == g) r += 1;
            if (a == i || a == m) r += 1;
            if (b == f || b == k) r -= 1;
            if (b == j || b == l) r -= 1;
            product3 = (r > 0) ? a : (r < 0) ? b : qinterp(a, b, c, d);
        }
    } else {
        product3 = qinterp(a, b, c, d);
    }

    // Top-right output pixel (product1)
    uint product1;
    if (a == d && b != c && a == e && a != l) {
        product1 = interp3_1(a, b);
    } else if (b == c && a != d && b == f && b != h) {
        product1 = interp3_1(b, a);
    } else if (b == c && a != d && b == e && b != j) {
        product1 = interp3_1(b, a);
    } else if (a == d && b != c && a == f && a != k) {
        product1 = interp3_1(a, b);
    } else {
        product1 = interp(a, b);
    }

    // Bottom-left output pixel (product2)
    uint product2;
    if (a == d && b != c && a == g && a != o) {
        product2 = interp3_1(a, c);
    } else if (b == c && a != d && c == h && c != e) {
        product2 = interp3_1(c, a);
    } else if (b == c && a != d && c == g && c != m) {
        product2 = interp3_1(c, a);
    } else if (a == d && b != c && a == h && a != l) {
        product2 = interp3_1(a, c);
    } else {
        product2 = interp(a, c);
    }

    // Top-left output pixel (product0)
    uint product0;
    if (a == d && b != c && (a == e || a == g)) {
        product0 = a;
    } else if (b == c && a != d && (b == f || c == h)) {
        product0 = interp(a, b);
    } else {
        product0 = a;
    }

    // Select output pixel based on sub-pixel position
    uint result;
    if (sub.x == 0 && sub.y == 0) result = product0;
    else if (sub.x == 1 && sub.y == 0) result = product1;
    else if (sub.x == 0 && sub.y == 1) result = product2;
    else result = product3;

    output_img.Store(int2(opx, opy), (result).unpack_rgba());
}
