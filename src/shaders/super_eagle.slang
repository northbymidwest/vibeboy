// Super Eagle compute shader by Kreed -- diagonal-aware 2x scaling.

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
[[vk::binding(0, 2)]] ConstantBuffer<Uniforms> u;

[[vk::binding(0, 0)]] StructuredBuffer<uint> pixels;
[[vk::image_format("rgba8")]]
[[vk::binding(0, 1)]] WTexture2D<float4> output_img;

uint fetch(int2 coord) { return px_raw(pixels, int(u.src_w), int(u.src_h), coord); }

// GetResult vote: bias toward a or b based on neighbors c, d.
int get_result(uint a, uint b, uint c, uint d) {
    int x = 0;
    int y = 0;
    if (a == c) x += 1; else if (b == c) y += 1;
    if (a == d) x += 1; else if (b == d) y += 1;
    int r = 0;
    if (x <= 1) r += 1;
    if (y <= 1) r -= 1;
    return r;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint opx = tid.x;
    uint opy = tid.y;
    if (opx >= u.out_w || opy >= u.out_h) return;

    int2 out_pos = int2(opx, opy);
    int2 src_i = out_pos / 2;
    int2 sub = out_pos - src_i * 2;

    // Map to Kreed naming from 4x4 neighborhood
    uint color_b1 = fetch(src_i + int2(-1, -1));
    uint color_b2 = fetch(src_i + int2( 0, -1));
    uint color4   = fetch(src_i + int2(-1,  0));
    uint color5   = fetch(src_i + int2( 0,  0));  // center
    uint color6   = fetch(src_i + int2( 1,  0));  // right
    uint color_s2 = fetch(src_i + int2( 2,  0));
    uint color1   = fetch(src_i + int2(-1,  1));
    uint color2   = fetch(src_i + int2( 0,  1));  // below
    uint color3   = fetch(src_i + int2( 1,  1));  // below-right
    uint color_s1 = fetch(src_i + int2( 2,  1));
    uint color_a1 = fetch(src_i + int2( 0,  2));
    uint color_a2 = fetch(src_i + int2( 1,  2));

    uint e0, e1, e2, e3;

    if (color2 == color6 && color5 != color3) {
        // Anti-diagonal wins
        e1 = color2;
        e2 = color2;
        if (color1 == color2 || color6 == color_s2) {
            e0 = interp3_1(color2, color5);
        } else {
            e0 = interp(color5, color6);
        }
        if (color2 == color_a1 || color6 == color_b2) {
            e3 = interp3_1(color2, color3);
        } else {
            e3 = interp(color2, color3);
        }
    } else if (color5 == color3 && color2 != color6) {
        // Main diagonal wins
        e3 = color5;
        e0 = color5;
        if (color_b1 == color5 || color3 == color_a2) {
            e1 = interp3_1(color5, color6);
        } else {
            e1 = interp(color5, color6);
        }
        if (color4 == color5 || color3 == color_s1) {
            e2 = interp3_1(color5, color2);
        } else {
            e2 = interp(color2, color3);
        }
    } else if (color5 == color3 && color2 == color6 && color5 != color6) {
        // Both diagonals match -- vote
        int r = get_result(color6, color5, color1, color_a1)
              + get_result(color6, color5, color4, color_b1)
              + get_result(color6, color5, color_a2, color_s1)
              + get_result(color6, color5, color_b2, color_s2);

        if (r > 0) {
            e1 = color2;
            e2 = color2;
            e0 = interp(color5, color6);
            e3 = interp(color5, color6);
        } else if (r < 0) {
            e0 = color5;
            e3 = color5;
            e1 = interp(color5, color6);
            e2 = interp(color5, color6);
        } else {
            e0 = color5;
            e3 = color5;
            e1 = color2;
            e2 = color2;
        }
    } else {
        // Center output on the input pixel to avoid half-pixel shift
        e0 = color5;
        e1 = interp(color5, color6);
        e2 = interp(color5, color2);
        e3 = qinterp(color5, color6, color2, color3);
    }

    // Select output pixel based on sub-pixel position
    uint result;
    if (sub.x == 0 && sub.y == 0) result = e0;
    else if (sub.x == 1 && sub.y == 0) result = e1;
    else if (sub.x == 0 && sub.y == 1) result = e2;
    else result = e3;

    output_img.Store(int2(opx, opy), (result).unpack_rgba());
}
