// DCCI -- Directional Cubic Convolution Interpolation (2x upscaling).
// Sobel-based edge direction detection with 1D cubic convolution along edges.

import modules.color;
import modules.pixel_source;

struct Uniforms {
    uint src_w;
    uint src_h;
    uint out_w;
    uint out_h;
    uint iscale;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
[[vk::binding(0, 2)]] ConstantBuffer<Uniforms> u : register(b0, space2);

[[vk::binding(0, 0)]] StructuredBuffer<uint> pixels : register(t0, space0);
[[vk::image_format("rgba8")]]
[[vk::binding(0, 1)]] WTexture2D<float4> output_img : register(u0, space1);

float4 fetch(int2 coord) { return px_rgba(pixels, int(u.src_w), int(u.src_h), coord); }

float luma(float4 c) {
    return 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
}

float luma_at(int2 coord) {
    return luma(fetch(coord));
}

// Catmull-Rom cubic weight (a = -0.5)
float cubic_weight(float d) {
    d = abs(d);
    if (d < 1.0) return (1.5 * d - 2.5) * d * d + 1.0;
    if (d < 2.0) return ((-0.5 * d + 2.5) * d - 4.0) * d + 2.0;
    return 0.0;
}

// Blend 4 colors with cubic weights (auto-normalized)
float4 cubic_blend4(float4 c0, float4 c1, float4 c2, float4 c3, float w0, float w1, float w2, float w3) {
    float s = w0 + w1 + w2 + w3;
    float inv = abs(s) > 0.001 ? 1.0 / s : 0.25;
    return c0 * (w0 * inv) + c1 * (w1 * inv) + c2 * (w2 * inv) + c3 * (w3 * inv);
}

// Detect edge direction: 0=flat, 1=vertical edge (horiz interp),
// 2=horizontal edge (vert interp), 3=main diagonal(\), 4=anti diagonal(/)
int detect_direction(int2 p) {
    // 3x3 Sobel on luma, centered on (p.x, p.y) to (p.x+1, p.y+1)
    float tl = luma_at(p);
    float tc = luma_at(p + int2(1, 0));
    float tr = luma_at(p + int2(2, 0));
    float ml = luma_at(p + int2(0, 1));
    float mr = luma_at(p + int2(2, 1));
    float bl = luma_at(p + int2(0, 2));
    float bc = luma_at(p + int2(1, 2));
    float br = luma_at(p + int2(2, 2));

    float gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
    float gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);

    float strength = sqrt(gx * gx + gy * gy);
    if (strength < 10.0) return 0;

    float deg = degrees(atan2(gy, gx));
    float abs_deg = abs(deg);

    if (abs_deg < 22.5 || abs_deg > 157.5) return 1;
    if (abs_deg > 67.5 && abs_deg < 112.5) return 2;
    if ((deg > 22.5 && deg < 67.5) || (deg < -112.5 && deg > -157.5)) return 4;
    return 3;
}

float4 diamond_interp(int2 p) {
    int dir = detect_direction(p);

    float w0 = cubic_weight(1.5);
    float w1 = cubic_weight(0.5);

    if (dir == 1) {
        // Horizontal interpolation (vertical edge)
        float4 top = cubic_blend4(
            fetch(p + int2(-1, 0)), fetch(p), fetch(p + int2(1, 0)), fetch(p + int2(2, 0)),
            w0, w1, w1, w0);
        float4 bot = cubic_blend4(
            fetch(p + int2(-1, 1)), fetch(p + int2(0, 1)), fetch(p + int2(1, 1)), fetch(p + int2(2, 1)),
            w0, w1, w1, w0);
        return lerp(top, bot, 0.5);
    }
    if (dir == 2) {
        // Vertical interpolation (horizontal edge)
        float4 left = cubic_blend4(
            fetch(p + int2(0, -1)), fetch(p), fetch(p + int2(0, 1)), fetch(p + int2(0, 2)),
            w0, w1, w1, w0);
        float4 right = cubic_blend4(
            fetch(p + int2(1, -1)), fetch(p + int2(1, 0)), fetch(p + int2(1, 1)), fetch(p + int2(1, 2)),
            w0, w1, w1, w0);
        return lerp(left, right, 0.5);
    }
    if (dir == 3) {
        // Main diagonal (\)
        return cubic_blend4(
            fetch(p + int2(-1, -1)), fetch(p), fetch(p + int2(1, 1)), fetch(p + int2(2, 2)),
            w0, w1, w1, w0);
    }
    if (dir == 4) {
        // Anti-diagonal (/)
        return cubic_blend4(
            fetch(p + int2(2, -1)), fetch(p + int2(1, 0)), fetch(p + int2(0, 1)), fetch(p + int2(-1, 2)),
            w0, w1, w1, w0);
    }
    // Flat: standard bilinear average
    return lerp(lerp(fetch(p), fetch(p + int2(1, 0)), 0.5),
               lerp(fetch(p + int2(0, 1)), fetch(p + int2(1, 1)), 0.5), 0.5);
}

float4 hedge_interp(int2 p) {
    float w0 = cubic_weight(1.5);
    float w1 = cubic_weight(0.5);
    return cubic_blend4(
        fetch(p + int2(-1, 0)), fetch(p), fetch(p + int2(1, 0)), fetch(p + int2(2, 0)),
        w0, w1, w1, w0);
}

float4 vedge_interp(int2 p) {
    float w0 = cubic_weight(1.5);
    float w1 = cubic_weight(0.5);
    return cubic_blend4(
        fetch(p + int2(0, -1)), fetch(p), fetch(p + int2(0, 1)), fetch(p + int2(0, 2)),
        w0, w1, w1, w0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint opx = tid.x;
    uint opy = tid.y;
    if (opx >= u.out_w || opy >= u.out_h) return;

    int2 src_i = int2(opx / 2u, opy / 2u);
    int2 sub = int2(opx & 1u, opy & 1u);

    float4 result;
    if (sub.x == 0 && sub.y == 0) {
        result = fetch(src_i);
    } else if (sub.x == 1 && sub.y == 0) {
        result = hedge_interp(src_i);
    } else if (sub.x == 0 && sub.y == 1) {
        result = vedge_interp(src_i);
    } else {
        result = diamond_interp(src_i);
    }

    output_img.Store(int2(opx, opy), result);
}
