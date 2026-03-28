// OmniScale Legacy -- diagonal-aware pixel-art upscaler with quantized fallback.
//
// Algorithm designed by Lior Halphon.
//
// Detects and preserves diagonal edges in 2x2 source quads. When a diagonal
// is found, the sub-pixel position along that diagonal picks the winning color.
// No diagonal -> bilinear blend quantized to the nearest source color.

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

float4 fetch(int2 coord) { return px_rgba(pixels, int(u.src_w), int(u.src_h), coord); }

// Manhattan RGB distance.
float channel_dist(float4 a, float4 b) {
    float3 d = abs(a.rgb - b.rgb);
    return (d.r + d.g + d.b) * 255.0;
}

// Perceptual similarity -- threshold of 15 in Manhattan RGB.
bool alike(float4 a, float4 b) {
    return all(a == b) || channel_dist(a, b) < 15.0;
}

// Fixed-point blend: t in [0, 1].
float4 blend_colors(float4 a, float4 b, float t) {
    return lerp(a, b, clamp(t, 0.0, 1.0));
}

// Module-scope globals for array parameters.
static float4 g_c[4];

// Pick color for a rising (/) diagonal.
float4 pick_rising(float fx, float fy) {
    float s = fx + fy;
    if (s < 0.5) return g_c[0];
    if (s > 1.5) return g_c[3];
    return g_c[2]; // on-diagonal
}

// Pick color for a falling (\) diagonal.
float4 pick_falling(float fx, float fy) {
    float d = 1.0 - fx + fy;
    if (d < 0.5) return g_c[1];
    if (d > 1.5) return g_c[2];
    return g_c[0]; // on-diagonal
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    int2 pos = int2(tid.xy);
    if (pos.x >= int(u.out_w) || pos.y >= int(u.out_h)) return;

    // Map output pixel to source coordinates (centered on pixel boundaries).
    float2 src_coord = (float2(pos) + 0.5) / float2(u.out_w, u.out_h) * float2(u.src_w, u.src_h) - 0.5;
    int2 src_i = int2(floor(src_coord));
    float2 f = src_coord - float2(src_i); // fractional [0, 1)

    // Sample the 2x2 quad.
    g_c[0] = fetch(src_i);               // top-left
    g_c[1] = fetch(src_i + int2(1, 0)); // top-right
    g_c[2] = fetch(src_i + int2(0, 1)); // bottom-left
    g_c[3] = fetch(src_i + int2(1, 1)); // bottom-right

    float4 result;

    // Diagonal detection.
    bool rising  = alike(g_c[2], g_c[1]); // / diagonal
    bool falling = alike(g_c[0], g_c[3]); // \ diagonal

    if (rising && falling) {
        if (alike(g_c[0], g_c[2])) {
            // All four similar -- flat fill
            result = g_c[0];
        } else {
            // Both diagonals -- 4x4 neighborhood vote to break tie
            int bias = 0;
            for (int row = -1; row < 3; row++) {
                for (int col = -1; col < 3; col++) {
                    float4 n = fetch(src_i + int2(col, row));
                    if (alike(n, g_c[0])) bias++;
                    if (alike(n, g_c[2])) bias--;
                }
            }
            if (bias < 0) {
                result = pick_falling(f.x, f.y);
            } else if (bias > 0) {
                result = pick_rising(f.x, f.y);
            } else {
                // Tie -- average both
                result = blend_colors(pick_falling(f.x, f.y), pick_rising(f.x, f.y), 0.5);
            }
        }
    } else if (rising) {
        result = pick_rising(f.x, f.y);
    } else if (falling) {
        result = pick_falling(f.x, f.y);
    } else {
        // No diagonal -- bilinear, then quantize to nearest source color
        float4 top = blend_colors(g_c[0], g_c[1], f.x);
        float4 bot = blend_colors(g_c[2], g_c[3], f.x);
        float4 mixed = blend_colors(top, bot, f.y);

        float d0 = channel_dist(mixed, g_c[0]);
        float d1 = channel_dist(mixed, g_c[1]);
        float d2 = channel_dist(mixed, g_c[2]);
        float d3 = channel_dist(mixed, g_c[3]);
        float min_d = min(min(d0, d1), min(d2, d3));

        if (d0 == min_d) result = g_c[0];
        else if (d1 == min_d) result = g_c[1];
        else if (d2 == min_d) result = g_c[2];
        else result = g_c[3];
    }

    output_img.Store(pos, result);
}
