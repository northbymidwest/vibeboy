#version 450

// OmniScale Legacy — diagonal-aware pixel-art upscaler with quantized fallback.
//
// Algorithm designed by Lior Halphon.
//
// Detects and preserves diagonal edges in 2x2 source quads. When a diagonal
// is found, the sub-pixel position along that diagonal picks the winning color.
// No diagonal → bilinear blend quantized to the nearest source color.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 2, binding = 0) uniform sampler2D src_tex;

layout(std140, set = 3, binding = 0) uniform Uniforms {
    vec2 src_size;
    vec2 dst_size;
};

vec4 fetch(ivec2 coord) {
    coord = clamp(coord, ivec2(0), ivec2(src_size) - 1);
    return texelFetch(src_tex, coord, 0);
}

// Manhattan RGB distance.
float channel_dist(vec4 a, vec4 b) {
    vec3 d = abs(a.rgb - b.rgb);
    return (d.r + d.g + d.b) * 255.0;
}

// Perceptual similarity — threshold of 15 in Manhattan RGB.
bool alike(vec4 a, vec4 b) {
    return a == b || channel_dist(a, b) < 15.0;
}

// Fixed-point blend: t in [0, 1].
vec4 blend(vec4 a, vec4 b, float t) {
    return mix(a, b, clamp(t, 0.0, 1.0));
}

// Pick color for a rising (/) diagonal.
vec4 pick_rising(vec4 c[4], float fx, float fy) {
    float s = fx + fy;
    if (s < 0.5) return c[0];
    if (s > 1.5) return c[3];
    return c[2]; // on-diagonal
}

// Pick color for a falling (\) diagonal.
vec4 pick_falling(vec4 c[4], float fx, float fy) {
    float d = 1.0 - fx + fy;
    if (d < 0.5) return c[1];
    if (d > 1.5) return c[2];
    return c[0]; // on-diagonal
}

void main() {
    // Map output pixel to source coordinates (centered on pixel boundaries).
    vec2 src_coord = v_uv * src_size - 0.5;
    ivec2 src_i = ivec2(floor(src_coord));
    vec2 f = src_coord - vec2(src_i); // fractional [0, 1)

    // Sample the 2x2 quad.
    vec4 c[4];
    c[0] = fetch(src_i);             // top-left
    c[1] = fetch(src_i + ivec2(1, 0)); // top-right
    c[2] = fetch(src_i + ivec2(0, 1)); // bottom-left
    c[3] = fetch(src_i + ivec2(1, 1)); // bottom-right

    // Diagonal detection.
    bool rising  = alike(c[2], c[1]); // / diagonal
    bool falling = alike(c[0], c[3]); // \ diagonal

    if (rising && falling) {
        if (alike(c[0], c[2])) {
            // All four similar — flat fill
            frag_color = c[0];
            return;
        }
        // Both diagonals — 4x4 neighborhood vote to break tie
        int bias = 0;
        for (int row = -1; row < 3; row++) {
            for (int col = -1; col < 3; col++) {
                vec4 n = fetch(src_i + ivec2(col, row));
                if (alike(n, c[0])) bias++;
                if (alike(n, c[2])) bias--;
            }
        }
        if (bias < 0) {
            frag_color = pick_falling(c, f.x, f.y);
        } else if (bias > 0) {
            frag_color = pick_rising(c, f.x, f.y);
        } else {
            // Tie — average both
            frag_color = blend(pick_falling(c, f.x, f.y), pick_rising(c, f.x, f.y), 0.5);
        }
    } else if (rising) {
        frag_color = pick_rising(c, f.x, f.y);
    } else if (falling) {
        frag_color = pick_falling(c, f.x, f.y);
    } else {
        // No diagonal — bilinear, then quantize to nearest source color
        vec4 top = blend(c[0], c[1], f.x);
        vec4 bot = blend(c[2], c[3], f.x);
        vec4 mixed = blend(top, bot, f.y);

        float d0 = channel_dist(mixed, c[0]);
        float d1 = channel_dist(mixed, c[1]);
        float d2 = channel_dist(mixed, c[2]);
        float d3 = channel_dist(mixed, c[3]);
        float min_d = min(min(d0, d1), min(d2, d3));

        if (d0 == min_d) frag_color = c[0];
        else if (d1 == min_d) frag_color = c[1];
        else if (d2 == min_d) frag_color = c[2];
        else frag_color = c[3];
    }
}
