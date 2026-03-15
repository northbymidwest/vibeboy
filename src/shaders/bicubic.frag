#version 450

// Bicubic (Catmull-Rom) interpolation — resolution-independent smooth scaling.
//
// Uses a 4x4 source neighborhood with Catmull-Rom spline weights for each
// output pixel. Produces sharper results than bilinear while remaining smooth.

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

// Catmull-Rom cubic weight function.
vec4 cubic_weight(float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    return vec4(
        -0.5 * t3 + t2 - 0.5 * t,
         1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
         0.5 * t3 - 0.5 * t2
    );
}

void main() {
    vec2 src_coord = v_uv * src_size - 0.5;
    ivec2 src_i = ivec2(floor(src_coord));
    vec2 f = src_coord - vec2(src_i);

    vec4 wx = cubic_weight(f.x);
    vec4 wy = cubic_weight(f.y);

    vec3 color = vec3(0.0);
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            float w = wx[i] * wy[j];
            color += fetch(src_i + ivec2(i - 1, j - 1)).rgb * w;
        }
    }

    frag_color = vec4(clamp(color, 0.0, 1.0), 1.0);
}
