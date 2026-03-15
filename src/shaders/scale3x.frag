#version 450

// Scale3x — 3x pixel-art scaling.
//
// Extension of Scale2x to 3x3 output blocks. Uses guard condition B!=H && D!=F,
// then diagonal neighbor checks for edge and corner pixels.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D src_tex;

layout(std140, set = 3, binding = 0) uniform Uniforms {
    vec2 src_size;
    float pad0;
    float pad1;
};

vec4 fetch(ivec2 coord) {
    coord = clamp(coord, ivec2(0), ivec2(src_size) - 1);
    return texelFetch(src_tex, coord, 0);
}

void main() {
    vec2 src_coord = v_uv * src_size;
    ivec2 src_i = clamp(ivec2(floor(src_coord)), ivec2(0), ivec2(src_size) - 1);
    vec2 frac_val = src_coord - vec2(src_i);
    ivec2 sub = clamp(ivec2(floor(frac_val * 3.0)), ivec2(0), ivec2(2));

    vec4 a = fetch(src_i + ivec2(-1, -1));
    vec4 b = fetch(src_i + ivec2( 0, -1));
    vec4 c = fetch(src_i + ivec2( 1, -1));
    vec4 d = fetch(src_i + ivec2(-1,  0));
    vec4 e = fetch(src_i);
    vec4 f = fetch(src_i + ivec2( 1,  0));
    vec4 g = fetch(src_i + ivec2(-1,  1));
    vec4 h = fetch(src_i + ivec2( 0,  1));
    vec4 i = fetch(src_i + ivec2( 1,  1));

    vec4 result = e;

    if (b != h && d != f) {
        int idx = sub.y * 3 + sub.x;
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

    frag_color = result;
}
