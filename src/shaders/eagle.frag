#version 450

// Eagle — 2x pixel-art scaling.
//
// Checks 3-way diagonal corner matches to decide whether to smooth corners.
// For each pixel P with 8 neighbors S,T,U,V,W,X,Y,Z:
//   TL = (S==T && S==V) ? S : P
//   TR = (T==U && U==W) ? U : P
//   BL = (V==X && X==Y) ? X : P
//   BR = (W==Z && Z==Y) ? Z : P

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
    ivec2 sub = clamp(ivec2(floor(frac_val * 2.0)), ivec2(0), ivec2(1));

    vec4 p = fetch(src_i);
    vec4 s = fetch(src_i + ivec2(-1, -1));
    vec4 t = fetch(src_i + ivec2( 0, -1));
    vec4 u = fetch(src_i + ivec2( 1, -1));
    vec4 v = fetch(src_i + ivec2(-1,  0));
    vec4 w = fetch(src_i + ivec2( 1,  0));
    vec4 x = fetch(src_i + ivec2(-1,  1));
    vec4 y = fetch(src_i + ivec2( 0,  1));
    vec4 z = fetch(src_i + ivec2( 1,  1));

    if      (sub.x == 0 && sub.y == 0) frag_color = (s==t && s==v) ? s : p;
    else if (sub.x == 1 && sub.y == 0) frag_color = (t==u && u==w) ? u : p;
    else if (sub.x == 0 && sub.y == 1) frag_color = (v==x && x==y) ? x : p;
    else                               frag_color = (w==z && z==y) ? z : p;
}
