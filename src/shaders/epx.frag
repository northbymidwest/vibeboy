#version 450

// EPX / Scale2x / Scale4x — pixel-art scaling by Eric Johnston.
//
// Scale2x: each pixel becomes 2x2 with corners smoothed by cardinal matching.
// Scale4x: EPX applied twice (single-pass, second pass computed inline).

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D src_tex;

layout(std140, set = 3, binding = 0) uniform Uniforms {
    vec2 src_size;
    float scale;  // 2.0 or 4.0
    float pad0;
};

vec4 fetch(ivec2 coord) {
    coord = clamp(coord, ivec2(0), ivec2(src_size) - 1);
    return texelFetch(src_tex, coord, 0);
}

// EPX rule: given center P and cardinal neighbors A(up), B(right), C(left), D(down),
// compute corner at sub-pixel (sx, sy) where sx,sy ∈ {0,1}.
vec4 epx(vec4 p, vec4 a, vec4 b, vec4 c, vec4 d, int sx, int sy) {
    //  TL(0,0) = (C==A && C!=D && A!=B) ? A : P
    //  TR(1,0) = (A==B && A!=C && B!=D) ? B : P
    //  BL(0,1) = (D==C && D!=B && C!=A) ? C : P
    //  BR(1,1) = (B==D && B!=A && D!=C) ? D : P
    if (sx == 0 && sy == 0) return (c==a && c!=d && a!=b) ? a : p;
    if (sx == 1 && sy == 0) return (a==b && a!=c && b!=d) ? b : p;
    if (sx == 0 && sy == 1) return (d==c && d!=b && c!=a) ? c : p;
    return (b==d && b!=a && d!=c) ? d : p;
}

// Compute the EPX 2x result at a given 2x-buffer position.
// pos is in the 2x coordinate space; returns the interpolated color.
vec4 epx_at(ivec2 pos) {
    ivec2 src = pos / 2;
    ivec2 sub = pos - src * 2;  // 0 or 1
    vec4 p = fetch(src);
    vec4 a = fetch(src + ivec2(0, -1));
    vec4 b = fetch(src + ivec2(1, 0));
    vec4 c = fetch(src + ivec2(-1, 0));
    vec4 d = fetch(src + ivec2(0, 1));
    return epx(p, a, b, c, d, sub.x, sub.y);
}

void main() {
    int iscale = int(scale);
    vec2 src_coord = v_uv * src_size;
    ivec2 src_i = clamp(ivec2(floor(src_coord)), ivec2(0), ivec2(src_size) - 1);
    vec2 frac_val = src_coord - vec2(src_i);
    ivec2 sub = clamp(ivec2(floor(frac_val * scale)), ivec2(0), ivec2(iscale - 1));

    if (iscale == 2) {
        // Scale2x: direct EPX
        vec4 p = fetch(src_i);
        vec4 a = fetch(src_i + ivec2(0, -1));
        vec4 b = fetch(src_i + ivec2(1, 0));
        vec4 c = fetch(src_i + ivec2(-1, 0));
        vec4 d = fetch(src_i + ivec2(0, 1));
        frag_color = epx(p, a, b, c, d, sub.x, sub.y);
    } else {
        // Scale4x: EPX applied twice. Each 4x sub-pixel maps to a 2x position,
        // and we compute that 2x pixel + its 4 cardinal neighbors in the 2x
        // buffer to apply EPX again.
        ivec2 mid_pos = src_i * 2 + sub / 2;  // position in 2x buffer
        ivec2 sub4 = sub - (sub / 2) * 2;     // sub-pixel within the second EPX pass

        vec4 p = epx_at(mid_pos);
        vec4 a = epx_at(mid_pos + ivec2(0, -1));
        vec4 b = epx_at(mid_pos + ivec2(1, 0));
        vec4 c = epx_at(mid_pos + ivec2(-1, 0));
        vec4 d = epx_at(mid_pos + ivec2(0, 1));
        frag_color = epx(p, a, b, c, d, sub4.x, sub4.y);
    }
}
