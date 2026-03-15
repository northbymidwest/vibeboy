#version 450

// Super xBR — edge-directed 2x scaling by Hyllian (single-pass GPU version).
//
// Folds the three-stage pipeline (diagonal → cardinal → polish) into a
// single fragment shader. Each pixel lazily computes any intermediate
// values it needs from the source texture.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D src_tex;

layout(std140, set = 3, binding = 0) uniform Uniforms {
    vec2 src_size;
    float pad0;
    float pad1;
};

// ── Constants ───────────────────────────────────────────────────────────────

// Gradient profile weights per stage.
// [adjacent, parallel, cross_near, core, cross_far, outer]
const float DIAG_WP[6]    = float[](2.0, 1.0, -1.0, 4.0, -1.0, 1.0);
const float CARD_WP[6]    = float[](1.0, 0.0,  0.0, 0.0,  0.0, 0.0);
const float POLISH_WP[6]  = float[](0.0, 0.0,  0.0, 1.0,  0.0, 0.0);

// Cubic filter sharpness parameters (negative lobe magnitude).
const float DIAG_SHARP  = 1.29633 / 10.0;
const float ORTHO_SHARP = 1.75068 / 20.0;

// Smoothstep threshold for diagonal/cardinal blending.
const float EDGE_LIMIT = 5.000001;

// ── Helpers ─────────────────────────────────────────────────────────────────

vec4 src_fetch(ivec2 coord) {
    coord = clamp(coord, ivec2(0), ivec2(src_size) - 1);
    return texelFetch(src_tex, coord, 0);
}

float luma(vec4 c) {
    return dot(c.rgb * 255.0, vec3(0.2126, 0.7152, 0.0722));
}

float ld(float a, float b) { return abs(a - b); }

float smooth_threshold(float x, float limit) {
    float t = clamp(x / limit, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// ── Gradient analysis ───────────────────────────────────────────────────────

// 14-sample diagonal gradient strength.
float diag_grad(float wp[6],
    float outer_a, float outer_b,
    float axis_l, float axis_c, float axis_r,
    float core_a, float core_b, float far_a, float far_b,
    float par_a, float par_b, float par_c,
    float cross_a, float cross_b
) {
    return wp[0] * (ld(axis_c, axis_r) + ld(axis_c, axis_l)
                  + ld(par_b, par_a) + ld(par_b, par_c))
         + wp[1] * (ld(far_a, far_b) + ld(core_a, core_b))
         + wp[2] * (ld(core_b, far_b) + ld(core_a, far_a))
         + wp[3] * ld(core_b, far_a)
         + wp[4] * (ld(axis_l, axis_r) + ld(par_a, par_c))
         + wp[5] * (ld(outer_a, outer_b) + ld(cross_a, cross_b));
}

// 8-sample cardinal gradient strength.
float card_grad(float wp[6], float i[4], float e[4]) {
    return wp[3] * (ld(i[0], i[1]) + ld(i[2], i[3]))
         + wp[0] * (ld(i[0], e[0]) + ld(i[1], e[1]) + ld(i[2], e[2]) + ld(i[3], e[3]))
         + wp[2] * (ld(i[0], e[1]) + ld(i[2], e[3]) + ld(e[0], i[1]) + ld(e[2], i[3]));
}

// ── Core interpolation ─────────────────────────────────────────────────────

// 4-tap cubic filter on 4 colors.
vec3 cubic4(float sharp, vec4 c0, vec4 c1, vec4 c2, vec4 c3) {
    float t[4] = float[](-sharp, 0.5 + sharp, 0.5 + sharp, -sharp);
    return t[0]*c0.rgb + t[1]*c1.rgb + t[2]*c2.rgb + t[3]*c3.rgb;
}

// 4-tap cubic filter on 4 paired colors (each tap is sum of two pixels).
vec3 cubic4_paired(float sharp, vec4 a0, vec4 b0, vec4 a1, vec4 b1,
                                vec4 a2, vec4 b2, vec4 a3, vec4 b3) {
    float t[4] = float[](-sharp, 0.25 + sharp, 0.25 + sharp, -sharp);
    return t[0]*(a0.rgb+b0.rgb) + t[1]*(a1.rgb+b1.rgb)
         + t[2]*(a2.rgb+b2.rgb) + t[3]*(a3.rgb+b3.rgb);
}

// Full edge-directed interpolation from a 4x4 pixel neighborhood.
// p[0..15] in row-major order:
//   0  1  2  3
//   4  5  6  7
//   8  9  10 11
//   12 13 14 15
vec4 edge_interp(vec4 p[16], float wp[6]) {
    float l[16];
    for (int i = 0; i < 16; i++) l[i] = luma(p[i]);

    // Position aliases
    const int CTL=0, TL=1, TR=2, CTR=3;
    const int LT=4,  ITL=5, ITR=6, RT=7;
    const int LB=8,  IBL=9, IBR=10, RB=11;
    const int CBL=12, BL=13, BR=14, CBR=15;

    // Diagonal gradients
    float grad_bs = diag_grad(wp,
        l[LT], l[TL], l[LB], l[ITL], l[TR],
        l[CBL], l[IBL], l[ITR], l[CTR],
        l[BL], l[IBR], l[RT], l[BR], l[RB]);
    float grad_sl = diag_grad(wp,
        l[TR], l[RT], l[TL], l[ITR], l[RB],
        l[CTL], l[ITL], l[IBR], l[CBR],
        l[LT], l[IBL], l[BR], l[LB], l[BL]);

    float diag_bias = grad_bs - grad_sl;
    float diag_conf = smooth_threshold(abs(diag_bias), EDGE_LIMIT);

    // Cardinal gradients
    float ih[4] = float[](l[ITR], l[IBR], l[ITL], l[IBL]);
    float eh[4] = float[](l[TR], l[BR], l[TL], l[BL]);
    float iv[4] = float[](l[ITL], l[ITR], l[IBL], l[IBR]);
    float ev[4] = float[](l[LT], l[RT], l[LB], l[RB]);
    float grad_h = card_grad(wp, ih, eh);
    float grad_v = card_grad(wp, iv, ev);
    float card_bias = grad_h - grad_v;

    // Candidate colors
    vec3 slash     = cubic4(DIAG_SHARP, p[CBL], p[IBL], p[ITR], p[CTR]);
    vec3 backslash = cubic4(DIAG_SHARP, p[CTL], p[ITL], p[IBR], p[CBR]);
    vec3 horiz = cubic4_paired(ORTHO_SHARP,
        p[LT],p[LB], p[ITL],p[IBL], p[ITR],p[IBR], p[RT],p[RB]);
    vec3 vert = cubic4_paired(ORTHO_SHARP,
        p[TR],p[TL], p[ITR],p[ITL], p[IBR],p[IBL], p[BR],p[BL]);

    vec3 diag_pick = (diag_bias >= 0.0) ? backslash : slash;
    vec3 card_pick = (card_bias >= 0.0) ? vert : horiz;

    vec3 result = card_pick + diag_conf * (diag_pick - card_pick);

    // Anti-ringing: clamp to inner 2x2 range
    vec3 lo = min(min(p[ITL].rgb, p[ITR].rgb), min(p[IBL].rgb, p[IBR].rgb));
    vec3 hi = max(max(p[ITL].rgb, p[ITR].rgb), max(p[IBL].rgb, p[IBR].rgb));
    result = clamp(result, lo, hi);

    return vec4(result, 1.0);
}

// ── Lazy 2x buffer sampling ────────────────────────────────────────────────

// Compute the diagonal-stage value at 2x position (ox, oy) where ox,oy are odd.
vec4 compute_diagonal(ivec2 pos) {
    ivec2 src = pos / 2; // source pixel at top-left of the 2x2 quad
    vec4 p[16];
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++)
            p[j*4+i] = src_fetch(src + ivec2(i-1, j-1));
    return edge_interp(p, DIAG_WP);
}

// Sample from the "after-diagonal" buffer: source at (even,even), diagonal at (odd,odd).
vec4 sample_diag_buf(ivec2 pos, ivec2 out_size) {
    pos = clamp(pos, ivec2(0), out_size - 1);
    int px = pos.x & 1, py = pos.y & 1;
    if (px == 0 && py == 0) return src_fetch(pos / 2);
    if (px == 1 && py == 1) return compute_diagonal(pos);
    // Shouldn't reach here from rotated cardinal grid, but fallback:
    return src_fetch(pos / 2);
}

// Compute the cardinal-stage value at 2x position (ox, oy).
// Horizontal midpoint: ox odd, oy even.
// Vertical midpoint: ox even, oy odd.
vec4 compute_cardinal(ivec2 pos, ivec2 out_size) {
    int ox = pos.x, oy = pos.y;
    vec4 p[16];

    if ((ox & 1) == 1 && (oy & 1) == 0) {
        // Horizontal midpoint — rotated 45° grid
        ivec2 offsets[16] = ivec2[](
            ivec2(ox-3,oy), ivec2(ox-2,oy-1), ivec2(ox-1,oy-2), ivec2(ox,oy-3),
            ivec2(ox-2,oy+1), ivec2(ox-1,oy), ivec2(ox,oy-1), ivec2(ox+1,oy-2),
            ivec2(ox-1,oy+2), ivec2(ox,oy+1), ivec2(ox+1,oy), ivec2(ox+2,oy-1),
            ivec2(ox,oy+3), ivec2(ox+1,oy+2), ivec2(ox+2,oy+1), ivec2(ox+3,oy)
        );
        for (int i = 0; i < 16; i++)
            p[i] = sample_diag_buf(offsets[i], out_size);
    } else {
        // Vertical midpoint — rotated 45° grid (transposed)
        ivec2 offsets[16] = ivec2[](
            ivec2(ox,oy-3), ivec2(ox-1,oy-2), ivec2(ox-2,oy-1), ivec2(ox-3,oy),
            ivec2(ox+1,oy-2), ivec2(ox,oy-1), ivec2(ox-1,oy), ivec2(ox-2,oy+1),
            ivec2(ox+2,oy-1), ivec2(ox+1,oy), ivec2(ox,oy+1), ivec2(ox-1,oy+2),
            ivec2(ox+3,oy), ivec2(ox+2,oy+1), ivec2(ox+1,oy+2), ivec2(ox,oy+3)
        );
        for (int i = 0; i < 16; i++)
            p[i] = sample_diag_buf(offsets[i], out_size);
    }
    return edge_interp(p, CARD_WP);
}

// Sample from the "after-cardinal" buffer (pre-polish): handles all 4 position types.
vec4 sample_full_buf(ivec2 pos, ivec2 out_size) {
    pos = clamp(pos, ivec2(0), out_size - 1);
    int px = pos.x & 1, py = pos.y & 1;
    if (px == 0 && py == 0) return src_fetch(pos / 2);
    if (px == 1 && py == 1) return compute_diagonal(pos);
    return compute_cardinal(pos, out_size);
}

// ── Main ────────────────────────────────────────────────────────────────────

void main() {
    ivec2 out_size = ivec2(src_size) * 2;
    vec2 out_coord = v_uv * vec2(out_size);
    ivec2 pos = clamp(ivec2(floor(out_coord)), ivec2(0), out_size - 1);

    // Compute pre-polish value
    vec4 pre_polish = sample_full_buf(pos, out_size);

    // Polish pass: gentle refinement with simplified weights.
    // Sample 4x4 neighborhood from the pre-polish buffer.
    vec4 p[16];
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++)
            p[j*4+i] = sample_full_buf(pos + ivec2(i-1, j-1), out_size);

    frag_color = edge_interp(p, POLISH_WP);
}
