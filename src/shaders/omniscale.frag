#version 450

// OmniScale — resolution-independent pixel-art upscaler (GPU fragment shader).
//
// Algorithm designed by Lior Halphon.
//
// Pattern-based edge interpolation for arbitrary output resolutions.
// Classifies the 3×3 source neighborhood into an 8-bit edge descriptor,
// then walks a prioritized rule set to produce anti-aliased output.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

// SDL3 SPIRV: fragment sampled textures at set 2
layout(set = 2, binding = 0) uniform sampler2D src_tex;

// SDL3 SPIRV: fragment uniform buffers at set 3
layout(std140, set = 3, binding = 0) uniform Uniforms {
    vec2 src_size;    // source texture dimensions (e.g. 160, 144)
    vec2 dst_size;    // output dimensions
    float pixel_size; // sqrt((src_w/dst_w)^2 + (src_h/dst_h)^2)
    float _pad0;
    float _pad1;
    float _pad2;
};

// ── Color distance ─────────────────────────────────────────────────────
//
// YCbCr-like perceptual test (integer-domain, all values scaled ×8):
//   Y·8  = 2·(ΔR + ΔG + ΔB)    threshold 36
//   Cb·8 = 2·(ΔR − ΔB)         threshold  4
//   Cr·8 = −ΔR + 2·ΔG − ΔB     threshold 10

bool colors_differ(vec4 a, vec4 b) {
    vec3 d = (a.rgb - b.rgb) * 255.0;
    return abs(2.0 * (d.r + d.g + d.b)) > 36.0
        || abs(2.0 * (d.r - d.b)) > 4.0
        || abs(-d.r + 2.0 * d.g - d.b) > 10.0;
}

// ── Blending helpers ───────────────────────────────────────────────────

vec4 blend(vec4 a, vec4 b, float t) {
    return mix(a, b, clamp(t, 0.0, 1.0));
}

vec4 blend3(vec4 a, vec4 b, vec4 c, float wa, float wb, float wc) {
    return vec4(clamp(a.rgb * wa + b.rgb * wb + c.rgb * wc, 0.0, 1.0), 1.0);
}

// ── Texel fetch with clamping ──────────────────────────────────────────

vec4 fetch(ivec2 coord, ivec2 bounds) {
    return texelFetch(src_tex, clamp(coord, ivec2(0), bounds - 1), 0);
}

// ── Corner blend ───────────────────────────────────────────────────────
//
// When the corner pixel differs from both cardinals, interpolate along
// the diagonal; otherwise do a smooth three-way blend.

vec4 corner_blend(vec4 w0, vec4 w1, vec4 w3, float px, float py) {
    if (colors_differ(w0, w1) || colors_differ(w0, w3))
        return blend(w1, w3, py - px + 0.5);
    else
        return blend(blend(blend3(w1, w0, w3, 0.375, 0.25, 0.375), w3, py * 2.0), w1, px * 2.0);
}

// ── Anti-alias band factor ─────────────────────────────────────────────
//
// Smooth transition: d = signed distance from boundary, hw = half-width.
// Returns 0 at -hw, 1 at +hw.

float aa_band(float d, float hw) {
    return (d + hw) / (2.0 * hw);
}

// ── Rule table constants ───────────────────────────────────────────────
//
// Each rule is encoded as a uvec4:
//   x = mask, y = value, z = condition type, w = action type
//
// Condition types:  0=none, 1=w1≠w5, 2=w7≠w3, 3=w3≠w1
// Action types:     0=SlideX, 1=SlideY, 2=Solid, 3=GentleCorner,
//                   4=PartialCornerTop, 5=PartialCornerLeft,
//                   6=CircularDiag, 7=SteepH, 8=SteepV,
//                   9=ReverseH, 10=ReverseV, 11=LinearCorner,
//                   12=AaDiag, 13=BilinearMid, 14=BilinearFull

const int NUM_RULES = 55;
// Rule table packed as int arrays (mask, value, condition, action).
// Values stored as decimal to avoid GLSL hex literal type issues.
int RULE_MASK[55] = int[](
    191,219,219,239,
    11,254,254,
    111,91,191,223,159,207,239,
    63,251,187,127,175,235,
    11,11,
    47,
    191,219,219,239,
    191,126,126,239,
    27,79,139,107,
    75,139,31,59,
    251,111,63,251,223,223,
    79,159,47,190,238,126,235,59,
    11,11
);
int RULE_VAL[55] = int[](
    55,19,73,109,
    11,74,26,
    42,10,58,90,138,138,78,
    14,90,138,90,138,138,
    8,2,
    47,
    55,19,73,109,
    143,14,42,171,
    3,67,131,67,
    9,137,25,25,
    106,110,62,250,222,30,
    75,27,11,10,10,10,75,27,
    1,0
);
int RULE_COND[55] = int[](
    1,1,2,2,
    3,3,3,
    3,3,3,3,3,3,3,
    3,3,3,3,3,3,
    0,0,
    0,
    0,0,0,0,
    0,0,0,0,
    0,0,0,0,
    0,0,0,0,
    0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0
);
int RULE_ACT[55] = int[](
    0,0,1,1,
    2,2,2,
    3,3,3,3,3,3,3,
    3,3,3,3,3,3,
    4,5,
    6,
    7,7,8,8,
    9,9,10,10,
    0,0,0,0,
    1,1,1,1,
    11,11,11,11,11,11,
    12,12,12,12,12,12,12,12,
    13,14
);

// ── Execute a matched action ───────────────────────────────────────────

vec4 exec_action(
    uint action, vec4 w[9],
    float px, float py, float ps
) {
    vec4 w0 = w[0], w1 = w[1], w3 = w[3], w4 = w[4];

    switch (action) {
    case 0u: // SlideX
        return blend(w4, w3, 0.5 - px);

    case 1u: // SlideY
        return blend(w4, w1, 0.5 - py);

    case 2u: // Solid
        return w4;

    case 3u: // GentleCorner
        return blend(w4, blend(w4, w0, 0.5 - px), 0.5 - py);

    case 4u: { // PartialCornerTop
        vec4 tip = blend3(w0, w1, w4, 0.375, 0.25, 0.375);
        vec4 edge = blend(w4, w1, 0.5);
        return blend(blend(tip, edge, px * 2.0), w4, py * 2.0);
    }
    case 5u: { // PartialCornerLeft
        vec4 tip = blend3(w0, w3, w4, 0.375, 0.25, 0.375);
        vec4 edge = blend(w4, w3, 0.5);
        return blend(blend(tip, edge, py * 2.0), w4, px * 2.0);
    }

    case 6u: { // CircularDiag
        float dx = 0.5 - px, dy = 0.5 - py;
        float d2 = dx * dx + dy * dy;
        float inner = 0.5 - ps / 2.0;
        if (d2 < inner * inner) return w4;
        vec4 r = corner_blend(w0, w1, w3, px, py);
        float outer = 0.5 + ps / 2.0;
        if (d2 > outer * outer) return r;
        return blend(w4, r, (sqrt(d2) - 0.5 + ps / 2.0) / ps);
    }

    case 7u: { // SteepH
        float d = px - 2.0 * py;
        float hw = ps * 2.236 / 2.0;
        if (d > hw) return w1;
        vec4 base = blend(w3, w4, px + 0.5);
        if (d < -hw) return base;
        return blend(base, w1, aa_band(d, hw));
    }
    case 8u: { // SteepV
        float d = py - 2.0 * px;
        float hw = ps * 2.236 / 2.0;
        if (d > hw) return w3;
        vec4 base = blend(w1, w4, px + 0.5);
        if (d < -hw) return base;
        return blend(base, w3, aa_band(d, hw));
    }

    case 9u: { // ReverseH
        float d = px + 2.0 * py;
        float hw = ps * 2.236 / 2.0;
        if (d > 1.0 + hw) return w4;
        vec4 r = corner_blend(w0, w1, w3, px, py);
        if (d < 1.0 - hw) return r;
        return blend(r, w4, aa_band(d - 1.0, hw));
    }
    case 10u: { // ReverseV
        float d = py + 2.0 * px;
        float hw = ps * 2.236 / 2.0;
        if (d > 1.0 + hw) return w4;
        vec4 r = corner_blend(w0, w1, w3, px, py);
        if (d < 1.0 - hw) return r;
        return blend(r, w4, aa_band(d - 1.0, hw));
    }

    case 11u: // LinearCorner
        return blend(w4, w0, (1.0 - px - py) / 2.0);

    case 12u: { // AaDiag
        float d = px + py;
        float hw = ps / 2.0;
        if (d > 0.5 + hw) return w4;
        vec4 r = corner_blend(w0, w1, w3, px, py);
        if (d < 0.5 - hw) return r;
        return blend(r, w4, aa_band(d - 0.5, hw));
    }

    case 13u: // BilinearMid
        return blend(
            blend(w4, w3, 0.5 - px),
            blend(w1, blend(w1, w3, 0.5), 0.5 - px),
            0.5 - py);

    case 14u: // BilinearFull
        return blend(
            blend(w4, w3, 0.5 - px),
            blend(w1, w0, 0.5 - px),
            0.5 - py);

    default:
        return w4;
    }
}

// ── Main ───────────────────────────────────────────────────────────────

void main() {
    ivec2 src_bounds = ivec2(src_size);

    // Map output pixel center to source coordinates
    vec2 src_coord = v_uv * src_size;
    ivec2 src_i = clamp(ivec2(floor(src_coord)), ivec2(0), src_bounds - 1);
    vec2 frac_xy = src_coord - vec2(src_i);

    // Uniform fast path: all 8 neighbors identical to center
    vec4 center = fetch(src_i, src_bounds);
    bool is_uniform = true;
    for (int dy = -1; dy <= 1 && is_uniform; dy++) {
        for (int dx = -1; dx <= 1 && is_uniform; dx++) {
            if (dx == 0 && dy == 0) continue;
            if (any(greaterThan(abs(center.rgb - fetch(src_i + ivec2(dx, dy), src_bounds).rgb), vec3(0.001))))
                is_uniform = false;
        }
    }
    if (is_uniform) {
        frag_color = vec4(center.rgb, 1.0);
        return;
    }

    // Reflect sub-pixel offset into top-left quadrant
    ivec2 orient;
    float px, py;
    if (frac_xy.x > 0.5) { px = 1.0 - frac_xy.x; orient.x = -1; }
    else                  { px = frac_xy.x;         orient.x = 1; }
    if (frac_xy.y > 0.5) { py = 1.0 - frac_xy.y; orient.y = -1; }
    else                  { py = frac_xy.y;         orient.y = 1; }

    // Sample oriented 3×3 neighborhood
    vec4 w[9];
    w[0] = fetch(src_i + ivec2(-orient.x, -orient.y), src_bounds);
    w[1] = fetch(src_i + ivec2(0, -orient.y), src_bounds);
    w[2] = fetch(src_i + ivec2(orient.x, -orient.y), src_bounds);
    w[3] = fetch(src_i + ivec2(-orient.x, 0), src_bounds);
    w[4] = center;
    w[5] = fetch(src_i + ivec2(orient.x, 0), src_bounds);
    w[6] = fetch(src_i + ivec2(-orient.x, orient.y), src_bounds);
    w[7] = fetch(src_i + ivec2(0, orient.y), src_bounds);
    w[8] = fetch(src_i + ivec2(orient.x, orient.y), src_bounds);

    // Build 8-bit difference pattern
    uint pat = 0u;
    if (colors_differ(w[0], center)) pat |= 1u;
    if (colors_differ(w[1], center)) pat |= 2u;
    if (colors_differ(w[2], center)) pat |= 4u;
    if (colors_differ(w[3], center)) pat |= 8u;
    if (colors_differ(w[5], center)) pat |= 16u;
    if (colors_differ(w[6], center)) pat |= 32u;
    if (colors_differ(w[7], center)) pat |= 64u;
    if (colors_differ(w[8], center)) pat |= 128u;

    // Walk rule table — first match wins
    for (int i = 0; i < NUM_RULES; i++) {
        if ((pat & uint(RULE_MASK[i])) != uint(RULE_VAL[i])) continue;

        // Check secondary condition
        bool cond_ok;
        switch (RULE_COND[i]) {
        case 0: cond_ok = true; break;
        case 1: cond_ok = colors_differ(w[1], w[5]); break;
        case 2: cond_ok = colors_differ(w[7], w[3]); break;
        case 3: cond_ok = colors_differ(w[3], w[1]); break;
        default: cond_ok = false; break;
        }
        if (!cond_ok) continue;

        frag_color = vec4(exec_action(uint(RULE_ACT[i]), w, px, py, pixel_size).rgb, 1.0);
        return;
    }

    // Extended sampling fallback
    float dist = px + py;
    if (dist > 0.5 + pixel_size / 2.0) {
        frag_color = vec4(center.rgb, 1.0);
        return;
    }

    // 7 samples in the L-shaped region beyond the 3×3 window
    vec4 extras[7];
    extras[0] = fetch(src_i - orient * ivec2(2, 2), src_bounds);
    extras[1] = fetch(src_i - orient * ivec2(1, 2), src_bounds);
    extras[2] = fetch(src_i - ivec2(0, orient.y * 2), src_bounds);
    extras[3] = fetch(src_i + ivec2(orient.x, -orient.y * 2), src_bounds);
    extras[4] = fetch(src_i - ivec2(orient.x * 2, orient.y), src_bounds);
    extras[5] = fetch(src_i - ivec2(orient.x * 2, 0), src_bounds);
    extras[6] = fetch(src_i + ivec2(-orient.x * 2, orient.y), src_bounds);

    uint ext = pat;
    for (int i = 0; i < 7; i++) {
        if (colors_differ(extras[i], center))
            ext |= (1u << (8 + i));
    }

    // Majority vote: if most extended samples match center, it's a diagonal
    if (int(bitCount(ext)) <= 7) {
        vec4 edge = blend(w[1], w[3], py - px + 0.5);
        if (dist < 0.5 - pixel_size / 2.0)
            frag_color = vec4(edge.rgb, 1.0);
        else
            frag_color = vec4(blend(edge, center, aa_band(dist - 0.5, pixel_size / 2.0)).rgb, 1.0);
        return;
    }

    frag_color = vec4(center.rgb, 1.0);
}
