// OmniScale -- resolution-independent pixel-art upscaler (compute shader).
// Algorithm by Lior Halphon.

import modules.color;
import modules.pixel_source;

struct Uniforms {
    uint src_w;
    uint src_h;
    uint out_w;
    uint out_h;
    float pixel_size; // sqrt((src_w/out_w)^2 + (src_h/out_h)^2)
    float _pad0;
    float _pad1;
    float _pad2;
};
[[vk::binding(0, 2)]]
#ifdef DXIL_TARGET
ConstantBuffer<Uniforms> u : register(b0, space2);
#else
ConstantBuffer<Uniforms> u;
#endif

[[vk::binding(0, 0)]]
#ifdef DXIL_TARGET
StructuredBuffer<uint> pixels : register(t0, space0);
#else
StructuredBuffer<uint> pixels;
#endif
[[vk::image_format("rgba8")]]
[[vk::binding(0, 1)]]
#ifdef DXIL_TARGET
WTexture2D<float4> output_img : register(u0, space1);
#else
WTexture2D<float4> output_img;
#endif

float4 fetch(int2 coord) { return px_rgba(pixels, int(u.src_w), int(u.src_h), coord); }

bool colors_differ(float4 a, float4 b) {
    float3 d = (a.rgb - b.rgb) * 255.0;
    return abs(2.0 * (d.r + d.g + d.b)) > 36.0
        || abs(2.0 * (d.r - d.b)) > 4.0
        || abs(-d.r + 2.0 * d.g - d.b) > 10.0;
}

float4 blend_colors(float4 a, float4 b, float t) {
    return lerp(a, b, clamp(t, 0.0, 1.0));
}

float4 blend3(float4 a, float4 b, float4 c, float wa, float wb, float wc) {
    return float4(clamp(a.rgb * wa + b.rgb * wb + c.rgb * wc, 0.0, 1.0), 1.0);
}

float4 corner_blend(float4 w0, float4 w1, float4 w3, float px, float py) {
    if (colors_differ(w0, w1) || colors_differ(w0, w3))
        return blend_colors(w1, w3, py - px + 0.5);
    else
        return blend_colors(blend_colors(blend3(w1, w0, w3, 0.375, 0.25, 0.375), w3, py * 2.0), w1, px * 2.0);
}

float aa_band(float d, float hw) {
    return (d + hw) / (2.0 * hw);
}

static const int NUM_RULES = 55;
static int RULE_MASK[55] = {
    191,219,219,239, 11,254,254,
    111,91,191,223,159,207,239,
    63,251,187,127,175,235, 11,11, 47,
    191,219,219,239, 191,126,126,239,
    27,79,139,107, 75,139,31,59,
    251,111,63,251,223,223,
    79,159,47,190,238,126,235,59,
    11,11
};
static int RULE_VAL[55] = {
    55,19,73,109, 11,74,26,
    42,10,58,90,138,138,78,
    14,90,138,90,138,138, 8,2, 47,
    55,19,73,109, 143,14,42,171,
    3,67,131,67, 9,137,25,25,
    106,110,62,250,222,30,
    75,27,11,10,10,10,75,27,
    1,0
};
static int RULE_COND[55] = {
    1,1,2,2, 3,3,3,
    3,3,3,3,3,3,3,
    3,3,3,3,3,3, 0,0, 0,
    0,0,0,0, 0,0,0,0,
    0,0,0,0, 0,0,0,0,
    0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0
};
static int RULE_ACT[55] = {
    0,0,1,1, 2,2,2,
    3,3,3,3,3,3,3,
    3,3,3,3,3,3, 4,5, 6,
    7,7,8,8, 9,9,10,10,
    0,0,0,0, 1,1,1,1,
    11,11,11,11,11,11,
    12,12,12,12,12,12,12,12,
    13,14
};

// Shared neighborhood for exec_action
static float4 g_w[9];

float4 exec_action(uint action, float px, float py, float ps) {
    float4 w0 = g_w[0], w1 = g_w[1], w3 = g_w[3], w4 = g_w[4];

    switch (action) {
    case 0u: return blend_colors(w4, w3, 0.5 - px);
    case 1u: return blend_colors(w4, w1, 0.5 - py);
    case 2u: return w4;
    case 3u: return blend_colors(w4, blend_colors(w4, w0, 0.5 - px), 0.5 - py);
    case 4u: {
        float4 tip = blend3(w0, w1, w4, 0.375, 0.25, 0.375);
        float4 edge = blend_colors(w4, w1, 0.5);
        return blend_colors(blend_colors(tip, edge, px * 2.0), w4, py * 2.0);
    }
    case 5u: {
        float4 tip = blend3(w0, w3, w4, 0.375, 0.25, 0.375);
        float4 edge = blend_colors(w4, w3, 0.5);
        return blend_colors(blend_colors(tip, edge, py * 2.0), w4, px * 2.0);
    }
    case 6u: {
        float dx = 0.5 - px, dy = 0.5 - py;
        float d2 = dx * dx + dy * dy;
        float inner = 0.5 - ps / 2.0;
        if (d2 < inner * inner) return w4;
        float4 r = corner_blend(w0, w1, w3, px, py);
        float outer = 0.5 + ps / 2.0;
        if (d2 > outer * outer) return r;
        return blend_colors(w4, r, (sqrt(d2) - 0.5 + ps / 2.0) / ps);
    }
    case 7u: {
        float d = px - 2.0 * py;
        float hw = ps * 2.236 / 2.0;
        if (d > hw) return w1;
        float4 base = blend_colors(w3, w4, px + 0.5);
        if (d < -hw) return base;
        return blend_colors(base, w1, aa_band(d, hw));
    }
    case 8u: {
        float d = py - 2.0 * px;
        float hw = ps * 2.236 / 2.0;
        if (d > hw) return w3;
        float4 base = blend_colors(w1, w4, px + 0.5);
        if (d < -hw) return base;
        return blend_colors(base, w3, aa_band(d, hw));
    }
    case 9u: {
        float d = px + 2.0 * py;
        float hw = ps * 2.236 / 2.0;
        if (d > 1.0 + hw) return w4;
        float4 r = corner_blend(w0, w1, w3, px, py);
        if (d < 1.0 - hw) return r;
        return blend_colors(r, w4, aa_band(d - 1.0, hw));
    }
    case 10u: {
        float d = py + 2.0 * px;
        float hw = ps * 2.236 / 2.0;
        if (d > 1.0 + hw) return w4;
        float4 r = corner_blend(w0, w1, w3, px, py);
        if (d < 1.0 - hw) return r;
        return blend_colors(r, w4, aa_band(d - 1.0, hw));
    }
    case 11u: return blend_colors(w4, w0, (1.0 - px - py) / 2.0);
    case 12u: {
        float d = px + py;
        float hw = ps / 2.0;
        if (d > 0.5 + hw) return w4;
        float4 r = corner_blend(w0, w1, w3, px, py);
        if (d < 0.5 - hw) return r;
        return blend_colors(r, w4, aa_band(d - 0.5, hw));
    }
    case 13u:
        return blend_colors(
            blend_colors(w4, w3, 0.5 - px),
            blend_colors(w1, blend_colors(w1, w3, 0.5), 0.5 - px),
            0.5 - py);
    case 14u:
        return blend_colors(
            blend_colors(w4, w3, 0.5 - px),
            blend_colors(w1, w0, 0.5 - px),
            0.5 - py);
    default: return w4;
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint opx = tid.x;
    uint opy = tid.y;
    if (opx >= u.out_w || opy >= u.out_h) return;

    float2 src_size = float2(float(u.src_w), float(u.src_h));
    float2 uv = (float2(opx, opy) + 0.5) / float2(float(u.out_w), float(u.out_h));
    float2 src_coord = uv * src_size;
    int2 src_i = clamp(int2(floor(src_coord)), int2(0), int2(int(u.src_w) - 1, int(u.src_h) - 1));
    float2 frac_xy = src_coord - float2(src_i);

    float4 center = fetch(src_i);
    bool is_uniform = true;
    for (int dy = -1; dy <= 1 && is_uniform; dy++) {
        for (int dx = -1; dx <= 1 && is_uniform; dx++) {
            if (dx == 0 && dy == 0) continue;
            if (any(abs(center.rgb - fetch(src_i + int2(dx, dy)).rgb) > float3(0.001)))
                is_uniform = false;
        }
    }
    if (is_uniform) {
        output_img.Store(int2(opx, opy), float4(center.rgb, 1.0));
        return;
    }

    float px = (frac_xy.x > 0.5) ? 1.0 - frac_xy.x : frac_xy.x;
    float py = (frac_xy.y > 0.5) ? 1.0 - frac_xy.y : frac_xy.y;
    int ox = (frac_xy.x > 0.5) ? -1 : 1;
    int oy = (frac_xy.y > 0.5) ? -1 : 1;
    int2 orient = int2(ox, oy);

    g_w[0] = fetch(src_i + int2(-orient.x, -orient.y));
    g_w[1] = fetch(src_i + int2(0, -orient.y));
    g_w[2] = fetch(src_i + int2(orient.x, -orient.y));
    g_w[3] = fetch(src_i + int2(-orient.x, 0));
    g_w[4] = center;
    g_w[5] = fetch(src_i + int2(orient.x, 0));
    g_w[6] = fetch(src_i + int2(-orient.x, orient.y));
    g_w[7] = fetch(src_i + int2(0, orient.y));
    g_w[8] = fetch(src_i + int2(orient.x, orient.y));

    uint pat = 0u;
    if (colors_differ(g_w[0], center)) pat |= 1u;
    if (colors_differ(g_w[1], center)) pat |= 2u;
    if (colors_differ(g_w[2], center)) pat |= 4u;
    if (colors_differ(g_w[3], center)) pat |= 8u;
    if (colors_differ(g_w[5], center)) pat |= 16u;
    if (colors_differ(g_w[6], center)) pat |= 32u;
    if (colors_differ(g_w[7], center)) pat |= 64u;
    if (colors_differ(g_w[8], center)) pat |= 128u;

    for (int i = 0; i < NUM_RULES; i++) {
        if ((pat & uint(RULE_MASK[i])) != uint(RULE_VAL[i])) continue;

        bool cond_ok;
        switch (RULE_COND[i]) {
        case 0: cond_ok = true; break;
        case 1: cond_ok = colors_differ(g_w[1], g_w[5]); break;
        case 2: cond_ok = colors_differ(g_w[7], g_w[3]); break;
        case 3: cond_ok = colors_differ(g_w[3], g_w[1]); break;
        default: cond_ok = false; break;
        }
        if (!cond_ok) continue;

        output_img.Store(int2(opx, opy), float4(exec_action(uint(RULE_ACT[i]), px, py, u.pixel_size).rgb, 1.0));
        return;
    }

    float dist = px + py;
    if (dist > 0.5 + u.pixel_size / 2.0) {
        output_img.Store(int2(opx, opy), float4(center.rgb, 1.0));
        return;
    }

    float4 extras[7];
    extras[0] = fetch(src_i - orient * int2(2, 2));
    extras[1] = fetch(src_i - orient * int2(1, 2));
    extras[2] = fetch(src_i - int2(0, orient.y * 2));
    extras[3] = fetch(src_i + int2(orient.x, -orient.y * 2));
    extras[4] = fetch(src_i - int2(orient.x * 2, orient.y));
    extras[5] = fetch(src_i - int2(orient.x * 2, 0));
    extras[6] = fetch(src_i + int2(-orient.x * 2, orient.y));

    uint ext = pat;
    if (colors_differ(extras[0], center)) ext |= (1u << 8u);
    if (colors_differ(extras[1], center)) ext |= (1u << 9u);
    if (colors_differ(extras[2], center)) ext |= (1u << 10u);
    if (colors_differ(extras[3], center)) ext |= (1u << 11u);
    if (colors_differ(extras[4], center)) ext |= (1u << 12u);
    if (colors_differ(extras[5], center)) ext |= (1u << 13u);
    if (colors_differ(extras[6], center)) ext |= (1u << 14u);

    // Manual popcount
    uint bc = ext;
    bc = bc - ((bc >> 1u) & 0x55555555u);
    bc = (bc & 0x33333333u) + ((bc >> 2u) & 0x33333333u);
    bc = (bc + (bc >> 4u)) & 0x0F0F0F0Fu;
    uint pop = (bc * 0x01010101u) >> 24u;
    if (pop <= 7u) {
        float4 edge = blend_colors(g_w[1], g_w[3], py - px + 0.5);
        if (dist < 0.5 - u.pixel_size / 2.0)
            output_img.Store(int2(opx, opy), float4(edge.rgb, 1.0));
        else
            output_img.Store(int2(opx, opy), float4(blend_colors(edge, center, aa_band(dist - 0.5, u.pixel_size / 2.0)).rgb, 1.0));
        return;
    }

    output_img.Store(int2(opx, opy), float4(center.rgb, 1.0));
}
