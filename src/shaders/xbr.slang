// xBR -- pixel-art scaling (2x, 3x, 4x) by Hyllian.
// Compute shader variant with resolved CP lookups.
// Color equality comparisons use packed u32 for exact integer matching.

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

uint fetch_raw(int2 coord) { return px_raw(pixels, int(u.src_w), int(u.src_h), coord); }

float4 fetch(int2 coord) {
    return (fetch_raw(coord)).unpack_rgba();
}

float color_dist(float4 a, float4 b) {
    return yuv_dist_sq(a, b);
}

float4 blend_colors(float4 a, float4 b, float alpha) {
    return lerp(a, b, clamp(alpha, 0.0, 1.0));
}

static const float B32  = 32.0 / 256.0;
static const float B64  = 64.0 / 256.0;
static const float B128 = 128.0 / 256.0;
static const float B192 = 192.0 / 256.0;
static const float B224 = 224.0 / 256.0;

// Resolved corner pixels (float4 for blending, uint for equality).
static float4 cF, cH, cI, cC, cG, cB, cD, cF4, cH5, cI4, cI5;
static float4 cE;
static uint uF, uH, uI, uC, uG, uB, uD, uF4, uH5, uI4, uI5;
static uint uE;

void resolve_cp(float4 nb[25], uint nb_raw[25], int cp[11]) {
    cF = nb[cp[0]]; cH = nb[cp[1]]; cI = nb[cp[2]];
    cC = nb[cp[3]]; cG = nb[cp[4]]; cB = nb[cp[5]];
    cD = nb[cp[6]]; cF4 = nb[cp[7]]; cH5 = nb[cp[8]];
    cI4 = nb[cp[9]]; cI5 = nb[cp[10]];
    cE = nb[12];
    uF = nb_raw[cp[0]]; uH = nb_raw[cp[1]]; uI = nb_raw[cp[2]];
    uC = nb_raw[cp[3]]; uG = nb_raw[cp[4]]; uB = nb_raw[cp[5]];
    uD = nb_raw[cp[6]]; uF4 = nb_raw[cp[7]]; uH5 = nb_raw[cp[8]];
    uI4 = nb_raw[cp[9]]; uI5 = nb_raw[cp[10]];
    uE = nb_raw[12];
}

float2 edge_weights() {
    float ew = color_dist(cE, cC) + color_dist(cE, cG)
        + color_dist(cI, cH5) + color_dist(cI, cF4)
        + 4.0 * color_dist(cH, cF);
    float iw = color_dist(cH, cD) + color_dist(cH, cI5)
        + color_dist(cF, cI4) + color_dist(cF, cB)
        + 4.0 * color_dist(cE, cI);
    return float2(ew, iw);
}

float4 pick_color() {
    return color_dist(cE, cF) <= color_dist(cE, cH) ? cF : cH;
}

void detect_direction(out bool is_left, out bool is_up) {
    float ke = color_dist(cF, cG);
    float ki = color_dist(cH, cC);
    is_left = ke * 2.0 <= ki && uE != uG && uD != uG;
    is_up   = ke >= ki * 2.0 && uE != uC && uB != uC;
}

bool sub_cond_24() {
    return (uF != uB && uH != uD)
        || (uE == uI && uF != uI4 && uH != uI5)
        || uE == uG
        || uE == uC;
}

bool sub_cond_3() {
    return (uF != uB && uF != uC)
        || (uH != uD && uH != uG)
        || (uE == uI
            && ((uF != uF4 && uF != uI4)
                || (uH != uH5 && uH != uI5)))
        || uE == uG
        || uE == uC;
}

float4 filt2x(float4 e) {
    if (uE == uH || uE == uF) return e;
    float2 ew_iw = edge_weights();
    if (ew_iw.x > ew_iw.y) return e;
    float4 px = pick_color();
    if (ew_iw.x < ew_iw.y && sub_cond_24()) {
        bool left, up;
        detect_direction(left, up);
        if (left && up) return blend_colors(e, px, B224);
        if (left)       return blend_colors(e, px, B192);
        if (up)         return blend_colors(e, px, B192);
        return blend_colors(e, px, B128);
    }
    return blend_colors(e, px, B128);
}

static float4 g_out_block[16];

void filt3x(float4 e, inout float4 corner, inout float4 side_h, inout float4 side_f) {
    if (uE == uH || uE == uF) return;
    float2 ew_iw = edge_weights();
    if (ew_iw.x > ew_iw.y) return;
    float4 px = pick_color();
    if (ew_iw.x < ew_iw.y && sub_cond_3()) {
        bool left, up;
        detect_direction(left, up);
        if (left && up) {
            float4 bh = blend_colors(side_h, px, B192);
            side_h = bh; side_f = bh;
            corner = px;
        } else if (left) {
            side_h = blend_colors(side_h, px, B192);
            side_f = blend_colors(side_f, px, B64);
            corner = px;
        } else if (up) {
            side_f = blend_colors(side_f, px, B192);
            side_h = blend_colors(side_h, px, B64);
            corner = px;
        } else {
            corner = blend_colors(corner, px, B224);
            side_f = blend_colors(side_f, px, B32);
            side_h = blend_colors(side_h, px, B32);
        }
    } else {
        corner = blend_colors(corner, px, B128);
    }
}

void filt4x(float4 e, int n15, int n14, int n13, int n12,
            int n11, int n10, int n7, int n3) {
    if (uE == uH || uE == uF) return;
    float2 ew_iw = edge_weights();
    if (ew_iw.x > ew_iw.y) return;
    float4 px = pick_color();
    if (ew_iw.x < ew_iw.y && sub_cond_24()) {
        bool left, up;
        detect_direction(left, up);
        if (left && up) {
            float4 b13 = blend_colors(g_out_block[n13], px, B192);
            float4 b12 = blend_colors(g_out_block[n12], px, B64);
            g_out_block[n13] = b13; g_out_block[n12] = b12;
            g_out_block[n15] = px; g_out_block[n14] = px;
            g_out_block[n11] = px; g_out_block[n10] = b12;
            g_out_block[n3] = b12; g_out_block[n7] = b13;
        } else if (left) {
            g_out_block[n11] = blend_colors(g_out_block[n11], px, B192);
            g_out_block[n13] = blend_colors(g_out_block[n13], px, B192);
            g_out_block[n10] = blend_colors(g_out_block[n10], px, B64);
            g_out_block[n12] = blend_colors(g_out_block[n12], px, B64);
            g_out_block[n14] = px; g_out_block[n15] = px;
        } else if (up) {
            g_out_block[n14] = blend_colors(g_out_block[n14], px, B192);
            g_out_block[n7]  = blend_colors(g_out_block[n7],  px, B192);
            g_out_block[n10] = blend_colors(g_out_block[n10], px, B64);
            g_out_block[n3]  = blend_colors(g_out_block[n3],  px, B64);
            g_out_block[n11] = px; g_out_block[n15] = px;
        } else {
            g_out_block[n11] = blend_colors(g_out_block[n11], px, B128);
            g_out_block[n14] = blend_colors(g_out_block[n14], px, B128);
            g_out_block[n15] = px;
        }
    } else {
        g_out_block[n15] = blend_colors(g_out_block[n15], px, B128);
    }
}

// Corner parameter indices: F, H, I, C, G, B, D, F4, H5, I4, I5
static const int CP_BR[11] = { 13,17,18, 8,16, 7,11, 14,22, 19,23 };
static const int CP_BL[11] = { 11,17,16, 6,18, 7,13, 10,22, 15,21 };
static const int CP_TR[11] = { 13, 7, 8, 18, 6, 17,11, 14, 2,  9, 3 };
static const int CP_TL[11] = { 11, 7, 6, 16, 8, 17,13, 10, 2,  5, 1 };

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    int2 out_pos = int2(tid.xy);
    if (out_pos.x >= int(u.out_w) || out_pos.y >= int(u.out_h)) return;

    int isc = int(u.iscale);
    int2 src_i = out_pos / isc;
    int2 sub = out_pos - src_i * isc;
    src_i = clamp(src_i, int2(0), int2(int(u.src_w) - 1, int(u.src_h) - 1));

    // Sample 5x5 neighborhood into local arrays (float4 + uint)
    float4 nb[25];
    uint nb_raw[25];
    for (int j = 0; j < 5; j++)
        for (int i = 0; i < 5; i++) {
            uint raw = fetch_raw(src_i + int2(i - 2, j - 2));
            nb_raw[j * 5 + i] = raw;
            nb[j * 5 + i] = (raw).unpack_rgba();
        }

    float4 e = nb[12];

    if (isc == 2) {
        int cp[11];
        if (sub.x == 0 && sub.y == 0) cp = CP_TL;
        else if (sub.x == 1 && sub.y == 0) cp = CP_TR;
        else if (sub.x == 0 && sub.y == 1) cp = CP_BL;
        else cp = CP_BR;
        resolve_cp(nb, nb_raw, cp);
        output_img.Store(out_pos, filt2x(e));

    } else if (isc == 3) {
        if (sub.x == 1 && sub.y == 1) {
            output_img.Store(out_pos, e);
        } else {
            int cx = sub.x < 1 ? 0 : (sub.x > 1 ? 1 : 0);
            int cy = sub.y < 1 ? 0 : (sub.y > 1 ? 1 : 0);
            int cp[11];
            if (cx == 0 && cy == 0) cp = CP_TL;
            else if (cx == 1 && cy == 0) cp = CP_TR;
            else if (cx == 0 && cy == 1) cp = CP_BL;
            else cp = CP_BR;
            resolve_cp(nb, nb_raw, cp);

            float4 corner = e, side_h = e, side_f = e;
            filt3x(e, corner, side_h, side_f);

            if (sub.x != 1 && sub.y != 1) output_img.Store(out_pos, corner);
            else if (sub.x == 1) output_img.Store(out_pos, side_h);
            else output_img.Store(out_pos, side_f);
        }

    } else {
        for (int i = 0; i < 16; i++) g_out_block[i] = e;

        resolve_cp(nb, nb_raw, CP_BR);
        filt4x(e, 15,14,13,12, 11,10, 7, 3);
        resolve_cp(nb, nb_raw, CP_BL);
        filt4x(e, 12,13,14,15,  8, 9, 4, 0);
        resolve_cp(nb, nb_raw, CP_TR);
        filt4x(e,  3, 2, 1, 0,  7, 6,11,15);
        resolve_cp(nb, nb_raw, CP_TL);
        filt4x(e,  0, 1, 2, 3,  4, 5, 8,12);

        output_img.Store(out_pos, g_out_block[sub.y * 4 + sub.x]);
    }
}
