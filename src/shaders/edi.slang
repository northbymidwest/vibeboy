// EDI -- Edge-Directed Interpolation (2x upscaling).
// Structure-tensor edge detection with direction-adaptive blending.

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

static const float FLAT_THRESHOLD = 500.0;
static const float SHARP_THRESHOLD = 8000.0;

uint fetch_raw(int2 coord) { return px_raw(pixels, int(u.src_w), int(u.src_h), coord); }

float4 fetch(int2 coord) {
    return (fetch_raw(coord)).unpack_rgba();
}

float luma(float4 c) {
    return 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
}

// Squared color distance (RGB, 0-255 scale)
float color_dist(uint a, uint b) {
    float dr = float(int((a >> 16u) & 0xFFu) - int((b >> 16u) & 0xFFu));
    float dg = float(int((a >> 8u)  & 0xFFu) - int((b >> 8u)  & 0xFFu));
    float db = float(int( a         & 0xFFu) - int( b         & 0xFFu));
    return dr * dr + dg * dg + db * db;
}

float4 blend2(float4 a, float4 b, float t) {
    return lerp(a, b, t);
}

float4 avg4(float4 a, float4 b, float4 c, float4 d) {
    return (a + b + c + d) * 0.25;
}

// Structure tensor edge direction at diamond point (x+0.5, y+0.5)
void edge_direction(int2 p, out float angle, out float strength) {
    float gxx = 0.0, gyy = 0.0, gxy = 0.0;
    for (int ky = 0; ky < 2; ky++) {
        for (int kx = 0; kx < 2; kx++) {
            int2 c = p + int2(kx, ky);
            float lc = luma(fetch(c));
            float lr = luma(fetch(c + int2(1, 0)));
            float ld = luma(fetch(c + int2(0, 1)));
            float dx = lr - lc;
            float dy = ld - lc;
            gxx += dx * dx;
            gyy += dy * dy;
            gxy += dx * dy;
        }
    }
    strength = gxx + gyy;
    if (strength < 0.001) {
        angle = 0.0;
        strength = 0.0;
        return;
    }
    angle = 0.5 * atan2(2.0 * gxy, gxx - gyy);
}

float4 diamond_interp(int2 p) {
    uint raw_tl = fetch_raw(p);
    uint raw_tr = fetch_raw(p + int2(1, 0));
    uint raw_bl = fetch_raw(p + int2(0, 1));
    uint raw_br = fetch_raw(p + int2(1, 1));

    float4 tl = (raw_tl).unpack_rgba();
    float4 tr = (raw_tr).unpack_rgba();
    float4 bl = (raw_bl).unpack_rgba();
    float4 br = (raw_br).unpack_rgba();

    float d_main = color_dist(raw_tl, raw_br);
    float d_anti = color_dist(raw_tr, raw_bl);
    float d_h1 = color_dist(raw_tl, raw_tr);
    float d_h2 = color_dist(raw_bl, raw_br);
    float d_v1 = color_dist(raw_tl, raw_bl);
    float d_v2 = color_dist(raw_tr, raw_br);

    float max_d = max(max(max(d_main, d_anti), max(d_h1, d_h2)), max(d_v1, d_v2));

    if (max_d < FLAT_THRESHOLD) {
        return avg4(tl, tr, bl, br);
    }

    if (max_d > SHARP_THRESHOLD) {
        if (d_main < d_anti * 0.25) return blend2(tl, br, 0.5);
        if (d_anti < d_main * 0.25) return blend2(tr, bl, 0.5);
        return avg4(tl, tr, bl, br);
    }

    // Medium edge -- directional interpolation
    float angle, strength;
    edge_direction(p, angle, strength);

    float cos2 = cos(2.0 * angle);
    float sin2 = sin(2.0 * angle);

    float w_main = (1.0 + sin2) * 0.5;
    float w_anti = (1.0 - sin2) * 0.5;
    float w_horz = (1.0 + cos2) * 0.5;
    float w_vert = (1.0 - cos2) * 0.5;

    float4 c_main = blend2(tl, br, 0.5);
    float4 c_anti = blend2(tr, bl, 0.5);
    float4 c_horz = blend2(blend2(tl, tr, 0.5), blend2(bl, br, 0.5), 0.5);
    float4 c_vert = blend2(blend2(tl, bl, 0.5), blend2(tr, br, 0.5), 0.5);

    // Diagonal weights scaled down by 0.5 (matches CPU)
    w_main *= 0.5;
    w_anti *= 0.5;

    // Find top 2 by weight and blend
    float weights[4] = { w_main, w_anti, w_horz, w_vert };
    float4 colors[4] = { c_main, c_anti, c_horz, c_vert };

    // Simple top-2 selection (sort 4 elements)
    for (int i = 0; i < 3; i++) {
        for (int j = i + 1; j < 4; j++) {
            if (weights[j] > weights[i]) {
                float tw = weights[i]; weights[i] = weights[j]; weights[j] = tw;
                float4 tc = colors[i]; colors[i] = colors[j]; colors[j] = tc;
            }
        }
    }

    float total = weights[0] + weights[1];
    if (total < 0.001) return avg4(tl, tr, bl, br);
    return blend2(colors[0], colors[1], weights[1] / total);
}

float4 hedge_interp(int2 p) {
    uint raw_l = fetch_raw(p);
    uint raw_r = fetch_raw(p + int2(1, 0));
    float4 left = (raw_l).unpack_rgba();
    float4 right = (raw_r).unpack_rgba();
    float d = color_dist(raw_l, raw_r);

    if (d > SHARP_THRESHOLD) {
        uint raw_al = fetch_raw(p + int2(0, -1));
        uint raw_ar = fetch_raw(p + int2(1, -1));
        uint raw_bl = fetch_raw(p + int2(0,  1));
        uint raw_br = fetch_raw(p + int2(1,  1));
        float cont_l = color_dist(raw_al, raw_l) + color_dist(raw_l, raw_bl);
        float cont_r = color_dist(raw_ar, raw_r) + color_dist(raw_r, raw_br);
        if (cont_l < cont_r * 0.5) return left;
        if (cont_r < cont_l * 0.5) return right;
    }
    return blend2(left, right, 0.5);
}

float4 vedge_interp(int2 p) {
    uint raw_t = fetch_raw(p);
    uint raw_b = fetch_raw(p + int2(0, 1));
    float4 top = (raw_t).unpack_rgba();
    float4 bot = (raw_b).unpack_rgba();
    float d = color_dist(raw_t, raw_b);

    if (d > SHARP_THRESHOLD) {
        uint raw_lt = fetch_raw(p + int2(-1, 0));
        uint raw_lb = fetch_raw(p + int2(-1, 1));
        uint raw_rt = fetch_raw(p + int2( 1, 0));
        uint raw_rb = fetch_raw(p + int2( 1, 1));
        float cont_t = color_dist(raw_lt, raw_t) + color_dist(raw_t, raw_rt);
        float cont_b = color_dist(raw_lb, raw_b) + color_dist(raw_b, raw_rb);
        if (cont_t < cont_b * 0.5) return top;
        if (cont_b < cont_t * 0.5) return bot;
    }
    return blend2(top, bot, 0.5);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint opx = tid.x;
    uint opy = tid.y;
    if (opx >= u.out_w || opy >= u.out_h) return;

    // Map output pixel to source pixel + sub-pixel within 2x2 block
    int2 src_i = int2(opx / 2u, opy / 2u);
    int2 sub = int2(opx & 1u, opy & 1u);

    float4 result;
    if (sub.x == 0 && sub.y == 0) {
        result = fetch(src_i);
    } else if (sub.x == 1 && sub.y == 0) {
        result = hedge_interp(src_i);
    } else if (sub.x == 0 && sub.y == 1) {
        result = vedge_interp(src_i);
    } else {
        result = diamond_interp(src_i);
    }

    output_img.Store(int2(opx, opy), result);
}
