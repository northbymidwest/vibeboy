// xBRZ -- pixel-art scaling (2x-6x) by Zenju.
//
// Compute shader version. Single-pass: each thread computes one output pixel,
// folding both phases (preProcessCorners + scalePixel) into one invocation.
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

// -- Constants --

static const float DOMINANT_RATIO = 3.6;
static const float LINE_DETECT_RATIO = 2.2;
static const float EQ_TOLERANCE = 30.0;

// Blend types (packed 2 bits each)
static const int BL_NONE = 0;
static const int BL_NORMAL = 1;
static const int BL_DOMINANT = 2;

// -- Helpers --

uint fetch_raw(int2 coord) { return px_raw(pixels, int(u.src_w), int(u.src_h), coord); }

float4 fetch(int2 coord) {
    return (fetch_raw(coord)).unpack_rgba();
}

float dist(float4 a, float4 b) {
    return ycbcr_dist(a, b);
}

// Distance with packed u32 early-out for identical colors.
float dist_raw(uint ra, uint rb, float4 a, float4 b) {
    if (ra == rb) return 0.0;
    return dist(a, b);
}

bool eq(float4 a, float4 b) { return dist(a, b) < EQ_TOLERANCE; }

float4 blend_colors(float4 a, float4 b, float alpha) {
    return lerp(a, b, clamp(alpha, 0.0, 1.0));
}

// -- Blend info packing --
// TL=[1:0], TR=[3:2], BR=[5:4], BL=[7:6]

int get_tl(int b) { return b & 3; }
int get_tr(int b) { return (b >> 2) & 3; }
int get_br(int b) { return (b >> 4) & 3; }
int get_bl(int b) { return (b >> 6) & 3; }

// -- preProcessCorners (per 2x2 block) --

int preprocess_block(int2 bxy) {
    uint rf = fetch_raw(bxy);
    uint rg = fetch_raw(bxy + int2(1, 0));
    uint rj = fetch_raw(bxy + int2(0, 1));
    uint rk = fetch_raw(bxy + int2(1, 1));

    // Skip uniform blocks (exact comparison using packed u32)
    if ((rf == rg && rj == rk) || (rf == rj && rg == rk)) return 0;

    float4 f = (rf).unpack_rgba();
    float4 g = (rg).unpack_rgba();
    float4 j = (rj).unpack_rgba();
    float4 k = (rk).unpack_rgba();

    float4 b_px = fetch(bxy + int2(0, -1));
    float4 c    = fetch(bxy + int2(1, -1));
    float4 e    = fetch(bxy + int2(-1, 0));
    float4 h    = fetch(bxy + int2(2, 0));
    float4 i    = fetch(bxy + int2(-1, 1));
    float4 l    = fetch(bxy + int2(2, 1));
    float4 n    = fetch(bxy + int2(0, 2));
    float4 o    = fetch(bxy + int2(1, 2));

    float jg = dist(i, f) + dist(f, c) + dist(n, k) + dist(k, h) + 4.0 * dist(j, g);
    float fk = dist(e, j) + dist(j, o) + dist(b_px, g) + dist(g, l) + 4.0 * dist(f, k);

    int result = 0;

    if (jg < fk) {
        int bt = (DOMINANT_RATIO * jg < fk) ? BL_DOMINANT : BL_NORMAL;
        if (rf != rg && rf != rj) result |= (bt << 4); // blend_f -> BR
        if (rk != rj && rk != rg) result |= bt;        // blend_k -> TL
    } else if (fk < jg) {
        int bt = (DOMINANT_RATIO * fk < jg) ? BL_DOMINANT : BL_NORMAL;
        if (rj != rf && rj != rk) result |= (bt << 2); // blend_j -> TR
        if (rg != rf && rg != rk) result |= (bt << 6); // blend_g -> BL
    }
    return result;
}

// -- Compute full blend info for source pixel (sx, sy) --

int compute_blend_info(int2 sxy) {
    int bi = 0;
    int r0 = preprocess_block(sxy);
    bi |= (get_br(r0) << 4);

    int r1 = preprocess_block(sxy + int2(-1, 0));
    bi |= (get_bl(r1) << 6);

    int r2 = preprocess_block(sxy + int2(0, -1));
    bi |= (get_tr(r2) << 2);

    int r3 = preprocess_block(sxy + int2(-1, -1));
    bi |= get_tl(r3);

    return bi;
}

// -- Rotate blend info --

int rotate_blend(int bi, int rot) {
    if (rot == 0) return bi;
    int r = (rot * 2) & 7;
    return ((bi >> r) | (bi << (8 - r))) & 0xFF;
}

// -- doLineBlend check --

bool do_line_blend(int rbi, float4 e, float4 g, float4 h, float4 i, float4 f, float4 c) {
    if (get_br(rbi) >= BL_DOMINANT) return true;
    if (get_tr(rbi) != BL_NONE && !eq(e, g)) return false;
    if (get_bl(rbi) != BL_NONE && !eq(e, c)) return false;
    if (!eq(e, i) && eq(g, h) && eq(h, i) && eq(i, f) && eq(f, c)) return false;
    return true;
}

// -- Blend patterns --

float4 apply_corner_blend(float4 center, float4 target, int n, int sub_r, int sub_c) {
    int m = n - 1;
    if (sub_r == m && sub_c == m) {
        float alpha = (n == 2) ? 0.21 : (n == 3) ? 0.45 : (n == 4) ? 0.68 : (n == 5) ? 0.86 : 0.97;
        return blend_colors(center, target, alpha);
    }
    if (n >= 4 && ((sub_r == m-1 && sub_c == m) || (sub_r == m && sub_c == m-1))) {
        float alpha = (n == 4) ? 0.09 : (n == 5) ? 0.23 : 0.42;
        return blend_colors(center, target, alpha);
    }
    if (n >= 6 && ((sub_r == m && sub_c == m-2) || (sub_r == m-2 && sub_c == m))) {
        return blend_colors(center, target, 0.06);
    }
    return center;
}

float4 apply_shallow(float4 center, float4 t, int n, int r, int c) {
    int m = n - 1;
    if (n == 2) {
        if (r == 1 && c == 0) return blend_colors(center, t, 0.25);
        if (r == 1 && c == 1) return blend_colors(center, t, 0.75);
    } else if (n == 3) {
        if (r == 2 && c == 0) return blend_colors(center, t, 0.25);
        if (r == 1 && c == 2) return blend_colors(center, t, 0.25);
        if (r == 2 && c == 1) return blend_colors(center, t, 0.75);
        if (r == 2 && c == 2) return blend_colors(center, t, 1.0);
    } else if (n == 4) {
        if (r == 3 && c == 0) return blend_colors(center, t, 0.25);
        if (r == 2 && c == 2) return blend_colors(center, t, 0.25);
        if (r == 3 && c == 1) return blend_colors(center, t, 0.75);
        if (r == 2 && c == 3) return blend_colors(center, t, 0.75);
        if (r == 3 && c == 2) return blend_colors(center, t, 1.0);
        if (r == 3 && c == 3) return blend_colors(center, t, 1.0);
    } else if (n == 5) {
        if (r == 4 && c == 0) return blend_colors(center, t, 0.25);
        if (r == 3 && c == 2) return blend_colors(center, t, 0.25);
        if (r == 2 && c == 4) return blend_colors(center, t, 0.25);
        if (r == 4 && c == 1) return blend_colors(center, t, 0.75);
        if (r == 3 && c == 3) return blend_colors(center, t, 0.75);
        if (r == 4 && c == 2) return blend_colors(center, t, 1.0);
        if (r == 4 && c == 3) return blend_colors(center, t, 1.0);
        if (r == 4 && c == 4) return blend_colors(center, t, 1.0);
        if (r == 3 && c == 4) return blend_colors(center, t, 1.0);
    } else { // 6x
        if (r == m && c == 0) return blend_colors(center, t, 0.25);
        if (r == m-1 && c == 2) return blend_colors(center, t, 0.25);
        if (r == m-2 && c == 4) return blend_colors(center, t, 0.25);
        if (r == m && c == 1) return blend_colors(center, t, 0.75);
        if (r == m-1 && c == 3) return blend_colors(center, t, 0.75);
        if (r == m-2 && c == m) return blend_colors(center, t, 0.75);
        if (r == m && c >= 2) return blend_colors(center, t, 1.0);
        if (r == m-1 && c >= m-1) return blend_colors(center, t, 1.0);
    }
    return center;
}

float4 apply_steep(float4 center, float4 t, int n, int r, int c) {
    // Steep = transpose of shallow
    return apply_shallow(center, t, n, c, r);
}

float4 apply_steep_and_shallow(float4 center, float4 t, int n, int r, int c) {
    int m = n - 1;
    if (n == 2) {
        if (r == 1 && c == 0) return blend_colors(center, t, 0.25);
        if (r == 0 && c == 1) return blend_colors(center, t, 0.25);
        if (r == 1 && c == 1) return blend_colors(center, t, 5.0/6.0);
    } else if (n == 3) {
        if (r == 2 && c == 0) return blend_colors(center, t, 0.25);
        if (r == 0 && c == 2) return blend_colors(center, t, 0.25);
        if (r == 2 && c == 1) return blend_colors(center, t, 0.75);
        if (r == 1 && c == 2) return blend_colors(center, t, 0.75);
        if (r == 2 && c == 2) return blend_colors(center, t, 1.0);
    } else if (n == 4) {
        if (r == 3 && c == 1) return blend_colors(center, t, 0.75);
        if (r == 1 && c == 3) return blend_colors(center, t, 0.75);
        if (r == 3 && c == 0) return blend_colors(center, t, 0.25);
        if (r == 0 && c == 3) return blend_colors(center, t, 0.25);
        if (r == 2 && c == 2) return blend_colors(center, t, 1.0/3.0);
        if (r == 3 && c == 2) return blend_colors(center, t, 1.0);
        if (r == 2 && c == 3) return blend_colors(center, t, 1.0);
        if (r == 3 && c == 3) return blend_colors(center, t, 1.0);
    } else if (n == 5) {
        if (r == 0 && c == 4) return blend_colors(center, t, 0.25);
        if (r == 2 && c == 3) return blend_colors(center, t, 0.25);
        if (r == 1 && c == 4) return blend_colors(center, t, 0.75);
        if (r == 4 && c == 0) return blend_colors(center, t, 0.25);
        if (r == 3 && c == 2) return blend_colors(center, t, 0.25);
        if (r == 4 && c == 1) return blend_colors(center, t, 0.75);
        if (r == 3 && c == 3) return blend_colors(center, t, 2.0/3.0);
        if (r == 2 && c == 4) return blend_colors(center, t, 1.0);
        if (r == 3 && c == 4) return blend_colors(center, t, 1.0);
        if (r == 4 && c == 4) return blend_colors(center, t, 1.0);
        if (r == 4 && c == 2) return blend_colors(center, t, 1.0);
        if (r == 4 && c == 3) return blend_colors(center, t, 1.0);
    } else { // 6x
        if (r == 0 && c == m) return blend_colors(center, t, 0.25);
        if (r == 2 && c == m-1) return blend_colors(center, t, 0.25);
        if (r == 1 && c == m) return blend_colors(center, t, 0.75);
        if (r == 3 && c == m-1) return blend_colors(center, t, 0.75);
        if (r == m && c == 0) return blend_colors(center, t, 0.25);
        if (r == m-1 && c == 2) return blend_colors(center, t, 0.25);
        if (r == m && c == 1) return blend_colors(center, t, 0.75);
        if (r == m-1 && c == 3) return blend_colors(center, t, 0.75);
        if (r == 2 && c == m) return blend_colors(center, t, 1.0);
        if (r == 3 && c == m) return blend_colors(center, t, 1.0);
        if (r == 4 && c == m) return blend_colors(center, t, 1.0);
        if (r == m && c == m) return blend_colors(center, t, 1.0);
        if (r == m-1 && c == m-1) return blend_colors(center, t, 1.0);
        if (r == m && c == m-1) return blend_colors(center, t, 1.0);
        if (r == m && c == 2) return blend_colors(center, t, 1.0);
        if (r == m && c == 3) return blend_colors(center, t, 1.0);
    }
    return center;
}

float4 apply_diagonal(float4 center, float4 t, int n, int r, int c) {
    int m = n - 1;
    if (n == 2) {
        if (r == 1 && c == 1) return blend_colors(center, t, 0.5);
    } else if (n == 3) {
        if (r == 1 && c == 2) return blend_colors(center, t, 1.0/8.0);
        if (r == 2 && c == 1) return blend_colors(center, t, 1.0/8.0);
        if (r == 2 && c == 2) return blend_colors(center, t, 7.0/8.0);
    } else if (n == 4) {
        if (r == m && c == n/2) return blend_colors(center, t, 0.5);
        if (r == m-1 && c == n/2+1) return blend_colors(center, t, 0.5);
        if (r == m && c == m) return blend_colors(center, t, 1.0);
    } else if (n == 5) {
        if (r == m && c == n/2) return blend_colors(center, t, 1.0/8.0);
        if (r == m-1 && c == n/2+1) return blend_colors(center, t, 1.0/8.0);
        if (r == m-2 && c == n/2+2) return blend_colors(center, t, 1.0/8.0);
        if (r == 4 && c == 3) return blend_colors(center, t, 7.0/8.0);
        if (r == 3 && c == 4) return blend_colors(center, t, 7.0/8.0);
        if (r == 4 && c == 4) return blend_colors(center, t, 1.0);
    } else { // 6x
        if (r == m && c == n/2) return blend_colors(center, t, 0.5);
        if (r == m-1 && c == n/2+1) return blend_colors(center, t, 0.5);
        if (r == m-2 && c == n/2+2) return blend_colors(center, t, 0.5);
        if (r == m-1 && c == m) return blend_colors(center, t, 1.0);
        if (r == m && c == m) return blend_colors(center, t, 1.0);
        if (r == m && c == m-1) return blend_colors(center, t, 1.0);
    }
    return center;
}

// -- Rotation helpers --

int2 rotate_sub(int sub_r, int sub_c, int n, int rot) {
    int m = n - 1;
    if (rot == 0) return int2(sub_r, sub_c);
    if (rot == 1) return int2(sub_r, m - sub_c);
    if (rot == 2) return int2(m - sub_r, m - sub_c);
    return int2(m - sub_r, sub_c);
}

int2 rot_dir(int rot) {
    if (rot == 0) return int2(1, 1);
    if (rot == 1) return int2(-1, 1);
    if (rot == 2) return int2(-1, -1);
    return int2(1, -1);
}

// -- Main --

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    int2 pos = int2(tid.xy);
    if (pos.x >= int(u.out_w) || pos.y >= int(u.out_h)) return;

    int n = int(u.iscale);

    // Map output pixel to source pixel + sub-pixel position
    int2 src_i = pos / n;
    int2 sub = pos - src_i * n;

    float4 e = fetch(src_i);

    // Compute blend info for this source pixel
    int bi = compute_blend_info(src_i);

    if (bi == 0) {
        output_img.Store(pos, e);
        return;
    }

    float4 result = e;

    // Process 4 rotations, each handling one corner
    for (int rot = 0; rot < 4; rot++) {
        int rbi = rotate_blend(bi, rot);
        if (get_br(rbi) == BL_NONE) continue;

        // Rotate sub-pixel coords so current corner is at bottom-right
        int2 rs = rotate_sub(sub.y, sub.x, n, rot);

        // Sample rotated 3x3 kernel
        int2 d = rot_dir(rot);
        float4 f_px = fetch(src_i + int2(d.x, 0));
        float4 h_px = fetch(src_i + int2(0, d.y));
        float4 c_px = fetch(src_i + int2(d.x, -d.y));
        float4 g_px = fetch(src_i + int2(-d.x, d.y));
        float4 i_px = fetch(src_i + int2(d.x, d.y));

        // Blend target: more similar axis neighbor
        float4 target = (dist(e, f_px) <= dist(e, h_px)) ? f_px : h_px;

        bool line = do_line_blend(rbi, e, g_px, h_px, i_px, f_px, c_px);

        if (!line) {
            result = apply_corner_blend(result, target, n, rs.x, rs.y);
        } else {
            float4 d_px = fetch(src_i + int2(-d.x, 0));
            float4 b_px = fetch(src_i + int2(0, -d.y));
            float fg = dist(f_px, g_px);
            float hc = dist(h_px, c_px);
            bool shallow = LINE_DETECT_RATIO * fg <= hc && !eq(e, g_px) && !eq(d_px, g_px);
            bool steep   = LINE_DETECT_RATIO * hc <= fg && !eq(e, c_px) && !eq(b_px, c_px);

            if (steep && shallow)
                result = apply_steep_and_shallow(result, target, n, rs.x, rs.y);
            else if (shallow)
                result = apply_shallow(result, target, n, rs.x, rs.y);
            else if (steep)
                result = apply_steep(result, target, n, rs.x, rs.y);
            else
                result = apply_diagonal(result, target, n, rs.x, rs.y);
        }
    }

    output_img.Store(pos, result);
}
