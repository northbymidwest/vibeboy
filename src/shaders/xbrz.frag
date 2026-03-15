#version 450

// xBRZ — pixel-art scaling (2x–6x) by Zenju.
//
// Single-pass fragment shader that folds both phases (preProcessCorners +
// scalePixel) into one invocation. Each fragment determines its source pixel,
// computes blend info from the 4 surrounding 2x2 blocks, then applies the
// appropriate blending pattern for its sub-pixel position.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D src_tex;

layout(std140, set = 3, binding = 0) uniform Uniforms {
    vec2 src_size;
    float scale;
    float pad0;
};

// ── Constants ───────────────────────────────────────────────────────────────

const float DOMINANT_RATIO = 3.6;
const float LINE_DETECT_RATIO = 2.2;
const float EQ_TOLERANCE = 30.0;

// Blend types (packed 2 bits each)
const int BL_NONE = 0;
const int BL_NORMAL = 1;
const int BL_DOMINANT = 2;

// ── Helpers ─────────────────────────────────────────────────────────────────

vec4 fetch(ivec2 coord) {
    coord = clamp(coord, ivec2(0), ivec2(src_size) - 1);
    return texelFetch(src_tex, coord, 0);
}

// BT.2020 YCbCr color distance (matches CPU xBRZ implementation).
float dist(vec4 a, vec4 b) {
    if (a == b) return 0.0;
    vec3 d = (a.rgb - b.rgb) * 255.0;
    const float K_R = 0.2627;
    const float K_G = 0.6780;
    const float K_B = 0.0593;
    const float S_B = 0.5 / (1.0 - K_B);
    const float S_R = 0.5 / (1.0 - K_R);
    float y  = K_R * d.r + K_G * d.g + K_B * d.b;
    float cb = S_B * (d.b - y);
    float cr = S_R * (d.r - y);
    return sqrt(y * y + cb * cb + cr * cr);
}

bool eq(vec4 a, vec4 b) { return dist(a, b) < EQ_TOLERANCE; }

vec4 blend(vec4 a, vec4 b, float alpha) {
    return mix(a, b, clamp(alpha, 0.0, 1.0));
}

// ── Blend info packing ──────────────────────────────────────────────────────
// TL=[1:0], TR=[3:2], BR=[5:4], BL=[7:6]

int get_tl(int b) { return b & 3; }
int get_tr(int b) { return (b >> 2) & 3; }
int get_br(int b) { return (b >> 4) & 3; }
int get_bl(int b) { return (b >> 6) & 3; }

// ── preProcessCorners (per 2x2 block) ───────────────────────────────────────
//
// For the 2x2 block at (bx, by):
//   F = (bx, by), G = (bx+1, by), J = (bx, by+1), K = (bx+1, by+1)
//
// Returns packed result: blend_f in BR, blend_g in BL, blend_j in TR, blend_k in TL

int preprocess_block(ivec2 bxy) {
    vec4 f = fetch(bxy);
    vec4 g = fetch(bxy + ivec2(1, 0));
    vec4 j = fetch(bxy + ivec2(0, 1));
    vec4 k = fetch(bxy + ivec2(1, 1));

    // Skip uniform blocks (exact comparison)
    if ((f == g && j == k) || (f == j && g == k)) return 0;

    vec4 b_px = fetch(bxy + ivec2(0, -1));
    vec4 c    = fetch(bxy + ivec2(1, -1));
    vec4 e    = fetch(bxy + ivec2(-1, 0));
    vec4 h    = fetch(bxy + ivec2(2, 0));
    vec4 i    = fetch(bxy + ivec2(-1, 1));
    vec4 l    = fetch(bxy + ivec2(2, 1));
    vec4 n    = fetch(bxy + ivec2(0, 2));
    vec4 o    = fetch(bxy + ivec2(1, 2));

    float jg = dist(i, f) + dist(f, c) + dist(n, k) + dist(k, h) + 4.0 * dist(j, g);
    float fk = dist(e, j) + dist(j, o) + dist(b_px, g) + dist(g, l) + 4.0 * dist(f, k);

    int result = 0;

    if (jg < fk) {
        int bt = (DOMINANT_RATIO * jg < fk) ? BL_DOMINANT : BL_NORMAL;
        if (f != g && f != j) result |= (bt << 4); // blend_f → BR
        if (k != j && k != g) result |= bt;        // blend_k → TL
    } else if (fk < jg) {
        int bt = (DOMINANT_RATIO * fk < jg) ? BL_DOMINANT : BL_NORMAL;
        if (j != f && j != k) result |= (bt << 2); // blend_j → TR
        if (g != f && g != k) result |= (bt << 6); // blend_g → BL
    }
    return result;
}

// ── Compute full blend info for source pixel (sx, sy) ───────────────────────
//
// Each corner comes from a different 2x2 block:
//   BR corner: block (sx, sy)
//   BL corner: block (sx-1, sy)
//   TR corner: block (sx, sy-1)
//   TL corner: block (sx-1, sy-1)

int compute_blend_info(ivec2 sxy) {
    int bi = 0;
    // BR corner from block (sx, sy) — stored in BR position
    int r0 = preprocess_block(sxy);
    bi |= (get_br(r0) << 4); // F's BR → our BR

    // BL corner from block (sx-1, sy) — G's BL → our BL
    int r1 = preprocess_block(sxy + ivec2(-1, 0));
    bi |= (get_bl(r1) << 6); // G's BL → our BL

    // TR corner from block (sx, sy-1) — J's TR → our TR
    int r2 = preprocess_block(sxy + ivec2(0, -1));
    bi |= (get_tr(r2) << 2); // J's TR → our TR

    // TL corner from block (sx-1, sy-1) — K's TL → our TL
    int r3 = preprocess_block(sxy + ivec2(-1, -1));
    bi |= get_tl(r3); // K's TL → our TL

    return bi;
}

// ── Rotate blend info ───────────────────────────────────────────────────────

int rotate_blend(int bi, int rot) {
    if (rot == 0) return bi;
    // General rotation: shift the 8-bit value by rot*2 bits
    int r = (rot * 2) & 7;
    return ((bi >> r) | (bi << (8 - r))) & 0xFF;
}

// ── doLineBlend check ───────────────────────────────────────────────────────

bool do_line_blend(int rbi, vec4 e, vec4 g, vec4 h, vec4 i, vec4 f, vec4 c) {
    if (get_br(rbi) >= BL_DOMINANT) return true;
    if (get_tr(rbi) != BL_NONE && !eq(e, g)) return false;
    if (get_bl(rbi) != BL_NONE && !eq(e, c)) return false;
    if (!eq(e, i) && eq(g, h) && eq(h, i) && eq(i, f) && eq(f, c)) return false;
    return true;
}

// ── Blend patterns ──────────────────────────────────────────────────────────
// For a given (sub_row, sub_col) within the NxN block, compute the blend
// contribution for one corner rotation. Returns the blended color.

vec4 apply_corner_blend(vec4 center, vec4 target, int n, int sub_r, int sub_c) {
    int m = n - 1;
    // Corner-only mode: conservative alpha at the corner pixel
    if (sub_r == m && sub_c == m) {
        float alpha = (n == 2) ? 0.21 : (n == 3) ? 0.45 : (n == 4) ? 0.68 : (n == 5) ? 0.86 : 0.97;
        return blend(center, target, alpha);
    }
    if (n >= 4 && ((sub_r == m-1 && sub_c == m) || (sub_r == m && sub_c == m-1))) {
        float alpha = (n == 4) ? 0.09 : (n == 5) ? 0.23 : 0.42;
        return blend(center, target, alpha);
    }
    if (n >= 6 && ((sub_r == m && sub_c == m-2) || (sub_r == m-2 && sub_c == m))) {
        return blend(center, target, 0.06);
    }
    return center;
}

vec4 apply_shallow(vec4 center, vec4 t, int n, int r, int c) {
    int m = n - 1;
    // Shallow: bottom row + ascending diagonal
    if (n == 2) {
        if (r == 1 && c == 0) return blend(center, t, 0.25);
        if (r == 1 && c == 1) return blend(center, t, 0.75);
    } else if (n == 3) {
        if (r == 2 && c == 0) return blend(center, t, 0.25);
        if (r == 1 && c == 2) return blend(center, t, 0.25);
        if (r == 2 && c == 1) return blend(center, t, 0.75);
        if (r == 2 && c == 2) return blend(center, t, 1.0);
    } else if (n == 4) {
        if (r == 3 && c == 0) return blend(center, t, 0.25);
        if (r == 2 && c == 2) return blend(center, t, 0.25);
        if (r == 3 && c == 1) return blend(center, t, 0.75);
        if (r == 2 && c == 3) return blend(center, t, 0.75);
        if (r == 3 && c == 2) return blend(center, t, 1.0);
        if (r == 3 && c == 3) return blend(center, t, 1.0);
    } else if (n == 5) {
        if (r == 4 && c == 0) return blend(center, t, 0.25);
        if (r == 3 && c == 2) return blend(center, t, 0.25);
        if (r == 2 && c == 4) return blend(center, t, 0.25);
        if (r == 4 && c == 1) return blend(center, t, 0.75);
        if (r == 3 && c == 3) return blend(center, t, 0.75);
        if (r == 4 && c == 2) return blend(center, t, 1.0);
        if (r == 4 && c == 3) return blend(center, t, 1.0);
        if (r == 4 && c == 4) return blend(center, t, 1.0);
        if (r == 3 && c == 4) return blend(center, t, 1.0);
    } else { // 6x
        if (r == m && c == 0) return blend(center, t, 0.25);
        if (r == m-1 && c == 2) return blend(center, t, 0.25);
        if (r == m-2 && c == 4) return blend(center, t, 0.25);
        if (r == m && c == 1) return blend(center, t, 0.75);
        if (r == m-1 && c == 3) return blend(center, t, 0.75);
        if (r == m-2 && c == m) return blend(center, t, 0.75);
        if (r == m && c >= 2) return blend(center, t, 1.0);
        if (r == m-1 && c >= m-1) return blend(center, t, 1.0);
    }
    return center;
}

vec4 apply_steep(vec4 center, vec4 t, int n, int r, int c) {
    // Steep = transpose of shallow
    return apply_shallow(center, t, n, c, r);
}

vec4 apply_steep_and_shallow(vec4 center, vec4 t, int n, int r, int c) {
    int m = n - 1;
    if (n == 2) {
        if (r == 1 && c == 0) return blend(center, t, 0.25);
        if (r == 0 && c == 1) return blend(center, t, 0.25);
        if (r == 1 && c == 1) return blend(center, t, 5.0/6.0);
    } else if (n == 3) {
        if (r == 2 && c == 0) return blend(center, t, 0.25);
        if (r == 0 && c == 2) return blend(center, t, 0.25);
        if (r == 2 && c == 1) return blend(center, t, 0.75);
        if (r == 1 && c == 2) return blend(center, t, 0.75);
        if (r == 2 && c == 2) return blend(center, t, 1.0);
    } else if (n == 4) {
        if (r == 3 && c == 1) return blend(center, t, 0.75);
        if (r == 1 && c == 3) return blend(center, t, 0.75);
        if (r == 3 && c == 0) return blend(center, t, 0.25);
        if (r == 0 && c == 3) return blend(center, t, 0.25);
        if (r == 2 && c == 2) return blend(center, t, 1.0/3.0);
        if (r == 3 && c == 2) return blend(center, t, 1.0);
        if (r == 2 && c == 3) return blend(center, t, 1.0);
        if (r == 3 && c == 3) return blend(center, t, 1.0);
    } else if (n == 5) {
        if (r == 0 && c == 4) return blend(center, t, 0.25);
        if (r == 2 && c == 3) return blend(center, t, 0.25);
        if (r == 1 && c == 4) return blend(center, t, 0.75);
        if (r == 4 && c == 0) return blend(center, t, 0.25);
        if (r == 3 && c == 2) return blend(center, t, 0.25);
        if (r == 4 && c == 1) return blend(center, t, 0.75);
        if (r == 3 && c == 3) return blend(center, t, 2.0/3.0);
        if (r == 2 && c == 4) return blend(center, t, 1.0);
        if (r == 3 && c == 4) return blend(center, t, 1.0);
        if (r == 4 && c == 4) return blend(center, t, 1.0);
        if (r == 4 && c == 2) return blend(center, t, 1.0);
        if (r == 4 && c == 3) return blend(center, t, 1.0);
    } else { // 6x
        if (r == 0 && c == m) return blend(center, t, 0.25);
        if (r == 2 && c == m-1) return blend(center, t, 0.25);
        if (r == 1 && c == m) return blend(center, t, 0.75);
        if (r == 3 && c == m-1) return blend(center, t, 0.75);
        if (r == m && c == 0) return blend(center, t, 0.25);
        if (r == m-1 && c == 2) return blend(center, t, 0.25);
        if (r == m && c == 1) return blend(center, t, 0.75);
        if (r == m-1 && c == 3) return blend(center, t, 0.75);
        if (r == 2 && c == m) return blend(center, t, 1.0);
        if (r == 3 && c == m) return blend(center, t, 1.0);
        if (r == 4 && c == m) return blend(center, t, 1.0);
        if (r == m && c == m) return blend(center, t, 1.0);
        if (r == m-1 && c == m-1) return blend(center, t, 1.0);
        if (r == m && c == m-1) return blend(center, t, 1.0);
        if (r == m && c == 2) return blend(center, t, 1.0);
        if (r == m && c == 3) return blend(center, t, 1.0);
    }
    return center;
}

vec4 apply_diagonal(vec4 center, vec4 t, int n, int r, int c) {
    int m = n - 1;
    if (n == 2) {
        if (r == 1 && c == 1) return blend(center, t, 0.5);
    } else if (n == 3) {
        if (r == 1 && c == 2) return blend(center, t, 1.0/8.0);
        if (r == 2 && c == 1) return blend(center, t, 1.0/8.0);
        if (r == 2 && c == 2) return blend(center, t, 7.0/8.0);
    } else if (n == 4) {
        if (r == m && c == n/2) return blend(center, t, 0.5);
        if (r == m-1 && c == n/2+1) return blend(center, t, 0.5);
        if (r == m && c == m) return blend(center, t, 1.0);
    } else if (n == 5) {
        if (r == m && c == n/2) return blend(center, t, 1.0/8.0);
        if (r == m-1 && c == n/2+1) return blend(center, t, 1.0/8.0);
        if (r == m-2 && c == n/2+2) return blend(center, t, 1.0/8.0);
        if (r == 4 && c == 3) return blend(center, t, 7.0/8.0);
        if (r == 3 && c == 4) return blend(center, t, 7.0/8.0);
        if (r == 4 && c == 4) return blend(center, t, 1.0);
    } else { // 6x
        if (r == m && c == n/2) return blend(center, t, 0.5);
        if (r == m-1 && c == n/2+1) return blend(center, t, 0.5);
        if (r == m-2 && c == n/2+2) return blend(center, t, 0.5);
        if (r == m-1 && c == m) return blend(center, t, 1.0);
        if (r == m && c == m) return blend(center, t, 1.0);
        if (r == m && c == m-1) return blend(center, t, 1.0);
    }
    return center;
}

// ── Rotation helpers ────────────────────────────────────────────────────────

// Rotate (sub_r, sub_c) so that corner `rot` maps to bottom-right.
ivec2 rotate_sub(int sub_r, int sub_c, int n, int rot) {
    int m = n - 1;
    if (rot == 0) return ivec2(sub_r, sub_c);
    if (rot == 1) return ivec2(sub_r, m - sub_c);
    if (rot == 2) return ivec2(m - sub_r, m - sub_c);
    return ivec2(m - sub_r, sub_c);
}

// Direction offsets for each rotation: (dx, dy)
ivec2 rot_dir(int rot) {
    if (rot == 0) return ivec2(1, 1);
    if (rot == 1) return ivec2(-1, 1);
    if (rot == 2) return ivec2(-1, -1);
    return ivec2(1, -1);
}

// ── Main ────────────────────────────────────────────────────────────────────

void main() {
    int n = int(scale);
    vec2 src_coord = v_uv * src_size;
    ivec2 src_i = clamp(ivec2(floor(src_coord)), ivec2(0), ivec2(src_size) - 1);
    vec2 frac_val = src_coord - vec2(src_i);
    ivec2 sub = clamp(ivec2(floor(frac_val * scale)), ivec2(0), ivec2(n - 1));

    vec4 e = fetch(src_i);

    // Compute blend info for this source pixel
    int bi = compute_blend_info(src_i);

    if (bi == 0) {
        frag_color = e;
        return;
    }

    vec4 result = e;

    // Process 4 rotations, each handling one corner
    for (int rot = 0; rot < 4; rot++) {
        int rbi = rotate_blend(bi, rot);
        if (get_br(rbi) == BL_NONE) continue;

        // Rotate sub-pixel coords so current corner is at bottom-right
        ivec2 rs = rotate_sub(sub.y, sub.x, n, rot);

        // Sample rotated 3x3 kernel
        ivec2 d = rot_dir(rot);
        vec4 f_px = fetch(src_i + ivec2(d.x, 0));
        vec4 h_px = fetch(src_i + ivec2(0, d.y));
        vec4 c_px = fetch(src_i + ivec2(d.x, -d.y));
        vec4 g_px = fetch(src_i + ivec2(-d.x, d.y));
        vec4 i_px = fetch(src_i + ivec2(d.x, d.y));

        // Blend target: more similar axis neighbor
        vec4 target = (dist(e, f_px) <= dist(e, h_px)) ? f_px : h_px;

        bool line = do_line_blend(rbi, e, g_px, h_px, i_px, f_px, c_px);

        if (!line) {
            result = apply_corner_blend(result, target, n, rs.x, rs.y);
        } else {
            vec4 d_px = fetch(src_i + ivec2(-d.x, 0));
            vec4 b_px = fetch(src_i + ivec2(0, -d.y));
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

    frag_color = result;
}
