#version 450

// xBR — pixel-art scaling (2x, 3x, 4x) by Hyllian.
//
// Uses weighted color distance on a 5x5 neighborhood to detect diagonal edges
// at each corner of the output block, then applies two-level directional
// interpolation with steep/shallow line detection.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D src_tex;

layout(std140, set = 3, binding = 0) uniform Uniforms {
    vec2 src_size;
    float scale;
    float pad0;
};

// ── Helpers ─────────────────────────────────────────────────────────────────

vec4 fetch(ivec2 coord) {
    coord = clamp(coord, ivec2(0), ivec2(src_size) - 1);
    return texelFetch(src_tex, coord, 0);
}

// BT.601 YCbCr weighted color distance (matches CPU xBR implementation).
float color_dist(vec4 a, vec4 b) {
    vec3 d = (a.rgb - b.rgb) * 255.0;
    float dy  =  0.299 * d.r + 0.587 * d.g + 0.114 * d.b;
    float dcb = -0.169 * d.r - 0.331 * d.g + 0.500 * d.b;
    float dcr =  0.500 * d.r - 0.419 * d.g - 0.081 * d.b;
    return 48.0 * dy * dy + 7.0 * dcb * dcb + 6.0 * dcr * dcr;
}

vec4 blend(vec4 a, vec4 b, float alpha) {
    return mix(a, b, clamp(alpha, 0.0, 1.0));
}

// ── Blend weight constants ──────────────────────────────────────────────────

const float B32  = 32.0 / 256.0;
const float B64  = 64.0 / 256.0;
const float B128 = 128.0 / 256.0;
const float B192 = 192.0 / 256.0;
const float B224 = 224.0 / 256.0;

// ── 5x5 neighborhood ────────────────────────────────────────────────────────
//
//   0  1  2  3  4
//   5  6  7  8  9
//  10 11 [12] 13 14
//  15 16  17 18 19
//  20 21  22 23 24

// Corner parameter indices: F, H, I, C, G, B, D, F4, H5, I4, I5
// BR corner
const int CP_BR[11] = int[](13,17,18, 8,16, 7,11, 14,22, 19,23);
// BL corner
const int CP_BL[11] = int[](11,17,16, 6,18, 7,13, 10,22, 15,21);
// TR corner
const int CP_TR[11] = int[](13, 7, 8, 18, 6, 17,11, 14, 2,  9, 3);
// TL corner
const int CP_TL[11] = int[](11, 7, 6, 16, 8, 17,13, 10, 2,  5, 1);

// ── Edge detection ──────────────────────────────────────────────────────────

// Compute edge weights for a corner.
//   ew = d(E,C) + d(E,G) + d(I,H5) + d(I,F4) + 4*d(H,F)
//   iw = d(H,D) + d(H,I5) + d(F,I4) + d(F,B) + 4*d(E,I)
vec2 edge_weights(vec4 nb[25], int cp[11]) {
    float ew = color_dist(nb[12], nb[cp[3]])
        + color_dist(nb[12], nb[cp[4]])
        + color_dist(nb[cp[2]], nb[cp[8]])
        + color_dist(nb[cp[2]], nb[cp[7]])
        + 4.0 * color_dist(nb[cp[1]], nb[cp[0]]);

    float iw = color_dist(nb[cp[1]], nb[cp[6]])
        + color_dist(nb[cp[1]], nb[cp[10]])
        + color_dist(nb[cp[0]], nb[cp[9]])
        + color_dist(nb[cp[0]], nb[cp[5]])
        + 4.0 * color_dist(nb[12], nb[cp[2]]);

    return vec2(ew, iw);
}

// Select blend color: the axis neighbor closer to center.
vec4 pick_color(vec4 nb[25], int cp[11]) {
    return color_dist(nb[12], nb[cp[0]]) <= color_dist(nb[12], nb[cp[1]])
        ? nb[cp[0]] : nb[cp[1]];
}

// Steep/shallow line detection.
void detect_direction(vec4 nb[25], int cp[11], out bool is_left, out bool is_up) {
    float ke = color_dist(nb[cp[0]], nb[cp[4]]);
    float ki = color_dist(nb[cp[1]], nb[cp[3]]);
    is_left = ke * 2.0 <= ki && nb[12] != nb[cp[4]] && nb[cp[6]] != nb[cp[4]];
    is_up   = ke >= ki * 2.0 && nb[12] != nb[cp[3]] && nb[cp[5]] != nb[cp[3]];
}

// Sub-condition for Level 2 (2x/4x variant).
bool sub_cond_24(vec4 nb[25], int cp[11]) {
    return (nb[cp[0]] != nb[cp[5]] && nb[cp[1]] != nb[cp[6]])
        || (nb[12] == nb[cp[2]] && nb[cp[0]] != nb[cp[9]] && nb[cp[1]] != nb[cp[10]])
        || nb[12] == nb[cp[4]]
        || nb[12] == nb[cp[3]];
}

// Sub-condition for Level 2 (3x variant — more permissive).
bool sub_cond_3(vec4 nb[25], int cp[11]) {
    return (nb[cp[0]] != nb[cp[5]] && nb[cp[0]] != nb[cp[3]])
        || (nb[cp[1]] != nb[cp[6]] && nb[cp[1]] != nb[cp[4]])
        || (nb[12] == nb[cp[2]]
            && ((nb[cp[0]] != nb[cp[7]] && nb[cp[0]] != nb[cp[9]])
                || (nb[cp[1]] != nb[cp[8]] && nb[cp[1]] != nb[cp[10]])))
        || nb[12] == nb[cp[4]]
        || nb[12] == nb[cp[3]];
}

// ── Per-corner blending (2x) ────────────────────────────────────────────────

vec4 filt2x(vec4 nb[25], int cp[11], vec4 e) {
    if (nb[12] == nb[cp[1]] || nb[12] == nb[cp[0]]) return e;

    vec2 ew_iw = edge_weights(nb, cp);
    if (ew_iw.x > ew_iw.y) return e;

    vec4 px = pick_color(nb, cp);

    if (ew_iw.x < ew_iw.y && sub_cond_24(nb, cp)) {
        bool left, up;
        detect_direction(nb, cp, left, up);
        if (left && up) return blend(e, px, B224);
        if (left)       return blend(e, px, B192);
        if (up)         return blend(e, px, B192);
        return blend(e, px, B128);
    }
    return blend(e, px, B128);
}

// ── Per-corner blending (3x) ────────────────────────────────────────────────
// Returns vec4[3]: corner, side_h, side_f

void filt3x(vec4 nb[25], int cp[11], vec4 e,
            inout vec4 corner, inout vec4 side_h, inout vec4 side_f) {
    if (nb[12] == nb[cp[1]] || nb[12] == nb[cp[0]]) return;

    vec2 ew_iw = edge_weights(nb, cp);
    if (ew_iw.x > ew_iw.y) return;

    vec4 px = pick_color(nb, cp);

    if (ew_iw.x < ew_iw.y && sub_cond_3(nb, cp)) {
        bool left, up;
        detect_direction(nb, cp, left, up);
        if (left && up) {
            vec4 bh = blend(side_h, px, B192);
            vec4 bf = blend(side_f, px, B64);
            side_h = bh; side_f = bh;
            corner = px;
            // Note: ext_h and ext_f would get bf, but we skip those
            // since we only return corner + two adjacent sides
        } else if (left) {
            side_h = blend(side_h, px, B192);
            side_f = blend(side_f, px, B64);
            corner = px;
        } else if (up) {
            side_f = blend(side_f, px, B192);
            side_h = blend(side_h, px, B64);
            corner = px;
        } else {
            corner = blend(corner, px, B224);
            side_f = blend(side_f, px, B32);
            side_h = blend(side_h, px, B32);
        }
    } else {
        corner = blend(corner, px, B128);
    }
}

// ── Per-corner blending (4x) ────────────────────────────────────────────────
// Modifies specific pixels in a 4x4 output grid.

void filt4x(vec4 nb[25], int cp[11], vec4 e,
            inout vec4 out_block[16], int n15, int n14, int n13, int n12,
            int n11, int n10, int n7, int n3) {
    if (nb[12] == nb[cp[1]] || nb[12] == nb[cp[0]]) return;

    vec2 ew_iw = edge_weights(nb, cp);
    if (ew_iw.x > ew_iw.y) return;

    vec4 px = pick_color(nb, cp);

    if (ew_iw.x < ew_iw.y && sub_cond_24(nb, cp)) {
        bool left, up;
        detect_direction(nb, cp, left, up);
        if (left && up) {
            vec4 b13 = blend(out_block[n13], px, B192);
            vec4 b12 = blend(out_block[n12], px, B64);
            out_block[n13] = b13;
            out_block[n12] = b12;
            out_block[n15] = px;
            out_block[n14] = px;
            out_block[n11] = px;
            out_block[n10] = b12;
            out_block[n3]  = b12;
            out_block[n7]  = b13;
        } else if (left) {
            out_block[n11] = blend(out_block[n11], px, B192);
            out_block[n13] = blend(out_block[n13], px, B192);
            out_block[n10] = blend(out_block[n10], px, B64);
            out_block[n12] = blend(out_block[n12], px, B64);
            out_block[n14] = px;
            out_block[n15] = px;
        } else if (up) {
            out_block[n14] = blend(out_block[n14], px, B192);
            out_block[n7]  = blend(out_block[n7],  px, B192);
            out_block[n10] = blend(out_block[n10], px, B64);
            out_block[n3]  = blend(out_block[n3],  px, B64);
            out_block[n11] = px;
            out_block[n15] = px;
        } else {
            out_block[n11] = blend(out_block[n11], px, B128);
            out_block[n14] = blend(out_block[n14], px, B128);
            out_block[n15] = px;
        }
    } else {
        out_block[n15] = blend(out_block[n15], px, B128);
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

void main() {
    vec2 src_coord = v_uv * src_size;
    ivec2 src_i = clamp(ivec2(floor(src_coord)), ivec2(0), ivec2(src_size) - 1);
    vec2 frac_val = src_coord - vec2(src_i);
    int iscale = int(scale);
    ivec2 sub = clamp(ivec2(floor(frac_val * scale)), ivec2(0), ivec2(iscale - 1));

    // Sample 5x5 neighborhood
    vec4 nb[25];
    for (int j = 0; j < 5; j++) {
        for (int i = 0; i < 5; i++) {
            nb[j * 5 + i] = fetch(src_i + ivec2(i - 2, j - 2));
        }
    }

    vec4 e = nb[12]; // center pixel

    if (iscale == 2) {
        // 2x: each output pixel is one corner
        int cp[11];
        if (sub.x == 0 && sub.y == 0) cp = CP_TL;
        else if (sub.x == 1 && sub.y == 0) cp = CP_TR;
        else if (sub.x == 0 && sub.y == 1) cp = CP_BL;
        else cp = CP_BR;

        frag_color = filt2x(nb, cp, e);

    } else if (iscale == 3) {
        // 3x: center pixel unchanged, corners and edges computed
        if (sub.x == 1 && sub.y == 1) {
            frag_color = e;
        } else {
            // Determine which corner governs this sub-pixel
            int cx = sub.x < 1 ? 0 : (sub.x > 1 ? 1 : -1);
            int cy = sub.y < 1 ? 0 : (sub.y > 1 ? 1 : -1);

            // For edge pixels (cx or cy == -1), find the closest corner
            if (cx == -1) cx = 0; // center column → use left corner
            if (cy == -1) cy = 0; // center row → use top corner

            int cp[11];
            if (cx == 0 && cy == 0) cp = CP_TL;
            else if (cx == 1 && cy == 0) cp = CP_TR;
            else if (cx == 0 && cy == 1) cp = CP_BL;
            else cp = CP_BR;

            // Compute the 3x3 sub-block for this corner
            vec4 corner = e, side_h = e, side_f = e;
            filt3x(nb, cp, e, corner, side_h, side_f);

            // Map sub-pixel to corner/side_h/side_f
            bool at_corner = (sub.x != 1 && sub.y != 1);
            bool at_side_h = (sub.x == 1); // horizontal edge
            bool at_side_f = (sub.y == 1); // vertical edge

            if (at_corner) frag_color = corner;
            else if (at_side_h) frag_color = side_h;
            else frag_color = side_f;
        }

    } else {
        // 4x: compute full 16-pixel block, pick the right sub-pixel
        vec4 out_block[16];
        for (int i = 0; i < 16; i++) out_block[i] = e;

        // BR corner
        filt4x(nb, CP_BR, e, out_block, 15,14,13,12, 11,10, 7, 3);
        // BL corner
        filt4x(nb, CP_BL, e, out_block, 12,13,14,15,  8, 9, 4, 0);
        // TR corner
        filt4x(nb, CP_TR, e, out_block,  3, 2, 1, 0,  7, 6,11,15);
        // TL corner
        filt4x(nb, CP_TL, e, out_block,  0, 1, 2, 3,  4, 5, 8,12);

        frag_color = out_block[sub.y * 4 + sub.x];
    }
}
