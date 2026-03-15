#version 450

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

bool colors_differ(vec4 a, vec4 b) {
    vec3 d = (a.rgb - b.rgb) * 255.0;
    // BT.601 YUV with individual thresholds (matching CPU hqx.rs)
    float dy = abs(0.299 * d.r + 0.587 * d.g + 0.114 * d.b);
    float du = abs(-0.169 * d.r - 0.331 * d.g + 0.500 * d.b);
    float dv = abs(0.500 * d.r - 0.419 * d.g - 0.081 * d.b);
    return dy > 48.0 || du > 7.0 || dv > 6.0;
}

vec4 mix2(vec4 c1, float w1, vec4 c2, float w2) {
    return vec4((c1.rgb * w1 + c2.rgb * w2) / (w1 + w2), 1.0);
}

vec4 mix3(vec4 c1, float w1, vec4 c2, float w2, vec4 c3, float w3) {
    return vec4((c1.rgb * w1 + c2.rgb * w2 + c3.rgb * w3) / (w1 + w2 + w3), 1.0);
}

// ── Corner view ─────────────────────────────────────────────────────────────
// A rotated/mirrored view of the 3x3 neighborhood for one corner.
// Named fields match the CPU CornerView struct.

struct CornerView {
    int edge;   // shuffled 8-bit edge pattern
    vec4 diag;  // corner diagonal
    vec4 top;   // top neighbor
    vec4 left;  // left neighbor
    vec4 center;// center pixel
    vec4 right; // right neighbor
    vec4 bot;   // bottom neighbor
};

// Symmetry transforms (index permutations for 4 corners)
const int TRANSFORMS[4*9] = int[](
    0,1,2,3,4,5,6,7,8,  // top-left (identity)
    2,1,0,5,4,3,8,7,6,  // top-right (flip x)
    6,7,8,3,4,5,0,1,2,  // bottom-left (flip y)
    8,7,6,5,4,3,2,1,0   // bottom-right (flip both)
);

// Rotations for hq3x edge pixels
const int ROTATIONS[4*9] = int[](
    0,1,2,3,4,5,6,7,8,  // 0 degrees
    2,5,8,1,4,7,0,3,6,  // 90 CW
    6,3,0,7,4,1,8,5,2,  // 90 CCW
    8,7,6,5,4,3,2,1,0   // 180
);

int remap_edge_pattern(int pattern, int perm_offset, bool reverse) {
    const int indices[8] = int[](0,1,2,3,5,6,7,8);
    int result = 0;
    for (int i = 0; i < 8; i++) {
        int idx = indices[i];
        int src_idx = (perm_offset >= 0) ?
            TRANSFORMS[perm_offset + idx] : ROTATIONS[(-perm_offset - 1) + idx];
        int src_bit = src_idx > 4 ? src_idx - 1 : src_idx;
        int actual_bit = reverse ? 7 - src_bit : src_bit;
        result |= ((pattern >> actual_bit) & 1) << i;
    }
    return result;
}

CornerView build_view(vec4 nb[9], int pattern, int perm_offset, bool reverse) {
    int edge = remap_edge_pattern(pattern, perm_offset, reverse);
    int p0, p1, p3, p4, p5, p7;
    if (perm_offset >= 0) {
        p0 = TRANSFORMS[perm_offset]; p1 = TRANSFORMS[perm_offset+1];
        p3 = TRANSFORMS[perm_offset+3]; p4 = TRANSFORMS[perm_offset+4];
        p5 = TRANSFORMS[perm_offset+5]; p7 = TRANSFORMS[perm_offset+7];
    } else {
        int off = -perm_offset - 1;
        p0 = ROTATIONS[off]; p1 = ROTATIONS[off+1];
        p3 = ROTATIONS[off+3]; p4 = ROTATIONS[off+4];
        p5 = ROTATIONS[off+5]; p7 = ROTATIONS[off+7];
    }
    return CornerView(edge, nb[p0], nb[p1], nb[p3], nb[p4], nb[p5], nb[p7]);
}

// ── Edge pattern helpers ────────────────────────────────────────────────────

bool em(CornerView v, int mask, int expected) { return (v.edge & mask) == expected; }

bool edge_horiz(CornerView v) { return em(v,0xbf,0x37) || em(v,0xdb,0x13); }
bool edge_vert(CornerView v) { return em(v,0xdb,0x49) || em(v,0xef,0x6d); }
bool edge_sharp(CornerView v) { return em(v,0x0b,0x0b) || em(v,0xfe,0x4a) || em(v,0xfe,0x1a); }

bool edge_partial(CornerView v) {
    return em(v,0x6f,0x2a)||em(v,0x5b,0x0a)||em(v,0xbf,0x3a)||em(v,0xdf,0x5a)||
           em(v,0x9f,0x8a)||em(v,0xcf,0x8a)||em(v,0xef,0x4e)||em(v,0x3f,0x0e)||
           em(v,0xfb,0x5a)||em(v,0xbb,0x8a)||em(v,0x7f,0x5a)||em(v,0xaf,0x8a)||em(v,0xeb,0x8a);
}

// ── hq2x corner ─────────────────────────────────────────────────────────────

vec4 hq2x_corner(CornerView v) {
    vec4 D=v.diag, T=v.top, L=v.left, C=v.center, R=v.right, B=v.bot;
    if (edge_horiz(v) && colors_differ(T,R)) return mix2(C,3.0,L,1.0);
    if (edge_vert(v) && colors_differ(B,L)) return mix2(C,3.0,T,1.0);
    if (edge_sharp(v) && colors_differ(L,T)) return C;
    if (edge_partial(v) && colors_differ(L,T)) return mix2(C,3.0,D,1.0);
    if (em(v,0x0b,0x08)) return mix3(C,2.0,D,1.0,T,1.0);
    if (em(v,0x0b,0x02)) return mix3(C,2.0,D,1.0,L,1.0);
    if (em(v,0x2f,0x2f)) return mix3(C,14.0,L,1.0,T,1.0);
    if (edge_horiz(v)) return mix3(C,5.0,T,2.0,L,1.0);
    if (edge_vert(v)) return mix3(C,5.0,L,2.0,T,1.0);
    if (em(v,0x1b,0x03)||em(v,0x4f,0x43)||em(v,0x8b,0x83)||em(v,0x6b,0x43)) return mix2(C,3.0,L,1.0);
    if (em(v,0x4b,0x09)||em(v,0x8b,0x89)||em(v,0x1f,0x19)||em(v,0x3b,0x19)) return mix2(C,3.0,T,1.0);
    if (em(v,0x7e,0x2a)||em(v,0xef,0xab)||em(v,0xbf,0x8f)||em(v,0x7e,0x0e)) return mix3(C,2.0,L,3.0,T,3.0);
    if (em(v,0xfb,0x6a)||em(v,0x6f,0x6e)||em(v,0x3f,0x3e)||em(v,0xfb,0xfa)||em(v,0xdf,0xde)||em(v,0xdf,0x1e)) return mix2(C,3.0,D,1.0);
    if (em(v,0x0a,0x00)||em(v,0x4f,0x4b)||em(v,0x9f,0x1b)||em(v,0x2f,0x0b)||em(v,0xbe,0x0a)||em(v,0xee,0x0a)||em(v,0x7e,0x0a)||em(v,0xeb,0x4b)||em(v,0x3b,0x1b)) return mix3(C,2.0,L,1.0,T,1.0);
    return mix3(C,6.0,L,1.0,T,1.0);
}

// ── hq3x corner + edge ──────────────────────────────────────────────────────

void hq3x_corner_and_edge(CornerView v, out vec4 corner, out vec4 edge) {
    vec4 D=v.diag, T=v.top, L=v.left, C=v.center, R=v.right, B=v.bot;

    if (edge_vert(v) && colors_differ(B,L)) corner = mix2(C,3.0,T,1.0);
    else if (edge_horiz(v) && colors_differ(T,R)) corner = mix2(C,3.0,L,1.0);
    else if (edge_sharp(v) && colors_differ(L,T)) corner = C;
    else if (edge_partial(v) && colors_differ(L,T)) corner = mix2(C,3.0,D,1.0);
    else if (em(v,0x4b,0x09)||em(v,0x8b,0x89)||em(v,0x1f,0x19)||em(v,0x3b,0x19)) corner = mix2(C,3.0,T,1.0);
    else if (em(v,0x1b,0x03)||em(v,0x4f,0x43)||em(v,0x8b,0x83)||em(v,0x6b,0x43)) corner = mix2(C,3.0,L,1.0);
    else if (em(v,0x7e,0x2a)||em(v,0xef,0xab)||em(v,0xbf,0x8f)||em(v,0x7e,0x0e)) corner = mix2(L,1.0,T,1.0);
    else if (em(v,0x4f,0x4b)||em(v,0x9f,0x1b)||em(v,0x2f,0x0b)||em(v,0xbe,0x0a)||em(v,0xee,0x0a)||em(v,0x7e,0x0a)||em(v,0xeb,0x4b)||em(v,0x3b,0x1b)) corner = mix3(C,2.0,L,7.0,T,7.0);
    else if (em(v,0x0b,0x08)||em(v,0xf9,0x68)||em(v,0xf3,0x62)||em(v,0x6d,0x6c)||em(v,0x67,0x66)||em(v,0x3d,0x3c)||em(v,0x37,0x36)||em(v,0xf9,0xf8)||em(v,0xdd,0xdc)||em(v,0xf3,0xf2)||em(v,0xd7,0xd6)||em(v,0xdd,0x1c)||em(v,0xd7,0x16)||em(v,0x0b,0x02)) corner = mix2(C,3.0,D,1.0);
    else corner = mix3(C,2.0,L,1.0,T,1.0);

    if ((em(v,0xfe,0xde)||em(v,0x9e,0x16)||em(v,0xda,0x12)||em(v,0x17,0x16)||em(v,0x5b,0x12)||em(v,0xbb,0x12)) && colors_differ(T,R)) edge = C;
    else if ((em(v,0x0f,0x0b)||em(v,0x5e,0x0a)||em(v,0xfb,0x7b)||em(v,0x3b,0x0b)||em(v,0xbe,0x0a)||em(v,0x7a,0x0a)) && colors_differ(L,T)) edge = C;
    else if (em(v,0xbf,0x8f)||em(v,0x7e,0x0e)||em(v,0xbf,0x37)||em(v,0xdb,0x13)) edge = mix2(T,3.0,C,1.0);
    else if (em(v,0x02,0x00)||em(v,0x7c,0x28)||em(v,0xed,0xa9)||em(v,0xf5,0xb4)||em(v,0xd9,0x90)) edge = mix2(C,3.0,T,1.0);
    else if (em(v,0x4f,0x4b)||em(v,0xfb,0x7b)||em(v,0xfe,0x7e)||em(v,0x9f,0x1b)||em(v,0x2f,0x0b)||em(v,0xbe,0x0a)||em(v,0x7e,0x0a)||em(v,0xfb,0x4b)||em(v,0xfb,0xdb)||em(v,0xfe,0xde)||em(v,0xfe,0x56)||em(v,0x57,0x56)||em(v,0x97,0x16)||em(v,0x3f,0x1e)||em(v,0xdb,0x12)||em(v,0xbb,0x12)) edge = mix2(C,7.0,T,1.0);
    else edge = C;
}

// ── hq4x quadrant ───────────────────────────────────────────────────────────

void hq4x_quadrant(CornerView v, out vec4 px00, out vec4 px01, out vec4 px10, out vec4 px11) {
    vec4 D=v.diag, T=v.top, L=v.left, C=v.center, R=v.right, B=v.bot;
    bool h_edge = edge_horiz(v) && colors_differ(T,R);
    bool v_edge = edge_vert(v) && colors_differ(B,L);
    bool p_diag = edge_partial(v) && colors_differ(L,T);
    bool h_vert = edge_vert(v), h_horiz = edge_horiz(v);
    bool l_edge = em(v,0x1b,0x03)||em(v,0x4f,0x43)||em(v,0x8b,0x83)||em(v,0x6b,0x43);
    bool t_edge = em(v,0x4b,0x09)||em(v,0x8b,0x89)||em(v,0x1f,0x19)||em(v,0x3b,0x19);
    bool n_diag = em(v,0x0b,0x08)||em(v,0xf9,0x68)||em(v,0xf3,0x62)||em(v,0x6d,0x6c)||em(v,0x67,0x66)||em(v,0x3d,0x3c)||em(v,0x37,0x36)||em(v,0xf9,0xf8)||em(v,0xdd,0xdc)||em(v,0xf3,0xf2)||em(v,0xd7,0xd6)||em(v,0xdd,0x1c)||em(v,0xd7,0x16)||em(v,0x0b,0x02);
    bool sharp = edge_sharp(v) && colors_differ(L,T);
    bool cross = (em(v,0x0f,0x0b)||em(v,0x2b,0x0b)||em(v,0xfe,0x4a)||em(v,0xfe,0x1a)) && colors_differ(L,T);
    bool a_diff = em(v,0x2f,0x2f);
    bool smth = em(v,0x0a,0x00);
    bool i_top = em(v,0x0b,0x09);
    bool da = em(v,0x7e,0x2a)||em(v,0xef,0xab);
    bool db = em(v,0xbf,0x8f)||em(v,0x7e,0x0e);
    bool grad = em(v,0x4f,0x4b)||em(v,0x9f,0x1b)||em(v,0x2f,0x0b)||em(v,0xbe,0x0a)||em(v,0xee,0x0a)||em(v,0x7e,0x0a)||em(v,0xeb,0x4b)||em(v,0x3b,0x1b);
    bool i_left = em(v,0x0b,0x03);

    if (h_edge) px00=mix2(C,5.0,L,3.0); else if (v_edge) px00=mix2(C,5.0,T,3.0); else if (sharp) px00=C; else if (p_diag) px00=mix2(C,5.0,D,3.0); else if (h_vert) px00=mix2(C,3.0,L,1.0); else if (h_horiz) px00=mix2(C,3.0,T,1.0); else if (l_edge) px00=mix2(C,5.0,L,3.0); else if (t_edge) px00=mix2(C,5.0,T,3.0); else if (em(v,0x0f,0x0b)||em(v,0x5e,0x0a)||em(v,0x2b,0x0b)||em(v,0xbe,0x0a)||em(v,0x7a,0x0a)||em(v,0xee,0x0a)) px00=mix2(T,1.0,L,1.0); else if (n_diag) px00=mix2(C,5.0,D,3.0); else px00=mix3(C,2.0,T,1.0,L,1.0);

    if (h_edge) px01=mix2(C,7.0,L,1.0); else if (cross) px01=C; else if (p_diag) px01=mix2(C,3.0,D,1.0); else if (a_diff) px01=C; else if (smth) px01=mix3(C,5.0,T,2.0,L,1.0); else if (em(v,0x0b,0x08)) px01=mix3(C,5.0,T,2.0,D,1.0); else if (i_top) px01=mix2(C,5.0,T,3.0); else if (h_horiz) px01=mix2(T,3.0,C,1.0); else if (da) px01=mix3(T,2.0,C,1.0,L,1.0); else if (db) px01=mix2(T,5.0,L,3.0); else if (l_edge) px01=mix2(C,7.0,L,1.0); else if (em(v,0xf3,0x62)||em(v,0x67,0x66)||em(v,0x37,0x36)||em(v,0xf3,0xf2)||em(v,0xd7,0xd6)||em(v,0xd7,0x16)||em(v,0x0b,0x02)) px01=mix2(C,3.0,D,1.0); else if (grad) px01=mix2(T,1.0,C,1.0); else px01=mix2(C,3.0,T,1.0);

    if (v_edge) px10=mix2(C,7.0,T,1.0); else if (cross) px10=C; else if (p_diag) px10=mix2(C,3.0,D,1.0); else if (a_diff) px10=C; else if (smth) px10=mix3(C,5.0,L,2.0,T,1.0); else if (em(v,0x0b,0x02)) px10=mix3(C,5.0,L,2.0,D,1.0); else if (i_left) px10=mix2(C,5.0,L,3.0); else if (h_vert) px10=mix2(L,3.0,C,1.0); else if (db) px10=mix3(L,2.0,C,1.0,T,1.0); else if (da) px10=mix2(L,5.0,T,3.0); else if (t_edge) px10=mix2(C,7.0,T,1.0); else if (em(v,0x0b,0x08)||em(v,0xf9,0x68)||em(v,0x6d,0x6c)||em(v,0x3d,0x3c)||em(v,0xf9,0xf8)||em(v,0xdd,0xdc)||em(v,0xdd,0x1c)) px10=mix2(C,3.0,D,1.0); else if (grad) px10=mix2(L,1.0,C,1.0); else px10=mix2(C,3.0,L,1.0);

    if ((em(v,0x7f,0x2b)||em(v,0xef,0xab)||em(v,0xbf,0x8f)||em(v,0x7f,0x0f)) && colors_differ(L,T)) px11=C; else if (p_diag) px11=mix2(C,7.0,D,1.0); else if (i_left) px11=mix2(C,7.0,L,1.0); else if (i_top) px11=mix2(C,7.0,T,1.0); else if (smth||em(v,0x7e,0x2a)||em(v,0xef,0xab)||em(v,0xbf,0x8f)||em(v,0x7e,0x0e)) px11=mix3(C,6.0,L,1.0,T,1.0); else if (n_diag) px11=mix2(C,7.0,D,1.0); else px11=C;
}

// ── Main ────────────────────────────────────────────────────────────────────

void main() {
    vec2 src_coord = v_uv * src_size;
    ivec2 src_i = clamp(ivec2(floor(src_coord)), ivec2(0), ivec2(src_size) - 1);
    vec2 frac_val = src_coord - vec2(src_i);
    int iscale = int(scale);
    ivec2 sub = clamp(ivec2(floor(frac_val * scale)), ivec2(0), ivec2(iscale - 1));

    // Sample 3x3 neighborhood
    vec4 nb[9];
    nb[0]=fetch(src_i+ivec2(-1,-1)); nb[1]=fetch(src_i+ivec2(0,-1)); nb[2]=fetch(src_i+ivec2(1,-1));
    nb[3]=fetch(src_i+ivec2(-1, 0)); nb[4]=fetch(src_i);             nb[5]=fetch(src_i+ivec2(1, 0));
    nb[6]=fetch(src_i+ivec2(-1, 1)); nb[7]=fetch(src_i+ivec2(0, 1)); nb[8]=fetch(src_i+ivec2(1, 1));

    // Compute edge pattern
    int pat = 0;
    if (colors_differ(nb[4],nb[0])) pat |= 1;
    if (colors_differ(nb[4],nb[1])) pat |= 2;
    if (colors_differ(nb[4],nb[2])) pat |= 4;
    if (colors_differ(nb[4],nb[3])) pat |= 8;
    if (colors_differ(nb[4],nb[5])) pat |= 16;
    if (colors_differ(nb[4],nb[6])) pat |= 32;
    if (colors_differ(nb[4],nb[7])) pat |= 64;
    if (colors_differ(nb[4],nb[8])) pat |= 128;

    if (iscale == 2) {
        // hq2x: each output pixel is one corner, selected by sub-pixel position
        // Transform index: TL=0, TR=1, BL=2, BR=3
        int ti = sub.y * 2 + sub.x;
        CornerView v = build_view(nb, pat, ti * 9, false);
        frag_color = hq2x_corner(v);

    } else if (iscale == 3) {
        // hq3x: center pixel unchanged, corners and edges via rotated views.
        // Each rotation produces a (corner, edge) pair:
        //   ROT[0] false → c00(0,0) + e01(1,0)   top-left corner + top edge
        //   ROT[1] true  → c02(2,0) + e12(2,1)   top-right corner + right edge
        //   ROT[2] true  → c20(0,2) + e10(0,1)   bottom-left corner + left edge
        //   ROT[3] false → c22(2,2) + e21(1,2)   bottom-right corner + bottom edge
        if (sub.x == 1 && sub.y == 1) {
            frag_color = nb[4];
        } else {
            vec4 corner, edge;
            if (sub.y == 0 && sub.x <= 1) {
                // (0,0)=corner, (1,0)=edge — ROT[0], false
                hq3x_corner_and_edge(build_view(nb, pat, -1, false), corner, edge);
                frag_color = (sub.x == 0) ? corner : edge;
            } else if (sub.x == 2 && sub.y <= 1) {
                // (2,0)=corner, (2,1)=edge — ROT[1], true
                hq3x_corner_and_edge(build_view(nb, pat, -10, true), corner, edge);
                frag_color = (sub.y == 0) ? corner : edge;
            } else if (sub.x == 0 && sub.y >= 1) {
                // (0,2)=corner, (0,1)=edge — ROT[2], true
                hq3x_corner_and_edge(build_view(nb, pat, -19, true), corner, edge);
                frag_color = (sub.y == 2) ? corner : edge;
            } else {
                // (2,2)=corner, (1,2)=edge — ROT[3], false
                hq3x_corner_and_edge(build_view(nb, pat, -28, false), corner, edge);
                frag_color = (sub.x == 2) ? corner : edge;
            }
        }

    } else {
        // hq4x: each quadrant is a 2x2 block
        vec4 d00, d01, d10, d11;
        int qx = sub.x < 2 ? 0 : 1;
        int qy = sub.y < 2 ? 0 : 1;
        int ti = qy * 2 + qx;
        hq4x_quadrant(build_view(nb, pat, ti * 9, false), d00, d01, d10, d11);
        int lx = qx == 0 ? sub.x : 3 - sub.x;
        int ly = qy == 0 ? sub.y : 3 - sub.y;
        frag_color = (ly==0 && lx==0) ? d00 : (ly==0) ? d01 : (lx==0) ? d10 : d11;
    }
}
