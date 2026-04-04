// xBRZ pixel-art scaling (2x-6x) — GPU compute implementation.
//
// Implementation of the xBRZ algorithm by Zenju. Single-pass compute shader:
// each thread computes one output pixel by analyzing its source pixel's 4x4
// neighborhood, detecting edges, and applying scale-specific blend patterns.
//
// Blend patterns are encoded in flat lookup tables indexed by
// (scale, edge_type, sub_pixel_index), avoiding per-scale branching.

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

// ── Constants ───────────────────────────────────────────────────────────────

static const float DOMINANT_THRESHOLD = 3.6;
static const float STEEP_THRESHOLD = 2.2;
static const float EQUAL_THRESHOLD = 30.0;

static const int EDGE_NONE = 0;
static const int EDGE_NORMAL = 1;
static const int EDGE_DOMINANT = 2;

// ── Helpers ─────────────────────────────────────────────────────────────────

uint read_raw(int2 coord) { return px_raw(pixels, int(u.src_w), int(u.src_h), coord); }
float4 read_pixel(int2 coord) { return (read_raw(coord)).unpack_rgba(); }
float color_dist(float4 a, float4 b) { return ycbcr_dist(a, b); }
bool colors_equal(float4 a, float4 b) { return color_dist(a, b) < EQUAL_THRESHOLD; }

// ── Edge info packing (4 corners × 2 bits) ─────────────────────────────────

int read_edge(int packed, int corner) { return (packed >> (corner * 2)) & 3; }

int rotate_edges(int packed, int rot) {
    if (rot == 0) return packed;
    int shift = (rot * 2) & 7;
    return ((packed >> shift) | (packed << (8 - shift))) & 0xFF;
}

// ── Edge analysis ───────────────────────────────────────────────────────────

int analyze_junction(int2 bxy) {
    uint raw_f = read_raw(bxy);
    uint raw_g = read_raw(bxy + int2(1, 0));
    uint raw_j = read_raw(bxy + int2(0, 1));
    uint raw_k = read_raw(bxy + int2(1, 1));

    if ((raw_f == raw_g && raw_j == raw_k) || (raw_f == raw_j && raw_g == raw_k))
        return 0;

    float4 f = (raw_f).unpack_rgba();
    float4 g = (raw_g).unpack_rgba();
    float4 j = (raw_j).unpack_rgba();
    float4 k = (raw_k).unpack_rgba();

    float4 bp = read_pixel(bxy + int2(0, -1));
    float4 c  = read_pixel(bxy + int2(1, -1));
    float4 e  = read_pixel(bxy + int2(-1, 0));
    float4 hp = read_pixel(bxy + int2(2, 0));
    float4 i  = read_pixel(bxy + int2(-1, 1));
    float4 l  = read_pixel(bxy + int2(2, 1));
    float4 n  = read_pixel(bxy + int2(0, 2));
    float4 o  = read_pixel(bxy + int2(1, 2));

    float d_jg = color_dist(i, f) + color_dist(f, c) + color_dist(n, k) + color_dist(k, hp) + 4.0 * color_dist(j, g);
    float d_fk = color_dist(e, j) + color_dist(j, o) + color_dist(bp, g) + color_dist(g, l) + 4.0 * color_dist(f, k);

    int result = 0;
    if (d_jg < d_fk) {
        int s = (DOMINANT_THRESHOLD * d_jg < d_fk) ? EDGE_DOMINANT : EDGE_NORMAL;
        if (raw_f != raw_g && raw_f != raw_j) result |= (s << 4);
        if (raw_k != raw_j && raw_k != raw_g) result |= s;
    } else if (d_fk < d_jg) {
        int s = (DOMINANT_THRESHOLD * d_fk < d_jg) ? EDGE_DOMINANT : EDGE_NORMAL;
        if (raw_j != raw_f && raw_j != raw_k) result |= (s << 2);
        if (raw_g != raw_f && raw_g != raw_k) result |= (s << 6);
    }
    return result;
}

int gather_edges(int2 sxy) {
    int info = 0;
    info |= (read_edge(analyze_junction(sxy), 2) << 4);
    info |= (read_edge(analyze_junction(sxy + int2(-1, 0)), 3) << 6);
    info |= (read_edge(analyze_junction(sxy + int2(0, -1)), 1) << 2);
    info |= read_edge(analyze_junction(sxy + int2(-1, -1)), 0);
    return info;
}

// ── Edge classification ─────────────────────────────────────────────────────

bool should_blend(int rotated_info, float4 center,
                  float4 g, float4 hp, float4 i, float4 f, float4 c) {
    if (read_edge(rotated_info, 2) >= EDGE_DOMINANT) return true;
    if (read_edge(rotated_info, 1) != EDGE_NONE && !colors_equal(center, g)) return false;
    if (read_edge(rotated_info, 3) != EDGE_NONE && !colors_equal(center, c)) return false;
    if (!colors_equal(center, i) && colors_equal(g, hp) && colors_equal(hp, i)
        && colors_equal(i, f) && colors_equal(f, c)) return false;
    return true;
}

// Edge direction: 0=shallow, 1=steep, 2=steep+shallow, 3=diagonal, 4=corner
static const int DIR_SHALLOW = 0;
static const int DIR_STEEP = 1;
static const int DIR_BOTH = 2;
static const int DIR_DIAGONAL = 3;
static const int DIR_CORNER = 4;

// ── Blend pattern tables ────────────────────────────────────────────────────
//
// Encoded as flat arrays of (row, col, weight*1000) triples. A sentinel
// row value of 99 marks the end. Each scale/direction combination has a
// contiguous run in one of the table arrays.
//
// Row/col are relative to bottom-right corner (rotation 0).

// Table entry: row, col, weight (scaled by 1000 to store as int)
struct PatternEntry { int r; int c; int w; };

// Sentinel value marking end of a pattern
static const int END = 99;

// Scale 2
static const PatternEntry PAT_2_SHALLOW[3]   = { {1,0,250}, {1,1,750}, {END,0,0} };
static const PatternEntry PAT_2_STEEP[3]     = { {0,1,250}, {1,1,750}, {END,0,0} };
static const PatternEntry PAT_2_BOTH[4]      = { {1,0,250}, {0,1,250}, {1,1,833}, {END,0,0} };
static const PatternEntry PAT_2_DIAG[2]      = { {1,1,500}, {END,0,0} };
static const PatternEntry PAT_2_CORNER[2]    = { {1,1,210}, {END,0,0} };

// Scale 3
static const PatternEntry PAT_3_SHALLOW[4]   = { {2,0,250}, {2,1,750}, {2,2,1000}, {END,0,0} };
static const PatternEntry PAT_3_STEEP[4]     = { {0,2,250}, {1,2,750}, {2,2,1000}, {END,0,0} };
static const PatternEntry PAT_3_BOTH[6]      = { {2,0,250}, {0,2,250}, {2,1,750}, {1,2,750}, {2,2,1000}, {END,0,0} };
static const PatternEntry PAT_3_DIAG[4]      = { {1,2,125}, {2,1,125}, {2,2,875}, {END,0,0} };
static const PatternEntry PAT_3_CORNER[2]    = { {2,2,450}, {END,0,0} };

// Scale 4
static const PatternEntry PAT_4_SHALLOW[7]   = { {3,0,250}, {2,2,250}, {3,1,750}, {2,3,750}, {3,2,1000}, {3,3,1000}, {END,0,0} };
static const PatternEntry PAT_4_STEEP[7]     = { {0,3,250}, {2,2,250}, {1,3,750}, {3,2,750}, {2,3,1000}, {3,3,1000}, {END,0,0} };
static const PatternEntry PAT_4_BOTH[9]      = { {3,1,750}, {1,3,750}, {3,0,250}, {0,3,250}, {2,2,333}, {3,2,1000}, {2,3,1000}, {3,3,1000}, {END,0,0} };
static const PatternEntry PAT_4_DIAG[4]      = { {3,2,500}, {2,3,500}, {3,3,1000}, {END,0,0} };
static const PatternEntry PAT_4_CORNER[4]    = { {3,3,680}, {3,2,90}, {2,3,90}, {END,0,0} };

// Scale 5
static const PatternEntry PAT_5_SHALLOW[10]  = { {4,0,250}, {3,2,250}, {2,4,250}, {4,1,750}, {3,3,750}, {4,2,1000}, {4,3,1000}, {4,4,1000}, {3,4,1000}, {END,0,0} };
static const PatternEntry PAT_5_STEEP[10]    = { {0,4,250}, {2,3,250}, {4,2,250}, {1,4,750}, {3,3,750}, {2,4,1000}, {3,4,1000}, {4,4,1000}, {4,3,1000}, {END,0,0} };
static const PatternEntry PAT_5_BOTH[13]     = { {0,4,250}, {2,3,250}, {1,4,750}, {4,0,250}, {3,2,250}, {4,1,750}, {3,3,667}, {2,4,1000}, {3,4,1000}, {4,4,1000}, {4,2,1000}, {4,3,1000}, {END,0,0} };
static const PatternEntry PAT_5_DIAG[7]      = { {4,2,125}, {3,3,125}, {2,4,125}, {4,3,875}, {3,4,875}, {4,4,1000}, {END,0,0} };
static const PatternEntry PAT_5_CORNER[4]    = { {4,4,860}, {3,4,230}, {4,3,230}, {END,0,0} };

// Scale 6
static const PatternEntry PAT_6_SHALLOW[13]  = { {5,0,250}, {4,2,250}, {3,4,250}, {5,1,750}, {4,3,750}, {3,5,750}, {5,2,1000}, {5,3,1000}, {5,4,1000}, {5,5,1000}, {4,4,1000}, {4,5,1000}, {END,0,0} };
static const PatternEntry PAT_6_STEEP[13]    = { {0,5,250}, {2,4,250}, {4,3,250}, {1,5,750}, {3,4,750}, {5,3,750}, {2,5,1000}, {3,5,1000}, {4,5,1000}, {5,5,1000}, {4,4,1000}, {5,4,1000}, {END,0,0} };
static const PatternEntry PAT_6_BOTH[17]     = { {0,5,250}, {2,4,250}, {1,5,750}, {3,4,750}, {5,0,250}, {4,2,250}, {5,1,750}, {4,3,750}, {2,5,1000}, {3,5,1000}, {4,5,1000}, {5,5,1000}, {4,4,1000}, {5,4,1000}, {5,2,1000}, {5,3,1000}, {END,0,0} };
static const PatternEntry PAT_6_DIAG[7]      = { {5,3,500}, {4,4,500}, {3,5,500}, {4,5,1000}, {5,4,1000}, {5,5,1000}, {END,0,0} };
static const PatternEntry PAT_6_CORNER[6]    = { {5,5,970}, {4,5,420}, {5,4,420}, {3,5,60}, {5,3,60}, {END,0,0} };

// ── Pattern lookup ──────────────────────────────────────────────────────────

// Look up the blend weight for a given (row, col) in a pattern table.
// Returns 0 if the sub-pixel is not in the pattern.
float lookup_weight(int r, int c, int scale, int dir) {
    // There's no good way to do indirect array indexing in Slang for static
    // const arrays, so we use a flat switch to select the right table.

    // Macro to search a pattern table for matching (r,c)
    #define SEARCH(PAT) { \
        for (int idx = 0; PAT[idx].r != END; idx++) { \
            if (PAT[idx].r == r && PAT[idx].c == c) \
                return float(PAT[idx].w) / 1000.0; \
        } \
        return 0.0; \
    }

    if (scale == 2) {
        if (dir == DIR_SHALLOW)  SEARCH(PAT_2_SHALLOW)
        if (dir == DIR_STEEP)    SEARCH(PAT_2_STEEP)
        if (dir == DIR_BOTH)     SEARCH(PAT_2_BOTH)
        if (dir == DIR_DIAGONAL) SEARCH(PAT_2_DIAG)
        SEARCH(PAT_2_CORNER)
    }
    if (scale == 3) {
        if (dir == DIR_SHALLOW)  SEARCH(PAT_3_SHALLOW)
        if (dir == DIR_STEEP)    SEARCH(PAT_3_STEEP)
        if (dir == DIR_BOTH)     SEARCH(PAT_3_BOTH)
        if (dir == DIR_DIAGONAL) SEARCH(PAT_3_DIAG)
        SEARCH(PAT_3_CORNER)
    }
    if (scale == 4) {
        if (dir == DIR_SHALLOW)  SEARCH(PAT_4_SHALLOW)
        if (dir == DIR_STEEP)    SEARCH(PAT_4_STEEP)
        if (dir == DIR_BOTH)     SEARCH(PAT_4_BOTH)
        if (dir == DIR_DIAGONAL) SEARCH(PAT_4_DIAG)
        SEARCH(PAT_4_CORNER)
    }
    if (scale == 5) {
        if (dir == DIR_SHALLOW)  SEARCH(PAT_5_SHALLOW)
        if (dir == DIR_STEEP)    SEARCH(PAT_5_STEEP)
        if (dir == DIR_BOTH)     SEARCH(PAT_5_BOTH)
        if (dir == DIR_DIAGONAL) SEARCH(PAT_5_DIAG)
        SEARCH(PAT_5_CORNER)
    }
    // scale == 6
    if (dir == DIR_SHALLOW)  SEARCH(PAT_6_SHALLOW)
    if (dir == DIR_STEEP)    SEARCH(PAT_6_STEEP)
    if (dir == DIR_BOTH)     SEARCH(PAT_6_BOTH)
    if (dir == DIR_DIAGONAL) SEARCH(PAT_6_DIAG)
    SEARCH(PAT_6_CORNER)

    #undef SEARCH
}

// ── Rotation ────────────────────────────────────────────────────────────────

int2 rotate_subpixel(int row, int col, int n, int rot) {
    int m = n - 1;
    if (rot == 0) return int2(row, col);
    if (rot == 1) return int2(row, m - col);
    if (rot == 2) return int2(m - row, m - col);
    return int2(m - row, col);
}

int2 corner_direction(int rot) {
    if (rot == 0) return int2(1, 1);
    if (rot == 1) return int2(-1, 1);
    if (rot == 2) return int2(-1, -1);
    return int2(1, -1);
}

// ── Entry point ─────────────────────────────────────────────────────────────

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    int2 pos = int2(tid.xy);
    if (pos.x >= int(u.out_w) || pos.y >= int(u.out_h)) return;

    int n = int(u.iscale);
    int2 src_pos = pos / n;
    int2 sub = pos - src_pos * n;

    float4 center = read_pixel(src_pos);
    int edge_info = gather_edges(src_pos);

    if (edge_info == 0) {
        output_img.Store(pos, center);
        return;
    }

    float4 result = center;

    for (int rot = 0; rot < 4; rot++) {
        int rotated = rotate_edges(edge_info, rot);
        if (read_edge(rotated, 2) == EDGE_NONE) continue;

        int2 rs = rotate_subpixel(sub.y, sub.x, n, rot);
        int2 d = corner_direction(rot);

        float4 f_nb = read_pixel(src_pos + int2(d.x, 0));
        float4 h_nb = read_pixel(src_pos + int2(0, d.y));
        float4 c_nb = read_pixel(src_pos + int2(d.x, -d.y));
        float4 g_nb = read_pixel(src_pos + int2(-d.x, d.y));
        float4 i_nb = read_pixel(src_pos + int2(d.x, d.y));

        float4 target = (color_dist(center, f_nb) <= color_dist(center, h_nb)) ? f_nb : h_nb;

        int dir;
        if (!should_blend(rotated, center, g_nb, h_nb, i_nb, f_nb, c_nb)) {
            dir = DIR_CORNER;
        } else {
            float4 d_nb = read_pixel(src_pos + int2(-d.x, 0));
            float4 b_nb = read_pixel(src_pos + int2(0, -d.y));
            float fg = color_dist(f_nb, g_nb);
            float hc = color_dist(h_nb, c_nb);
            bool is_shallow = STEEP_THRESHOLD * fg <= hc && !colors_equal(center, g_nb) && !colors_equal(d_nb, g_nb);
            bool is_steep   = STEEP_THRESHOLD * hc <= fg && !colors_equal(center, c_nb) && !colors_equal(b_nb, c_nb);

            dir = (is_steep && is_shallow) ? DIR_BOTH :
                  is_shallow ? DIR_SHALLOW :
                  is_steep ? DIR_STEEP : DIR_DIAGONAL;
        }

        float w = lookup_weight(rs.x, rs.y, n, dir);
        if (w > 0.0) {
            result = lerp(result, target, w);
        }
    }

    output_img.Store(pos, result);
}
