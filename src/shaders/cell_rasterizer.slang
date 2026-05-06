// Tile-based nearest-boundary cell rasterizer with adaptive 2x2 AA.
//
// Each workgroup handles a 2x2 source-pixel tile. CPs in the tile's
// neighborhood are loaded into shared memory ONCE, then all output
// pixels in the tile evaluate against shared memory.
//
// Magic numbers:
//   2x2 source-pixel tile per workgroup — chosen to keep enough output
//     pixels per tile at typical scales (4x-12x) for the 256 threads to
//     amortize the shared-memory load cost.
//   numthreads(256, 1, 1) — workgroup size; loads CPs cooperatively then
//     splits output pixels across threads.
//   MAX_TILE_CPS = 26 — absolute upper bound on candidate CPs for a 2x2
//     tile: 18 base slots (3x3 unique cell corners x 2 slots) + 8
//     T-junction stem fan-out slots (one per non-center corner; center's
//     stem-bottom is always inside the 3x3 walk so no outward fan-out).
//   26 = 18+8 — CP scan slot count: 9 unique corners covering the tile's
//     4 cells (each cell's 4 corners) plus the per-T-junction fan-out
//     to catch stem-bottom CPs that sit outside the 3x3 walk.

import modules.color;
import modules.bspline;
import modules.vectorize;

struct Uniforms {
    uint img_w;
    uint img_h;
    uint out_w;
    uint out_h;
    float scale;
    uint corners_w;
    uint tiles_w;   // ceil(img_w / 2)
    uint tiles_h;   // ceil(img_h / 2)
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
[[vk::binding(1, 0)]]
#ifdef DXIL_TARGET
StructuredBuffer<float> cp_positions : register(t1, space0);
#else
StructuredBuffer<float> cp_positions;
#endif
[[vk::binding(2, 0)]]
#ifdef DXIL_TARGET
StructuredBuffer<float> orig_positions : register(t2, space0);
#else
StructuredBuffer<float> orig_positions;
#endif
[[vk::binding(3, 0)]]
#ifdef DXIL_TARGET
StructuredBuffer<uint> cp_flags : register(t3, space0);
#else
StructuredBuffer<uint> cp_flags;
#endif
[[vk::binding(4, 0)]]
#ifdef DXIL_TARGET
StructuredBuffer<int> cp_neighbors : register(t4, space0);
#else
StructuredBuffer<int> cp_neighbors;
#endif
[[vk::binding(1, 1)]]
#ifdef DXIL_TARGET
RWStructuredBuffer<float> crossing_t : register(u1, space1);
#else
RWStructuredBuffer<float> crossing_t;
#endif
[[vk::image_format("rgba8")]]
[[vk::binding(0, 1)]]
#ifdef DXIL_TARGET
WTexture2D<float4> output_img : register(u0, space1);
#else
WTexture2D<float4> output_img;
#endif

// Cache cap = absolute worst-case candidate count for the 26-slot walk:
// 18 base (3x3 unique corners x 2 slots) + 8 fan-out (every non-center
// corner has a T-junction whose stem-bottom sits outside the 3x3 walk).
static const uint MAX_TILE_CPS = 26u;

struct SlimCP {
    float2 pos;
    float2 prev_pos;
    float2 next_pos;
    float2 bbox_min;
    float2 bbox_max;
    int ci;
    // Branch threshold for prev_dir vs next_dir in resolve_color. For
    // non-clamped (interior) spans this is 0.5; for clamped Bezier spans
    // (Q0=prev_endpoint or Q2=next_endpoint), the parameterization is
    // shifted, so t_branch is the t value at which the clamped curve reaches
    // the same physical "before/after sc" boundary that the interior
    // B-spline reaches at t=0.5.
    float t_branch;
    // 2-CP chain — degenerate stem with both ends being endpoint markers.
    // The span is geometrically a line segment, so the rasterizer uses a
    // closed-form line-segment distance instead of bspline_refine,
    // sidestepping the cubic-solver's c3≈0 numerical instability.
    bool is_line;
};

groupshared SlimCP sh_cps[MAX_TILE_CPS];
groupshared Atomic<uint> sh_num_cps;

/// Closest-point on a 2-CP-chain span. The straight-line construction
/// p0=(3·a0-a1)/2, p1=midpoint, p2=(3·a1-a0)/2 gives B(t)=lerp(a0,a1,t),
/// with B(0)=midpoint(prev_pos,pos)=a0 and B(1)=midpoint(pos,next_pos)=a1.
/// Returns (t, d²) matching bspline_refine's signature.
float2 closest_on_chain_segment(SlimCP sc, float2 pt) {
    float2 a0 = 0.5 * (sc.prev_pos + sc.pos);
    float2 a1 = 0.5 * (sc.pos + sc.next_pos);
    float2 v = a1 - a0;
    float vv = dot(v, v);
    float t = (vv > 0.0) ? saturate(dot(pt - a0, v) / vv) : 0.0;
    float2 closest = a0 + t * v;
    float2 d = pt - closest;
    return float2(t, dot(d, d));
}

struct ResolveData {
    float2 orig_pos;
    float2 orig_prev;
    float2 orig_next;
    int prev_dir;
    int next_dir;
    uint icx, icy;
    int prev_ci;
    int next_ci;
};

float2 read_pos(int idx) {
    if (idx < 0) return float2(-1e10, -1e10);
    return float2(cp_positions[idx * 2], cp_positions[idx * 2 + 1]);
}

float2 read_orig(int idx) {
    if (idx < 0) return float2(-1e10, -1e10);
    return float2(orig_positions[idx * 2], orig_positions[idx * 2 + 1]);
}

ResolveData read_resolve_data(int ci) {
    ResolveData rd;
    rd.prev_ci = cp_neighbors[ci * 4];
    rd.next_ci = cp_neighbors[ci * 4 + 1];
    rd.prev_dir = cp_neighbors[ci * 4 + 2];
    rd.next_dir = cp_neighbors[ci * 4 + 3];
    rd.orig_pos = read_orig(ci);
    rd.orig_prev = (rd.prev_ci >= 0) ? read_orig(rd.prev_ci) : rd.orig_pos;
    rd.orig_next = (rd.next_ci >= 0) ? read_orig(rd.next_ci) : rd.orig_pos;
    rd.icx = uint((ci / 2) % int(u.corners_w));
    rd.icy = uint((ci / 2) / int(u.corners_w));
    return rd;
}

uint get_px_color(int px, int py) {
    px = clamp(px, 0, int(u.img_w) - 1);
    py = clamp(py, 0, int(u.img_h) - 1);
    return pixels[py * int(u.img_w) + px];
}

void get_edge_colors(uint icx, uint icy, int dir, out uint color_left, out uint color_right) {
    switch (dir) {
        case 0: color_left = get_px_color(int(icx)-1, int(icy)-1); color_right = get_px_color(int(icx), int(icy)-1); break;
        case 1: color_left = get_px_color(int(icx), int(icy)-1); color_right = get_px_color(int(icx), int(icy)); break;
        case 2: color_left = get_px_color(int(icx), int(icy)); color_right = get_px_color(int(icx)-1, int(icy)); break;
        case 3: color_left = get_px_color(int(icx)-1, int(icy)); color_right = get_px_color(int(icx)-1, int(icy)-1); break;
        default: color_left = 0u; color_right = 0u; break;
    }
}

bool resolve_color(float2 pt, SlimCP sc, ResolveData rd, float t, out uint out_color) {
    out_color = 0u;
    // Endpoint defer: when this CP's curve has its closest point exactly at
    // an endpoint that's structurally co-located with another CP's curve
    // (degenerate stem at t=1, clamped Bezier extension at t=0), the local
    // edges describe a different region. Defer to the next-closest candidate.
    //
    // The t == 0.0 / t == 1.0 exact compares are sound because closest_on_*
    // routines clamp t to exactly 0 or 1 at the endpoints — they don't return
    // 1e-7 noise. Don't relax this to an epsilon: a curve whose closest
    // point is *just* inside (t=0.001) shouldn't defer.
    bool prev_extends =
        (rd.prev_ci < 0) || ((cp_flags[rd.prev_ci] & IS_ENDPOINT) != 0u);
    bool next_extends =
        (rd.next_ci < 0) || ((cp_flags[rd.next_ci] & IS_ENDPOINT) != 0u);
    if ((prev_extends && t == 0.0) || (next_extends && t == 1.0)) {
        return false;
    }
    uint pl = 0u, pr = 0u, nl = 0u, nr = 0u;
    if (rd.prev_dir >= 0) get_edge_colors(rd.icx, rd.icy, rd.prev_dir, pl, pr);
    if (rd.next_dir >= 0) get_edge_colors(rd.icx, rd.icy, rd.next_dir, nl, nr);

    uint color_left = 0u, color_right = 0u;
    float ref_t;
    bool prev_valid = (pl != pr);
    bool next_valid = (nl != nr);

    if (rd.prev_ci < 0) {
        if (next_valid) { color_left = nr; color_right = nl; ref_t = 1.0; }
        else return false;
    } else if (rd.next_ci < 0) {
        if (prev_valid) { color_left = pl; color_right = pr; ref_t = 0.0; }
        else return false;
    } else if (t < sc.t_branch) {
        if (prev_valid) { color_left = pl; color_right = pr; ref_t = 0.0; }
        else if (next_valid) { color_left = nr; color_right = nl; ref_t = 1.0; }
        else return false;
    } else {
        if (next_valid) { color_left = nr; color_right = nl; ref_t = 1.0; }
        else if (prev_valid) { color_left = pl; color_right = pr; ref_t = 0.0; }
        else return false;
    }

    float2 orig_tangent = bspline_deriv(rd.orig_prev, rd.orig_pos, rd.orig_next, ref_t);
    float tl2 = length(orig_tangent);
    if (tl2 < 1e-4) return false;
    orig_tangent /= tl2;
    float2 orig_normal = float2(-orig_tangent.y, orig_tangent.x);

    float2 cpt = bspline_eval(sc.prev_pos, sc.pos, sc.next_pos, t);
    float2 opt_tangent = bspline_deriv(sc.prev_pos, sc.pos, sc.next_pos, t);
    float otl = length(opt_tangent);
    if (otl < 1e-4) return false;
    opt_tangent /= otl;
    float2 opt_normal = float2(-opt_tangent.y, opt_tangent.x);

    bool normals_agree = dot(opt_normal, orig_normal) > 0.0;
    float side = dot(pt - cpt, opt_normal);
    if (normals_agree) {
        out_color = (side > 0.0) ? color_left : color_right;
    } else {
        out_color = (side > 0.0) ? color_right : color_left;
    }
    return true;
}

/// Linear approximation of a curve at the AA hit's closest_t, plus the
/// curve's pos/neg side colors. Drives single-curve AA, dual-curve AA,
/// and (with cpt anchored at J) wedge AA.
struct AaLine {
    float2 cpt;
    float2 normal;
    uint pos_color;
    uint neg_color;
    bool valid;
};

AaLine build_aa_line(SlimCP sc, ResolveData rd, float t) {
    AaLine result;
    result.valid = false;
    result.cpt = float2(0, 0);
    result.normal = float2(0, 0);
    result.pos_color = 0u;
    result.neg_color = 0u;

    float2 cpt = bspline_eval(sc.prev_pos, sc.pos, sc.next_pos, t);
    float2 tang = bspline_deriv(sc.prev_pos, sc.pos, sc.next_pos, t);
    float tl = length(tang);
    if (tl < 1e-4) return result;
    float2 normal = float2(-tang.y, tang.x) / tl;

    uint aa_pl = 0u, aa_pr = 0u, aa_nl = 0u, aa_nr = 0u;
    if (rd.prev_dir >= 0) get_edge_colors(rd.icx, rd.icy, rd.prev_dir, aa_pl, aa_pr);
    if (rd.next_dir >= 0) get_edge_colors(rd.icx, rd.icy, rd.next_dir, aa_nl, aa_nr);

    uint color_left = 0u, color_right = 0u;
    bool have_edge = false;
    float ref_t = 0.0;
    if (t < sc.t_branch && aa_pl != aa_pr) {
        color_left = aa_pl; color_right = aa_pr; ref_t = 0.0; have_edge = true;
    } else if (t >= sc.t_branch && aa_nl != aa_nr) {
        color_left = aa_nr; color_right = aa_nl; ref_t = 1.0; have_edge = true;
    } else if (aa_nl != aa_nr) {
        color_left = aa_nr; color_right = aa_nl; ref_t = 1.0; have_edge = true;
    } else if (aa_pl != aa_pr) {
        color_left = aa_pl; color_right = aa_pr; ref_t = 0.0; have_edge = true;
    }
    if (!have_edge) return result;

    float2 orig_tang = bspline_deriv(rd.orig_prev, rd.orig_pos, rd.orig_next, ref_t);
    float2 orig_norm = float2(-orig_tang.y, orig_tang.x);
    bool flip = dot(normal, orig_norm) < 0.0;
    result.cpt = cpt;
    result.normal = normal;
    result.pos_color = flip ? color_right : color_left;
    result.neg_color = flip ? color_left : color_right;
    result.valid = true;
    return result;
}

/// Junction-anchored AA line: cpt = J (junction position), tangent at
/// `t_eval`. Returns the unit tangent in `out_tang_unit` so caller can
/// align line_b's normal to line_a's tangent direction. valid=true iff
/// the curve has at least one edge with a color discontinuity (otherwise
/// the line wouldn't separate distinct color regions).
AaLine build_junction_aa_line(SlimCP sc, ResolveData rd, float t_eval, float2 j,
                              out float2 out_tang_unit) {
    AaLine result;
    result.valid = false;
    result.cpt = j;
    result.normal = float2(0, 0);
    result.pos_color = 0u;
    result.neg_color = 0u;
    out_tang_unit = float2(0, 0);

    float2 tang = bspline_deriv(sc.prev_pos, sc.pos, sc.next_pos, t_eval);
    float tl = length(tang);
    if (tl < 1e-4) return result;
    float2 tu = tang / tl;
    out_tang_unit = tu;

    uint aa_pl = 0u, aa_pr = 0u, aa_nl = 0u, aa_nr = 0u;
    bool prev_split = false, next_split = false;
    if (rd.prev_dir >= 0) {
        get_edge_colors(rd.icx, rd.icy, rd.prev_dir, aa_pl, aa_pr);
        prev_split = (aa_pl != aa_pr);
    }
    if (rd.next_dir >= 0) {
        get_edge_colors(rd.icx, rd.icy, rd.next_dir, aa_nl, aa_nr);
        next_split = (aa_nl != aa_nr);
    }
    if (!prev_split && !next_split) return result;

    result.normal = float2(-tu.y, tu.x);
    result.valid = true;
    return result;
}

/// Find the closest cached hit that's not on `sc_a_ci`'s chain (1-step
/// neighbor test) and is within `aa_threshold`. Returns the cached slot
/// index in [0, 4) or -1 if no qualifying hit exists.
int find_second_chain_hit(int sc_a_ci, int best_idx,
                          int idx0, int idx1, int idx2,
                          float d2_0, float d2_1, float d2_2,
                          int cached_n, float aa_threshold) {
    int sa_prev = cp_neighbors[sc_a_ci * 4];
    int sa_next = cp_neighbors[sc_a_ci * 4 + 1];

    // Each iteration is a closure over (slot, ci, d2). Unrolled across
    // s ∈ {0, 1, 2} on constant indices to keep idxs/d2s in registers
    // (dynamic-indexed local arrays force GPU compilers to spill).
    #define TRY_SEC_HIT(s, ci_v, d2_v) \
        if ((s) < cached_n) { \
            int _ci = (ci_v); \
            if (_ci >= 0 && _ci != best_idx) { \
                if ((d2_v) >= aa_threshold) return -1; \
                SlimCP _cand = sh_cps[uint(_ci)]; \
                bool _same = (sc_a_ci == _cand.ci) \
                    || (sc_a_ci == cp_neighbors[_cand.ci * 4]) \
                    || (sc_a_ci == cp_neighbors[_cand.ci * 4 + 1]) \
                    || (_cand.ci == sa_prev) \
                    || (_cand.ci == sa_next); \
                if (!_same) return (s); \
            } \
        }
    TRY_SEC_HIT(0, idx0, d2_0)
    TRY_SEC_HIT(1, idx1, d2_1)
    TRY_SEC_HIT(2, idx2, d2_2)
    #undef TRY_SEC_HIT
    return -1;
}

/// Exact pixel coverage by a tangent line. d_perp_pixel is the line's
/// signed perpendicular distance from pixel center in pixel-side units.
/// Returns the area on the line's positive side, in [0, 1].
///   |d_p| <= (a-b)/2     → linear regime (line crosses parallel edges)
///   (a-b)/2 < |d_p| < (a+b)/2 → corner-cut quadratic
///   |d_p| >= (a+b)/2     → saturated
float line_coverage_pos(float2 normal, float d_perp_pixel) {
    float nx = abs(normal.x);
    float ny = abs(normal.y);
    float a = max(nx, ny);
    float b = min(nx, ny);
    float half_ext = (a + b) * 0.5;
    float lin_ext = (a - b) * 0.5;
    if (d_perp_pixel >= half_ext) return 1.0;
    if (d_perp_pixel <= -half_ext) return 0.0;
    if (abs(d_perp_pixel) <= lin_ext) return 0.5 + d_perp_pixel / a;
    if (d_perp_pixel > 0.0) {
        float t = half_ext - d_perp_pixel;
        return 1.0 - 0.5 * t * t / (a * b);
    }
    float t = half_ext + d_perp_pixel;
    return 0.5 * t * t / (a * b);
}

/// Decode an ARGB packed color into linear sRGB (gamma 2.2).
float3 srgb_to_linear(uint c) {
    float3 srgb = float3(float((c >> 16u) & 0xFFu),
                         float((c >> 8u) & 0xFFu),
                         float(c & 0xFFu)) / 255.0;
    return pow(srgb, float3(2.2));
}

struct SHit { float d2; float t; int idx; };

uint sample_from_shared(float2 pt, uint fallback_color,
                        out int out_idx, out float out_t, out float out_d2,
                        out int cached_idx_0, out int cached_idx_1,
                        out int cached_idx_2,
                        out float cached_t_0, out float cached_t_1,
                        out float cached_t_2,
                        out float cached_d2_0, out float cached_d2_1,
                        out float cached_d2_2,
                        out int cached_n, out float second_d2) {
    SHit hits[3];
    int num_hits = 0;
    // Init via constant indices — keeps hits[] in registers vs spilling
    // to local memory (dynamic-indexed loop would force the latter on
    // most GPU compilers).
    hits[0].d2 = 1e10; hits[0].idx = -1; hits[0].t = 0.0;
    hits[1].d2 = 1e10; hits[1].idx = -1; hits[1].t = 0.0;
    hits[2].d2 = 1e10; hits[2].idx = -1; hits[2].t = 0.0;

    // Clamp to MAX_TILE_CPS: the load loop's atomic add can exceed the cap
    // in dense tiles, but only slots [0, MAX_TILE_CPS) are actually written.
    uint n = min(sh_num_cps.load(), MAX_TILE_CPS);
    for (uint i = 0u; i < n; i++) {
        SlimCP sc = sh_cps[i];

        if (pt.x < sc.bbox_min.x - 1.0 || pt.x > sc.bbox_max.x + 1.0 ||
            pt.y < sc.bbox_min.y - 1.0 || pt.y > sc.bbox_max.y + 1.0)
            continue;

        float2 quick = bspline_quick_screen(sc.prev_pos, sc.pos, sc.next_pos, pt);
        if (quick.y > 2.0) continue;

        float2 result = sc.is_line
            ? closest_on_chain_segment(sc, pt)
            : bspline_refine(sc.prev_pos, sc.pos, sc.next_pos, pt, quick.x, quick.y);
        float span_best_t = result.x;
        float span_best_d2 = result.y;

        if (span_best_d2 < 1.0) {
            // Insertion sort via explicit if/else over constant indices —
            // keeps hits[] register-allocated.
            if (span_best_d2 < hits[0].d2) {
                hits[2] = hits[1];
                hits[1] = hits[0];
                hits[0].d2 = span_best_d2; hits[0].t = span_best_t; hits[0].idx = int(i);
                if (num_hits < 3) num_hits++;
            } else if (span_best_d2 < hits[1].d2) {
                hits[2] = hits[1];
                hits[1].d2 = span_best_d2; hits[1].t = span_best_t; hits[1].idx = int(i);
                if (num_hits < 3) num_hits++;
            } else if (span_best_d2 < hits[2].d2) {
                hits[2].d2 = span_best_d2; hits[2].t = span_best_t; hits[2].idx = int(i);
                if (num_hits < 3) num_hits++;
            }
        }
    }

    // Export the d²-sorted top-3 indices for the wedge AA path to reuse
    // (line_b search at the pixel center). Passed as 3 individual ints
    // (not an array) — Slang/SPIRV-Cross handles array out-params unevenly
    // across backends; scalars are uniformly safe.
    cached_idx_0 = (num_hits > 0) ? hits[0].idx : -1;
    cached_idx_1 = (num_hits > 1) ? hits[1].idx : -1;
    cached_idx_2 = (num_hits > 2) ? hits[2].idx : -1;
    cached_t_0 = (num_hits > 0) ? hits[0].t : 0.0;
    cached_t_1 = (num_hits > 1) ? hits[1].t : 0.0;
    cached_t_2 = (num_hits > 2) ? hits[2].t : 0.0;
    cached_d2_0 = (num_hits > 0) ? hits[0].d2 : 1e10;
    cached_d2_1 = (num_hits > 1) ? hits[1].d2 : 1e10;
    cached_d2_2 = (num_hits > 2) ? hits[2].d2 : 1e10;
    cached_n = num_hits;
    // second_d2 lets the wedge-AA gate skip ALL its work (line_a build,
    // line_b search, polygon clip, centroid sampling) when the next-closest
    // curve isn't within aa_threshold — most AA pixels (single-curve edges,
    // not junctions) hit this fast path.
    second_d2 = (num_hits > 1) ? hits[1].d2 : 1e10;

    // Default outs to the closest hit (used by AA only if no hit resolves).
    out_idx = (num_hits > 0) ? hits[0].idx : -1;
    out_t = (num_hits > 0) ? hits[0].t : 0.0;
    out_d2 = (num_hits > 0) ? hits[0].d2 : 1e10;

    // Try resolve on each hit in order — unrolled across constant indices
    // so hits[] stays register-allocated. Promote the resolved hit's data
    // to out-params so the AA path uses the same CP that produced the
    // center color, not a closer one that was endpoint-deferred.
    if (num_hits >= 1) {
        SlimCP sc = sh_cps[uint(hits[0].idx)];
        ResolveData rd = read_resolve_data(sc.ci);
        uint c;
        if (resolve_color(pt, sc, rd, hits[0].t, c)) {
            out_idx = hits[0].idx; out_t = hits[0].t; out_d2 = hits[0].d2;
            return c;
        }
    }
    if (num_hits >= 2) {
        SlimCP sc = sh_cps[uint(hits[1].idx)];
        ResolveData rd = read_resolve_data(sc.ci);
        uint c;
        if (resolve_color(pt, sc, rd, hits[1].t, c)) {
            out_idx = hits[1].idx; out_t = hits[1].t; out_d2 = hits[1].d2;
            return c;
        }
    }
    if (num_hits >= 3) {
        SlimCP sc = sh_cps[uint(hits[2].idx)];
        ResolveData rd = read_resolve_data(sc.ci);
        uint c;
        if (resolve_color(pt, sc, rd, hits[2].t, c)) {
            out_idx = hits[2].idx; out_t = hits[2].t; out_d2 = hits[2].d2;
            return c;
        }
    }
    return fallback_color;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID, uint3 ltid: SV_GroupThreadID, uint3 gid: SV_GroupID) {
    uint tile_idx = gid.x;
    uint local_tid = ltid.x;

    uint tile_x = tile_idx % u.tiles_w;
    uint tile_y = tile_idx / u.tiles_w;
    bool valid_tile = (tile_y < u.tiles_h);

    int src_x0 = int(tile_x) * 2;
    int src_y0 = int(tile_y) * 2;

    // Phase 1: Cooperatively load CPs into shared memory
    if (local_tid == 0u) sh_num_cps.store(0u);
    GroupMemoryBarrierWithGroupSync();

    if (valid_tile) {
        // Slot layout for the 26-candidate scan over a 2x2 source tile:
        //   slots 0..17 — 3x3 unique corners covering the tile's 4 cells,
        //                 corner_idx = slot/2 ∈ 0..8 row-major (dx, dy ∈
        //                 {0, 1, 2}), slot_within = slot & 1.
        //   slots 18..25 — T-junction stem fan-out for the 8 non-center
        //                  corners. The center corner (corner_idx==4) is
        //                  skipped because all 4 cardinal stem directions
        //                  point INTO the 3x3 walk — its stem-bottom is
        //                  always already loaded by the base slots.
        for (uint slot_idx = local_tid; slot_idx < 26u; slot_idx += 256u) {
            int ci;
            if (slot_idx < 18u) {
                uint corner_idx = slot_idx / 2u;
                uint slot = slot_idx & 1u;
                int cx = src_x0 + int(corner_idx % 3u);
                int cy = src_y0 + int(corner_idx / 3u);
                if (cx < 0 || cy < 0 || cx >= int(u.corners_w) || cy > int(u.img_h)) continue;
                ci = (cy * int(u.corners_w) + cx) * 2 + int(slot);
            } else {
                // Fan-out target: at corner C (non-center), if slot 0 is a
                // T-junction through-CP, the stem-bottom CP can sit outside
                // the 3x3 walk. Look it up via slot 1's chain link. Skip
                // when stem-bottom IS already a corner in the 3x3 walk.
                uint fan_idx = slot_idx - 18u;
                uint corner_idx = fan_idx + ((fan_idx >= 4u) ? 1u : 0u);
                int cx = src_x0 + int(corner_idx % 3u);
                int cy = src_y0 + int(corner_idx / 3u);
                if (cx < 0 || cy < 0 || cx >= int(u.corners_w) || cy > int(u.img_h)) continue;
                int slot0_ci = (cy * int(u.corners_w) + cx) * 2;
                if ((cp_flags[slot0_ci] & IS_TJUNCTION) == 0u) continue;
                int slot1_ci = slot0_ci + 1;
                int sp = cp_neighbors[slot1_ci * 4];
                int sn = cp_neighbors[slot1_ci * 4 + 1];
                ci = (sp >= 0) ? sp : sn;
                if (ci < 0) continue;
                int s_cx = (ci / 2) % int(u.corners_w);
                int s_cy = (ci / 2) / int(u.corners_w);
                if (s_cx >= src_x0 && s_cx <= src_x0 + 2
                 && s_cy >= src_y0 && s_cy <= src_y0 + 2) continue;
            }
            uint flag = cp_flags[ci];
            if (flag == 0u) continue;

            int prev_ci = cp_neighbors[ci * 4];
            int next_ci = cp_neighbors[ci * 4 + 1];
            // Clamped model: endpoint CPs don't own a span; their interior
            // neighbor's span extends to reach them via knot multiplicity 3.
            // Exception: 2-CP chains have no interior neighbor to do the
            // clamping, so we render here with the degenerate fallback.
            if (prev_ci < 0 && next_ci < 0) continue;
            bool i_am_endpoint = (prev_ci < 0 || next_ci < 0);
            bool two_cp_chain = false;
            if (i_am_endpoint) {
                int other = (prev_ci < 0) ? next_ci : prev_ci;
                bool other_is_end = (cp_flags[other] & IS_ENDPOINT) != 0u;
                if (!other_is_end) continue;
                // 2-CP chain: render only ONCE — the lower-indexed endpoint
                // owns the span. Without this guard both endpoints render the
                // same physical line with opposite-bowing ghost curvatures.
                if (ci > other) continue;
                two_cp_chain = true;
            }

            float2 cp_real = read_pos(ci);
            // 2-CP-chain fallback: if either neighbor is missing, use the
            // CP itself for that side (degenerate, straight-line span).
            float2 prev_pos = (prev_ci >= 0) ? read_pos(prev_ci) : cp_real;
            float2 next_pos = (next_ci >= 0) ? read_pos(next_ci) : cp_real;

            // Tile bbox cull: the per-pixel scan accepts a CP if its bbox
            // (extended by 1 unit) reaches the pixel. Tile spans src_x0..
            // src_x0+2 in source coords, so a CP's bbox-extended region
            // intersects the tile iff bbox_max ≥ tile_min-1 and bbox_min ≤
            // tile_max+1. CPs that fail this can never affect any pixel in
            // the tile and don't need to take a shared-memory slot.
            float2 bbox_min = min(min(prev_pos, cp_real), next_pos);
            float2 bbox_max = max(max(prev_pos, cp_real), next_pos);
            float tile_min_x = float(src_x0) - 1.0;
            float tile_min_y = float(src_y0) - 1.0;
            float tile_max_x = float(src_x0 + 2) + 1.0;
            float tile_max_y = float(src_y0 + 2) + 1.0;
            if (bbox_max.x < tile_min_x || bbox_min.x > tile_max_x ||
                bbox_max.y < tile_min_y || bbox_min.y > tile_max_y) continue;

            uint idx = sh_num_cps.add(1u);
            if (idx < MAX_TILE_CPS) {
                // If a neighbor is an endpoint, replace its position with a
                // virtual ghost = 2*real - cp so that bspline_eval(...,t=0|1)
                // lands exactly on the real endpoint position. For interior
                // neighbors, ghost == real; nothing changes.
                bool prev_is_end = (prev_ci >= 0) && ((cp_flags[prev_ci] & IS_ENDPOINT) != 0u);
                bool next_is_end = (next_ci >= 0) && ((cp_flags[next_ci] & IS_ENDPOINT) != 0u);
                float2 cp;
                float2 pp;
                float2 np;
                if (two_cp_chain) {
                    // Render as a straight line between the two markers.
                    // Pick p0=(3·a0-a1)/2, p1=midpoint, p2=(3·a1-a0)/2 so
                    // bspline_eval gives B(t)=lerp(a0,a1,t) — needed for the
                    // AA path's cpt/tangent evaluation, which goes through
                    // beval. Closest-point queries dispatch to a line
                    // solver via is_line, bypassing bspline_refine entirely.
                    int other = (prev_ci < 0) ? next_ci : prev_ci;
                    float2 other_pos = read_pos(other);
                    float2 a0 = (prev_ci < 0) ? cp_real : other_pos;
                    float2 a1 = (next_ci < 0) ? cp_real : other_pos;
                    pp = 1.5 * a0 - 0.5 * a1;
                    cp = 0.5 * (a0 + a1);
                    np = 1.5 * a1 - 0.5 * a0;
                } else {
                    cp = cp_real;
                    pp = prev_is_end ? (2.0 * prev_pos - cp) : prev_pos;
                    np = next_is_end ? (2.0 * next_pos - cp) : next_pos;
                }

                // Bbox: include the real prev/next positions (not ghosts) so
                // that pixels in the cell-radius around endpoint neighbors
                // still find this span as a candidate. Ghosts can be far
                // outside the curve's actual extent.
                sh_cps[idx].pos = cp;
                sh_cps[idx].prev_pos = pp;
                sh_cps[idx].next_pos = np;
                sh_cps[idx].bbox_min = bbox_min;
                sh_cps[idx].bbox_max = bbox_max;
                sh_cps[idx].ci = ci;
                sh_cps[idx].is_line = two_cp_chain;

                // t_branch: branch threshold for prev_dir vs next_dir in
                // resolve_color. For non-clamped spans the curve passes
                // through cp at t=0.5 trivially. For clamped spans, project
                // the rendered curve's B(0.5) (= 0.125*pp + 0.75*cp +
                // 0.125*np with ghost adjustment) back onto itself to find
                // t. This aligns t_branch with the algebraic ghost-aware
                // stem snap performed by update_tjunction. Crossings
                // override the default with the actual curve-curve
                // intersection parameter from `crossing_t` (Stage 5b), so
                // the rasterizer's J = beval(...t_branch) lands on the
                // geometric intersection, not the integer grid corner.
                if ((flag & IS_CROSSING) != 0u) {
                    sh_cps[idx].t_branch = crossing_t[ci];
                } else if (two_cp_chain) {
                    // Straight-line span: B(t) is linear, interior_mid = cp =
                    // midpoint, closest t is trivially 0.5.
                    sh_cps[idx].t_branch = 0.5;
                } else if (prev_is_end || next_is_end) {
                    float2 interior_mid = 0.125 * pp + 0.75 * cp + 0.125 * np;
                    sh_cps[idx].t_branch = bspline_refine(pp, cp, np, interior_mid, 0.5, 1e10).x;
                } else {
                    sh_cps[idx].t_branch = 0.5;
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (!valid_tile) return;

    // Phase 2: Each thread processes output pixels in the tile.
    uint out_x0 = uint(floor(float(src_x0) * u.scale));
    uint out_y0 = uint(floor(float(src_y0) * u.scale));
    uint out_x1 = min(uint(ceil(float(src_x0 + 2) * u.scale)), u.out_w);
    uint out_y1 = min(uint(ceil(float(src_y0 + 2) * u.scale)), u.out_h);
    uint tile_out_w = out_x1 - out_x0;
    uint tile_out_h = out_y1 - out_y0;
    uint pixels_per_tile = tile_out_w * tile_out_h;

    for (uint px_idx = local_tid; px_idx < pixels_per_tile; px_idx += 256u) {
        uint local_x = px_idx % tile_out_w;
        uint local_y = px_idx / tile_out_w;
        uint opx = out_x0 + local_x;
        uint opy = out_y0 + local_y;
        if (opx >= u.out_w || opy >= u.out_h) continue;

        float inv_scale = 1.0 / u.scale;
        float2 center = float2(float(opx) + 0.5, float(opy) + 0.5) * inv_scale;

        int fb_x = clamp(int(floor(center.x)), 0, int(u.img_w) - 1);
        int fb_y = clamp(int(floor(center.y)), 0, int(u.img_h) - 1);
        uint fallback = pixels[fb_y * int(u.img_w) + fb_x];

        int best_idx = -1;
        float best_t = 0.0;
        float best_d2 = 1e10;
        int cached_idx_0 = -1, cached_idx_1 = -1, cached_idx_2 = -1;
        float cached_t_0 = 0.0, cached_t_1 = 0.0, cached_t_2 = 0.0;
        float cached_d2_0 = 1e10, cached_d2_1 = 1e10, cached_d2_2 = 1e10;
        int cached_n = 0;
        float second_d2 = 1e10;
        uint center_color = sample_from_shared(center, fallback, best_idx, best_t, best_d2,
                                                cached_idx_0, cached_idx_1, cached_idx_2,
                                                cached_t_0, cached_t_1, cached_t_2,
                                                cached_d2_0, cached_d2_1, cached_d2_2,
                                                cached_n, second_d2);

        // AA threshold: a pixel's half-diagonal² in source units is
        // 0.5/scale². Beyond that, the curve passes outside the pixel
        // and the analytical pixel-coverage formula saturates to a
        // solid color anyway, so AA work would be wasted. The same
        // tight bound applies to single-curve AA and the wedge AA
        // gate so each fires only on pixels the curve(s) pass through.
        float aa_threshold = 0.5 / (u.scale * u.scale);
        bool need_aa = (best_idx >= 0 && best_d2 < aa_threshold);

        // Wedge AA model
        // ==============
        // line_a: tangent at the through curve (carries up to 4 colors via
        //   t_branch — segment 0 covers t < t_branch with prev edge colors,
        //   segment 1 covers t >= t_branch with next edge colors).
        // line_b: tangent at the stem render curve. Selects which line_a
        //   segment a wedge belongs to via sb-sign — but only when line_b
        //   actually crosses line_a's render range (b_separates). Outside
        //   that range all wedges share the segment containing the pixel
        //   center (center_in_seg1).
        // Pixel is clipped against line_a then line_b → 4 wedges. Each
        // picks one of {pos_seg0, neg_seg0, pos_seg1, neg_seg1} via two
        // sign tests on the centroid.
        //
        // Wedge AA fires only when the structural junction relationship
        // holds. At a T-junction corner, slot 0 holds the through curve;
        // slot 1 holds the original stem CP (filtered — its render span is
        // owned by an adjacent corner via clamped Bezier). Slot 1's
        // surviving non-(-1) neighbor IS the stem render CP. So candidate
        // A is the through-curve CP at a junction iff
        //   (a) A is IS_TJUNCTION-flagged, AND
        //   (b) slot 1 at A's corner has B's ci as one of its neighbors.
        // Crossings (valence-4): both hits are IS_CROSSING at the same
        // corner — partner identification is direct.
        AaLine line_a;
        AaLine line_b;
        line_a.valid = false;
        line_b.valid = false;
        SlimCP sc_a;
        ResolveData rd_a;
        float line_a_t = 0.0;
        if (need_aa && best_idx >= 0 && (second_d2 < aa_threshold)) {
            int second_idx = (cached_n > 0 && cached_idx_0 != best_idx) ? cached_idx_0
                           : (cached_n > 1 && cached_idx_1 != best_idx) ? cached_idx_1
                           : -1;
            if (second_idx >= 0) {
                SlimCP sc0 = sh_cps[uint(best_idx)];
                SlimCP sc1 = sh_cps[uint(second_idx)];
                int ci0 = sc0.ci;
                int ci1 = sc1.ci;
                uint f0 = cp_flags[ci0];
                uint f1 = cp_flags[ci1];

                int slot1_ci_0 = (ci0 / 2) * 2 + 1;
                int slot1_ci_1 = (ci1 / 2) * 2 + 1;
                // Structural lookup of t_b_eval — the parameter on sc_b's
                // curve at which it passes through (or terminates at) the
                // junction. Three cases, all without closest_t:
                //   (a) sc_b IS slot 1 of sc_a's corner — 2-CP-chain stem.
                //       Cell graph creates slot 1 with next_ci=-1, so the
                //       2-CP construction puts a1 = slot1.pos = junction
                //       side at t=1.
                //   (b) sc_b is slot 1's neighbor — slot 1 was filtered;
                //       sc_b ghost-extends through slot 1's pos at whichever
                //       of sc_b's prev/next is IS_ENDPOINT-flagged (t=0 for
                //       prev, t=1 for next).
                //   Crossings: sc_b's curve passes through J at t_branch.
                int cp0_t_match = -1;  // -1 = no match, 0 = t=0, 1 = t=1, 2 = t=t_branch
                int cp1_t_match = -1;
                if ((f0 & IS_TJUNCTION) != 0u) {
                    if (ci1 == slot1_ci_0) {
                        cp0_t_match = 1;  // case (a)
                    } else if (cp_neighbors[slot1_ci_0 * 4] == ci1
                            || cp_neighbors[slot1_ci_0 * 4 + 1] == ci1) {
                        // case (b): junction side = sc_b's IS_ENDPOINT neighbor
                        int sc_b_prev = cp_neighbors[ci1 * 4];
                        int sc_b_next = cp_neighbors[ci1 * 4 + 1];
                        bool prev_is_end = sc_b_prev >= 0
                            && (cp_flags[sc_b_prev] & IS_ENDPOINT) != 0u;
                        bool next_is_end = sc_b_next >= 0
                            && (cp_flags[sc_b_next] & IS_ENDPOINT) != 0u;
                        if (prev_is_end) cp0_t_match = 0;
                        else if (next_is_end) cp0_t_match = 1;
                    }
                }
                if ((f1 & IS_TJUNCTION) != 0u) {
                    if (ci0 == slot1_ci_1) {
                        cp1_t_match = 1;
                    } else if (cp_neighbors[slot1_ci_1 * 4] == ci0
                            || cp_neighbors[slot1_ci_1 * 4 + 1] == ci0) {
                        int sc_b_prev = cp_neighbors[ci0 * 4];
                        int sc_b_next = cp_neighbors[ci0 * 4 + 1];
                        bool prev_is_end = sc_b_prev >= 0
                            && (cp_flags[sc_b_prev] & IS_ENDPOINT) != 0u;
                        bool next_is_end = sc_b_next >= 0
                            && (cp_flags[sc_b_next] & IS_ENDPOINT) != 0u;
                        if (prev_is_end) cp1_t_match = 0;
                        else if (next_is_end) cp1_t_match = 1;
                    }
                }
                bool both_x_same =
                    ((f0 & IS_CROSSING) != 0u)
                    && ((f1 & IS_CROSSING) != 0u)
                    && (ci0 / 2) == (ci1 / 2);

                int through_idx = -1;
                int stem_idx = -1;
                float t_b_eval = 0.0;
                if (cp0_t_match >= 0) {
                    through_idx = best_idx;
                    stem_idx = second_idx;
                    t_b_eval = (cp0_t_match == 0) ? 0.0 : 1.0;
                } else if (cp1_t_match >= 0) {
                    through_idx = second_idx;
                    stem_idx = best_idx;
                    t_b_eval = (cp1_t_match == 0) ? 0.0 : 1.0;
                } else if (both_x_same) {
                    through_idx = best_idx;
                    stem_idx = second_idx;
                    t_b_eval = sh_cps[uint(second_idx)].t_branch;
                }

                // The wedge AA model assumes the 3-color region (the wedge
                // emanating from the junction) overlaps the pixel. That's
                // true only when the junction — the geometric meeting point
                // of the through and stem curves, which equals the through
                // CP's pos post-correction — lies inside the pixel square.
                // When two curves pass near a pixel but their junction is
                // outside it, the local geometry is 2-color; falling back to
                // single-curve AA avoids mixing in a spurious third color.
                if (through_idx >= 0) {
                    SlimCP sc_through_test = sh_cps[uint(through_idx)];
                    // J = beval(through, t_branch): the actual point on the
                    // rendered curve at the junction parameter. NOT sc.pos
                    // (= polynomial control point), since for non-ghost-
                    // extended through CPs (e.g. an arch with all interior
                    // neighbors) the curve at t_branch differs from pos by
                    // the 0.125*p0 + 0.75*p1 + 0.125*p2 vs p1 offset.
                    float2 J_test = bspline_eval(sc_through_test.prev_pos,
                                                  sc_through_test.pos,
                                                  sc_through_test.next_pos,
                                                  sc_through_test.t_branch);
                    float pix_half_test = inv_scale * 0.5;
                    if (abs(J_test.x - center.x) > pix_half_test
                        || abs(J_test.y - center.y) > pix_half_test) {
                        through_idx = -1;
                        stem_idx = -1;
                    }
                }

                if (through_idx >= 0) {
                    sc_a = sh_cps[uint(through_idx)];
                    rd_a = read_resolve_data(sc_a.ci);
                    // J = beval(sc_a, t_branch) — point on the rendered through
                    // curve at the junction parameter (not sc_a.pos).
                    float2 J = bspline_eval(sc_a.prev_pos, sc_a.pos, sc_a.next_pos,
                                             sc_a.t_branch);
                    line_a_t = sc_a.t_branch;

                    // line_a: through tangent at t_branch (where the curve
                    // passes through its own pos by construction).
                    float2 tang_a_unit;
                    line_a = build_junction_aa_line(sc_a, rd_a, sc_a.t_branch, J, tang_a_unit);

                    // line_b: stem tangent at the junction. t_b_eval was
                    // determined structurally by the gate above.
                    SlimCP sc_b = sh_cps[uint(stem_idx)];
                    ResolveData rd_b = read_resolve_data(sc_b.ci);
                    float2 tang_b_unit;
                    line_b = build_junction_aa_line(sc_b, rd_b, t_b_eval, J, tang_b_unit);

                    // Align line_b.normal so sb > 0 = "in tang_a direction"
                    // along line_a = post-junction (seg 1). If line_b's normal
                    // opposes tang_a, flip it.
                    if (line_b.valid && dot(line_b.normal, tang_a_unit) < 0.0) {
                        line_b.normal = -line_b.normal;
                    }
                }
            }
        }

        if (need_aa && line_a.valid && line_b.valid) {
            // Through-curve color LUT (seg0 / seg1 × pos / neg). At a
            // T-junction the through curve (= sc_a) carries all 3 colors
            // via its prev/next edge segments split at t_branch. Build
            // each segment's (pos, neg) using the same edge-selection
            // logic as resolve_color (so a missing edge falls back to
            // the other one — both segments share colors), then align
            // to line_a.normal.
            uint a_pl = 0u, a_pr = 0u, a_nl = 0u, a_nr = 0u;
            if (rd_a.prev_dir >= 0) get_edge_colors(rd_a.icx, rd_a.icy, rd_a.prev_dir, a_pl, a_pr);
            if (rd_a.next_dir >= 0) get_edge_colors(rd_a.icx, rd_a.icy, rd_a.next_dir, a_nl, a_nr);
            bool a_prev_valid = (a_pl != a_pr);
            bool a_next_valid = (a_nl != a_nr);

            uint pos_seg0 = center_color, neg_seg0 = center_color;
            uint pos_seg1 = center_color, neg_seg1 = center_color;
            for (int seg = 0; seg < 2; seg++) {
                uint cl = 0u, cr = 0u;
                float ref_t = 0.0;
                bool ok = false;
                if (rd_a.prev_ci < 0) {
                    if (a_next_valid) { cl = a_nr; cr = a_nl; ref_t = 1.0; ok = true; }
                } else if (rd_a.next_ci < 0) {
                    if (a_prev_valid) { cl = a_pl; cr = a_pr; ref_t = 0.0; ok = true; }
                } else if (seg == 0) {
                    if (a_prev_valid)      { cl = a_pl; cr = a_pr; ref_t = 0.0; ok = true; }
                    else if (a_next_valid) { cl = a_nr; cr = a_nl; ref_t = 1.0; ok = true; }
                } else {
                    if (a_next_valid)      { cl = a_nr; cr = a_nl; ref_t = 1.0; ok = true; }
                    else if (a_prev_valid) { cl = a_pl; cr = a_pr; ref_t = 0.0; ok = true; }
                }
                if (!ok) continue;
                float2 ot = bspline_deriv(rd_a.orig_prev, rd_a.orig_pos, rd_a.orig_next, ref_t);
                float2 on = float2(-ot.y, ot.x);
                bool flip_o = dot(line_a.normal, on) < 0.0;
                uint op = flip_o ? cr : cl;
                uint ng = flip_o ? cl : cr;
                if (seg == 0) { pos_seg0 = op; neg_seg0 = ng; }
                else          { pos_seg1 = op; neg_seg1 = ng; }
            }

            // LUT decoded once. Selected by (sa>0) | ((sb>0) << 1):
            //   0 = neg_seg0, 1 = pos_seg0, 2 = neg_seg1, 3 = pos_seg1.
            // Held as 4 scalar float3s (not an array) so the dynamic
            // selection compiles to a register select tree instead of
            // spilling 48 bytes of indexed scratch.
            float3 lut0, lut1, lut2, lut3;
            {
                #define DECODE_TO_LIN(cc) pow(float3(float(((cc) >> 16u) & 0xFFu), \
                                                    float(((cc) >>  8u) & 0xFFu), \
                                                    float( (cc)         & 0xFFu)) / 255.0, \
                                              float3(2.2))
                lut0 = DECODE_TO_LIN(neg_seg0);
                lut1 = DECODE_TO_LIN(pos_seg0);
                lut2 = DECODE_TO_LIN(neg_seg1);
                lut3 = DECODE_TO_LIN(pos_seg1);
                #undef DECODE_TO_LIN
            }

            // Boundary-sweep area accumulation. Walk the pixel boundary
            // CCW; on each edge find up to 2 line crossings (line_a and
            // line_b), split the edge into ≤3 sub-segments, classify
            // each sub-segment by (sa, sb)-sign at its midpoint, and
            // add a signed-triangle area (J as origin) to the matching
            // wedge bucket. J interior + consistent walk direction →
            // |cross| gives the wedge area directly.
            float pix_h = inv_scale * 0.5;
            float2 corners[4] = {
                float2(center.x - pix_h, center.y - pix_h),
                float2(center.x + pix_h, center.y - pix_h),
                float2(center.x + pix_h, center.y + pix_h),
                float2(center.x - pix_h, center.y + pix_h),
            };
            float2 J = line_a.cpt;

            float3 sum_lin = float3(0, 0, 0);
            float total_area = 0.0;

            for (int ei = 0; ei < 4; ei++) {
                float2 p = corners[ei];
                float2 q = corners[(ei + 1) & 3];
                float sa_p = dot(p - line_a.cpt, line_a.normal);
                float sa_q = dot(q - line_a.cpt, line_a.normal);
                float sb_p = dot(p - line_b.cpt, line_b.normal);
                float sb_q = dot(q - line_b.cpt, line_b.normal);

                float splits[2] = { 0.0, 0.0 };
                int nsplit = 0;
                if (sa_p * sa_q < 0.0) {
                    splits[nsplit] = sa_p / (sa_p - sa_q);
                    nsplit++;
                }
                if (sb_p * sb_q < 0.0) {
                    splits[nsplit] = sb_p / (sb_p - sb_q);
                    nsplit++;
                }
                if (nsplit == 2 && splits[0] > splits[1]) {
                    float tmp = splits[0]; splits[0] = splits[1]; splits[1] = tmp;
                }

                // Walk sub-segments [0, splits[0]], [splits[0], splits[1]], [splits[1], 1].
                float t_prev = 0.0;
                float2 u = p;
                for (int k = 0; k < nsplit; k++) {
                    float t_curr = splits[k];
                    if (t_curr > t_prev + 1e-9) {
                        float2 v = p + t_curr * (q - p);
                        float2 m = p + (0.5 * (t_prev + t_curr)) * (q - p);
                        float sa_m = dot(m - line_a.cpt, line_a.normal);
                        float sb_m = dot(m - line_b.cpt, line_b.normal);
                        float2 uj = u - J;
                        float2 vj = v - J;
                        float area = abs(uj.x * vj.y - uj.y * vj.x) * 0.5;
                        if (area >= 1e-9) {
                            int idx = (sa_m > 0.0 ? 1 : 0) | (sb_m > 0.0 ? 2 : 0);
                            float3 lut_sel = (idx == 0) ? lut0
                                           : (idx == 1) ? lut1
                                           : (idx == 2) ? lut2
                                                        : lut3;
                            sum_lin += lut_sel * area;
                            total_area += area;
                        }
                        u = v;
                    }
                    t_prev = t_curr;
                }
                if (t_prev < 1.0 - 1e-9) {
                    float2 m = p + (0.5 * (t_prev + 1.0)) * (q - p);
                    float sa_m = dot(m - line_a.cpt, line_a.normal);
                    float sb_m = dot(m - line_b.cpt, line_b.normal);
                    float2 uj = u - J;
                    float2 vj = q - J;
                    float area = abs(uj.x * vj.y - uj.y * vj.x) * 0.5;
                    if (area >= 1e-9) {
                        int idx = (sa_m > 0.0 ? 1 : 0) | (sb_m > 0.0 ? 2 : 0);
                        float3 lut_sel = (idx == 0) ? lut0
                                       : (idx == 1) ? lut1
                                       : (idx == 2) ? lut2
                                                    : lut3;
                        sum_lin += lut_sel * area;
                        total_area += area;
                    }
                }
            }

            if (total_area > 0.0) {
                float3 blended = pow(sum_lin / total_area, float3(1.0 / 2.2));
                output_img.Store(int2(opx, opy), float4(blended, 1.0));
            } else {
                output_img.Store(int2(opx, opy), (center_color).unpack_rgba());
            }
        } else if (!need_aa) {
            output_img.Store(int2(opx, opy), (center_color).unpack_rgba());
        } else {
            // Try dual-curve AA first; fall through to single-curve if any
            // gate fails. Dual-curve handles two distinct-chain curves
            // passing through a pixel that isn't a structural junction.
            // Both paths share build_aa_line + line_coverage_pos.
            bool handled = false;
            SlimCP sc_a = sh_cps[uint(best_idx)];
            ResolveData rd_a = read_resolve_data(sc_a.ci);
            AaLine la = build_aa_line(sc_a, rd_a, best_t);

            // ---- Dual-curve attempt ----
            if (la.valid && la.pos_color != la.neg_color && cached_n >= 2) {
                int sec_slot = find_second_chain_hit(
                    sc_a.ci, best_idx,
                    cached_idx_0, cached_idx_1, cached_idx_2,
                    cached_d2_0, cached_d2_1, cached_d2_2,
                    cached_n, aa_threshold);
                if (sec_slot >= 0) {
                    int sec_idx = (sec_slot == 0) ? cached_idx_0
                                : (sec_slot == 1) ? cached_idx_1
                                : cached_idx_2;
                    float sec_t = (sec_slot == 0) ? cached_t_0
                                : (sec_slot == 1) ? cached_t_1
                                : cached_t_2;
                    SlimCP sc_b = sh_cps[uint(sec_idx)];
                    ResolveData rd_b = read_resolve_data(sc_b.ci);
                    AaLine lb = build_aa_line(sc_b, rd_b, sec_t);
                    if (lb.valid && lb.pos_color != lb.neg_color) {
                        // Inner-side: which side of A faces B (and vice versa)?
                        float2 delta_ba = lb.cpt - la.cpt;
                        bool inner_a_pos = dot(delta_ba, la.normal) > 0.0;
                        bool inner_b_pos = dot(-delta_ba, lb.normal) > 0.0;
                        uint a_inner = inner_a_pos ? la.pos_color : la.neg_color;
                        uint b_inner = inner_b_pos ? lb.pos_color : lb.neg_color;
                        // Dual-curve assumes a shared middle region. If
                        // inner colors disagree, geometry has 3 distinct
                        // colors (junction missed by structural detector).
                        if (a_inner == b_inner) {
                            float d_a = dot(center - la.cpt, la.normal) / inv_scale;
                            float d_b = dot(center - lb.cpt, lb.normal) / inv_scale;
                            float fa_pos = line_coverage_pos(la.normal, d_a);
                            float fb_pos = line_coverage_pos(lb.normal, d_b);
                            float fa_outer = inner_a_pos ? (1.0 - fa_pos) : fa_pos;
                            float fb_outer = inner_b_pos ? (1.0 - fb_pos) : fb_pos;
                            // Lines crossing inside pixel → A_outer ∩ B_outer
                            // overlap → fractions sum > 1. Defer.
                            if (fa_outer + fb_outer <= 1.0 + 1e-4) {
                                float fm = max(0.0, 1.0 - fa_outer - fb_outer);
                                uint a_outer = inner_a_pos ? la.neg_color : la.pos_color;
                                uint b_outer = inner_b_pos ? lb.neg_color : lb.pos_color;
                                float3 lin = srgb_to_linear(a_outer) * fa_outer
                                           + srgb_to_linear(b_outer) * fb_outer
                                           + srgb_to_linear(a_inner) * fm;
                                output_img.Store(int2(opx, opy),
                                    float4(pow(lin, float3(1.0 / 2.2)), 1.0));
                                handled = true;
                            }
                        }
                    }
                }
            }

            // ---- Single-curve fallback ----
            if (!handled) {
                if (!la.valid || la.pos_color == la.neg_color) {
                    output_img.Store(int2(opx, opy), (center_color).unpack_rgba());
                } else {
                    float d_p = dot(center - la.cpt, la.normal) / inv_scale;
                    float frac = line_coverage_pos(la.normal, d_p);
                    float3 lin = srgb_to_linear(la.pos_color) * frac
                               + srgb_to_linear(la.neg_color) * (1.0 - frac);
                    output_img.Store(int2(opx, opy),
                        float4(pow(lin, float3(1.0 / 2.2)), 1.0));
                }
            }
        }
    }
}
