// Tile-based nearest-boundary cell rasterizer with adaptive 2x2 AA.
//
// Each workgroup handles a 2x2 source-pixel tile. CPs in the tile's
// neighborhood are loaded into shared memory ONCE, then all output
// pixels in the tile evaluate against shared memory.

import modules.color;
import modules.bspline;

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
[[vk::image_format("rgba8")]]
[[vk::binding(0, 1)]]
#ifdef DXIL_TARGET
WTexture2D<float4> output_img : register(u0, space1);
#else
WTexture2D<float4> output_img;
#endif

static const uint MAX_TILE_CPS = 98u;  // 7x7 corners x 2 slots

struct SlimCP {
    float2 pos;
    float2 prev_pos;
    float2 next_pos;
    float2 bbox_min;
    float2 bbox_max;
    int ci;
    int _pad;
};

groupshared SlimCP sh_cps[MAX_TILE_CPS];
groupshared Atomic<uint> sh_num_cps;

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
    } else if (t < 0.5) {
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

struct SHit { float d2; float t; int idx; };

uint sample_from_shared(float2 pt, uint fallback_color,
                        out int out_idx, out float out_t, out float out_d2) {
    SHit hits[4];
    int num_hits = 0;
    for (int i = 0; i < 4; i++) { hits[i].d2 = 1e10; hits[i].idx = -1; }

    uint n = sh_num_cps.load();
    for (uint i = 0u; i < n; i++) {
        SlimCP sc = sh_cps[i];

        if (pt.x < sc.bbox_min.x - 1.0 || pt.x > sc.bbox_max.x + 1.0 ||
            pt.y < sc.bbox_min.y - 1.0 || pt.y > sc.bbox_max.y + 1.0)
            continue;

        float2 quick = bspline_quick_screen(sc.prev_pos, sc.pos, sc.next_pos, pt);
        if (quick.y > 2.0) continue;

        float2 result = bspline_refine(sc.prev_pos, sc.pos, sc.next_pos, pt,
                                       quick.x, quick.y);
        float span_best_t = result.x;
        float span_best_d2 = result.y;

        if (span_best_d2 < 1.0) {
            for (int h = 0; h < 4; h++) {
                if (span_best_d2 < hits[h].d2) {
                    for (int j = 3; j > h; j--) hits[j] = hits[j-1];
                    hits[h].d2 = span_best_d2;
                    hits[h].t = span_best_t;
                    hits[h].idx = int(i);
                    if (num_hits < 4) num_hits++;
                    break;
                }
            }
        }
    }

    out_idx = (num_hits > 0) ? hits[0].idx : -1;
    out_t = (num_hits > 0) ? hits[0].t : 0.0;
    out_d2 = (num_hits > 0) ? hits[0].d2 : 1e10;

    for (int h = 0; h < num_hits; h++) {
        SlimCP sc = sh_cps[uint(hits[h].idx)];
        ResolveData rd = read_resolve_data(sc.ci);
        uint c;
        if (resolve_color(pt, sc, rd, hits[h].t, c)) {
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
        int center_cx = src_x0 + 1;
        int center_cy = src_y0 + 1;

        for (uint slot_idx = local_tid; slot_idx < 98u; slot_idx += 256u) {
            int local_idx = int(slot_idx) / 2;
            int slot = int(slot_idx) % 2;
            int dy = local_idx / 7 - 3;
            int dx = local_idx % 7 - 3;
            int cx = center_cx + dx;
            int cy = center_cy + dy;

            if (cx < 0 || cy < 0 || cx >= int(u.corners_w) || cy > int(u.img_h)) continue;
            int ci = (cy * int(u.corners_w) + cx) * 2 + slot;
            uint flag = cp_flags[ci];
            if (flag == 0u) continue;

            int prev_ci = cp_neighbors[ci * 4];
            int next_ci = cp_neighbors[ci * 4 + 1];
            if (prev_ci < 0 && next_ci < 0) continue;

            uint idx;
            idx = sh_num_cps.add(1u);
            if (idx < MAX_TILE_CPS) {
                float2 cp = read_pos(ci);
                float2 pp = (prev_ci >= 0) ? read_pos(prev_ci) : cp;
                float2 np = (next_ci >= 0) ? read_pos(next_ci) : cp;
                sh_cps[idx].pos = cp;
                sh_cps[idx].prev_pos = pp;
                sh_cps[idx].next_pos = np;
                sh_cps[idx].bbox_min = min(min(pp, cp), np);
                sh_cps[idx].bbox_max = max(max(pp, cp), np);
                sh_cps[idx].ci = ci;
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
        uint center_color = sample_from_shared(center, fallback, best_idx, best_t, best_d2);

        float aa_threshold = 2.0 / (u.scale * u.scale);
        bool need_aa = (best_idx >= 0 && best_d2 < aa_threshold);

        if (!need_aa) {
            output_img.Store(int2(opx, opy), (center_color).unpack_rgba());
        } else {
            SlimCP sc = sh_cps[uint(best_idx)];
            ResolveData rd = read_resolve_data(sc.ci);

            float2 cpt = bspline_eval(sc.prev_pos, sc.pos, sc.next_pos, best_t);
            float2 tang = bspline_deriv(sc.prev_pos, sc.pos, sc.next_pos, best_t);
            float tl = length(tang);

            if (tl < 1e-4) {
                output_img.Store(int2(opx, opy), (center_color).unpack_rgba());
            } else {
                float2 normal = float2(-tang.y, tang.x) / tl;
                float d = dot(center - cpt, normal);

                uint aa_pl = 0u, aa_pr = 0u, aa_nl = 0u, aa_nr = 0u;
                if (rd.prev_dir >= 0) get_edge_colors(rd.icx, rd.icy, rd.prev_dir, aa_pl, aa_pr);
                if (rd.next_dir >= 0) get_edge_colors(rd.icx, rd.icy, rd.next_dir, aa_nl, aa_nr);

                uint color_left, color_right;
                bool have_edge = false;
                float ref_t;

                if (best_t < 0.5 && aa_pl != aa_pr) {
                    color_left = aa_pl; color_right = aa_pr; ref_t = 0.0; have_edge = true;
                } else if (best_t >= 0.5 && aa_nl != aa_nr) {
                    color_left = aa_nr; color_right = aa_nl; ref_t = 1.0; have_edge = true;
                } else if (aa_nl != aa_nr) {
                    color_left = aa_nr; color_right = aa_nl; ref_t = 1.0; have_edge = true;
                } else if (aa_pl != aa_pr) {
                    color_left = aa_pl; color_right = aa_pr; ref_t = 0.0; have_edge = true;
                }

                if (!have_edge) {
                    output_img.Store(int2(opx, opy), (center_color).unpack_rgba());
                } else {
                    float2 orig_tang = bspline_deriv(rd.orig_prev, rd.orig_pos, rd.orig_next, ref_t);
                    float2 orig_norm = float2(-orig_tang.y, orig_tang.x);
                    bool flip = dot(normal, orig_norm) < 0.0;
                    uint pos_side = flip ? color_right : color_left;
                    uint neg_side = flip ? color_left : color_right;

                    if (pos_side == neg_side) {
                        output_img.Store(int2(opx, opy), (center_color).unpack_rgba());
                    } else {
                        float pw = inv_scale;
                        float proj_w = pw * max(abs(normal.x), abs(normal.y));
                        float frac = clamp(0.5 + d / proj_w, 0.0, 1.0);

                        // Blend in linear light (sRGB decode → blend → sRGB encode)
                        float3 c0 = float3(float((pos_side >> 16u) & 0xFFu),
                                           float((pos_side >> 8u) & 0xFFu),
                                           float(pos_side & 0xFFu)) / 255.0;
                        float3 c1 = float3(float((neg_side >> 16u) & 0xFFu),
                                           float((neg_side >> 8u) & 0xFFu),
                                           float(neg_side & 0xFFu)) / 255.0;
                        c0 = pow(c0, float3(2.2));
                        c1 = pow(c1, float3(2.2));
                        float3 blended = pow(frac * c0 + (1.0 - frac) * c1, float3(1.0 / 2.2));
                        output_img.Store(int2(opx, opy), float4(blended, 1.0));
                    }
                }
            }
        }
    }
}
