// ScaleFX 3x pixel-art scaling — GPU compute implementation.
//
// Implementation of the ScaleFX algorithm by Sp00kyFox.
// Five-stage pipeline: color metrics → corner weights → junction arbitration →
// edge tracing (6 levels) → 3x block assembly.
//
// Stages 1-4 dispatch at source resolution, stage 5 at 3x output resolution.
// A single compute entry point branches on the `pass` uniform (0-4).

import modules.color;
import modules.pixel_source;

struct Uniforms {
    uint src_w;
    uint src_h;
    uint out_w;    // = src_w * 3
    uint out_h;    // = src_h * 3
    uint pass;     // 0-4
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[[vk::binding(0, 3)]]
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

[[vk::binding(0, 1)]] RWStructuredBuffer<float4> stage1_out;   // metrics
[[vk::binding(1, 1)]] RWStructuredBuffer<float4> stage2_out;   // corner weights
[[vk::binding(2, 1)]] RWStructuredBuffer<float4> stage3_out;   // junction flags
[[vk::binding(3, 1)]] RWStructuredBuffer<float4> stage4_out;   // edge tags
[[vk::binding(4, 1)]] RWStructuredBuffer<uint>   packed_pixels; // 9x chaining output

[[vk::image_format("rgba8")]]
[[vk::binding(0, 2)]]
#ifdef DXIL_TARGET
WTexture2D<float4> output_img : register(u0, space1);
#else
WTexture2D<float4> output_img;
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

float4 read_pixel(int2 coord) {
    return px_rgba(pixels, int(u.src_w), int(u.src_h), coord);
}

float4 read_buf(RWStructuredBuffer<float4> buf, int2 coord) {
    coord = clamp(coord, int2(0), int2(int(u.src_w) - 1, int(u.src_h) - 1));
    return buf[coord.y * int(u.src_w) + coord.x];
}

// Red-weighted perceptual color distance (Riemersma / compuphase).
float color_distance(float3 p, float3 q) {
    float mean_r = 0.5 * (p.r + q.r);
    float3 delta = p - q;
    float3 w = float3(2.0 + mean_r, 4.0, 3.0 - mean_r);
    return sqrt(dot(w * delta, delta)) / 3.0;
}

// ---------------------------------------------------------------------------
// Stage 1: Color distance metrics
// ---------------------------------------------------------------------------

void stage_metrics(uint2 tid) {
    if (tid.x >= u.src_w || tid.y >= u.src_h) return;
    int2 p = int2(tid);
    float3 center = read_pixel(p).rgb;
    stage1_out[tid.y * u.src_w + tid.x] = float4(
        color_distance(center, read_pixel(p + int2(-1, -1)).rgb),
        color_distance(center, read_pixel(p + int2( 0, -1)).rgb),
        color_distance(center, read_pixel(p + int2( 1, -1)).rgb),
        color_distance(center, read_pixel(p + int2( 1,  0)).rgb)
    );
}

// ---------------------------------------------------------------------------
// Stage 2: Corner interpolation weights
// ---------------------------------------------------------------------------

static const float CLARITY_THRESHOLD = 0.50;

float corner_weight(float edge_dist, float2 primary, float2 secondary) {
    float proximity = max(CLARITY_THRESHOLD - edge_dist, 0.0) / CLARITY_THRESHOLD;
    float bias = primary.x - primary.y;
    float stronger = (min(primary.x, secondary.x) + primary.x > min(primary.y, secondary.y) + primary.y)
                     ? bias : -bias;
    float direction = clamp(1.0 - edge_dist + stronger, 0.0, 1.0);
    return proximity * direction * primary.x * primary.y;
}

void stage_weights(uint2 tid) {
    if (tid.x >= u.src_w || tid.y >= u.src_h) return;
    int2 p = int2(tid);

    float4 tl = read_buf(stage1_out, p + int2(-1, -1));
    float4 t  = read_buf(stage1_out, p + int2( 0, -1));
    float4 l  = read_buf(stage1_out, p + int2(-1,  0));
    float4 c  = read_buf(stage1_out, p);
    float4 r  = read_buf(stage1_out, p + int2( 1,  0));
    float4 bl = read_buf(stage1_out, p + int2(-1,  1));
    float4 b  = read_buf(stage1_out, p + int2( 0,  1));
    float4 br = read_buf(stage1_out, p + int2( 1,  1));

    float4 result;
    result.x = corner_weight(l.z, float2(l.w, c.y), float2(tl.w, l.y));
    result.y = corner_weight(r.x, float2(c.w, c.y), float2(t.w, r.y));
    result.z = corner_weight(b.z, float2(c.w, b.y), float2(b.w, br.y));
    result.w = corner_weight(b.x, float2(l.w, b.y), float2(bl.w, bl.y));

    stage2_out[tid.y * u.src_w + tid.x] = result;
}

// ---------------------------------------------------------------------------
// Stage 3: Junction arbitration and edge classification
// ---------------------------------------------------------------------------

// Branchless comparison helpers matching GPU step semantics.
float4 vec_ge(float4 a, float4 b)  { return 1.0 - step(a, b); }
float4 vec_leq(float4 a, float4 b) { return step(a, b); }

float scalar_ge(float a, float b)  { return 1.0 - step(a, b); }
float scalar_le(float a, float b)  { return 1.0 - step(b, a); }

float4 dominance(float3 a, float3 b, float3 c, float3 d) {
    return 2.0 * float4(a.y, b.y, c.y, d.y)
         - (float4(a.x, b.x, c.x, d.x) + float4(a.z, b.z, c.z, d.z));
}

float junction_condition(float2 diag, float2 orth_a, float2 orth_b) {
    return (diag.x >= max(min(orth_a.x, orth_a.y), min(orth_b.x, orth_b.y)))
        && (diag.y >= max(min(orth_a.x, orth_b.y), min(orth_b.x, orth_a.y)))
        ? 1.0 : 0.0;
}

void stage_junctions(uint2 tid) {
    if (tid.x >= u.src_w || tid.y >= u.src_h) return;
    int2 p = int2(tid);

    // Metric neighborhood
    float4 m_tl = read_buf(stage1_out, p + int2(-1, -1));
    float4 m_t  = read_buf(stage1_out, p + int2( 0, -1));
    float4 m_l  = read_buf(stage1_out, p + int2(-1,  0));
    float4 m_c  = read_buf(stage1_out, p);
    float4 m_r  = read_buf(stage1_out, p + int2( 1,  0));
    float4 m_bl = read_buf(stage1_out, p + int2(-1,  1));
    float4 m_b  = read_buf(stage1_out, p + int2( 0,  1));
    float4 m_br = read_buf(stage1_out, p + int2( 1,  1));

    // Weight neighborhood
    float4 s_tl = read_buf(stage2_out, p + int2(-1, -1));
    float4 s_t  = read_buf(stage2_out, p + int2( 0, -1));
    float4 s_tr = read_buf(stage2_out, p + int2( 1, -1));
    float4 s_l  = read_buf(stage2_out, p + int2(-1,  0));
    float4 s_c  = read_buf(stage2_out, p);
    float4 s_r  = read_buf(stage2_out, p + int2( 1,  0));
    float4 s_bl = read_buf(stage2_out, p + int2(-1,  1));
    float4 s_b  = read_buf(stage2_out, p + int2( 0,  1));
    float4 s_br = read_buf(stage2_out, p + int2( 1,  1));

    // Four junctions around center pixel
    float4 js0 = float4(s_tl.z, s_t.w, s_c.x, s_l.y);
    float4 jd0 = dominance(s_tl.yzw, s_t.zwx, s_c.wxy, s_l.xyz);
    float4 js1 = float4(s_t.z, s_tr.w, s_r.x, s_c.y);
    float4 jd1 = dominance(s_t.yzw, s_tr.zwx, s_r.wxy, s_c.xyz);
    float4 js2 = float4(s_c.z, s_r.w, s_br.x, s_b.y);
    float4 jd2 = dominance(s_c.yzw, s_r.zwx, s_br.wxy, s_b.xyz);
    float4 js3 = float4(s_l.z, s_c.w, s_b.x, s_bl.y);
    float4 jd3 = dominance(s_l.yzw, s_c.zwx, s_b.wxy, s_bl.xyz);

    // Majority vote
    float4 zero4 = float4(0.0);
    float4 v0 = min(vec_ge(jd0, zero4) * (vec_leq(jd0.yzwx, zero4) * vec_leq(jd0.wxyz, zero4) + vec_ge(jd0 + jd0.zwxy, jd0.yzwx + jd0.wxyz)), float4(1.0));
    float4 v1 = min(vec_ge(jd1, zero4) * (vec_leq(jd1.yzwx, zero4) * vec_leq(jd1.wxyz, zero4) + vec_ge(jd1 + jd1.zwxy, jd1.yzwx + jd1.wxyz)), float4(1.0));
    float4 v2 = min(vec_ge(jd2, zero4) * (vec_leq(jd2.yzwx, zero4) * vec_leq(jd2.wxyz, zero4) + vec_ge(jd2 + jd2.zwxy, jd2.yzwx + jd2.wxyz)), float4(1.0));
    float4 v3 = min(vec_ge(jd3, zero4) * (vec_leq(jd3.yzwx, zero4) * vec_leq(jd3.wxyz, zero4) + vec_ge(jd3 + jd3.zwxy, jd3.yzwx + jd3.wxyz)), float4(1.0));

    // Strength injection
    float4 active;
    active.x = min(v0.z + (1.0 - v0.y) * (1.0 - v0.w) * scalar_ge(js0.z, 0.0) * (v0.x + scalar_ge(js0.x + js0.z, js0.y + js0.w)), 1.0);
    active.y = min(v1.w + (1.0 - v1.z) * (1.0 - v1.x) * scalar_ge(js1.w, 0.0) * (v1.y + scalar_ge(js1.y + js1.w, js1.x + js1.z)), 1.0);
    active.z = min(v2.x + (1.0 - v2.w) * (1.0 - v2.y) * scalar_ge(js2.x, 0.0) * (v2.z + scalar_ge(js2.x + js2.z, js2.y + js2.w)), 1.0);
    active.w = min(v3.y + (1.0 - v3.x) * (1.0 - v3.z) * scalar_ge(js3.y, 0.0) * (v3.w + scalar_ge(js3.y + js3.w, js3.x + js3.z)), 1.0);

    // End-of-line suppression
    active = min(active * (float4(v0.z, v1.w, v2.x, v3.y) + 1.0 - active.wxyz * active.yzwx), float4(1.0));

    // Edge classification
    float4 jc;
    jc.x = junction_condition(float2(m_l.z, m_c.x), float2(m_l.w, m_c.y), float2(m_tl.w, m_l.y));
    jc.y = junction_condition(float2(m_r.x, m_c.z), float2(m_c.w, m_c.y), float2(m_t.w, m_r.y));
    jc.z = junction_condition(float2(m_b.z, m_br.x), float2(m_c.w, m_b.y), float2(m_b.w, m_br.y));
    jc.w = junction_condition(float2(m_b.x, m_bl.z), float2(m_l.w, m_b.y), float2(m_bl.w, m_bl.y));

    float4 he = float4(min(m_l.w, m_tl.w), min(m_c.w, m_t.w), min(m_c.w, m_b.w), min(m_l.w, m_bl.w));
    float4 ve = float4(min(m_c.y, m_l.y), min(m_c.y, m_r.y), min(m_b.y, m_br.y), min(m_b.y, m_bl.y));
    float4 hx = float4(m_l.w, m_c.w, m_c.w, m_l.w);
    float4 vx = float4(m_c.y, m_c.y, m_b.y, m_b.y);

    float4 orient = vec_ge(he + hx, ve + vx);
    float4 horiz  = (1.0 - step(ve, he)) * jc;
    float4 vert   = (1.0 - step(he, ve)) * jc;

    stage3_out[tid.y * u.src_w + tid.x] = (active + 2.0 * horiz + 4.0 * vert + 8.0 * orient) / 15.0;
}

// ---------------------------------------------------------------------------
// Stage 4: Multi-level edge tracing
// ---------------------------------------------------------------------------

bool4 decode_corner(float4 v) { return bool4(floor(fmod(v * 15.0 + 0.5, float4(2.0)))); }
bool4 decode_horiz(float4 v)  { return bool4(floor(fmod(v * 7.5  + 0.25, float4(2.0)))); }
bool4 decode_vert(float4 v)   { return bool4(floor(fmod(v * 3.75 + 0.125, float4(2.0)))); }
bool4 decode_orient(float4 v) { return bool4(floor(fmod(v * 1.875 + 0.0625, float4(2.0)))); }

void stage_trace(uint2 tid) {
    if (tid.x >= u.src_w || tid.y >= u.src_h) return;
    int2 p = int2(tid);

    // Read junction flags for center ± 3 in each cardinal direction
    float4 fE  = read_buf(stage3_out, p);
    float4 fL  = read_buf(stage3_out, p + int2(-1,  0));
    float4 fL2 = read_buf(stage3_out, p + int2(-2,  0));
    float4 fL3 = read_buf(stage3_out, p + int2(-3,  0));
    float4 fR  = read_buf(stage3_out, p + int2( 1,  0));
    float4 fR2 = read_buf(stage3_out, p + int2( 2,  0));
    float4 fR3 = read_buf(stage3_out, p + int2( 3,  0));
    float4 fU  = read_buf(stage3_out, p + int2( 0, -1));
    float4 fU2 = read_buf(stage3_out, p + int2( 0, -2));
    float4 fU3 = read_buf(stage3_out, p + int2( 0, -3));
    float4 fD  = read_buf(stage3_out, p + int2( 0,  1));
    float4 fD2 = read_buf(stage3_out, p + int2( 0,  2));
    float4 fD3 = read_buf(stage3_out, p + int2( 0,  3));

    // Decode all needed flags
    bool4 Ec = decode_corner(fE); bool4 Eh = decode_horiz(fE); bool4 Ev = decode_vert(fE); bool4 Eo = decode_orient(fE);
    bool4 Lc = decode_corner(fL); bool4 Lh = decode_horiz(fL); bool4 Lo = decode_orient(fL);
    bool4 L2c = decode_corner(fL2); bool4 L2h = decode_horiz(fL2); bool4 L3h = decode_horiz(fL3);
    bool4 Rc = decode_corner(fR); bool4 Rh = decode_horiz(fR); bool4 Ro = decode_orient(fR);
    bool4 R2c = decode_corner(fR2); bool4 R2h = decode_horiz(fR2); bool4 R3h = decode_horiz(fR3);
    bool4 Uc = decode_corner(fU); bool4 Uv = decode_vert(fU); bool4 Uo = decode_orient(fU);
    bool4 U2c = decode_corner(fU2); bool4 U2v = decode_vert(fU2); bool4 U3v = decode_vert(fU3);
    bool4 Dc = decode_corner(fD); bool4 Dv = decode_vert(fD); bool4 Do = decode_orient(fD);
    bool4 D2c = decode_corner(fD2); bool4 D2v = decode_vert(fD2); bool4 D3v = decode_vert(fD3);

    // Level 1: corner with adjacent support (SFX_SCN=1: always allow)
    bool l1_0 = Ec.x && (Lc.z || Uc.z || true);
    bool l1_1 = Ec.y && (Rc.w || Uc.w || true);
    bool l1_2 = Ec.z && (Rc.x || Dc.x || true);
    bool l1_3 = Ec.w && (Lc.y || Dc.y || true);

    // Level 2: mid-edge (horizontal or vertical continuation)
    bool2 l2_0 = bool2((Ec.x && Eh.y) && Lc.z, (Ec.y && Eh.x) && Rc.w);
    bool2 l2_1 = bool2((Ec.y && Ev.z) && Uc.w, (Ec.z && Ev.y) && Dc.x);
    bool2 l2_2 = bool2((Ec.w && Eh.z) && Lc.y, (Ec.z && Eh.w) && Rc.x);
    bool2 l2_3 = bool2((Ec.x && Ev.w) && Uc.z, (Ec.w && Ev.x) && Dc.y);

    // Level 3: extended corner
    bool2 l3_0 = bool2(l2_0.y && (Lh.y && Lh.x) && Rh.z, l2_3.y && (Uv.w && Uv.x) && Dv.z);
    bool2 l3_1 = bool2(l2_0.x && (Rh.x && Rh.y) && Lh.w, l2_1.y && (Uv.z && Uv.y) && Dv.w);
    bool2 l3_2 = bool2(l2_2.x && (Rh.w && Rh.z) && Lh.x, l2_1.x && (Dv.y && Dv.z) && Uv.x);
    bool2 l3_3 = bool2(l2_2.y && (Lh.z && Lh.w) && Rh.y, l2_3.x && (Dv.x && Dv.w) && Uv.y);

    // Level 4: 4-pixel run
    bool2 l4_0 = bool2(Lc.x && Lh.y && Eh.x && Eh.y && Rh.x && Rh.y && L2c.z && L2h.w,
                        Uc.x && Uv.w && Ev.x && Ev.w && Dv.x && Dv.w && U2c.z && U2v.y);
    bool2 l4_1 = bool2(Rc.y && Rh.x && Eh.y && Eh.x && Lh.y && Lh.x && R2c.w && R2h.z,
                        Uc.y && Uv.z && Ev.y && Ev.z && Dv.y && Dv.z && U2c.w && U2v.x);
    bool2 l4_2 = bool2(Rc.z && Rh.w && Eh.z && Eh.w && Lh.z && Lh.w && R2c.x && R2h.y,
                        Dc.z && Dv.y && Ev.z && Ev.y && Uv.z && Uv.y && D2c.x && D2v.w);
    bool2 l4_3 = bool2(Lc.w && Lh.z && Eh.w && Eh.z && Rh.w && Rh.z && L2c.y && L2h.x,
                        Dc.w && Dv.x && Ev.w && Ev.x && Uv.w && Uv.x && D2c.y && D2v.z);

    // Level 5: mid of 4-pixel run
    bool2 l5_0 = bool2(l4_0.x && R2h.x && R2h.y && L3h.z && L3h.w, l4_1.x && L2h.y && L2h.x && R3h.w && R3h.z);
    bool2 l5_1 = bool2(l4_1.y && D2v.y && D2v.z && U3v.w && U3v.x, l4_2.y && U2v.z && U2v.y && D3v.x && D3v.w);
    bool2 l5_2 = bool2(l4_3.x && R2h.w && R2h.z && L3h.y && L3h.x, l4_2.x && L2h.z && L2h.w && R3h.x && R3h.y);
    bool2 l5_3 = bool2(l4_0.y && D2v.x && D2v.w && U3v.z && U3v.y, l4_3.y && U2v.w && U2v.x && D3v.y && D3v.z);

    // Level 6: 6-pixel run
    bool2 l6_0 = bool2(l5_0.y && L3h.y && L3h.x, l5_3.y && U3v.w && U3v.x);
    bool2 l6_1 = bool2(l5_0.x && R3h.x && R3h.y, l5_1.y && U3v.z && U3v.y);
    bool2 l6_2 = bool2(l5_2.x && R3h.w && R3h.z, l5_1.x && D3v.y && D3v.z);
    bool2 l6_3 = bool2(l5_2.y && L3h.z && L3h.w, l5_3.x && D3v.x && D3v.w);

    // Subpixel tags: 0=center, 1=left, 2=left2, 3=right, 4=right2, 5=up, 6=up2, 7=down, 8=down2
    float4 ct; // corner tags
    ct.x = (l1_0 && Eo.x || l3_0.x && Eo.y || l4_0.x && Lo.x || l6_0.x && Ro.y) ? 5 : (l1_0 || l3_0.y && !Eo.w || l4_0.y && !Uo.x || l6_0.y && !Do.w) ? 1 : l3_0.x ? 3 : l3_0.y ? 7 : l4_0.x ? 2 : l4_0.y ? 6 : l6_0.x ? 4 : l6_0.y ? 8 : 0;
    ct.y = (l1_1 && Eo.y || l3_1.x && Eo.x || l4_1.x && Ro.y || l6_1.x && Lo.x) ? 5 : (l1_1 || l3_1.y && !Eo.z || l4_1.y && !Uo.y || l6_1.y && !Do.z) ? 3 : l3_1.x ? 1 : l3_1.y ? 7 : l4_1.x ? 4 : l4_1.y ? 6 : l6_1.x ? 2 : l6_1.y ? 8 : 0;
    ct.z = (l1_2 && Eo.z || l3_2.x && Eo.w || l4_2.x && Ro.z || l6_2.x && Lo.w) ? 7 : (l1_2 || l3_2.y && !Eo.y || l4_2.y && !Do.z || l6_2.y && !Uo.y) ? 3 : l3_2.x ? 1 : l3_2.y ? 5 : l4_2.x ? 4 : l4_2.y ? 8 : l6_2.x ? 2 : l6_2.y ? 6 : 0;
    ct.w = (l1_3 && Eo.w || l3_3.x && Eo.z || l4_3.x && Lo.w || l6_3.x && Ro.z) ? 7 : (l1_3 || l3_3.y && !Eo.x || l4_3.y && !Do.w || l6_3.y && !Uo.x) ? 1 : l3_3.x ? 3 : l3_3.y ? 5 : l4_3.x ? 2 : l4_3.y ? 8 : l6_3.x ? 4 : l6_3.y ? 6 : 0;

    float4 mt; // mid tags
    mt.x = (l2_0.x &&  Eo.x || l2_0.y &&  Eo.y || l5_0.x &&  Lo.x || l5_0.y &&  Ro.y) ? 5 : l2_0.x ? 1 : l2_0.y ? 3 : l5_0.x ? 2 : l5_0.y ? 4 : (Ec.x && Lc.z && Ec.y && Rc.w) ? ( Eo.x ?  Eo.y ? 5 : 3 : 1) : 0;
    mt.y = (l2_1.x && !Eo.y || l2_1.y && !Eo.z || l5_1.x && !Uo.y || l5_1.y && !Do.z) ? 3 : l2_1.x ? 5 : l2_1.y ? 7 : l5_1.x ? 6 : l5_1.y ? 8 : (Ec.y && Uc.w && Ec.z && Dc.x) ? (!Eo.y ? !Eo.z ? 3 : 7 : 5) : 0;
    mt.z = (l2_2.x &&  Eo.w || l2_2.y &&  Eo.z || l5_2.x &&  Lo.w || l5_2.y &&  Ro.z) ? 7 : l2_2.x ? 1 : l2_2.y ? 3 : l5_2.x ? 2 : l5_2.y ? 4 : (Ec.z && Rc.x && Ec.w && Lc.y) ? ( Eo.z ?  Eo.w ? 7 : 1 : 3) : 0;
    mt.w = (l2_3.x && !Eo.x || l2_3.y && !Eo.w || l5_3.x && !Uo.x || l5_3.y && !Do.w) ? 1 : l2_3.x ? 5 : l2_3.y ? 7 : l5_3.x ? 6 : l5_3.y ? 8 : (Ec.w && Dc.y && Ec.x && Uc.z) ? (!Eo.w ? !Eo.x ? 1 : 5 : 7) : 0;

    stage4_out[tid.y * u.src_w + tid.x] = (ct + 9.0 * mt) / 80.0;
}

// ---------------------------------------------------------------------------
// Stage 5: 3x block assembly
// ---------------------------------------------------------------------------

static const int2 NEIGHBOR_OFFSETS[9] = {
    int2( 0,  0), int2(-1,  0), int2(-2,  0), int2( 1,  0), int2( 2,  0),
    int2( 0, -1), int2( 0, -2), int2( 0,  1), int2( 0,  2),
};

void stage_assemble(uint2 tid) {
    if (tid.x >= u.out_w || tid.y >= u.out_h) return;

    int2 src_pos = int2(tid.xy) / 3;
    int2 block_pos = int2(tid.xy) % 3;

    float4 packed = read_buf(stage4_out, src_pos);
    float4 corner_tags = floor(fmod(packed * 80.0 + 0.5, float4(9.0)));
    float4 mid_tags = floor(fmod(packed * 80.0 / 9.0 + 0.5 / 9.0, float4(9.0)));

    // Map block position to the appropriate tag
    float tag = 0.0;
    if      (block_pos.y == 0 && block_pos.x == 0) tag = corner_tags.x;
    else if (block_pos.y == 0 && block_pos.x == 1) tag = mid_tags.x;
    else if (block_pos.y == 0 && block_pos.x == 2) tag = corner_tags.y;
    else if (block_pos.y == 1 && block_pos.x == 0) tag = mid_tags.w;
    else if (block_pos.y == 1 && block_pos.x == 2) tag = mid_tags.y;
    else if (block_pos.y == 2 && block_pos.x == 0) tag = corner_tags.w;
    else if (block_pos.y == 2 && block_pos.x == 1) tag = mid_tags.z;
    else if (block_pos.y == 2 && block_pos.x == 2) tag = corner_tags.z;
    // (1,1) stays 0 = center pixel

    int2 offset = NEIGHBOR_OFFSETS[clamp(int(tag + 0.5), 0, 8)];
    float4 color = read_pixel(src_pos + offset);
    output_img.Store(int2(tid.xy), color);

    // Write packed XRGB for 9x chaining (second pass reads this as pixel source)
    uint pr = uint(color.r * 255.0 + 0.5);
    uint pg = uint(color.g * 255.0 + 0.5);
    uint pb = uint(color.b * 255.0 + 0.5);
    packed_pixels[tid.y * u.out_w + tid.x] = (pr << 16u) | (pg << 8u) | pb;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    switch (u.pass) {
        case 0: stage_metrics(tid.xy); break;
        case 1: stage_weights(tid.xy); break;
        case 2: stage_junctions(tid.xy); break;
        case 3: stage_trace(tid.xy); break;
        case 4: stage_assemble(tid.xy); break;
        default: break;
    }
}
