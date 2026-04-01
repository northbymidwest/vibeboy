// Super xBR -- edge-directed 2x scaling by Hyllian (3-pass compute version).
//
// This shader handles all 3 passes, selected by the `pass` uniform:
//   pass 0: Diagonal -- compute (odd,odd) positions from source pixels
//   pass 1: Cardinal -- compute (odd,even) and (even,odd) from pass 0 + source
//   pass 2: Polish   -- refine all positions from pass 1 output
//
// Pass 0 and 1 write to an intermediate RGBA buffer (storage buffer).
// Pass 2 writes to the output image.

import modules.color;
import modules.pixel_source;

// SDL3 compute binding convention:
//   set 0 = readonly storage buffers
//   set 1 = readwrite storage buffers
//   set 2 = readwrite storage textures
//   set 3 = uniform buffers
struct Uniforms {
    uint src_w;
    uint src_h;
    uint out_w;
    uint out_h;
    uint pass;   // 0=diagonal, 1=cardinal, 2=polish
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
[[vk::binding(0, 1)]]
#ifdef DXIL_TARGET
RWStructuredBuffer<uint> intermed : register(u0, space1);
#else
RWStructuredBuffer<uint> intermed;
#endif
[[vk::image_format("rgba8")]]
[[vk::binding(0, 2)]]
#ifdef DXIL_TARGET
WTexture2D<float4> output_img : register(u1, space1);
#else
WTexture2D<float4> output_img;
#endif

// -- Module-scope globals --
static float4 g_p[16];
static float g_wp[6];
static float g_i[4];
static float g_e[4];

// -- Constants --
static const float DIAG_WP[6]    = { 2.0, 1.0, -1.0, 4.0, -1.0, 1.0 };
static const float CARD_WP[6]    = { 1.0, 0.0,  0.0, 0.0,  0.0, 0.0 };
static const float POLISH_WP[6]  = { 0.0, 0.0,  0.0, 1.0,  0.0, 0.0 };
static const float DIAG_SHARP  = 1.29633 / 10.0;
static const float ORTHO_SHARP = 1.75068 / 20.0;
static const float EDGE_LIMIT = 5.000001;

// -- Helpers --
uint pack(float4 c) {
    uint r = uint(clamp(c.r, 0.0, 1.0) * 255.0 + 0.5);
    uint g = uint(clamp(c.g, 0.0, 1.0) * 255.0 + 0.5);
    uint b = uint(clamp(c.b, 0.0, 1.0) * 255.0 + 0.5);
    return (r << 16u) | (g << 8u) | b;
}

float4 src_fetch(int2 coord) { return px_rgba(pixels, int(u.src_w), int(u.src_h), coord); }

float4 buf_fetch(int2 coord) {
    coord = clamp(coord, int2(0), int2(int(u.out_w) - 1, int(u.out_h) - 1));
    return (intermed[coord.y * int(u.out_w) + coord.x]).unpack_rgba();
}

void buf_write(int2 coord, float4 c) {
    if (coord.x >= 0 && coord.y >= 0 && coord.x < int(u.out_w) && coord.y < int(u.out_h))
        intermed[coord.y * int(u.out_w) + coord.x] = pack(c);
}

float luma(float4 c) { return dot(c.rgb * 255.0, float3(0.2126, 0.7152, 0.0722)); }
float ld(float a, float b) { return abs(a - b); }

float smooth_threshold(float x, float limit) {
    float t = clamp(x / limit, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

float diag_grad(
    float outer_a, float outer_b,
    float axis_l, float axis_c, float axis_r,
    float core_a, float core_b, float far_a, float far_b,
    float par_a, float par_b, float par_c,
    float cross_a, float cross_b
) {
    return g_wp[0] * (ld(axis_c, axis_r) + ld(axis_c, axis_l)
                  + ld(par_b, par_a) + ld(par_b, par_c))
         + g_wp[1] * (ld(far_a, far_b) + ld(core_a, core_b))
         + g_wp[2] * (ld(core_b, far_b) + ld(core_a, far_a))
         + g_wp[3] * ld(core_b, far_a)
         + g_wp[4] * (ld(axis_l, axis_r) + ld(par_a, par_c))
         + g_wp[5] * (ld(outer_a, outer_b) + ld(cross_a, cross_b));
}

float card_grad() {
    return g_wp[3] * (ld(g_i[0], g_i[1]) + ld(g_i[2], g_i[3]))
         + g_wp[0] * (ld(g_i[0], g_e[0]) + ld(g_i[1], g_e[1]) + ld(g_i[2], g_e[2]) + ld(g_i[3], g_e[3]))
         + g_wp[2] * (ld(g_i[0], g_e[1]) + ld(g_i[2], g_e[3]) + ld(g_e[0], g_i[1]) + ld(g_e[2], g_i[3]));
}

float3 cubic4(float sharp, float4 c0, float4 c1, float4 c2, float4 c3) {
    float t0 = -sharp, t1 = 0.5 + sharp, t2 = 0.5 + sharp, t3 = -sharp;
    return t0*c0.rgb + t1*c1.rgb + t2*c2.rgb + t3*c3.rgb;
}

float3 cubic4_paired(float sharp, float4 a0, float4 b0, float4 a1, float4 b1,
                                float4 a2, float4 b2, float4 a3, float4 b3) {
    float t0 = -sharp, t1 = 0.25 + sharp, t2 = 0.25 + sharp, t3 = -sharp;
    return t0*(a0.rgb+b0.rgb) + t1*(a1.rgb+b1.rgb)
         + t2*(a2.rgb+b2.rgb) + t3*(a3.rgb+b3.rgb);
}

float4 edge_interp() {
    float l[16];
    for (int i = 0; i < 16; i++) l[i] = luma(g_p[i]);

    static const int CTL=0, TL=1, TR=2, CTR=3;
    static const int LT=4,  ITL=5, ITR=6, RT=7;
    static const int LB=8,  IBL=9, IBR=10, RB=11;
    static const int CBL=12, BL=13, BR=14, CBR=15;

    float grad_bs = diag_grad(l[LT], l[TL], l[LB], l[ITL], l[TR],
        l[CBL], l[IBL], l[ITR], l[CTR], l[BL], l[IBR], l[RT], l[BR], l[RB]);
    float grad_sl = diag_grad(l[TR], l[RT], l[TL], l[ITR], l[RB],
        l[CTL], l[ITL], l[IBR], l[CBR], l[LT], l[IBL], l[BR], l[LB], l[BL]);

    float diag_bias = grad_bs - grad_sl;
    float diag_conf = smooth_threshold(abs(diag_bias), EDGE_LIMIT);

    g_i[0] = l[ITR]; g_i[1] = l[IBR]; g_i[2] = l[ITL]; g_i[3] = l[IBL];
    g_e[0] = l[TR]; g_e[1] = l[BR]; g_e[2] = l[TL]; g_e[3] = l[BL];
    float grad_h = card_grad();
    g_i[0] = l[ITL]; g_i[1] = l[ITR]; g_i[2] = l[IBL]; g_i[3] = l[IBR];
    g_e[0] = l[LT]; g_e[1] = l[RT]; g_e[2] = l[LB]; g_e[3] = l[RB];
    float grad_v = card_grad();
    float card_bias = grad_h - grad_v;

    float3 slash     = cubic4(DIAG_SHARP, g_p[CBL], g_p[IBL], g_p[ITR], g_p[CTR]);
    float3 backslash = cubic4(DIAG_SHARP, g_p[CTL], g_p[ITL], g_p[IBR], g_p[CBR]);
    float3 horiz = cubic4_paired(ORTHO_SHARP,
        g_p[LT],g_p[LB], g_p[ITL],g_p[IBL], g_p[ITR],g_p[IBR], g_p[RT],g_p[RB]);
    float3 vert = cubic4_paired(ORTHO_SHARP,
        g_p[TR],g_p[TL], g_p[ITR],g_p[ITL], g_p[IBR],g_p[IBL], g_p[BR],g_p[BL]);

    float3 diag_pick = (diag_bias >= 0.0) ? backslash : slash;
    float3 card_pick = (card_bias >= 0.0) ? vert : horiz;
    float3 result = card_pick + diag_conf * (diag_pick - card_pick);

    float3 lo = min(min(g_p[ITL].rgb, g_p[ITR].rgb), min(g_p[IBL].rgb, g_p[IBR].rgb));
    float3 hi = max(max(g_p[ITL].rgb, g_p[ITR].rgb), max(g_p[IBL].rgb, g_p[IBR].rgb));
    result = clamp(result, lo, hi);

    return float4(result, 1.0);
}

// -- Main --
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    int2 pos = int2(tid.xy);
    if (pos.x >= int(u.out_w) || pos.y >= int(u.out_h)) return;

    if (u.pass == 0u) {
        // Pass 0: Write source pixels to (even,even) and diagonal to (odd,odd)
        int px = pos.x & 1, py = pos.y & 1;
        if (px == 0 && py == 0) {
            // Copy source pixel
            buf_write(pos, src_fetch(pos / 2));
        } else if (px == 1 && py == 1) {
            // Diagonal interpolation
            int2 src = pos / 2;
            for (int j = 0; j < 4; j++)
                for (int i = 0; i < 4; i++)
                    g_p[j*4+i] = src_fetch(src + int2(i-1, j-1));
            g_wp = DIAG_WP;
            buf_write(pos, edge_interp());
        }
        // (odd,even) and (even,odd) positions left uninitialized -- filled in pass 1

    } else if (u.pass == 1u) {
        // Pass 1: Cardinal interpolation for (odd,even) and (even,odd)
        int px = pos.x & 1, py = pos.y & 1;
        if ((px == 1 && py == 0) || (px == 0 && py == 1)) {
            int ox = pos.x, oy = pos.y;
            if (px == 1 && py == 0) {
                // Horizontal midpoint -- rotated 45 degree grid
                g_p[0]  = buf_fetch(int2(ox-3,oy));   g_p[1]  = buf_fetch(int2(ox-2,oy-1));
                g_p[2]  = buf_fetch(int2(ox-1,oy-2)); g_p[3]  = buf_fetch(int2(ox,  oy-3));
                g_p[4]  = buf_fetch(int2(ox-2,oy+1)); g_p[5]  = buf_fetch(int2(ox-1,oy));
                g_p[6]  = buf_fetch(int2(ox,  oy-1)); g_p[7]  = buf_fetch(int2(ox+1,oy-2));
                g_p[8]  = buf_fetch(int2(ox-1,oy+2)); g_p[9]  = buf_fetch(int2(ox,  oy+1));
                g_p[10] = buf_fetch(int2(ox+1,oy));   g_p[11] = buf_fetch(int2(ox+2,oy-1));
                g_p[12] = buf_fetch(int2(ox,  oy+3)); g_p[13] = buf_fetch(int2(ox+1,oy+2));
                g_p[14] = buf_fetch(int2(ox+2,oy+1)); g_p[15] = buf_fetch(int2(ox+3,oy));
            } else {
                // Vertical midpoint -- rotated 45 degree grid (transposed)
                g_p[0]  = buf_fetch(int2(ox,  oy-3)); g_p[1]  = buf_fetch(int2(ox-1,oy-2));
                g_p[2]  = buf_fetch(int2(ox-2,oy-1)); g_p[3]  = buf_fetch(int2(ox-3,oy));
                g_p[4]  = buf_fetch(int2(ox+1,oy-2)); g_p[5]  = buf_fetch(int2(ox,  oy-1));
                g_p[6]  = buf_fetch(int2(ox-1,oy));   g_p[7]  = buf_fetch(int2(ox-2,oy+1));
                g_p[8]  = buf_fetch(int2(ox+2,oy-1)); g_p[9]  = buf_fetch(int2(ox+1,oy));
                g_p[10] = buf_fetch(int2(ox,  oy+1)); g_p[11] = buf_fetch(int2(ox-1,oy+2));
                g_p[12] = buf_fetch(int2(ox+3,oy));   g_p[13] = buf_fetch(int2(ox+2,oy+1));
                g_p[14] = buf_fetch(int2(ox+1,oy+2)); g_p[15] = buf_fetch(int2(ox,  oy+3));
            }
            g_wp = CARD_WP;
            buf_write(pos, edge_interp());
        }

    } else {
        // Pass 2: Polish -- refine all positions
        for (int j = 0; j < 4; j++)
            for (int i = 0; i < 4; i++)
                g_p[j*4+i] = buf_fetch(pos + int2(i-2, j-2));
        g_wp = POLISH_WP;
        output_img.Store(pos, edge_interp());
    }
}
