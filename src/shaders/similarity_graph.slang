// Build similarity graph: connect pixels with identical colors.
// No YUV threshold — any color difference creates a boundary edge.
//
// Each thread processes one pixel. Output is a (2W+1)×(2H+1) buffer:
//   Odd coords (2x+1, 2y+1) = pixel color
//   Even coords = edge/corner flags:
//     Horizontal (2x+2, 2y+1): 1 if pixel(x,y) == pixel(x+1,y)
//     Vertical (2x+1, 2y+2): 1 if pixel(x,y) == pixel(x,y+1)
//     Corner (2x+2, 2y+2): bit 0 = down-right diagonal (x,y)→(x+1,y+1)
//                           bit 1 = down-left diagonal (x+1,y)→(x,y+1)
//
// Also writes an 8-bit valence mask per pixel to `valence`, indexed by
// (y * img_w + x). Bits match the DIR_* constants in resolve_crossings
// (NW=1, W=2, SW=4, S=8, SE=16, E=32, NE=64, N=128). Lets resolve_crossings
// replace its 8-fetch valence_mask helper with one buffer read.

struct Uniforms {
    uint img_w;
    uint img_h;
    uint graph_stride; // = 2*img_w + 1
    uint _pad;
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
[[vk::binding(0, 1)]]
#ifdef DXIL_TARGET
RWStructuredBuffer<uint> graph : register(u0, space1);
#else
RWStructuredBuffer<uint> graph;
#endif
[[vk::binding(1, 1)]]
#ifdef DXIL_TARGET
RWStructuredBuffer<uint> valence : register(u1, space1);
#else
RWStructuredBuffer<uint> valence;
#endif

uint px(uint x, uint y) { return pixels[y * u.img_w + x]; }

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint x = tid.x;
    uint y = tid.y;
    if (x >= u.img_w || y >= u.img_h) return;

    uint c = px(x, y);

    // Store pixel color at odd coordinate
    graph[(2u * y + 1u) * u.graph_stride + (2u * x + 1u)] = c;

    // Horizontal edge to right neighbor
    if (x + 1u < u.img_w) {
        graph[(2u * y + 1u) * u.graph_stride + (2u * x + 2u)] =
            (c == px(x + 1u, y)) ? 1u : 0u;
    }

    // Vertical edge to down neighbor
    if (y + 1u < u.img_h) {
        graph[(2u * y + 2u) * u.graph_stride + (2u * x + 1u)] =
            (c == px(x, y + 1u)) ? 1u : 0u;
    }

    // Diagonal corner: both diagonals written by the upper-left pixel of the 2×2 block
    if (x + 1u < u.img_w && y + 1u < u.img_h) {
        uint flags = 0u;
        if (c == px(x + 1u, y + 1u)) flags |= 1u;
        if (px(x + 1u, y) == px(x, y + 1u)) flags |= 2u;
        graph[(2u * y + 2u) * u.graph_stride + (2u * x + 2u)] = flags;
    }

    // Border zeros
    if (x == 0u) graph[(2u * y + 1u) * u.graph_stride] = 0u;
    if (y == 0u) graph[2u * x + 1u] = 0u;

    // 8-bit valence mask: each bit set when the corresponding 8-neighbor
    // pixel shares this pixel's color. Diagonals use the same logic as
    // the corner-flag computation above (TL==BR for backslash, TR==BL for
    // slash) — color equality with the diagonal neighbor.
    uint v = 0u;
    if (x > 0u           && c == px(x - 1u, y))         v |= 2u;   // W
    if (x + 1u < u.img_w && c == px(x + 1u, y))         v |= 32u;  // E
    if (y > 0u           && c == px(x, y - 1u))         v |= 128u; // N
    if (y + 1u < u.img_h && c == px(x, y + 1u))         v |= 8u;   // S
    if (x > 0u           && y > 0u           && c == px(x - 1u, y - 1u)) v |= 1u;  // NW
    if (x + 1u < u.img_w && y > 0u           && c == px(x + 1u, y - 1u)) v |= 64u; // NE
    if (x > 0u           && y + 1u < u.img_h && c == px(x - 1u, y + 1u)) v |= 4u;  // SW
    if (x + 1u < u.img_w && y + 1u < u.img_h && c == px(x + 1u, y + 1u)) v |= 16u; // SE
    valence[y * u.img_w + x] = v;
}
