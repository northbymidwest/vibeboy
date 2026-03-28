// LCD grid effect -- simulates subpixel structure of an LCD panel.
// Each source pixel is rendered with visible R/G/B subpixel columns
// and inter-pixel gaps.

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
[[vk::binding(0, 2)]] ConstantBuffer<Uniforms> u;

[[vk::binding(0, 0)]] StructuredBuffer<uint> pixels;
[[vk::image_format("rgba8")]]
[[vk::binding(0, 1)]] WTexture2D<float4> output_img;

static const float BLEED = 0.12;
static const float BOOST = 2.2;
static const float GAP_BRIGHTNESS = 0.08;
static const float HGAP_BRIGHTNESS = 0.12;

float4 fetch(int2 coord) { return px_rgba(pixels, int(u.src_w), int(u.src_h), coord); }

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint opx = tid.x;
    uint opy = tid.y;
    if (opx >= u.out_w || opy >= u.out_h) return;

    uint scale = max(u.iscale, 3u);

    // Source pixel and position within the scaled block
    int2 src_i = int2(opx / scale, opy / scale);
    uint px = opx % scale;  // column within pixel block
    uint py = opy % scale;  // row within pixel block

    float4 c = fetch(src_i);

    // Determine subpixel phase: map column to R(0), G(1), B(2)
    uint phase = (px * 3u) / scale;

    // Vertical gap: last column (only if scale > 3)
    bool is_vgap = (scale > 3u) && (px == scale - 1u);
    // Horizontal gap: last row
    bool is_hgap = (py == scale - 1u);

    float3 rgb;

    if (is_vgap || is_hgap) {
        // Gap regions: very dim
        rgb = c.rgb * GAP_BRIGHTNESS;
    } else {
        // Subpixel mask: amplify the primary channel, bleed the others
        float3 mask;
        if (phase == 0u)      mask = float3(1.0, BLEED, BLEED);
        else if (phase == 1u) mask = float3(BLEED, 1.0, BLEED);
        else                  mask = float3(BLEED, BLEED, 1.0);

        rgb = min(c.rgb * mask * BOOST, float3(1.0));
    }

    output_img.Store(int2(opx, opy), float4(rgb, 1.0));
}
