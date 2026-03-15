#version 450

// Antialiased nearest-neighbor scaling.
//
// Standard nearest-neighbor picks the single closest source pixel. This
// variant considers the area of the source grid that each output pixel
// covers: if an output pixel straddles a source-pixel boundary, the
// overlapping source pixels are blended proportionally to their area
// coverage. At integer scales the result is identical to nearest-neighbor.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 2, binding = 0) uniform sampler2D src_tex;

layout(std140, set = 3, binding = 0) uniform Uniforms {
    vec2 src_size;
    vec2 dst_size;
};

vec4 fetch(ivec2 coord) {
    coord = clamp(coord, ivec2(0), ivec2(src_size) - 1);
    return texelFetch(src_tex, coord, 0);
}

void main() {
    // Size of one output pixel in source coordinates.
    vec2 pixel = src_size / dst_size;

    // Source-space span of this output pixel.
    vec2 src0 = v_uv * src_size - pixel * 0.5;
    vec2 src1 = src0 + pixel;

    // Integer source pixel range covered.
    ivec2 i0 = ivec2(clamp(floor(src0), vec2(0.0), src_size - 1.0));
    ivec2 i1 = ivec2(clamp(floor(src1 - vec2(1e-5)), vec2(0.0), src_size - 1.0));

    if (i0 == i1) {
        // Output pixel falls entirely within one source pixel.
        frag_color = fetch(i0);
    } else {
        // Accumulate weighted color across overlapping source pixels.
        vec3 acc = vec3(0.0);
        float area_total = 0.0;

        for (int iy = i0.y; iy <= i1.y; iy++) {
            float top = max(float(iy), src0.y);
            float bot = min(float(iy + 1), src1.y);
            float h = bot - top;
            if (h <= 0.0) continue;

            for (int ix = i0.x; ix <= i1.x; ix++) {
                float left = max(float(ix), src0.x);
                float right = min(float(ix + 1), src1.x);
                float w = right - left;
                if (w <= 0.0) continue;

                float area = w * h;
                acc += fetch(ivec2(ix, iy)).rgb * area;
                area_total += area;
            }
        }

        frag_color = vec4(acc / max(area_total, 1e-5), 1.0);
    }
}
