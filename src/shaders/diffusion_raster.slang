// Voronoi diffusion rasterizer with packed diagonal state ownership.

import modules.color;

struct Uniforms {
    uint out_w;
    uint out_h;
    uint src_w;
    uint src_h;
    float inv_scale;
    float gauss_k;
    float radius;
    uint _pad;
};
[[vk::binding(0, 2)]] ConstantBuffer<Uniforms> u : register(b0, space2);

[[vk::binding(0, 0)]] StructuredBuffer<uint> src_pixels : register(t0, space0);
[[vk::binding(1, 0)]] StructuredBuffer<uint> src_regions : register(t1, space0);
[[vk::binding(2, 0)]] StructuredBuffer<uint> diag_states : register(t2, space0);
[[vk::image_format("rgba8")]]
[[vk::binding(0, 1)]] WTexture2D<float4> output_img : register(u0, space1);

// Unpack corner diagonal state from packed uint.
// corner: 0=TL, 1=TR, 2=BR, 3=BL
uint get_diag(uint packed, uint corner) {
    return (packed >> (corner * 2u)) & 3u;
}

// Build Voronoi cell vertices for source pixel (px, py).
// Returns vertex count; fills verts[0..count].
int build_cell(uint px, uint py, out float2 verts[8]) {
    uint packed = diag_states[py * u.src_w + px];
    uint tl = get_diag(packed, 0u);
    uint tr = get_diag(packed, 1u);
    uint br = get_diag(packed, 2u);
    uint bl = get_diag(packed, 3u);

    float bx = float(px);
    float by = float(py);
    int n = 0;

    // TL corner (rel=BR)
    if (tl == 1u) {
        verts[n++] = float2(bx - 0.25, by + 0.25);
        verts[n++] = float2(bx + 0.25, by - 0.25);
    } else if (tl == 2u) {
        verts[n++] = float2(bx + 0.25, by + 0.25);
    } else {
        verts[n++] = float2(bx, by);
    }

    // TR corner (rel=BL)
    if (tr == 1u) {
        verts[n++] = float2(bx + 0.75, by + 0.25);
    } else if (tr == 2u) {
        verts[n++] = float2(bx + 0.75, by - 0.25);
        verts[n++] = float2(bx + 1.25, by + 0.25);
    } else {
        verts[n++] = float2(bx + 1.0, by);
    }

    // BR corner (rel=TL)
    if (br == 1u) {
        verts[n++] = float2(bx + 1.25, by + 0.75);
        verts[n++] = float2(bx + 0.75, by + 1.25);
    } else if (br == 2u) {
        verts[n++] = float2(bx + 0.75, by + 0.75);
    } else {
        verts[n++] = float2(bx + 1.0, by + 1.0);
    }

    // BL corner (rel=TR)
    if (bl == 1u) {
        verts[n++] = float2(bx + 0.25, by + 0.75);
    } else if (bl == 2u) {
        verts[n++] = float2(bx + 0.25, by + 1.25);
        verts[n++] = float2(bx - 0.25, by + 0.75);
    } else {
        verts[n++] = float2(bx, by + 1.0);
    }

    return n;
}

// Point-in-convex-polygon via cross-product winding test.
bool point_in_cell(float2 verts[8], int nv, float2 p) {
    int sign = 0;
    for (int i = 0; i < nv; i++) {
        float2 v0 = verts[i];
        float2 v1 = verts[(i + 1) % nv];
        float cross_val = (v1.x - v0.x) * (p.y - v0.y) - (v1.y - v0.y) * (p.x - v0.x);
        int s = (cross_val > 1e-6) ? 1 : ((cross_val < -1e-6) ? -1 : 0);
        if (s != 0) {
            if (sign == 0) sign = s;
            else if (sign != s) return false;
        }
    }
    return true;
}

// Find which source pixel owns this point via Voronoi cell test.
uint find_owner(float2 sp) {
    uint home_x = min(uint(sp.x), u.src_w - 1u);
    uint home_y = min(uint(sp.y), u.src_h - 1u);
    uint home_id = home_y * u.src_w + home_x;

    // Fast path: no diagonals at any corner -> cell is a perfect square
    uint packed = diag_states[home_id];
    if (packed == 0u) return home_id;

    // Check home cell
    float2 verts[8];
    int nv = build_cell(home_x, home_y, verts);
    if (point_in_cell(verts, nv, sp)) return home_id;

    // Check 8 neighbors
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = int(home_x) + dx;
            int ny = int(home_y) + dy;
            if (nx < 0 || ny < 0 || nx >= int(u.src_w) || ny >= int(u.src_h)) continue;
            nv = build_cell(uint(nx), uint(ny), verts);
            if (point_in_cell(verts, nv, sp)) return uint(ny) * u.src_w + uint(nx);
        }
    }

    return home_id; // fallback
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint ox = tid.x;
    uint oy = tid.y;
    if (ox >= u.out_w || oy >= u.out_h) return;

    // Output pixel center in source coordinates
    float sx = (float(ox) + 0.5) * u.inv_scale;
    float sy = (float(oy) + 0.5) * u.inv_scale;

    // Find owner via Voronoi cell geometry
    uint owner = find_owner(float2(sx, sy));
    uint my_region = src_regions[owner];

    // Search window in source pixels
    float r_sq = u.radius * u.radius;
    int min_px = max(int(floor(sx - u.radius)), 0);
    int max_px = min(int(ceil(sx + u.radius)), int(u.src_w) - 1);
    int min_py = max(int(floor(sy - u.radius)), 0);
    int max_py = min(int(ceil(sy + u.radius)), int(u.src_h) - 1);

    // Gaussian-weighted color accumulation
    float tr = 0.0, tg = 0.0, tb = 0.0, tw = 0.0;

    for (int py = min_py; py <= max_py; py++) {
        for (int px = min_px; px <= max_px; px++) {
            uint idx = uint(py) * u.src_w + uint(px);
            if (src_regions[idx] != my_region) continue;

            float dx = sx - (float(px) + 0.5);
            float dy = sy - (float(py) + 0.5);
            float d_sq = dx * dx + dy * dy;
            if (d_sq > r_sq) continue;

            float w = exp(-d_sq * u.gauss_k);
            uint c = src_pixels[idx];
            tr += w * float(c.channel(16u));
            tg += w * float(c.channel(8u));
            tb += w * float(c.channel(0u));
            tw += w;
        }
    }

    float4 rgba;
    if (tw > 0.0) {
        float inv_tw = 1.0 / tw;
        rgba = float4(
            clamp(tr * inv_tw / 255.0, 0.0, 1.0),
            clamp(tg * inv_tw / 255.0, 0.0, 1.0),
            clamp(tb * inv_tw / 255.0, 0.0, 1.0),
            1.0
        );
    } else {
        rgba = float4(0.0, 0.0, 0.0, 1.0);
    }

    output_img.Store(int2(ox, oy), rgba);
}
