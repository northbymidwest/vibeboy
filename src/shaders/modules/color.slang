// Color packing, unpacking, and perceptual distance metrics.
//
// All color operations are extension methods on uint:
//   c.unpack_rgba()  c.unpack_rgb()  c.channel(shift)

module;

// ── Extension methods on uint ───────────────────────────────────────────────

public extension uint {
    /// Unpack this packed XRGB8888 color to float4 (R, G, B, 1.0).
    [ForceInline] public float4 unpack_rgba() {
        return float4(
            float((this >> 16u) & 0xFFu) / 255.0,
            float((this >> 8u)  & 0xFFu) / 255.0,
            float( this         & 0xFFu) / 255.0,
            1.0
        );
    }

    /// Unpack this packed XRGB8888 color to float3 (R, G, B).
    [ForceInline] public float3 unpack_rgb() {
        return float3(
            float((this >> 16u) & 0xFFu) / 255.0,
            float((this >> 8u)  & 0xFFu) / 255.0,
            float( this         & 0xFFu) / 255.0
        );
    }

    /// Extract a single 8-bit channel (shift: 16=R, 8=G, 0=B).
    [ForceInline] public uint channel(uint shift) { return (this >> shift) & 0xFFu; }
}

// ── Free functions ──────────────────────────────────────────────────────────

/// Coverage-weighted blend of two packed colors.
[ForceInline] public uint blend_over(uint bg, uint fg, uint fg_coverage, uint total) {
    uint inv = total - fg_coverage;
    uint r = (bg.channel(16u) * inv + fg.channel(16u) * fg_coverage) / total;
    uint g = (bg.channel( 8u) * inv + fg.channel( 8u) * fg_coverage) / total;
    uint b = (bg.channel( 0u) * inv + fg.channel( 0u) * fg_coverage) / total;
    return 0xFF000000u | (r << 16u) | (g << 8u) | b;
}

/// BT.601 YUV color-difference check (hqx thresholds: Y>48, U>7, V>6).
[ForceInline] public bool yuv_differs(float4 a, float4 b) {
    float3 d = (a.rgb - b.rgb) * 255.0;
    float dy = abs( 0.299 * d.r + 0.587 * d.g + 0.114 * d.b);
    float du = abs(-0.169 * d.r - 0.331 * d.g + 0.500 * d.b);
    float dv = abs( 0.500 * d.r - 0.419 * d.g - 0.081 * d.b);
    return dy > 48.0 || du > 7.0 || dv > 6.0;
}

/// BT.601 weighted YUV squared distance (xBR).
[ForceInline] public float yuv_dist_sq(float4 a, float4 b) {
    float3 d = (a.rgb - b.rgb) * 255.0;
    float dy  =  0.299 * d.r + 0.587 * d.g + 0.114 * d.b;
    float dcb = -0.169 * d.r - 0.331 * d.g + 0.500 * d.b;
    float dcr =  0.500 * d.r - 0.419 * d.g - 0.081 * d.b;
    return 48.0 * dy * dy + 7.0 * dcb * dcb + 6.0 * dcr * dcr;
}

/// BT.2020 YCbCr Euclidean distance (xBRZ).
[ForceInline] public float ycbcr_dist(float4 a, float4 b) {
    float3 d = (a.rgb - b.rgb) * 255.0;
    static const float K_R = 0.2627;
    static const float K_G = 0.6780;
    static const float K_B = 0.0593;
    static const float S_B = 0.5 / (1.0 - K_B);
    static const float S_R = 0.5 / (1.0 - K_R);
    float y  = K_R * d.r + K_G * d.g + K_B * d.b;
    float cb = S_B * (d.b - y);
    float cr = S_R * (d.r - y);
    return sqrt(y * y + cb * cb + cr * cr);
}
