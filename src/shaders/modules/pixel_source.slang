// Clamped pixel buffer access with automatic unpacking.
//
// Wraps a StructuredBuffer<uint> + dimensions into a struct with convenience
// methods for clamped reads in packed, float4, and float3 formats.

module;

import modules.color;

/// Clamped read of a raw packed XRGB8888 pixel.
[ForceInline] public uint px_raw(StructuredBuffer<uint> buf, int w, int h, int2 coord) {
    coord = clamp(coord, int2(0), int2(w - 1, h - 1));
    return buf[coord.y * w + coord.x];
}

/// Clamped read of a pixel as float4 (R, G, B, 1.0).
[ForceInline] public float4 px_rgba(StructuredBuffer<uint> buf, int w, int h, int2 coord) {
    return (px_raw(buf, w, h, coord)).unpack_rgba();
}

/// Clamped read of a pixel as float3 (R, G, B).
[ForceInline] public float3 px_rgb(StructuredBuffer<uint> buf, int w, int h, int2 coord) {
    return (px_raw(buf, w, h, coord)).unpack_rgb();
}
