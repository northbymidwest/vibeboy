// Bit-manipulation color blending for the 2xSaI family (sai2x, super_sai2x, super_eagle).
//
// These operate on packed XRGB8888 uint values without unpacking to float,
// using bit tricks to avoid overflow.

module;

/// 50/50 blend without overflow (bit-manipulation on packed u32).
[ForceInline] public uint interp(uint a, uint b) {
    return ((a & 0x00FEFEFEu) >> 1u) + ((b & 0x00FEFEFEu) >> 1u) + (a & b & 0x00010101u);
}

/// 25% each of four colors.
[ForceInline] public uint qinterp(uint a, uint b, uint c, uint d) {
    return interp(interp(a, b), interp(c, d));
}

/// 75% a + 25% b.
[ForceInline] public uint interp3_1(uint a, uint b) {
    return interp(a, interp(a, b));
}
