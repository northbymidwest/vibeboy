// Shared constants for the Kopf-Lischinski vectorize GPU pipeline.

module;

/// Control-point flag: node is a geometric corner (excluded from curvature energy).
public static const uint IS_CORNER    = 16u;

/// Control-point flag: node is a T-junction (position corrected post-optimization).
public static const uint IS_TJUNCTION = 32u;

/// Sentinel color marking void/boundary edges in the scanline rasterizer.
public static const uint VOID_COLOR   = 0x01000000u;
