// Shared constants for the Kopf-Lischinski vectorize GPU pipeline.

module;

/// Control-point flag: node is a geometric corner (excluded from curvature energy).
public static const uint IS_CORNER    = 16u;

/// Control-point flag: node is a T-junction (position corrected post-optimization).
public static const uint IS_TJUNCTION = 32u;

/// Control-point flag: node is a valence-4 crossing (inverse-corrected so curve
/// passes through grid corner).
public static const uint IS_CROSSING  = 64u;

/// Control-point flag: node is a chain endpoint (prev or next neighbor is -1).
/// The rasterizer treats endpoint neighbors with clamped (knot-multiplicity-3)
/// boundary conditions instead of degenerate self-fallback, so the final span
/// of a chain is a real curved Bezier ending exactly at the endpoint position.
public static const uint IS_ENDPOINT  = 128u;

/// Sentinel color marking void/boundary edges in the scanline rasterizer.
public static const uint VOID_COLOR   = 0x01000000u;
