//! HQx (High Quality Magnification) scaling algorithms.
//!
//! Implements hq2x, hq3x, and hq4x pixel-art scaling filters based on
//! Maxim Stepin's original algorithm. Uses YUV color-space distance comparison
//! to detect edges, then applies smooth interpolation patterns.
//!
//! Each function takes a source buffer of u32 ARGB pixels and produces a
//! scaled output buffer.

// ── YUV color distance ──────────────────────────────────────────────────────

/// Default color-distance threshold (tuned to match the original hqx C code).
const THRESHOLD: u32 = 48;
/// Luma weight for Y difference.
const Y_WEIGHT: u32 = 48;
/// Chroma weight for U difference.
const U_WEIGHT: u32 = 7;
/// Chroma weight for V difference.
const V_WEIGHT: u32 = 6;

/// Convert ARGB u32 to (Y, U, V) components, each as i32.
#[inline(always)]
fn to_yuv(c: u32) -> (i32, i32, i32) {
    let r = ((c >> 16) & 0xFF) as i32;
    let g = ((c >> 8) & 0xFF) as i32;
    let b = (c & 0xFF) as i32;
    let y = (r + g + b) as i32;         // simplified luminance (0..765)
    let u = (r - b) as i32;             // -255..255
    let v = (2 * g - r - b) as i32;     // -510..510
    (y, u, v)
}

/// Return true if two colors are "different" (exceed the YUV threshold).
#[inline(always)]
fn diff(a: u32, b: u32) -> bool {
    if a == b {
        return false;
    }
    let (y1, u1, v1) = to_yuv(a);
    let (y2, u2, v2) = to_yuv(b);
    let dy = (y1 - y2).unsigned_abs();
    let du = (u1 - u2).unsigned_abs();
    let dv = (v1 - v2).unsigned_abs();
    dy * Y_WEIGHT + du * U_WEIGHT + dv * V_WEIGHT > THRESHOLD * Y_WEIGHT
}

// ── Pixel interpolation helpers ─────────────────────────────────────────────

/// Linear interpolation: (a + b) / 2
#[inline(always)]
fn interp_2(a: u32, b: u32) -> u32 {
    // Per-channel average without overflow
    let mask = 0xFEFEFEFE_u32;
    ((a & mask) >> 1) + ((b & mask) >> 1) + (a & b & 0x01010101)
}

/// Weighted interpolation: (2a + b) / 3
#[inline(always)]
fn interp_3(a: u32, b: u32) -> u32 {
    blend(a, b, 2, 1)
}

/// Weighted interpolation: (2a + b + c) / 4
#[inline(always)]
fn interp_4(a: u32, b: u32, c: u32) -> u32 {
    blend3(a, b, c, 2, 1, 1)
}

/// Weighted interpolation: (a + b) / 2  (same as interp_2)
#[inline(always)]
fn interp_5(a: u32, b: u32) -> u32 {
    interp_2(a, b)
}

/// Weighted: (5a + 2b + c) / 8
#[inline(always)]
fn interp_6(a: u32, b: u32, c: u32) -> u32 {
    blend3(a, b, c, 5, 2, 1)
}

/// Weighted: (6a + b + c) / 8
#[inline(always)]
fn interp_7(a: u32, b: u32, c: u32) -> u32 {
    blend3(a, b, c, 6, 1, 1)
}

/// Weighted: (5a + 3b) / 8
#[inline(always)]
fn interp_8(a: u32, b: u32) -> u32 {
    blend(a, b, 5, 3)
}

/// Weighted: (2a + 3b + 3c) / 8
#[inline(always)]
fn interp_9(a: u32, b: u32, c: u32) -> u32 {
    blend3(a, b, c, 2, 3, 3)
}

/// Weighted: (14a + b + c) / 16
#[inline(always)]
fn interp_10(a: u32, b: u32, c: u32) -> u32 {
    blend3(a, b, c, 14, 1, 1)
}

/// Two-component blend: (wa*a + wb*b) / (wa+wb)
#[inline(always)]
fn blend(a: u32, b: u32, wa: u32, wb: u32) -> u32 {
    let total = wa + wb;
    let r = (((a >> 16) & 0xFF) * wa + ((b >> 16) & 0xFF) * wb) / total;
    let g = (((a >> 8) & 0xFF) * wa + ((b >> 8) & 0xFF) * wb) / total;
    let bl = ((a & 0xFF) * wa + (b & 0xFF) * wb) / total;
    0xFF000000 | (r << 16) | (g << 8) | bl
}

/// Three-component blend: (wa*a + wb*b + wc*c) / (wa+wb+wc)
#[inline(always)]
fn blend3(a: u32, b: u32, c: u32, wa: u32, wb: u32, wc: u32) -> u32 {
    let total = wa + wb + wc;
    let r = (((a >> 16) & 0xFF) * wa + ((b >> 16) & 0xFF) * wb + ((c >> 16) & 0xFF) * wc) / total;
    let g = (((a >> 8) & 0xFF) * wa + ((b >> 8) & 0xFF) * wb + ((c >> 8) & 0xFF) * wc) / total;
    let bl = ((a & 0xFF) * wa + (b & 0xFF) * wb + (c & 0xFF) * wc) / total;
    0xFF000000 | (r << 16) | (g << 8) | bl
}

// ── Neighbor sampling ───────────────────────────────────────────────────────

/// Neighbor indices:
///   0 1 2
///   3 4 5
///   6 7 8
/// where 4 is the center pixel.
struct Neighbors {
    w: [u32; 9],
    /// Bit pattern: bit i set if w[i] differs from w[4]
    pattern: u16,
}

impl Neighbors {
    fn sample(src: &[u32], src_w: usize, src_h: usize, x: usize, y: usize) -> Self {
        let c = src[y * src_w + x];

        let x0 = if x > 0 { x - 1 } else { 0 };
        let x2 = if x + 1 < src_w { x + 1 } else { src_w - 1 };
        let y0 = if y > 0 { y - 1 } else { 0 };
        let y2 = if y + 1 < src_h { y + 1 } else { src_h - 1 };

        let w = [
            src[y0 * src_w + x0], // 0: top-left
            src[y0 * src_w + x],  // 1: top
            src[y0 * src_w + x2], // 2: top-right
            src[y * src_w + x0],  // 3: left
            c,                    // 4: center
            src[y * src_w + x2],  // 5: right
            src[y2 * src_w + x0], // 6: bottom-left
            src[y2 * src_w + x],  // 7: bottom
            src[y2 * src_w + x2], // 8: bottom-right
        ];

        let mut pattern: u16 = 0;
        for i in 0..9 {
            if i != 4 && diff(c, w[i]) {
                pattern |= 1 << i;
            }
        }

        Neighbors { w, pattern }
    }
}

// ── Scaling mode enum ───────────────────────────────────────────────────────

/// HQx scaling factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HqxScale {
    Hq2x,
    Hq3x,
    Hq4x,
}

impl HqxScale {
    pub fn factor(self) -> u32 {
        match self {
            HqxScale::Hq2x => 2,
            HqxScale::Hq3x => 3,
            HqxScale::Hq4x => 4,
        }
    }
}

/// Scaling mode for the renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleFilter {
    Nearest,
    Hqx(HqxScale),
}

impl ScaleFilter {
    pub fn factor(self) -> u32 {
        match self {
            ScaleFilter::Nearest => 1,
            ScaleFilter::Hqx(h) => h.factor(),
        }
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Scale a source image using hq2x.
/// `src` must have `src_w * src_h` pixels.
/// Returns a buffer of `(src_w*2) * (src_h*2)` pixels.
pub fn hq2x(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let dst_w = src_w * 2;
    let dst_h = src_h * 2;
    let mut dst = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let n = Neighbors::sample(src, src_w, src_h, x, y);
            let mut out = [n.w[4]; 4]; // [top-left, top-right, bottom-left, bottom-right]
            hq2x_kernel(&n, &mut out);
            let dx = x * 2;
            let dy = y * 2;
            dst[dy * dst_w + dx] = out[0];
            dst[dy * dst_w + dx + 1] = out[1];
            dst[(dy + 1) * dst_w + dx] = out[2];
            dst[(dy + 1) * dst_w + dx + 1] = out[3];
        }
    }

    dst
}

/// Scale a source image using hq3x.
pub fn hq3x(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let dst_w = src_w * 3;
    let dst_h = src_h * 3;
    let mut dst = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let n = Neighbors::sample(src, src_w, src_h, x, y);
            let mut out = [n.w[4]; 9];
            hq3x_kernel(&n, &mut out);
            let dx = x * 3;
            let dy = y * 3;
            for oy in 0..3 {
                for ox in 0..3 {
                    dst[(dy + oy) * dst_w + dx + ox] = out[oy * 3 + ox];
                }
            }
        }
    }

    dst
}

/// Scale a source image using hq4x.
pub fn hq4x(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let dst_w = src_w * 4;
    let dst_h = src_h * 4;
    let mut dst = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let n = Neighbors::sample(src, src_w, src_h, x, y);
            let mut out = [n.w[4]; 16];
            hq4x_kernel(&n, &mut out);
            let dx = x * 4;
            let dy = y * 4;
            for oy in 0..4 {
                for ox in 0..4 {
                    dst[(dy + oy) * dst_w + dx + ox] = out[oy * 4 + ox];
                }
            }
        }
    }

    dst
}

/// Apply the appropriate hqx scale to a source buffer.
pub fn scale(src: &[u32], src_w: usize, src_h: usize, mode: HqxScale) -> Vec<u32> {
    match mode {
        HqxScale::Hq2x => hq2x(src, src_w, src_h),
        HqxScale::Hq3x => hq3x(src, src_w, src_h),
        HqxScale::Hq4x => hq4x(src, src_w, src_h),
    }
}

// ── HQ2x kernel ─────────────────────────────────────────────────────────────

/// Compute 4 output pixels (2x2) for center pixel based on neighbor pattern.
/// out layout: [0]=top-left, [1]=top-right, [2]=bottom-left, [3]=bottom-right
fn hq2x_kernel(n: &Neighbors, out: &mut [u32; 4]) {
    let w = &n.w;
    let p = n.pattern;
    let c = w[4];

    // Each output pixel is determined by which neighbors differ from center.
    // This uses simplified hq2x rules derived from the original lookup tables.

    // Top-left pixel
    out[0] = if !diff_bit(p, 3) && !diff_bit(p, 1) {
        interp_2(interp_2(c, w[3]), interp_2(c, w[1]))
    } else if !diff_bit(p, 3) && diff_bit(p, 1) {
        interp_2(c, w[3])
    } else if diff_bit(p, 3) && !diff_bit(p, 1) {
        interp_2(c, w[1])
    } else if !diff(w[3], w[1]) {
        interp_2(c, interp_2(w[3], w[1]))
    } else {
        c
    };

    // Top-right pixel
    out[1] = if !diff_bit(p, 1) && !diff_bit(p, 5) {
        interp_2(interp_2(c, w[1]), interp_2(c, w[5]))
    } else if !diff_bit(p, 1) && diff_bit(p, 5) {
        interp_2(c, w[1])
    } else if diff_bit(p, 1) && !diff_bit(p, 5) {
        interp_2(c, w[5])
    } else if !diff(w[1], w[5]) {
        interp_2(c, interp_2(w[1], w[5]))
    } else {
        c
    };

    // Bottom-left pixel
    out[2] = if !diff_bit(p, 7) && !diff_bit(p, 3) {
        interp_2(interp_2(c, w[7]), interp_2(c, w[3]))
    } else if !diff_bit(p, 7) && diff_bit(p, 3) {
        interp_2(c, w[7])
    } else if diff_bit(p, 7) && !diff_bit(p, 3) {
        interp_2(c, w[3])
    } else if !diff(w[7], w[3]) {
        interp_2(c, interp_2(w[7], w[3]))
    } else {
        c
    };

    // Bottom-right pixel
    out[3] = if !diff_bit(p, 5) && !diff_bit(p, 7) {
        interp_2(interp_2(c, w[5]), interp_2(c, w[7]))
    } else if !diff_bit(p, 5) && diff_bit(p, 7) {
        interp_2(c, w[5])
    } else if diff_bit(p, 5) && !diff_bit(p, 7) {
        interp_2(c, w[7])
    } else if !diff(w[5], w[7]) {
        interp_2(c, interp_2(w[5], w[7]))
    } else {
        c
    };
}

// ── HQ3x kernel ─────────────────────────────────────────────────────────────

/// Compute 9 output pixels (3x3) for center pixel.
/// out layout: row-major [0..8] = 3x3 grid
fn hq3x_kernel(n: &Neighbors, out: &mut [u32; 9]) {
    let w = &n.w;
    let p = n.pattern;
    let c = w[4];

    // Center pixel always stays
    out[4] = c;

    // Top-left corner (out[0])
    out[0] = if !diff_bit(p, 3) && !diff_bit(p, 1) {
        interp_2(interp_2(c, w[3]), interp_2(c, w[1]))
    } else if !diff_bit(p, 3) && diff_bit(p, 1) {
        interp_3(c, w[3])
    } else if diff_bit(p, 3) && !diff_bit(p, 1) {
        interp_3(c, w[1])
    } else if !diff(w[3], w[1]) {
        interp_4(c, w[3], w[1])
    } else {
        c
    };

    // Top center (out[1])
    out[1] = if !diff_bit(p, 1) {
        interp_3(c, w[1])
    } else {
        c
    };

    // Top-right corner (out[2])
    out[2] = if !diff_bit(p, 1) && !diff_bit(p, 5) {
        interp_2(interp_2(c, w[1]), interp_2(c, w[5]))
    } else if !diff_bit(p, 1) && diff_bit(p, 5) {
        interp_3(c, w[1])
    } else if diff_bit(p, 1) && !diff_bit(p, 5) {
        interp_3(c, w[5])
    } else if !diff(w[1], w[5]) {
        interp_4(c, w[1], w[5])
    } else {
        c
    };

    // Middle-left (out[3])
    out[3] = if !diff_bit(p, 3) {
        interp_3(c, w[3])
    } else {
        c
    };

    // Middle-right (out[5])
    out[5] = if !diff_bit(p, 5) {
        interp_3(c, w[5])
    } else {
        c
    };

    // Bottom-left corner (out[6])
    out[6] = if !diff_bit(p, 7) && !diff_bit(p, 3) {
        interp_2(interp_2(c, w[7]), interp_2(c, w[3]))
    } else if !diff_bit(p, 7) && diff_bit(p, 3) {
        interp_3(c, w[7])
    } else if diff_bit(p, 7) && !diff_bit(p, 3) {
        interp_3(c, w[3])
    } else if !diff(w[7], w[3]) {
        interp_4(c, w[7], w[3])
    } else {
        c
    };

    // Bottom center (out[7])
    out[7] = if !diff_bit(p, 7) {
        interp_3(c, w[7])
    } else {
        c
    };

    // Bottom-right corner (out[8])
    out[8] = if !diff_bit(p, 5) && !diff_bit(p, 7) {
        interp_2(interp_2(c, w[5]), interp_2(c, w[7]))
    } else if !diff_bit(p, 5) && diff_bit(p, 7) {
        interp_3(c, w[5])
    } else if diff_bit(p, 5) && !diff_bit(p, 7) {
        interp_3(c, w[7])
    } else if !diff(w[5], w[7]) {
        interp_4(c, w[5], w[7])
    } else {
        c
    };

    // Diagonal refinement: blend corners with diagonals when they're similar
    if !diff_bit(p, 0) && diff_bit(p, 3) && diff_bit(p, 1) {
        out[0] = interp_3(out[0], w[0]);
    }
    if !diff_bit(p, 2) && diff_bit(p, 1) && diff_bit(p, 5) {
        out[2] = interp_3(out[2], w[2]);
    }
    if !diff_bit(p, 6) && diff_bit(p, 7) && diff_bit(p, 3) {
        out[6] = interp_3(out[6], w[6]);
    }
    if !diff_bit(p, 8) && diff_bit(p, 5) && diff_bit(p, 7) {
        out[8] = interp_3(out[8], w[8]);
    }
}

// ── HQ4x kernel ─────────────────────────────────────────────────────────────

/// Compute 16 output pixels (4x4) for center pixel.
/// out layout: row-major [0..15] = 4x4 grid
fn hq4x_kernel(n: &Neighbors, out: &mut [u32; 16]) {
    let w = &n.w;
    let p = n.pattern;
    let c = w[4];

    // Start with all center color
    for px in out.iter_mut() {
        *px = c;
    }

    // ── Top-left quadrant (out[0], out[1], out[4], out[5]) ──
    if !diff_bit(p, 3) && !diff_bit(p, 1) {
        // Smooth corner: blend with both neighbors
        out[0] = interp_2(interp_2(c, w[3]), interp_2(c, w[1]));
        out[1] = interp_3(c, w[1]);
        out[4] = interp_3(c, w[3]);
        out[5] = c;
    } else if !diff_bit(p, 3) {
        out[0] = interp_3(c, w[3]);
        out[4] = interp_3(c, w[3]);
        out[1] = c;
        out[5] = c;
    } else if !diff_bit(p, 1) {
        out[0] = interp_3(c, w[1]);
        out[1] = interp_3(c, w[1]);
        out[4] = c;
        out[5] = c;
    } else if !diff(w[3], w[1]) {
        out[0] = interp_4(c, w[3], w[1]);
        out[1] = interp_3(c, w[1]);
        out[4] = interp_3(c, w[3]);
        out[5] = c;
    }
    // Diagonal refinement
    if !diff_bit(p, 0) && diff_bit(p, 3) && diff_bit(p, 1) {
        out[0] = interp_3(out[0], w[0]);
    }

    // ── Top-right quadrant (out[2], out[3], out[6], out[7]) ──
    if !diff_bit(p, 1) && !diff_bit(p, 5) {
        out[2] = interp_3(c, w[1]);
        out[3] = interp_2(interp_2(c, w[1]), interp_2(c, w[5]));
        out[6] = c;
        out[7] = interp_3(c, w[5]);
    } else if !diff_bit(p, 1) {
        out[2] = interp_3(c, w[1]);
        out[3] = interp_3(c, w[1]);
        out[6] = c;
        out[7] = c;
    } else if !diff_bit(p, 5) {
        out[2] = c;
        out[3] = interp_3(c, w[5]);
        out[6] = c;
        out[7] = interp_3(c, w[5]);
    } else if !diff(w[1], w[5]) {
        out[2] = interp_3(c, w[1]);
        out[3] = interp_4(c, w[1], w[5]);
        out[6] = c;
        out[7] = interp_3(c, w[5]);
    }
    if !diff_bit(p, 2) && diff_bit(p, 1) && diff_bit(p, 5) {
        out[3] = interp_3(out[3], w[2]);
    }

    // ── Bottom-left quadrant (out[8], out[9], out[12], out[13]) ──
    if !diff_bit(p, 7) && !diff_bit(p, 3) {
        out[8] = interp_3(c, w[3]);
        out[9] = c;
        out[12] = interp_2(interp_2(c, w[7]), interp_2(c, w[3]));
        out[13] = interp_3(c, w[7]);
    } else if !diff_bit(p, 3) {
        out[8] = interp_3(c, w[3]);
        out[9] = c;
        out[12] = interp_3(c, w[3]);
        out[13] = c;
    } else if !diff_bit(p, 7) {
        out[8] = c;
        out[9] = c;
        out[12] = interp_3(c, w[7]);
        out[13] = interp_3(c, w[7]);
    } else if !diff(w[7], w[3]) {
        out[8] = interp_3(c, w[3]);
        out[9] = c;
        out[12] = interp_4(c, w[7], w[3]);
        out[13] = interp_3(c, w[7]);
    }
    if !diff_bit(p, 6) && diff_bit(p, 7) && diff_bit(p, 3) {
        out[12] = interp_3(out[12], w[6]);
    }

    // ── Bottom-right quadrant (out[10], out[11], out[14], out[15]) ──
    if !diff_bit(p, 5) && !diff_bit(p, 7) {
        out[10] = c;
        out[11] = interp_3(c, w[5]);
        out[14] = interp_3(c, w[7]);
        out[15] = interp_2(interp_2(c, w[5]), interp_2(c, w[7]));
    } else if !diff_bit(p, 5) {
        out[10] = c;
        out[11] = interp_3(c, w[5]);
        out[14] = c;
        out[15] = interp_3(c, w[5]);
    } else if !diff_bit(p, 7) {
        out[10] = c;
        out[11] = c;
        out[14] = interp_3(c, w[7]);
        out[15] = interp_3(c, w[7]);
    } else if !diff(w[5], w[7]) {
        out[10] = c;
        out[11] = interp_3(c, w[5]);
        out[14] = interp_3(c, w[7]);
        out[15] = interp_4(c, w[5], w[7]);
    }
    if !diff_bit(p, 8) && diff_bit(p, 5) && diff_bit(p, 7) {
        out[15] = interp_3(out[15], w[8]);
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Check if bit `i` is set in the neighbor-difference pattern.
#[inline(always)]
fn diff_bit(pattern: u16, bit: u8) -> bool {
    pattern & (1 << bit) != 0
}
