//! xBRZ — pixel-art scaling by Zenju (2x–6x).
//!
//! A refined variant of xBR with improved edge detection using "dominance
//! counting" — each corner tests how many of the surrounding pixels support
//! a diagonal vs orthogonal edge, then applies steep/shallow line detection
//! for more accurate sub-pixel blending.

use super::get;
use super::color_dist;
use super::blend_argb;

/// xBRZ scaling factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XbrzScale {
    Xbrz2x,
    Xbrz3x,
    Xbrz4x,
    Xbrz5x,
    Xbrz6x,
}

impl XbrzScale {
    pub fn factor(self) -> u32 {
        match self {
            XbrzScale::Xbrz2x => 2,
            XbrzScale::Xbrz3x => 3,
            XbrzScale::Xbrz4x => 4,
            XbrzScale::Xbrz5x => 5,
            XbrzScale::Xbrz6x => 6,
        }
    }
}

// ── Color equality threshold ────────────────────────────────────────────────

const EQ_THRESHOLD: f32 = 30.0;
const DOMINANT_DIR_THRESHOLD: f32 = 3.6;
const STEEP_DIR_THRESHOLD: f32 = 2.2;

#[inline(always)]
fn colors_equal(a: u32, b: u32) -> bool {
    color_dist(a, b) < EQ_THRESHOLD
}

// ── Rotated kernel access ──────────────────────────────────────────────────

/// Sample a rotated 4x4 kernel for a given corner.
/// rot: 0=BR, 1=BL (flip x), 2=TL (flip both), 3=TR (flip y)
fn sample_kernel(src: &[u32], w: usize, h: usize, cx: isize, cy: isize, rot: u8) -> [u32; 16] {
    let (dx, dy) = match rot {
        0 => (1_isize, 1_isize),
        1 => (-1_isize, 1_isize),
        2 => (-1_isize, -1_isize),
        3 => (1_isize, -1_isize),
        _ => unreachable!(),
    };

    let mut p = [0u32; 16];
    for j in 0..4_isize {
        for i in 0..4_isize {
            p[(j * 4 + i) as usize] = get(src, w, h, cx + (i - 1) * dx, cy + (j - 1) * dy);
        }
    }
    p
}

// ── Edge classification ─────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum BlendType {
    None,
    Normal,
    Dominant,
}

/// Classify corner blend type and steep/shallow flags from the rotated 4x4 kernel.
/// Kernel layout (oriented toward BR corner):
/// ```text
///   0  1  2  3
///   4  5  6  7
///   8  9 10 11
///  12 13 14 15
/// ```
/// p[5] = center, p[6] = right, p[9] = below, p[10] = diagonal
fn classify_corner(p: &[u32; 16]) -> (BlendType, bool, bool) {
    let f = p[5];
    let g = p[6];
    let j = p[9];
    let k = p[10];

    if colors_equal(f, k) {
        return (BlendType::None, false, false);
    }

    let d_fg = color_dist(f, g);
    let d_fj = color_dist(f, j);
    let d_gk = color_dist(g, k);
    let d_jk = color_dist(j, k);

    let diag_weight = d_fg + d_fj + d_gk + d_jk;
    let ortho_weight = color_dist(f, k) * 2.0;

    if diag_weight >= ortho_weight {
        return (BlendType::None, false, false);
    }

    let is_dominant = diag_weight * DOMINANT_DIR_THRESHOLD < ortho_weight;

    let is_steep = d_fj + d_jk > STEEP_DIR_THRESHOLD * (d_fg + d_gk) &&
                   !colors_equal(f, j) && !colors_equal(p[8], j);
    let is_shallow = d_fg + d_gk > STEEP_DIR_THRESHOLD * (d_fj + d_jk) &&
                     !colors_equal(f, g) && !colors_equal(p[2], g);

    let blend = if is_dominant { BlendType::Dominant } else { BlendType::Normal };
    (blend, is_steep, is_shallow)
}

// ── Rotation helper ─────────────────────────────────────────────────────────

/// Map (row, col) in an NxN block to the rotated position for a given corner.
#[inline(always)]
fn rotate_idx(row: usize, col: usize, n: usize, rot: u8) -> usize {
    match rot {
        0 => row * n + col,
        1 => row * n + (n - 1 - col),
        2 => (n - 1 - row) * n + (n - 1 - col),
        3 => (n - 1 - row) * n + col,
        _ => unreachable!(),
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

pub fn scale(src: &[u32], src_w: usize, src_h: usize, mode: XbrzScale) -> Vec<u32> {
    let n = mode.factor() as usize;
    let dst_w = src_w * n;
    let dst_h = src_h * n;
    let mut dst = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let ix = x as isize;
            let iy = y as isize;
            let center = get(src, src_w, src_h, ix, iy);

            let nn = n * n;
            let mut block = vec![center; nn];

            for rot in 0..4u8 {
                let p = sample_kernel(src, src_w, src_h, ix, iy, rot);
                let (blend_type, steep, shallow) = classify_corner(&p);

                if blend_type != BlendType::None {
                    let right = p[6];
                    let down = p[9];

                    let blend_color = if colors_equal(right, down) {
                        right
                    } else if color_dist(center, right) < color_dist(center, down) {
                        right
                    } else {
                        down
                    };

                    let strong = blend_type == BlendType::Dominant;
                    apply_rotated_blend(&mut block, n, rot, strong, steep, shallow, blend_color);
                }
            }

            let dx = x * n;
            let dy = y * n;
            for r in 0..n {
                for c in 0..n {
                    dst[(dy + r) * dst_w + dx + c] = block[r * n + c];
                }
            }
        }
    }
    dst
}

/// Apply blending for a specific corner rotation directly into the block.
fn apply_rotated_blend(
    block: &mut [u32],
    n: usize,
    rot: u8,
    strong: bool,
    steep: bool,
    shallow: bool,
    blend_color: u32,
) {
    let (a1, a2, a3, a4) = if strong {
        (0.5_f32, 0.375, 0.25, 0.125)
    } else {
        (0.375, 0.25, 0.125, 0.0625)
    };

    let last = n - 1;

    // Corner pixel
    let idx = rotate_idx(last, last, n, rot);
    block[idx] = blend_argb(block[idx], blend_color, a1);

    if n >= 3 {
        let idx1 = rotate_idx(last, last - 1, n, rot);
        let idx2 = rotate_idx(last - 1, last, n, rot);
        block[idx1] = blend_argb(block[idx1], blend_color, a2);
        block[idx2] = blend_argb(block[idx2], blend_color, a2);
    }

    if n >= 4 {
        let idx1 = rotate_idx(last, last - 2, n, rot);
        let idx2 = rotate_idx(last - 2, last, n, rot);
        let idx3 = rotate_idx(last - 1, last - 1, n, rot);
        block[idx1] = blend_argb(block[idx1], blend_color, a3);
        block[idx2] = blend_argb(block[idx2], blend_color, a3);
        block[idx3] = blend_argb(block[idx3], blend_color, a3);

        if steep {
            let idx = rotate_idx(last - 1, last - 2, n, rot);
            block[idx] = blend_argb(block[idx], blend_color, a4);
        }
        if shallow {
            let idx = rotate_idx(last - 2, last - 1, n, rot);
            block[idx] = blend_argb(block[idx], blend_color, a4);
        }
    }

    if n >= 5 {
        let idx1 = rotate_idx(last, last - 3, n, rot);
        let idx2 = rotate_idx(last - 3, last, n, rot);
        let idx3 = rotate_idx(last - 1, last - 2, n, rot);
        let idx4 = rotate_idx(last - 2, last - 1, n, rot);
        block[idx1] = blend_argb(block[idx1], blend_color, a4);
        block[idx2] = blend_argb(block[idx2], blend_color, a4);
        block[idx3] = blend_argb(block[idx3], blend_color, a3);
        block[idx4] = blend_argb(block[idx4], blend_color, a3);

        if steep {
            let idx = rotate_idx(last - 2, last - 2, n, rot);
            block[idx] = blend_argb(block[idx], blend_color, a4);
        }
        if shallow {
            let idx = rotate_idx(last - 2, last - 2, n, rot);
            block[idx] = blend_argb(block[idx], blend_color, a4);
        }
    }

    if n >= 6 {
        let idx1 = rotate_idx(last, last - 4, n, rot);
        let idx2 = rotate_idx(last - 4, last, n, rot);
        block[idx1] = blend_argb(block[idx1], blend_color, a4);
        block[idx2] = blend_argb(block[idx2], blend_color, a4);

        let idx3 = rotate_idx(last - 1, last - 3, n, rot);
        let idx4 = rotate_idx(last - 3, last - 1, n, rot);
        block[idx3] = blend_argb(block[idx3], blend_color, a4);
        block[idx4] = blend_argb(block[idx4], blend_color, a4);

        let idx5 = rotate_idx(last - 2, last - 2, n, rot);
        block[idx5] = blend_argb(block[idx5], blend_color, a3);

        if steep {
            let idx = rotate_idx(last - 3, last - 2, n, rot);
            block[idx] = blend_argb(block[idx], blend_color, a4);
        }
        if shallow {
            let idx = rotate_idx(last - 2, last - 3, n, rot);
            block[idx] = blend_argb(block[idx], blend_color, a4);
        }
    }
}
