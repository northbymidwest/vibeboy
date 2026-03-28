//! NEDI — New Edge-Directed Interpolation (2x upscaling).
//!
//! Uses local covariance estimated from a small neighborhood to adapt
//! interpolation weights along detected edges. Falls back to bilinear
//! when the local covariance matrix is near-singular.
//!
//! Reference: Li & Orchard, "New Edge-Directed Interpolation" (2001).

use super::get;

/// Luma from ARGB pixel.
#[inline(always)]
fn luma(p: u32) -> f32 {
    let r = ((p >> 16) & 0xFF) as f32;
    let g = ((p >> 8) & 0xFF) as f32;
    let b = (p & 0xFF) as f32;
    0.299 * r + 0.587 * g + 0.114 * b
}

/// Blend four ARGB colors with weights (must sum to ~1.0).
#[inline(always)]
fn blend4(c: [u32; 4], w: [f32; 4]) -> u32 {
    let mut r = 0.0f32;
    let mut g = 0.0f32;
    let mut b = 0.0f32;
    for i in 0..4 {
        r += ((c[i] >> 16) & 0xFF) as f32 * w[i];
        g += ((c[i] >> 8) & 0xFF) as f32 * w[i];
        b += (c[i] & 0xFF) as f32 * w[i];
    }
    let r = r.round().clamp(0.0, 255.0) as u32;
    let g = g.round().clamp(0.0, 255.0) as u32;
    let b = b.round().clamp(0.0, 255.0) as u32;
    0xFF000000 | (r << 16) | (g << 8) | b
}

/// Compute NEDI weights for a diamond interpolation point.
///
/// The diamond point sits at (x+0.5, y+0.5) between four pixels:
///   d0=(x,y)  d1=(x+1,y)  d2=(x,y+1)  d3=(x+1,y+1)
///
/// We estimate local covariance from nearby diamond-pattern samples
/// (already known pixels at integer positions that also form diamond
/// patterns around known interpolated points).
///
/// Returns weights [w0, w1, w2, w3] for [d0, d1, d2, d3].
#[inline]
fn nedi_diamond_weights(src: &[u32], w: usize, h: usize, x: isize, y: isize) -> [f32; 4] {
    // Training samples: use the 4 diagonal neighbors' luma values
    // at known positions to build a local covariance model.
    //
    // For the diamond point at (x+0.5, y+0.5), the training set uses
    // nearby integer-grid pixels in a diamond pattern around this point.

    // Neighborhood size for covariance estimation (M training samples)
    // Use the 4x4 neighborhood of diamond-pattern pixels.
    let mut c = [[0.0f32; 4]; 12]; // training input vectors
    let mut y_vals = [0.0f32; 12]; // training output (known pixel values)
    let mut count = 0usize;

    // For each known pixel near (x+0.5, y+0.5), its 4 diagonal neighbors
    // form a training sample. The known pixel is the output.
    for dy in -1..=2_isize {
        for dx in -1..=2_isize {
            // Known pixel at (x+dx, y+dy) — only use those that form
            // a checkerboard pattern offset from our target
            if (dx + dy) % 2 != 0 {
                continue;
            }
            if count >= 12 {
                break;
            }
            let px = x + dx;
            let py = y + dy;
            let center = luma(get(src, w, h, px, py));

            // Its 4 diagonal neighbors
            let n0 = luma(get(src, w, h, px - 1, py - 1));
            let n1 = luma(get(src, w, h, px + 1, py - 1));
            let n2 = luma(get(src, w, h, px - 1, py + 1));
            let n3 = luma(get(src, w, h, px + 1, py + 1));

            c[count] = [n0, n1, n2, n3];
            y_vals[count] = center;
            count += 1;
        }
    }

    if count < 4 {
        return [0.25, 0.25, 0.25, 0.25];
    }

    // Build C^T * C (4x4) and C^T * y (4x1)
    let mut ctc = [[0.0f32; 4]; 4];
    let mut cty = [0.0f32; 4];

    for k in 0..count {
        for i in 0..4 {
            cty[i] += c[k][i] * y_vals[k];
            for j in 0..4 {
                ctc[i][j] += c[k][i] * c[k][j];
            }
        }
    }

    // Add small regularization to prevent singularity
    for i in 0..4 {
        ctc[i][i] += 0.001;
    }

    // Solve 4x4 system via Cramer's rule or simple Gaussian elimination
    if let Some(weights) = solve_4x4(&ctc, &cty) {
        // Normalize weights to sum to 1
        let sum: f32 = weights.iter().sum();
        if sum.abs() > 0.01 {
            let inv = 1.0 / sum;
            [weights[0] * inv, weights[1] * inv, weights[2] * inv, weights[3] * inv]
        } else {
            [0.25, 0.25, 0.25, 0.25]
        }
    } else {
        [0.25, 0.25, 0.25, 0.25]
    }
}

/// Solve a 4x4 linear system Ax = b using Gaussian elimination with partial pivoting.
fn solve_4x4(a: &[[f32; 4]; 4], b: &[f32; 4]) -> Option<[f32; 4]> {
    let mut m = [[0.0f32; 5]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = a[i][j];
        }
        m[i][4] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..4 {
        // Find pivot
        let mut max_row = col;
        let mut max_val = m[col][col].abs();
        for row in (col + 1)..4 {
            if m[row][col].abs() > max_val {
                max_val = m[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-10 {
            return None;
        }
        if max_row != col {
            m.swap(col, max_row);
        }
        let pivot = m[col][col];
        for row in (col + 1)..4 {
            let factor = m[row][col] / pivot;
            for j in col..5 {
                m[row][j] -= factor * m[col][j];
            }
        }
    }

    // Back substitution
    let mut x = [0.0f32; 4];
    for i in (0..4).rev() {
        let mut sum = m[i][4];
        for j in (i + 1)..4 {
            sum -= m[i][j] * x[j];
        }
        if m[i][i].abs() < 1e-10 {
            return None;
        }
        x[i] = sum / m[i][i];
    }
    Some(x)
}

pub fn scale(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let dst_w = src_w * 2;
    let dst_h = src_h * 2;
    let mut dst = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let ix = x as isize;
            let iy = y as isize;
            let dx = x * 2;
            let dy = y * 2;

            // Top-left: original pixel
            dst[dy * dst_w + dx] = get(src, src_w, src_h, ix, iy);

            // Top-right: horizontal interpolation (average of left and right)
            let left = get(src, src_w, src_h, ix, iy);
            let right = get(src, src_w, src_h, ix + 1, iy);
            dst[dy * dst_w + dx + 1] = super::blend_argb(left, right, 0.5);

            // Bottom-left: vertical interpolation (average of top and bottom)
            let top = get(src, src_w, src_h, ix, iy);
            let bot = get(src, src_w, src_h, ix, iy + 1);
            dst[(dy + 1) * dst_w + dx] = super::blend_argb(top, bot, 0.5);

            // Bottom-right: NEDI diamond interpolation
            let d0 = get(src, src_w, src_h, ix, iy);
            let d1 = get(src, src_w, src_h, ix + 1, iy);
            let d2 = get(src, src_w, src_h, ix, iy + 1);
            let d3 = get(src, src_w, src_h, ix + 1, iy + 1);
            let w = nedi_diamond_weights(src, src_w, src_h, ix, iy);

            // Clamp weights to avoid ringing
            let w = [
                w[0].clamp(-0.25, 1.25),
                w[1].clamp(-0.25, 1.25),
                w[2].clamp(-0.25, 1.25),
                w[3].clamp(-0.25, 1.25),
            ];
            let sum: f32 = w.iter().sum();
            let w = if sum.abs() > 0.01 {
                let inv = 1.0 / sum;
                [w[0] * inv, w[1] * inv, w[2] * inv, w[3] * inv]
            } else {
                [0.25, 0.25, 0.25, 0.25]
            };

            dst[(dy + 1) * dst_w + dx + 1] = blend4([d0, d1, d2, d3], w);
        }
    }

    dst
}
