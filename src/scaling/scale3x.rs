//! Scale3x — 3x pixel-art scaling.
//!
//! Extension of Scale2x to 3x3 output blocks. Uses the same guard
//! condition (B!=H && D!=F) and adds diagonal neighbor checks for
//! the edge and corner pixels.
//!
//! For each pixel E with neighbors:
//!     A B C
//!     D E F
//!     G H I
//! Output 3x3 block (E0..E8), with rules only active when B!=H && D!=F.

use super::get;

pub fn scale(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let dst_w = src_w * 3;
    let dst_h = src_h * 3;
    let mut dst = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let ix = x as isize;
            let iy = y as isize;
            let a = get(src, src_w, src_h, ix - 1, iy - 1);
            let b = get(src, src_w, src_h, ix, iy - 1);
            let c = get(src, src_w, src_h, ix + 1, iy - 1);
            let d = get(src, src_w, src_h, ix - 1, iy);
            let e = get(src, src_w, src_h, ix, iy);
            let f = get(src, src_w, src_h, ix + 1, iy);
            let g = get(src, src_w, src_h, ix - 1, iy + 1);
            let h = get(src, src_w, src_h, ix, iy + 1);
            let i = get(src, src_w, src_h, ix + 1, iy + 1);

            let (e0, e1, e2, e3, e4, e5, e6, e7, e8) = if b != h && d != f {
                (
                    if d == b { d } else { e },
                    if (d == b && e != c) || (b == f && e != a) { b } else { e },
                    if b == f { f } else { e },
                    if (d == b && e != g) || (d == h && e != a) { d } else { e },
                    e,
                    if (b == f && e != i) || (h == f && e != c) { f } else { e },
                    if d == h { d } else { e },
                    if (d == h && e != i) || (h == f && e != g) { h } else { e },
                    if h == f { f } else { e },
                )
            } else {
                (e, e, e, e, e, e, e, e, e)
            };

            let dx = x * 3;
            let dy = y * 3;
            dst[dy * dst_w + dx] = e0;
            dst[dy * dst_w + dx + 1] = e1;
            dst[dy * dst_w + dx + 2] = e2;
            dst[(dy + 1) * dst_w + dx] = e3;
            dst[(dy + 1) * dst_w + dx + 1] = e4;
            dst[(dy + 1) * dst_w + dx + 2] = e5;
            dst[(dy + 2) * dst_w + dx] = e6;
            dst[(dy + 2) * dst_w + dx + 1] = e7;
            dst[(dy + 2) * dst_w + dx + 2] = e8;
        }
    }
    dst
}
