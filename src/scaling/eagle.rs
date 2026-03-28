//! Eagle — 2x pixel-art scaling.
//!
//! One of the earliest pixel scaling algorithms. Checks diagonal
//! neighbors (3-way corner match) to decide whether to smooth corners.
//!
//! For each pixel P with all 8 neighbors:
//!     S T U
//!     V P W
//!     X Y Z
//! Output 2x2 (starts as P, then):
//!   1 = (S==T && S==V) ? S : P
//!   2 = (T==U && U==W) ? U : P
//!   3 = (V==X && X==Y) ? X : P
//!   4 = (W==Z && Z==Y) ? Z : P

use super::get;

pub fn scale(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let dst_w = src_w * 2;
    let dst_h = src_h * 2;
    let mut dst = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let ix = x as isize;
            let iy = y as isize;
            let p = get(src, src_w, src_h, ix, iy);
            let s = get(src, src_w, src_h, ix - 1, iy - 1);
            let t = get(src, src_w, src_h, ix, iy - 1);
            let u = get(src, src_w, src_h, ix + 1, iy - 1);
            let v = get(src, src_w, src_h, ix - 1, iy);
            let w = get(src, src_w, src_h, ix + 1, iy);
            let xx = get(src, src_w, src_h, ix - 1, iy + 1);
            let yy = get(src, src_w, src_h, ix, iy + 1);
            let z = get(src, src_w, src_h, ix + 1, iy + 1);

            let p1 = if s == t && s == v { s } else { p };
            let p2 = if t == u && u == w { u } else { p };
            let p3 = if v == xx && xx == yy { xx } else { p };
            let p4 = if w == z && z == yy { z } else { p };

            let dx = x * 2;
            let dy = y * 2;
            dst[dy * dst_w + dx] = p1;
            dst[dy * dst_w + dx + 1] = p2;
            dst[(dy + 1) * dst_w + dx] = p3;
            dst[(dy + 1) * dst_w + dx + 1] = p4;
        }
    }
    dst
}
