//! EPX (Eric's Pixel Expansion) — 2x pixel-art scaling.
//!
//! Developed by Eric Johnston at LucasArts (~1992) for porting SCUMM engine
//! games to early color Macintosh. Each pixel becomes a 2x2 block with
//! corners smoothed based on cardinal neighbor matching.
//!
//! For each pixel P with neighbors:
//!       A
//!     C P B
//!       D
//! Output:
//!   1 = (C==A && C!=D && A!=B) ? A : P
//!   2 = (A==B && A!=C && B!=D) ? B : P
//!   3 = (D==C && D!=B && C!=A) ? C : P
//!   4 = (B==D && B!=A && D!=C) ? D : P

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
            let a = get(src, src_w, src_h, ix, iy - 1);
            let b = get(src, src_w, src_h, ix + 1, iy);
            let c = get(src, src_w, src_h, ix - 1, iy);
            let d = get(src, src_w, src_h, ix, iy + 1);

            let p1 = if c == a && c != d && a != b { a } else { p };
            let p2 = if a == b && a != c && b != d { b } else { p };
            let p3 = if d == c && d != b && c != a { c } else { p };
            let p4 = if b == d && b != a && d != c { d } else { p };

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
