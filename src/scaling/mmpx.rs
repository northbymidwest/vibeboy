//! MMPX — Style-Preserving Pixel-Art Magnification (2x).
//!
//! Algorithm by Morgan McGuire & Mara Gagiu (2021).
//! Rule-based 2x upscaler that extends EPX with additional clauses for
//! thick lines, intersections, triangle tips, and 2:1 slopes.
//! All output values are chosen from existing source pixels — no new
//! colors are introduced.

use super::get;

/// Luma approximation incorporating alpha (assumes bright background).
/// Transparent pixels get high luma (treated as background).
#[inline(always)]
fn luma(c: u32) -> u32 {
    let r = (c >> 16) & 0xFF;
    let g = (c >> 8) & 0xFF;
    let b = c & 0xFF;
    let a = (c >> 24) & 0xFF;
    (r + g + b + 1) * (256 - a)
}

#[inline(always)]
fn all_eq2(b: u32, a0: u32, a1: u32) -> bool {
    ((b ^ a0) | (b ^ a1)) == 0
}

#[inline(always)]
fn all_eq3(b: u32, a0: u32, a1: u32, a2: u32) -> bool {
    ((b ^ a0) | (b ^ a1) | (b ^ a2)) == 0
}

#[inline(always)]
fn all_eq4(b: u32, a0: u32, a1: u32, a2: u32, a3: u32) -> bool {
    ((b ^ a0) | (b ^ a1) | (b ^ a2) | (b ^ a3)) == 0
}

#[inline(always)]
fn any_eq3(b: u32, a0: u32, a1: u32, a2: u32) -> bool {
    b == a0 || b == a1 || b == a2
}

#[inline(always)]
fn none_eq2(b: u32, a0: u32, a1: u32) -> bool {
    b != a0 && b != a1
}

#[inline(always)]
fn none_eq4(b: u32, a0: u32, a1: u32, a2: u32, a3: u32) -> bool {
    b != a0 && b != a1 && b != a2 && b != a3
}

pub fn scale(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let dst_w = src_w * 2;
    let dst_h = src_h * 2;
    let mut dst = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let ix = x as isize;
            let iy = y as isize;

            let e = get(src, src_w, src_h, ix, iy);

            // 3x3 neighborhood
            let a = get(src, src_w, src_h, ix - 1, iy - 1);
            let b = get(src, src_w, src_h, ix,     iy - 1);
            let c = get(src, src_w, src_h, ix + 1, iy - 1);
            let d = get(src, src_w, src_h, ix - 1, iy);
            let f = get(src, src_w, src_h, ix + 1, iy);
            let g = get(src, src_w, src_h, ix - 1, iy + 1);
            let h = get(src, src_w, src_h, ix,     iy + 1);
            let i = get(src, src_w, src_h, ix + 1, iy + 1);

            // Default: nearest neighbor
            let mut j = e;
            let mut k = e;
            let mut l = e;
            let mut m = e;

            // Early exit if neighborhood is uniform
            if ((a ^ e) | (b ^ e) | (c ^ e) | (d ^ e) |
                (f ^ e) | (g ^ e) | (h ^ e) | (i ^ e)) != 0
            {
                // Diamond points
                let p = get(src, src_w, src_h, ix,     iy - 2);
                let q = get(src, src_w, src_h, ix - 2, iy);
                let r = get(src, src_w, src_h, ix + 2, iy);
                let s = get(src, src_w, src_h, ix,     iy + 2);

                // Luma values for tie-breaking
                let bl = luma(b);
                let dl = luma(d);
                let el = luma(e);
                let fl = luma(f);
                let hl = luma(h);

                // ── 1:1 slope rules ──

                // J (top-left)
                if d == b && d != h && d != f
                    && (el >= dl || e == a)
                    && any_eq3(e, a, c, g)
                    && (el < dl || a != d || e != p || e != q)
                {
                    j = d;
                }

                // K (top-right)
                if b == f && b != d && b != h
                    && (el >= bl || e == c)
                    && any_eq3(e, a, c, i)
                    && (el < bl || c != b || e != p || e != r)
                {
                    k = b;
                }

                // L (bottom-left)
                if h == d && h != f && h != b
                    && (el >= hl || e == g)
                    && any_eq3(e, a, g, i)
                    && (el < hl || g != h || e != s || e != q)
                {
                    l = h;
                }

                // M (bottom-right)
                if f == h && f != b && f != d
                    && (el >= fl || e == i)
                    && any_eq3(e, c, g, i)
                    && (el < fl || i != h || e != r || e != s)
                {
                    m = f;
                }

                // ── Intersection rules (structural) ──

                if e != f && all_eq4(e, c, i, d, q) && all_eq2(f, b, h)
                    && f != get(src, src_w, src_h, ix + 3, iy)
                {
                    k = f; m = f;
                }

                if e != d && all_eq4(e, a, g, f, r) && all_eq2(d, b, h)
                    && d != get(src, src_w, src_h, ix - 3, iy)
                {
                    j = d; l = d;
                }

                if e != h && all_eq4(e, g, i, b, p) && all_eq2(h, d, f)
                    && h != get(src, src_w, src_h, ix, iy + 3)
                {
                    l = h; m = h;
                }

                if e != b && all_eq4(e, a, c, h, s) && all_eq2(b, d, f)
                    && b != get(src, src_w, src_h, ix, iy - 3)
                {
                    j = b; k = b;
                }

                // ── Intersection rules (triangle tip) ──

                if bl < el && all_eq4(e, g, h, i, s) && none_eq4(e, a, d, c, f) {
                    j = b; k = b;
                }

                if hl < el && all_eq4(e, a, b, c, p) && none_eq4(e, d, g, i, f) {
                    l = h; m = h;
                }

                if fl < el && all_eq4(e, a, d, g, q) && none_eq4(e, b, c, i, h) {
                    k = f; m = f;
                }

                if dl < el && all_eq4(e, c, f, i, r) && none_eq4(e, b, a, g, h) {
                    j = d; l = d;
                }

                // ── 2:1 slope rules ──

                if h != b {
                    if h != a && h != e && h != c {
                        if all_eq3(h, g, f, r)
                            && none_eq2(h, d, get(src, src_w, src_h, ix + 2, iy - 1))
                        {
                            l = m;
                        }
                        if all_eq3(h, i, d, q)
                            && none_eq2(h, f, get(src, src_w, src_h, ix - 2, iy - 1))
                        {
                            m = l;
                        }
                    }

                    if b != i && b != g && b != e {
                        if all_eq3(b, a, f, r)
                            && none_eq2(b, d, get(src, src_w, src_h, ix + 2, iy + 1))
                        {
                            j = k;
                        }
                        if all_eq3(b, c, d, q)
                            && none_eq2(b, f, get(src, src_w, src_h, ix - 2, iy + 1))
                        {
                            k = j;
                        }
                    }
                }

                if f != d {
                    if d != i && d != e && d != c {
                        if all_eq3(d, a, h, s)
                            && none_eq2(d, b, get(src, src_w, src_h, ix + 1, iy + 2))
                        {
                            j = l;
                        }
                        if all_eq3(d, g, b, p)
                            && none_eq2(d, h, get(src, src_w, src_h, ix + 1, iy - 2))
                        {
                            l = j;
                        }
                    }

                    if f != e && f != a && f != g {
                        if all_eq3(f, c, h, s)
                            && none_eq2(f, b, get(src, src_w, src_h, ix - 1, iy + 2))
                        {
                            k = m;
                        }
                        if all_eq3(f, i, b, p)
                            && none_eq2(f, h, get(src, src_w, src_h, ix - 1, iy - 2))
                        {
                            m = k;
                        }
                    }
                }
            }

            // Write 2x2 output block
            let dx = x * 2;
            let dy = y * 2;
            dst[dy * dst_w + dx] = j;
            dst[dy * dst_w + dx + 1] = k;
            dst[(dy + 1) * dst_w + dx] = l;
            dst[(dy + 1) * dst_w + dx + 1] = m;
        }
    }

    dst
}
