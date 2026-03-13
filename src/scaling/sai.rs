//! 2xSaI, Super 2xSaI, and Super Eagle — classic 2x pixel-art scaling by Kreed.
//!
//! All three algorithms by Derek Liauw Kie Fa (Kreed), circa 1999-2001.
//! They sample a 4x4 neighborhood around each pixel and use color equality
//! tests to detect edges, then interpolate to produce smooth 2x output.

use super::get;

// ── Color helpers ───────────────────────────────────────────────────────────

/// Fast 50/50 color blend without overflow.
#[inline(always)]
fn interp(a: u32, b: u32) -> u32 {
    ((a & 0x00FE_FEFE) >> 1) + ((b & 0x00FE_FEFE) >> 1) + (a & b & 0x0001_0101)
}

/// 25% each of four colors.
#[inline(always)]
fn qinterp(a: u32, b: u32, c: u32, d: u32) -> u32 {
    interp(interp(a, b), interp(c, d))
}

/// 75% a + 25% b.
#[inline(always)]
fn interp3_1(a: u32, b: u32) -> u32 {
    interp(a, interp(a, b))
}

// ── 4x4 neighborhood ───────────────────────────────────────────────────────

/// 4x4 pixel neighborhood around center pixel (index 5).
///
/// ```text
///  p0  p1  p2  p3       (x-1,y-1) (x,y-1) (x+1,y-1) (x+2,y-1)
///  p4  p5  p6  p7       (x-1,y)   (x,y)   (x+1,y)   (x+2,y)
///  p8  p9 p10 p11       (x-1,y+1) (x,y+1) (x+1,y+1) (x+2,y+1)
/// p12 p13 p14 p15       (x-1,y+2) (x,y+2) (x+1,y+2) (x+2,y+2)
/// ```
#[inline(always)]
fn sample4x4(src: &[u32], w: usize, h: usize, x: isize, y: isize) -> [u32; 16] {
    let mut p = [0u32; 16];
    for j in 0..4_isize {
        for i in 0..4_isize {
            p[(j * 4 + i) as usize] = get(src, w, h, x + i - 1, y + j - 1);
        }
    }
    p
}

// ── 2xSaI ───────────────────────────────────────────────────────────────────

/// 2xSaI (Scale and Interpolation) by Kreed.
pub fn scale_2xsai(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let dst_w = src_w * 2;
    let dst_h = src_h * 2;
    let mut dst = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let p = sample4x4(src, src_w, src_h, x as isize, y as isize);

            // Kreed naming: I=p0 E=p1 F=p2 J=p3
            //               G=p4 A=p5 B=p6 K=p7
            //               H=p8 C=p9 D=p10 L=p11
            //               M=p12 N=p13 O=p14 P=p15
            let (i, e, f, j) = (p[0], p[1], p[2], p[3]);
            let (g, a, b, k) = (p[4], p[5], p[6], p[7]);
            let (h, c, d, l) = (p[8], p[9], p[10], p[11]);
            let (_m, _n, o, _p) = (p[12], p[13], p[14], p[15]);

            // Output: top-left=A, top-right=product, bottom-left=product1, bottom-right=product2
            let (product, product1, product2);

            if a == d && b != c {
                product = if (a == e && b == l) || (a == c && a == f && b != e && b == j) {
                    a
                } else {
                    interp(a, b)
                };
                product1 = if (a == g && c == o) || (a == b && a == h && g != c && c == _m) {
                    a
                } else {
                    interp(a, c)
                };
                product2 = a;
            } else if b == c && a != d {
                product = if (b == f && a == h) || (b == e && b == d && a != f && a == i) {
                    b
                } else {
                    interp(a, b)
                };
                product1 = if (c == h && a == f) || (c == g && c == d && a != h && a == i) {
                    c
                } else {
                    interp(a, c)
                };
                product2 = b;
            } else if a == d && b == c {
                if a == b {
                    product = a;
                    product1 = a;
                    product2 = a;
                } else {
                    // Count surrounding pixel affinities to break the tie
                    let mut r: i32 = 0;
                    product = interp(a, b);
                    product1 = interp(a, c);

                    if a == g || a == e { r += 1; }
                    if a == i || a == _m { r += 1; }
                    if b == k || b == f { r -= 1; }
                    if b == j || b == l { r -= 1; }

                    product2 = if r > 0 {
                        a
                    } else if r < 0 {
                        b
                    } else {
                        qinterp(a, b, c, d)
                    };
                }
            } else {
                product2 = qinterp(a, b, c, d);

                product = if a == c && a == f && b != e && b == j {
                    a
                } else if b == e && b == d && a != f && a == i {
                    b
                } else {
                    interp(a, b)
                };

                product1 = if a == b && a == h && g != c && c == _m {
                    a
                } else if c == g && c == d && a != h && a == i {
                    c
                } else {
                    interp(a, c)
                };
            }

            let dx = x * 2;
            let dy = y * 2;
            dst[dy * dst_w + dx] = a;
            dst[dy * dst_w + dx + 1] = product;
            dst[(dy + 1) * dst_w + dx] = product1;
            dst[(dy + 1) * dst_w + dx + 1] = product2;
        }
    }
    dst
}

// ── Super 2xSaI ─────────────────────────────────────────────────────────────

/// Super 2xSaI by Kreed — smoother variant that interpolates all four output pixels.
pub fn scale_super2xsai(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let dst_w = src_w * 2;
    let dst_h = src_h * 2;
    let mut dst = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let p = sample4x4(src, src_w, src_h, x as isize, y as isize);

            let (i, e, f, j) = (p[0], p[1], p[2], p[3]);
            let (g, a, b, k) = (p[4], p[5], p[6], p[7]);
            let (h, c, d, l) = (p[8], p[9], p[10], p[11]);
            let (_m, _n, o, _p) = (p[12], p[13], p[14], p[15]);

            let (product0, product1, product2, product3);

            // Bottom-right output pixel (product3)
            if b == c && a != d {
                product3 = b;
            } else if a == d && b != c {
                product3 = a;
            } else if a == d && b == c {
                if a == b {
                    product3 = a;
                } else {
                    let mut r: i32 = 0;
                    if a == e || a == g { r += 1; }
                    if a == i || a == _m { r += 1; }
                    if b == f || b == k { r -= 1; }
                    if b == j || b == l { r -= 1; }
                    product3 = if r > 0 { a } else if r < 0 { b } else { qinterp(a, b, c, d) };
                }
            } else {
                product3 = qinterp(a, b, c, d);
            }

            // Top-right output pixel (product1)
            if a == d && b != c && a == e && a != l {
                product1 = interp3_1(a, b);
            } else if b == c && a != d && b == f && b != h {
                product1 = interp3_1(b, a);
            } else if b == c && a != d && b == e && b != j {
                product1 = interp3_1(b, a);
            } else if a == d && b != c && a == f && a != k {
                product1 = interp3_1(a, b);
            } else if a == d && b == c {
                product1 = interp(a, b);
            } else if a == c {
                product1 = interp(a, b);
            } else if b == d {
                product1 = interp(a, b);
            } else {
                product1 = interp(a, b);
            }

            // Bottom-left output pixel (product2)
            if a == d && b != c && a == g && a != o {
                product2 = interp3_1(a, c);
            } else if b == c && a != d && c == h && c != e {
                product2 = interp3_1(c, a);
            } else if b == c && a != d && c == g && c != _m {
                product2 = interp3_1(c, a);
            } else if a == d && b != c && a == h && a != l {
                product2 = interp3_1(a, c);
            } else if a == d && b == c {
                product2 = interp(a, c);
            } else if a == b {
                product2 = interp(a, c);
            } else if c == d {
                product2 = interp(a, c);
            } else {
                product2 = interp(a, c);
            }

            // Top-left output pixel (product0) — center-weighted blend
            product0 = if a == d && b != c && (a == e || a == g) {
                a
            } else if b == c && a != d && (b == f || c == h) {
                interp(a, b)
            } else {
                a
            };

            let dx = x * 2;
            let dy = y * 2;
            dst[dy * dst_w + dx] = product0;
            dst[dy * dst_w + dx + 1] = product1;
            dst[(dy + 1) * dst_w + dx] = product2;
            dst[(dy + 1) * dst_w + dx + 1] = product3;
        }
    }
    dst
}

// ── Super Eagle ─────────────────────────────────────────────────────────────

/// GetResult vote: bias toward a or b based on neighbors c, d.
#[inline(always)]
fn get_result(a: u32, b: u32, c: u32, d: u32) -> i32 {
    let mut x = 0i32;
    let mut y = 0i32;
    if a == c { x += 1; } else if b == c { y += 1; }
    if a == d { x += 1; } else if b == d { y += 1; }
    let mut r = 0i32;
    if x <= 1 { r += 1; }
    if y <= 1 { r -= 1; }
    r
}

/// Super Eagle by Kreed — diagonal-aware 2x scaling of the 2x2 block
/// {color5, color6, color2, color3} using extended neighbor voting.
///
/// Snes9x naming mapped to our 4x4 grid:
/// ```text
///              colorB1(p1) colorB2(p2)
///  color4(p4)  color5(p5)  color6(p6) colorS2(p7)
///  color1(p8)  color2(p9)  color3(p10) colorS1(p11)
///              colorA1(p13) colorA2(p14)
/// ```
pub fn scale_super_eagle(src: &[u32], src_w: usize, src_h: usize) -> Vec<u32> {
    let dst_w = src_w * 2;
    let dst_h = src_h * 2;
    let mut dst = vec![0u32; dst_w * dst_h];

    for y in 0..src_h {
        for x in 0..src_w {
            let p = sample4x4(src, src_w, src_h, x as isize, y as isize);

            // Map to Kreed naming
            let color_b1 = p[1];
            let color_b2 = p[2];
            let color4  = p[4];
            let color5  = p[5];  // center (top-left of 2x2 block)
            let color6  = p[6];  // right
            let color_s2 = p[7];
            let color1  = p[8];
            let color2  = p[9];  // below
            let color3  = p[10]; // below-right (diagonal)
            let color_s1 = p[11];
            let color_a1 = p[13];
            let color_a2 = p[14];

            let (e0, e1, e2, e3);

            if color2 == color6 && color5 != color3 {
                // Anti-diagonal wins
                e1 = color2;
                e2 = color2;
                if (color1 == color2 && color6 == color_s2)
                    || (color2 == color_a1 && color6 == color_b2)
                {
                    e0 = interp3_1(color2, color5);
                    e3 = interp3_1(color2, color3);
                } else {
                    e0 = interp(color5, color6);
                    e3 = interp(color2, color3);
                }
            } else if color5 == color3 && color2 != color6 {
                // Main diagonal wins
                e3 = color5;
                e0 = color5;
                if (color_b1 == color5 && color3 == color_a2)
                    || (color4 == color5 && color3 == color_s1)
                {
                    e1 = interp3_1(color5, color6);
                    e2 = interp3_1(color5, color2);
                } else {
                    e1 = interp(color5, color6);
                    e2 = interp(color2, color3);
                }
            } else if color5 == color3 && color2 == color6 && color5 != color6 {
                // Both diagonals match — vote among extended neighbors
                let r = get_result(color6, color5, color1, color_a1)
                    + get_result(color6, color5, color4, color_b1)
                    + get_result(color6, color5, color_a2, color_s1)
                    + get_result(color6, color5, color_b2, color_s2);

                if r > 0 {
                    e1 = color2;
                    e2 = color2;
                    e0 = interp(color5, color6);
                    e3 = interp(color5, color6);
                } else if r < 0 {
                    e0 = color5;
                    e3 = color5;
                    e1 = interp(color5, color6);
                    e2 = interp(color5, color6);
                } else {
                    e0 = color5;
                    e3 = color5;
                    e1 = color2;
                    e2 = color2;
                }
            } else {
                if color2 == color5 || color3 == color6 {
                    // Horizontal/vertical edge — nearest neighbor
                    e0 = color5;
                    e1 = color6;
                    e2 = color2;
                    e3 = color3;
                } else {
                    // No clear feature — 75/25 blend toward each corner
                    e0 = interp3_1(color5, color6);
                    e1 = interp3_1(color6, color5);
                    e2 = interp3_1(color2, color3);
                    e3 = interp3_1(color3, color2);
                }
            }

            let dx = x * 2;
            let dy = y * 2;
            dst[dy * dst_w + dx] = e0;
            dst[dy * dst_w + dx + 1] = e1;
            dst[(dy + 1) * dst_w + dx] = e2;
            dst[(dy + 1) * dst_w + dx + 1] = e3;
        }
    }
    dst
}
