// Quadratic uniform B-spline curve-curve intersection.
//
// Used by the crossing_pack stage: at each 4-way crossing, the optimizer
// has placed both N-S and E-W spans through the same geometric point —
// `a_p1 == b_p1` after energy minimization. We need the actual curve-
// curve intersection parameters (t_a, t_b) so the rasterizer can split
// prev/next color regions and anchor wedge-AA junction lines exactly at
// the geometric meeting point.
//
// Pipeline:
//   1. Seed (t, s) by finding the closest-t on each curve to a_p1 (the
//      optimizer's intersection estimate). Accounts for ghost-extended
//      neighbors and chain asymmetry that a fixed t=0.5 seed misses.
//   2. 2D Newton iterate on F(t, s) = B_a(t) − B_b(s) = 0. Quadratic
//      convergence — 2–3 iterations to f32 precision from this seed.

module;

// Try parameter t against the running best for closest-point search.
[ForceInline] void try_closest_t(float t, float2 a, float2 b, float2 e,
                                  inout float best_t, inout float best_d2) {
    if (t > 0.0 && t < 1.0) {
        float2 d = (a * t + b) * t + e;
        float d2 = dot(d, d);
        if (d2 < best_d2) { best_d2 = d2; best_t = t; }
    }
}

// Closest t in [0, 1] on a quadratic B(t) = a·t² + b·t + c to point pt.
// |B(t) − pt|² has derivative D'(t) = 2·B'(t)·(B(t) − pt), a cubic in t.
// Solve D'(t)/2 = c3·t³ + c2·t² + c1·t + c0 = 0 via Cardano on the
// depressed cubic; degenerate cases (linear D' from a near-line span)
// fall through to the lower-degree solve. Compare candidates plus the
// endpoints by D(t).
[ForceInline] float closest_t_on_quadratic(float2 a, float2 b, float2 c, float2 pt) {
    float2 e = c - pt;
    float c3 = 2.0 * dot(a, a);
    float c2 = 3.0 * dot(a, b);
    float c1 = 2.0 * dot(a, e) + dot(b, b);
    float c0 = dot(b, e);

    float best_t = 0.0;
    float best_d2 = dot(e, e);
    float2 d1 = a + b + e;
    float d1_2 = dot(d1, d1);
    if (d1_2 < best_d2) { best_d2 = d1_2; best_t = 1.0; }

    if (abs(c3) < 1e-12) {
        if (abs(c2) > 1e-12) {
            float disc = c1 * c1 - 4.0 * c2 * c0;
            if (disc >= 0.0) {
                float sq = sqrt(disc);
                float q = -0.5 * (c1 + sign(c1) * sq);
                try_closest_t(q / c2, a, b, e, best_t, best_d2);
                if (abs(q) > 1e-20) try_closest_t(c0 / q, a, b, e, best_t, best_d2);
            }
        } else if (abs(c1) > 1e-12) {
            try_closest_t(-c0 / c1, a, b, e, best_t, best_d2);
        }
        return best_t;
    }

    // Cardano on the depressed cubic.
    float aa = c2 / c3;
    float bb = c1 / c3;
    float cc = c0 / c3;
    float p = bb - aa * aa / 3.0;
    float q = 2.0 * aa * aa * aa / 27.0 - aa * bb / 3.0 + cc;
    float shift = -aa / 3.0;
    float disc = q * q / 4.0 + p * p * p / 27.0;
    if (disc >= 0.0) {
        float sq = sqrt(disc);
        float u1 = sign(-q * 0.5 + sq) * pow(abs(-q * 0.5 + sq), 1.0 / 3.0);
        float u2 = sign(-q * 0.5 - sq) * pow(abs(-q * 0.5 - sq), 1.0 / 3.0);
        try_closest_t(u1 + u2 + shift, a, b, e, best_t, best_d2);
    } else {
        float r = sqrt(max(-p / 3.0, 0.0));
        float cos_arg = (r > 1e-20) ? (-q * 0.5) / (r * r * r) : 0.0;
        float theta = acos(clamp(cos_arg, -1.0, 1.0));
        float TAU = 6.283185307;
        for (int k = 0; k < 3; ++k) {
            try_closest_t(2.0 * r * cos((theta + TAU * float(k)) / 3.0) + shift,
                          a, b, e, best_t, best_d2);
        }
    }
    return best_t;
}

// Intersect two quadratic uniform B-spline spans via 2D Newton from a
// closest-point seed. Returns float2(t_a, t_b).
public float2 bspline_intersect(float2 a_p0, float2 a_p1, float2 a_p2,
                                 float2 b_p0, float2 b_p1, float2 b_p2) {
    // B(t) = aa·t² + ba·t + ca for the standard uniform-B-spline blend.
    float2 aa = 0.5 * a_p0 - a_p1 + 0.5 * a_p2;
    float2 ba = -a_p0 + a_p1;
    float2 ca = 0.5 * a_p0 + 0.5 * a_p1;
    float2 ab = 0.5 * b_p0 - b_p1 + 0.5 * b_p2;
    float2 bb = -b_p0 + b_p1;
    float2 cb = 0.5 * b_p0 + 0.5 * b_p1;

    // Seed: a_p1 == b_p1 after the optimizer. Closest-t on each curve to
    // that point handles ghost-extended neighbors and chain asymmetry.
    float t = closest_t_on_quadratic(aa, ba, ca, a_p1);
    float s = closest_t_on_quadratic(ab, bb, cb, a_p1);

    // 2D Newton on F(t, s) = B_a(t) − B_b(s) = 0.
    //   J = [B_a'(t).x  −B_b'(s).x ]  det = −B_a'(t).x·B_b'(s).y
    //       [B_a'(t).y  −B_b'(s).y ]      + B_a'(t).y·B_b'(s).x
    // (Δt, Δs) = J⁻¹·F; (t, s) ← (t, s) − (Δt, Δs).
    [unroll] for (int iter = 0; iter < 6; ++iter) {
        float2 ba_t = (aa * t + ba) * t + ca;
        float2 bb_s = (ab * s + bb) * s + cb;
        float2 f = ba_t - bb_s;
        float2 da = 2.0 * aa * t + ba;
        float2 db = 2.0 * ab * s + bb;
        float det = -da.x * db.y + da.y * db.x;
        if (abs(det) < 1e-12) break;
        float inv = 1.0 / det;
        float step_t = inv * (-db.y * f.x + db.x * f.y);
        float step_s = inv * (-da.y * f.x + da.x * f.y);
        t -= step_t;
        s -= step_s;
        if (abs(step_t) < 1e-7 && abs(step_s) < 1e-7) break;
    }
    t = clamp(t, 0.0, 1.0);
    s = clamp(s, 0.0, 1.0);

    return float2(t, s);
}
