// Quadratic uniform B-spline curve-curve intersection.
//
// Used by the crossing_pack stage: at each 4-way crossing, the optimizer
// has placed both N-S and E-W spans through nearly the same point. We need
// the actual curve-curve intersection parameters (t_a, t_b) so the
// rasterizer can split prev/next color regions and anchor wedge-AA
// junction lines exactly at the geometric meeting point.
//
// Pipeline structure:
//   1. Build the Sylvester resultant of the two quadratics (a quartic in t).
//   2. Ferrari method for the quartic via a real root of its resolvent cubic.
//   3. From t (root closest to 0.5), evaluate B_a(t) to get the intersection
//      point, then back-solve the better-conditioned axis of B_b for s.

module;

// Update best_t/best_dist if `t` is in [0,1] and closer to 0.5 than current.
[ForceInline] void try_root_in_unit(float t, inout float best_t, inout float best_dist) {
    if (t >= 0.0 && t <= 1.0) {
        float d = abs(t - 0.5);
        if (d < best_dist) { best_dist = d; best_t = t; }
    }
}

// One real root of c3·y³ + c2·y² + c1·y + c0 = 0. Used by the quartic
// resolvent — any real cubic has at least one real root, so this always
// returns a usable value.
[ForceInline] float solve_cubic_real_root(float c3, float c2, float c1, float c0) {
    float a = c2 / c3;
    float b = c1 / c3;
    float cc = c0 / c3;
    float p = b - a * a / 3.0;
    float q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + cc;
    float shift = -a / 3.0;
    float disc = q * q / 4.0 + p * p * p / 27.0;

    if (disc >= 0.0) {
        float sq = sqrt(disc);
        // cbrt via sign-preserving pow (HLSL/Slang have no cbrt).
        float u1 = sign(-q * 0.5 + sq) * pow(abs(-q * 0.5 + sq), 1.0 / 3.0);
        float u2 = sign(-q * 0.5 - sq) * pow(abs(-q * 0.5 - sq), 1.0 / 3.0);
        return u1 + u2 + shift;
    }
    float r = sqrt(-p / 3.0);
    float theta = acos(clamp((-q * 0.5) / (r * r * r), -1.0, 1.0));
    return 2.0 * r * cos(theta / 3.0) + shift;
}

// Track in-[0,1] roots of c3·t³ + c2·t² + c1·t + c0 = 0 via best_t/best_dist.
// Cardano's method on the depressed cubic; degenerates cleanly to quadratic /
// linear when leading coefficients vanish (relative to the others).
[ForceInline] void solve_cubic_in_unit(float c3, float c2, float c1, float c0,
                                        inout float best_t, inout float best_dist) {
    float scale = max(max(abs(c0), abs(c1)), max(abs(c2), abs(c3)));
    float eps = 1e-6 * max(scale, 1.0);

    if (abs(c3) < eps) {
        // Quadratic
        if (abs(c2) < eps) {
            if (abs(c1) > eps) {
                try_root_in_unit(-c0 / c1, best_t, best_dist);
            }
            return;
        }
        float disc = c1 * c1 - 4.0 * c2 * c0;
        if (disc < 0.0) return;
        float sq = sqrt(disc);
        try_root_in_unit((-c1 + sq) / (2.0 * c2), best_t, best_dist);
        try_root_in_unit((-c1 - sq) / (2.0 * c2), best_t, best_dist);
        return;
    }

    // Cardano on the depressed cubic: t = u - c2/(3·c3), u³ + p·u + q = 0.
    float a = c2 / c3;
    float b = c1 / c3;
    float cc = c0 / c3;
    float p = b - a * a / 3.0;
    float q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + cc;
    float shift = -a / 3.0;
    float disc = q * q / 4.0 + p * p * p / 27.0;

    if (disc > 0.0) {
        float sq = sqrt(disc);
        float u1 = sign(-q * 0.5 + sq) * pow(abs(-q * 0.5 + sq), 1.0 / 3.0);
        float u2 = sign(-q * 0.5 - sq) * pow(abs(-q * 0.5 - sq), 1.0 / 3.0);
        try_root_in_unit(u1 + u2 + shift, best_t, best_dist);
        return;
    }
    // Three real roots (trigonometric form)
    float r = sqrt(max(-p / 3.0, 0.0));
    float cos_arg = (r > 1e-20) ? (-q * 0.5) / (r * r * r) : 0.0;
    float theta = acos(clamp(cos_arg, -1.0, 1.0));
    float TAU = 6.283185307;
    for (int k = 0; k < 3; k++) {
        float t = 2.0 * r * cos((theta + TAU * float(k)) / 3.0) + shift;
        try_root_in_unit(t, best_t, best_dist);
    }
}

// Track in-[0,1] roots of c4·t⁴ + c3·t³ + c2·t² + c1·t + c0 = 0 via
// best_t/best_dist. Ferrari's method: depress to u⁴ + p·u² + q·u + r = 0,
// pick a real root y of the resolvent cubic, factor into two quadratics in u.
[ForceInline] void solve_quartic_in_unit(float c4, float c3, float c2, float c1, float c0,
                                          inout float best_t, inout float best_dist) {
    // Relative degeneracy threshold: two B-spline spans whose 2nd-derivatives
    // are parallel (e.g. both diagonal) make c4 ≈ 0 even when other coefficients
    // are not, so an absolute cutoff would miss the cubic-degenerate case.
    float scale = max(max(max(abs(c0), abs(c1)), max(abs(c2), abs(c3))), abs(c4));
    float eps = 1e-6 * max(scale, 1.0);
    if (abs(c4) < eps) {
        solve_cubic_in_unit(c3, c2, c1, c0, best_t, best_dist);
        return;
    }

    // Normalize: t⁴ + a·t³ + b·t² + c·t + d = 0
    float a = c3 / c4;
    float b = c2 / c4;
    float cc = c1 / c4;
    float dd = c0 / c4;

    // Depress via t = u - a/4: u⁴ + p·u² + q·u + r = 0
    float a2 = a * a;
    float p = b - 3.0 * a2 / 8.0;
    float q = cc - a * b / 2.0 + a2 * a / 8.0;
    float r = dd - a * cc / 4.0 + a2 * b / 16.0 - 3.0 * a2 * a2 / 256.0;
    float shift = -a / 4.0;

    if (abs(q) < 1e-12) {
        // Biquadratic: u⁴ + p·u² + r = 0 → solve for u²
        float disc = p * p - 4.0 * r;
        if (disc < 0.0) return;
        float sq = sqrt(disc);
        float u2_a = (-p + sq) * 0.5;
        float u2_b = (-p - sq) * 0.5;
        if (u2_a >= 0.0) {
            float uu = sqrt(u2_a);
            try_root_in_unit( uu + shift, best_t, best_dist);
            try_root_in_unit(-uu + shift, best_t, best_dist);
        }
        if (u2_b >= 0.0) {
            float uu = sqrt(u2_b);
            try_root_in_unit( uu + shift, best_t, best_dist);
            try_root_in_unit(-uu + shift, best_t, best_dist);
        }
        return;
    }

    // Resolvent cubic: 8·y³ + 8·p·y² + (2p² - 8r)·y - q² = 0; pick any real y.
    float cy = solve_cubic_real_root(8.0, 8.0 * p, 2.0 * p * p - 8.0 * r, -q * q);
    float two_y = 2.0 * cy;
    if (two_y < 0.0) return;
    float m = sqrt(two_y);
    if (abs(m) < 1e-12) return;
    float q_over_2m = q / (2.0 * m);
    float half_p_plus_y = p * 0.5 + cy;

    // Two quadratics: u² ± m·u + (p/2 + y ∓ q/(2m)) = 0
    float bb1 = m;
    float cc1 = half_p_plus_y - q_over_2m;
    float disc1 = bb1 * bb1 - 4.0 * cc1;
    if (disc1 >= 0.0) {
        float sq1 = sqrt(disc1);
        try_root_in_unit((-bb1 + sq1) * 0.5 + shift, best_t, best_dist);
        try_root_in_unit((-bb1 - sq1) * 0.5 + shift, best_t, best_dist);
    }
    float bb2 = -m;
    float cc2 = half_p_plus_y + q_over_2m;
    float disc2 = bb2 * bb2 - 4.0 * cc2;
    if (disc2 >= 0.0) {
        float sq2 = sqrt(disc2);
        try_root_in_unit((-bb2 + sq2) * 0.5 + shift, best_t, best_dist);
        try_root_in_unit((-bb2 - sq2) * 0.5 + shift, best_t, best_dist);
    }
}

// Intersect two quadratic uniform B-spline spans.
// Returns float2(t_a, t_b) — the parameters at which curve A and curve B
// cross. The pipeline guarantees they cross within [0,1]² at a 4-way
// junction, so we pick the in-unit root closest to (0.5, 0.5).
public float2 bspline_intersect(float2 a_p0, float2 a_p1, float2 a_p2,
                                 float2 b_p0, float2 b_p1, float2 b_p2) {
    // B(t) = aa·t² + ba·t + ca for the standard uniform-B-spline blend.
    float2 aa = 0.5 * a_p0 - a_p1 + 0.5 * a_p2;
    float2 ba = -a_p0 + a_p1;
    float2 ca = 0.5 * a_p0 + 0.5 * a_p1;
    float2 ab = 0.5 * b_p0 - b_p1 + 0.5 * b_p2;
    float2 bb = -b_p0 + b_p1;
    float2 cb = 0.5 * b_p0 + 0.5 * b_p1;

    // Sylvester resultant of (B_a − B_b) on (s) eliminates s, leaving
    // R(t) = U(t)² − M·V(t), a quartic in t.
    float m = ab.x * bb.y - bb.x * ab.y;
    float u2 = ab.y * aa.x - ab.x * aa.y;
    float u1 = ab.y * ba.x - ab.x * ba.y;
    float u0 = ab.y * ca.x - ab.x * ca.y + ab.x * cb.y - ab.y * cb.x;
    float v2 = bb.y * aa.x - bb.x * aa.y;
    float v1 = bb.y * ba.x - bb.x * ba.y;
    float v0 = bb.y * ca.x - bb.x * ca.y + bb.x * cb.y - bb.y * cb.x;

    float r4 = u2 * u2;
    float r3 = 2.0 * u1 * u2;
    float r2 = u1 * u1 + 2.0 * u0 * u2 - m * v2;
    float r1 = 2.0 * u0 * u1 - m * v1;
    float r0 = u0 * u0 - m * v0;

    float best_t = 0.5;
    float best_dist = 1e30;
    solve_quartic_in_unit(r4, r3, r2, r1, r0, best_t, best_dist);
    float t = best_t;

    // Intersection point on curve A (= aa·t² + ba·t + ca).
    float2 p = aa * t * t + ba * t + ca;

    // Back-solve B_b(s) = p on the better-conditioned axis to get t_b.
    bool prefer_x = abs(ab.x) >= abs(ab.y);
    float a_use = prefer_x ? ab.x : ab.y;
    float b_use = prefer_x ? bb.x : bb.y;
    float c_use = prefer_x ? cb.x : cb.y;
    float t_use = prefer_x ? p.x : p.y;
    float kk = c_use - t_use;

    float best_s = 0.5;
    float best_s_dist = 1e30;
    if (abs(a_use) < 1e-12) {
        if (abs(b_use) > 1e-12) {
            try_root_in_unit(-kk / b_use, best_s, best_s_dist);
        }
    } else {
        float disc = b_use * b_use - 4.0 * a_use * kk;
        if (disc >= 0.0) {
            float sq = sqrt(disc);
            try_root_in_unit((-b_use + sq) / (2.0 * a_use), best_s, best_s_dist);
            try_root_in_unit((-b_use - sq) / (2.0 * a_use), best_s, best_s_dist);
        }
    }

    return float2(t, best_s);
}
