// Quadratic uniform B-spline evaluation and closest-point queries.
//
// Used by the cell rasterizer for curve-distance computation.

module;

/// Evaluate quadratic B-spline at parameter t in [0, 1].
[ForceInline] public float2 bspline_eval(float2 p0, float2 p1, float2 p2, float t) {
    float u = 1.0 - t;
    return 0.5 * u * u * p0 + (u * t + 0.5) * p1 + 0.5 * t * t * p2;
}

/// First derivative of quadratic B-spline at parameter t.
[ForceInline] public float2 bspline_deriv(float2 p0, float2 p1, float2 p2, float t) {
    return (t - 1.0) * p0 + (1.0 - 2.0 * t) * p1 + t * p2;
}

/// Quick 3-point screen for nearest-parameter estimation.
/// Returns float2(best_t, best_distance_squared).
public float2 bspline_quick_screen(float2 p0, float2 p1, float2 p2, float2 pt) {
    float2 b0 = 0.5 * (p0 + p1);
    float2 bm = 0.125 * p0 + 0.75 * p1 + 0.125 * p2;
    float2 b1 = 0.5 * (p1 + p2);

    float dd0 = dot(pt - b0, pt - b0);
    float ddm = dot(pt - bm, pt - bm);
    float dd1 = dot(pt - b1, pt - b1);

    float best = dd0; float t = 0.0;
    if (ddm < best) { best = ddm; t = 0.5; }
    if (dd1 < best) { best = dd1; t = 1.0; }
    return float2(t, best);
}

/// Exact closest point via analytical cubic solve.
/// D(t) = |B(t)-pt|² is degree 4, D'(t) is cubic → closed-form roots.
/// Evaluates D at all real roots in [0,1] + endpoints, returns global min.
/// No iterative Newton, no degenerate endpoint traps.
public float2 bspline_refine(float2 p0, float2 p1, float2 p2, float2 pt,
                              float seed_t, float seed_d2) {
    // Polynomial form: B(t) = a*t² + b*t + c
    float2 a = 0.5 * (p0 - 2.0 * p1 + p2);
    float2 b = p1 - p0;
    float2 e = 0.5 * (p0 + p1) - pt;  // c - pt

    // D'(t)/2 = (a·t²+b·t+e)·(2a·t+b) per component, summed.
    // Cubic coefficients: c3·t³ + c2·t² + c1·t + c0
    float c3 = 2.0 * dot(a, a);
    float c2 = 3.0 * dot(a, b);
    float c1 = 2.0 * dot(a, e) + dot(b, b);
    float c0 = dot(b, e);

    // Evaluate D(t) via Horner
    #define EVAL_D2(t) dot((a*(t)+b)*(t)+e, (a*(t)+b)*(t)+e)

    // Start with endpoints
    float d0 = EVAL_D2(0.0);
    float d1 = EVAL_D2(1.0);
    float best_d2 = d0; float best_t = 0.0;
    if (d1 < best_d2) { best_d2 = d1; best_t = 1.0; }

    // 1e-12 is essentially "is c3 zero?" — the full-cubic branch divides by
    // c3 (and c3², c3³), so anything strictly positive but tiny would blow up.
    // Genuine straight-line spans (2-CP chains) are dispatched to a closed-
    // form line solver upstream and never reach here, so this only fires for
    // numerically-degenerate spans where the linear approximation is correct
    // anyway.
    if (abs(c3) < 1e-12) {
        // Degenerate: quadratic or linear
        if (abs(c2) > 1e-12) {
            // Numerically stable quadratic root formula (Numerical Recipes 5.6):
            // q = -0.5*(c1 + sgn(c1)*sqrt(disc)); roots: q/c2, c0/q.
            // Avoids catastrophic cancellation when c2 is small and the
            // (-c1 ± sqrt(disc)) form subtracts two near-equal numbers.
            float disc = c1*c1 - 4.0*c2*c0;
            if (disc >= 0.0) {
                float sq = sqrt(disc);
                float qq = -0.5 * (c1 + sign(c1) * sq);
                float t1 = qq / c2;
                float t2 = c0 / qq;
                if (t1 > 0.0 && t1 < 1.0) { float dd = EVAL_D2(t1); if (dd < best_d2) { best_d2 = dd; best_t = t1; } }
                if (t2 > 0.0 && t2 < 1.0) { float dd = EVAL_D2(t2); if (dd < best_d2) { best_d2 = dd; best_t = t2; } }
            }
        } else if (abs(c1) > 1e-12) {
            float t1 = -c0 / c1;
            if (t1 > 0.0 && t1 < 1.0) { float dd = EVAL_D2(t1); if (dd < best_d2) { best_d2 = dd; best_t = t1; } }
        }
    } else {
        // Full cubic: depressed form u³ + p·u + q = 0 via t = u - c2/(3·c3)
        float inv3a = 1.0 / (3.0 * c3);
        float shift = -c2 * inv3a;
        float p = (3.0*c3*c1 - c2*c2) / (3.0*c3*c3);
        float q = (2.0*c2*c2*c2 - 9.0*c3*c2*c1 + 27.0*c3*c3*c0) / (27.0*c3*c3*c3);
        float disc = q*q/4.0 + p*p*p/27.0;

        if (disc > 1e-12) {
            // One real root
            float sq = sqrt(disc);
            float hq = -q * 0.5;
            // cbrt via sign-preserving pow
            float u1 = sign(hq+sq) * pow(abs(hq+sq), 1.0/3.0);
            float u2 = sign(hq-sq) * pow(abs(hq-sq), 1.0/3.0);
            float t1 = u1 + u2 + shift;
            if (t1 > 0.0 && t1 < 1.0) { float dd = EVAL_D2(t1); if (dd < best_d2) { best_d2 = dd; best_t = t1; } }
        } else {
            // Three real roots (trigonometric)
            float r = sqrt(max(-p*p*p/27.0, 0.0));
            float phi = (r < 1e-15) ? 0.0 : acos(clamp(-q/(2.0*r), -1.0, 1.0));
            float cube_r = 2.0 * sign(r) * pow(abs(r), 1.0/3.0);
            float TAU = 6.283185307;
            for (int k = 0; k < 3; k++) {
                float angle = (phi + TAU * float(k)) / 3.0;
                float t1 = cube_r * cos(angle) + shift;
                if (t1 > 0.0 && t1 < 1.0) { float dd = EVAL_D2(t1); if (dd < best_d2) { best_d2 = dd; best_t = t1; } }
            }
        }
    }

    #undef EVAL_D2
    return float2(best_t, best_d2);
}
