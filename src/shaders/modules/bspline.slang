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

/// Refine closest point via 2 extra samples + 3 Newton iterations.
/// Returns float2(best_t, best_distance_squared).
public float2 bspline_refine(float2 p0, float2 p1, float2 p2, float2 pt,
                              float seed_t, float seed_d2) {
    float best_d2 = seed_d2;
    float best_t = seed_t;

    float2 bq1 = bspline_eval(p0, p1, p2, 0.25);
    float dq1 = dot(pt - bq1, pt - bq1);
    if (dq1 < best_d2) { best_d2 = dq1; best_t = 0.25; }

    float2 bq3 = bspline_eval(p0, p1, p2, 0.75);
    float dq3 = dot(pt - bq3, pt - bq3);
    if (dq3 < best_d2) { best_d2 = dq3; best_t = 0.75; }

    float2 b2 = p0 - 2.0 * p1 + p2;
    for (int i = 0; i < 3; i++) {
        float2 bt = bspline_eval(p0, p1, p2, best_t);
        float2 bt1 = bspline_deriv(p0, p1, p2, best_t);
        float2 diff = bt - pt;
        float f = dot(bt1, diff);
        float g = dot(b2, diff) + dot(bt1, bt1);
        if (abs(g) > 1e-10)
            best_t = clamp(best_t - f / g, 0.0, 1.0);
    }

    float2 bp = bspline_eval(p0, p1, p2, best_t);
    best_d2 = dot(pt - bp, pt - bp);
    return float2(best_t, best_d2);
}
