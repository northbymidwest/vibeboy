//! SVG serialization: convert color paths to an SVG document using the `svg` crate.

use super::contour::{ColorPath, PathSegment};

/// Render a collection of color paths as a complete SVG document string.
pub fn render_svg(paths: &[ColorPath], width: usize, height: usize, bg_color: u32) -> String {
    let mut doc = svg::Document::new()
        .set("viewBox", (0, 0, width, height))
        .set("width", width * 4)
        .set("height", height * 4)
        .set("shape-rendering", "geometricPrecision");

    // Background rect
    let bg_r = (bg_color >> 16) & 0xFF;
    let bg_g = (bg_color >> 8) & 0xFF;
    let bg_b = bg_color & 0xFF;
    doc = doc.add(
        svg::node::element::Rectangle::new()
            .set("width", width)
            .set("height", height)
            .set("fill", format!("#{bg_r:02X}{bg_g:02X}{bg_b:02X}")),
    );

    // Group paths by color to reduce SVG size
    let mut by_color: std::collections::BTreeMap<u32, Vec<String>> =
        std::collections::BTreeMap::new();

    for path in paths {
        if path.segments.is_empty() {
            continue;
        }
        let d = path_to_svg_d(&path.segments);
        if !d.is_empty() {
            by_color.entry(path.color).or_default().push(d);
        }
    }

    for (color, ds) in &by_color {
        // Skip background color — covered by the background rect
        if *color == bg_color {
            continue;
        }
        let r = (color >> 16) & 0xFF;
        let g = (color >> 8) & 0xFF;
        let b = color & 0xFF;
        let hex = format!("#{r:02X}{g:02X}{b:02X}");

        let mut combined = String::new();
        for d in ds {
            if !combined.is_empty() {
                combined.push(' ');
            }
            combined.push_str(d);
        }

        doc = doc.add(
            svg::node::element::Path::new()
                .set("fill", hex.clone())
                .set("fill-rule", "nonzero")
                .set("stroke", hex)
                .set("stroke-width", 0.025)
                .set("stroke-linejoin", "round")
                .set("d", combined),
        );
    }

    doc.to_string()
}

/// Convert path segments to an SVG `d` attribute string.
/// Handles multiple sub-polygons: when a segment's start doesn't match
/// the previous end, a new subpath (M...Z) is started.
fn path_to_svg_d(segments: &[PathSegment]) -> String {
    if segments.is_empty() {
        return String::new();
    }

    let mut d = String::with_capacity(segments.len() * 32);
    let mut last_x = f64::NAN;
    let mut last_y = f64::NAN;
    let mut in_subpath = false;

    for seg in segments {
        let (start, end) = match seg {
            PathSegment::Line(a, b) => (*a, *b),
            PathSegment::QuadBezier(a, _, c) => (*a, *c),
        };

        // Start new subpath if start doesn't match previous end
        if !in_subpath
            || (start.x - last_x).abs() > 1e-9
            || (start.y - last_y).abs() > 1e-9
        {
            if in_subpath {
                d.push('Z');
            }
            d.push_str(&format!("M{} {}", fmt_f64(start.x), fmt_f64(start.y)));
            in_subpath = true;
        }

        match seg {
            PathSegment::Line(_, b) => {
                d.push_str(&format!("L{} {}", fmt_f64(b.x), fmt_f64(b.y)));
            }
            PathSegment::QuadBezier(_, ctrl, end) => {
                d.push_str(&format!(
                    "Q{} {} {} {}",
                    fmt_f64(ctrl.x),
                    fmt_f64(ctrl.y),
                    fmt_f64(end.x),
                    fmt_f64(end.y)
                ));
            }
        }

        last_x = end.x;
        last_y = end.y;
    }

    if in_subpath {
        d.push('Z');
    }
    d
}

/// Format a float with minimal precision (up to 4 decimal places, trailing zeros stripped).
fn fmt_f64(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        let s = format!("{:.4}", v);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}
