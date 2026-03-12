//! SVG serialization: convert color paths to an SVG document.

use super::contour::{ColorPath, PathSegment};
use std::fmt::Write;

/// Render a collection of color paths as a complete SVG document string.
pub fn render_svg(paths: &[ColorPath], width: usize, height: usize) -> String {
    let mut svg = String::with_capacity(paths.len() * 128);

    writeln!(
        svg,
        r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{sw}" height="{sh}" shape-rendering="geometricPrecision">"#,
        w = width,
        h = height,
        sw = width * 4,
        sh = height * 4,
    )
    .unwrap();

    // Background rect (black)
    writeln!(
        svg,
        "<rect width=\"{}\" height=\"{}\" fill=\"#000000\"/>",
        width, height
    )
    .unwrap();

    // Group paths by color to reduce SVG size
    let mut by_color: std::collections::BTreeMap<u32, Vec<String>> = std::collections::BTreeMap::new();

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
        let r = (color >> 16) & 0xFF;
        let g = (color >> 8) & 0xFF;
        let b = color & 0xFF;

        // Merge all paths of the same color into one <path> element
        let mut combined = String::new();
        for d in ds {
            if !combined.is_empty() {
                combined.push(' ');
            }
            combined.push_str(d);
        }

        writeln!(
            svg,
            "<path fill=\"#{:02X}{:02X}{:02X}\" d=\"{}\"/>",
            r, g, b, combined
        )
        .unwrap();
    }

    writeln!(svg, "</svg>").unwrap();
    svg
}

/// Convert path segments to an SVG `d` attribute string.
fn path_to_svg_d(segments: &[PathSegment]) -> String {
    if segments.is_empty() {
        return String::new();
    }

    let mut d = String::with_capacity(segments.len() * 32);

    // Move to the start of the first segment
    let start = match &segments[0] {
        PathSegment::Line(a, _) => *a,
        PathSegment::QuadBezier(a, _, _) => *a,
    };
    write!(d, "M{} {}", fmt_f64(start.x), fmt_f64(start.y)).unwrap();

    let mut last_x = start.x;
    let mut last_y = start.y;

    for seg in segments {
        match seg {
            PathSegment::Line(_, b) => {
                // Skip zero-length lines
                if (b.x - last_x).abs() < 1e-9 && (b.y - last_y).abs() < 1e-9 {
                    continue;
                }
                write!(d, "L{} {}", fmt_f64(b.x), fmt_f64(b.y)).unwrap();
                last_x = b.x;
                last_y = b.y;
            }
            PathSegment::QuadBezier(_, ctrl, end) => {
                write!(
                    d,
                    "Q{} {} {} {}",
                    fmt_f64(ctrl.x),
                    fmt_f64(ctrl.y),
                    fmt_f64(end.x),
                    fmt_f64(end.y)
                )
                .unwrap();
                last_x = end.x;
                last_y = end.y;
            }
        }
    }

    d.push('Z');
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
