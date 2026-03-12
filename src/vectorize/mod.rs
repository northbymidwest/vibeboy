//! Kopf-Lischinski pixel-art vectorization.
//!
//! Converts a pixel-art frame buffer into smooth SVG vector graphics.
//! Pipeline:
//! 1. Build similarity graph (connect similar adjacent pixels)
//! 2. Resolve ambiguous diagonal crossings with heuristics
//! 3. Compute Voronoi-reshaped corner positions
//! 4. Flood-fill connected regions
//! 5. Trace boundary contours of each region
//! 6. Fit B-splines to the contour points
//! 7. Serialize to SVG

pub mod contour;
pub mod graph;
pub mod svg;
pub mod voronoi;

/// Vectorize a pixel buffer to an SVG string.
///
/// `pixels` is a flat array of ARGB u32 values (0x00RRGGBB).
/// Returns a complete SVG document as a string.
pub fn vectorize_to_svg(pixels: &[u32], width: usize, height: usize) -> String {
    // Step 1: Build similarity graph with diagonal crossing resolution
    let graph = graph::build(pixels, width, height);

    // Step 2: Build reshaped corner grid
    let corners = voronoi::build_corners(&graph);

    // Step 3: Find connected regions and trace their boundary contours
    let contours = contour::build_contours(pixels, &graph, &corners);

    // Step 4: Smooth contours with B-splines and convert to ColorPaths
    let paths: Vec<contour::ColorPath> = contours
        .iter()
        .map(|c| contour::ColorPath {
            color: c.color,
            segments: contour::smooth_contour(c),
        })
        .collect();

    // Step 5: Serialize to SVG
    svg::render_svg(&paths, width, height)
}
