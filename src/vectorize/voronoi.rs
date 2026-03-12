//! Voronoi corner reshaping for the Kopf-Lischinski algorithm (Section 3.2).
//!
//! At each interior grid corner where 4 pixel cells meet, a diagonal connection
//! in the resolved similarity graph shifts the corner by ±¼ pixel, forming the
//! generalized Voronoi diagram. This ensures diagonally-connected pixels share
//! an edge in the reshaped cell graph.
//!
//! The paper: "The reshaped cell graph can be computed as a generalized Voronoi
//! diagram, where each Voronoi cell contains the points that are closest to the
//! union of a node and its half-edges."

use super::graph::SimilarityGraph;

/// A point in continuous 2D space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// Get the reshaped corner position at grid corner (cx, cy) for a directed
/// boundary edge traveling in the given direction.
///
/// At each interior grid corner, at most one diagonal can exist (after crossing
/// resolution). The diagonal shifts the corner into two nodes offset by ¼ pixel,
/// and the correct node depends on the travel direction of the contour edge.
///
/// Convention: the traced region is to the LEFT of the travel direction.
///
/// For \ diagonal (TL-BR connected) at corner (cx, cy):
///   Nodes A=(cx-¼, cy+¼) and B=(cx+¼, cy-¼)
///   Going right or up → B; going down or left → A
///
/// For / diagonal (TR-BL connected) at corner (cx, cy):
///   Nodes C=(cx-¼, cy-¼) and D=(cx+¼, cy+¼)
///   Going left or up → C; going right or down → D
///
/// dir: 0=right, 1=down, 2=left, 3=up
pub fn reshaped_corner(cx: i32, cy: i32, dir: u8, graph: &SimilarityGraph) -> Point {
    let w = graph.width as i32;
    let h = graph.height as i32;

    // Only interior corners (not on the image boundary) can be reshaped
    if cx <= 0 || cy <= 0 || cx >= w || cy >= h {
        return Point::new(cx as f64, cy as f64);
    }

    // \ diagonal: pixel (cx-1, cy-1) connected to (cx, cy)
    let has_backslash = graph.edge((cx - 1) as usize, (cy - 1) as usize).down_right;

    // / diagonal: pixel (cx, cy-1) connected to (cx-1, cy)
    let has_slash = graph.edge(cx as usize, (cy - 1) as usize).down_left;

    if has_backslash {
        match dir {
            0 | 3 => Point::new(cx as f64 + 0.25, cy as f64 - 0.25),
            _ => Point::new(cx as f64 - 0.25, cy as f64 + 0.25),
        }
    } else if has_slash {
        match dir {
            2 | 3 => Point::new(cx as f64 - 0.25, cy as f64 - 0.25),
            _ => Point::new(cx as f64 + 0.25, cy as f64 + 0.25),
        }
    } else {
        Point::new(cx as f64, cy as f64)
    }
}
