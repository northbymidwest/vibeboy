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
