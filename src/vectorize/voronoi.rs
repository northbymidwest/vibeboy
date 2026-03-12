//! Voronoi reshaping for cell-corner positions.
//!
//! At each grid intersection where 4 pixels meet, diagonal edges in the
//! similarity graph shift the corner position to create non-square cell
//! boundaries that follow the pixel-art features.

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

/// Precomputed corner positions for the reshaped Voronoi grid.
/// Grid has (width+1) x (height+1) corners.
pub struct CornersGrid {
    pub cw: usize, // width+1
    pub _ch: usize, // height+1
    /// For each corner, store the base position and diagonal type.
    /// Diagonal type: 0 = no diagonal, 1 = main (\), 2 = anti (/)
    pub corners: Vec<(Point, u8)>,
}

impl CornersGrid {
    /// Get the reshaped corner position for a specific pixel at a specific corner.
    /// `px, py` is the pixel; `corner` is which corner (0=TL, 1=TR, 2=BR, 3=BL).
    pub fn corner_for_pixel(&self, px: usize, py: usize, corner: u8) -> Point {
        let (cx, cy) = match corner {
            0 => (px, py),       // TL
            1 => (px + 1, py),   // TR
            2 => (px + 1, py + 1), // BR
            3 => (px, py + 1),   // BL
            _ => unreachable!(),
        };

        let (base, diag) = self.corners[cy * self.cw + cx];

        if diag == 0 {
            return base;
        }

        let shift = 0.25;

        if diag == 1 {
            // Main diagonal (\): TL and BR pixels are connected
            match corner {
                0 => Point::new(base.x + shift, base.y + shift), // TL pixel: shift toward BR
                1 => Point::new(base.x + shift, base.y - shift), // TR pixel: shift away
                2 => Point::new(base.x - shift, base.y - shift), // BR pixel: shift toward TL
                3 => Point::new(base.x - shift, base.y + shift), // BL pixel: shift away
                _ => base,
            }
        } else {
            // Anti diagonal (/): TR and BL pixels are connected
            match corner {
                0 => Point::new(base.x - shift, base.y + shift), // TL pixel: shift away
                1 => Point::new(base.x - shift, base.y - shift), // TR pixel: shift toward BL
                2 => Point::new(base.x + shift, base.y - shift), // BR pixel: shift away
                3 => Point::new(base.x + shift, base.y + shift), // BL pixel: shift toward TR
                _ => base,
            }
        }
    }
}

/// Build the corner grid from the similarity graph.
pub fn build_corners(graph: &SimilarityGraph) -> CornersGrid {
    let w = graph.width;
    let h = graph.height;
    let cw = w + 1;
    let ch = h + 1;
    let mut corners = vec![(Point::new(0.0, 0.0), 0u8); cw * ch];

    for cy in 0..ch {
        for cx in 0..cw {
            let base = Point::new(cx as f64, cy as f64);

            // Interior corners only
            let diag = if cx > 0 && cy > 0 && cx < w && cy < h {
                let bx = cx - 1; // top-left pixel of the 2x2 block
                let by = cy - 1;
                let has_main = graph.edge(bx, by).down_right;
                let has_anti = graph.edge(bx + 1, by).down_left;

                if has_main && !has_anti {
                    1 // main diagonal
                } else if has_anti && !has_main {
                    2 // anti diagonal
                } else {
                    0 // no diagonal or both (resolved to none)
                }
            } else {
                0
            };

            corners[cy * cw + cx] = (base, diag);
        }
    }

    CornersGrid { cw, _ch: ch, corners }
}
