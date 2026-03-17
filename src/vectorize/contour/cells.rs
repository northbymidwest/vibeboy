//! Voronoi cell template table and precomputation (Paper Section 3.2).

use super::NodeId;
use crate::vectorize::graph::SimilarityGraph;

// --- Precomputed cells via template matching (Paper Section 3.2) ---
//
// The paper: "The shape of a Voronoi cell is fully determined by its local
// neighborhood in the similarity graph. The possible distinct shapes are easy
// to enumerate, enabling an extremely efficient algorithm, which walks in
// scanline order over the similarity graph, matches specific edge configurations
// in a 3x3 block at a time, and pastes together the corresponding cell templates."
//
// Each pixel's cell shape depends on the diagonal state at its 4 corners:
// none (0), backslash (1), or slash (2). With 4 corners x 3 states = 81
// templates, each mapping to a fixed cell polygon (4-8 vertices in x4 coords).

/// A pre-computed Voronoi cell template: up to 8 vertex offsets in x4 coordinates
/// relative to the pixel's top-left corner (4*px, 4*py).
#[derive(Clone, Copy)]
struct CellTemplate {
    offsets: [(i8, i8); 8],
    len: u8,
}

/// Build one cell template from the 4 corner diagonal states.
/// Corner states: 0 = no diagonal, 1 = backslash, 2 = slash.
/// Vertices are in CW order: TL(BR), TR(BL), BR(TL), BL(TR).
const fn build_cell_template(tl: u8, tr: u8, br: u8, bl: u8) -> CellTemplate {
    let mut offsets = [(0i8, 0i8); 8];
    let mut len = 0u8;

    // TL corner, pixel visits as CornerRel::BR
    if tl == 1 {        // backslash: a=(-1,1), b=(1,-1)
        offsets[len as usize] = (-1, 1); len += 1;
        offsets[len as usize] = (1, -1); len += 1;
    } else if tl == 2 { // slash: d=(1,1)
        offsets[len as usize] = (1, 1); len += 1;
    } else {             // none: (0,0)
        offsets[len as usize] = (0, 0); len += 1;
    }

    // TR corner, pixel visits as CornerRel::BL
    if tr == 1 {        // backslash: a=(3,1)
        offsets[len as usize] = (3, 1); len += 1;
    } else if tr == 2 { // slash: c=(3,-1), d=(5,1)
        offsets[len as usize] = (3, -1); len += 1;
        offsets[len as usize] = (5, 1); len += 1;
    } else {             // none: (4,0)
        offsets[len as usize] = (4, 0); len += 1;
    }

    // BR corner, pixel visits as CornerRel::TL
    if br == 1 {        // backslash: b=(5,3), a=(3,5)
        offsets[len as usize] = (5, 3); len += 1;
        offsets[len as usize] = (3, 5); len += 1;
    } else if br == 2 { // slash: c=(3,3)
        offsets[len as usize] = (3, 3); len += 1;
    } else {             // none: (4,4)
        offsets[len as usize] = (4, 4); len += 1;
    }

    // BL corner, pixel visits as CornerRel::TR
    if bl == 1 {        // backslash: b=(1,3)
        offsets[len as usize] = (1, 3); len += 1;
    } else if bl == 2 { // slash: d=(1,5), c=(-1,3)
        offsets[len as usize] = (1, 5); len += 1;
        offsets[len as usize] = (-1, 3); len += 1;
    } else {             // none: (0,4)
        offsets[len as usize] = (0, 4); len += 1;
    }

    CellTemplate { offsets, len }
}

/// All 81 Voronoi cell templates, indexed by:
///   tl_state * 27 + tr_state * 9 + br_state * 3 + bl_state
const fn compute_cell_templates() -> [CellTemplate; 81] {
    let mut templates = [CellTemplate { offsets: [(0, 0); 8], len: 0 }; 81];
    let mut i = 0u8;
    while i < 81 {
        let tl = i / 27;
        let tr = (i / 9) % 3;
        let br = (i / 3) % 3;
        let bl = i % 3;
        templates[i as usize] = build_cell_template(tl, tr, br, bl);
        i += 1;
    }
    templates
}

static CELL_TEMPLATES: [CellTemplate; 81] = compute_cell_templates();

/// Get diagonal state at grid corner (cx, cy): 0=none, 1=backslash, 2=slash.
/// Boundary corners (on image edge) always return 0.
#[inline(always)]
pub(super) fn corner_diag_state(graph: &SimilarityGraph, cx: usize, cy: usize) -> usize {
    let w = graph.width;
    let h = graph.height;
    if cx == 0 || cy == 0 || cx >= w || cy >= h {
        return 0;
    }
    if graph.edge(cx - 1, cy - 1).down_right { return 1; }
    if graph.edge(cx, cy - 1).down_left { return 2; }
    0
}

/// Inline fixed-size cell: max 8 vertices per Voronoi cell, avoids heap allocation.
#[derive(Clone, Copy)]
pub(super) struct InlineCell {
    pub(super) nodes: [NodeId; 8],
    pub(super) len: u8,
}

impl InlineCell {
    #[inline]
    pub(super) fn as_slice(&self) -> &[NodeId] {
        &self.nodes[..self.len as usize]
    }
}

/// Precompute NodeId cell polygons for all pixels using template matching.
/// For each pixel, reads the diagonal state at its 4 corners, looks up the
/// corresponding cell template, and stamps out NodeId vertices directly
/// from x4 integer offsets -- no floating point or per-corner branching.
pub(super) fn precompute_cells(w: usize, h: usize, graph: &SimilarityGraph) -> Vec<InlineCell> {
    let zero = NodeId { x4: 0, y4: 0 };
    let mut cells = vec![InlineCell { nodes: [zero; 8], len: 0 }; w * h];
    for y in 0..h {
        for x in 0..w {
            let tl = corner_diag_state(graph, x, y);
            let tr = corner_diag_state(graph, x + 1, y);
            let br = corner_diag_state(graph, x + 1, y + 1);
            let bl = corner_diag_state(graph, x, y + 1);

            let template = &CELL_TEMPLATES[tl * 27 + tr * 9 + br * 3 + bl];
            let cell = &mut cells[y * w + x];
            let base_x4 = (x * 4) as i32;
            let base_y4 = (y * 4) as i32;
            cell.len = template.len;
            for i in 0..template.len as usize {
                let (dx, dy) = template.offsets[i];
                cell.nodes[i] = NodeId {
                    x4: base_x4 + dx as i32,
                    y4: base_y4 + dy as i32,
                };
            }
        }
    }
    cells
}
