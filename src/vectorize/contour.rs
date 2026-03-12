//! Connected region finding and boundary contour tracing.
//!
//! The Kopf-Lischinski pipeline:
//! 1. Flood-fill connected regions using the similarity graph
//! 2. For each region, trace boundary edges between inside/outside pixels
//! 3. At grid corners where 4 pixels meet, apply Voronoi reshaping
//! 4. Fit quadratic B-splines through the corner sequence

use super::graph::SimilarityGraph;
use super::voronoi::{CornersGrid, Point};

/// A closed contour path with a sequence of corner points.
#[derive(Clone, Debug)]
pub struct ContourPath {
    pub color: u32,
    pub points: Vec<Point>,
}

/// Find all connected regions using flood-fill on the similarity graph.
/// Returns per-pixel region labels and per-region color.
fn label_regions(pixels: &[u32], graph: &SimilarityGraph) -> (Vec<u32>, Vec<u32>) {
    let w = graph.width;
    let h = graph.height;
    let n = w * h;
    let mut labels = vec![u32::MAX; n];
    let mut colors = Vec::new();
    let mut region_id = 0u32;

    for start in 0..n {
        if labels[start] != u32::MAX {
            continue;
        }
        let color = pixels[start];
        colors.push(color);

        let mut stack = vec![start];
        while let Some(idx) = stack.pop() {
            if labels[idx] != u32::MAX {
                continue;
            }
            labels[idx] = region_id;
            let x = idx % w;
            let y = idx / w;
            let e = graph.edge(x, y);

            if e.right && x + 1 < w && labels[idx + 1] == u32::MAX {
                stack.push(idx + 1);
            }
            if e.down && y + 1 < h && labels[idx + w] == u32::MAX {
                stack.push(idx + w);
            }
            if x > 0 && graph.edge(x - 1, y).right && labels[idx - 1] == u32::MAX {
                stack.push(idx - 1);
            }
            if y > 0 && graph.edge(x, y - 1).down && labels[idx - w] == u32::MAX {
                stack.push(idx - w);
            }
        }
        region_id += 1;
    }

    (labels, colors)
}

/// A directed boundary edge between two grid corners.
/// Identified by starting corner (cx, cy) and direction (0=R, 1=D, 2=L, 3=U).
/// Convention: the inside region is always to the LEFT of the travel direction.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct DirEdge {
    cx: i32,
    cy: i32,
    dir: u8, // 0=right, 1=down, 2=left, 3=up
}

impl DirEdge {
    /// The pixel to the LEFT of this directed edge (the "inside" pixel).
    fn left_pixel(&self) -> (i32, i32) {
        match self.dir {
            0 => (self.cx, self.cy - 1),       // right: left = above
            1 => (self.cx, self.cy),            // down: left = right-side pixel
            2 => (self.cx - 1, self.cy),        // left: left = below
            3 => (self.cx - 1, self.cy - 1),    // up: left = left-side pixel
            _ => unreachable!(),
        }
    }

    /// End corner of this edge.
    fn end_corner(&self) -> (i32, i32) {
        match self.dir {
            0 => (self.cx + 1, self.cy),
            1 => (self.cx, self.cy + 1),
            2 => (self.cx - 1, self.cy),
            3 => (self.cx, self.cy - 1),
            _ => unreachable!(),
        }
    }

    /// The unique undirected edge ID for visited tracking.
    /// Horizontal edges: stored as (min_cx, cy, true)
    /// Vertical edges: stored as (cx, min_cy, false)
    fn canonical(&self) -> (usize, usize, bool) {
        match self.dir {
            0 => (self.cx as usize, self.cy as usize, true),
            2 => (self.cx as usize - 1, self.cy as usize, true),
            1 => (self.cx as usize, self.cy as usize, false),
            3 => (self.cx as usize, self.cy as usize - 1, false),
            _ => unreachable!(),
        }
    }
}

/// Trace all boundary contours of a region.
fn trace_region_boundary(
    labels: &[u32],
    region_id: u32,
    w: usize,
    h: usize,
    corners: &CornersGrid,
) -> Vec<Vec<Point>> {
    let iw = w as i32;
    let ih = h as i32;

    let inside = |x: i32, y: i32| -> bool {
        x >= 0 && y >= 0 && x < iw && y < ih && labels[y as usize * w + x as usize] == region_id
    };

    // Collect all directed boundary edges (inside to left).
    // For each horizontal grid edge (cx, cy)→(cx+1, cy):
    //   pixel above = (cx, cy-1), pixel below = (cx, cy)
    //   If above is inside and below is outside → directed edge going RIGHT (inside=left=above)
    //   If below is inside and above is outside → directed edge going LEFT (inside=left=below)
    // Similarly for vertical edges.

    let cw = w + 1;
    let ch = h + 1;
    // Track visited undirected edges
    let mut visited_h = vec![false; cw * ch]; // horizontal: idx = cy * cw + cx
    let mut visited_v = vec![false; cw * ch]; // vertical: idx = cy * cw + cx

    let mut contours = Vec::new();

    // Scan for unvisited boundary edges
    for cy in 0..ch {
        for cx in 0..w {
            if visited_h[cy * cw + cx] {
                continue;
            }
            let above_in = inside(cx as i32, cy as i32 - 1);
            let below_in = inside(cx as i32, cy as i32);
            if above_in == below_in {
                continue;
            }
            // Found a boundary edge
            let start = if above_in {
                // Inside is above → go right
                DirEdge { cx: cx as i32, cy: cy as i32, dir: 0 }
            } else {
                // Inside is below → go left from (cx+1, cy)
                DirEdge { cx: cx as i32 + 1, cy: cy as i32, dir: 2 }
            };

            let pts = trace_one_contour(
                start, &inside, w, h, corners,
                &mut visited_h, &mut visited_v, cw,
            );
            if pts.len() >= 3 {
                contours.push(pts);
            }
        }
    }
    // Also scan vertical edges for any missed starts
    for cy in 0..h {
        for cx in 0..cw {
            if visited_v[cy * cw + cx] {
                continue;
            }
            let left_in = inside(cx as i32 - 1, cy as i32);
            let right_in = inside(cx as i32, cy as i32);
            if left_in == right_in {
                continue;
            }
            let start = if left_in {
                DirEdge { cx: cx as i32, cy: cy as i32, dir: 1 }
            } else {
                DirEdge { cx: cx as i32, cy: cy as i32 + 1, dir: 3 }
            };

            let pts = trace_one_contour(
                start, &inside, w, h, corners,
                &mut visited_h, &mut visited_v, cw,
            );
            if pts.len() >= 3 {
                contours.push(pts);
            }
        }
    }

    contours
}

/// Trace a single contour starting from the given directed edge.
fn trace_one_contour(
    start: DirEdge,
    inside: &dyn Fn(i32, i32) -> bool,
    w: usize,
    h: usize,
    corners: &CornersGrid,
    visited_h: &mut [bool],
    visited_v: &mut [bool],
    cw: usize,
) -> Vec<Point> {
    let iw = w as i32;
    let ih = h as i32;
    let mut points = Vec::new();
    let mut edge = start;
    let max_iter = 2 * (w + h + 2) * (w + h + 2); // generous upper bound

    for iter in 0..max_iter {
        // Mark this edge as visited
        let (ex, ey, horiz) = edge.canonical();
        if horiz {
            visited_h[ey * cw + ex] = true;
        } else {
            visited_v[ey * cw + ex] = true;
        }

        // Emit the START corner of this edge (reshaped)
        let corner_pt = reshaped_corner_for_edge(&edge, inside, corners, iw, ih);
        // Only emit if it's a turn point or reshaped (we'll optimize later)
        points.push(corner_pt);

        // Find next edge: at the end corner, try turning left first, then straight, then right
        let (ecx, ecy) = edge.end_corner();
        let left_dir = (edge.dir + 3) % 4;
        let straight_dir = edge.dir;
        let right_dir = (edge.dir + 1) % 4;

        let next_edge;
        if is_boundary_dir(ecx, ecy, left_dir, inside, iw, ih) {
            next_edge = DirEdge { cx: ecx, cy: ecy, dir: left_dir };
        } else if is_boundary_dir(ecx, ecy, straight_dir, inside, iw, ih) {
            next_edge = DirEdge { cx: ecx, cy: ecy, dir: straight_dir };
        } else if is_boundary_dir(ecx, ecy, right_dir, inside, iw, ih) {
            next_edge = DirEdge { cx: ecx, cy: ecy, dir: right_dir };
        } else {
            // U-turn (shouldn't happen for well-formed regions)
            next_edge = DirEdge { cx: ecx, cy: ecy, dir: (edge.dir + 2) % 4 };
        }

        edge = next_edge;

        // Check closure
        if edge.cx == start.cx && edge.cy == start.cy && edge.dir == start.dir {
            break;
        }

        if iter > 0 {
            // Safety: check if this edge was already visited
            let (vx, vy, vh) = edge.canonical();
            let already = if vh { visited_h[vy * cw + vx] } else { visited_v[vy * cw + vx] };
            if already {
                break;
            }
        }
    }

    points
}

/// Check if a directed boundary edge exists starting at (cx, cy) in direction dir.
/// A boundary edge has the region inside to the left and outside to the right.
fn is_boundary_dir(cx: i32, cy: i32, dir: u8, inside: &dyn Fn(i32, i32) -> bool, iw: i32, ih: i32) -> bool {
    let edge = DirEdge { cx, cy, dir };
    let (lx, ly) = edge.left_pixel();
    let l_in = lx >= 0 && ly >= 0 && lx < iw && ly < ih && inside(lx, ly);

    // Right pixel
    let r = match dir {
        0 => (cx, cy),           // right: right = below
        1 => (cx - 1, cy),       // down: right = left-side
        2 => (cx - 1, cy - 1),   // left: right = above
        3 => (cx, cy - 1),       // up: right = right-side
        _ => unreachable!(),
    };
    let r_in = r.0 >= 0 && r.1 >= 0 && r.0 < iw && r.1 < ih && inside(r.0, r.1);

    l_in && !r_in
}

/// Get the reshaped corner position for the start of a directed edge.
fn reshaped_corner_for_edge(
    edge: &DirEdge,
    inside: &dyn Fn(i32, i32) -> bool,
    corners: &CornersGrid,
    iw: i32,
    ih: i32,
) -> Point {
    let (lx, ly) = edge.left_pixel();
    if lx >= 0 && ly >= 0 && lx < iw && ly < ih && inside(lx, ly) {
        let px = lx as usize;
        let py = ly as usize;
        let corner_idx = pixel_corner_at(px, py, edge.cx as usize, edge.cy as usize);
        corners.corner_for_pixel(px, py, corner_idx)
    } else {
        Point::new(edge.cx as f64, edge.cy as f64)
    }
}

/// Given pixel (px, py) and grid corner (cx, cy), determine which corner
/// index (0=TL, 1=TR, 2=BR, 3=BL) this is.
fn pixel_corner_at(px: usize, py: usize, cx: usize, cy: usize) -> u8 {
    if cx == px && cy == py { 0 }
    else if cx == px + 1 && cy == py { 1 }
    else if cx == px + 1 && cy == py + 1 { 2 }
    else { 3 }
}

/// Build contour paths for all regions.
pub fn build_contours(
    pixels: &[u32],
    graph: &SimilarityGraph,
    corners: &CornersGrid,
) -> Vec<ContourPath> {
    let w = graph.width;
    let h = graph.height;

    let (labels, colors) = label_regions(pixels, graph);
    let num_regions = colors.len();
    let mut paths = Vec::new();

    for rid in 0..num_regions {
        let contours = trace_region_boundary(&labels, rid as u32, w, h, corners);
        for contour_pts in contours {
            paths.push(ContourPath {
                color: colors[rid],
                points: contour_pts,
            });
        }
    }

    paths
}

/// Simplify contour points: collapse collinear runs, then B-spline smooth.
pub fn smooth_contour(contour: &ContourPath) -> Vec<PathSegment> {
    // Deduplicate consecutive identical points
    let mut deduped = Vec::with_capacity(contour.points.len());
    for &p in &contour.points {
        if deduped.is_empty() || {
            let last: Point = *deduped.last().unwrap();
            (last.x - p.x).abs() > 1e-9 || (last.y - p.y).abs() > 1e-9
        } {
            deduped.push(p);
        }
    }
    if deduped.len() > 1 {
        let first = deduped[0];
        let last = *deduped.last().unwrap();
        if (first.x - last.x).abs() < 1e-9 && (first.y - last.y).abs() < 1e-9 {
            deduped.pop();
        }
    }

    if deduped.len() < 3 {
        return deduped.windows(2).map(|w| PathSegment::Line(w[0], w[1])).collect();
    }

    // Collapse collinear runs: keep only points where direction changes
    let n = deduped.len();
    let mut corners = Vec::with_capacity(n);
    for i in 0..n {
        let prev = deduped[(i + n - 1) % n];
        let curr = deduped[i];
        let next = deduped[(i + 1) % n];

        let cross = (curr.x - prev.x) * (next.y - curr.y) - (curr.y - prev.y) * (next.x - curr.x);
        if cross.abs() > 1e-9 {
            corners.push(curr);
        }
    }

    let n = corners.len();
    if n < 3 {
        if n == 2 {
            return vec![
                PathSegment::Line(corners[0], corners[1]),
                PathSegment::Line(corners[1], corners[0]),
            ];
        }
        return Vec::new();
    }

    // Classify each corner: if both adjacent edges are axis-aligned (horizontal
    // or vertical), it's a hard corner (90° on grid). Otherwise B-spline smooth.
    let is_axis_aligned = |a: Point, b: Point| -> bool {
        (a.x - b.x).abs() < 1e-9 || (a.y - b.y).abs() < 1e-9
    };

    // For each corner, check if prev→curr and curr→next are both axis-aligned
    let mut is_hard = vec![false; n];
    for i in 0..n {
        let prev = corners[(i + n - 1) % n];
        let curr = corners[i];
        let next = corners[(i + 1) % n];
        if is_axis_aligned(prev, curr) && is_axis_aligned(curr, next) {
            is_hard[i] = true;
        }
    }

    // B-spline through soft corners, lines through hard corners.
    // Walk the corners, emitting segments.
    let mut segments = Vec::with_capacity(n);
    for i in 0..n {
        let p0 = corners[i];
        let p1 = corners[(i + 1) % n];
        let p2 = corners[(i + 2) % n];
        let i1 = (i + 1) % n;

        if is_hard[i1] {
            // Hard corner at p1: emit two lines (mid01→p1 and p1→mid12)
            let mid01 = if is_hard[i] {
                p0 // previous was also hard, start from p0 not midpoint
            } else {
                Point::new((p0.x + p1.x) * 0.5, (p0.y + p1.y) * 0.5)
            };
            let mid12 = if is_hard[(i + 2) % n] {
                p2
            } else {
                Point::new((p1.x + p2.x) * 0.5, (p1.y + p2.y) * 0.5)
            };
            // Single line from mid01 through p1 to mid12 (two segments)
            segments.push(PathSegment::Line(mid01, p1));
            segments.push(PathSegment::Line(p1, mid12));
        } else {
            // Soft corner: quadratic B-spline
            let mid01 = if is_hard[i] {
                p0
            } else {
                Point::new((p0.x + p1.x) * 0.5, (p0.y + p1.y) * 0.5)
            };
            let mid12 = if is_hard[(i + 2) % n] {
                p2
            } else {
                Point::new((p1.x + p2.x) * 0.5, (p1.y + p2.y) * 0.5)
            };
            segments.push(PathSegment::QuadBezier(mid01, p1, mid12));
        }
    }

    segments
}

/// A path segment for SVG output.
#[derive(Clone, Debug)]
pub enum PathSegment {
    Line(Point, Point),
    QuadBezier(Point, Point, Point),
}

/// A complete colored path ready for SVG serialization.
#[derive(Clone, Debug)]
pub struct ColorPath {
    pub color: u32,
    pub segments: Vec<PathSegment>,
}
