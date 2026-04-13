# Vectorize Scanline Rasterizer — Face-Tracing Design

## Goal

Replace the `rasterize_scanline()` stub with a winding-rule scanline rasterizer that traces closed faces from the cell graph. Each source pixel's Voronoi cell becomes a closed polygon filled via nonzero winding rule. Output must match the nearest-curve rasterizer at pixel-interior regions, with smooth AA at boundaries.

## Why Face-Tracing

Previous attempts failed because they tried to assign winding or fill direction per-edge without topology:

- **Edge-table sweep**: Fills between sorted edges left-to-right. Fails when near-horizontal boundaries produce no edges on a scanline — the sweep uses the wrong color for entire spans.
- **Per-edge winding**: Each boundary edge gets +1/-1 winding per color. Fails because screen-left/right determination is unreliable (curves move from grid, tangent direction varies along a span, cross-product flips at inflection points).

Face-tracing solves both: each face is a **closed polygon** with **guaranteed-consistent winding** from the traversal order. No per-edge side determination needed.

## Architecture

### Phase 1: Face Extraction

Reuse the `gpu_svg.rs` algorithm verbatim — it already works correctly:

1. **`pixel_cell()`** — For each source pixel, read the resolved similarity graph's diagonal state at the 4 surrounding grid corners. Produce a Voronoi cell polygon (4-8 vertices) in x4 coordinates (4× grid resolution, integer).

2. **`build_cell_edges()`** — Walk all pixel cells, emit directed half-edges. For each canonical edge (min→max node pair), track which pixel color is on the forward vs reverse side. Boundary edges (different colors) produce two `DirEdge` entries (one per color).

3. **`trace_faces()`** — Planar face algorithm: angle-sorted adjacency, next-edge lookup via binary search, loop tracing. Each face is a closed sequence of x4 node IDs with a single fill color.

4. **`build_node_map()`** — Map x4 node IDs to optimized CP positions. CPs are matched by rounding their original (grid) positions to x4 coordinates. Border nodes and T-junctions/crossings marked sharp.

This code lives in `gpu_svg.rs` today (test_runner crate). For the scanline rasterizer, it needs to be accessible from the main library. Two options:

- **Option A**: Move face-tracing functions into `src/scaling/vectorize.rs` (or a new `src/scaling/vectorize_faces.rs`), make `gpu_svg.rs` call into the library.
- **Option B**: Duplicate the face-tracing in `vectorize.rs`, keeping `gpu_svg.rs` separate.

**Recommendation: Option A** — single source of truth, and the face-tracing is ~200 lines that naturally belong alongside the vectorize pipeline.

### Phase 2: Curve Flattening

For each traced face, flatten its boundary into line segments:

1. Walk the face's node sequence (closed loop of x4 node IDs).
2. For each consecutive pair of nodes, determine the curve type:
   - **Grid-only node** (no optimized position): straight line segment to the node's grid position (x4/4).
   - **Sharp node** (T-junction/crossing/border): straight line segment to the optimized position.
   - **Smooth node**: quadratic B-spline segment where the node's optimized position is the control point, and endpoints are midpoints with the adjacent nodes' positions. Adaptively subdivide based on curvature × scale.

This matches the SVG exporter's `face_to_svg_d()` logic but produces line segments instead of SVG path commands.

Each line segment is tagged with:
- The face's color
- Winding direction: +1 if the segment goes downward (y increases), -1 if upward. Since faces are traced in a consistent CW/CCW order, the winding is automatically consistent around each face.

### Phase 3: Scanline Fill

Build a row-indexed edge table from all faces' flattened edges, then sweep:

1. **Row index**: For each output row, a sorted list of edges that cross it (same prefix-sum flat array as before).

2. **Per-row sweep**: For each output pixel on the row:
   - Advance edge cursor past all edges with x < pixel_center
   - For each edge passed, accumulate `winding[color] += edge.winding`
   - The pixel's color is the color with nonzero winding (if multiple, last-one-wins)
   - If no color has nonzero winding, the pixel is in the background (detect from border pixels)

3. **Background**: Detect the background color from image border pixels (like `detect_bg()` in gpu_svg.rs). Fill the output buffer with background first, then the winding sweep overwrites interior pixels.

### Phase 4: Analytical AA

At edge boundary pixels (where an edge crosses the pixel):

1. Compute trapezoidal coverage from the edge's x-intercepts at pixel top/bottom.
2. Blend the two adjacent colors using `blend_linear_srgb()`.
3. The two colors are: the current winding color (nonzero) and the color that becomes nonzero after crossing this edge. Since face edges carry a single color, the "other" color is whatever was at this pixel before the current face's winding kicked in.

For initial implementation, skip AA (fill with hard winding result). Add AA as a follow-up once the fill is visually correct.

### Phase 5: Parallelism

Same pattern as existing `rasterize()`:
- `std::thread::scope` with `available_parallelism()`
- Split output into row chunks
- Each thread processes its chunk independently (row index is read-only)
- wasm32: sequential fallback

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/scaling/vectorize_faces.rs` | Create | Face extraction from cell graph: `pixel_cell`, `build_cell_edges`, `trace_faces`, `build_node_map`, `flatten_face` |
| `src/scaling/vectorize.rs` | Modify | `rasterize_scanline()` calls face extraction + scanline fill |
| `src/scaling/mod.rs` | Modify | Add `mod vectorize_faces;` |
| `src/test_runner/gpu_svg.rs` | Modify | Remove duplicated face-tracing code, import from library |

## Testing

- Compare `--filter vectorize-scanline` output against `--filter vectorize` for multiple test sprites and Game Boy frames.
- Interior pixels should be identical (same color regions).
- Boundary pixels may differ slightly (analytical AA vs nearest-curve AA).

## Constraints

- Must use the same `VectorizeData` pipeline output (stages 1-5).
- Face boundaries must follow the same optimized B-spline curves as the nearest-curve rasterizer.
- No nearest-neighbor fallback — all pixel colors determined by winding rule.
- VOID_COLOR faces (outside image boundary) are skipped.
