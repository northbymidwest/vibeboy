# Vectorize Scanline Rasterizer (Face-Tracing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a winding-rule scanline rasterizer that traces closed faces from the vectorize cell graph and fills them, replacing the `rasterize_scanline()` stub.

**Architecture:** Extract face-tracing code from `gpu_svg.rs` into a shared library module (`vectorize_faces.rs`). The scanline rasterizer calls face extraction to get closed polygons, flattens their B-spline boundaries into line segments tagged with color and winding, builds a row-indexed edge table, and sweeps each row accumulating per-color winding to determine pixel colors.

**Tech Stack:** Rust 2024 edition, existing `VectorizeData` pipeline, `std::thread::scope` parallelism.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/scaling/vectorize_faces.rs` | Create | Face extraction: `pixel_cell`, `build_cell_edges`, `trace_faces`, `build_node_map`, node pack/unpack/angle helpers |
| `src/scaling/mod.rs` | Modify | Add `pub mod vectorize_faces;` |
| `src/scaling/vectorize.rs` | Modify | Replace `rasterize_scanline()` stub with face-tracing + winding fill |
| `src/test_runner/gpu_svg.rs` | Modify | Import face-tracing from library instead of local copies |

---

### Task 1: Extract face-tracing into `vectorize_faces.rs`

**Files:**
- Create: `src/scaling/vectorize_faces.rs`
- Modify: `src/scaling/mod.rs`

Move the face-tracing algorithm from `src/test_runner/gpu_svg.rs` into the main library so both the SVG exporter and the scanline rasterizer can use it.

- [ ] **Step 1: Create `src/scaling/vectorize_faces.rs` with the face-tracing code**

Copy the following functions from `gpu_svg.rs` into the new file, making them `pub`:

```rust
//! Face extraction from the vectorize cell graph.
//!
//! Traces closed Voronoi cell faces from the resolved similarity graph
//! and maps face nodes to optimized B-spline control point positions.

use std::collections::BTreeMap;
use super::vectorize::VectorizeData;

/// Sentinel color for the void outside the image boundary.
pub const VOID_COLOR: u32 = 0x01000000;

const IS_TJUNCTION: u32 = 32;
const IS_CROSSING: u32 = 64;
const SHARP_MASK: u32 = IS_TJUNCTION | IS_CROSSING;

/// Pack x4/y4 coordinates into a u64 node identifier.
pub fn pack_node(x4: i32, y4: i32) -> u64 {
    ((x4 as u64) << 32) | (y4 as u32 as u64)
}

/// Unpack a u64 node identifier into x4/y4 coordinates.
pub fn unpack_node(nid: u64) -> (i32, i32) {
    ((nid >> 32) as i32, nid as u32 as i32)
}

/// Cross-product angle comparison (exact integer).
#[inline]
pub fn angle_cmp(adx: i64, ady: i64, bdx: i64, bdy: i64) -> std::cmp::Ordering {
    let ha = ady > 0 || (ady == 0 && adx > 0);
    let hb = bdy > 0 || (bdy == 0 && bdx > 0);
    if ha != hb {
        return if ha { std::cmp::Ordering::Less } else { std::cmp::Ordering::Greater };
    }
    (adx * bdy - ady * bdx).cmp(&0).reverse()
}

/// A directed half-edge in the planar cell graph.
pub struct DirEdge {
    pub from: u64,
    pub to: u64,
    pub color: u32,
}

/// A traced face: closed loop of x4 node IDs with a fill color.
pub struct Face {
    pub nodes: Vec<u64>,
    pub color: u32,
}

// Then: corner_diag(), pixel_cell(), build_cell_edges(), trace_faces(), build_node_map()
// copied verbatim from gpu_svg.rs lines 65-265, with pub visibility.
// build_cell_edges signature: pub fn build_cell_edges(data: &VectorizeData, pixels: &[u32]) -> Vec<DirEdge>
// trace_faces signature: pub fn trace_faces(edges: &[DirEdge]) -> Vec<Face>
// build_node_map signature: pub fn build_node_map(data: &VectorizeData) -> (BTreeMap<u64, (f64, f64)>, BTreeMap<u64, bool>)
```

Copy each function verbatim from `gpu_svg.rs`. The only changes:
- Add `pub` to `corner_diag`, `pixel_cell`, `build_cell_edges`, `trace_faces`, `build_node_map`
- Change `trace_faces` return type from `Vec<(Vec<u64>, u32)>` to `Vec<Face>`
- Use `super::vectorize::VectorizeData` instead of `vibeboy::scaling::vectorize::VectorizeData`

- [ ] **Step 2: Add module to `src/scaling/mod.rs`**

After the `pub mod vectorize;` line (line 19), add:

```rust
pub mod vectorize_faces;
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo build --release 2>&1 | tail -5`

- [ ] **Step 4: Commit**

```bash
git add src/scaling/vectorize_faces.rs src/scaling/mod.rs
git commit -m "Extract face-tracing from gpu_svg into vectorize_faces library module"
```

---

### Task 2: Migrate `gpu_svg.rs` to use the library module

**Files:**
- Modify: `src/test_runner/gpu_svg.rs`

Replace the local face-tracing code in `gpu_svg.rs` with imports from the new library module. This ensures a single source of truth.

- [ ] **Step 1: Replace local code with imports**

In `gpu_svg.rs`, remove:
- `pack_node`, `unpack_node`, `angle_cmp` (lines 38-57)
- `corner_diag`, `pixel_cell` (lines 63-109)
- `DirEdge` struct (lines 111-115)
- `build_cell_edges` (lines 120-155)
- `trace_faces` (lines 163-230)
- `build_node_map` (lines 240-265)
- Constants `IS_TJUNCTION`, `IS_CROSSING`, `SHARP_MASK`, `VOID_COLOR` (lines 14-19)

Replace with imports:

```rust
use vibeboy::scaling::vectorize_faces::{
    self, DirEdge, Face, VOID_COLOR, pack_node, unpack_node, build_cell_edges, trace_faces, build_node_map,
};
```

Keep: `fmt()`, `hex()`, `face_to_svg_d()`, `detect_bg()`, `render_svg()`.

Update `render_svg()` to use `Face` struct fields instead of tuple destructuring:
- Change `for (nodes, color) in &faces` to `for face in &faces`
- Use `face.nodes` and `face.color` instead of destructured names

Update `face_to_svg_d()` signature if needed (it takes `&[u64]` for nodes, which is fine).

- [ ] **Step 2: Verify it compiles and SVG export still works**

```bash
cargo build --release 2>&1 | tail -5
cargo run --release --bin test_runner -- vectorize vectorize-tests/smw_mario_input.png --out /tmp/test.svg 2>&1 | tail -1
```

Expected: SVG file produced, same as before.

- [ ] **Step 3: Commit**

```bash
git add src/test_runner/gpu_svg.rs
git commit -m "Migrate gpu_svg to use shared vectorize_faces module"
```

---

### Task 3: Implement curve flattening for traced faces

**Files:**
- Modify: `src/scaling/vectorize_faces.rs`

Add a function that takes a traced face and produces flattened line segments suitable for scanline fill.

- [ ] **Step 1: Add `WindingEdge` struct and `flatten_face` function**

Add to `vectorize_faces.rs`:

```rust
/// A directed line segment for winding-rule scanline fill.
/// y_min < y_max. Winding is +1 if original direction was downward, -1 if upward.
pub struct WindingEdge {
    pub x_at_ymin: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub dx_per_dy: f32,
    pub color: u32,
    pub winding: i8,
}

/// Flatten a traced face's boundary into directed line segments for scanline fill.
///
/// Walks the face's node sequence. For each node:
/// - Grid-only nodes (no optimized position): straight line to grid position.
/// - Sharp nodes (T-junction/crossing/border): straight line to optimized position.
/// - Smooth nodes: quadratic B-spline with node as control point and endpoints at
///   midpoints with adjacent nodes. Adaptively subdivided.
///
/// Each segment is tagged with the face's color. Winding direction is +1 for
/// downward segments, -1 for upward, giving consistent winding around the face.
pub fn flatten_face(
    face: &Face,
    pos_map: &BTreeMap<u64, (f64, f64)>,
    sharp_map: &BTreeMap<u64, bool>,
    scale_factor: f32,
    edges_out: &mut Vec<WindingEdge>,
) {
    let n = face.nodes.len();
    if n < 3 { return; }

    let get_pos = |nid: u64| -> (f32, f32) {
        pos_map.get(&nid).map(|&(x, y)| (x as f32, y as f32)).unwrap_or_else(|| {
            let (x4, y4) = unpack_node(nid);
            (x4 as f32 / 4.0, y4 as f32 / 4.0)
        })
    };
    let is_optimized = |nid: u64| pos_map.contains_key(&nid);
    let is_sharp = |nid: u64| sharp_map.get(&nid).copied().unwrap_or(false);
    let is_grid = |nid: u64| !is_optimized(nid);
    let mid = |a: (f32, f32), b: (f32, f32)| ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5);

    // Collect the polyline/curve points that form the face boundary.
    // This mirrors face_to_svg_d() but produces (x,y) points instead of SVG commands.
    let mut pts: Vec<(f32, f32)> = Vec::new();

    // Starting point (same logic as face_to_svg_d)
    let first = get_pos(face.nodes[0]);
    let last = get_pos(face.nodes[n - 1]);
    let start = if is_grid(face.nodes[0]) || is_sharp(face.nodes[0]) {
        first
    } else {
        mid(last, first)
    };
    pts.push((start.0 * scale_factor, start.1 * scale_factor));

    for i in 0..n {
        let nid = face.nodes[i];
        let next_nid = face.nodes[(i + 1) % n];
        let p = get_pos(nid);
        let np = get_pos(next_nid);

        if is_grid(nid) || is_sharp(nid) {
            // Straight line to the node position
            pts.push((p.0 * scale_factor, p.1 * scale_factor));
            if !is_grid(next_nid) && !is_sharp(next_nid) {
                let m = mid(p, np);
                pts.push((m.0 * scale_factor, m.1 * scale_factor));
            }
        } else {
            // Quadratic B-spline: control point at p, endpoint at mid(p, np)
            let end = mid(p, np);
            let ctrl = (p.0 * scale_factor, p.1 * scale_factor);
            let end_s = (end.0 * scale_factor, end.1 * scale_factor);
            let start_pt = *pts.last().unwrap();

            // Adaptive subdivision based on deviation from chord
            let chord_mid = mid(start_pt, end_s);
            let dev = ((ctrl.0 - chord_mid.0).powi(2) + (ctrl.1 - chord_mid.1).powi(2)).sqrt();
            let subdiv = (dev * 0.5).ceil().clamp(1.0, 16.0) as usize;

            for s in 1..=subdiv {
                let t = s as f32 / subdiv as f32;
                let u = 1.0 - t;
                // Quadratic Bézier: (1-t)²·start + 2(1-t)t·ctrl + t²·end
                let x = u * u * start_pt.0 + 2.0 * u * t * ctrl.0 + t * t * end_s.0;
                let y = u * u * start_pt.1 + 2.0 * u * t * ctrl.1 + t * t * end_s.1;
                pts.push((x, y));
            }
        }
    }

    // Convert consecutive points into winding edges.
    // Close the polygon by connecting last point back to first.
    let color = face.color;
    for i in 0..pts.len() {
        let (x0, y0) = pts[i];
        let (x1, y1) = pts[(i + 1) % pts.len()];

        let dy = y1 - y0;
        if dy.abs() < 1e-6 { continue; } // skip near-horizontal

        let winding: i8 = if dy > 0.0 { 1 } else { -1 };
        let (ymin, ymax, x_at_ymin) = if y0 < y1 {
            (y0, y1, x0)
        } else {
            (y1, y0, x1)
        };
        let dx_per_dy = (x1 - x0) / dy;

        edges_out.push(WindingEdge {
            x_at_ymin, y_min: ymin, y_max: ymax, dx_per_dy,
            color, winding,
        });
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo build --release 2>&1 | tail -5`

- [ ] **Step 3: Commit**

```bash
git add src/scaling/vectorize_faces.rs
git commit -m "Add flatten_face: B-spline curve flattening for scanline fill"
```

---

### Task 4: Implement winding-rule scanline fill

**Files:**
- Modify: `src/scaling/vectorize.rs`

Replace the `rasterize_scanline()` stub with the real implementation.

- [ ] **Step 1: Update `scale_scanline()` to pass `VectorizeData` directly**

Change `scale_scanline()` to pass the whole `VectorizeData` to the rasterizer instead of unpacking fields:

```rust
pub fn scale_scanline(src: &[u32], src_w: usize, src_h: usize, scale_factor: f32) -> Vec<u32> {
    let out_w = (src_w as f32 * scale_factor).ceil() as usize;
    let out_h = (src_h as f32 * scale_factor).ceil() as usize;
    let data = vectorize(src, src_w, src_h);
    rasterize_scanline(src, &data, out_w, out_h, scale_factor)
}
```

- [ ] **Step 2: Replace `rasterize_scanline()` with face-tracing + winding fill**

```rust
fn rasterize_scanline(
    pixels: &[u32],
    data: &VectorizeData,
    out_w: usize,
    out_h: usize,
    scale_factor: f32,
) -> Vec<u32> {
    use super::vectorize_faces::{
        self, WindingEdge, VOID_COLOR, build_cell_edges, trace_faces, build_node_map, flatten_face,
    };

    // Phase 1: Trace faces
    let dir_edges = build_cell_edges(data, pixels);
    let faces = trace_faces(&dir_edges);
    let (pos_map, sharp_map) = build_node_map(data);

    // Phase 2: Flatten all face boundaries into winding edges
    let mut edges: Vec<WindingEdge> = Vec::new();
    for face in &faces {
        if face.color == VOID_COLOR { continue; }
        flatten_face(face, &pos_map, &sharp_map, scale_factor, &mut edges);
    }

    // Phase 3: Build row index
    let mut row_count = vec![0u32; out_h];
    for edge in &edges {
        let r_start = (edge.y_min.floor() as usize).min(out_h.saturating_sub(1));
        let r_end = (edge.y_max.ceil() as usize).min(out_h);
        for r in r_start..r_end { row_count[r] += 1; }
    }
    let mut row_offsets = vec![0u32; out_h + 1];
    for r in 0..out_h { row_offsets[r + 1] = row_offsets[r] + row_count[r]; }
    let total = row_offsets[out_h] as usize;
    let mut row_data = vec![0u32; total];
    let mut fill_pos = row_offsets[..out_h].to_vec();
    for (ei, edge) in edges.iter().enumerate() {
        let r_start = (edge.y_min.floor() as usize).min(out_h.saturating_sub(1));
        let r_end = (edge.y_max.ceil() as usize).min(out_h);
        for r in r_start..r_end {
            row_data[fill_pos[r] as usize] = ei as u32;
            fill_pos[r] += 1;
        }
    }

    // Detect background color from image border
    let bg = detect_bg(pixels, data.img_w, data.img_h);

    // Collect unique colors for winding map
    let mut color_set: Vec<u32> = Vec::new();
    for e in &edges {
        if !color_set.contains(&e.color) { color_set.push(e.color); }
    }
    let num_colors = color_set.len();

    // Phase 4: Fill output buffer
    let mut output = vec![pack_color(bg); out_w * out_h];

    let process_rows = |chunk: &mut [u32], start_row: usize| {
        let chunk_rows = chunk.len() / out_w;
        let mut winding = vec![0i16; num_colors];
        let mut row_edges: Vec<(f32, u32)> = Vec::new();

        for local_y in 0..chunk_rows {
            let opy = start_row + local_y;
            let row_center = opy as f32 + 0.5;
            let row_slice = &mut chunk[local_y * out_w..(local_y + 1) * out_w];

            let rd_start = row_offsets[opy] as usize;
            let rd_end = row_offsets[opy + 1] as usize;
            if rd_start == rd_end { continue; } // all background

            // Collect and sort edges by x at row center
            row_edges.clear();
            for &ei in &row_data[rd_start..rd_end] {
                let edge = &edges[ei as usize];
                let y_clamp = row_center.clamp(edge.y_min, edge.y_max);
                let x = edge.x_at_ymin + (y_clamp - edge.y_min) * edge.dx_per_dy;
                row_edges.push((x, ei));
            }
            row_edges.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Sweep left to right accumulating winding per color
            for w in winding.iter_mut() { *w = 0; }
            let mut edge_cursor = 0;

            for opx in 0..out_w {
                let px_center = opx as f32 + 0.5;

                while edge_cursor < row_edges.len() && row_edges[edge_cursor].0 < px_center {
                    let ei = row_edges[edge_cursor].1 as usize;
                    let edge = &edges[ei];
                    if let Some(ci) = color_set.iter().position(|&c| c == edge.color) {
                        winding[ci] += edge.winding as i16;
                    }
                    edge_cursor += 1;
                }

                // Find color with nonzero winding (last one wins if multiple)
                let mut color = 0u32;
                for (i, &w) in winding.iter().enumerate() {
                    if w != 0 { color = color_set[i]; }
                }
                if color != 0 {
                    row_slice[opx] = pack_color(color);
                }
            }
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let rows_per_thread = (out_h + num_threads - 1) / num_threads;
        std::thread::scope(|scope| {
            let chunks: Vec<&mut [u32]> = output.chunks_mut(out_w * rows_per_thread).collect();
            let handles: Vec<_> = chunks.into_iter().enumerate().map(|(ci, chunk)| {
                let start = ci * rows_per_thread;
                scope.spawn(move || { process_rows(chunk, start); })
            }).collect();
            for h in handles { h.join().unwrap(); }
        });
    }
    #[cfg(target_arch = "wasm32")]
    {
        process_rows(&mut output, 0);
    }

    output
}

fn detect_bg(pixels: &[u32], w: usize, h: usize) -> u32 {
    let mut counts = std::collections::HashMap::new();
    for x in 0..w {
        *counts.entry(pixels[x]).or_insert(0usize) += 1;
        *counts.entry(pixels[(h - 1) * w + x]).or_insert(0) += 1;
    }
    for y in 1..h - 1 {
        *counts.entry(pixels[y * w]).or_insert(0) += 1;
        *counts.entry(pixels[y * w + w - 1]).or_insert(0) += 1;
    }
    counts.into_iter().max_by_key(|&(_, n)| n).map(|(c, _)| c).unwrap_or(0)
}
```

- [ ] **Step 3: Remove the old TODO comment block and unused `#[allow(dead_code)]` on `blend_linear_srgb`**

Remove lines 89-101 (the TODO comment block).
Remove `#[allow(dead_code)]` from `blend_linear_srgb` (line 104).

- [ ] **Step 4: Verify it compiles**

Run: `cargo build --release 2>&1 | tail -5`

- [ ] **Step 5: Commit**

```bash
git add src/scaling/vectorize.rs
git commit -m "Implement winding-rule scanline rasterizer with face-tracing"
```

---

### Task 5: Visual validation

**Files:** None (testing only)

- [ ] **Step 1: Generate comparison outputs**

```bash
# Sprite comparison
cargo run --release --bin test_runner -- vectorize vectorize-tests/smw_mario_input.png --out /tmp/mario-nearest.png --filter vectorize --scale 8
cargo run --release --bin test_runner -- vectorize vectorize-tests/smw_mario_input.png --out /tmp/mario-scanline.png --filter vectorize-scanline --scale 8

# Game Boy frame
cargo run --release --bin test_runner -- screenshot game-boy-test-roms/cgb-acid2/cgb-acid2.gbc --frames 300 --out /tmp/acid2-nearest.png --filter vectorize --scale 6
cargo run --release --bin test_runner -- screenshot game-boy-test-roms/cgb-acid2/cgb-acid2.gbc --frames 300 --out /tmp/acid2-scanline.png --filter vectorize-scanline --scale 6
```

- [ ] **Step 2: Compare visually**

Check:
- All color regions present and correct (no missing faces, no wrong colors)
- No horizontal banding or streaking artifacts
- Smooth boundaries (curves, not pixelated)
- Background fills correctly

- [ ] **Step 3: Fix any issues found, commit fixes**

---

### Task 6: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update filter count and add vectorize_faces module**

In the scaling filter infrastructure section:
- Update filter count from 35 to 36
- Add line: `- \`vectorize_faces.rs\`: Face extraction from vectorize cell graph (Voronoi cells, planar face tracing, node-to-CP mapping). Shared between SVG export and scanline rasterizer.`

- [ ] **Step 2: Add CLI example**

In the test runner section, add:

```bash
# Scanline rasterizer (face-tracing winding fill, faster CPU path)
cargo run --release --bin test_runner -- vectorize input.png --out output.png --filter vectorize-scanline --scale 8
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "Document vectorize-scanline filter and vectorize_faces module"
```
