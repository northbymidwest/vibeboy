# Vectorize Scanline Rasterizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CPU scanline rasterizer (`vectorize-scanline`) that reuses the existing vectorize stages 1-5 but replaces the expensive per-pixel nearest-curve rasterizer with an edge-table scanline fill using analytical anti-aliasing.

**Architecture:** Flatten optimized B-spline curves into line segments, build a row-indexed edge table, then sweep scanlines computing exact fractional pixel coverage from edge x-intercepts. Each edge carries its left/right colors from the source pixels. Interior pixels get direct color assignment; boundary pixels get a two-color analytical blend in linear sRGB. At T-junctions/crossings where 3+ colors meet, pick the dominant edge and accept sub-pixel error.

**Tech Stack:** Rust (2024 edition), existing `VectorizeData` pipeline, rayon-style row parallelism on native, sequential on wasm.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/scaling/vectorize.rs` | Modify | Add `scale_scanline()` entry point and `rasterize_scanline()` function |
| `src/scaling/mod.rs` | Modify | Add `ScaleFilter::VectorizeScanline` variant, registry entry, `cpu_scale()` dispatch |
| `src/test_runner/commands.rs` | Modify | Wire up `vectorize-scanline` in test runner |

All new code goes in `vectorize.rs` alongside the existing `rasterize()`. No new files needed.

---

### Task 1: Add ScaleFilter::VectorizeScanline to the filter registry

**Files:**
- Modify: `src/scaling/mod.rs:136-170` (enum), `src/scaling/mod.rs:183-218` (REGISTRY), `src/scaling/mod.rs:428-434` (cpu_scale dispatch)

- [ ] **Step 1: Add enum variant**

In `src/scaling/mod.rs`, add `VectorizeScanline` after `Vectorize` (line 165):

```rust
    /// CPU implementation of the full GPU vectorize pipeline.
    Vectorize,
    /// CPU scanline rasterizer: analytical AA, edge-table fill.
    VectorizeScanline,
    /// ScaleFX 3x edge interpolation (Sp00kyFox).
    ScaleFx,
```

- [ ] **Step 2: Add registry entry**

In the `REGISTRY` array (after the Vectorize entry at line 209), add:

```rust
    FilterInfo { filter: ScaleFilter::VectorizeScanline, cli_name: "vectorize-scanline", display_name: "Vectorize Scanline", factor: 0 },
```

- [ ] **Step 3: Add cpu_scale dispatch**

In `cpu_scale()`, after the `ScaleFilter::Vectorize` arm (line 428-434), add:

```rust
        ScaleFilter::VectorizeScanline => {
            let scale_f = (disp_w as f32 / sw as f32).min(disp_h as f32 / sh as f32);
            let ow = (sw as f32 * scale_f).ceil() as usize;
            let oh = (sh as f32 * scale_f).ceil() as usize;
            let s = vectorize::scale_scanline(src, sw, sh, scale_f);
            (s, ow as u32, oh as u32)
        }
```

- [ ] **Step 4: Add stub `scale_scanline()` in vectorize.rs**

In `src/scaling/vectorize.rs`, after `scale()` (line 87), add a stub so it compiles:

```rust
/// Public entry point: runs stages 1-5, then scanline rasterizer.
pub fn scale_scanline(src: &[u32], src_w: usize, src_h: usize, scale_factor: f32) -> Vec<u32> {
    let out_w = (src_w as f32 * scale_factor).ceil() as usize;
    let out_h = (src_h as f32 * scale_factor).ceil() as usize;

    let data = vectorize(src, src_w, src_h);

    rasterize_scanline(
        src,
        &data.positions,
        &data.orig_positions,
        &data.flags,
        &data.neighbors,
        data.img_w,
        data.img_h,
        out_w,
        out_h,
        scale_factor,
    )
}

fn rasterize_scanline(
    pixels: &[u32],
    positions: &[f32],
    orig_positions: &[f32],
    flags: &[u32],
    cp_neighbors: &[i32],
    img_w: usize,
    img_h: usize,
    out_w: usize,
    out_h: usize,
    scale_factor: f32,
) -> Vec<u32> {
    // Stub: fall back to nearest-curve rasterizer until implemented
    rasterize(pixels, positions, orig_positions, flags, cp_neighbors, img_w, img_h, out_w, out_h, scale_factor)
}
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo build --release 2>&1 | tail -5`
Expected: successful build (warnings OK)

- [ ] **Step 6: Verify filter is accessible from CLI**

Run: `cargo run --release --bin test_runner -- vectorize test-assets/some-small-image.png --out /tmp/scanline-test.png --filter vectorize-scanline --scale 4`
Expected: produces output (uses stub fallback)

- [ ] **Step 7: Commit**

```bash
git add src/scaling/mod.rs src/scaling/vectorize.rs
git commit -m "Add VectorizeScanline filter variant with stub rasterizer"
```

---

### Task 2: Edge flattening — walk B-spline chains and subdivide into line segments

**Files:**
- Modify: `src/scaling/vectorize.rs`

The edge table is the core data structure. Each line segment stores:

```rust
struct Edge {
    x0: f32,         // x at y_min
    y_min: f32,      // top of segment (in output coords)
    y_max: f32,      // bottom of segment (in output coords)
    dx_per_dy: f32,  // slope: (x1 - x0) / (y1 - y0)
    color_left: u32,  // source pixel color on left side
    color_right: u32, // source pixel color on right side
}
```

- [ ] **Step 1: Add the Edge struct and edge flattening function**

In `src/scaling/vectorize.rs`, add after the `CpData` struct (~line 1571):

```rust
/// A line segment in the edge table, in output pixel coordinates.
struct ScanEdge {
    x_at_ymin: f32,
    y_min: f32,
    y_max: f32,
    dx_per_dy: f32,
    color_left: u32,
    color_right: u32,
}
```

- [ ] **Step 2: Implement the edge chain walker**

Add a function that walks CP chains and collects B-spline spans:

```rust
/// Walk all CP chains and flatten B-spline curves into line segments.
/// Returns edges sorted by y_min, in output pixel coordinates.
fn build_edge_table(
    pixels: &[u32],
    positions: &[f32],
    orig_positions: &[f32],
    flags: &[u32],
    cp_neighbors: &[i32],
    img_w: usize,
    img_h: usize,
    scale_factor: f32,
    out_h: usize,
) -> (Vec<ScanEdge>, Vec<u32>, Vec<u32>) {
    let corners_w = img_w + 1;
    let corners_h = img_h + 1;
    let num_cps = corners_w * corners_h * 2;

    let get_px_color = |px: i32, py: i32| -> u32 {
        if px < 0 || py < 0 || px >= img_w as i32 || py >= img_h as i32 {
            return 0xFF000000; // black border
        }
        pixels[py as usize * img_w + px as usize] | 0xFF000000
    };

    let get_edge_colors = |icx: i32, icy: i32, dir: i32| -> (u32, u32) {
        match dir {
            0 => (get_px_color(icx - 1, icy - 1), get_px_color(icx, icy - 1)),
            1 => (get_px_color(icx, icy - 1), get_px_color(icx, icy)),
            2 => (get_px_color(icx, icy), get_px_color(icx - 1, icy)),
            3 => (get_px_color(icx - 1, icy), get_px_color(icx - 1, icy - 1)),
            _ => (0, 0),
        }
    };

    let mut edges = Vec::new();
    let mut visited = vec![false; num_cps];

    for ci in 0..num_cps {
        if visited[ci] { continue; }
        let next_ci = cp_neighbors[ci * 4 + 1];
        if next_ci < 0 { continue; } // no outgoing edge

        // Walk chain from this CP
        let mut cur = ci as i32;
        while cur >= 0 && !visited[cur as usize] {
            visited[cur as usize] = true;
            let idx = cur as usize;
            let next = cp_neighbors[idx * 4 + 1];
            if next < 0 { break; }

            let next_dir = cp_neighbors[idx * 4 + 3];
            if next_dir < 0 { cur = next; continue; }

            // Get the three control points for this B-spline span
            let prev_ci = cp_neighbors[idx * 4]; // for the span, prev = cur, mid = midpoint, next = next
            let px = positions[idx * 2];
            let py = positions[idx * 2 + 1];
            let nx = positions[next as usize * 2];
            let ny = positions[next as usize * 2 + 1];

            // B-spline span: P0=cur, P1=midpoint(cur,next), P2=next
            // Actually the quadratic B-spline uses: prev_pos, pos, next_pos
            // where the curve goes from midpoint(prev,pos) to midpoint(pos,next)
            // For edge flattening, we need the span that this CP contributes to:
            // The span from this CP to the next uses (prev_of_next, cur, next) as control points
            // But simpler: use beval(cur_pos, next_pos_as_control, next_next_pos) 
            // 
            // Actually each CP pair (ci, next_ci) defines a curve segment where:
            //   P0 = position[ci], P1 = position[next_ci], P2 = position[next_next_ci]
            // and the curve evaluates as beval(P0, P1, P2, t) for t in [0, 1]
            //
            // For the edge table we just need the span between consecutive CPs.
            // The B-spline basis means the actual curve passes through midpoints
            // of consecutive CPs. We flatten each span into line segments.

            let icx = (idx % (corners_w * 2)) as i32 / 2;
            let icy = (idx / (corners_w * 2)) as i32;

            let (color_left, color_right) = get_edge_colors(icx, icy, next_dir);

            // Skip edges where both sides are the same color
            if color_left == color_right {
                cur = next;
                continue;
            }

            // Get the three control points for this B-spline segment
            let next_next = cp_neighbors[next as usize * 4 + 1];
            let (p0x, p0y) = (px, py);
            let (p1x, p1y) = (nx, ny);
            let (p2x, p2y) = if next_next >= 0 {
                (positions[next_next as usize * 2], positions[next_next as usize * 2 + 1])
            } else {
                (nx, ny) // degenerate: straight line
            };

            // Adaptive subdivision based on curvature
            // Estimate curvature from control point deviation
            let mid_x = (p0x + p2x) * 0.5;
            let mid_y = (p0y + p2y) * 0.5;
            let dev = ((p1x - mid_x).powi(2) + (p1y - mid_y).powi(2)).sqrt();
            let num_segments = ((dev * scale_factor * 2.0).ceil() as usize).clamp(1, 16);

            let inv_n = 1.0 / num_segments as f32;
            for seg in 0..num_segments {
                let t0 = seg as f32 * inv_n;
                let t1 = (seg + 1) as f32 * inv_n;

                let (x0, y0) = beval((p0x, p0y), (p1x, p1y), (p2x, p2y), t0);
                let (x1, y1) = beval((p0x, p0y), (p1x, p1y), (p2x, p2y), t1);

                // Scale to output coordinates
                let ox0 = x0 * scale_factor;
                let oy0 = y0 * scale_factor;
                let ox1 = x1 * scale_factor;
                let oy1 = y1 * scale_factor;

                // Ensure y_min < y_max (swap if needed, flip colors for winding)
                let (sx0, sy_min, sy_max, sdx, cl, cr) = if oy0 < oy1 {
                    let dy = oy1 - oy0;
                    if dy < 1e-6 { continue; } // horizontal edge, skip
                    (ox0, oy0, oy1, (ox1 - ox0) / dy, color_left, color_right)
                } else {
                    let dy = oy0 - oy1;
                    if dy < 1e-6 { continue; }
                    (ox1, oy1, oy0, (ox0 - ox1) / dy, color_right, color_left)
                };

                edges.push(ScanEdge {
                    x_at_ymin: sx0,
                    y_min: sy_min,
                    y_max: sy_max,
                    dx_per_dy: sdx,
                    color_left: cl,
                    color_right: cr,
                });
            }

            cur = next;
        }
    }

    // Build row index: for each output row, which edges cross it
    let mut row_starts = vec![0u32; out_h + 1];
    // Count edges per row
    for edge in &edges {
        let row_min = (edge.y_min.floor() as usize).min(out_h - 1);
        let row_max = (edge.y_max.ceil() as usize).min(out_h);
        for row in row_min..row_max {
            row_starts[row + 1] += 1;
        }
    }
    // Prefix sum
    for i in 1..=out_h {
        row_starts[i] += row_starts[i - 1];
    }
    // Fill row index
    let total = row_starts[out_h] as usize;
    let mut row_indices = vec![0u32; total];
    let mut row_counts = vec![0u32; out_h];
    for (ei, edge) in edges.iter().enumerate() {
        let row_min = (edge.y_min.floor() as usize).min(out_h - 1);
        let row_max = (edge.y_max.ceil() as usize).min(out_h);
        for row in row_min..row_max {
            let offset = row_starts[row] + row_counts[row];
            row_indices[offset as usize] = ei as u32;
            row_counts[row] += 1;
        }
    }

    (edges, row_starts, row_indices)
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo build --release 2>&1 | tail -5`
Expected: successful build (function not yet called, so no link errors)

- [ ] **Step 4: Commit**

```bash
git add src/scaling/vectorize.rs
git commit -m "Add edge flattening: walk B-spline chains into line segments with row index"
```

---

### Task 3: Scanline fill with analytical anti-aliasing

**Files:**
- Modify: `src/scaling/vectorize.rs`

This is the core rasterizer. For each output row:
1. Collect active edges from the row index
2. Sort edges by x-intercept at the scanline center
3. Walk edges left-to-right, tracking the current fill color
4. For boundary pixels where an edge crosses: compute exact fractional coverage from the edge's x-intercepts at the pixel's top and bottom scanlines, blend two colors analytically

- [ ] **Step 1: Implement the scanline rasterizer**

Replace the stub `rasterize_scanline()` with the real implementation:

```rust
fn rasterize_scanline(
    pixels: &[u32],
    positions: &[f32],
    orig_positions: &[f32],
    flags: &[u32],
    cp_neighbors: &[i32],
    img_w: usize,
    img_h: usize,
    out_w: usize,
    out_h: usize,
    scale_factor: f32,
) -> Vec<u32> {
    if out_w == 0 || out_h == 0 {
        return Vec::new();
    }

    let (edges, row_starts, row_indices) = build_edge_table(
        pixels, positions, orig_positions, flags, cp_neighbors,
        img_w, img_h, scale_factor, out_h,
    );

    let inv_scale = 1.0 / scale_factor;

    // Fallback color lookup from source pixels
    let get_src_color = |ox: usize, oy: usize| -> u32 {
        let sx = ((ox as f32 + 0.5) * inv_scale) as usize;
        let sy = ((oy as f32 + 0.5) * inv_scale) as usize;
        let sx = sx.min(img_w - 1);
        let sy = sy.min(img_h - 1);
        pixels[sy * img_w + sx] | 0xFF000000
    };

    let mut output = vec![0u32; out_w * out_h];

    // Process each row
    let rasterize_row = |row: usize, out_row: &mut [u32]| {
        let y_center = row as f32 + 0.5;
        let y_top = row as f32;
        let y_bot = (row + 1) as f32;

        // Collect edges for this row with their x-intercepts at row center
        let start = row_starts[row] as usize;
        let end = row_starts[row + 1] as usize;

        struct ActiveEdge {
            x_at_center: f32,  // x-intercept at y_center
            x_at_top: f32,     // x-intercept at y_top
            x_at_bot: f32,     // x-intercept at y_bot
            y_min: f32,
            y_max: f32,
            color_left: u32,
            color_right: u32,
        }

        let mut active: Vec<ActiveEdge> = Vec::with_capacity(end - start);
        for i in start..end {
            let ei = row_indices[i] as usize;
            let e = &edges[ei];

            // Clamp y range to this row for coverage computation
            let cy_top = y_top.max(e.y_min);
            let cy_bot = y_bot.min(e.y_max);
            if cy_top >= cy_bot { continue; }

            let x_center = e.x_at_ymin + (y_center - e.y_min) * e.dx_per_dy;
            let x_top = e.x_at_ymin + (cy_top - e.y_min) * e.dx_per_dy;
            let x_bot = e.x_at_ymin + (cy_bot - e.y_min) * e.dx_per_dy;

            active.push(ActiveEdge {
                x_at_center: x_center,
                x_at_top: x_top,
                x_at_bot: x_bot,
                y_min: cy_top,
                y_max: cy_bot,
                color_left: e.color_left,
                color_right: e.color_right,
            });
        }

        // Sort by x at center of row
        active.sort_by(|a, b| a.x_at_center.partial_cmp(&b.x_at_center).unwrap_or(std::cmp::Ordering::Equal));

        // Fill pixels
        // For each edge, it transitions from color_left to color_right as we cross it.
        // Pixels fully to the left of all edges get source pixel color (fallback).
        // Between edges, the color is determined by the rightmost edge we've crossed.

        // Precompute fallback colors for the row
        // Then overwrite with edge-derived colors
        for x in 0..out_w {
            out_row[x] = get_src_color(x, row);
        }

        for ae in &active {
            let x_min = ae.x_at_top.min(ae.x_at_center).min(ae.x_at_bot);
            let x_max = ae.x_at_top.max(ae.x_at_center).max(ae.x_at_bot);

            // Pixel range this edge affects
            let px_start = (x_min.floor() as i32).max(0) as usize;
            let px_end = ((x_max.ceil() as i32) + 1).min(out_w as i32) as usize;

            // Vertical coverage: fraction of the pixel row that the edge spans
            let v_coverage = (ae.y_max - ae.y_min); // 0..1 range within this pixel row

            for px in px_start..px_end {
                let px_left = px as f32;
                let px_right = (px + 1) as f32;

                // Compute area of the pixel that is to the RIGHT of the edge.
                // The edge crosses from (x_at_top, y_top) to (x_at_bot, y_bot).
                // Area to the right = pixel_area - area_to_left
                // For a line crossing a unit square, the covered area is a trapezoid.

                // X-intercepts clamped to pixel bounds
                let xt = ae.x_at_top.clamp(px_left, px_right);
                let xb = ae.x_at_bot.clamp(px_left, px_right);

                // Fraction of pixel width on the left side of the edge
                let frac_top = (xt - px_left); // 0..1
                let frac_bot = (xb - px_left); // 0..1

                // Average coverage (trapezoidal rule) * vertical coverage
                let left_coverage = ((frac_top + frac_bot) * 0.5 * v_coverage).clamp(0.0, 1.0);

                if left_coverage < 0.001 {
                    // Edge is at or past right side of pixel — pixel is fully left-side color
                    out_row[px] = ae.color_left;
                } else if left_coverage > 0.999 {
                    // Edge is at or past left side — pixel is fully right-side color
                    out_row[px] = ae.color_right;
                } else {
                    // Blend: left_coverage of color_left, rest is color_right
                    // No background — always two-color blend summing to 1.0
                    out_row[px] = blend_linear_srgb(ae.color_left, ae.color_right, left_coverage);
                }
            }
        }
    };

    // Row-parallel on native, sequential on wasm
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::Arc;
        let edges = Arc::new(edges);
        let row_starts = Arc::new(row_starts);
        let row_indices = Arc::new(row_indices);

        let chunk_size = (out_h / rayon_like_parallelism()).max(1);
        let threads: Vec<_> = output
            .chunks_mut(out_w * chunk_size)
            .enumerate()
            .map(|(chunk_i, chunk)| {
                let start_row = chunk_i * chunk_size;
                let num_rows = chunk.len() / out_w;
                for local_row in 0..num_rows {
                    let row = start_row + local_row;
                    let row_slice = &mut chunk[local_row * out_w..(local_row + 1) * out_w];
                    rasterize_row(row, row_slice);
                }
            })
            .collect();
    }

    #[cfg(target_arch = "wasm32")]
    {
        for row in 0..out_h {
            let row_slice = &mut output[row * out_w..(row + 1) * out_w];
            rasterize_row(row, row_slice);
        }
    }

    output
}
```

- [ ] **Step 2: Add sRGB linear blend helper**

Add near the other blending code in `vectorize.rs`:

```rust
/// Blend two packed ARGB colors in linear sRGB space.
/// `t` is the weight of `c0` (0.0 = all c1, 1.0 = all c0).
fn blend_linear_srgb(c0: u32, c1: u32, t: f32) -> u32 {
    let r0 = (((c0 >> 16) & 0xFF) as f32 / 255.0).powf(2.2);
    let g0 = (((c0 >> 8) & 0xFF) as f32 / 255.0).powf(2.2);
    let b0 = ((c0 & 0xFF) as f32 / 255.0).powf(2.2);

    let r1 = (((c1 >> 16) & 0xFF) as f32 / 255.0).powf(2.2);
    let g1 = (((c1 >> 8) & 0xFF) as f32 / 255.0).powf(2.2);
    let b1 = ((c1 & 0xFF) as f32 / 255.0).powf(2.2);

    let s = 1.0 - t;
    let r = (t * r0 + s * r1).powf(1.0 / 2.2) * 255.0;
    let g = (t * g0 + s * g1).powf(1.0 / 2.2) * 255.0;
    let b = (t * b0 + s * b1).powf(1.0 / 2.2) * 255.0;

    0xFF000000
        | ((r.round() as u32).min(255) << 16)
        | ((g.round() as u32).min(255) << 8)
        | (b.round() as u32).min(255)
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo build --release 2>&1 | tail -5`
Expected: successful build

- [ ] **Step 4: Commit**

```bash
git add src/scaling/vectorize.rs
git commit -m "Implement scanline rasterizer with analytical AA and sRGB blending"
```

---

### Task 4: Visual validation and iteration

**Files:**
- No file changes — testing only

The edge flattening logic in Task 2 walks CP chains using the neighbor connectivity data. The exact traversal pattern (which CPs form spans, how control points map to B-spline evaluation) may need adjustment based on visual output. This task validates correctness.

- [ ] **Step 1: Generate test output with both rasterizers**

Use a small test image to compare:

```bash
# Existing nearest-curve rasterizer
cargo run --release --bin test_runner -- vectorize test-input.png --out /tmp/vec-nearest.png --filter vectorize --scale 8

# New scanline rasterizer
cargo run --release --bin test_runner -- vectorize test-input.png --out /tmp/vec-scanline.png --filter vectorize-scanline --scale 8
```

- [ ] **Step 2: Compare outputs visually**

Open both images side by side. Check:
- Region colors match (no swapped left/right)
- Edges are smooth (no jagged artifacts from bad subdivision)
- AA blending is visible at boundaries (not blocky, no background bleed)
- No missing regions or black holes

- [ ] **Step 3: Test with a Game Boy frame**

```bash
# Capture a frame from a real game
cargo run --release --bin test_runner -- screenshot path/to/rom.gbc --frames 300 --out /tmp/frame-nearest.png --filter vectorize --scale 6
cargo run --release --bin test_runner -- screenshot path/to/rom.gbc --frames 300 --out /tmp/frame-scanline.png --filter vectorize-scanline --scale 6
```

- [ ] **Step 4: Fix any visual issues found**

Iterate on the edge flattening and coverage computation until output is visually correct. Common issues:
- Wrong winding direction: swap `color_left`/`color_right` in `ScanEdge`
- Missing edges: chain walker skipping CPs it shouldn't
- Subdivision too coarse: increase segment count or tune adaptive threshold
- Coverage math wrong: check x-intercept clamping and trapezoidal area formula

- [ ] **Step 5: Commit fixes**

```bash
git add src/scaling/vectorize.rs
git commit -m "Fix scanline rasterizer visual issues from validation"
```

---

### Task 5: Row-parallel execution

**Files:**
- Modify: `src/scaling/vectorize.rs`

The scanline rasterizer in Task 3 has a sketch of parallelism but needs to match the existing pattern used by `rasterize()`. Each row is independent once the edge table is built.

- [ ] **Step 1: Wire up thread pool parallelism**

Look at how the existing `rasterize()` function parallelizes (it uses a `ThreadPool` pattern around line 1978-2002) and match that pattern for `rasterize_scanline()`. The key constraint: the `rasterize_row` closure captures the edge table and row index by reference, so it needs to be `Send + Sync` safe. The edge table is read-only after construction, so this is straightforward.

- [ ] **Step 2: Verify parallel output matches sequential**

```bash
# Compare parallel vs sequential (set RAYON_NUM_THREADS=1 for sequential)
RAYON_NUM_THREADS=1 cargo run --release --bin test_runner -- vectorize test-input.png --out /tmp/seq.png --filter vectorize-scanline --scale 8
cargo run --release --bin test_runner -- vectorize test-input.png --out /tmp/par.png --filter vectorize-scanline --scale 8
# Diff should be identical
```

- [ ] **Step 3: Commit**

```bash
git add src/scaling/vectorize.rs
git commit -m "Add row-parallel execution to scanline rasterizer"
```

---

### Task 6: Wire up test runner vectorize subcommand

**Files:**
- Modify: `src/test_runner/commands.rs`

- [ ] **Step 1: Check test_runner filter dispatch**

Read `src/test_runner/commands.rs` to see how `vectorize_and_save()` and `cmd_vectorize()` dispatch filters. Add `vectorize-scanline` as a recognized format/filter name if needed.

The test runner already uses `ScaleFilter::from_name()` for `--filter` parsing, so `vectorize-scanline` should work automatically via the registry. Verify this.

- [ ] **Step 2: Test the full pipeline**

```bash
cargo run --release --bin test_runner -- vectorize test-input.png --out /tmp/scanline.png --filter vectorize-scanline --scale 4
cargo run --release --bin test_runner -- vectorize test-input.png --out /tmp/scanline.svg
```

SVG export should still work (it uses `vectorize()` stages 1-5, not the rasterizer).

- [ ] **Step 3: Commit if any changes were needed**

```bash
git add src/test_runner/commands.rs
git commit -m "Wire up vectorize-scanline in test runner"
```

---

### Task 7: Update CLAUDE.md documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update filter count and module list**

In the scaling filter infrastructure section, update the filter count (35 → 36) and add `vectorize` module note about the scanline rasterizer.

- [ ] **Step 2: Add vectorize-scanline to CLI examples**

Add example usage in the test runner section:

```bash
# Scanline rasterizer (faster CPU path)
cargo run --release --bin test_runner -- vectorize input.png --out output.png --filter vectorize-scanline --scale 8
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "Document vectorize-scanline filter in CLAUDE.md"
```
