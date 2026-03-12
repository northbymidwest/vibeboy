// GPU scanline rasterizer for vectorized pixel-art paths.
// Per-pixel 2x2 supersampling with nonzero winding rule.

struct Uniforms {
    out_w: u32,
    out_h: u32,
    num_paths: u32,
    bg_color: u32,
}

struct Edge {
    x0: f32,
    y0: f32,
    x1: f32,
    inv_dy: f32,
    y_min: f32,
    y_max: f32,
    dir: i32,
    _pad: u32,
}

struct PathMeta {
    color: u32,
    edge_start: u32,
    edge_count: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> edges: array<Edge>;
@group(0) @binding(2) var<storage, read> paths: array<PathMeta>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;

fn unpack_channel(color: u32, shift: u32) -> u32 {
    return (color >> shift) & 0xFFu;
}

fn blend4(bg: u32, fg: u32, coverage: u32) -> u32 {
    let inv = 4u - coverage;
    let r = (unpack_channel(bg, 16u) * inv + unpack_channel(fg, 16u) * coverage) >> 2u;
    let g = (unpack_channel(bg, 8u) * inv + unpack_channel(fg, 8u) * coverage) >> 2u;
    let b = (unpack_channel(bg, 0u) * inv + unpack_channel(fg, 0u) * coverage) >> 2u;
    return 0xFF000000u | (r << 16u) | (g << 8u) | b;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= uniforms.out_w || py >= uniforms.out_h) {
        return;
    }

    var color = uniforms.bg_color;
    let y_top = f32(py);
    let x_left = f32(px);

    // Sample points for 2x2 supersampling
    let sy0 = y_top + 0.25;
    let sy1 = y_top + 0.75;
    let sx0 = x_left + 0.25;
    let sx1 = x_left + 0.75;

    for (var pi = 0u; pi < uniforms.num_paths; pi++) {
        let path = paths[pi];
        if (path.color == uniforms.bg_color) {
            continue;
        }

        var coverage = 0u;
        let end = path.edge_start + path.edge_count;

        // Sub-sample (0.25, 0.25)
        var w00: i32 = 0;
        // Sub-sample (0.75, 0.25)
        var w10: i32 = 0;
        // Sub-sample (0.25, 0.75)
        var w01: i32 = 0;
        // Sub-sample (0.75, 0.75)
        var w11: i32 = 0;

        for (var ei = path.edge_start; ei < end; ei++) {
            let e = edges[ei];

            // Row 0 (sy = y + 0.25)
            if (sy0 >= e.y_min && sy0 < e.y_max) {
                let t = (sy0 - e.y0) * e.inv_dy;
                let ix = e.x0 + t * (e.x1 - e.x0);
                if (sx0 < ix) { w00 += e.dir; }
                if (sx1 < ix) { w10 += e.dir; }
            }

            // Row 1 (sy = y + 0.75)
            if (sy1 >= e.y_min && sy1 < e.y_max) {
                let t = (sy1 - e.y0) * e.inv_dy;
                let ix = e.x0 + t * (e.x1 - e.x0);
                if (sx0 < ix) { w01 += e.dir; }
                if (sx1 < ix) { w11 += e.dir; }
            }
        }

        if (w00 != 0) { coverage += 1u; }
        if (w10 != 0) { coverage += 1u; }
        if (w01 != 0) { coverage += 1u; }
        if (w11 != 0) { coverage += 1u; }

        if (coverage == 4u) {
            color = path.color;
        } else if (coverage > 0u) {
            color = blend4(color, path.color, coverage);
        }
    }

    output[py * uniforms.out_w + px] = color;
}
