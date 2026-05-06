use std::fs;
use std::path::Path;

use vibeboy::model::GbModel;
use vibeboy::scaling;
use crate::test_model::{detect_model_with_rom, resolve_boot_rom};
use crate::util::{make_emu, parse_keys, GB_FB_WIDTH, GB_FB_HEIGHT};

fn vectorize_to_svg(pixels: &[u32], width: usize, height: usize) -> String {
    let data = vibeboy::scaling::vectorize::vectorize(pixels, width, height);
    crate::gpu_svg::render_svg(&data, pixels)
}

fn save_pixels_png(pixels: &[u32], w: usize, h: usize, out: &str) {
    let mut rgb = Vec::with_capacity(w * h * 3);
    for &pixel in pixels {
        rgb.push(((pixel >> 16) & 0xFF) as u8);
        rgb.push(((pixel >> 8) & 0xFF) as u8);
        rgb.push((pixel & 0xFF) as u8);
    }
    image::save_buffer(out, &rgb, w as u32, h as u32, image::ColorType::Rgb8)
        .expect("Failed to write PNG");
}

fn save_pixels(pixels: &[u32], w: usize, h: usize, out: &str, format: &str, frames: u32) {
    if format == "svg" || out.ends_with(".svg") {
        let svg = vectorize_to_svg(pixels, w, h);
        fs::write(out, &svg).expect("Failed to write SVG");
        eprintln!("Wrote {} (frame {}, {} bytes SVG)", out, frames, svg.len());
    } else {
        save_pixels_png(pixels, w, h, out);
        eprintln!("Wrote {} (frame {}, {}x{})", out, frames, w, h);
    }
}

/// Try a GPU rasterization function, falling back to a CPU function on failure.
/// Returns the rasterized result as (pixels, width, height).
///
/// `filter_name` is included in diagnostic messages for clarity.
/// `gpu_fn` should return `Some((pixels, width, height))` on success, `None` on failure.
/// `cpu_fn` provides the CPU fallback and is called when GPU is unavailable or fails.
fn gpu_with_cpu_fallback(
    filter_name: &str,
    use_gpu: bool,
    gpu_fn: impl FnOnce() -> Option<(Vec<u32>, u32, u32)>,
    cpu_fn: impl FnOnce() -> (Vec<u32>, usize, usize),
) -> (Vec<u32>, usize, usize) {
    if use_gpu {
        #[cfg(feature = "sdl3-gpu-shaders")]
        {
            if let Some((pixels, w, h)) = gpu_fn() {
                return (pixels, w as usize, h as usize);
            }
            eprintln!(
                "GPU '{}' failed, falling back to CPU rasterization",
                filter_name
            );
        }
        #[cfg(not(feature = "sdl3-gpu-shaders"))]
        {
            let _ = gpu_fn; // suppress unused variable warning
            eprintln!(
                "GPU shaders not enabled for '{}', using CPU rasterization",
                filter_name
            );
        }
    }
    cpu_fn()
}

/// Rasterize pixels using the specified vectorize format and save to a file.
fn vectorize_and_save(
    pixels: &[u32], width: usize, height: usize,
    out: &str, format: &str, scale: usize, use_gpu: bool,
) {
    if out.ends_with(".svg") {
        let svg = vectorize_to_svg(pixels, width, height);
        fs::write(out, &svg).expect("Failed to write SVG");
        eprintln!(
            "Vectorized {}x{} image -> {} ({} bytes)",
            width, height, out, svg.len()
        );
        return;
    }

    let scale_f = scale as f32;
    let cpu_fallback = || {
        let ow = (width as f32 * scale_f).round() as usize;
        let oh = (height as f32 * scale_f).round() as usize;
        let r = vibeboy::scaling::vectorize::scale(pixels, width, height, scale_f);
        (r, ow, oh)
    };
    let (raster_pixels, out_w, out_h) = gpu_with_cpu_fallback(
        "vectorize",
        use_gpu,
        || {
            #[cfg(feature = "sdl3-gpu-shaders")]
            {
                vibeboy::scaling::sdl::gpu_full_pipeline_screenshot(
                    pixels, width, height, scale,
                )
            }
            #[cfg(not(feature = "sdl3-gpu-shaders"))]
            {
                None
            }
        },
        cpu_fallback,
    );
    save_pixels_png(&raster_pixels, out_w, out_h, out);
    let backend = if use_gpu { "gpu" } else { "cpu" };
    eprintln!(
        "Vectorized+rasterized {}x{} image -> {} ({}x{} at {}x, backend={}, format={})",
        width, height, out, out_w, out_h, scale, backend, format
    );
}

/// Apply a scaling filter via GPU, returning the scaled pixels on success.
/// Returns `None` if GPU is not available or the operation fails.
fn try_gpu_filter(
    raw_fb: &[u32],
    filter_name: &str,
    sf: scaling::ScaleFilter,
    scale: usize,
) -> Option<(Vec<u32>, usize, usize)> {
    #[cfg(feature = "sdl3-gpu-shaders")]
    {
        if filter_name == "vectorize" {
            if let Some((pix, w, h)) = scaling::sdl::gpu_full_pipeline_screenshot(
                raw_fb, GB_FB_WIDTH, GB_FB_HEIGHT, scale,
            ) {
                return Some((pix, w as usize, h as usize));
            }
            eprintln!(
                "GPU full pipeline screenshot failed for filter '{}', falling back to CPU",
                filter_name
            );
        } else if let Some((s, w, h)) = scaling::sdl::gpu_screenshot(
            raw_fb, GB_FB_WIDTH as u32, GB_FB_HEIGHT as u32, sf,
        ) {
            return Some((s, w as usize, h as usize));
        } else {
            eprintln!(
                "GPU screenshot failed for filter '{}', falling back to CPU",
                filter_name
            );
        }
    }
    #[cfg(not(feature = "sdl3-gpu-shaders"))]
    {
        let _ = (sf, scale);
        eprintln!(
            "GPU support not compiled in for filter '{}' (enable sdl3-gpu-shaders feature)",
            filter_name
        );
    }
    None
}

pub fn cmd_screenshot(
    path: &Path,
    force_model: Option<GbModel>,
    boot: bool,
    bootrom: Option<&Path>,
    frames: u32,
    out: &str,
    format: &str,
    scale: usize,
    keys: &str,
    filter: Option<&str>,
    use_gpu: bool,
) {
    let rom = fs::read(path).expect("Failed to read ROM");
    let model = force_model.unwrap_or_else(|| detect_model_with_rom(path, Some(&rom)));
    let br = resolve_boot_rom(boot, bootrom, model);
    let mut emu = make_emu(rom, br, model);
    let key_events = parse_keys(keys);
    let mut key_idx = 0;
    for f in 0..frames {
        while key_idx < key_events.len() && key_events[key_idx].0 == f {
            emu.set_button(key_events[key_idx].1, key_events[key_idx].2);
            key_idx += 1;
        }
        emu.step_frame();
    }
    let raw_fb = emu.frame_buffer();

    // SVG export from raw framebuffer (bypass scaling)
    if out.ends_with(".svg") {
        let svg = vectorize_to_svg(raw_fb, GB_FB_WIDTH, GB_FB_HEIGHT);
        fs::write(out, &svg).expect("Failed to write SVG");
        eprintln!("Wrote {} (frame {}, {} bytes SVG)", out, frames, svg.len());
        return;
    }

    // Apply scaling filter if requested
    let scaled_buf;
    let (fb, fb_w, fb_h) = if let Some(f) = filter {
        let sf = scaling::ScaleFilter::from_name(f).unwrap_or_else(|| {
            eprintln!("Unknown filter '{}', using nearest", f);
            scaling::ScaleFilter::Nearest
        });

        // Try GPU path if requested
        if use_gpu {
            if let Some((pixels, w, h)) =
                try_gpu_filter(raw_fb, f, sf, scale)
            {
                scaled_buf = pixels;
                return save_pixels(&scaled_buf, w, h, out, format, frames);
            }
        }

        // CPU path
        let disp_w = GB_FB_WIDTH * scale;
        let disp_h = GB_FB_HEIGHT * scale;
        let (s, w, h) = scaling::cpu_scale(sf, raw_fb, GB_FB_WIDTH, GB_FB_HEIGHT, disp_w, disp_h)
            .unwrap_or_else(|| (raw_fb.to_vec(), GB_FB_WIDTH as u32, GB_FB_HEIGHT as u32));
        scaled_buf = s;
        (scaled_buf.as_slice(), w as usize, h as usize)
    } else {
        (raw_fb, GB_FB_WIDTH, GB_FB_HEIGHT)
    };

    if matches!(format, "raster" | "vectorize") {
        vectorize_and_save(fb, GB_FB_WIDTH, GB_FB_HEIGHT, out, format, scale, use_gpu);
    } else {
        save_pixels(fb, fb_w, fb_h, out, format, frames);
    }
}

pub fn cmd_vectorize(input: &Path, out: &str, scale: usize, gpu: bool, dump_cps: Option<&str>) {
    let img = image::open(input).unwrap_or_else(|e| {
        eprintln!("Failed to open image '{}': {}", input.display(), e);
        std::process::exit(1);
    });
    let rgba = img.to_rgba8();
    let width = rgba.width() as usize;
    let height = rgba.height() as usize;

    // Convert RGBA to ARGB u32
    let pixels: Vec<u32> = rgba
        .pixels()
        .map(|p| {
            let [r, g, b, _a] = p.0;
            0xFF000000 | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
        })
        .collect();

    if let Some(path) = dump_cps {
        dump_cp_data(&pixels, width, height, path);
    }

    vectorize_and_save(&pixels, width, height, out, "vectorize", scale, gpu);

    if let Ok(spec) = std::env::var("VIBEBOY_OVERLAY_CURVES") {
        let nn = spec.parse::<usize>().unwrap_or(8);
        write_curve_overlay(&pixels, width, height, scale, nn, out);
    }
    // VIBEBOY_OVERLAY_FOCUS=cx,cy,radius,mag
    //   (cx,cy) = output-pixel center in the rendered (scale-x) image
    //   radius   = output-pixel half-width (radius=1 → 3x3 region)
    //   mag      = display pixels per output pixel
    if let Ok(spec) = std::env::var("VIBEBOY_OVERLAY_FOCUS") {
        let parts: Vec<&str> = spec.split(',').collect();
        if parts.len() == 4 {
            let cx: i32 = parts[0].parse().unwrap();
            let cy: i32 = parts[1].parse().unwrap();
            let radius: i32 = parts[2].parse().unwrap();
            let mag: usize = parts[3].parse().unwrap();
            write_focus_overlay(&pixels, width, height, scale, cx, cy, radius, mag, out);
        }
    }
}

use vibeboy::scaling::vectorize::{IS_CORNER, IS_CROSSING, IS_ENDPOINT, IS_TJUNCTION};

/// Decode a flag bitfield to a JSON array of named bits. Returns "[]" if no
/// known bits are set. Slot bits (1=junction, 2=interior) are emitted as
/// "JUNCTION" or "INTERIOR" so the caller can read intent without knowing
/// the encoding.
fn flag_names(flag: u32) -> String {
    let mut names: Vec<&'static str> = Vec::new();
    match flag & 3 {
        1 => names.push("JUNCTION"),
        2 => names.push("INTERIOR"),
        _ => {}
    }
    if flag & IS_CORNER != 0 { names.push("CORNER"); }
    if flag & IS_TJUNCTION != 0 { names.push("TJUNCTION"); }
    if flag & IS_CROSSING != 0 { names.push("CROSSING"); }
    if flag & IS_ENDPOINT != 0 { names.push("ENDPOINT"); }
    let mut s = String::from("[");
    for (i, n) in names.iter().enumerate() {
        if i > 0 { s.push_str(", "); }
        s.push('"');
        s.push_str(n);
        s.push('"');
    }
    s.push(']');
    s
}

/// Format an ARGB u32 as a `"#RRGGBB"` hex string.
fn argb_hex(c: u32) -> String {
    format!("\"#{:02X}{:02X}{:02X}\"", (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF)
}

/// Format an edge's (left, right) colors. The two pixels straddle the edge in
/// the direction the rasterizer uses for `resolve_color` / `build_aa_line`.
/// Returns `null` if `dir < 0` (inactive edge).
fn edge_colors_json(pixels: &[u32], w: usize, h: usize, icx: i32, icy: i32, dir: i32) -> String {
    if dir < 0 { return String::from("null"); }
    let (l, r) = vibeboy::scaling::vectorize::get_edge_colors(pixels, w, h, icx, icy, dir);
    format!("[{}, {}]", argb_hex(l), argb_hex(r))
}

/// Dump every control point's data to a JSON file (or stdout when path == "-").
/// Emits one record per CP with: index, position, original position, neighbors
/// (prev_ci, next_ci, prev_dir, next_dir), raw flag bits + decoded names,
/// crossing_t, and the (left, right) source-pixel colors straddling the prev
/// and next edges (the inputs the rasterizer feeds into resolve_color / AA).
/// Inactive CPs (flag == 0) are included so the output's index matches
/// `VectorizeData.positions[ci*2..]` 1:1.
fn dump_cp_data(pixels: &[u32], w: usize, h: usize, path: &str) {
    use std::io::Write;
    let data = vibeboy::scaling::vectorize::vectorize(pixels, w, h);
    let num_cps = data.flags.len();
    let corners_w = data.img_w + 1;

    let mut out = String::new();
    out.push_str(&format!(
        "{{\n  \"img_w\": {}, \"img_h\": {}, \"num_cps\": {},\n  \"control_points\": [\n",
        data.img_w, data.img_h, num_cps,
    ));
    for ci in 0..num_cps {
        let px = data.positions[ci * 2];
        let py = data.positions[ci * 2 + 1];
        let ox = data.orig_positions[ci * 2];
        let oy = data.orig_positions[ci * 2 + 1];
        let prev_ci = data.neighbors[ci * 4];
        let next_ci = data.neighbors[ci * 4 + 1];
        let prev_dir = data.neighbors[ci * 4 + 2];
        let next_dir = data.neighbors[ci * 4 + 3];
        let flag = data.flags[ci];
        let crossing_t = data.crossing_t[ci];
        // Corner coordinates for this CP (matches `read_resolve_data` in
        // both shader and CPU mirror: every 2 CPs share a corner slot).
        let icx = (ci / 2) % corners_w;
        let icy = (ci / 2) / corners_w;
        let prev_edge = edge_colors_json(pixels, data.img_w, data.img_h, icx as i32, icy as i32, prev_dir);
        let next_edge = edge_colors_json(pixels, data.img_w, data.img_h, icx as i32, icy as i32, next_dir);
        let comma = if ci + 1 == num_cps { "" } else { "," };
        out.push_str(&format!(
            "    {{\"ci\": {}, \"pos\": [{:.6}, {:.6}], \"orig_pos\": [{:.6}, {:.6}], \
             \"prev_ci\": {}, \"next_ci\": {}, \"prev_dir\": {}, \"next_dir\": {}, \
             \"flag\": {}, \"flag_names\": {}, \"crossing_t\": {:.6}, \
             \"icx\": {}, \"icy\": {}, \"prev_edge_colors\": {}, \"next_edge_colors\": {}}}{}\n",
            ci, px, py, ox, oy, prev_ci, next_ci, prev_dir, next_dir,
            flag, flag_names(flag), crossing_t, icx, icy, prev_edge, next_edge, comma,
        ));
    }
    out.push_str("  ]\n}\n");

    if path == "-" {
        print!("{}", out);
    } else {
        let mut f = std::fs::File::create(path).unwrap_or_else(|e| {
            eprintln!("Failed to create CP dump file '{}': {}", path, e);
            std::process::exit(1);
        });
        f.write_all(out.as_bytes()).unwrap_or_else(|e| {
            eprintln!("Failed to write CP dump: {}", e);
            std::process::exit(1);
        });
        let active = data.flags.iter().filter(|&&f| f != 0).count();
        eprintln!("Dumped {} CPs ({} active) to {}", num_cps, active, path);
    }
}

/// Decide whether a CP should render its span (mirrors `rasterize`'s filter:
/// skip CPs whose chain has no interior CP unless this is a 2-CP chain, and
/// in that case let only the lower-indexed endpoint own the span). Returns
/// (prev_pos, ghost-adjusted pp, ghost-adjusted np) when the CP renders.
fn cp_render_geometry(
    data: &vibeboy::scaling::vectorize::VectorizeData,
    ci: usize,
) -> Option<((f32, f32), (f32, f32), (f32, f32), (f32, f32), (f32, f32))> {
    let flag = data.flags[ci];
    if flag == 0 { return None; }
    let prev_ci = data.neighbors[ci * 4];
    let next_ci = data.neighbors[ci * 4 + 1];
    if prev_ci < 0 && next_ci < 0 { return None; }
    let i_am_endpoint = prev_ci < 0 || next_ci < 0;
    if i_am_endpoint {
        let other = if prev_ci < 0 { next_ci } else { prev_ci };
        if (data.flags[other as usize] & IS_ENDPOINT) == 0 { return None; }
        if (ci as i32) > other { return None; }
    }
    let cp = (data.positions[ci * 2], data.positions[ci * 2 + 1]);
    let prev_pos = if prev_ci >= 0 {
        (data.positions[prev_ci as usize * 2], data.positions[prev_ci as usize * 2 + 1])
    } else { cp };
    let next_pos = if next_ci >= 0 {
        (data.positions[next_ci as usize * 2], data.positions[next_ci as usize * 2 + 1])
    } else { cp };
    let prev_is_end = prev_ci >= 0 && (data.flags[prev_ci as usize] & IS_ENDPOINT) != 0;
    let next_is_end = next_ci >= 0 && (data.flags[next_ci as usize] & IS_ENDPOINT) != 0;
    let pp = if prev_is_end {
        (2.0 * prev_pos.0 - cp.0, 2.0 * prev_pos.1 - cp.1)
    } else { prev_pos };
    let np = if next_is_end {
        (2.0 * next_pos.0 - cp.0, 2.0 * next_pos.1 - cp.1)
    } else { next_pos };
    Some((cp, pp, np, prev_pos, next_pos))
}

fn write_curve_overlay(pixels: &[u32], w: usize, h: usize, scale: usize, nn: usize, base_out: &str) {
    let scaled_w = w * scale;
    let scaled_h = h * scale;
    let final_w = scaled_w * nn;
    let final_h = scaled_h * nn;

    let scaled_rgba = std::fs::read(base_out).ok()
        .and_then(|_| image::open(base_out).ok())
        .map(|i| i.to_rgba8())
        .expect("base render not found");

    // NN-upscale to final
    let mut buf = vec![0u8; final_w * final_h * 4];
    for fy in 0..final_h {
        let sy = fy / nn;
        for fx in 0..final_w {
            let sx = fx / nn;
            let p = scaled_rgba.get_pixel(sx as u32, sy as u32);
            let i = (fy * final_w + fx) * 4;
            buf[i] = p.0[0]; buf[i+1] = p.0[1]; buf[i+2] = p.0[2]; buf[i+3] = 255;
        }
    }

    // Get CPs
    let data = vibeboy::scaling::vectorize::vectorize(pixels, w, h);
    let num_cps = data.flags.len();

    // pixels-per-source-unit on the final image
    let pps = (scale * nn) as f32;

    // Plot a pixel
    let plot = |buf: &mut [u8], x: i32, y: i32, c: [u8;3]| {
        if x < 0 || y < 0 || x >= final_w as i32 || y >= final_h as i32 { return; }
        let i = (y as usize * final_w + x as usize) * 4;
        // Blend with existing pixel for visibility
        buf[i] = ((c[0] as u16 + buf[i] as u16) / 2) as u8;
        buf[i+1] = ((c[1] as u16 + buf[i+1] as u16) / 2) as u8;
        buf[i+2] = ((c[2] as u16 + buf[i+2] as u16) / 2) as u8;
    };

    // Draw a B-spline span by sampling
    let draw_span = |buf: &mut [u8], pp: (f32,f32), cp: (f32,f32), np: (f32,f32), color: [u8;3]| {
        const N: usize = 200;
        let mut last: Option<(f32,f32)> = None;
        for i in 0..=N {
            let t = i as f32 / N as f32;
            let u = 1.0 - t;
            let bx = 0.5*u*u*pp.0 + (u*t + 0.5)*cp.0 + 0.5*t*t*np.0;
            let by = 0.5*u*u*pp.1 + (u*t + 0.5)*cp.1 + 0.5*t*t*np.1;
            let px = (bx * pps).round() as i32;
            let py = (by * pps).round() as i32;
            if let Some((lx, ly)) = last {
                // Bresenham from (lx,ly) to (px,py)
                let lxi = lx.round() as i32;
                let lyi = ly.round() as i32;
                let dx = (px - lxi).abs();
                let dy = -(py - lyi).abs();
                let sx = if lxi < px { 1 } else { -1 };
                let sy = if lyi < py { 1 } else { -1 };
                let mut err = dx + dy;
                let mut x = lxi; let mut y = lyi;
                loop {
                    plot(buf, x, y, color);
                    if x == px && y == py { break; }
                    let e2 = 2 * err;
                    if e2 >= dy { err += dy; x += sx; }
                    if e2 <= dx { err += dx; y += sy; }
                }
            }
            last = Some((px as f32, py as f32));
        }
    };

    for ci in 0..num_cps {
        let Some((cp, pp, np, _, _)) = cp_render_geometry(&data, ci) else { continue; };
        let prev_ci = data.neighbors[ci * 4];
        let next_ci = data.neighbors[ci * 4 + 1];
        let stem = (prev_ci < 0 || next_ci < 0)
            || (prev_ci >= 0 && (data.flags[prev_ci as usize] & IS_ENDPOINT) != 0)
            || (next_ci >= 0 && (data.flags[next_ci as usize] & IS_ENDPOINT) != 0);
        let color = if stem { [255, 64, 255] } else { [0, 200, 255] };
        draw_span(&mut buf, pp, cp, np, color);
    }

    // Mark CPs with small dots
    for ci in 0..num_cps {
        let flag = data.flags[ci];
        if flag == 0 { continue; }
        let cp = (data.positions[ci*2], data.positions[ci*2+1]);
        let px = (cp.0 * pps).round() as i32;
        let py = (cp.1 * pps).round() as i32;
        let color = if (flag & IS_ENDPOINT) != 0 { [255, 255, 0] } else { [0, 255, 0] };
        for dy in -2..=2 { for dx in -2..=2 {
            plot(&mut buf, px+dx, py+dy, color);
        }}
    }

    let out_path = base_out.replace(".png", "_overlay.png");
    image::save_buffer(&out_path, &buf, final_w as u32, final_h as u32, image::ColorType::Rgba8)
        .expect("save overlay");
    eprintln!("Overlay -> {} ({}x{})", out_path, final_w, final_h);
}

/// Focused overlay: crop a (2*radius+1)x(2*radius+1) region of output
/// pixels around (cx, cy), upscale by `mag` per output pixel, and draw
/// curves and CPs at full polynomial precision.
fn write_focus_overlay(
    pixels: &[u32], w: usize, h: usize, scale: usize,
    cx: i32, cy: i32, radius: i32, mag: usize,
    base_out: &str,
) {
    let scaled_rgba = image::open(base_out).expect("base render not found").to_rgba8();
    let scaled_w = scaled_rgba.width() as i32;
    let scaled_h = scaled_rgba.height() as i32;

    let span = 2 * radius + 1;
    let final_size = span as usize * mag;

    // Source coordinate at the top-left of the crop:
    // pixel (cx-radius, cy-radius) covers source (cx-radius, cy-radius)/scale
    // to (cx-radius+1, cy-radius+1)/scale. Top-left corner of the display:
    let x0_pix = cx - radius;
    let y0_pix = cy - radius;
    let src_x0 = x0_pix as f32 / scale as f32;
    let src_y0 = y0_pix as f32 / scale as f32;
    // Display pixels per source unit:
    let pps = (mag * scale) as f32;

    // NN-upscale the cropped region into the buffer
    let mut buf = vec![0u8; final_size * final_size * 4];
    for fy in 0..final_size {
        let py = y0_pix + (fy as i32 / mag as i32);
        for fx in 0..final_size {
            let px = x0_pix + (fx as i32 / mag as i32);
            let i = (fy * final_size + fx) * 4;
            if px >= 0 && py >= 0 && px < scaled_w && py < scaled_h {
                let p = scaled_rgba.get_pixel(px as u32, py as u32);
                buf[i] = p.0[0]; buf[i+1] = p.0[1]; buf[i+2] = p.0[2]; buf[i+3] = 255;
            } else {
                buf[i+3] = 255;
            }
        }
    }

    let data = vibeboy::scaling::vectorize::vectorize(pixels, w, h);
    let num_cps = data.flags.len();

    let to_disp = |sx: f32, sy: f32| -> (f32, f32) {
        ((sx - src_x0) * pps, (sy - src_y0) * pps)
    };

    let plot = |buf: &mut [u8], x: i32, y: i32, c: [u8;3], blend: f32| {
        if x < 0 || y < 0 || x >= final_size as i32 || y >= final_size as i32 { return; }
        let i = (y as usize * final_size + x as usize) * 4;
        let a = blend.clamp(0.0, 1.0);
        buf[i]   = ((c[0] as f32 * a) + (buf[i] as f32 * (1.0 - a))) as u8;
        buf[i+1] = ((c[1] as f32 * a) + (buf[i+1] as f32 * (1.0 - a))) as u8;
        buf[i+2] = ((c[2] as f32 * a) + (buf[i+2] as f32 * (1.0 - a))) as u8;
    };

    // Draw a B-spline span as a thick antialiased line
    let draw_span = |buf: &mut [u8], pp: (f32,f32), cp: (f32,f32), np: (f32,f32), color: [u8;3]| {
        const N: usize = 2000;
        for i in 0..=N {
            let t = i as f32 / N as f32;
            let u = 1.0 - t;
            let bx = 0.5*u*u*pp.0 + (u*t + 0.5)*cp.0 + 0.5*t*t*np.0;
            let by = 0.5*u*u*pp.1 + (u*t + 0.5)*cp.1 + 0.5*t*t*np.1;
            let (dx, dy) = to_disp(bx, by);
            // Draw a 3-pixel wide centered dot
            for oy in -2..=2 { for ox in -2..=2 {
                let dist = ((ox*ox + oy*oy) as f32).sqrt();
                if dist <= 2.5 {
                    let alpha = (1.0 - dist / 2.5).max(0.0);
                    plot(buf, dx as i32 + ox, dy as i32 + oy, color, alpha * 0.85);
                }
            }}
        }
    };

    // Draw all curves whose bbox enters the visible area
    let view_min = (src_x0 - 0.5, src_y0 - 0.5);
    let view_max = (src_x0 + span as f32 / scale as f32 + 0.5,
                    src_y0 + span as f32 / scale as f32 + 0.5);
    for ci in 0..num_cps {
        let Some((cp, pp, np, prev_pos, next_pos)) = cp_render_geometry(&data, ci) else { continue; };
        let prev_ci = data.neighbors[ci * 4];
        let next_ci = data.neighbors[ci * 4 + 1];
        let stem = (prev_ci < 0 || next_ci < 0)
            || (prev_ci >= 0 && (data.flags[prev_ci as usize] & IS_ENDPOINT) != 0)
            || (next_ci >= 0 && (data.flags[next_ci as usize] & IS_ENDPOINT) != 0);

        // Cull if entire bbox is outside view
        let bmin = (prev_pos.0.min(cp.0).min(next_pos.0), prev_pos.1.min(cp.1).min(next_pos.1));
        let bmax = (prev_pos.0.max(cp.0).max(next_pos.0), prev_pos.1.max(cp.1).max(next_pos.1));
        if bmax.0 < view_min.0 || bmin.0 > view_max.0 || bmax.1 < view_min.1 || bmin.1 > view_max.1 {
            continue;
        }

        let color = if stem { [255, 64, 255] } else { [0, 200, 255] };
        draw_span(&mut buf, pp, cp, np, color);
    }

    // Draw CP positions (after curves so they're on top)
    for ci in 0..num_cps {
        let flag = data.flags[ci];
        if flag == 0 { continue; }
        let cp = (data.positions[ci*2], data.positions[ci*2+1]);
        if cp.0 < view_min.0 || cp.0 > view_max.0 || cp.1 < view_min.1 || cp.1 > view_max.1 {
            continue;
        }
        let (dx, dy) = to_disp(cp.0, cp.1);
        // Stems = yellow filled, regular = green filled. Larger radius for stems.
        let (color, r2) = if (flag & IS_ENDPOINT) != 0 { ([255u8, 255, 0], 8) } else { ([0u8, 255, 0], 6) };
        for oy in -10i32..=10 { for ox in -10i32..=10 {
            let d2 = ox*ox + oy*oy;
            if d2 <= r2*r2 {
                plot(&mut buf, dx as i32 + ox, dy as i32 + oy, color, 1.0);
            } else if d2 <= (r2+1)*(r2+1) {
                plot(&mut buf, dx as i32 + ox, dy as i32 + oy, [0, 0, 0], 1.0);
            }
        }}
    }

    // Mark each visible output pixel center with a white +
    for px in (cx - radius)..=(cx + radius) {
        for py in (cy - radius)..=(cy + radius) {
            let center_src_x = (px as f32 + 0.5) / scale as f32;
            let center_src_y = (py as f32 + 0.5) / scale as f32;
            let (dx, dy) = to_disp(center_src_x, center_src_y);
            let dxi = dx as i32; let dyi = dy as i32;
            // White cross 16 px arms
            for k in -16i32..=16 {
                plot(&mut buf, dxi + k, dyi, [255, 255, 255], 1.0);
                plot(&mut buf, dxi, dyi + k, [255, 255, 255], 1.0);
            }
            // Highlight the artifact pixel center with red
            if px == cx && py == cy {
                for k in -20i32..=20 {
                    plot(&mut buf, dxi + k, dyi - 1, [255, 0, 0], 1.0);
                    plot(&mut buf, dxi + k, dyi + 1, [255, 0, 0], 1.0);
                    plot(&mut buf, dxi - 1, dyi + k, [255, 0, 0], 1.0);
                    plot(&mut buf, dxi + 1, dyi + k, [255, 0, 0], 1.0);
                }
            }
        }
    }

    let out_path = base_out.replace(".png", "_focus.png");
    image::save_buffer(&out_path, &buf, final_size as u32, final_size as u32, image::ColorType::Rgba8)
        .expect("save focus");
    eprintln!("Focus overlay -> {} ({}x{}) at pps={}", out_path, final_size, final_size, pps);
}

pub fn cmd_audio_dump(
    rom_path: &Path,
    model: Option<GbModel>,
    frames: u32,
    out: &str,
    sample_rate: u32,
) {
    let rom = fs::read(rom_path).unwrap_or_else(|e| {
        eprintln!("Failed to read ROM: {e}");
        std::process::exit(1);
    });
    let resolved_model = model.unwrap_or_else(|| detect_model_with_rom(rom_path, Some(&rom)));
    let br = resolve_boot_rom(false, None, resolved_model);
    let mut emu = make_emu(rom, br, resolved_model);

    let mut all_samples: Vec<f32> = Vec::new();
    for _ in 0..frames {
        emu.step_frame();
        all_samples.extend_from_slice(&emu.drain_audio_samples());
    }

    // Write WAV: 32-bit float, 2ch, given sample rate
    let data_len = all_samples.len() * 4;
    let file_len = 36 + data_len;
    let mut wav: Vec<u8> = Vec::with_capacity(file_len + 8);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(file_len as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    wav.extend_from_slice(&3u16.to_le_bytes()); // format = IEEE float
    wav.extend_from_slice(&2u16.to_le_bytes()); // channels
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * 2 * 4;
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&8u16.to_le_bytes()); // block align
    wav.extend_from_slice(&32u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data_len as u32).to_le_bytes());
    for &s in &all_samples {
        wav.extend_from_slice(&s.to_le_bytes());
    }
    fs::write(out, &wav).expect("Failed to write WAV");
    eprintln!(
        "Wrote {} ({} frames, {} samples, {:.1}s at {}Hz)",
        out,
        frames,
        all_samples.len() / 2,
        all_samples.len() as f64 / (2.0 * sample_rate as f64),
        sample_rate,
    );
}
