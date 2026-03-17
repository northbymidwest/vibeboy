use std::fs;
use std::path::Path;

use crate::model::GbModel;
use crate::scaling;
use crate::vectorize;
use crate::test_runner::test_model::{detect_model_with_rom, resolve_boot_rom};
use crate::test_runner::util::{make_emu, parse_keys, GB_FB_WIDTH, GB_FB_HEIGHT};

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
        let svg = vectorize::vectorize_to_svg(pixels, w, h);
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
/// Handles raster, diffusion, and spline-diffusion formats with optional GPU.
fn vectorize_and_save(
    pixels: &[u32], width: usize, height: usize,
    out: &str, format: &str, scale: usize, use_gpu: bool,
) {
    if out.ends_with(".svg") {
        let svg = vectorize::vectorize_to_svg(pixels, width, height);
        fs::write(out, &svg).expect("Failed to write SVG");
        eprintln!(
            "Vectorized {}x{} image -> {} ({} bytes)",
            width, height, out, svg.len()
        );
        return;
    }

    let (raster_pixels, out_w, out_h) = match format {
        "spline-diffusion" => {
            let r = spline_diffusion_with_fallback(pixels, width, height, scale, use_gpu);
            (r, width * scale, height * scale)
        }
        "diffusion" => {
            vectorize::rasterize::rasterize_diffusion(pixels, width, height, scale)
        }
        "edge" => {
            gpu_with_cpu_fallback(
                "edge",
                use_gpu,
                || {
                    #[cfg(feature = "sdl3-gpu-shaders")]
                    {
                        crate::scaling::gpu::gpu_vectorize_shared_screenshot(
                            pixels, width, height, scale,
                        )
                    }
                    #[cfg(not(feature = "sdl3-gpu-shaders"))]
                    {
                        None
                    }
                },
                || vectorize::vectorize_to_raster_shared(pixels, width, height, scale),
            )
        }
        "cpu-dump" => {
            // Dump CPU control points for visualization
            vectorize::contour::dump_cpu_control_points(pixels, width, height);
            return;
        }
        "gpu-full" => {
            gpu_with_cpu_fallback(
                "gpu-full",
                use_gpu,
                || {
                    #[cfg(feature = "sdl3-gpu-shaders")]
                    {
                        crate::scaling::gpu::gpu_full_pipeline_screenshot(
                            pixels, width, height, scale,
                        )
                    }
                    #[cfg(not(feature = "sdl3-gpu-shaders"))]
                    {
                        None
                    }
                },
                || vectorize::vectorize_to_raster_shared(pixels, width, height, scale),
            )
        }
        _ => {
            // "raster" or default
            vectorize::vectorize_to_raster(pixels, width, height, scale)
        }
    };
    save_pixels_png(&raster_pixels, out_w, out_h, out);
    eprintln!(
        "Vectorized+rasterized {}x{} image -> {} ({}x{} at {}x, format={})",
        width, height, out, out_w, out_h, scale, format
    );
}

/// Spline-diffusion rasterization with optional GPU acceleration and CPU fallback.
fn spline_diffusion_with_fallback(
    pixels: &[u32], width: usize, height: usize, scale: usize, use_gpu: bool,
) -> Vec<u32> {
    #[cfg(feature = "sdl3-gpu-shaders")]
    if use_gpu {
        if let Some((px, _, _)) = scaling::gpu::gpu_spline_diffusion_screenshot(pixels, width, height, scale) {
            return px;
        }
        eprintln!("GPU 'spline-diffusion' failed, falling back to CPU rasterization");
    }
    #[cfg(not(feature = "sdl3-gpu-shaders"))]
    if use_gpu {
        eprintln!("GPU shaders not enabled for 'spline-diffusion', using CPU rasterization");
    }
    vectorize::contour::YUV_VISIBLE_EDGES.store(true, std::sync::atomic::Ordering::Relaxed);
    let (paths, bg_color) = vectorize::vectorize_core(pixels, width, height);
    vectorize::contour::YUV_VISIBLE_EDGES.store(false, std::sync::atomic::Ordering::Relaxed);
    let (r, _, _) = vectorize::rasterize::rasterize_spline_diffusion(
        &paths, pixels, width, height, bg_color, scale,
    );
    r
}

/// Apply a scaling filter via GPU, returning the scaled pixels on success.
/// Returns `None` if GPU is not available or the operation fails.
fn try_gpu_filter(
    raw_fb: &[u32],
    filter_name: &str,
    sf: scaling::ScaleFilter,
    scale: usize,
    is_vectorize: bool,
    is_adaptive: bool,
) -> Option<(Vec<u32>, usize, usize)> {
    #[cfg(feature = "sdl3-gpu-shaders")]
    {
        if is_vectorize {
            let s = scale as f64;
            if let Some((pix, w, h)) = scaling::gpu::gpu_vectorize_screenshot(
                raw_fb, GB_FB_WIDTH, GB_FB_HEIGHT, s, is_adaptive,
            ) {
                return Some((pix, w as usize, h as usize));
            }
            eprintln!(
                "GPU vectorize screenshot failed for filter '{}', falling back to CPU",
                filter_name
            );
        } else if let Some((s, w, h)) = scaling::gpu::gpu_screenshot(
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
        let _ = (sf, scale, is_vectorize, is_adaptive);
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

    // Apply scaling filter if requested
    let scaled_buf;
    let (fb, fb_w, fb_h) = if let Some(f) = filter {
        let sf = scaling::ScaleFilter::from_name(f).unwrap_or_else(|| {
            eprintln!("Unknown filter '{}', using nearest", f);
            scaling::ScaleFilter::Nearest
        });
        let is_vectorize = f == "vectorize-legacy" || f == "vectorize-legacy-adaptive";
        let is_adaptive = f == "vectorize-legacy-adaptive";

        // Try GPU path if requested
        if use_gpu {
            if let Some((pixels, w, h)) =
                try_gpu_filter(raw_fb, f, sf, scale, is_vectorize, is_adaptive)
            {
                scaled_buf = pixels;
                return save_pixels(&scaled_buf, w, h, out, format, frames);
            }
        }

        // CPU path
        if is_vectorize {
            let s = scale as f64;
            let mut cache = crate::vectorize::VectorizeCache::new_legacy(is_adaptive);
            let (raster, rw, rh) = cache.rasterize(raw_fb, GB_FB_WIDTH, GB_FB_HEIGHT, s);
            scaled_buf = raster.to_vec();
            (scaled_buf.as_slice(), rw, rh)
        } else {
            let disp_w = GB_FB_WIDTH * scale;
            let disp_h = GB_FB_HEIGHT * scale;
            let (s, w, h) = scaling::cpu_scale(sf, raw_fb, GB_FB_WIDTH, GB_FB_HEIGHT, disp_w, disp_h)
                .unwrap_or_else(|| (raw_fb.to_vec(), GB_FB_WIDTH as u32, GB_FB_HEIGHT as u32));
            scaled_buf = s;
            (scaled_buf.as_slice(), w as usize, h as usize)
        }
    } else {
        (raw_fb, GB_FB_WIDTH, GB_FB_HEIGHT)
    };

    if matches!(format, "raster" | "diffusion" | "spline-diffusion" | "edge" | "gpu-full" | "cpu-dump") {
        vectorize_and_save(fb, GB_FB_WIDTH, GB_FB_HEIGHT, out, format, scale, use_gpu);
    } else {
        save_pixels(fb, fb_w, fb_h, out, format, frames);
    }
}

pub fn cmd_vectorize(input: &Path, out: &str, format: &str, scale: usize, gpu: bool) {
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

    vectorize_and_save(&pixels, width, height, out, format, scale, gpu);
}
