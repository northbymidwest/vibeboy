use std::fs;
use std::path::Path;

use vibeboy::model::GbModel;
use vibeboy::scaling;
use crate::test_model::{detect_model_with_rom, resolve_boot_rom};
use crate::util::{make_emu, parse_keys, GB_FB_WIDTH, GB_FB_HEIGHT};

fn vectorize_gpu_to_svg(pixels: &[u32], width: usize, height: usize) -> String {
    let data = vibeboy::scaling::vectorize_gpu::vectorize(pixels, width, height);
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
        let svg = vectorize_gpu_to_svg(pixels, w, h);
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
        let svg = vectorize_gpu_to_svg(pixels, width, height);
        fs::write(out, &svg).expect("Failed to write SVG");
        eprintln!(
            "Vectorized {}x{} image -> {} ({} bytes)",
            width, height, out, svg.len()
        );
        return;
    }

    let (raster_pixels, out_w, out_h) = match format {
        "gpu-full" => {
            gpu_with_cpu_fallback(
                "gpu-full",
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
                || {
                    let scale_f = scale as f32;
                    let ow = (width as f32 * scale_f).round() as usize;
                    let oh = (height as f32 * scale_f).round() as usize;
                    let r = vibeboy::scaling::vectorize_gpu::scale(pixels, width, height, scale_f);
                    (r, ow, oh)
                },
            )
        }
        "vectorize-gpu" | _ => {
            let scale_f = scale as f32;
            let out_w = (width as f32 * scale_f).round() as usize;
            let out_h = (height as f32 * scale_f).round() as usize;
            let r = vibeboy::scaling::vectorize_gpu::scale(pixels, width, height, scale_f);
            (r, out_w, out_h)
        }
    };
    save_pixels_png(&raster_pixels, out_w, out_h, out);
    eprintln!(
        "Vectorized+rasterized {}x{} image -> {} ({}x{} at {}x, format={})",
        width, height, out, out_w, out_h, scale, format
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
        if filter_name == "vectorize-gpu" {
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
        let svg = vectorize_gpu_to_svg(raw_fb, GB_FB_WIDTH, GB_FB_HEIGHT);
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

    if matches!(format, "raster" | "gpu-full" | "vectorize-gpu") {
        vectorize_and_save(fb, GB_FB_WIDTH, GB_FB_HEIGHT, out, format, scale, use_gpu);
    } else {
        save_pixels(fb, fb_w, fb_h, out, format, frames);
    }
}

pub fn cmd_vectorize(input: &Path, out: &str, filter: &str, scale: usize, gpu: bool) {
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

    // Map --filter names to internal format names for vectorize_and_save
    // All legacy vectorize variants now map to vectorize-gpu
    let format = match filter {
        "vectorize-gpu" => "vectorize-gpu",
        "gpu-full" => "gpu-full",
        other => {
            // Any legacy vectorize name maps to vectorize-gpu
            if other.starts_with("vectorize") || other == "raster" || other == "diffusion"
                || other == "spline-diffusion" || other == "edge"
            {
                "vectorize-gpu"
            } else {
                other
            }
        }
    };

    vectorize_and_save(&pixels, width, height, out, format, scale, gpu);
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
