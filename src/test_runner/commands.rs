use std::collections::VecDeque;
use std::fs;
use std::path::Path;

use crate::emulator::Emulator;
use crate::model::GbModel;
use crate::scaling;
use crate::vectorize;
use crate::test_runner::test_model::{detect_model_with_rom, load_boot_rom, resolve_boot_rom};
use crate::test_runner::util::{make_emu, parse_keys};

fn save_pixels(pixels: &[u32], w: usize, h: usize, out: &str, format: &str, frames: u32) {
    if format == "svg" || out.ends_with(".svg") {
        let svg = vectorize::vectorize_to_svg(pixels, w, h);
        fs::write(out, &svg).expect("Failed to write SVG");
        eprintln!("Wrote {} (frame {}, {} bytes SVG)", out, frames, svg.len());
    } else {
        let mut rgb = Vec::with_capacity(w * h * 3);
        for &pixel in pixels {
            rgb.push(((pixel >> 16) & 0xFF) as u8);
            rgb.push(((pixel >> 8) & 0xFF) as u8);
            rgb.push((pixel & 0xFF) as u8);
        }
        image::save_buffer(out, &rgb, w as u32, h as u32, image::ColorType::Rgb8)
            .expect("Failed to write PNG");
        eprintln!("Wrote {} (frame {}, {}x{})", out, frames, w, h);
    }
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
        let sf = match f {
            "epx" => scaling::ScaleFilter::Epx,
            "scale2x" => scaling::ScaleFilter::Scale2x,
            "scale3x" => scaling::ScaleFilter::Scale3x,
            "scale4x" => scaling::ScaleFilter::Scale4x,
            "eagle" => scaling::ScaleFilter::Eagle,
            "2xsai" => scaling::ScaleFilter::Sai2x,
            "super-2xsai" => scaling::ScaleFilter::Super2xSai,
            "super-eagle" => scaling::ScaleFilter::SuperEagle,
            "hq2x" => scaling::ScaleFilter::Hqx(scaling::HqxScale::Hq2x),
            "hq3x" => scaling::ScaleFilter::Hqx(scaling::HqxScale::Hq3x),
            "hq4x" => scaling::ScaleFilter::Hqx(scaling::HqxScale::Hq4x),
            "xbr2x" => scaling::ScaleFilter::Xbr(scaling::XbrScale::Xbr2x),
            "xbr3x" => scaling::ScaleFilter::Xbr(scaling::XbrScale::Xbr3x),
            "xbr4x" => scaling::ScaleFilter::Xbr(scaling::XbrScale::Xbr4x),
            "xbrz2x" => scaling::ScaleFilter::Xbrz(scaling::xbrz::XbrzScale::Xbrz2x),
            "xbrz3x" => scaling::ScaleFilter::Xbrz(scaling::xbrz::XbrzScale::Xbrz3x),
            "xbrz4x" => scaling::ScaleFilter::Xbrz(scaling::xbrz::XbrzScale::Xbrz4x),
            "xbrz5x" => scaling::ScaleFilter::Xbrz(scaling::xbrz::XbrzScale::Xbrz5x),
            "xbrz6x" => scaling::ScaleFilter::Xbrz(scaling::xbrz::XbrzScale::Xbrz6x),
            "super-xbr" => scaling::ScaleFilter::SuperXbr,
            "omniscale" => scaling::ScaleFilter::OmniScale,
            "omniscale-legacy" => scaling::ScaleFilter::OmniScaleLegacy,
            "aa-nearest" => scaling::ScaleFilter::AaNearestNeighbor,
            "bicubic" => scaling::ScaleFilter::Bicubic,
            "vectorize" | "vectorize-adaptive" => scaling::ScaleFilter::Vectorize,
            other => {
                eprintln!("Unknown filter '{}', using nearest", other);
                scaling::ScaleFilter::Nearest
            }
        };
        let is_vectorize = f == "vectorize" || f == "vectorize-adaptive";
        let is_adaptive = f == "vectorize-adaptive";

        // Try GPU path if requested
        #[cfg(feature = "sdl3-gpu-shaders")]
        if use_gpu {
            if is_vectorize {
                let s = scale as f64;
                if let Some((pix, w, h)) = scaling::gpu::gpu_vectorize_screenshot(
                    raw_fb, 160, 144, s, is_adaptive,
                ) {
                    scaled_buf = pix;
                    return save_pixels(&scaled_buf, w as usize, h as usize, out, format, frames);
                }
                eprintln!("GPU vectorize screenshot failed, falling back to CPU");
            } else if let Some((s, w, h)) = scaling::gpu::gpu_screenshot(raw_fb, 160, 144, sf) {
                scaled_buf = s;
                return save_pixels(&scaled_buf, w as usize, h as usize, out, format, frames);
            } else {
                eprintln!("GPU screenshot failed for filter '{}', falling back to CPU", f);
            }
        }
        #[cfg(not(feature = "sdl3-gpu-shaders"))]
        if use_gpu {
            eprintln!("GPU support not compiled in (enable sdl3-gpu-shaders feature)");
        }

        // CPU path
        if is_vectorize {
            let s = scale as f64;
            let mut cache = crate::vectorize::VectorizeCache::new(is_adaptive);
            let (raster, rw, rh) = cache.rasterize(raw_fb, 160, 144, s);
            scaled_buf = raster.to_vec();
            (scaled_buf.as_slice(), rw, rh)
        } else {
            let disp_w = 160 * scale;
            let disp_h = 144 * scale;
            let (s, w, h) = scaling::cpu_scale(sf, raw_fb, 160, 144, disp_w, disp_h)
                .unwrap_or_else(|| (raw_fb.to_vec(), 160, 144));
            scaled_buf = s;
            (scaled_buf.as_slice(), w as usize, h as usize)
        }
    } else {
        (raw_fb, 160, 144)
    };

    if format == "raster" {
        let (pixels, out_w, out_h) = vectorize::vectorize_to_raster(fb, 160, 144, scale);
        save_pixels(&pixels, out_w, out_h, out, "png", frames);
    } else if format == "diffusion" {
        let (pixels, out_w, out_h) = vectorize::rasterize::rasterize_diffusion(fb, 160, 144, scale);
        save_pixels(&pixels, out_w, out_h, out, "png", frames);
    } else if format == "spline-diffusion" {
        // Paper's full pipeline: YUV visible edges + spline boundaries + Gaussian diffusion
        vectorize::contour::YUV_VISIBLE_EDGES.store(true, std::sync::atomic::Ordering::Relaxed);
        let (paths, bg_color) = vectorize::vectorize_core(fb, 160, 144);
        vectorize::contour::YUV_VISIBLE_EDGES.store(false, std::sync::atomic::Ordering::Relaxed);
        let (pixels, out_w, out_h) = vectorize::rasterize::rasterize_spline_diffusion(
            &paths, fb, 160, 144, bg_color, scale,
        );
        save_pixels(&pixels, out_w, out_h, out, "png", frames);
    } else {
        save_pixels(fb, fb_w, fb_h, out, format, frames);
    }
}

pub fn cmd_vectorize(input: &Path, out: &str, format: &str, scale: usize) {
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

    if out.ends_with(".svg") {
        let svg = vectorize::vectorize_to_svg(&pixels, width, height);
        fs::write(out, &svg).expect("Failed to write SVG");
        eprintln!(
            "Vectorized {}x{} image -> {} ({} bytes)",
            width, height, out,
            svg.len()
        );
    } else {
        let raster_pixels = if format == "spline-diffusion" {
            vectorize::contour::YUV_VISIBLE_EDGES.store(true, std::sync::atomic::Ordering::Relaxed);
            let (paths, bg_color) = vectorize::vectorize_core(&pixels, width, height);
            vectorize::contour::YUV_VISIBLE_EDGES.store(false, std::sync::atomic::Ordering::Relaxed);
            let (r, _, _) = vectorize::rasterize::rasterize_spline_diffusion(
                &paths, &pixels, width, height, bg_color, scale,
            );
            r
        } else if format == "diffusion" {
            let (r, _, _) = vectorize::rasterize::rasterize_diffusion(&pixels, width, height, scale);
            r
        } else {
            let (r, _, _) = vectorize::vectorize_to_raster(&pixels, width, height, scale);
            r
        };
        let out_w = width * scale;
        let out_h = height * scale;
        let mut rgb = Vec::with_capacity(out_w * out_h * 3);
        for pixel in &raster_pixels {
            rgb.push(((pixel >> 16) & 0xFF) as u8);
            rgb.push(((pixel >> 8) & 0xFF) as u8);
            rgb.push((pixel & 0xFF) as u8);
        }
        image::save_buffer(out, &rgb, out_w as u32, out_h as u32, image::ColorType::Rgb8)
            .expect("Failed to write PNG");
        eprintln!(
            "Vectorized+rasterized {}x{} image -> {} ({}x{} at {}x, format={})",
            width, height, out, out_w, out_h, scale, format
        );
    }
}

pub fn cmd_analyze(
    path: &Path,
    force_model: Option<GbModel>,
    boot: bool,
    bootrom: Option<&Path>,
    frames: u32,
) {
    let rom = fs::read(path).expect("Failed to read ROM");
    let model = force_model.unwrap_or(GbModel::Cgb);
    let br = resolve_boot_rom(boot, bootrom, model);
    let mut emu = make_emu(rom, br, model);

    for _ in 0..frames {
        emu.step_frame();
    }

    let fb = emu.frame_buffer();
    eprintln!("Frame {} analysis (right edge, columns 155-159):", frames);
    let mut yellow_count = 0;
    for y in 0..144 {
        let mut right_colors = Vec::new();
        for x in 155..160 {
            let pixel = fb[y * 160 + x];
            right_colors.push(pixel);
            let r = (pixel >> 16) & 0xFF;
            let g = (pixel >> 8) & 0xFF;
            let b = pixel & 0xFF;
            if r > 200 && g > 150 && b < 100 {
                yellow_count += 1;
            }
        }
        if right_colors.iter().any(|&p| {
            let r = (p >> 16) & 0xFF;
            let g = (p >> 8) & 0xFF;
            let b = p & 0xFF;
            r > 200 && g > 150 && b < 100
        }) {
            eprintln!(
                "  LY={:3}: {:08X} {:08X} {:08X} {:08X} {:08X}  SCX={}",
                y, right_colors[0], right_colors[1], right_colors[2], right_colors[3],
                right_colors[4], emu.bus.ppu.scx
            );
        }
    }
    eprintln!("\nLeft edge sample (LY=72):");
    for x in 0..8 {
        let pixel = fb[72 * 160 + x];
        eprint!("{:08X} ", pixel);
    }
    eprintln!("\nRight edge sample (LY=72):");
    for x in 152..160 {
        let pixel = fb[72 * 160 + x];
        eprint!("{:08X} ", pixel);
    }
    eprintln!(
        "\nTotal yellow pixels on right edge (cols 155-159): {}",
        yellow_count
    );
}

pub fn cmd_trace_timer(
    path: &Path,
    force_model: Option<GbModel>,
    boot: bool,
    bootrom: Option<&Path>,
) {
    let rom = fs::read(path).expect("Failed to read ROM");
    let model = force_model.unwrap_or_else(|| detect_model_with_rom(path, Some(&rom)));
    let br = resolve_boot_rom(boot, bootrom, model);
    let mut emu = Emulator::new(rom, br, None, model, None);
    emu.headless = true;
    emu.bus.apu.headless = true;
    if boot || bootrom.is_some() {
        let mut history: VecDeque<(u16, u8, u16, u16)> = VecDeque::new();
        loop {
            let pc = emu.cpu.regs.pc;
            let opcode = emu.bus.read_byte(pc);
            let tc_before = emu.bus.timer.counter();
            if pc == 0x0100 && !emu.bus.boot_rom_active {
                break;
            }
            emu.cpu.step(&mut emu.bus);
            let tc_after = emu.bus.timer.counter();
            history.push_back((pc, opcode, tc_before, tc_after));
            if history.len() > 30 {
                history.pop_front();
            }
        }
        eprintln!("Last {} instructions of boot ROM:", history.len());
        for (pc, opcode, tc_before, tc_after) in &history {
            let delta = tc_after.wrapping_sub(*tc_before);
            eprintln!(
                "  PC={:#06X} op={:#04X} timer: {:#06X} -> {:#06X} (+{}) DIV: {:02X} -> {:02X}",
                pc,
                opcode,
                tc_before,
                tc_after,
                delta,
                tc_before >> 8,
                tc_after >> 8
            );
        }
    }
    eprintln!(
        "At PC=$0100: timer_counter={:#06X} DIV={:02X}",
        emu.bus.timer.counter(),
        emu.bus.timer.counter() >> 8
    );
    for i in 0..20 {
        let pc = emu.cpu.regs.pc;
        let opcode = emu.bus.read_byte(pc);
        let tc_before = emu.bus.timer.counter();
        emu.cpu.step(&mut emu.bus);
        let tc_after = emu.bus.timer.counter();
        eprintln!(
            "  step {}: PC={:#06X} op={:#04X} timer: {:#06X} -> {:#06X} (DIV: {:02X} -> {:02X})",
            i,
            pc,
            opcode,
            tc_before,
            tc_after,
            tc_before >> 8,
            tc_after >> 8
        );
    }
}

pub fn cmd_calibrate(path: &Path) {
    let rom = fs::read(path).expect("Failed to read ROM");
    let models = [
        GbModel::Dmg0,
        GbModel::Dmg,
        GbModel::Mgb,
        GbModel::Sgb,
        GbModel::Sgb2,
        GbModel::Cgb,
        GbModel::Agb,
    ];
    for model in &models {
        if let Some(br) = load_boot_rom(*model) {
            let mut emu = Emulator::new(rom.clone(), Some(br), None, *model, None);
            emu.headless = true;
            emu.bus.apu.headless = true;
            for _ in 0..100_000_000u64 {
                if emu.cpu.regs.pc == 0x0100 && !emu.bus.boot_rom_active {
                    let div_val = emu.bus.read_byte(0xFF04);
                    eprintln!(
                        "{:?}: LY={} dot={} mode={} total_ticks={} DIV={:02X} timer_counter={:#06X} regs=A:{:02X} F:{:02X} B:{:02X} C:{:02X} D:{:02X} E:{:02X} H:{:02X} L:{:02X}",
                        model,
                        emu.bus.ppu.ly,
                        emu.bus.ppu.dot,
                        emu.bus.ppu.stat & 0x03,
                        emu.bus.ppu.total_ticks,
                        div_val,
                        emu.bus.timer.counter(),
                        emu.cpu.regs.a,
                        emu.cpu.regs.f,
                        emu.cpu.regs.b,
                        emu.cpu.regs.c,
                        emu.cpu.regs.d,
                        emu.cpu.regs.e,
                        emu.cpu.regs.h,
                        emu.cpu.regs.l
                    );
                    break;
                }
                emu.cpu.step(&mut emu.bus);
            }
        } else {
            eprintln!("{:?}: no boot ROM available", model);
        }
    }
}
