mod apu;
mod bus;
mod cartridge;
mod cpu;
mod emulator;
mod joypad;
mod model;
mod ppu;
mod printer;
mod serial;
mod sgb;
mod snapshot;
mod snes;
mod timer;
mod vectorize;

use clap::{Parser, Subcommand};
use emulator::Emulator;
use model::GbModel;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "test_runner", about = "Game Boy emulator test runner")]
struct Cli {
    /// Path to ROM file or directory of ROMs
    path: PathBuf,

    /// Hardware model override (auto-detected from filename if not specified)
    #[arg(long, value_parser = parse_model)]
    model: Option<GbModel>,

    /// Use boot ROM (auto-detected by model)
    #[arg(long)]
    boot: bool,

    /// Path to boot ROM file (implies --boot)
    #[arg(long)]
    bootrom: Option<PathBuf>,

    /// PPU renderer: fifo (default), line, screen
    #[arg(long, default_value = "fifo")]
    renderer: String,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run in Blargg test mode (serial output detection)
    Blargg,
    /// Run in Gambatte test mode (hex output comparison after 15 frames)
    Gambatte,
    /// Run in GBMicrotest mode (HRAM result check after 2 frames)
    Gbmicrotest,
    /// Run Mealybug Tearoom tests (screenshot comparison after LD B,B breakpoint)
    Tearoom,
    /// Take a screenshot after N frames
    Screenshot {
        /// Number of frames to run before capturing
        #[arg(long, default_value = "300")]
        frames: u32,
        /// Output file path
        #[arg(long, default_value = "screenshot.png")]
        out: String,
        /// Output format: png, svg, or raster (vectorized then rasterized)
        #[arg(long, default_value = "png")]
        format: String,
        /// Scale factor for raster format (default 4)
        #[arg(long, default_value = "4")]
        scale: usize,
    },
    /// Analyze frame buffer (debug tool)
    Analyze {
        /// Number of frames to run
        #[arg(long, default_value = "300")]
        frames: u32,
    },
    /// Trace timer state around boot ROM handoff (debug tool)
    TraceTimer,
    /// Dump PPU/timer state at PC=$0100 for all models with boot ROMs
    Calibrate,
    /// Vectorize an input PNG image to SVG using Kopf-Lischinski
    Vectorize {
        /// Output SVG file path
        #[arg(long, default_value = "output.svg")]
        out: String,
    },
}

fn parse_model(s: &str) -> Result<GbModel, String> {
    s.parse::<GbModel>()
}

fn parse_renderer(s: &str) -> ppu::RendererMode {
    match s.to_lowercase().as_str() {
        "fifo" => ppu::RendererMode::Fifo,
        "line" => ppu::RendererMode::Line,
        "screen" => ppu::RendererMode::Screen,
        other => {
            eprintln!("Unknown renderer '{}'. Options: fifo, line, screen", other);
            std::process::exit(1);
        }
    }
}

fn make_emu(rom: Vec<u8>, boot_rom: Option<Vec<u8>>, model: GbModel, renderer: ppu::RendererMode) -> Emulator {
    let mut emu = Emulator::new(rom, boot_rom, None, model, None);
    emu.bus.ppu.renderer_mode = renderer;
    emu
}

fn detect_model_with_rom(path: &Path, rom: Option<&[u8]>) -> GbModel {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let path_str = path.to_str().unwrap_or("");
    // OAM bug tests are DMG-specific despite having CGB flag 0x80
    if path_str.contains("oam_bug") {
        return GbModel::Dmg;
    }
    // Check longer suffixes first to avoid partial matches
    if stem.ends_with("-dmgABCmgb") || stem.ends_with("-dmgABC") {
        GbModel::Dmg
    } else if stem.ends_with("-dmg0") {
        GbModel::Dmg0
    } else if stem.ends_with("-mgb") {
        GbModel::Mgb
    } else if stem.ends_with("-sgb2") {
        GbModel::Sgb2
    } else if stem.ends_with("-sgb") || stem.ends_with("-S") {
        GbModel::Sgb
    } else if stem.ends_with("-A") {
        GbModel::Agb
    } else if stem.ends_with("-cgb0") {
        GbModel::Cgb0
    } else if stem.ends_with("-C") || stem.ends_with("-cgb") || stem.ends_with("-cgbABCDE") {
        GbModel::Cgb
    } else if stem.ends_with("-GS") || stem.ends_with("-G") {
        GbModel::Dmg
    } else if let Some(data) = rom {
        // Auto-detect from cart header CGB flag
        let cgb_flag = data.get(0x0143).copied().unwrap_or(0);
        if cgb_flag == 0x80 || cgb_flag == 0xC0 {
            GbModel::Cgb
        } else {
            GbModel::Dmg
        }
    } else {
        GbModel::Cgb
    }
}


/// Mooneye pass: B=3,C=5,D=8,E=13,H=21,L=34 (Fibonacci)
fn mooneye_passed(regs: [u8; 6]) -> bool {
    regs == [3, 5, 8, 13, 21, 34]
}

/// Load boot ROM for the given model from bootroms/ directory.
fn load_boot_rom(model: GbModel) -> Option<Vec<u8>> {
    let candidates: &[&str] = match model {
        GbModel::Dmg0 => &["bootroms/dmg0_boot.bin", "bootroms/dmg_boot.bin"],
        GbModel::Dmg  => &["bootroms/dmg_boot.bin"],
        GbModel::Mgb  => &["bootroms/mgb_boot.bin", "bootroms/dmg_boot.bin"],
        GbModel::Sgb  => &["bootroms/sgb_boot.bin"],
        GbModel::Sgb2 => &["bootroms/sgb2_boot.bin"],
        GbModel::Cgb0 => &["bootroms/cgb0_boot.bin", "bootroms/cgb_boot.bin"],
        GbModel::Cgb => &["bootroms/cgb_boot.bin", "gbc_bios.bin"],
        GbModel::Agb => &["bootroms/cgb_agb_boot.bin", "bootroms/cgb_boot.bin"],
    };
    for path in candidates {
        if let Ok(data) = fs::read(path) {
            return Some(data);
        }
    }
    None
}

fn resolve_boot_rom(cli: &Cli, model: GbModel) -> Option<Vec<u8>> {
    if let Some(ref p) = cli.bootrom {
        Some(fs::read(p).unwrap_or_else(|e| {
            eprintln!("Failed to read boot ROM '{}': {}", p.display(), e);
            std::process::exit(1);
        }))
    } else if cli.boot {
        load_boot_rom(model)
    } else {
        None
    }
}

fn run_test_mooneye(path: &Path, verbose: bool, force_model: Option<GbModel>, boot_rom: Option<Vec<u8>>) -> &'static str {
    let rom = match fs::read(path) {
        Ok(r) => r,
        Err(_) => return "ERR",
    };
    let model = force_model.unwrap_or_else(|| detect_model_with_rom(path, Some(&rom)));
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    // Use model-specific boot ROM variant (e.g. cgb0 for -cgb0 suffix)
    let br = if boot_rom.is_some() {
        if stem.contains("cgb0") {
            fs::read("bootroms/cgb0_boot.bin").ok()
                .or(boot_rom)
        } else {
            boot_rom
        }
    } else {
        None
    };
    let mut emu = Emulator::new(rom, br, None, model, None);
    match emu.run_until_breakpoint(300) {
        Some(regs) => {
            if mooneye_passed(regs) {
                "PASS"
            } else {
                if verbose {
                    eprintln!("  regs: B={:02X} C={:02X} D={:02X} E={:02X} H={:02X} L={:02X}",
                        regs[0], regs[1], regs[2], regs[3], regs[4], regs[5]);
                    if stem.contains("channel_") || stem.contains("div_") {
                        eprint!("  actual   @C000:");
                        for i in 0..128u16 {
                            if i % 16 == 0 && i > 0 { eprint!("\n                "); }
                            eprint!(" {:02X}", emu.bus.read_byte(0xC000 + i));
                        }
                        eprintln!();
                    }
                    if stem.contains("boot_hwio") {
                        // CGB expected values from boot_hwio-C.s
                        let expected: &[(u16, u8)] = &[
                            (0xFF00, 0xFF), (0xFF01, 0x00), (0xFF02, 0x7E), (0xFF03, 0xFF),
                            (0xFF05, 0x00), (0xFF06, 0x00), (0xFF07, 0xF8),
                            (0xFF08, 0xFF), (0xFF09, 0xFF), (0xFF0A, 0xFF), (0xFF0B, 0xFF),
                            (0xFF0C, 0xFF), (0xFF0D, 0xFF), (0xFF0E, 0xFF), (0xFF0F, 0xE1),
                            (0xFF10, 0x80), (0xFF11, 0xBF), (0xFF12, 0xF3), (0xFF13, 0xFF),
                            (0xFF14, 0xBF), (0xFF15, 0xFF), (0xFF16, 0x3F), (0xFF17, 0x00),
                            (0xFF18, 0xFF), (0xFF19, 0xBF), (0xFF1A, 0x7F), (0xFF1B, 0xFF),
                            (0xFF1C, 0x9F), (0xFF1D, 0xFF), (0xFF1E, 0xBF), (0xFF1F, 0xFF),
                            (0xFF20, 0xFF), (0xFF21, 0x00), (0xFF22, 0x00), (0xFF23, 0xBF),
                            (0xFF24, 0x77), (0xFF25, 0xF3), (0xFF26, 0xF1), (0xFF27, 0xFF),
                            (0xFF42, 0x00), (0xFF43, 0x00), (0xFF45, 0x00),
                            (0xFF47, 0xFC),
                            (0xFF4A, 0x00), (0xFF4B, 0x00),
                            // CGB-specific registers
                            (0xFF68, 0xC8), (0xFF6A, 0xD0),
                            (0xFF72, 0x00), (0xFF73, 0x00), (0xFF75, 0x8F),
                        ];
                        for &(addr, exp) in expected {
                            let got = emu.bus.read_byte(addr);
                            if got != exp {
                                eprintln!("  boot_hwio mismatch: ${:04X} expected=${:02X} got=${:02X}", addr, exp, got);
                            }
                        }
                    }
                }
                "FAIL"
            }
        }
        None => "TIMEOUT",
    }
}

fn run_test_gbmicrotest(path: &Path, verbose: bool, force_model: Option<GbModel>) -> &'static str {
    let rom = match fs::read(path) {
        Ok(r) => r,
        Err(_) => return "ERR",
    };
    let model = force_model.unwrap_or(GbModel::Dmg);
    let mut emu = Emulator::new(rom, None, None, model, None);
    // Run for 2 frames (sufficient per gbmicrotest docs)
    for _ in 0..2 {
        emu.step_frame();
    }

    // Check HRAM result at 0xFF82
    let result = emu.bus.read_byte(0xFF82);
    let actual = emu.bus.read_byte(0xFF80);
    let expected = emu.bus.read_byte(0xFF81);
    if result == 0x01 {
        "PASS"
    } else if result == 0xFF {
        if verbose {
            eprintln!("  actual=0x{:02X} expected=0x{:02X}", actual, expected);
        }
        "FAIL"
    } else {
        if verbose {
            eprintln!("  FF82=0x{:02X} (not 0x01 or 0xFF)", result);
        }
        "TIMEOUT"
    }
}

/// Convert our 0x00RRGGBB frame buffer pixel to the DMG shade index (0-3).
fn pixel_to_dmg_shade(pixel: u32) -> u8 {
    let r = ((pixel >> 16) & 0xFF) as u8;
    // Map to DMG shades: #FFFFFF=0, #AAAAAA=1, #555555=2, #000000=3
    if r >= 0xD0 { 0 } else if r >= 0x80 { 1 } else if r >= 0x2A { 2 } else { 3 }
}

/// Convert a DMG shade index to the tearoom reference RGB.
fn dmg_shade_to_rgb(shade: u8) -> (u8, u8, u8) {
    match shade {
        0 => (0xFF, 0xFF, 0xFF),
        1 => (0xAA, 0xAA, 0xAA),
        2 => (0x55, 0x55, 0x55),
        _ => (0x00, 0x00, 0x00),
    }
}

fn run_test_tearoom(path: &Path, verbose: bool, force_model: Option<GbModel>) -> &'static str {
    let rom = match fs::read(path) {
        Ok(r) => r,
        Err(_) => return "ERR",
    };
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let dir = path.parent().unwrap_or(Path::new("."));

    // Determine model: -C suffix = CGB, otherwise DMG
    let is_cgb_test = stem.ends_with("-C");
    let model = force_model.unwrap_or(if is_cgb_test { GbModel::Cgb } else { GbModel::Dmg });

    // Find reference image: try {stem}_dmg_blob.png, {stem}_dmg_b.png, etc.
    let ref_path = if is_cgb_test {
        // Try cgb_c, cgb_d variants
        ["_cgb_c.png", "_cgb_d.png"].iter()
            .map(|suffix| dir.join(format!("{}{}", stem, suffix)))
            .find(|p| p.exists())
    } else {
        ["_dmg_blob.png", "_dmg_b.png"].iter()
            .map(|suffix| dir.join(format!("{}{}", stem, suffix)))
            .find(|p| p.exists())
    };

    let ref_path = match ref_path {
        Some(p) => p,
        None => {
            if verbose { eprintln!("  no reference image found"); }
            return "SKIP";
        }
    };

    let mut emu = Emulator::new(rom, None, None, model, None);
    let hit = emu.run_until_breakpoint(300);
    if hit.is_none() {
        return "TIMEOUT";
    }

    // Load reference image
    let ref_img = match image::open(&ref_path) {
        Ok(img) => img.to_rgb8(),
        Err(_) => {
            if verbose { eprintln!("  failed to load reference: {}", ref_path.display()); }
            return "ERR";
        }
    };

    if ref_img.width() != 160 || ref_img.height() != 144 {
        if verbose { eprintln!("  reference image wrong size: {}x{}", ref_img.width(), ref_img.height()); }
        return "ERR";
    }

    let fb = emu.frame_buffer();

    // Compare pixel by pixel
    let mut mismatches = 0u32;
    for y in 0..144usize {
        for x in 0..160usize {
            let pixel = fb[y * 160 + x];
            let ref_pixel = ref_img.get_pixel(x as u32, y as u32);

            // Convert our pixel to standard shade RGB for comparison
            let our_shade = pixel_to_dmg_shade(pixel);
            let (er, eg, eb) = dmg_shade_to_rgb(our_shade);
            let (rr, rg, rb) = (ref_pixel[0], ref_pixel[1], ref_pixel[2]);

            if er != rr || eg != rg || eb != rb {
                mismatches += 1;
            }
        }
    }

    if mismatches == 0 {
        "PASS"
    } else {
        if verbose {
            eprintln!("  {} pixel mismatches vs {}", mismatches, ref_path.file_name().unwrap().to_str().unwrap());
        }
        "FAIL"
    }
}

fn run_test_blargg(path: &Path, verbose: bool) -> &'static str {
    let rom = match fs::read(path) {
        Ok(r) => r,
        Err(_) => return "ERR",
    };
    let model = detect_model_with_rom(path, Some(&rom));
    let br: Option<Vec<u8>> = None;
    let mut emu = Emulator::new(rom, br, None, model, None);
    let output = emu.run_until_serial_result(6000);
    if verbose && !output.is_empty() {
        for line in output.lines() {
            eprintln!("  {}", line);
        }
    }
    if output.contains("Passed") {
        "PASS"
    } else if output.contains("Failed") {
        "FAIL"
    } else {
        "TIMEOUT"
    }
}

fn cmd_screenshot(cli: &Cli, frames: u32, out: &str, format: &str, scale: usize) {
    let rom = fs::read(&cli.path).expect("Failed to read ROM");
    let model = cli.model.unwrap_or_else(|| detect_model_with_rom(&cli.path, Some(&rom)));
    let br = resolve_boot_rom(cli, model);
    let mut emu = make_emu(rom, br, model, parse_renderer(&cli.renderer));
    for _ in 0..frames {
        emu.step_frame();
    }
    let fb = emu.frame_buffer();

    if format == "svg" || (format == "png" && out.ends_with(".svg")) {
        let svg = vectorize::vectorize_to_svg(fb, 160, 144);
        fs::write(out, &svg).expect("Failed to write SVG");
        eprintln!("Wrote {} (frame {}, {} bytes SVG)", out, frames, svg.len());
    } else if format == "raster" {
        let (pixels, out_w, out_h) = vectorize::vectorize_to_raster(fb, 160, 144, scale);
        let mut rgb = Vec::with_capacity(out_w * out_h * 3);
        for pixel in &pixels {
            rgb.push(((pixel >> 16) & 0xFF) as u8);
            rgb.push(((pixel >> 8) & 0xFF) as u8);
            rgb.push((pixel & 0xFF) as u8);
        }
        image::save_buffer(out, &rgb, out_w as u32, out_h as u32, image::ColorType::Rgb8)
            .expect("Failed to write PNG");
        eprintln!("Wrote {} (frame {}, {}x{} at {}x scale, vectorized+rasterized)", out, frames, out_w, out_h, scale);
    } else {
        let mut rgb = Vec::with_capacity(160 * 144 * 3);
        for y in 0..144 {
            for x in 0..160 {
                let pixel = fb[y * 160 + x];
                rgb.push(((pixel >> 16) & 0xFF) as u8);
                rgb.push(((pixel >> 8) & 0xFF) as u8);
                rgb.push((pixel & 0xFF) as u8);
            }
        }
        image::save_buffer(out, &rgb, 160, 144, image::ColorType::Rgb8)
            .expect("Failed to write PNG");
        eprintln!("Wrote {} (frame {})", out, frames);
    }
}

fn cmd_vectorize(input: &Path, out: &str) {
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

    let svg = vectorize::vectorize_to_svg(&pixels, width, height);
    fs::write(out, &svg).expect("Failed to write SVG");
    eprintln!(
        "Vectorized {}x{} image → {} ({} bytes)",
        width, height, out, svg.len()
    );
}

fn cmd_analyze(cli: &Cli, frames: u32) {
    let rom = fs::read(&cli.path).expect("Failed to read ROM");
    let model = cli.model.unwrap_or(GbModel::Cgb);
    let br = resolve_boot_rom(cli, model);
    let mut emu = make_emu(rom, br, model, parse_renderer(&cli.renderer));

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
            eprintln!("  LY={:3}: {:08X} {:08X} {:08X} {:08X} {:08X}  SCX={}",
                y, right_colors[0], right_colors[1], right_colors[2],
                right_colors[3], right_colors[4], emu.bus.ppu.scx);
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
    eprintln!("\nTotal yellow pixels on right edge (cols 155-159): {}", yellow_count);
}

fn cmd_trace_timer(cli: &Cli) {
    let rom = fs::read(&cli.path).expect("Failed to read ROM");
    let model = cli.model.unwrap_or_else(|| detect_model_with_rom(&cli.path, Some(&rom)));
    let br = resolve_boot_rom(cli, model);
    let mut emu = Emulator::new(rom, br, None, model, None);
    if cli.boot || cli.bootrom.is_some() {
        let mut history: Vec<(u16, u8, u16, u16)> = Vec::new();
        loop {
            let pc = emu.cpu.regs.pc;
            let opcode = emu.bus.read_byte(pc);
            let tc_before = emu.bus.timer.counter();
            if pc == 0x0100 && !emu.bus.boot_rom_active { break; }
            emu.cpu.step(&mut emu.bus);
            let tc_after = emu.bus.timer.counter();
            history.push((pc, opcode, tc_before, tc_after));
            if history.len() > 30 { history.remove(0); }
        }
        eprintln!("Last {} instructions of boot ROM:", history.len());
        for (pc, opcode, tc_before, tc_after) in &history {
            let delta = tc_after.wrapping_sub(*tc_before);
            eprintln!("  PC={:#06X} op={:#04X} timer: {:#06X} -> {:#06X} (+{}) DIV: {:02X} -> {:02X}",
                pc, opcode, tc_before, tc_after, delta, tc_before >> 8, tc_after >> 8);
        }
    }
    eprintln!("At PC=$0100: timer_counter={:#06X} DIV={:02X}",
        emu.bus.timer.counter(), emu.bus.timer.counter() >> 8);
    for i in 0..20 {
        let pc = emu.cpu.regs.pc;
        let opcode = emu.bus.read_byte(pc);
        let tc_before = emu.bus.timer.counter();
        emu.cpu.step(&mut emu.bus);
        let tc_after = emu.bus.timer.counter();
        eprintln!("  step {}: PC={:#06X} op={:#04X} timer: {:#06X} -> {:#06X} (DIV: {:02X} -> {:02X})",
            i, pc, opcode, tc_before, tc_after, tc_before >> 8, tc_after >> 8);
    }
}

fn cmd_calibrate(cli: &Cli) {
    let rom = fs::read(&cli.path).expect("Failed to read ROM");
    let models = [GbModel::Dmg0, GbModel::Dmg, GbModel::Mgb, GbModel::Sgb, GbModel::Sgb2, GbModel::Cgb, GbModel::Agb];
    for model in &models {
        if let Some(br) = load_boot_rom(*model) {
            let mut emu = Emulator::new(rom.clone(), Some(br), None, *model, None);
            for _ in 0..100_000_000u64 {
                if emu.cpu.regs.pc == 0x0100 && !emu.bus.boot_rom_active {
                    let div_val = emu.bus.read_byte(0xFF04);
                    eprintln!("{:?}: LY={} dot={} mode={} total_ticks={} DIV={:02X} timer_counter={:#06X} regs=A:{:02X} F:{:02X} B:{:02X} C:{:02X} D:{:02X} E:{:02X} H:{:02X} L:{:02X}",
                        model, emu.bus.ppu.ly, emu.bus.ppu.dot, emu.bus.ppu.stat & 0x03,
                        emu.bus.ppu.total_ticks, div_val, emu.bus.timer.counter(),
                        emu.cpu.regs.a, emu.cpu.regs.f, emu.cpu.regs.b, emu.cpu.regs.c,
                        emu.cpu.regs.d, emu.cpu.regs.e, emu.cpu.regs.h, emu.cpu.regs.l);
                    break;
                }
                emu.cpu.step(&mut emu.bus);
            }
        } else {
            eprintln!("{:?}: no boot ROM available", model);
        }
    }
}

// Gambatte hex digit tile patterns (8x8 pixels each, bit 7=leftmost pixel)
// 1 = black (0x000000), 0 = white (0xF8F8F8)
const GAMBATTE_DIGITS: [[u8; 8]; 16] = [
    [0x00, 0x7F, 0x41, 0x41, 0x41, 0x41, 0x41, 0x7F], // 0
    [0x00, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08], // 1
    [0x00, 0x7F, 0x01, 0x01, 0x7F, 0x40, 0x40, 0x7F], // 2
    [0x00, 0x7F, 0x01, 0x01, 0x3F, 0x01, 0x01, 0x7F], // 3
    [0x00, 0x41, 0x41, 0x41, 0x7F, 0x01, 0x01, 0x01], // 4
    [0x00, 0x7F, 0x40, 0x40, 0x7E, 0x01, 0x01, 0x7E], // 5
    [0x00, 0x7F, 0x40, 0x40, 0x7F, 0x41, 0x41, 0x7F], // 6
    [0x00, 0x7F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10], // 7
    [0x00, 0x3E, 0x41, 0x41, 0x3E, 0x41, 0x41, 0x3E], // 8
    [0x00, 0x7F, 0x41, 0x41, 0x7F, 0x01, 0x01, 0x7F], // 9
    [0x00, 0x08, 0x22, 0x41, 0x7F, 0x41, 0x41, 0x41], // A
    [0x00, 0x7E, 0x41, 0x41, 0x7E, 0x41, 0x41, 0x7E], // B
    [0x00, 0x3E, 0x41, 0x40, 0x40, 0x40, 0x41, 0x3E], // C
    [0x00, 0x7E, 0x41, 0x41, 0x41, 0x41, 0x41, 0x7E], // D
    [0x00, 0x7F, 0x40, 0x40, 0x7F, 0x40, 0x40, 0x7F], // E
    [0x00, 0x7F, 0x40, 0x40, 0x7F, 0x40, 0x40, 0x40], // F
];

fn gambatte_digit_index(c: char) -> Option<usize> {
    match c {
        '0'..='9' => Some((c as u8 - b'0') as usize),
        'A'..='F' => Some((c as u8 - b'A') as usize + 10),
        'a'..='f' => Some((c as u8 - b'a') as usize + 10),
        _ => None,
    }
}

/// Check if framebuffer tile at (tile_x * 8, 0) matches expected digit pattern.
fn gambatte_tile_matches(fb: &[u32], tile_x: usize, digit: usize) -> bool {
    let pattern = &GAMBATTE_DIGITS[digit];
    for y in 0..8 {
        for x in 0..8 {
            let pixel = fb[y * 160 + tile_x * 8 + x];
            let masked = pixel & 0xF8F8F8;
            let expected_black = (pattern[y] >> (7 - x)) & 1 == 1;
            let is_black = masked == 0;
            if expected_black != is_black {
                return false;
            }
        }
    }
    true
}

/// Parse gambatte test: extract expected hex from filename, determine model(s).
/// Returns (expected_hex, is_dmg, is_cgb).
fn parse_gambatte_test(path: &Path) -> Option<(String, bool, bool)> {
    let stem = path.file_stem()?.to_str()?;
    // Find _out pattern and extract hex string
    // Patterns: "dmg08_cgb04c_out<HEX>", "dmg08_out<HEX>", "cgb04c_out<HEX>", "_out<HEX>"
    let is_dmg;
    let is_cgb;
    let hex_str;

    if let Some(pos) = stem.find("dmg08_cgb04c_out") {
        is_dmg = true;
        is_cgb = true;
        hex_str = &stem[pos + 16..];
    } else if let Some(pos) = stem.find("dmg08_out") {
        is_dmg = true;
        is_cgb = stem.contains("cgb04c_out");
        hex_str = &stem[pos + 9..];
    } else if let Some(pos) = stem.find("cgb04c_out") {
        is_dmg = false;
        is_cgb = true;
        hex_str = &stem[pos + 10..];
    } else if let Some(pos) = stem.rfind("_out") {
        is_dmg = false;
        is_cgb = true;
        hex_str = &stem[pos + 4..];
    } else {
        return None;
    }

    // Skip audio tests
    if hex_str.starts_with("audio") {
        return None;
    }

    if hex_str.is_empty() || !hex_str.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }

    Some((hex_str.to_uppercase(), is_dmg, is_cgb))
}

fn run_test_gambatte(path: &Path, verbose: bool, force_model: Option<GbModel>) -> (&'static str, &'static str) {
    let rom = match fs::read(path) {
        Ok(r) => r,
        Err(_) => return ("ERR", ""),
    };

    let (expected_hex, is_dmg, is_cgb) = match parse_gambatte_test(path) {
        Some(v) => v,
        None => return ("SKIP", ""),
    };

    // Determine which model to test
    let model = if let Some(m) = force_model {
        m
    } else if is_dmg && !is_cgb {
        GbModel::Dmg
    } else {
        // Default to CGB for cgb-only or dual tests
        GbModel::Cgb
    };

    let mut emu = Emulator::new(rom, None, None, model, None);

    // Run for 15 frames
    for _ in 0..15 {
        emu.step_frame();
    }

    let fb = emu.frame_buffer();

    // Compare each hex digit
    for (i, c) in expected_hex.chars().enumerate() {
        let digit = match gambatte_digit_index(c) {
            Some(d) => d,
            None => return ("ERR", "bad hex"),
        };
        if !gambatte_tile_matches(fb, i, digit) {
            if verbose {
                // Show what we got vs expected
                let mut actual = String::new();
                for d in 0..16usize {
                    if gambatte_tile_matches(fb, i, d) {
                        actual = format!("{:X}", d);
                        break;
                    }
                }
                if actual.is_empty() { actual = "?".to_string(); }
                eprintln!("  digit {}: expected={} got={}", i, c, actual);
            }
            return ("FAIL", "");
        }
    }

    ("PASS", "")
}

fn run_tests(cli: &Cli) {
    let blargg_mode = matches!(cli.command, Some(Command::Blargg));
    let gambatte_mode = matches!(cli.command, Some(Command::Gambatte));
    let gbmicrotest_mode = matches!(cli.command, Some(Command::Gbmicrotest));
    let tearoom_mode = matches!(cli.command, Some(Command::Tearoom));
    let mut roms: Vec<PathBuf> = Vec::new();
    collect_roms(&cli.path, &mut roms);
    roms.sort();

    if blargg_mode {
        eprintln!("Blargg mode (serial output detection)");
    } else if gambatte_mode {
        eprintln!("Gambatte mode (hex output comparison, 15 frames)");
    } else if gbmicrotest_mode {
        eprintln!("GBMicrotest mode (HRAM result check, 2 frames)");
    } else if tearoom_mode {
        eprintln!("Tearoom mode (screenshot comparison after LD B,B breakpoint)");
    }

    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut timeout = 0usize;
    let mut skipped = 0usize;

    for rom in &roms {
        let result = if blargg_mode {
            run_test_blargg(rom, true)
        } else if gambatte_mode {
            let (r, _) = run_test_gambatte(rom, true, cli.model);
            r
        } else if gbmicrotest_mode {
            run_test_gbmicrotest(rom, true, cli.model)
        } else if tearoom_mode {
            run_test_tearoom(rom, true, cli.model)
        } else {
            let model = cli.model.unwrap_or_else(|| detect_model_with_rom(rom, None));
            let br = if cli.boot || cli.bootrom.is_some() {
                if let Some(ref p) = cli.bootrom {
                    fs::read(p).ok()
                } else {
                    load_boot_rom(model)
                }
            } else {
                None
            };
            run_test_mooneye(rom, true, cli.model, br)
        };
        let label = rom.strip_prefix(&cli.path).unwrap_or(rom).display().to_string();
        match result {
            "SKIP" => { skipped += 1; continue; }
            _ => {}
        }
        println!("{:<12} {}", result, label);
        match result {
            "PASS"    => passed += 1,
            "FAIL"    => failed += 1,
            "TIMEOUT" => timeout += 1,
            _ => {}
        }
    }

    print!("\n--- {} passed, {} failed, {} timeout", passed, failed, timeout);
    if skipped > 0 {
        print!(", {} skipped", skipped);
    }
    println!(" ({} total) ---", passed + failed + timeout);
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    match &cli.command {
        Some(Command::Screenshot { frames, out, format, scale }) => cmd_screenshot(&cli, *frames, out, format, *scale),
        Some(Command::Vectorize { out }) => cmd_vectorize(&cli.path, out),
        Some(Command::Analyze { frames }) => cmd_analyze(&cli, *frames),
        Some(Command::TraceTimer) => cmd_trace_timer(&cli),
        Some(Command::Calibrate) => cmd_calibrate(&cli),
        Some(Command::Blargg) | Some(Command::Gambatte) | Some(Command::Gbmicrotest) | Some(Command::Tearoom) | None => run_tests(&cli),
    }
}

fn collect_roms(dir: &Path, out: &mut Vec<PathBuf>) {
    if dir.is_file() {
        let ext = dir.extension().and_then(|e| e.to_str());
        if ext == Some("gb") || ext == Some("gbc") {
            out.push(dir.to_path_buf());
        }
        return;
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            collect_roms(&entry.path(), out);
        }
    }
}
