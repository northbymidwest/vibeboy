use std::fs;
use std::path::Path;

use crate::emulator::Emulator;
use crate::model::GbModel;
use crate::test_runner::harness::{TestHarness, TestResult};

pub struct TearoomHarness {
    pub force_model: Option<GbModel>,
}

/// Convert our 0x00RRGGBB frame buffer pixel to the DMG shade index (0-3).
fn pixel_to_dmg_shade(pixel: u32) -> u8 {
    let r = ((pixel >> 16) & 0xFF) as u8;
    // Map to DMG shades: #FFFFFF=0, #AAAAAA=1, #555555=2, #000000=3
    if r >= 0xD0 {
        0
    } else if r >= 0x80 {
        1
    } else if r >= 0x2A {
        2
    } else {
        3
    }
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

impl TestHarness for TearoomHarness {
    fn name(&self) -> &str {
        "Tearoom (screenshot comparison after LD B,B breakpoint)"
    }

    fn run_test(&self, path: &Path, verbose: bool) -> TestResult {
        let rom = match fs::read(path) {
            Ok(r) => r,
            Err(_) => return TestResult::Err,
        };
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let dir = path.parent().unwrap_or(Path::new("."));

        // Determine model: -C suffix = CGB, otherwise DMG
        let is_cgb_test = stem.ends_with("-C");
        let model = self
            .force_model
            .unwrap_or(if is_cgb_test { GbModel::Cgb } else { GbModel::Dmg });

        // Find reference image: try {stem}_dmg_blob.png, {stem}_dmg_b.png, etc.
        let ref_path = if is_cgb_test {
            ["_cgb_c.png", "_cgb_d.png"]
                .iter()
                .map(|suffix| dir.join(format!("{}{}", stem, suffix)))
                .find(|p| p.exists())
        } else {
            ["_dmg_blob.png", "_dmg_b.png"]
                .iter()
                .map(|suffix| dir.join(format!("{}{}", stem, suffix)))
                .find(|p| p.exists())
        };

        let ref_path = match ref_path {
            Some(p) => p,
            None => {
                if verbose {
                    eprintln!("  no reference image found");
                }
                return TestResult::Skip;
            }
        };

        let mut emu = Emulator::new(rom, None, None, model, None);
        let hit = emu.run_until_breakpoint(300);
        if hit.is_none() {
            return TestResult::Timeout;
        }

        // Load reference image
        let ref_img = match image::open(&ref_path) {
            Ok(img) => img.to_rgb8(),
            Err(_) => {
                if verbose {
                    eprintln!("  failed to load reference: {}", ref_path.display());
                }
                return TestResult::Err;
            }
        };

        if ref_img.width() != 160 || ref_img.height() != 144 {
            if verbose {
                eprintln!(
                    "  reference image wrong size: {}x{}",
                    ref_img.width(),
                    ref_img.height()
                );
            }
            return TestResult::Err;
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
            TestResult::Pass
        } else {
            if verbose {
                eprintln!(
                    "  {} pixel mismatches vs {}",
                    mismatches,
                    ref_path.file_name().unwrap().to_str().unwrap()
                );
            }
            TestResult::Fail
        }
    }
}
