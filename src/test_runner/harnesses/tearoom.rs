use std::fs;
use std::path::Path;

use crate::harness::{TestHarness, TestResult};
use crate::util::{GB_FB_HEIGHT, GB_FB_WIDTH, make_emu};
use vibeboy_core::model::GbModel;
use vibeboy_core::ppu::Ppu;

pub struct TearoomHarness {
    pub force_model: Option<GbModel>,
}

/// Reference RGB for the four DMG shades (lightest to darkest), as used by
/// the mealybug expected screenshots.
const REF_DMG_GRAYS: [u32; 4] = [0x00FFFFFF, 0x00AAAAAA, 0x00555555, 0x00000000];

/// Palettes the CGB boot ROM assigns in DMG compatibility mode for these ROMs
/// (RGB555). The reference screenshots were captured with these; they convert
/// with the same `(X << 3) | (X >> 2)` expansion the PPU uses.
const CGB_COMPAT_BG: [u16; 4] = [0x7FFF, 0x1BEF, 0x6180, 0x0000];
const CGB_COMPAT_OBJ: [u16; 4] = [0x7FFF, 0x421F, 0x1CF2, 0x0000];

/// Map a DMG/MGB frame buffer pixel to the reference gray for the same shade.
/// The PPU renders DMG shades with an LCD tint (`Ppu::DMG_SHADES` or
/// `Ppu::MGB_SHADES`) and renders pure white while the LCD is blanked after
/// being switched on; anything else is not a DMG shade and is kept as is so
/// it registers as a mismatch.
fn dmg_pixel_to_reference(pixel: u32) -> u32 {
    let pixel = pixel & 0x00FF_FFFF;
    if pixel == 0x00FF_FFFF {
        return REF_DMG_GRAYS[0];
    }
    for shades in [Ppu::DMG_SHADES, Ppu::MGB_SHADES] {
        if let Some(i) = shades.iter().position(|&s| s == pixel) {
            return REF_DMG_GRAYS[i];
        }
    }
    pixel | 0xFF00_0000
}

/// Reference screenshot suffixes in order of preference for a model.
fn reference_suffixes(model: GbModel) -> &'static [&'static str] {
    if model.is_cgb() {
        &["_cgb_c.png", "_cgb_d.png"]
    } else {
        &["_dmg_blob.png", "_dmg_b.png"]
    }
}

impl TestHarness for TearoomHarness {
    fn name(&self) -> &str {
        "Tearoom (screenshot comparison after LD B,B breakpoint)"
    }

    fn run_test(&self, path: &Path, verbose: bool) -> TestResult {
        let Ok(rom) = fs::read(path) else {
            return TestResult::Err;
        };
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let dir = path.parent().unwrap_or(Path::new("."));

        // ROMs with a -C suffix are CGB-only. The rest are DMG ROMs with
        // expected screenshots for DMG and for CGB compatibility mode; they
        // run on DMG unless a model is forced (e.g. `--model cgb`).
        let model = self.force_model.unwrap_or(if stem.ends_with("-C") {
            GbModel::Cgb
        } else {
            GbModel::Dmg
        });

        let Some(ref_path) = reference_suffixes(model)
            .iter()
            .map(|suffix| dir.join(format!("{stem}{suffix}")))
            .find(|p| p.exists())
        else {
            if verbose {
                eprintln!("  no reference image for {model:?}");
            }
            return TestResult::Skip;
        };

        let mut emu = make_emu(rom, None, model);
        let dmg_compat = model.is_cgb() && emu.bus().ppu.dmg_compat;
        if dmg_compat {
            // No boot ROM runs, so install the palettes it would have chosen.
            let ppu = &mut emu.bus_mut().ppu;
            ppu.dmg_bg_ref = CGB_COMPAT_BG;
            ppu.dmg_obj_ref = [CGB_COMPAT_OBJ, CGB_COMPAT_OBJ];
            let (bgp, obp0, obp1) = (ppu.bgp, ppu.obp0, ppu.obp1);
            ppu.sync_dmg_palette_to_cgb(bgp, false, 0);
            ppu.sync_dmg_palette_to_cgb(obp0, true, 0);
            ppu.sync_dmg_palette_to_cgb(obp1, true, 1);
        }

        if emu.run_until_breakpoint(300).is_none() {
            return TestResult::Timeout;
        }

        let Ok(ref_img) = image::open(&ref_path).map(|img| img.to_rgb8()) else {
            if verbose {
                eprintln!("  failed to load reference: {}", ref_path.display());
            }
            return TestResult::Err;
        };

        if ref_img.width() != GB_FB_WIDTH as u32 || ref_img.height() != GB_FB_HEIGHT as u32 {
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
        let mut mismatches = 0u32;
        let mut first_mismatch = None;
        for y in 0..GB_FB_HEIGHT {
            for x in 0..GB_FB_WIDTH {
                let pixel = fb[y * GB_FB_WIDTH + x] & 0x00FF_FFFF;
                // CGB output (native and compatibility mode) already uses the
                // reference RGB555 expansion, so it compares directly.
                let ours = if model.is_cgb() {
                    pixel
                } else {
                    dmg_pixel_to_reference(pixel)
                };
                let r = ref_img.get_pixel(x as u32, y as u32);
                let expected = (r[0] as u32) << 16 | (r[1] as u32) << 8 | r[2] as u32;
                if ours != expected {
                    mismatches += 1;
                    first_mismatch.get_or_insert((x, y, ours, expected));
                }
            }
        }

        if mismatches == 0 {
            TestResult::Pass
        } else {
            if verbose {
                let (x, y, ours, expected) = first_mismatch.unwrap();
                eprintln!(
                    "  {} pixel mismatches vs {} (first at {},{}: {:06X} != {:06X})",
                    mismatches,
                    ref_path.file_name().unwrap().to_string_lossy(),
                    x,
                    y,
                    ours & 0x00FF_FFFF,
                    expected
                );
            }
            TestResult::Fail
        }
    }
}
