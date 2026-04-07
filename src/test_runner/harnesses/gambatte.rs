use std::fs;
use std::path::Path;

use vibeboy::model::GbModel;
use crate::harness::{TestHarness, TestResult};
use crate::util::{make_emu, GB_FB_WIDTH};

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
/// Uses relative brightness instead of absolute color values so it works
/// with any palette (grayscale, DMG green, CGB colors).
fn gambatte_tile_matches(fb: &[u32], tile_x: usize, digit: usize) -> bool {
    let pattern = &GAMBATTE_DIGITS[digit];
    // Determine the darkest and lightest pixel values in the tile
    // to classify pixels as "black" or "white" relative to the palette.
    let mut darkest = u32::MAX;
    let mut lightest = 0u32;
    for y in 0..8 {
        for x in 0..8 {
            let pixel = fb[y * GB_FB_WIDTH + tile_x * 8 + x] & 0x00FFFFFF;
            let luma = (pixel >> 16) & 0xFF; // use red channel as brightness proxy
            if luma < (darkest & 0xFF) { darkest = luma; }
            if luma > (lightest & 0xFF) { lightest = luma; }
        }
    }
    if darkest == lightest { return false; } // uniform tile — can't match any digit
    let threshold = (darkest + lightest) / 2;

    for y in 0..8 {
        for x in 0..8 {
            let pixel = fb[y * GB_FB_WIDTH + tile_x * 8 + x] & 0x00FFFFFF;
            let luma = (pixel >> 16) & 0xFF;
            let expected_black = (pattern[y] >> (7 - x)) & 1 == 1;
            let is_black = luma <= threshold;
            if expected_black != is_black {
                return false;
            }
        }
    }
    true
}

/// Parse gambatte test: extract expected hex from filename, determine model(s).
/// Returns (expected_hex, is_dmg, is_cgb).
/// Parsed expected outputs from a gambatte test ROM filename.
/// Possible formats:
///   - `..._dmg08_cgb04c_outHEX.gbc`: same expected hex for DMG and CGB
///   - `..._dmg08_outHEX_cgb04c_outHEX.gbc`: separate DMG and CGB outputs
///   - `..._dmg08_outHEX.gbc`: DMG-only
///   - `..._cgb04c_outHEX.gbc`: CGB-only
///   - `..._outHEX.gbc`: model unspecified, default CGB
struct GambatteExpected {
    dmg_hex: Option<String>,
    cgb_hex: Option<String>,
}

fn is_hex(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_hexdigit())
}

fn parse_gambatte_test(path: &Path) -> Option<GambatteExpected> {
    let stem = path.file_stem()?.to_str()?;

    let dmg_hex;
    let cgb_hex;

    if let Some(pos) = stem.find("dmg08_cgb04c_out") {
        // Shared output: both DMG and CGB expect the same hex.
        let hex = &stem[pos + 16..];
        if !is_hex(hex) || hex.starts_with("audio") {
            return None;
        }
        dmg_hex = Some(hex.to_uppercase());
        cgb_hex = Some(hex.to_uppercase());
    } else if let Some(d_pos) = stem.find("dmg08_out") {
        // DMG output present. May also have separate CGB output.
        let after_dmg = &stem[d_pos + 9..];
        // DMG hex runs until '_' or end of stem.
        let dmg_end = after_dmg.find('_').unwrap_or(after_dmg.len());
        let dh = &after_dmg[..dmg_end];
        if !is_hex(dh) || dh.starts_with("audio") {
            return None;
        }
        dmg_hex = Some(dh.to_uppercase());
        cgb_hex = if let Some(c_pos) = stem.find("cgb04c_out") {
            let ch = &stem[c_pos + 10..];
            let ch_end = ch.find('_').unwrap_or(ch.len());
            let ch = &ch[..ch_end];
            if is_hex(ch) { Some(ch.to_uppercase()) } else { None }
        } else {
            None
        };
    } else if let Some(c_pos) = stem.find("cgb04c_out") {
        let hex = &stem[c_pos + 10..];
        let hex_end = hex.find('_').unwrap_or(hex.len());
        let hex = &hex[..hex_end];
        if !is_hex(hex) || hex.starts_with("audio") {
            return None;
        }
        dmg_hex = None;
        cgb_hex = Some(hex.to_uppercase());
    } else if let Some(pos) = stem.rfind("_out") {
        let hex = &stem[pos + 4..];
        let hex_end = hex.find('_').unwrap_or(hex.len());
        let hex = &hex[..hex_end];
        if !is_hex(hex) || hex.starts_with("audio") {
            return None;
        }
        dmg_hex = None;
        cgb_hex = Some(hex.to_uppercase());
    } else {
        return None;
    }

    if dmg_hex.is_none() && cgb_hex.is_none() {
        return None;
    }

    Some(GambatteExpected { dmg_hex, cgb_hex })
}

pub struct GambatteHarness {
    pub force_model: Option<GbModel>,
}

impl TestHarness for GambatteHarness {
    fn name(&self) -> &str {
        "Gambatte (hex output comparison, 15 frames)"
    }

    fn run_test(&self, path: &Path, verbose: bool) -> TestResult {
        let Ok(rom) = fs::read(path) else {
            return TestResult::Err;
        };

        let Some(expected) = parse_gambatte_test(path) else {
            return TestResult::Skip;
        };

        // Determine which model variants to run.
        let mut runs: Vec<(GbModel, &str)> = Vec::new();
        if let Some(m) = self.force_model {
            // Forced model: use whichever expected value applies, prefer the
            // matching one. If neither applies, skip.
            let hex = match m {
                GbModel::Dmg | GbModel::Dmg0 | GbModel::Mgb => expected.dmg_hex.as_deref(),
                _ => expected.cgb_hex.as_deref(),
            };
            if let Some(h) = hex {
                runs.push((m, h));
            } else {
                return TestResult::Skip;
            }
        } else {
            if let Some(ref h) = expected.dmg_hex {
                runs.push((GbModel::Dmg, h));
            }
            if let Some(ref h) = expected.cgb_hex {
                runs.push((GbModel::Cgb, h));
            }
        }

        if runs.is_empty() {
            return TestResult::Skip;
        }

        // Run each variant. The ROM passes only if all variants pass.
        for (model, expected_hex) in &runs {
            let mut emu = make_emu(rom.clone(), None, *model);
            for _ in 0..15 {
                emu.step_frame();
            }
            let fb = emu.frame_buffer();

            for (i, c) in expected_hex.chars().enumerate() {
                let Some(digit) = gambatte_digit_index(c) else {
                    return TestResult::Err;
                };
                if !gambatte_tile_matches(fb, i, digit) {
                    if verbose {
                        let mut actual = String::new();
                        for d in 0..16usize {
                            if gambatte_tile_matches(fb, i, d) {
                                actual = format!("{:X}", d);
                                break;
                            }
                        }
                        if actual.is_empty() {
                            actual = "?".to_string();
                        }
                        eprintln!(
                            "  [{:?}] digit {}: expected={} got={}",
                            model, i, c, actual
                        );
                    }
                    return TestResult::Fail;
                }
            }
        }

        TestResult::Pass
    }
}
