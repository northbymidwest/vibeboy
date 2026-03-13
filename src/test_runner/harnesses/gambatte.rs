use std::fs;
use std::path::Path;

use crate::emulator::Emulator;
use crate::model::GbModel;
use crate::test_runner::harness::{TestHarness, TestResult};

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

pub struct GambatteHarness {
    pub force_model: Option<GbModel>,
}

impl TestHarness for GambatteHarness {
    fn name(&self) -> &str {
        "Gambatte (hex output comparison, 15 frames)"
    }

    fn run_test(&self, path: &Path, verbose: bool) -> TestResult {
        let rom = match fs::read(path) {
            Ok(r) => r,
            Err(_) => return TestResult::Err,
        };

        let (expected_hex, is_dmg, is_cgb) = match parse_gambatte_test(path) {
            Some(v) => v,
            None => return TestResult::Skip,
        };

        // Determine which model to test
        let model = if let Some(m) = self.force_model {
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
                None => return TestResult::Err,
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
                    if actual.is_empty() {
                        actual = "?".to_string();
                    }
                    eprintln!("  digit {}: expected={} got={}", i, c, actual);
                }
                return TestResult::Fail;
            }
        }

        TestResult::Pass
    }
}
