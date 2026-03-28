use std::fs;
use std::path::Path;

use vibeboy::model::GbModel;
use crate::harness::{TestHarness, TestResult};
use crate::util::make_emu;

pub struct GbMicrotestHarness {
    pub force_model: Option<GbModel>,
}

impl TestHarness for GbMicrotestHarness {
    fn name(&self) -> &str {
        "GBMicrotest (HRAM/VRAM result check)"
    }

    fn run_test(&self, path: &Path, verbose: bool) -> TestResult {
        let Ok(rom) = fs::read(path) else {
            return TestResult::Err;
        };
        let filename = path.file_name().and_then(|f| f.to_str()).unwrap_or("");

        // Most tests complete within 2 frames, but some need more:
        // - is_if_set_during_ime0 needs ~380ms (~23 frames)
        // - Tests with long delay loops may need 4+ frames
        let frames = if filename == "is_if_set_during_ime0.gb" { 30 } else { 4 };

        let model = self.force_model.unwrap_or(GbModel::Dmg);
        let rom_copy = rom.clone();
        let mut emu = make_emu(rom, None, model);
        for _ in 0..frames {
            emu.step_frame();
        }

        // Primary check: HRAM result at $FF80-$FF82
        let result = emu.bus_mut().read_byte(0xFF82);
        let actual = emu.bus_mut().read_byte(0xFF80);
        let expected = emu.bus_mut().read_byte(0xFF81);
        if result == 0x01 {
            return TestResult::Pass;
        }
        if result == 0xFF {
            if verbose {
                eprintln!("  actual=0x{:02X} expected=0x{:02X}", actual, expected);
            }
            return TestResult::Fail;
        }

        // Fallback: some gbmicrotest ROMs write their result byte to VRAM
        // $8000 instead of HRAM. Read the expected value from the ROM's
        // comparison instruction to determine pass/fail.
        let vram_result = emu.bus().ppu.vram[0][0];
        if vram_result != 0 {
            // Search ROM for the CP instruction that checks the result.
            // Pattern: F0 41/44 (LDH A,(STAT/LY)) ... FE xx (CP expected)
            // or: EA 00 80 (LD ($8000),A) preceded by the test logic.
            let expected_vram = self.find_expected_value(&rom_copy);
            if let Some(exp) = expected_vram {
                if vram_result == exp {
                    return TestResult::Pass;
                }
                if verbose {
                    eprintln!("  VRAM[0]=0x{:02X} expected=0x{:02X}", vram_result, exp);
                }
                return TestResult::Fail;
            }
            // No expected value found — report what we have
            if verbose {
                eprintln!("  VRAM[0]=0x{:02X} (no expected value found)", vram_result);
            }
            return TestResult::Timeout;
        }

        if verbose {
            eprintln!("  FF82=0x{:02X} (not 0x01 or 0xFF), VRAM[0]=0x00", result);
        }
        TestResult::Timeout
    }
}

impl GbMicrotestHarness {
    /// Search the ROM for a CP instruction that checks the VRAM result byte.
    /// These tests typically do: read a register → LD ($8000),A → JR done,
    /// with a CP xx nearby that holds the expected value.
    fn find_expected_value(&self, rom: &[u8]) -> Option<u8> {
        // Look for pattern: FE xx 28 (CP xx; JR Z) near EA 00 80 (LD ($8000),A)
        for i in 0..rom.len().saturating_sub(6) {
            if rom[i] == 0xEA && rom.get(i+1) == Some(&0x00) && rom.get(i+2) == Some(&0x80) {
                // Found LD ($8000),A. Search backwards for the CP instruction.
                for j in (i.saturating_sub(20)..i).rev() {
                    if rom[j] == 0xFE && j + 2 < rom.len() && rom[j+2] == 0x28 {
                        return Some(rom[j+1]);
                    }
                }
            }
        }
        None
    }
}
