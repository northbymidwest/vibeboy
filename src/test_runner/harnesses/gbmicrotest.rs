use std::fs;
use std::path::Path;

use crate::emulator::Emulator;
use crate::model::GbModel;
use crate::test_runner::harness::{TestHarness, TestResult};

pub struct GbMicrotestHarness {
    pub force_model: Option<GbModel>,
}

impl TestHarness for GbMicrotestHarness {
    fn name(&self) -> &str {
        "GBMicrotest (HRAM result check, 2 frames)"
    }

    fn run_test(&self, path: &Path, verbose: bool) -> TestResult {
        let Ok(rom) = fs::read(path) else {
            return TestResult::Err;
        };
        let model = self.force_model.unwrap_or(GbModel::Dmg);
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
            TestResult::Pass
        } else if result == 0xFF {
            if verbose {
                eprintln!("  actual=0x{:02X} expected=0x{:02X}", actual, expected);
            }
            TestResult::Fail
        } else {
            if verbose {
                eprintln!("  FF82=0x{:02X} (not 0x01 or 0xFF)", result);
            }
            TestResult::Timeout
        }
    }
}
