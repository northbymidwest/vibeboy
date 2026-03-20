use std::fs;
use std::path::Path;

use vibeboy::model::GbModel;
use crate::harness::{TestHarness, TestResult};
use crate::test_model::detect_model_with_rom;
use crate::util::make_emu;

pub struct BlarggHarness {
    pub force_model: Option<GbModel>,
}

impl TestHarness for BlarggHarness {
    fn name(&self) -> &str {
        "Blargg (serial output detection)"
    }

    fn run_test(&self, path: &Path, verbose: bool) -> TestResult {
        let Ok(rom) = fs::read(path) else {
            return TestResult::Err;
        };
        let model = self
            .force_model
            .unwrap_or_else(|| detect_model_with_rom(path, Some(&rom)));
        let mut emu = make_emu(rom, None, model);
        let output = emu.run_until_serial_result(6000);
        if verbose && !output.is_empty() {
            for line in output.lines() {
                eprintln!("  {}", line);
            }
        }
        if output.contains("Passed") {
            TestResult::Pass
        } else if output.contains("Failed") {
            TestResult::Fail
        } else {
            TestResult::Timeout
        }
    }
}
