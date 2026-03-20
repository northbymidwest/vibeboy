use std::fs;
use std::path::Path;

use vibeboy::model::GbModel;
use crate::harness::{TestHarness, TestResult};
use crate::test_model::detect_model_with_rom;
use crate::util::make_emu;

pub struct MooneyeHarness {
    pub force_model: Option<GbModel>,
    pub boot_rom: Option<Vec<u8>>,
}

/// Mooneye pass: B=3,C=5,D=8,E=13,H=21,L=34 (Fibonacci)
fn mooneye_passed(regs: [u8; 6]) -> bool {
    regs == [3, 5, 8, 13, 21, 34]
}

impl TestHarness for MooneyeHarness {
    fn name(&self) -> &str {
        "Mooneye (breakpoint + Fibonacci check)"
    }

    fn run_test(&self, path: &Path, verbose: bool) -> TestResult {
        let Ok(rom) = fs::read(path) else {
            return TestResult::Err;
        };
        let model = self
            .force_model
            .unwrap_or_else(|| detect_model_with_rom(path, Some(&rom)));
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        // Use model-specific boot ROM variant (e.g. cgb0 for -cgb0 suffix)
        let br = if self.boot_rom.is_some() {
            if stem.contains("cgb0") {
                fs::read("bootroms/cgb0_boot.bin")
                    .ok()
                    .or_else(|| self.boot_rom.clone())
            } else {
                self.boot_rom.clone()
            }
        } else {
            None
        };
        let mut emu = make_emu(rom, br, model);
        match emu.run_until_breakpoint(300) {
            Some(regs) => {
                if mooneye_passed(regs) {
                    TestResult::Pass
                } else {
                    if verbose {
                        eprintln!(
                            "  regs: B={:02X} C={:02X} D={:02X} E={:02X} H={:02X} L={:02X}",
                            regs[0], regs[1], regs[2], regs[3], regs[4], regs[5]
                        );
                        if stem.contains("channel_") || stem.contains("div_") {
                            eprint!("  actual   @C000:");
                            for i in 0..128u16 {
                                if i % 16 == 0 && i > 0 {
                                    eprint!("\n                ");
                                }
                                eprint!(" {:02X}", emu.bus_mut().read_byte(0xC000 + i));
                            }
                            eprintln!();
                        }
                        if stem.contains("boot_hwio") {
                            // CGB expected values from boot_hwio-C.s
                            let expected: &[(u16, u8)] = &[
                                (0xFF00, 0xFF),
                                (0xFF01, 0x00),
                                (0xFF02, 0x7E),
                                (0xFF03, 0xFF),
                                (0xFF05, 0x00),
                                (0xFF06, 0x00),
                                (0xFF07, 0xF8),
                                (0xFF08, 0xFF),
                                (0xFF09, 0xFF),
                                (0xFF0A, 0xFF),
                                (0xFF0B, 0xFF),
                                (0xFF0C, 0xFF),
                                (0xFF0D, 0xFF),
                                (0xFF0E, 0xFF),
                                (0xFF0F, 0xE1),
                                (0xFF10, 0x80),
                                (0xFF11, 0xBF),
                                (0xFF12, 0xF3),
                                (0xFF13, 0xFF),
                                (0xFF14, 0xBF),
                                (0xFF15, 0xFF),
                                (0xFF16, 0x3F),
                                (0xFF17, 0x00),
                                (0xFF18, 0xFF),
                                (0xFF19, 0xBF),
                                (0xFF1A, 0x7F),
                                (0xFF1B, 0xFF),
                                (0xFF1C, 0x9F),
                                (0xFF1D, 0xFF),
                                (0xFF1E, 0xBF),
                                (0xFF1F, 0xFF),
                                (0xFF20, 0xFF),
                                (0xFF21, 0x00),
                                (0xFF22, 0x00),
                                (0xFF23, 0xBF),
                                (0xFF24, 0x77),
                                (0xFF25, 0xF3),
                                (0xFF26, 0xF1),
                                (0xFF27, 0xFF),
                                (0xFF42, 0x00),
                                (0xFF43, 0x00),
                                (0xFF45, 0x00),
                                (0xFF47, 0xFC),
                                (0xFF4A, 0x00),
                                (0xFF4B, 0x00),
                                // CGB-specific registers
                                (0xFF68, 0xC8),
                                (0xFF6A, 0xD0),
                                (0xFF72, 0x00),
                                (0xFF73, 0x00),
                                (0xFF75, 0x8F),
                            ];
                            for &(addr, exp) in expected {
                                let got = emu.bus_mut().read_byte(addr);
                                if got != exp {
                                    eprintln!(
                                        "  boot_hwio mismatch: ${:04X} expected={:02X} got={:02X}",
                                        addr, exp, got
                                    );
                                }
                            }
                        }
                    }
                    TestResult::Fail
                }
            }
            None => TestResult::Timeout,
        }
    }
}
