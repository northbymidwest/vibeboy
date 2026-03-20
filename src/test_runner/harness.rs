use std::fmt;
use std::path::Path;

use crate::util::collect_roms;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestResult {
    Pass,
    Fail,
    Timeout,
    Skip,
    Err,
}

impl fmt::Display for TestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TestResult::Pass => write!(f, "PASS"),
            TestResult::Fail => write!(f, "FAIL"),
            TestResult::Timeout => write!(f, "TIMEOUT"),
            TestResult::Skip => write!(f, "SKIP"),
            TestResult::Err => write!(f, "ERR"),
        }
    }
}

pub trait TestHarness {
    fn name(&self) -> &str;
    fn run_test(&self, path: &Path, verbose: bool) -> TestResult;
}

pub fn run_tests(path: &Path, harness: &dyn TestHarness, verbose: bool, quiet: bool) {
    let mut roms = Vec::new();
    collect_roms(path, &mut roms);
    roms.sort();

    if !quiet {
        eprintln!("{} mode", harness.name());
    }

    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut timeout = 0usize;
    let mut skipped = 0usize;

    for rom in &roms {
        let result = harness.run_test(rom, verbose);
        let label = rom.strip_prefix(path).unwrap_or(rom).display().to_string();
        match result {
            TestResult::Skip => {
                skipped += 1;
                continue;
            }
            _ => {}
        }
        if !quiet {
            println!("{:<12} {}", result, label);
        }
        match result {
            TestResult::Pass => passed += 1,
            TestResult::Fail => failed += 1,
            TestResult::Timeout => timeout += 1,
            _ => {}
        }
    }

    print!(
        "\n--- {} passed, {} failed, {} timeout",
        passed, failed, timeout
    );
    if skipped > 0 {
        print!(", {} skipped", skipped);
    }
    println!(" ({} total) ---", passed + failed + timeout);
}
