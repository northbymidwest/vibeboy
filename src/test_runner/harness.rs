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

/// Counts from one `run_tests` call. Skipped ROMs are not results and are
/// not part of `total()`.
#[derive(Debug, Default, Clone, Copy)]
pub struct Summary {
    pub passed: usize,
    pub failed: usize,
    pub timeout: usize,
    pub errors: usize,
    pub skipped: usize,
}

impl Summary {
    pub fn total(&self) -> usize {
        self.passed + self.failed + self.timeout + self.errors
    }

    /// True when every test that ran passed.
    pub fn all_passed(&self) -> bool {
        self.passed == self.total()
    }
}

pub fn run_tests(path: &Path, harness: &dyn TestHarness, verbose: bool, quiet: bool) -> Summary {
    let mut roms = Vec::new();
    collect_roms(path, &mut roms);
    roms.sort();

    if !quiet {
        eprintln!("{} mode", harness.name());
    }

    let mut summary = Summary::default();

    for rom in &roms {
        let result = harness.run_test(rom, verbose);
        let label = rom.strip_prefix(path).unwrap_or(rom).display().to_string();
        if result == TestResult::Skip {
            summary.skipped += 1;
            continue;
        }
        if !quiet {
            println!("{:<12} {}", result, label);
        }
        match result {
            TestResult::Pass => summary.passed += 1,
            TestResult::Fail => summary.failed += 1,
            TestResult::Timeout => summary.timeout += 1,
            TestResult::Err => summary.errors += 1,
            TestResult::Skip => {}
        }
    }

    print!(
        "\n--- {} passed, {} failed, {} timeout",
        summary.passed, summary.failed, summary.timeout
    );
    if summary.errors > 0 {
        print!(", {} error", summary.errors);
    }
    if summary.skipped > 0 {
        print!(", {} skipped", summary.skipped);
    }
    println!(" ({} total) ---", summary.total());
    summary
}
