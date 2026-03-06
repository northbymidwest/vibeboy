mod apu;
mod bus;
mod cartridge;
mod cpu;
mod emulator;
mod joypad;
mod model;
mod ppu;
mod printer;
mod serial;
mod sgb;
mod snapshot;
mod snes;
mod timer;

use emulator::Emulator;
use model::GbModel;
use std::fs;
use std::path::Path;

/// Detect hardware model from test filename suffix.
fn detect_model(path: &Path) -> GbModel {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    // Check longer suffixes first to avoid partial matches
    if stem.ends_with("-dmgABCmgb") || stem.ends_with("-dmgABC") {
        GbModel::Dmg
    } else if stem.ends_with("-dmg0") {
        GbModel::Dmg0
    } else if stem.ends_with("-mgb") {
        GbModel::Mgb
    } else if stem.ends_with("-sgb2") {
        GbModel::Sgb2
    } else if stem.ends_with("-sgb") || stem.ends_with("-S") {
        GbModel::Sgb
    } else if stem.ends_with("-GS") || stem.ends_with("-G") {
        GbModel::Dmg
    } else {
        GbModel::Cgb
    }
}


/// Mooneye pass: B=3,C=5,D=8,E=13,H=21,L=34 (Fibonacci)
fn mooneye_passed(regs: [u8; 6]) -> bool {
    regs == [3, 5, 8, 13, 21, 34]
}

/// Load boot ROM for the given model from bootroms/ directory.
fn load_boot_rom(model: GbModel) -> Option<Vec<u8>> {
    let candidates: &[&str] = match model {
        GbModel::Dmg0 => &["bootroms/dmg0_boot.bin", "bootroms/dmg_boot.bin"],
        GbModel::Dmg  => &["bootroms/dmg_boot.bin"],
        GbModel::Mgb  => &["bootroms/mgb_boot.bin", "bootroms/dmg_boot.bin"],
        GbModel::Sgb  => &["bootroms/sgb_boot.bin"],
        GbModel::Sgb2 => &["bootroms/sgb2_boot.bin"],
        GbModel::Cgb  => &["bootroms/cgb_boot.bin", "gbc_bios.bin"],
    };
    for path in candidates {
        if let Ok(data) = fs::read(path) {
            return Some(data);
        }
    }
    None
}

fn run_test_mooneye(path: &Path, verbose: bool, force_model: Option<GbModel>, use_boot_rom: bool) -> &'static str {
    let rom = match fs::read(path) {
        Ok(r) => r,
        Err(_) => return "ERR",
    };
    let model = force_model.unwrap_or_else(|| detect_model(path));
    let br = if use_boot_rom { load_boot_rom(model) } else { None };
    let mut emu = Emulator::new(rom, br, None, model, None);
    match emu.run_until_breakpoint(300) {
        Some(regs) => {
            if mooneye_passed(regs) {
                "PASS"
            } else {
                if verbose {
                    eprintln!("  regs: B={:02X} C={:02X} D={:02X} E={:02X} H={:02X} L={:02X}",
                        regs[0], regs[1], regs[2], regs[3], regs[4], regs[5]);
                }
                "FAIL"
            }
        }
        None => "TIMEOUT",
    }
}

fn run_test_blargg(path: &Path, verbose: bool) -> &'static str {
    let rom = match fs::read(path) {
        Ok(r) => r,
        Err(_) => return "ERR",
    };
    let model = detect_model(path);
    let br = load_boot_rom(model);
    let mut emu = Emulator::new(rom, br, None, model, None);
    let output = emu.run_until_serial_result(1800); // ~30 seconds at 60fps
    if verbose && !output.is_empty() {
        // Print serial output indented
        for line in output.lines() {
            eprintln!("  {}", line);
        }
    }
    if output.contains("Passed") {
        "PASS"
    } else if output.contains("Failed") {
        "FAIL"
    } else {
        "TIMEOUT"
    }
}

fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: test_runner <dir_or_glob> [--blargg]");
        std::process::exit(1);
    }

    let root = Path::new(&args[1]);
    let blargg_mode = args.iter().any(|a| a == "--blargg");
    let use_boot_rom = args.iter().any(|a| a == "--boot");
    let force_model = if args.iter().any(|a| a == "--dmg") {
        Some(GbModel::Dmg)
    } else if args.iter().any(|a| a == "--dmg0") {
        Some(GbModel::Dmg0)
    } else if args.iter().any(|a| a == "--mgb") {
        Some(GbModel::Mgb)
    } else if args.iter().any(|a| a == "--sgb") {
        Some(GbModel::Sgb)
    } else if args.iter().any(|a| a == "--sgb2") {
        Some(GbModel::Sgb2)
    } else if args.iter().any(|a| a == "--cgb") {
        Some(GbModel::Cgb)
    } else {
        None
    };

    let mut roms: Vec<std::path::PathBuf> = Vec::new();
    collect_roms(root, &mut roms);
    roms.sort();

    if blargg_mode {
        eprintln!("Blargg mode (serial output detection)");
    }

    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut timeout = 0usize;

    for rom in &roms {
        let result = if blargg_mode {
            run_test_blargg(rom, true)
        } else {
            run_test_mooneye(rom, true, force_model, use_boot_rom)
        };
        let label = rom.strip_prefix(root).unwrap_or(rom).display().to_string();
        println!("{:<12} {}", result, label);
        match result {
            "PASS"    => passed += 1,
            "FAIL"    => failed += 1,
            "TIMEOUT" => timeout += 1,
            _ => {}
        }
    }

    println!("\n--- {} passed, {} failed, {} timeout ({} total) ---",
        passed, failed, timeout, roms.len());
}

fn collect_roms(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    if dir.is_file() {
        let ext = dir.extension().and_then(|e| e.to_str());
        if ext == Some("gb") || ext == Some("gbc") {
            out.push(dir.to_path_buf());
        }
        return;
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            collect_roms(&entry.path(), out);
        }
    }
}
