mod apu;
mod bus;
mod cartridge;
mod cpu;
mod emulator;
mod joypad;
mod ppu;
mod timer;

use emulator::Emulator;
use std::fs;
use std::path::Path;

/// Mooneye pass: B=3,C=5,D=8,E=13,H=21,L=34 (Fibonacci)
fn mooneye_passed(regs: [u8; 6]) -> bool {
    regs == [3, 5, 8, 13, 21, 34]
}

fn run_test_mooneye(path: &Path, verbose: bool, boot_rom: &Option<Vec<u8>>) -> &'static str {
    let rom = match fs::read(path) {
        Ok(r) => r,
        Err(_) => return "ERR",
    };
    let mut emu = Emulator::new(rom, boot_rom.clone(), None);
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

fn run_test_blargg(path: &Path, verbose: bool, boot_rom: &Option<Vec<u8>>) -> &'static str {
    let rom = match fs::read(path) {
        Ok(r) => r,
        Err(_) => return "ERR",
    };
    let mut emu = Emulator::new(rom, boot_rom.clone(), None);
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
        eprintln!("Usage: test_runner <dir_or_glob> [--boot-rom <path>] [--blargg]");
        std::process::exit(1);
    }

    let root = Path::new(&args[1]);
    let blargg_mode = args.iter().any(|a| a == "--blargg");

    // Parse optional --boot-rom argument
    let boot_rom = if let Some(pos) = args.iter().position(|a| a == "--boot-rom") {
        args.get(pos + 1).and_then(|p| fs::read(p).ok())
    } else {
        // Auto-detect boot ROM in crate root
        fs::read("gbc_bios.bin").ok()
    };

    let mut roms: Vec<std::path::PathBuf> = Vec::new();
    collect_roms(root, &mut roms);
    roms.sort();

    if boot_rom.is_some() {
        eprintln!("Using GBC boot ROM");
    }
    if blargg_mode {
        eprintln!("Blargg mode (serial output detection)");
    }

    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut timeout = 0usize;

    for rom in &roms {
        let result = if blargg_mode {
            run_test_blargg(rom, true, &boot_rom)
        } else {
            run_test_mooneye(rom, true, &boot_rom)
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
