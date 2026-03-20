use std::collections::VecDeque;
use std::fs;
use std::path::Path;

use vibeboy::emulator::Emulator;
use vibeboy::model::GbModel;
use crate::test_model::{detect_model_with_rom, load_boot_rom, resolve_boot_rom};
use crate::util::{make_emu, GB_FB_WIDTH, GB_FB_HEIGHT};

pub fn cmd_analyze(
    path: &Path,
    force_model: Option<GbModel>,
    boot: bool,
    bootrom: Option<&Path>,
    frames: u32,
) {
    let rom = fs::read(path).expect("Failed to read ROM");
    let model = force_model.unwrap_or(GbModel::Cgb);
    let br = resolve_boot_rom(boot, bootrom, model);
    let mut emu = make_emu(rom, br, model);

    for _ in 0..frames {
        emu.step_frame();
    }

    let fb = emu.frame_buffer();
    eprintln!("Frame {} analysis (right edge, columns 155-159):", frames);
    let mut yellow_count = 0;
    for y in 0..GB_FB_HEIGHT {
        let mut right_colors = Vec::new();
        for x in 155..GB_FB_WIDTH {
            let pixel = fb[y * GB_FB_WIDTH + x];
            right_colors.push(pixel);
            let r = (pixel >> 16) & 0xFF;
            let g = (pixel >> 8) & 0xFF;
            let b = pixel & 0xFF;
            if r > 200 && g > 150 && b < 100 {
                yellow_count += 1;
            }
        }
        if right_colors.iter().any(|&p| {
            let r = (p >> 16) & 0xFF;
            let g = (p >> 8) & 0xFF;
            let b = p & 0xFF;
            r > 200 && g > 150 && b < 100
        }) {
            eprintln!(
                "  LY={:3}: {:08X} {:08X} {:08X} {:08X} {:08X}  SCX={}",
                y, right_colors[0], right_colors[1], right_colors[2], right_colors[3],
                right_colors[4], emu.bus().ppu.scx
            );
        }
    }
    eprintln!("\nLeft edge sample (LY=72):");
    for x in 0..8 {
        let pixel = fb[72 * GB_FB_WIDTH + x];
        eprint!("{:08X} ", pixel);
    }
    eprintln!("\nRight edge sample (LY=72):");
    for x in 152..GB_FB_WIDTH {
        let pixel = fb[72 * GB_FB_WIDTH + x];
        eprint!("{:08X} ", pixel);
    }
    eprintln!(
        "\nTotal yellow pixels on right edge (cols 155-159): {}",
        yellow_count
    );
}

pub fn cmd_trace_timer(
    path: &Path,
    force_model: Option<GbModel>,
    boot: bool,
    bootrom: Option<&Path>,
) {
    let rom = fs::read(path).expect("Failed to read ROM");
    let model = force_model.unwrap_or_else(|| detect_model_with_rom(path, Some(&rom)));
    let br = resolve_boot_rom(boot, bootrom, model);
    let mut emu = make_emu(rom, br, model);
    if boot || bootrom.is_some() {
        let mut history: VecDeque<(u16, u8, u16, u16)> = VecDeque::new();
        loop {
            let pc = emu.cpu().regs.pc;
            let opcode = emu.bus_mut().read_byte(pc);
            let tc_before = emu.bus().timer.counter();
            if pc == 0x0100 && !emu.bus().boot_rom_active {
                break;
            }
            emu.cpu_step();
            let tc_after = emu.bus().timer.counter();
            history.push_back((pc, opcode, tc_before, tc_after));
            if history.len() > 30 {
                history.pop_front();
            }
        }
        eprintln!("Last {} instructions of boot ROM:", history.len());
        for (pc, opcode, tc_before, tc_after) in &history {
            let delta = tc_after.wrapping_sub(*tc_before);
            eprintln!(
                "  PC={:#06X} op={:#04X} timer: {:#06X} -> {:#06X} (+{}) DIV: {:02X} -> {:02X}",
                pc,
                opcode,
                tc_before,
                tc_after,
                delta,
                tc_before >> 8,
                tc_after >> 8
            );
        }
    }
    {
        let tc = emu.bus().timer.counter();
        eprintln!(
            "At PC=$0100: timer_counter={:#06X} DIV={:02X}",
            tc,
            tc >> 8
        );
    }
    for i in 0..20 {
        let pc = emu.cpu().regs.pc;
        let opcode = emu.bus_mut().read_byte(pc);
        let tc_before = emu.bus().timer.counter();
        emu.cpu_step();
        let tc_after = emu.bus().timer.counter();
        eprintln!(
            "  step {}: PC={:#06X} op={:#04X} timer: {:#06X} -> {:#06X} (DIV: {:02X} -> {:02X})",
            i,
            pc,
            opcode,
            tc_before,
            tc_after,
            tc_before >> 8,
            tc_after >> 8
        );
    }
}

pub fn cmd_calibrate(path: &Path) {
    let rom = fs::read(path).expect("Failed to read ROM");
    let models = [
        GbModel::Dmg0,
        GbModel::Dmg,
        GbModel::Mgb,
        GbModel::Sgb,
        GbModel::Sgb2,
        GbModel::Cgb,
        GbModel::Agb,
    ];
    for model in &models {
        if let Some(br) = load_boot_rom(*model) {
            let mut emu = Emulator::new(rom.clone(), Some(br), None, *model, None);
            emu.set_headless(true);
            for _ in 0..100_000_000u64 {
                if emu.cpu().regs.pc == 0x0100 && !emu.bus().boot_rom_active {
                    let div_val = emu.bus_mut().read_byte(0xFF04);
                    eprintln!(
                        "{:?}: LY={} dot={} mode={} total_ticks={} DIV={:02X} timer_counter={:#06X} regs=A:{:02X} F:{:02X} B:{:02X} C:{:02X} D:{:02X} E:{:02X} H:{:02X} L:{:02X}",
                        model,
                        emu.bus().ppu.ly,
                        emu.bus().ppu.dot,
                        emu.bus().ppu.stat & 0x03,
                        emu.bus().ppu.total_ticks,
                        div_val,
                        emu.bus().timer.counter(),
                        emu.cpu().regs.a,
                        emu.cpu().regs.f,
                        emu.cpu().regs.b,
                        emu.cpu().regs.c,
                        emu.cpu().regs.d,
                        emu.cpu().regs.e,
                        emu.cpu().regs.h,
                        emu.cpu().regs.l
                    );
                    break;
                }
                emu.cpu_step();
            }
        } else {
            eprintln!("{:?}: no boot ROM available", model);
        }
    }
}
