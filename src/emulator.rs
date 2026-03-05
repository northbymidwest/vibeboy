use crate::bus::Bus;
use crate::cpu::{Cpu, Registers};
use crate::joypad::{
    BTN_A, BTN_B, BTN_DOWN, BTN_LEFT, BTN_RIGHT, BTN_SELECT, BTN_START, BTN_UP,
};
use std::path::Path;

/// T-cycles per frame at normal speed (70224 = 456 × 154).
pub const CYCLES_PER_FRAME: u32 = 70_224;

pub struct Emulator {
    pub cpu: Cpu,
    pub bus: Bus,
}

impl Emulator {
    /// Create a new emulator. If `boot_rom` is Some, the CPU starts at PC=0x0000
    /// with hardware reset registers and executes the boot ROM. Otherwise the CPU
    /// starts at PC=0x0100 with post-boot GBC register values (no boot animation).
    pub fn new(rom: Vec<u8>, boot_rom: Option<Vec<u8>>, rom_path: Option<&Path>) -> Self {
        let has_boot = boot_rom.is_some();
        let mut cpu = Cpu::new();
        if has_boot {
            cpu.regs = Registers::reset();
        }
        Emulator {
            cpu,
            bus: Bus::new(rom, boot_rom, rom_path),
        }
    }

    /// Persist battery-backed cartridge RAM to disk.
    pub fn save(&self) {
        self.bus.save_to_disk();
    }

    /// Run until one full frame has been rendered (VBlank).
    pub fn step_frame(&mut self) {
        self.bus.clear_frame_ready();
        while !self.bus.frame_ready() {
            self.step();
        }
    }

    /// Execute one CPU instruction and advance bus components.
    fn step(&mut self) {
        self.cpu.step(&mut self.bus);
        // Ticking is now done inline by CPU during each M-cycle
    }

    /// Run until the Mooneye LD B,B breakpoint (opcode 0x40 at PC) or
    /// `max_frames` frames elapse. Returns Some((b,c,d,e,h,l)) on breakpoint.
    pub fn run_until_breakpoint(&mut self, max_frames: u32) -> Option<[u8; 6]> {
        let cycles_per_frame = 70_224u32;
        let mut cycles = 0u32;
        let limit = cycles_per_frame * max_frames;
        loop {
            // Check for Mooneye LD B,B breakpoint before executing
            if self.bus.read_byte(self.cpu.regs.pc) == 0x40 {
                let r = &self.cpu.regs;
                return Some([r.b, r.c, r.d, r.e, r.h, r.l]);
            }
            self.cpu.step(&mut self.bus);
            cycles += 4; // approximate; good enough for frame counting
            if cycles >= limit {
                return None;
            }
        }
    }

    /// Return the current frame buffer (160 × 144 pixels, 0x00RRGGBB).
    pub fn frame_buffer(&self) -> &[u32] {
        self.bus.ppu.frame_buffer()
    }

    // ── Input handling ────────────────────────────────────────────────────────

    pub fn set_button(&mut self, button: u8, pressed: bool) {
        self.bus.joypad.set_button(button, pressed);
    }

    // Expose button constants for main.rs
    pub const BTN_RIGHT:  u8 = BTN_RIGHT;
    pub const BTN_LEFT:   u8 = BTN_LEFT;
    pub const BTN_UP:     u8 = BTN_UP;
    pub const BTN_DOWN:   u8 = BTN_DOWN;
    pub const BTN_A:      u8 = BTN_A;
    pub const BTN_B:      u8 = BTN_B;
    pub const BTN_SELECT: u8 = BTN_SELECT;
    pub const BTN_START:  u8 = BTN_START;
}
