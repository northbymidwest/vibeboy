use crate::bus::Bus;
use crate::cpu::Cpu;
use crate::joypad::{
    BTN_A, BTN_B, BTN_DOWN, BTN_LEFT, BTN_RIGHT, BTN_SELECT, BTN_START, BTN_UP,
};

/// T-cycles per frame at normal speed (70224 = 456 × 154).
pub const CYCLES_PER_FRAME: u32 = 70_224;

pub struct Emulator {
    pub cpu: Cpu,
    pub bus: Bus,
}

impl Emulator {
    pub fn new(rom: Vec<u8>) -> Self {
        Emulator {
            cpu: Cpu::new(),
            bus: Bus::new(rom),
        }
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
        let cpu_cycles = self.cpu.step(&mut self.bus);

        // In double-speed mode the CPU runs twice as fast but PPU/timer run at
        // the same rate, so we halve the effective cycle count for bus tick.
        let bus_cycles = if self.bus.double_speed {
            cpu_cycles / 2
        } else {
            cpu_cycles
        };

        self.bus.tick(bus_cycles);
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
