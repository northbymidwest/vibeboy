/// Audio Processing Unit — stub implementation.
/// Handles I/O register reads/writes correctly; produces no actual audio.
pub struct Apu {
    regs: [u8; 0x30],
}

impl Apu {
    pub fn new() -> Self {
        let mut apu = Apu { regs: [0u8; 0x30] };
        // Post-boot register state
        apu.regs[0x10] = 0x80; // NR10
        apu.regs[0x11] = 0xBF; // NR11
        apu.regs[0x12] = 0xF3; // NR12
        apu.regs[0x13] = 0xFF; // NR13
        apu.regs[0x14] = 0xBF; // NR14
        apu.regs[0x16] = 0x3F; // NR21
        apu.regs[0x17] = 0x00; // NR22
        apu.regs[0x19] = 0xBF; // NR24
        apu.regs[0x1A] = 0x7F; // NR30
        apu.regs[0x1B] = 0xFF; // NR31
        apu.regs[0x1C] = 0x9F; // NR32
        apu.regs[0x1E] = 0xBF; // NR34
        apu.regs[0x20] = 0xFF; // NR41
        apu.regs[0x23] = 0xBF; // NR44
        apu.regs[0x24] = 0x77; // NR50
        apu.regs[0x25] = 0xF3; // NR51
        apu.regs[0x26] = 0xF1; // NR52
        apu
    }

    pub fn read(&self, addr: u16) -> u8 {
        let idx = (addr - 0xFF10) as usize;
        if idx < self.regs.len() {
            self.regs[idx]
        } else {
            0xFF
        }
    }

    pub fn write(&mut self, addr: u16, val: u8) {
        let idx = (addr - 0xFF10) as usize;
        if idx < self.regs.len() {
            self.regs[idx] = val;
        }
    }

    pub fn step(&mut self, _cycles: u32) {
        // No audio output yet
    }
}
