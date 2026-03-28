/// WDC 65C816 CPU core for SGB SNES-side emulation.
///
/// Implements all 256 opcodes in both emulation and native modes,
/// 24 addressing modes, NMI/IRQ handling, and WAI instruction.

/// Status register flag bits.
const FLAG_C: u8 = 0x01; // Carry
const FLAG_Z: u8 = 0x02; // Zero
const FLAG_I: u8 = 0x04; // IRQ disable
const FLAG_D: u8 = 0x08; // Decimal
const FLAG_X: u8 = 0x10; // Index register size (0=16, 1=8) [native only]
const FLAG_M: u8 = 0x20; // Memory/Accumulator size (0=16, 1=8) [native only]
const FLAG_V: u8 = 0x40; // Overflow
const FLAG_N: u8 = 0x80; // Negative

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Cpu65816 {
    pub a: u16,   // Accumulator (full 16-bit; use low 8 when M=1)
    pub x: u16,   // Index X
    pub y: u16,   // Index Y
    pub s: u16,   // Stack pointer
    pub pc: u16,  // Program counter
    pub pbr: u8,  // Program bank register
    pub dbr: u8,  // Data bank register
    pub d: u16,   // Direct page register
    pub p: u8,    // Processor status
    pub emulation: bool,  // Emulation mode (starts true after RESET)
    pub waiting: bool,    // WAI state — waiting for interrupt
    pub stopped: bool,    // STP state
    pub nmi_pending: bool,
    nmi_line: bool,       // Current NMI line level (edge-triggered)
    nmi_prev: bool,       // Previous NMI line level
    pub irq_line: bool,   // IRQ line level (active low)
    pub cycles: u64,      // Master cycles consumed
}

impl Cpu65816 {
    pub fn new() -> Self {
        Cpu65816 {
            a: 0, x: 0, y: 0,
            s: 0x01FF,
            pc: 0, pbr: 0, dbr: 0, d: 0,
            p: FLAG_M | FLAG_X | FLAG_I,
            emulation: true,
            waiting: false,
            stopped: false,
            nmi_pending: false,
            nmi_line: false,
            nmi_prev: false,
            irq_line: false,
            cycles: 0,
        }
    }

    /// Reset the CPU. Reads reset vector from bus.
    pub fn reset(&mut self, read: &dyn Fn(u32) -> u8) {
        self.emulation = true;
        self.p = FLAG_M | FLAG_X | FLAG_I;
        self.s = 0x01FF;
        self.pbr = 0;
        self.dbr = 0;
        self.d = 0;
        self.waiting = false;
        self.stopped = false;
        let lo = read(0x00FFFC) as u16;
        let hi = read(0x00FFFD) as u16;
        self.pc = lo | (hi << 8);
        self.cycles = 0;
    }

    #[inline]
    fn a_is_8(&self) -> bool { self.emulation || self.p & FLAG_M != 0 }
    #[inline]
    fn x_is_8(&self) -> bool { self.emulation || self.p & FLAG_X != 0 }
    #[inline]
    fn al(&self) -> u8 { self.a as u8 }
    #[inline]
    fn ah(&self) -> u8 { (self.a >> 8) as u8 }
    #[inline]
    fn set_al(&mut self, v: u8) { self.a = (self.a & 0xFF00) | v as u16; }

    fn set_nz8(&mut self, v: u8) {
        self.p = (self.p & !(FLAG_N | FLAG_Z))
            | if v == 0 { FLAG_Z } else { 0 }
            | (v & 0x80);
    }
    fn set_nz16(&mut self, v: u16) {
        self.p = (self.p & !(FLAG_N | FLAG_Z))
            | if v == 0 { FLAG_Z } else { 0 }
            | ((v >> 8) as u8 & 0x80);
    }

    // Stack operations
    fn push8(&mut self, v: u8, write: &mut dyn FnMut(u32, u8)) {
        write(self.s as u32, v);
        if self.emulation {
            let lo = self.s as u8;
            self.s = 0x0100 | lo.wrapping_sub(1) as u16;
        } else {
            self.s = self.s.wrapping_sub(1);
        }
    }
    fn push16(&mut self, v: u16, write: &mut dyn FnMut(u32, u8)) {
        self.push8((v >> 8) as u8, write);
        self.push8(v as u8, write);
    }
    fn pull8(&mut self, read: &dyn Fn(u32) -> u8) -> u8 {
        if self.emulation {
            let lo = self.s as u8;
            self.s = 0x0100 | lo.wrapping_add(1) as u16;
        } else {
            self.s = self.s.wrapping_add(1);
        }
        read(self.s as u32)
    }
    fn pull16(&mut self, read: &dyn Fn(u32) -> u8) -> u16 {
        let lo = self.pull8(read) as u16;
        let hi = self.pull8(read) as u16;
        lo | (hi << 8)
    }

    // Fetch from PC
    fn fetch8(&mut self, read: &dyn Fn(u32) -> u8) -> u8 {
        let addr = (self.pbr as u32) << 16 | self.pc as u32;
        let v = read(addr);
        self.pc = self.pc.wrapping_add(1);
        v
    }
    fn fetch16(&mut self, read: &dyn Fn(u32) -> u8) -> u16 {
        let lo = self.fetch8(read) as u16;
        let hi = self.fetch8(read) as u16;
        lo | (hi << 8)
    }
    fn fetch24(&mut self, read: &dyn Fn(u32) -> u8) -> u32 {
        let lo = self.fetch8(read) as u32;
        let mi = self.fetch8(read) as u32;
        let hi = self.fetch8(read) as u32;
        lo | (mi << 8) | (hi << 16)
    }

    // Read 16-bit from linear address
    fn read16(&self, addr: u32, read: &dyn Fn(u32) -> u8) -> u16 {
        let lo = read(addr) as u16;
        let hi = read(addr.wrapping_add(1)) as u16;
        lo | (hi << 8)
    }

    // Write 16-bit to linear address
    fn write16(&self, addr: u32, v: u16, write: &mut dyn FnMut(u32, u8)) {
        write(addr, v as u8);
        write(addr.wrapping_add(1), (v >> 8) as u8);
    }

    /// Set NMI line level. Edge-detect happens on step().
    pub fn set_nmi(&mut self, level: bool) {
        self.nmi_line = level;
    }

    /// Check and handle NMI edge (falling edge = false->true transition of nmi_line).
    fn check_nmi(&mut self) {
        if self.nmi_line && !self.nmi_prev {
            self.nmi_pending = true;
        }
        self.nmi_prev = self.nmi_line;
    }

    /// Execute one instruction. Returns master cycles consumed.
    pub fn step(&mut self, read: &dyn Fn(u32) -> u8, write: &mut dyn FnMut(u32, u8)) -> u32 {
        let start_cycles = self.cycles;

        // Check for pending interrupts
        self.check_nmi();

        if self.waiting {
            if self.nmi_pending || (self.irq_line && self.p & FLAG_I == 0) {
                self.waiting = false;
                self.cycles += 6; // WAI recovery
            } else {
                self.cycles += 6; // Idle cycles while waiting
                return (self.cycles - start_cycles) as u32;
            }
        }

        if self.stopped {
            self.cycles += 6;
            return (self.cycles - start_cycles) as u32;
        }

        // Handle NMI
        if self.nmi_pending {
            self.nmi_pending = false;
            self.do_interrupt(read, write, false);
            return (self.cycles - start_cycles) as u32;
        }

        // Handle IRQ
        if self.irq_line && self.p & FLAG_I == 0 {
            self.do_interrupt(read, write, true);
            return (self.cycles - start_cycles) as u32;
        }

        let opcode = self.fetch8(read);
        self.execute(opcode, read, write);

        (self.cycles - start_cycles) as u32
    }

    fn do_interrupt(&mut self, read: &dyn Fn(u32) -> u8, write: &mut dyn FnMut(u32, u8), is_irq: bool) {
        if !self.emulation {
            self.push8(self.pbr, write);
        }
        self.push16(self.pc, write);
        self.push8(self.p, write);
        self.p |= FLAG_I;
        self.p &= !FLAG_D;
        self.pbr = 0;
        let vector = if is_irq {
            if self.emulation { 0x00FFFE } else { 0x00FFEE }
        } else {
            if self.emulation { 0x00FFFA } else { 0x00FFEA }
        };
        self.pc = self.read16(vector, read);
        self.cycles += 7 * 6;
    }

    // ── Addressing mode helpers ──
    // Each returns a 24-bit linear address.

    fn addr_imm8(&mut self, _read: &dyn Fn(u32) -> u8) -> u32 {
        let addr = (self.pbr as u32) << 16 | self.pc as u32;
        self.pc = self.pc.wrapping_add(1);
        self.cycles += 6;
        addr
    }
    fn addr_imm16(&mut self, _read: &dyn Fn(u32) -> u8) -> u32 {
        let addr = (self.pbr as u32) << 16 | self.pc as u32;
        self.pc = self.pc.wrapping_add(2);
        self.cycles += 6;
        addr
    }

    fn addr_dp(&mut self, read: &dyn Fn(u32) -> u8) -> u32 {
        let off = self.fetch8(read) as u16;
        self.cycles += 6;
        if self.d & 0xFF != 0 { self.cycles += 6; }
        self.d.wrapping_add(off) as u32
    }
    fn addr_dp_x(&mut self, read: &dyn Fn(u32) -> u8) -> u32 {
        let off = self.fetch8(read) as u16;
        self.cycles += 12;
        if self.d & 0xFF != 0 { self.cycles += 6; }
        self.d.wrapping_add(off).wrapping_add(self.x) as u32
    }
    fn addr_dp_y(&mut self, read: &dyn Fn(u32) -> u8) -> u32 {
        let off = self.fetch8(read) as u16;
        self.cycles += 12;
        if self.d & 0xFF != 0 { self.cycles += 6; }
        self.d.wrapping_add(off).wrapping_add(self.y) as u32
    }
    fn addr_dp_ind(&mut self, read: &dyn Fn(u32) -> u8) -> u32 {
        let off = self.fetch8(read) as u16;
        self.cycles += 6;
        if self.d & 0xFF != 0 { self.cycles += 6; }
        let ptr = self.d.wrapping_add(off) as u32;
        let lo = read(ptr) as u32;
        let hi = read(ptr + 1) as u32;
        (self.dbr as u32) << 16 | (hi << 8) | lo
    }
    fn addr_dp_ind_long(&mut self, read: &dyn Fn(u32) -> u8) -> u32 {
        let off = self.fetch8(read) as u16;
        self.cycles += 6;
        if self.d & 0xFF != 0 { self.cycles += 6; }
        let ptr = self.d.wrapping_add(off) as u32;
        let lo = read(ptr) as u32;
        let mi = read(ptr + 1) as u32;
        let hi = read(ptr + 2) as u32;
        (hi << 16) | (mi << 8) | lo
    }
    fn addr_dp_x_ind(&mut self, read: &dyn Fn(u32) -> u8) -> u32 {
        let off = self.fetch8(read) as u16;
        self.cycles += 12;
        if self.d & 0xFF != 0 { self.cycles += 6; }
        let ptr = self.d.wrapping_add(off).wrapping_add(self.x) as u32;
        let lo = read(ptr) as u32;
        let hi = read(ptr + 1) as u32;
        (self.dbr as u32) << 16 | (hi << 8) | lo
    }
    fn addr_dp_ind_y(&mut self, read: &dyn Fn(u32) -> u8) -> u32 {
        let off = self.fetch8(read) as u16;
        self.cycles += 6;
        if self.d & 0xFF != 0 { self.cycles += 6; }
        let ptr = self.d.wrapping_add(off) as u32;
        let lo = read(ptr) as u32;
        let hi = read(ptr + 1) as u32;
        let base = (self.dbr as u32) << 16 | (hi << 8) | lo;
        base.wrapping_add(self.y as u32)
    }
    fn addr_dp_ind_long_y(&mut self, read: &dyn Fn(u32) -> u8) -> u32 {
        let off = self.fetch8(read) as u16;
        self.cycles += 6;
        if self.d & 0xFF != 0 { self.cycles += 6; }
        let ptr = self.d.wrapping_add(off) as u32;
        let lo = read(ptr) as u32;
        let mi = read(ptr + 1) as u32;
        let hi = read(ptr + 2) as u32;
        let base = (hi << 16) | (mi << 8) | lo;
        base.wrapping_add(self.y as u32)
    }
    fn addr_abs(&mut self, read: &dyn Fn(u32) -> u8) -> u32 {
        let addr = self.fetch16(read);
        self.cycles += 6;
        (self.dbr as u32) << 16 | addr as u32
    }
    fn addr_abs_x(&mut self, read: &dyn Fn(u32) -> u8) -> u32 {
        let addr = self.fetch16(read);
        self.cycles += 6;
        let base = (self.dbr as u32) << 16 | addr as u32;
        base.wrapping_add(self.x as u32)
    }
    fn addr_abs_y(&mut self, read: &dyn Fn(u32) -> u8) -> u32 {
        let addr = self.fetch16(read);
        self.cycles += 6;
        let base = (self.dbr as u32) << 16 | addr as u32;
        base.wrapping_add(self.y as u32)
    }
    fn addr_abs_long(&mut self, read: &dyn Fn(u32) -> u8) -> u32 {
        let addr = self.fetch24(read);
        self.cycles += 6;
        addr
    }
    fn addr_abs_long_x(&mut self, read: &dyn Fn(u32) -> u8) -> u32 {
        let addr = self.fetch24(read);
        self.cycles += 6;
        addr.wrapping_add(self.x as u32)
    }
    fn addr_sr(&mut self, read: &dyn Fn(u32) -> u8) -> u32 {
        let off = self.fetch8(read) as u16;
        self.cycles += 12;
        self.s.wrapping_add(off) as u32
    }
    fn addr_sr_ind_y(&mut self, read: &dyn Fn(u32) -> u8) -> u32 {
        let off = self.fetch8(read) as u16;
        self.cycles += 12;
        let ptr = self.s.wrapping_add(off) as u32;
        let lo = read(ptr) as u32;
        let hi = read(ptr + 1) as u32;
        let base = (self.dbr as u32) << 16 | (hi << 8) | lo;
        base.wrapping_add(self.y as u32)
    }

    // ── ALU helpers ──

    fn op_adc8(&mut self, val: u8) {
        let a = self.al() as u16;
        let v = val as u16;
        let c = (self.p & FLAG_C) as u16;
        if self.p & FLAG_D != 0 {
            // BCD
            let mut lo = (a & 0x0F) + (v & 0x0F) + c;
            if lo > 9 { lo += 6; }
            let mut hi = (a >> 4) + (v >> 4) + if lo > 0x0F { 1 } else { 0 };
            let result = ((hi & 0x0F) << 4) | (lo & 0x0F);
            // Set V before BCD adjust
            let signed = !(a ^ v) & (a ^ (result as u16)) & 0x80;
            self.p = (self.p & !FLAG_V) | if signed != 0 { FLAG_V } else { 0 };
            if hi > 9 { hi += 6; }
            self.p = (self.p & !FLAG_C) | if hi > 0x0F { FLAG_C } else { 0 };
            let r = result as u8;
            self.set_al(r);
            self.set_nz8(r);
        } else {
            let result = a + v + c;
            let r = result as u8;
            self.p = (self.p & !(FLAG_C | FLAG_V))
                | if result > 0xFF { FLAG_C } else { 0 }
                | if (!(a ^ v) & (a ^ result)) & 0x80 != 0 { FLAG_V } else { 0 };
            self.set_al(r);
            self.set_nz8(r);
        }
    }
    fn op_adc16(&mut self, val: u16) {
        let a = self.a as u32;
        let v = val as u32;
        let c = (self.p & FLAG_C) as u32;
        if self.p & FLAG_D != 0 {
            let mut r = 0u32;
            let mut carry = c;
            for nibble in 0..4 {
                let shift = nibble * 4;
                let mut n = ((a >> shift) & 0xF) + ((v >> shift) & 0xF) + carry;
                if n > 9 { n += 6; }
                carry = n >> 4;
                r |= (n & 0xF) << shift;
            }
            self.p = (self.p & !FLAG_C) | if carry != 0 { FLAG_C } else { 0 };
            let result = r as u16;
            self.p = (self.p & !FLAG_V)
                | if (!(a ^ v) & (a ^ r)) & 0x8000 != 0 { FLAG_V } else { 0 };
            self.a = result;
            self.set_nz16(result);
        } else {
            let result = a + v + c;
            let r = result as u16;
            self.p = (self.p & !(FLAG_C | FLAG_V))
                | if result > 0xFFFF { FLAG_C } else { 0 }
                | if (!(a ^ v) & (a ^ result)) & 0x8000 != 0 { FLAG_V } else { 0 };
            self.a = r;
            self.set_nz16(r);
        }
    }
    fn op_sbc8(&mut self, val: u8) {
        let a = self.al() as u16;
        let v = val as u16;
        let c = (self.p & FLAG_C) as u16;
        if self.p & FLAG_D != 0 {
            let mut lo = (a & 0x0F).wrapping_sub(v & 0x0F).wrapping_sub(1 - c);
            let borrow_lo = if lo > 0x0F { lo = lo.wrapping_add(10); 1u16 } else { 0 };
            let mut hi = (a >> 4).wrapping_sub(v >> 4).wrapping_sub(borrow_lo);
            if hi > 0x0F { hi = hi.wrapping_add(10); }
            let result = ((hi & 0x0F) << 4) | (lo & 0x0F);
            let bin_result = a.wrapping_sub(v).wrapping_sub(1 - c);
            self.p = (self.p & !(FLAG_C | FLAG_V))
                | if bin_result <= 0xFF { FLAG_C } else { 0 }
                | if ((a ^ v) & (a ^ bin_result)) & 0x80 != 0 { FLAG_V } else { 0 };
            let r = result as u8;
            self.set_al(r);
            self.set_nz8(r);
        } else {
            let result = a.wrapping_sub(v).wrapping_sub(1 - c);
            let r = result as u8;
            self.p = (self.p & !(FLAG_C | FLAG_V))
                | if result <= 0xFF { FLAG_C } else { 0 }
                | if ((a ^ v) & (a ^ result)) & 0x80 != 0 { FLAG_V } else { 0 };
            self.set_al(r);
            self.set_nz8(r);
        }
    }
    fn op_sbc16(&mut self, val: u16) {
        let a = self.a as u32;
        let v = val as u32;
        let c = (self.p & FLAG_C) as u32;
        if self.p & FLAG_D != 0 {
            let mut r = 0u32;
            let mut borrow = 1 - c;
            for nibble in 0..4 {
                let shift = nibble * 4;
                let mut n = ((a >> shift) & 0xF).wrapping_sub(((v >> shift) & 0xF) + borrow);
                if n > 0xF { n = n.wrapping_add(10); borrow = 1; } else { borrow = 0; }
                r |= (n & 0xF) << shift;
            }
            self.p = (self.p & !FLAG_C) | if borrow == 0 { FLAG_C } else { 0 };
            let bin_result = a.wrapping_sub(v).wrapping_sub(1 - c);
            self.p = (self.p & !FLAG_V)
                | if ((a ^ v) & (a ^ bin_result)) & 0x8000 != 0 { FLAG_V } else { 0 };
            self.a = r as u16;
            self.set_nz16(r as u16);
        } else {
            let result = a.wrapping_sub(v).wrapping_sub(1 - c);
            let r = result as u16;
            self.p = (self.p & !(FLAG_C | FLAG_V))
                | if result <= 0xFFFF { FLAG_C } else { 0 }
                | if ((a ^ v) & (a ^ result)) & 0x8000 != 0 { FLAG_V } else { 0 };
            self.a = r;
            self.set_nz16(r);
        }
    }

    fn op_cmp8(&mut self, a: u8, b: u8) {
        let result = (a as u16).wrapping_sub(b as u16);
        self.p = (self.p & !FLAG_C) | if a >= b { FLAG_C } else { 0 };
        self.set_nz8(result as u8);
    }
    fn op_cmp16(&mut self, a: u16, b: u16) {
        let result = (a as u32).wrapping_sub(b as u32);
        self.p = (self.p & !FLAG_C) | if a >= b { FLAG_C } else { 0 };
        self.set_nz16(result as u16);
    }

    fn branch(&mut self, cond: bool, read: &dyn Fn(u32) -> u8) {
        let offset = self.fetch8(read) as i8;
        self.cycles += 6;
        if cond {
            self.pc = self.pc.wrapping_add(offset as u16);
            self.cycles += 6;
        }
    }

    fn set_p(&mut self, val: u8) {
        self.p = val;
        if self.emulation {
            self.p |= FLAG_M | FLAG_X;
        }
        if self.p & FLAG_X != 0 {
            self.x &= 0xFF;
            self.y &= 0xFF;
        }
    }

    // ── Main decode & execute ──

    fn execute(&mut self, op: u8, read: &dyn Fn(u32) -> u8, write: &mut dyn FnMut(u32, u8)) {
        self.cycles += 6; // Base cycle for opcode fetch

        match op {
            // ── ADC ──
            0x69 => { // ADC #imm
                if self.a_is_8() {
                    let v = self.fetch8(read);
                    self.op_adc8(v);
                } else {
                    let v = self.fetch16(read);
                    self.op_adc16(v);
                    self.cycles += 6;
                }
            }
            0x65 => { let a = self.addr_dp(read); self.op_adc8_or_16(a, read); }
            0x75 => { let a = self.addr_dp_x(read); self.op_adc8_or_16(a, read); }
            0x6D => { let a = self.addr_abs(read); self.op_adc8_or_16(a, read); }
            0x7D => { let a = self.addr_abs_x(read); self.op_adc8_or_16(a, read); }
            0x79 => { let a = self.addr_abs_y(read); self.op_adc8_or_16(a, read); }
            0x72 => { let a = self.addr_dp_ind(read); self.op_adc8_or_16(a, read); }
            0x61 => { let a = self.addr_dp_x_ind(read); self.op_adc8_or_16(a, read); }
            0x71 => { let a = self.addr_dp_ind_y(read); self.op_adc8_or_16(a, read); }
            0x67 => { let a = self.addr_dp_ind_long(read); self.op_adc8_or_16(a, read); }
            0x77 => { let a = self.addr_dp_ind_long_y(read); self.op_adc8_or_16(a, read); }
            0x6F => { let a = self.addr_abs_long(read); self.op_adc8_or_16(a, read); }
            0x7F => { let a = self.addr_abs_long_x(read); self.op_adc8_or_16(a, read); }
            0x63 => { let a = self.addr_sr(read); self.op_adc8_or_16(a, read); }
            0x73 => { let a = self.addr_sr_ind_y(read); self.op_adc8_or_16(a, read); }

            // ── SBC ──
            0xE9 => {
                if self.a_is_8() {
                    let v = self.fetch8(read);
                    self.op_sbc8(v);
                } else {
                    let v = self.fetch16(read);
                    self.op_sbc16(v);
                    self.cycles += 6;
                }
            }
            0xE5 => { let a = self.addr_dp(read); self.op_sbc8_or_16(a, read); }
            0xF5 => { let a = self.addr_dp_x(read); self.op_sbc8_or_16(a, read); }
            0xED => { let a = self.addr_abs(read); self.op_sbc8_or_16(a, read); }
            0xFD => { let a = self.addr_abs_x(read); self.op_sbc8_or_16(a, read); }
            0xF9 => { let a = self.addr_abs_y(read); self.op_sbc8_or_16(a, read); }
            0xF2 => { let a = self.addr_dp_ind(read); self.op_sbc8_or_16(a, read); }
            0xE1 => { let a = self.addr_dp_x_ind(read); self.op_sbc8_or_16(a, read); }
            0xF1 => { let a = self.addr_dp_ind_y(read); self.op_sbc8_or_16(a, read); }
            0xE7 => { let a = self.addr_dp_ind_long(read); self.op_sbc8_or_16(a, read); }
            0xF7 => { let a = self.addr_dp_ind_long_y(read); self.op_sbc8_or_16(a, read); }
            0xEF => { let a = self.addr_abs_long(read); self.op_sbc8_or_16(a, read); }
            0xFF => { let a = self.addr_abs_long_x(read); self.op_sbc8_or_16(a, read); }
            0xE3 => { let a = self.addr_sr(read); self.op_sbc8_or_16(a, read); }
            0xF3 => { let a = self.addr_sr_ind_y(read); self.op_sbc8_or_16(a, read); }

            // ── AND ──
            0x29 => {
                if self.a_is_8() {
                    let v = self.fetch8(read);
                    self.set_al(self.al() & v);
                    self.set_nz8(self.al());
                } else {
                    let v = self.fetch16(read);
                    self.a &= v;
                    self.set_nz16(self.a);
                    self.cycles += 6;
                }
            }
            0x25 => { let a = self.addr_dp(read); self.op_and(a, read); }
            0x35 => { let a = self.addr_dp_x(read); self.op_and(a, read); }
            0x2D => { let a = self.addr_abs(read); self.op_and(a, read); }
            0x3D => { let a = self.addr_abs_x(read); self.op_and(a, read); }
            0x39 => { let a = self.addr_abs_y(read); self.op_and(a, read); }
            0x32 => { let a = self.addr_dp_ind(read); self.op_and(a, read); }
            0x21 => { let a = self.addr_dp_x_ind(read); self.op_and(a, read); }
            0x31 => { let a = self.addr_dp_ind_y(read); self.op_and(a, read); }
            0x27 => { let a = self.addr_dp_ind_long(read); self.op_and(a, read); }
            0x37 => { let a = self.addr_dp_ind_long_y(read); self.op_and(a, read); }
            0x2F => { let a = self.addr_abs_long(read); self.op_and(a, read); }
            0x3F => { let a = self.addr_abs_long_x(read); self.op_and(a, read); }
            0x23 => { let a = self.addr_sr(read); self.op_and(a, read); }
            0x33 => { let a = self.addr_sr_ind_y(read); self.op_and(a, read); }

            // ── ORA ──
            0x09 => {
                if self.a_is_8() {
                    let v = self.fetch8(read);
                    self.set_al(self.al() | v);
                    self.set_nz8(self.al());
                } else {
                    let v = self.fetch16(read);
                    self.a |= v;
                    self.set_nz16(self.a);
                    self.cycles += 6;
                }
            }
            0x05 => { let a = self.addr_dp(read); self.op_ora(a, read); }
            0x15 => { let a = self.addr_dp_x(read); self.op_ora(a, read); }
            0x0D => { let a = self.addr_abs(read); self.op_ora(a, read); }
            0x1D => { let a = self.addr_abs_x(read); self.op_ora(a, read); }
            0x19 => { let a = self.addr_abs_y(read); self.op_ora(a, read); }
            0x12 => { let a = self.addr_dp_ind(read); self.op_ora(a, read); }
            0x01 => { let a = self.addr_dp_x_ind(read); self.op_ora(a, read); }
            0x11 => { let a = self.addr_dp_ind_y(read); self.op_ora(a, read); }
            0x07 => { let a = self.addr_dp_ind_long(read); self.op_ora(a, read); }
            0x17 => { let a = self.addr_dp_ind_long_y(read); self.op_ora(a, read); }
            0x0F => { let a = self.addr_abs_long(read); self.op_ora(a, read); }
            0x1F => { let a = self.addr_abs_long_x(read); self.op_ora(a, read); }
            0x03 => { let a = self.addr_sr(read); self.op_ora(a, read); }
            0x13 => { let a = self.addr_sr_ind_y(read); self.op_ora(a, read); }

            // ── EOR ──
            0x49 => {
                if self.a_is_8() {
                    let v = self.fetch8(read);
                    self.set_al(self.al() ^ v);
                    self.set_nz8(self.al());
                } else {
                    let v = self.fetch16(read);
                    self.a ^= v;
                    self.set_nz16(self.a);
                    self.cycles += 6;
                }
            }
            0x45 => { let a = self.addr_dp(read); self.op_eor(a, read); }
            0x55 => { let a = self.addr_dp_x(read); self.op_eor(a, read); }
            0x4D => { let a = self.addr_abs(read); self.op_eor(a, read); }
            0x5D => { let a = self.addr_abs_x(read); self.op_eor(a, read); }
            0x59 => { let a = self.addr_abs_y(read); self.op_eor(a, read); }
            0x52 => { let a = self.addr_dp_ind(read); self.op_eor(a, read); }
            0x41 => { let a = self.addr_dp_x_ind(read); self.op_eor(a, read); }
            0x51 => { let a = self.addr_dp_ind_y(read); self.op_eor(a, read); }
            0x47 => { let a = self.addr_dp_ind_long(read); self.op_eor(a, read); }
            0x57 => { let a = self.addr_dp_ind_long_y(read); self.op_eor(a, read); }
            0x4F => { let a = self.addr_abs_long(read); self.op_eor(a, read); }
            0x5F => { let a = self.addr_abs_long_x(read); self.op_eor(a, read); }
            0x43 => { let a = self.addr_sr(read); self.op_eor(a, read); }
            0x53 => { let a = self.addr_sr_ind_y(read); self.op_eor(a, read); }

            // ── CMP ──
            0xC9 => {
                if self.a_is_8() {
                    let v = self.fetch8(read);
                    self.op_cmp8(self.al(), v);
                } else {
                    let v = self.fetch16(read);
                    self.op_cmp16(self.a, v);
                    self.cycles += 6;
                }
            }
            0xC5 => { let a = self.addr_dp(read); self.op_cmp_mem(a, read); }
            0xD5 => { let a = self.addr_dp_x(read); self.op_cmp_mem(a, read); }
            0xCD => { let a = self.addr_abs(read); self.op_cmp_mem(a, read); }
            0xDD => { let a = self.addr_abs_x(read); self.op_cmp_mem(a, read); }
            0xD9 => { let a = self.addr_abs_y(read); self.op_cmp_mem(a, read); }
            0xD2 => { let a = self.addr_dp_ind(read); self.op_cmp_mem(a, read); }
            0xC1 => { let a = self.addr_dp_x_ind(read); self.op_cmp_mem(a, read); }
            0xD1 => { let a = self.addr_dp_ind_y(read); self.op_cmp_mem(a, read); }
            0xC7 => { let a = self.addr_dp_ind_long(read); self.op_cmp_mem(a, read); }
            0xD7 => { let a = self.addr_dp_ind_long_y(read); self.op_cmp_mem(a, read); }
            0xCF => { let a = self.addr_abs_long(read); self.op_cmp_mem(a, read); }
            0xDF => { let a = self.addr_abs_long_x(read); self.op_cmp_mem(a, read); }
            0xC3 => { let a = self.addr_sr(read); self.op_cmp_mem(a, read); }
            0xD3 => { let a = self.addr_sr_ind_y(read); self.op_cmp_mem(a, read); }

            // ── CPX ──
            0xE0 => {
                if self.x_is_8() {
                    let v = self.fetch8(read);
                    self.op_cmp8(self.x as u8, v);
                } else {
                    let v = self.fetch16(read);
                    self.op_cmp16(self.x, v);
                    self.cycles += 6;
                }
            }
            0xE4 => { let a = self.addr_dp(read); self.op_cpx_mem(a, read); }
            0xEC => { let a = self.addr_abs(read); self.op_cpx_mem(a, read); }

            // ── CPY ──
            0xC0 => {
                if self.x_is_8() {
                    let v = self.fetch8(read);
                    self.op_cmp8(self.y as u8, v);
                } else {
                    let v = self.fetch16(read);
                    self.op_cmp16(self.y, v);
                    self.cycles += 6;
                }
            }
            0xC4 => { let a = self.addr_dp(read); self.op_cpy_mem(a, read); }
            0xCC => { let a = self.addr_abs(read); self.op_cpy_mem(a, read); }

            // ── LDA ──
            0xA9 => {
                if self.a_is_8() {
                    let v = self.fetch8(read);
                    self.set_al(v);
                    self.set_nz8(v);
                } else {
                    let v = self.fetch16(read);
                    self.a = v;
                    self.set_nz16(v);
                    self.cycles += 6;
                }
            }
            0xA5 => { let a = self.addr_dp(read); self.op_lda(a, read); }
            0xB5 => { let a = self.addr_dp_x(read); self.op_lda(a, read); }
            0xAD => { let a = self.addr_abs(read); self.op_lda(a, read); }
            0xBD => { let a = self.addr_abs_x(read); self.op_lda(a, read); }
            0xB9 => { let a = self.addr_abs_y(read); self.op_lda(a, read); }
            0xB2 => { let a = self.addr_dp_ind(read); self.op_lda(a, read); }
            0xA1 => { let a = self.addr_dp_x_ind(read); self.op_lda(a, read); }
            0xB1 => { let a = self.addr_dp_ind_y(read); self.op_lda(a, read); }
            0xA7 => { let a = self.addr_dp_ind_long(read); self.op_lda(a, read); }
            0xB7 => { let a = self.addr_dp_ind_long_y(read); self.op_lda(a, read); }
            0xAF => { let a = self.addr_abs_long(read); self.op_lda(a, read); }
            0xBF => { let a = self.addr_abs_long_x(read); self.op_lda(a, read); }
            0xA3 => { let a = self.addr_sr(read); self.op_lda(a, read); }
            0xB3 => { let a = self.addr_sr_ind_y(read); self.op_lda(a, read); }

            // ── LDX ──
            0xA2 => {
                if self.x_is_8() {
                    let v = self.fetch8(read);
                    self.x = v as u16;
                    self.set_nz8(v);
                } else {
                    let v = self.fetch16(read);
                    self.x = v;
                    self.set_nz16(v);
                    self.cycles += 6;
                }
            }
            0xA6 => { let a = self.addr_dp(read); self.op_ldx(a, read); }
            0xB6 => { let a = self.addr_dp_y(read); self.op_ldx(a, read); }
            0xAE => { let a = self.addr_abs(read); self.op_ldx(a, read); }
            0xBE => { let a = self.addr_abs_y(read); self.op_ldx(a, read); }

            // ── LDY ──
            0xA0 => {
                if self.x_is_8() {
                    let v = self.fetch8(read);
                    self.y = v as u16;
                    self.set_nz8(v);
                } else {
                    let v = self.fetch16(read);
                    self.y = v;
                    self.set_nz16(v);
                    self.cycles += 6;
                }
            }
            0xA4 => { let a = self.addr_dp(read); self.op_ldy(a, read); }
            0xB4 => { let a = self.addr_dp_x(read); self.op_ldy(a, read); }
            0xAC => { let a = self.addr_abs(read); self.op_ldy(a, read); }
            0xBC => { let a = self.addr_abs_x(read); self.op_ldy(a, read); }

            // ── STA ──
            0x85 => { let a = self.addr_dp(read); self.op_sta(a, write); }
            0x95 => { let a = self.addr_dp_x(read); self.op_sta(a, write); }
            0x8D => { let a = self.addr_abs(read); self.op_sta(a, write); }
            0x9D => { let a = self.addr_abs_x(read); self.op_sta(a, write); }
            0x99 => { let a = self.addr_abs_y(read); self.op_sta(a, write); }
            0x92 => { let a = self.addr_dp_ind(read); self.op_sta(a, write); }
            0x81 => { let a = self.addr_dp_x_ind(read); self.op_sta(a, write); }
            0x91 => { let a = self.addr_dp_ind_y(read); self.op_sta(a, write); }
            0x87 => { let a = self.addr_dp_ind_long(read); self.op_sta(a, write); }
            0x97 => { let a = self.addr_dp_ind_long_y(read); self.op_sta(a, write); }
            0x8F => { let a = self.addr_abs_long(read); self.op_sta(a, write); }
            0x9F => { let a = self.addr_abs_long_x(read); self.op_sta(a, write); }
            0x83 => { let a = self.addr_sr(read); self.op_sta(a, write); }
            0x93 => { let a = self.addr_sr_ind_y(read); self.op_sta(a, write); }

            // ── STX ──
            0x86 => { let a = self.addr_dp(read); self.op_stx(a, write); }
            0x96 => { let a = self.addr_dp_y(read); self.op_stx(a, write); }
            0x8E => { let a = self.addr_abs(read); self.op_stx(a, write); }

            // ── STY ──
            0x84 => { let a = self.addr_dp(read); self.op_sty(a, write); }
            0x94 => { let a = self.addr_dp_x(read); self.op_sty(a, write); }
            0x8C => { let a = self.addr_abs(read); self.op_sty(a, write); }

            // ── STZ ──
            0x64 => { let a = self.addr_dp(read); self.op_stz(a, write); }
            0x74 => { let a = self.addr_dp_x(read); self.op_stz(a, write); }
            0x9C => { let a = self.addr_abs(read); self.op_stz(a, write); }
            0x9E => { let a = self.addr_abs_x(read); self.op_stz(a, write); }

            // ── INC ──
            0x1A => { // INC A
                if self.a_is_8() {
                    let v = self.al().wrapping_add(1);
                    self.set_al(v);
                    self.set_nz8(v);
                } else {
                    self.a = self.a.wrapping_add(1);
                    self.set_nz16(self.a);
                }
                self.cycles += 6;
            }
            0xE6 => { let a = self.addr_dp(read); self.op_inc_mem(a, read, write); }
            0xF6 => { let a = self.addr_dp_x(read); self.op_inc_mem(a, read, write); }
            0xEE => { let a = self.addr_abs(read); self.op_inc_mem(a, read, write); }
            0xFE => { let a = self.addr_abs_x(read); self.op_inc_mem(a, read, write); }

            // ── DEC ──
            0x3A => { // DEC A
                if self.a_is_8() {
                    let v = self.al().wrapping_sub(1);
                    self.set_al(v);
                    self.set_nz8(v);
                } else {
                    self.a = self.a.wrapping_sub(1);
                    self.set_nz16(self.a);
                }
                self.cycles += 6;
            }
            0xC6 => { let a = self.addr_dp(read); self.op_dec_mem(a, read, write); }
            0xD6 => { let a = self.addr_dp_x(read); self.op_dec_mem(a, read, write); }
            0xCE => { let a = self.addr_abs(read); self.op_dec_mem(a, read, write); }
            0xDE => { let a = self.addr_abs_x(read); self.op_dec_mem(a, read, write); }

            // ── INX, DEX, INY, DEY ──
            0xE8 => { // INX
                if self.x_is_8() {
                    self.x = (self.x as u8).wrapping_add(1) as u16;
                    self.set_nz8(self.x as u8);
                } else {
                    self.x = self.x.wrapping_add(1);
                    self.set_nz16(self.x);
                }
                self.cycles += 6;
            }
            0xCA => { // DEX
                if self.x_is_8() {
                    self.x = (self.x as u8).wrapping_sub(1) as u16;
                    self.set_nz8(self.x as u8);
                } else {
                    self.x = self.x.wrapping_sub(1);
                    self.set_nz16(self.x);
                }
                self.cycles += 6;
            }
            0xC8 => { // INY
                if self.x_is_8() {
                    self.y = (self.y as u8).wrapping_add(1) as u16;
                    self.set_nz8(self.y as u8);
                } else {
                    self.y = self.y.wrapping_add(1);
                    self.set_nz16(self.y);
                }
                self.cycles += 6;
            }
            0x88 => { // DEY
                if self.x_is_8() {
                    self.y = (self.y as u8).wrapping_sub(1) as u16;
                    self.set_nz8(self.y as u8);
                } else {
                    self.y = self.y.wrapping_sub(1);
                    self.set_nz16(self.y);
                }
                self.cycles += 6;
            }

            // ── ASL ──
            0x0A => { // ASL A
                if self.a_is_8() {
                    let v = self.al();
                    self.p = (self.p & !FLAG_C) | if v & 0x80 != 0 { FLAG_C } else { 0 };
                    let r = v << 1;
                    self.set_al(r);
                    self.set_nz8(r);
                } else {
                    self.p = (self.p & !FLAG_C) | if self.a & 0x8000 != 0 { FLAG_C } else { 0 };
                    self.a <<= 1;
                    self.set_nz16(self.a);
                }
                self.cycles += 6;
            }
            0x06 => { let a = self.addr_dp(read); self.op_asl_mem(a, read, write); }
            0x16 => { let a = self.addr_dp_x(read); self.op_asl_mem(a, read, write); }
            0x0E => { let a = self.addr_abs(read); self.op_asl_mem(a, read, write); }
            0x1E => { let a = self.addr_abs_x(read); self.op_asl_mem(a, read, write); }

            // ── LSR ──
            0x4A => { // LSR A
                if self.a_is_8() {
                    let v = self.al();
                    self.p = (self.p & !FLAG_C) | (v & 1);
                    let r = v >> 1;
                    self.set_al(r);
                    self.set_nz8(r);
                } else {
                    self.p = (self.p & !FLAG_C) | (self.a as u8 & 1);
                    self.a >>= 1;
                    self.set_nz16(self.a);
                }
                self.cycles += 6;
            }
            0x46 => { let a = self.addr_dp(read); self.op_lsr_mem(a, read, write); }
            0x56 => { let a = self.addr_dp_x(read); self.op_lsr_mem(a, read, write); }
            0x4E => { let a = self.addr_abs(read); self.op_lsr_mem(a, read, write); }
            0x5E => { let a = self.addr_abs_x(read); self.op_lsr_mem(a, read, write); }

            // ── ROL ──
            0x2A => { // ROL A
                if self.a_is_8() {
                    let v = self.al();
                    let c = self.p & FLAG_C;
                    self.p = (self.p & !FLAG_C) | if v & 0x80 != 0 { FLAG_C } else { 0 };
                    let r = (v << 1) | c;
                    self.set_al(r);
                    self.set_nz8(r);
                } else {
                    let c = (self.p & FLAG_C) as u16;
                    self.p = (self.p & !FLAG_C) | if self.a & 0x8000 != 0 { FLAG_C } else { 0 };
                    self.a = (self.a << 1) | c;
                    self.set_nz16(self.a);
                }
                self.cycles += 6;
            }
            0x26 => { let a = self.addr_dp(read); self.op_rol_mem(a, read, write); }
            0x36 => { let a = self.addr_dp_x(read); self.op_rol_mem(a, read, write); }
            0x2E => { let a = self.addr_abs(read); self.op_rol_mem(a, read, write); }
            0x3E => { let a = self.addr_abs_x(read); self.op_rol_mem(a, read, write); }

            // ── ROR ──
            0x6A => { // ROR A
                if self.a_is_8() {
                    let v = self.al();
                    let c = self.p & FLAG_C;
                    self.p = (self.p & !FLAG_C) | (v & 1);
                    let r = (v >> 1) | (c << 7);
                    self.set_al(r);
                    self.set_nz8(r);
                } else {
                    let c = (self.p & FLAG_C) as u16;
                    self.p = (self.p & !FLAG_C) | (self.a as u8 & 1);
                    self.a = (self.a >> 1) | (c << 15);
                    self.set_nz16(self.a);
                }
                self.cycles += 6;
            }
            0x66 => { let a = self.addr_dp(read); self.op_ror_mem(a, read, write); }
            0x76 => { let a = self.addr_dp_x(read); self.op_ror_mem(a, read, write); }
            0x6E => { let a = self.addr_abs(read); self.op_ror_mem(a, read, write); }
            0x7E => { let a = self.addr_abs_x(read); self.op_ror_mem(a, read, write); }

            // ── BIT ──
            0x89 => { // BIT #imm — only sets Z, does not affect N/V
                if self.a_is_8() {
                    let v = self.fetch8(read);
                    let r = self.al() & v;
                    self.p = (self.p & !FLAG_Z) | if r == 0 { FLAG_Z } else { 0 };
                } else {
                    let v = self.fetch16(read);
                    let r = self.a & v;
                    self.p = (self.p & !FLAG_Z) | if r == 0 { FLAG_Z } else { 0 };
                    self.cycles += 6;
                }
            }
            0x24 => { let a = self.addr_dp(read); self.op_bit(a, read); }
            0x34 => { let a = self.addr_dp_x(read); self.op_bit(a, read); }
            0x2C => { let a = self.addr_abs(read); self.op_bit(a, read); }
            0x3C => { let a = self.addr_abs_x(read); self.op_bit(a, read); }

            // ── TRB / TSB ──
            0x14 => { let a = self.addr_dp(read); self.op_trb(a, read, write); }
            0x1C => { let a = self.addr_abs(read); self.op_trb(a, read, write); }
            0x04 => { let a = self.addr_dp(read); self.op_tsb(a, read, write); }
            0x0C => { let a = self.addr_abs(read); self.op_tsb(a, read, write); }

            // ── Branches ──
            0x10 => self.branch(self.p & FLAG_N == 0, read), // BPL
            0x30 => self.branch(self.p & FLAG_N != 0, read), // BMI
            0x50 => self.branch(self.p & FLAG_V == 0, read), // BVC
            0x70 => self.branch(self.p & FLAG_V != 0, read), // BVS
            0x90 => self.branch(self.p & FLAG_C == 0, read), // BCC
            0xB0 => self.branch(self.p & FLAG_C != 0, read), // BCS
            0xD0 => self.branch(self.p & FLAG_Z == 0, read), // BNE
            0xF0 => self.branch(self.p & FLAG_Z != 0, read), // BEQ
            0x80 => self.branch(true, read),                  // BRA
            0x82 => { // BRL (16-bit relative)
                let off = self.fetch16(read) as i16;
                self.pc = self.pc.wrapping_add(off as u16);
                self.cycles += 6;
            }

            // ── JMP / JSR / RTS / RTI ──
            0x4C => { // JMP abs
                let addr = self.fetch16(read);
                self.pc = addr;
            }
            0x6C => { // JMP (abs)
                let ptr = self.fetch16(read) as u32;
                self.pc = self.read16(ptr, read);
            }
            0x7C => { // JMP (abs,X)
                let base = self.fetch16(read);
                let ptr = (self.pbr as u32) << 16 | base.wrapping_add(self.x) as u32;
                self.pc = self.read16(ptr, read);
            }
            0x5C => { // JML abs long
                let addr = self.fetch24(read);
                self.pbr = (addr >> 16) as u8;
                self.pc = addr as u16;
            }
            0xDC => { // JML [abs]
                let ptr = self.fetch16(read) as u32;
                let lo = read(ptr) as u32;
                let mi = read(ptr + 1) as u32;
                let hi = read(ptr + 2) as u32;
                self.pc = (lo | (mi << 8)) as u16;
                self.pbr = hi as u8;
            }
            0x20 => { // JSR abs
                let addr = self.fetch16(read);
                self.push16(self.pc.wrapping_sub(1), write);
                self.pc = addr;
                self.cycles += 6;
            }
            0xFC => { // JSR (abs,X)
                let base = self.fetch16(read);
                self.push16(self.pc.wrapping_sub(1), write);
                let ptr = (self.pbr as u32) << 16 | base.wrapping_add(self.x) as u32;
                self.pc = self.read16(ptr, read);
                self.cycles += 6;
            }
            0x22 => { // JSL abs long
                let addr = self.fetch24(read);
                self.push8(self.pbr, write);
                self.push16(self.pc.wrapping_sub(1), write);
                self.pbr = (addr >> 16) as u8;
                self.pc = addr as u16;
                self.cycles += 6;
            }
            0x60 => { // RTS
                self.pc = self.pull16(read).wrapping_add(1);
                self.cycles += 18;
            }
            0x6B => { // RTL
                self.pc = self.pull16(read).wrapping_add(1);
                self.pbr = self.pull8(read);
                self.cycles += 12;
            }
            0x40 => { // RTI
                let p = self.pull8(read);
                self.set_p(p);
                self.pc = self.pull16(read);
                if !self.emulation {
                    self.pbr = self.pull8(read);
                }
                self.cycles += 12;
            }

            // ── Flag set/clear ──
            0x18 => { self.p &= !FLAG_C; self.cycles += 6; } // CLC
            0x38 => { self.p |= FLAG_C; self.cycles += 6; }  // SEC
            0x58 => { self.p &= !FLAG_I; self.cycles += 6; } // CLI
            0x78 => { self.p |= FLAG_I; self.cycles += 6; }  // SEI
            0xD8 => { self.p &= !FLAG_D; self.cycles += 6; } // CLD
            0xF8 => { self.p |= FLAG_D; self.cycles += 6; }  // SED
            0xB8 => { self.p &= !FLAG_V; self.cycles += 6; } // CLV

            // ── Transfer ──
            0xAA => { // TAX
                if self.x_is_8() {
                    self.x = self.al() as u16;
                    self.set_nz8(self.x as u8);
                } else {
                    self.x = self.a;
                    self.set_nz16(self.x);
                }
                self.cycles += 6;
            }
            0xA8 => { // TAY
                if self.x_is_8() {
                    self.y = self.al() as u16;
                    self.set_nz8(self.y as u8);
                } else {
                    self.y = self.a;
                    self.set_nz16(self.y);
                }
                self.cycles += 6;
            }
            0x8A => { // TXA
                if self.a_is_8() {
                    self.set_al(self.x as u8);
                    self.set_nz8(self.al());
                } else {
                    self.a = self.x;
                    self.set_nz16(self.a);
                }
                self.cycles += 6;
            }
            0x98 => { // TYA
                if self.a_is_8() {
                    self.set_al(self.y as u8);
                    self.set_nz8(self.al());
                } else {
                    self.a = self.y;
                    self.set_nz16(self.a);
                }
                self.cycles += 6;
            }
            0xBA => { // TSX
                if self.x_is_8() {
                    self.x = self.s as u8 as u16;
                    self.set_nz8(self.x as u8);
                } else {
                    self.x = self.s;
                    self.set_nz16(self.x);
                }
                self.cycles += 6;
            }
            0x9A => { // TXS
                if self.emulation {
                    self.s = 0x0100 | self.x as u8 as u16;
                } else {
                    self.s = self.x;
                }
                self.cycles += 6;
            }
            0x9B => { // TXY
                if self.x_is_8() {
                    self.y = self.x as u8 as u16;
                    self.set_nz8(self.y as u8);
                } else {
                    self.y = self.x;
                    self.set_nz16(self.y);
                }
                self.cycles += 6;
            }
            0xBB => { // TYX
                if self.x_is_8() {
                    self.x = self.y as u8 as u16;
                    self.set_nz8(self.x as u8);
                } else {
                    self.x = self.y;
                    self.set_nz16(self.x);
                }
                self.cycles += 6;
            }
            0x5B => { // TCD
                self.d = self.a;
                self.set_nz16(self.d);
                self.cycles += 6;
            }
            0x7B => { // TDC
                self.a = self.d;
                self.set_nz16(self.a);
                self.cycles += 6;
            }
            0x1B => { // TCS
                self.s = self.a;
                if self.emulation { self.s = 0x0100 | (self.s & 0xFF); }
                self.cycles += 6;
            }
            0x3B => { // TSC
                self.a = self.s;
                self.set_nz16(self.a);
                self.cycles += 6;
            }

            // ── Stack push/pull ──
            0x48 => { // PHA
                if self.a_is_8() {
                    self.push8(self.al(), write);
                } else {
                    self.push16(self.a, write);
                }
                self.cycles += 6;
            }
            0x68 => { // PLA
                if self.a_is_8() {
                    let v = self.pull8(read);
                    self.set_al(v);
                    self.set_nz8(v);
                } else {
                    let v = self.pull16(read);
                    self.a = v;
                    self.set_nz16(v);
                }
                self.cycles += 12;
            }
            0xDA => { // PHX
                if self.x_is_8() {
                    self.push8(self.x as u8, write);
                } else {
                    self.push16(self.x, write);
                }
                self.cycles += 6;
            }
            0xFA => { // PLX
                if self.x_is_8() {
                    self.x = self.pull8(read) as u16;
                    self.set_nz8(self.x as u8);
                } else {
                    self.x = self.pull16(read);
                    self.set_nz16(self.x);
                }
                self.cycles += 12;
            }
            0x5A => { // PHY
                if self.x_is_8() {
                    self.push8(self.y as u8, write);
                } else {
                    self.push16(self.y, write);
                }
                self.cycles += 6;
            }
            0x7A => { // PLY
                if self.x_is_8() {
                    self.y = self.pull8(read) as u16;
                    self.set_nz8(self.y as u8);
                } else {
                    self.y = self.pull16(read);
                    self.set_nz16(self.y);
                }
                self.cycles += 12;
            }
            0x08 => { // PHP
                self.push8(self.p, write);
                self.cycles += 6;
            }
            0x28 => { // PLP
                let p = self.pull8(read);
                self.set_p(p);
                self.cycles += 12;
            }
            0x8B => { // PHB
                self.push8(self.dbr, write);
                self.cycles += 6;
            }
            0xAB => { // PLB
                self.dbr = self.pull8(read);
                self.set_nz8(self.dbr);
                self.cycles += 12;
            }
            0x0B => { // PHD
                self.push16(self.d, write);
                self.cycles += 6;
            }
            0x2B => { // PLD
                self.d = self.pull16(read);
                self.set_nz16(self.d);
                self.cycles += 12;
            }
            0x4B => { // PHK
                self.push8(self.pbr, write);
                self.cycles += 6;
            }
            0xF4 => { // PEA abs
                let v = self.fetch16(read);
                self.push16(v, write);
            }
            0xD4 => { // PEI (dp)
                let off = self.fetch8(read) as u16;
                let ptr = self.d.wrapping_add(off) as u32;
                let v = self.read16(ptr, read);
                self.push16(v, write);
            }
            0x62 => { // PER rel16
                let off = self.fetch16(read) as i16;
                let v = self.pc.wrapping_add(off as u16);
                self.push16(v, write);
            }

            // ── REP / SEP ──
            0xC2 => { // REP — reset bits
                let mask = self.fetch8(read);
                self.set_p(self.p & !mask);
                self.cycles += 6;
            }
            0xE2 => { // SEP — set bits
                let mask = self.fetch8(read);
                self.set_p(self.p | mask);
                self.cycles += 6;
            }

            // ── XCE ──
            0xFB => { // Exchange carry and emulation
                let old_c = self.p & FLAG_C != 0;
                let old_e = self.emulation;
                self.emulation = old_c;
                self.p = (self.p & !FLAG_C) | if old_e { FLAG_C } else { 0 };
                if self.emulation {
                    self.p |= FLAG_M | FLAG_X;
                    self.x &= 0xFF;
                    self.y &= 0xFF;
                    self.s = 0x0100 | (self.s & 0xFF);
                }
                self.cycles += 6;
            }

            // ── Block move ──
            0x44 => { // MVP (block move previous — decrementing)
                let dst_bank = self.fetch8(read);
                let src_bank = self.fetch8(read);
                self.dbr = dst_bank;
                let src = (src_bank as u32) << 16 | self.x as u32;
                let dst = (dst_bank as u32) << 16 | self.y as u32;
                let v = read(src);
                write(dst, v);
                if self.x_is_8() {
                    self.x = (self.x as u8).wrapping_sub(1) as u16;
                    self.y = (self.y as u8).wrapping_sub(1) as u16;
                } else {
                    self.x = self.x.wrapping_sub(1);
                    self.y = self.y.wrapping_sub(1);
                }
                self.a = self.a.wrapping_sub(1);
                if self.a != 0xFFFF { self.pc = self.pc.wrapping_sub(3); }
                self.cycles += 6;
            }
            0x54 => { // MVN (block move next — incrementing)
                let dst_bank = self.fetch8(read);
                let src_bank = self.fetch8(read);
                self.dbr = dst_bank;
                let src = (src_bank as u32) << 16 | self.x as u32;
                let dst = (dst_bank as u32) << 16 | self.y as u32;
                let v = read(src);
                write(dst, v);
                if self.x_is_8() {
                    self.x = (self.x as u8).wrapping_add(1) as u16;
                    self.y = (self.y as u8).wrapping_add(1) as u16;
                } else {
                    self.x = self.x.wrapping_add(1);
                    self.y = self.y.wrapping_add(1);
                }
                self.a = self.a.wrapping_sub(1);
                if self.a != 0xFFFF { self.pc = self.pc.wrapping_sub(3); }
                self.cycles += 6;
            }

            // ── NOP, WAI, STP, WDM, BRK, COP ──
            0xEA => { self.cycles += 6; } // NOP
            0xCB => { // WAI
                self.waiting = true;
            }
            0xDB => { // STP
                self.stopped = true;
            }
            0x42 => { // WDM — 2-byte NOP
                let _ = self.fetch8(read);
            }
            0x00 => { // BRK
                let _ = self.fetch8(read); // signature byte
                if !self.emulation { self.push8(self.pbr, write); }
                self.push16(self.pc, write);
                self.push8(self.p, write);
                self.p |= FLAG_I;
                self.p &= !FLAG_D;
                self.pbr = 0;
                let vector = if self.emulation { 0x00FFFE } else { 0x00FFE6 };
                self.pc = self.read16(vector, read);
            }
            0x02 => { // COP
                let _ = self.fetch8(read);
                if !self.emulation { self.push8(self.pbr, write); }
                self.push16(self.pc, write);
                self.push8(self.p, write);
                self.p |= FLAG_I;
                self.p &= !FLAG_D;
                self.pbr = 0;
                let vector = if self.emulation { 0x00FFF4 } else { 0x00FFE4 };
                self.pc = self.read16(vector, read);
            }

            // ── XBA ──
            0xEB => {
                let lo = self.al();
                let hi = self.ah();
                self.a = (lo as u16) << 8 | hi as u16;
                self.set_nz8(hi); // Sets flags based on new low byte
                self.cycles += 6;
            }

            _ => {
                // Unknown opcode — treat as 1-byte NOP
                log::warn!("65C816: unknown opcode ${:02X} at {:02X}:{:04X}", op, self.pbr, self.pc.wrapping_sub(1));
            }
        }
    }

    // ── Grouped memory operation helpers ──

    fn op_adc8_or_16(&mut self, addr: u32, read: &dyn Fn(u32) -> u8) {
        if self.a_is_8() {
            let v = read(addr);
            self.op_adc8(v);
        } else {
            let v = self.read16(addr, read);
            self.op_adc16(v);
        }
    }
    fn op_sbc8_or_16(&mut self, addr: u32, read: &dyn Fn(u32) -> u8) {
        if self.a_is_8() {
            let v = read(addr);
            self.op_sbc8(v);
        } else {
            let v = self.read16(addr, read);
            self.op_sbc16(v);
        }
    }
    fn op_and(&mut self, addr: u32, read: &dyn Fn(u32) -> u8) {
        if self.a_is_8() {
            let v = read(addr);
            self.set_al(self.al() & v);
            self.set_nz8(self.al());
        } else {
            let v = self.read16(addr, read);
            self.a &= v;
            self.set_nz16(self.a);
        }
    }
    fn op_ora(&mut self, addr: u32, read: &dyn Fn(u32) -> u8) {
        if self.a_is_8() {
            let v = read(addr);
            self.set_al(self.al() | v);
            self.set_nz8(self.al());
        } else {
            let v = self.read16(addr, read);
            self.a |= v;
            self.set_nz16(self.a);
        }
    }
    fn op_eor(&mut self, addr: u32, read: &dyn Fn(u32) -> u8) {
        if self.a_is_8() {
            let v = read(addr);
            self.set_al(self.al() ^ v);
            self.set_nz8(self.al());
        } else {
            let v = self.read16(addr, read);
            self.a ^= v;
            self.set_nz16(self.a);
        }
    }
    fn op_cmp_mem(&mut self, addr: u32, read: &dyn Fn(u32) -> u8) {
        if self.a_is_8() {
            let v = read(addr);
            self.op_cmp8(self.al(), v);
        } else {
            let v = self.read16(addr, read);
            self.op_cmp16(self.a, v);
        }
    }
    fn op_cpx_mem(&mut self, addr: u32, read: &dyn Fn(u32) -> u8) {
        if self.x_is_8() {
            let v = read(addr);
            self.op_cmp8(self.x as u8, v);
        } else {
            let v = self.read16(addr, read);
            self.op_cmp16(self.x, v);
        }
    }
    fn op_cpy_mem(&mut self, addr: u32, read: &dyn Fn(u32) -> u8) {
        if self.x_is_8() {
            let v = read(addr);
            self.op_cmp8(self.y as u8, v);
        } else {
            let v = self.read16(addr, read);
            self.op_cmp16(self.y, v);
        }
    }
    fn op_lda(&mut self, addr: u32, read: &dyn Fn(u32) -> u8) {
        if self.a_is_8() {
            let v = read(addr);
            self.set_al(v);
            self.set_nz8(v);
        } else {
            let v = self.read16(addr, read);
            self.a = v;
            self.set_nz16(v);
        }
    }
    fn op_ldx(&mut self, addr: u32, read: &dyn Fn(u32) -> u8) {
        if self.x_is_8() {
            let v = read(addr);
            self.x = v as u16;
            self.set_nz8(v);
        } else {
            let v = self.read16(addr, read);
            self.x = v;
            self.set_nz16(v);
        }
    }
    fn op_ldy(&mut self, addr: u32, read: &dyn Fn(u32) -> u8) {
        if self.x_is_8() {
            let v = read(addr);
            self.y = v as u16;
            self.set_nz8(v);
        } else {
            let v = self.read16(addr, read);
            self.y = v;
            self.set_nz16(v);
        }
    }
    fn op_sta(&self, addr: u32, write: &mut dyn FnMut(u32, u8)) {
        if self.a_is_8() {
            write(addr, self.al());
        } else {
            self.write16(addr, self.a, write);
        }
    }
    fn op_stx(&self, addr: u32, write: &mut dyn FnMut(u32, u8)) {
        if self.x_is_8() {
            write(addr, self.x as u8);
        } else {
            self.write16(addr, self.x, write);
        }
    }
    fn op_sty(&self, addr: u32, write: &mut dyn FnMut(u32, u8)) {
        if self.x_is_8() {
            write(addr, self.y as u8);
        } else {
            self.write16(addr, self.y, write);
        }
    }
    fn op_stz(&self, addr: u32, write: &mut dyn FnMut(u32, u8)) {
        if self.a_is_8() {
            write(addr, 0);
        } else {
            self.write16(addr, 0, write);
        }
    }
    fn op_inc_mem(&mut self, addr: u32, read: &dyn Fn(u32) -> u8, write: &mut dyn FnMut(u32, u8)) {
        if self.a_is_8() {
            let v = read(addr).wrapping_add(1);
            write(addr, v);
            self.set_nz8(v);
        } else {
            let v = self.read16(addr, read).wrapping_add(1);
            self.write16(addr, v, write);
            self.set_nz16(v);
        }
        self.cycles += 6;
    }
    fn op_dec_mem(&mut self, addr: u32, read: &dyn Fn(u32) -> u8, write: &mut dyn FnMut(u32, u8)) {
        if self.a_is_8() {
            let v = read(addr).wrapping_sub(1);
            write(addr, v);
            self.set_nz8(v);
        } else {
            let v = self.read16(addr, read).wrapping_sub(1);
            self.write16(addr, v, write);
            self.set_nz16(v);
        }
        self.cycles += 6;
    }
    fn op_asl_mem(&mut self, addr: u32, read: &dyn Fn(u32) -> u8, write: &mut dyn FnMut(u32, u8)) {
        if self.a_is_8() {
            let v = read(addr);
            self.p = (self.p & !FLAG_C) | if v & 0x80 != 0 { FLAG_C } else { 0 };
            let r = v << 1;
            write(addr, r);
            self.set_nz8(r);
        } else {
            let v = self.read16(addr, read);
            self.p = (self.p & !FLAG_C) | if v & 0x8000 != 0 { FLAG_C } else { 0 };
            let r = v << 1;
            self.write16(addr, r, write);
            self.set_nz16(r);
        }
        self.cycles += 6;
    }
    fn op_lsr_mem(&mut self, addr: u32, read: &dyn Fn(u32) -> u8, write: &mut dyn FnMut(u32, u8)) {
        if self.a_is_8() {
            let v = read(addr);
            self.p = (self.p & !FLAG_C) | (v & 1);
            let r = v >> 1;
            write(addr, r);
            self.set_nz8(r);
        } else {
            let v = self.read16(addr, read);
            self.p = (self.p & !FLAG_C) | (v as u8 & 1);
            let r = v >> 1;
            self.write16(addr, r, write);
            self.set_nz16(r);
        }
        self.cycles += 6;
    }
    fn op_rol_mem(&mut self, addr: u32, read: &dyn Fn(u32) -> u8, write: &mut dyn FnMut(u32, u8)) {
        if self.a_is_8() {
            let v = read(addr);
            let c = self.p & FLAG_C;
            self.p = (self.p & !FLAG_C) | if v & 0x80 != 0 { FLAG_C } else { 0 };
            let r = (v << 1) | c;
            write(addr, r);
            self.set_nz8(r);
        } else {
            let v = self.read16(addr, read);
            let c = (self.p & FLAG_C) as u16;
            self.p = (self.p & !FLAG_C) | if v & 0x8000 != 0 { FLAG_C } else { 0 };
            let r = (v << 1) | c;
            self.write16(addr, r, write);
            self.set_nz16(r);
        }
        self.cycles += 6;
    }
    fn op_ror_mem(&mut self, addr: u32, read: &dyn Fn(u32) -> u8, write: &mut dyn FnMut(u32, u8)) {
        if self.a_is_8() {
            let v = read(addr);
            let c = self.p & FLAG_C;
            self.p = (self.p & !FLAG_C) | (v & 1);
            let r = (v >> 1) | (c << 7);
            write(addr, r);
            self.set_nz8(r);
        } else {
            let v = self.read16(addr, read);
            let c = (self.p & FLAG_C) as u16;
            self.p = (self.p & !FLAG_C) | (v as u8 & 1);
            let r = (v >> 1) | (c << 15);
            self.write16(addr, r, write);
            self.set_nz16(r);
        }
        self.cycles += 6;
    }
    fn op_bit(&mut self, addr: u32, read: &dyn Fn(u32) -> u8) {
        if self.a_is_8() {
            let v = read(addr);
            let r = self.al() & v;
            self.p = (self.p & !(FLAG_N | FLAG_V | FLAG_Z))
                | (v & (FLAG_N | FLAG_V))
                | if r == 0 { FLAG_Z } else { 0 };
        } else {
            let v = self.read16(addr, read);
            let r = self.a & v;
            self.p = (self.p & !(FLAG_N | FLAG_V | FLAG_Z))
                | ((v >> 8) as u8 & (FLAG_N | FLAG_V))
                | if r == 0 { FLAG_Z } else { 0 };
        }
    }
    fn op_trb(&mut self, addr: u32, read: &dyn Fn(u32) -> u8, write: &mut dyn FnMut(u32, u8)) {
        if self.a_is_8() {
            let v = read(addr);
            self.p = (self.p & !FLAG_Z) | if v & self.al() == 0 { FLAG_Z } else { 0 };
            write(addr, v & !self.al());
        } else {
            let v = self.read16(addr, read);
            self.p = (self.p & !FLAG_Z) | if v & self.a == 0 { FLAG_Z } else { 0 };
            self.write16(addr, v & !self.a, write);
        }
        self.cycles += 6;
    }
    fn op_tsb(&mut self, addr: u32, read: &dyn Fn(u32) -> u8, write: &mut dyn FnMut(u32, u8)) {
        if self.a_is_8() {
            let v = read(addr);
            self.p = (self.p & !FLAG_Z) | if v & self.al() == 0 { FLAG_Z } else { 0 };
            write(addr, v | self.al());
        } else {
            let v = self.read16(addr, read);
            self.p = (self.p & !FLAG_Z) | if v & self.a == 0 { FLAG_Z } else { 0 };
            self.write16(addr, v | self.a, write);
        }
        self.cycles += 6;
    }
}
