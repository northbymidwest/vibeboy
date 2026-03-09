mod registers;
pub use registers::Registers;
use crate::bus::Bus;

#[derive(Clone)]
pub struct Cpu {
    pub regs: Registers,
    pub ime: bool,
    pub ime_pending: bool,
    pub halted: bool,
    pub halt_bug: bool,
}

impl Cpu {
    pub fn new() -> Self {
        Cpu {
            regs: Registers::new(),
            ime: false,
            ime_pending: false,
            halted: false,
            halt_bug: false,
        }
    }

    pub fn halted(&self) -> bool {
        self.halted
    }

    pub fn step(&mut self, bus: &mut Bus) -> u32 {
        // Save pending_ime state before anything else (EI delay)
        let pending_ime = self.ime_pending;

        // Handle halted state: HALT NOP cycle with 2T-split IF check.
        // Hardware checks IF at the midpoint (2T) of each HALT NOP cycle.
        // If IF is set in the second half of a HALT NOP, it's not detected
        // until the NEXT HALT NOP's midpoint check, effectively delaying
        // detection by 4T compared to a naive post-tick check.
        let halt_wake = if self.halted {
            bus.tick_half_mcycle(); // first 2T of HALT NOP
            let pending = bus.ie() & bus.if_reg() & 0x1F;
            bus.tick_half_mcycle(); // second 2T of HALT NOP
            bus.step_oam_dma();
            if pending != 0 {
                self.halted = false;
                if self.ime {
                    self.dispatch_interrupt(bus);
                    return 24; // 4T HALT wake + 20T dispatch
                }
                // Wake from HALT without dispatch (IME=false, HALT bug behavior)
                true
            } else {
                return 4; // still halted, NOP cycle
            }
        } else {
            // Not halted: check for interrupt dispatch
            if self.ime {
                let pending = bus.ie() & bus.if_reg() & 0x1F;
                if pending != 0 {
                    self.dispatch_interrupt(bus);
                    return 20;
                }
            }
            false
        };

        // Fetch opcode
        let op = if self.halt_bug {
            // Halt bug: PC is not incremented for the next byte
            self.halt_bug = false;
            let v = bus.read_byte(self.regs.pc);
            bus.tick_mcycle();
            v
        } else {
            self.fetch_byte(bus)
        };

        bus.debug_oam_pc = self.regs.pc.wrapping_sub(1); // PC of current instruction
        let cycles = self.execute(bus, op);

        // Apply EI delay: IME is enabled after the instruction following EI.
        // If DI executed this step it cleared ime_pending, so we skip the apply.
        if pending_ime && self.ime_pending {
            self.ime = true;
            self.ime_pending = false;
        }

        // Account for extra cycles consumed by GDMA/HDMA (bus already ticked)
        let dma_extra = bus.dma_halt_cycles;
        bus.dma_halt_cycles = 0;

        cycles + dma_extra + if halt_wake { 4 } else { 0 }
    }

    fn dispatch_interrupt(&mut self, bus: &mut Bus) {
        self.halted = false;
        self.ime = false;
        bus.tick_mcycle(); // internal 1
        bus.trigger_oam_bug(self.regs.pc);
        bus.trigger_oam_bug(self.regs.sp);
        bus.tick_mcycle(); // internal 2

        let pc = self.regs.pc;

        // Push high byte of PC. If SP-1 happens to be 0xFFFF (IE register),
        // this write can change which interrupt is pending or cancel the dispatch.
        self.regs.sp = self.regs.sp.wrapping_sub(1);
        bus.write_byte(self.regs.sp, (pc >> 8) as u8);
        bus.tick_mcycle();

        // Re-read pending AFTER the high-byte push (IE may have changed).
        let pending = bus.ie() & bus.if_reg() & 0x1F;
        let vector = if pending != 0 {
            let bit = pending.trailing_zeros() as u8;
            *bus.if_mut() &= !(1 << bit); // clear the winning interrupt
            match bit {
                0 => 0x0040u16, // VBlank
                1 => 0x0048,    // STAT
                2 => 0x0050,    // Timer
                3 => 0x0058,    // Serial
                4 => 0x0060,    // Joypad
                _ => 0x0040,
            }
        } else {
            // All interrupt bits cleared from IE — dispatch cancelled, jump to $0000.
            0x0000
        };

        // Push low byte of PC (may also land on 0xFFFF, but we don't re-check).
        self.regs.sp = self.regs.sp.wrapping_sub(1);
        bus.write_byte(self.regs.sp, (pc & 0xFF) as u8);
        bus.tick_mcycle();

        bus.tick_mcycle(); // vector fetch (internal)
        self.regs.pc = vector;
    }

    fn push(&mut self, bus: &mut Bus, val: u16) {
        self.regs.sp = self.regs.sp.wrapping_sub(1);
        bus.write_byte(self.regs.sp, (val >> 8) as u8);
        bus.tick_mcycle();
        self.regs.sp = self.regs.sp.wrapping_sub(1);
        bus.write_byte(self.regs.sp, (val & 0xFF) as u8);
        bus.tick_mcycle();
    }

    fn pop(&mut self, bus: &mut Bus) -> u16 {
        // POP has no write-style OAM bug. OAM bug triggers only through
        // the memory read path (read-style OAM corruption).
        bus.trigger_oam_bug_read(self.regs.sp);
        let lo = bus.read_byte(self.regs.sp) as u16;
        self.regs.sp = self.regs.sp.wrapping_add(1);
        bus.tick_mcycle();
        bus.trigger_oam_bug_read(self.regs.sp);
        let hi = bus.read_byte(self.regs.sp) as u16;
        self.regs.sp = self.regs.sp.wrapping_add(1);
        bus.tick_mcycle();
        (hi << 8) | lo
    }

    fn fetch_byte(&mut self, bus: &mut Bus) -> u8 {
        let v = bus.read_byte(self.regs.pc);
        self.regs.pc = self.regs.pc.wrapping_add(1);
        bus.tick_mcycle();
        v
    }

    fn fetch_word(&mut self, bus: &mut Bus) -> u16 {
        let lo = self.fetch_byte(bus) as u16;
        let hi = self.fetch_byte(bus) as u16;
        (hi << 8) | lo
    }

    fn r8(&self, bus: &mut Bus, id: u8) -> u8 {
        match id {
            0 => self.regs.b,
            1 => self.regs.c,
            2 => self.regs.d,
            3 => self.regs.e,
            4 => self.regs.h,
            5 => self.regs.l,
            6 => {
                let addr = self.regs.hl();
                let v = bus.read_byte(addr);
                bus.tick_mcycle();
                v
            }
            7 => self.regs.a,
            _ => unreachable!(),
        }
    }

    fn set_r8(&mut self, bus: &mut Bus, id: u8, val: u8) {
        match id {
            0 => self.regs.b = val,
            1 => self.regs.c = val,
            2 => self.regs.d = val,
            3 => self.regs.e = val,
            4 => self.regs.h = val,
            5 => self.regs.l = val,
            6 => { bus.write_byte(self.regs.hl(), val); bus.tick_mcycle(); }
            7 => self.regs.a = val,
            _ => unreachable!(),
        }
    }

    fn r16(&self, id: u8) -> u16 {
        match id {
            0 => self.regs.bc(),
            1 => self.regs.de(),
            2 => self.regs.hl(),
            3 => self.regs.sp,
            _ => unreachable!(),
        }
    }

    fn set_r16(&mut self, id: u8, val: u16) {
        match id {
            0 => self.regs.set_bc(val),
            1 => self.regs.set_de(val),
            2 => self.regs.set_hl(val),
            3 => self.regs.sp = val,
            _ => unreachable!(),
        }
    }

    // ALU operations

    /// Add val (+ optional carry) to A, setting all flags.
    fn alu_add(&mut self, val: u8, carry: bool) {
        let c = carry as u16;
        let a = self.regs.a as u16;
        let v = val as u16;
        let result = a + v + c;
        self.regs.set_flags(
            result as u8 == 0,
            false,
            (a & 0xF) + (v & 0xF) + c > 0xF,
            result > 0xFF,
        );
        self.regs.a = result as u8;
    }

    /// Subtract val (+ optional carry) from A, setting all flags. Returns result WITHOUT storing to A.
    fn alu_sub(&mut self, val: u8, carry: bool) -> u8 {
        let c = carry as u16;
        let a = self.regs.a as u16;
        let v = val as u16;
        let result = a.wrapping_sub(v).wrapping_sub(c);
        self.regs.set_flags(
            result as u8 == 0,
            true,
            (a & 0xF) < (v & 0xF) + c,
            a < v + c,
        );
        result as u8
    }

    fn alu_and(&mut self, val: u8) {
        self.regs.a &= val;
        let z = self.regs.a == 0;
        self.regs.set_flags(z, false, true, false);
    }

    fn alu_xor(&mut self, val: u8) {
        self.regs.a ^= val;
        let z = self.regs.a == 0;
        self.regs.set_flags(z, false, false, false);
    }

    fn alu_or(&mut self, val: u8) {
        self.regs.a |= val;
        let z = self.regs.a == 0;
        self.regs.set_flags(z, false, false, false);
    }

    // INC/DEC helpers — preserve C flag, set Z and H accordingly.

    fn inc8(&mut self, val: u8) -> u8 {
        let result = val.wrapping_add(1);
        let z = result == 0;
        let h = (val & 0xF) == 0xF;
        // Preserve C (bit 4 of F), clear N (bit 6), Z (bit 7), H (bit 5)
        self.regs.f &= 0x10;
        if z { self.regs.f |= 0x80; }
        if h { self.regs.f |= 0x20; }
        result
    }

    fn dec8(&mut self, val: u8) -> u8 {
        let result = val.wrapping_sub(1);
        let z = result == 0;
        // H flag: set if there was a borrow from bit 4 (lower nibble was 0)
        let h = (val & 0xF) == 0x0;
        // Preserve C, set N
        self.regs.f &= 0x10;
        self.regs.f |= 0x40; // N = 1
        if z { self.regs.f |= 0x80; }
        if h { self.regs.f |= 0x20; }
        result
    }

    /// ADD HL, r16 — preserves Z flag, clears N, sets H and C.
    fn add_hl(&mut self, val: u16) {
        let hl = self.regs.hl() as u32;
        let v = val as u32;
        let result = hl + v;
        let h = (hl & 0xFFF) + (v & 0xFFF) > 0xFFF;
        let c = result > 0xFFFF;
        // Preserve Z (bit 7), clear N (bit 6), set/clear H (bit 5) and C (bit 4)
        self.regs.f &= 0x80;
        if h { self.regs.f |= 0x20; }
        if c { self.regs.f |= 0x10; }
        self.regs.set_hl(result as u16);
    }

    /// Compute SP + signed byte offset, setting flags Z=0, N=0, H, C.
    /// Returns the resulting address without modifying SP.
    fn add_sp_signed(&mut self, e: u8) -> u16 {
        let sp = self.regs.sp;
        let e16 = e as i8 as i16 as u16;
        let result = sp.wrapping_add(e16);
        // H and C are based on the low byte addition (unsigned)
        let h = (sp & 0xF) + (e as u16 & 0xF) > 0xF;
        let c = (sp & 0xFF) + (e as u16 & 0xFF) > 0xFF;
        self.regs.set_flags(false, false, h, c);
        result
    }

    fn daa(&mut self) {
        let mut a = self.regs.a;
        if !self.regs.flag_n() {
            // High nibble correction first (using original a)
            if self.regs.flag_c() || a > 0x99 {
                a = a.wrapping_add(0x60);
                self.regs.f |= 0x10; // set C
            }
            // Low nibble correction second
            if self.regs.flag_h() || (a & 0x0F) > 9 {
                a = a.wrapping_add(0x06);
            }
        } else {
            if self.regs.flag_c() {
                a = a.wrapping_sub(0x60);
            }
            if self.regs.flag_h() {
                a = a.wrapping_sub(0x06);
            }
        }
        // Keep N and C flags, clear H, update Z
        self.regs.f &= 0x50; // keep N (bit 6) and C (bit 4)
        if a == 0 {
            self.regs.f |= 0x80; // set Z
        }
        self.regs.a = a;
    }

    fn execute_cb(&mut self, bus: &mut Bus) -> u32 {
        let op = self.fetch_byte(bus);
        let reg = op & 0x07;
        let bit = (op >> 3) & 0x07;
        let val = self.r8(bus, reg);
        let is_hl = reg == 6;

        match op >> 6 {
            0 => {
                // Rotate/shift group
                let result = match (op >> 3) & 0x07 {
                    0 => {
                        // RLC: rotate left through copy (new bit 0 = old bit 7)
                        let c = val >> 7;
                        let r = (val << 1) | c;
                        self.regs.set_flags(r == 0, false, false, c != 0);
                        r
                    }
                    1 => {
                        // RRC: rotate right through copy (new bit 7 = old bit 0)
                        let c = val & 1;
                        let r = (val >> 1) | (c << 7);
                        self.regs.set_flags(r == 0, false, false, c != 0);
                        r
                    }
                    2 => {
                        // RL: rotate left through carry
                        let old_c = self.regs.flag_c() as u8;
                        let c = val >> 7;
                        let r = (val << 1) | old_c;
                        self.regs.set_flags(r == 0, false, false, c != 0);
                        r
                    }
                    3 => {
                        // RR: rotate right through carry
                        let old_c = self.regs.flag_c() as u8;
                        let c = val & 1;
                        let r = (val >> 1) | (old_c << 7);
                        self.regs.set_flags(r == 0, false, false, c != 0);
                        r
                    }
                    4 => {
                        // SLA: shift left arithmetic (bit 0 = 0)
                        let c = val >> 7;
                        let r = val << 1;
                        self.regs.set_flags(r == 0, false, false, c != 0);
                        r
                    }
                    5 => {
                        // SRA: shift right arithmetic (bit 7 preserved)
                        let c = val & 1;
                        let r = (val >> 1) | (val & 0x80);
                        self.regs.set_flags(r == 0, false, false, c != 0);
                        r
                    }
                    6 => {
                        // SWAP: swap upper and lower nibbles
                        let r = (val >> 4) | (val << 4);
                        self.regs.set_flags(r == 0, false, false, false);
                        r
                    }
                    7 => {
                        // SRL: shift right logical (bit 7 = 0)
                        let c = val & 1;
                        let r = val >> 1;
                        self.regs.set_flags(r == 0, false, false, c != 0);
                        r
                    }
                    _ => unreachable!(),
                };
                self.set_r8(bus, reg, result);
                if is_hl { 16 } else { 8 }
            }
            1 => {
                // BIT: test bit, Z = NOT(bit), N=0, H=1, C unchanged
                let b = (val >> bit) & 1;
                self.regs.f &= 0x10; // keep C only
                self.regs.f |= 0x20; // set H
                if b == 0 {
                    self.regs.f |= 0x80; // set Z
                }
                if is_hl { 12 } else { 8 }
            }
            2 => {
                // RES: reset bit
                self.set_r8(bus, reg, val & !(1 << bit));
                if is_hl { 16 } else { 8 }
            }
            3 => {
                // SET: set bit
                self.set_r8(bus, reg, val | (1 << bit));
                if is_hl { 16 } else { 8 }
            }
            _ => unreachable!(),
        }
    }

    fn execute(&mut self, bus: &mut Bus, op: u8) -> u32 {
        match op {
            // ---- 0x00-0x0F ----
            0x00 => 4, // NOP

            0x01 => {
                // LD BC, n16
                let v = self.fetch_word(bus);
                self.regs.set_bc(v);
                12
            }
            0x02 => {
                // LD (BC), A
                bus.write_byte(self.regs.bc(), self.regs.a);
                bus.tick_mcycle();
                8
            }
            0x03 => {
                // INC BC
                let v = self.regs.bc();
                bus.trigger_oam_bug(v);
                bus.tick_mcycle();
                self.regs.set_bc(v.wrapping_add(1));
                8
            }
            0x04 => {
                // INC B
                let v = self.inc8(self.regs.b);
                self.regs.b = v;
                4
            }
            0x05 => {
                // DEC B
                let v = self.dec8(self.regs.b);
                self.regs.b = v;
                4
            }
            0x06 => {
                // LD B, n8
                self.regs.b = self.fetch_byte(bus);
                8
            }
            0x07 => {
                // RLCA: rotate A left, old bit 7 goes to C and bit 0
                let c = self.regs.a >> 7;
                self.regs.a = (self.regs.a << 1) | c;
                self.regs.set_flags(false, false, false, c != 0);
                4
            }
            0x08 => {
                // LD (a16), SP
                let addr = self.fetch_word(bus);
                bus.write_byte(addr, (self.regs.sp & 0xFF) as u8);
                bus.tick_mcycle();
                bus.write_byte(addr.wrapping_add(1), (self.regs.sp >> 8) as u8);
                bus.tick_mcycle();
                20
            }
            0x09 => {
                // ADD HL, BC
                let v = self.regs.bc();
                self.add_hl(v);
                bus.tick_mcycle();
                8
            }
            0x0A => {
                // LD A, (BC)
                self.regs.a = bus.read_byte(self.regs.bc());
                bus.tick_mcycle();
                8
            }
            0x0B => {
                // DEC BC
                let v = self.regs.bc();
                bus.trigger_oam_bug(v);
                bus.tick_mcycle();
                self.regs.set_bc(v.wrapping_sub(1));
                8
            }
            0x0C => {
                // INC C
                let v = self.inc8(self.regs.c);
                self.regs.c = v;
                4
            }
            0x0D => {
                // DEC C
                let v = self.dec8(self.regs.c);
                self.regs.c = v;
                4
            }
            0x0E => {
                // LD C, n8
                self.regs.c = self.fetch_byte(bus);
                8
            }
            0x0F => {
                // RRCA: rotate A right, old bit 0 goes to C and bit 7
                let c = self.regs.a & 1;
                self.regs.a = (self.regs.a >> 1) | (c << 7);
                self.regs.set_flags(false, false, false, c != 0);
                4
            }

            // ---- 0x10-0x1F ----
            0x10 => {
                // STOP: consume 0x00 argument, then perform speed switch if KEY1 bit0 set
                let _next = self.fetch_byte(bus);
                bus.do_speed_switch();
                4
            }
            0x11 => {
                // LD DE, n16
                let v = self.fetch_word(bus);
                self.regs.set_de(v);
                12
            }
            0x12 => {
                // LD (DE), A
                bus.write_byte(self.regs.de(), self.regs.a);
                bus.tick_mcycle();
                8
            }
            0x13 => {
                // INC DE
                let v = self.regs.de();
                bus.trigger_oam_bug(v);
                bus.tick_mcycle();
                self.regs.set_de(v.wrapping_add(1));
                8
            }
            0x14 => {
                // INC D
                let v = self.inc8(self.regs.d);
                self.regs.d = v;
                4
            }
            0x15 => {
                // DEC D
                let v = self.dec8(self.regs.d);
                self.regs.d = v;
                4
            }
            0x16 => {
                // LD D, n8
                self.regs.d = self.fetch_byte(bus);
                8
            }
            0x17 => {
                // RLA: rotate A left through carry
                let c = self.regs.a >> 7;
                let old_c = self.regs.flag_c() as u8;
                self.regs.a = (self.regs.a << 1) | old_c;
                self.regs.set_flags(false, false, false, c != 0);
                4
            }
            0x18 => {
                // JR e8: unconditional relative jump
                let e = self.fetch_byte(bus) as i8;
                bus.tick_mcycle();
                bus.trigger_oam_bug(self.regs.pc);
                self.regs.pc = self.regs.pc.wrapping_add(e as u16);
                12
            }
            0x19 => {
                // ADD HL, DE
                let v = self.regs.de();
                self.add_hl(v);
                bus.tick_mcycle();
                8
            }
            0x1A => {
                // LD A, (DE)
                self.regs.a = bus.read_byte(self.regs.de());
                bus.tick_mcycle();
                8
            }
            0x1B => {
                // DEC DE
                let v = self.regs.de();
                bus.trigger_oam_bug(v);
                bus.tick_mcycle();
                self.regs.set_de(v.wrapping_sub(1));
                8
            }
            0x1C => {
                // INC E
                let v = self.inc8(self.regs.e);
                self.regs.e = v;
                4
            }
            0x1D => {
                // DEC E
                let v = self.dec8(self.regs.e);
                self.regs.e = v;
                4
            }
            0x1E => {
                // LD E, n8
                self.regs.e = self.fetch_byte(bus);
                8
            }
            0x1F => {
                // RRA: rotate A right through carry
                let c = self.regs.a & 1;
                let old_c = self.regs.flag_c() as u8;
                self.regs.a = (self.regs.a >> 1) | (old_c << 7);
                self.regs.set_flags(false, false, false, c != 0);
                4
            }

            // ---- 0x20-0x2F ----
            0x20 => {
                // JR NZ, e8
                let e = self.fetch_byte(bus) as i8;
                if !self.regs.flag_z() {
                    bus.tick_mcycle();
                    bus.trigger_oam_bug(self.regs.pc);
                    self.regs.pc = self.regs.pc.wrapping_add(e as u16);
                    12
                } else {
                    8
                }
            }
            0x21 => {
                // LD HL, n16
                let v = self.fetch_word(bus);
                self.regs.set_hl(v);
                12
            }
            0x22 => {
                // LD (HL+), A
                let hl = self.regs.hl();
                bus.write_byte(hl, self.regs.a);
                bus.tick_mcycle();
                self.regs.set_hl(hl.wrapping_add(1));
                8
            }
            0x23 => {
                // INC HL
                let v = self.regs.hl();
                bus.trigger_oam_bug(v);
                bus.tick_mcycle();
                self.regs.set_hl(v.wrapping_add(1));
                8
            }
            0x24 => {
                // INC H
                let v = self.inc8(self.regs.h);
                self.regs.h = v;
                4
            }
            0x25 => {
                // DEC H
                let v = self.dec8(self.regs.h);
                self.regs.h = v;
                4
            }
            0x26 => {
                // LD H, n8
                self.regs.h = self.fetch_byte(bus);
                8
            }
            0x27 => {
                // DAA
                self.daa();
                4
            }
            0x28 => {
                // JR Z, e8
                let e = self.fetch_byte(bus) as i8;
                if self.regs.flag_z() {
                    bus.tick_mcycle();
                    bus.trigger_oam_bug(self.regs.pc);
                    self.regs.pc = self.regs.pc.wrapping_add(e as u16);
                    12
                } else {
                    8
                }
            }
            0x29 => {
                // ADD HL, HL
                let v = self.regs.hl();
                self.add_hl(v);
                bus.tick_mcycle();
                8
            }
            0x2A => {
                // LD A, (HL+)
                let hl = self.regs.hl();
                bus.trigger_oam_bug_read(hl);
                self.regs.a = bus.read_byte(hl);
                bus.tick_mcycle();
                self.regs.set_hl(hl.wrapping_add(1));
                8
            }
            0x2B => {
                // DEC HL
                let v = self.regs.hl();
                bus.trigger_oam_bug(v);
                bus.tick_mcycle();
                self.regs.set_hl(v.wrapping_sub(1));
                8
            }
            0x2C => {
                // INC L
                let v = self.inc8(self.regs.l);
                self.regs.l = v;
                4
            }
            0x2D => {
                // DEC L
                let v = self.dec8(self.regs.l);
                self.regs.l = v;
                4
            }
            0x2E => {
                // LD L, n8
                self.regs.l = self.fetch_byte(bus);
                8
            }
            0x2F => {
                // CPL: complement A, set N and H
                self.regs.a = !self.regs.a;
                self.regs.f |= 0x60; // set N (bit 6) and H (bit 5)
                4
            }

            // ---- 0x30-0x3F ----
            0x30 => {
                // JR NC, e8
                let e = self.fetch_byte(bus) as i8;
                if !self.regs.flag_c() {
                    bus.tick_mcycle();
                    bus.trigger_oam_bug(self.regs.pc);
                    self.regs.pc = self.regs.pc.wrapping_add(e as u16);
                    12
                } else {
                    8
                }
            }
            0x31 => {
                // LD SP, n16
                self.regs.sp = self.fetch_word(bus);
                12
            }
            0x32 => {
                // LD (HL-), A
                let hl = self.regs.hl();
                bus.write_byte(hl, self.regs.a);
                bus.tick_mcycle();
                self.regs.set_hl(hl.wrapping_sub(1));
                8
            }
            0x33 => {
                // INC SP
                let v = self.regs.sp;
                bus.trigger_oam_bug(v);
                bus.tick_mcycle();
                self.regs.sp = v.wrapping_add(1);
                8
            }
            0x34 => {
                // INC (HL)
                let hl = self.regs.hl();
                let v = bus.read_byte(hl);
                bus.tick_mcycle();
                let r = self.inc8(v);
                bus.write_byte(hl, r);
                bus.tick_mcycle();
                12
            }
            0x35 => {
                // DEC (HL)
                let hl = self.regs.hl();
                let v = bus.read_byte(hl);
                bus.tick_mcycle();
                let r = self.dec8(v);
                bus.write_byte(hl, r);
                bus.tick_mcycle();
                12
            }
            0x36 => {
                // LD (HL), n8
                let n = self.fetch_byte(bus);
                bus.write_byte(self.regs.hl(), n);
                bus.tick_mcycle();
                12
            }
            0x37 => {
                // SCF: set carry flag, clear N and H, preserve Z
                self.regs.f &= 0x80; // keep Z only
                self.regs.f |= 0x10; // set C
                4
            }
            0x38 => {
                // JR C, e8
                let e = self.fetch_byte(bus) as i8;
                if self.regs.flag_c() {
                    bus.tick_mcycle();
                    bus.trigger_oam_bug(self.regs.pc);
                    self.regs.pc = self.regs.pc.wrapping_add(e as u16);
                    12
                } else {
                    8
                }
            }
            0x39 => {
                // ADD HL, SP
                let v = self.regs.sp;
                self.add_hl(v);
                bus.tick_mcycle();
                8
            }
            0x3A => {
                // LD A, (HL-)
                let hl = self.regs.hl();
                bus.trigger_oam_bug_read(hl);
                self.regs.a = bus.read_byte(hl);
                bus.tick_mcycle();
                self.regs.set_hl(hl.wrapping_sub(1));
                8
            }
            0x3B => {
                // DEC SP
                let v = self.regs.sp;
                bus.trigger_oam_bug(v);
                bus.tick_mcycle();
                self.regs.sp = v.wrapping_sub(1);
                8
            }
            0x3C => {
                // INC A
                let v = self.inc8(self.regs.a);
                self.regs.a = v;
                4
            }
            0x3D => {
                // DEC A
                let v = self.dec8(self.regs.a);
                self.regs.a = v;
                4
            }
            0x3E => {
                // LD A, n8
                self.regs.a = self.fetch_byte(bus);
                8
            }
            0x3F => {
                // CCF: complement carry flag, Z unchanged, N=0, H=0
                let z = self.regs.flag_z();
                let c = !self.regs.flag_c();
                self.regs.set_flags(z, false, false, c);
                4
            }

            // ---- 0x40-0x7F: LD r8, r8 (HALT at 0x76) ----
            0x40..=0x75 | 0x77..=0x7F => {
                let dst = (op >> 3) & 0x07;
                let src = op & 0x07;
                let val = self.r8(bus, src);
                self.set_r8(bus, dst, val);
                // (HL) access costs extra 4 cycles
                if src == 6 || dst == 6 { 8 } else { 4 }
            }

            0x76 => {
                // HALT
                if !self.ime {
                    let pending = bus.ie() & bus.if_reg() & 0x1F;
                    if pending != 0 {
                        // Halt bug: next byte is read twice
                        self.halt_bug = true;
                    } else {
                        self.halted = true;
                    }
                } else {
                    self.halted = true;
                }
                4
            }

            // ---- 0x80-0xBF: ALU A, r8 ----
            0x80..=0xBF => {
                let src = op & 0x07;
                let val = self.r8(bus, src);
                let cycles = if src == 6 { 8 } else { 4 };
                match (op >> 3) & 0x07 {
                    0 => {
                        // ADD A, r8
                        self.alu_add(val, false);
                    }
                    1 => {
                        // ADC A, r8 — capture carry before alu_add modifies flags
                        let c = self.regs.flag_c();
                        self.alu_add(val, c);
                    }
                    2 => {
                        // SUB A, r8
                        let r = self.alu_sub(val, false);
                        self.regs.a = r;
                    }
                    3 => {
                        // SBC A, r8 — capture carry before alu_sub modifies flags
                        let c = self.regs.flag_c();
                        let r = self.alu_sub(val, c);
                        self.regs.a = r;
                    }
                    4 => {
                        // AND A, r8
                        self.alu_and(val);
                    }
                    5 => {
                        // XOR A, r8
                        self.alu_xor(val);
                    }
                    6 => {
                        // OR A, r8
                        self.alu_or(val);
                    }
                    7 => {
                        // CP A, r8: set flags as SUB but discard result
                        self.alu_sub(val, false);
                    }
                    _ => unreachable!(),
                }
                cycles
            }

            // ---- 0xC0-0xFF: Control / misc ----
            0xC0 => {
                // RET NZ
                bus.tick_mcycle(); // internal (condition check)
                if !self.regs.flag_z() {
                    let pc = self.pop(bus); // 2 M-cycles
                    bus.tick_mcycle(); // internal after pop
                    self.regs.pc = pc;
                    20
                } else {
                    8
                }
            }
            0xC1 => {
                // POP BC
                let v = self.pop(bus);
                self.regs.set_bc(v);
                12
            }
            0xC2 => {
                // JP NZ, a16
                let addr = self.fetch_word(bus);
                if !self.regs.flag_z() {
                    self.regs.pc = addr;
                    bus.tick_mcycle();
                    16
                } else {
                    12
                }
            }
            0xC3 => {
                // JP a16
                let addr = self.fetch_word(bus);
                self.regs.pc = addr;
                bus.tick_mcycle();
                16
            }
            0xC4 => {
                // CALL NZ, a16
                let addr = self.fetch_word(bus);
                if !self.regs.flag_z() {
                    let pc = self.regs.pc;
                    bus.tick_mcycle();
                    bus.trigger_oam_bug(self.regs.sp);
                    self.push(bus, pc);
                    self.regs.pc = addr;
                    24
                } else {
                    12
                }
            }
            0xC5 => {
                // PUSH BC
                let v = self.regs.bc();
                bus.trigger_oam_bug(self.regs.sp);
                bus.tick_mcycle();
                self.push(bus, v);
                16
            }
            0xC6 => {
                // ADD A, n8
                let n = self.fetch_byte(bus);
                self.alu_add(n, false);
                8
            }
            0xC7 => {
                // RST 00H
                let pc = self.regs.pc;
                bus.tick_mcycle();
                bus.trigger_oam_bug(self.regs.sp);
                self.push(bus, pc);
                self.regs.pc = 0x0000;
                16
            }
            0xC8 => {
                // RET Z
                bus.tick_mcycle(); // internal (condition check)
                if self.regs.flag_z() {
                    let pc = self.pop(bus); // 2 M-cycles
                    bus.tick_mcycle(); // internal after pop
                    self.regs.pc = pc;
                    20
                } else {
                    8
                }
            }
            0xC9 => {
                // RET
                let pc = self.pop(bus); // 2 M-cycles
                bus.tick_mcycle(); // internal
                self.regs.pc = pc;
                16
            }
            0xCA => {
                // JP Z, a16
                let addr = self.fetch_word(bus);
                if self.regs.flag_z() {
                    self.regs.pc = addr;
                    bus.tick_mcycle();
                    16
                } else {
                    12
                }
            }
            0xCB => {
                // PREFIX CB
                self.execute_cb(bus)
            }
            0xCC => {
                // CALL Z, a16
                let addr = self.fetch_word(bus);
                if self.regs.flag_z() {
                    let pc = self.regs.pc;
                    bus.tick_mcycle();
                    bus.trigger_oam_bug(self.regs.sp);
                    self.push(bus, pc);
                    self.regs.pc = addr;
                    24
                } else {
                    12
                }
            }
            0xCD => {
                // CALL a16
                let addr = self.fetch_word(bus);
                let pc = self.regs.pc;
                bus.tick_mcycle();
                bus.trigger_oam_bug(self.regs.sp);
                self.push(bus, pc);
                self.regs.pc = addr;
                24
            }
            0xCE => {
                // ADC A, n8
                let n = self.fetch_byte(bus);
                let c = self.regs.flag_c();
                self.alu_add(n, c);
                8
            }
            0xCF => {
                // RST 08H
                let pc = self.regs.pc;
                bus.tick_mcycle();
                bus.trigger_oam_bug(self.regs.sp);
                self.push(bus, pc);
                self.regs.pc = 0x0008;
                16
            }

            0xD0 => {
                // RET NC
                bus.tick_mcycle(); // internal (condition check)
                if !self.regs.flag_c() {
                    let pc = self.pop(bus); // 2 M-cycles
                    bus.tick_mcycle(); // internal after pop
                    self.regs.pc = pc;
                    20
                } else {
                    8
                }
            }
            0xD1 => {
                // POP DE
                let v = self.pop(bus);
                self.regs.set_de(v);
                12
            }
            0xD2 => {
                // JP NC, a16
                let addr = self.fetch_word(bus);
                if !self.regs.flag_c() {
                    self.regs.pc = addr;
                    bus.tick_mcycle();
                    16
                } else {
                    12
                }
            }
            0xD3 => 4, // ILLEGAL (undefined on SM83)
            0xD4 => {
                // CALL NC, a16
                let addr = self.fetch_word(bus);
                if !self.regs.flag_c() {
                    let pc = self.regs.pc;
                    bus.tick_mcycle();
                    bus.trigger_oam_bug(self.regs.sp);
                    self.push(bus, pc);
                    self.regs.pc = addr;
                    24
                } else {
                    12
                }
            }
            0xD5 => {
                // PUSH DE
                let v = self.regs.de();
                bus.trigger_oam_bug(self.regs.sp);
                bus.tick_mcycle();
                self.push(bus, v);
                16
            }
            0xD6 => {
                // SUB A, n8
                let n = self.fetch_byte(bus);
                let r = self.alu_sub(n, false);
                self.regs.a = r;
                8
            }
            0xD7 => {
                // RST 10H
                let pc = self.regs.pc;
                bus.tick_mcycle();
                bus.trigger_oam_bug(self.regs.sp);
                self.push(bus, pc);
                self.regs.pc = 0x0010;
                16
            }
            0xD8 => {
                // RET C
                bus.tick_mcycle(); // internal (condition check)
                if self.regs.flag_c() {
                    let pc = self.pop(bus); // 2 M-cycles
                    bus.tick_mcycle(); // internal after pop
                    self.regs.pc = pc;
                    20
                } else {
                    8
                }
            }
            0xD9 => {
                // RETI: return and enable interrupts
                let pc = self.pop(bus); // 2 M-cycles
                bus.tick_mcycle(); // internal
                self.regs.pc = pc;
                self.ime = true;
                16
            }
            0xDA => {
                // JP C, a16
                let addr = self.fetch_word(bus);
                if self.regs.flag_c() {
                    self.regs.pc = addr;
                    bus.tick_mcycle();
                    16
                } else {
                    12
                }
            }
            0xDB => 4, // ILLEGAL
            0xDC => {
                // CALL C, a16
                let addr = self.fetch_word(bus);
                if self.regs.flag_c() {
                    let pc = self.regs.pc;
                    bus.tick_mcycle();
                    bus.trigger_oam_bug(self.regs.sp);
                    self.push(bus, pc);
                    self.regs.pc = addr;
                    24
                } else {
                    12
                }
            }
            0xDD => 4, // ILLEGAL
            0xDE => {
                // SBC A, n8
                let n = self.fetch_byte(bus);
                let c = self.regs.flag_c();
                let r = self.alu_sub(n, c);
                self.regs.a = r;
                8
            }
            0xDF => {
                // RST 18H
                let pc = self.regs.pc;
                bus.tick_mcycle();
                bus.trigger_oam_bug(self.regs.sp);
                self.push(bus, pc);
                self.regs.pc = 0x0018;
                16
            }

            0xE0 => {
                // LDH (a8), A  — write A to 0xFF00 + n
                let n = self.fetch_byte(bus);
                bus.write_byte(0xFF00 | n as u16, self.regs.a);
                bus.tick_mcycle();
                12
            }
            0xE1 => {
                // POP HL
                let v = self.pop(bus);
                self.regs.set_hl(v);
                12
            }
            0xE2 => {
                // LD (C), A  — write A to 0xFF00 + C
                bus.write_byte(0xFF00 | self.regs.c as u16, self.regs.a);
                bus.tick_mcycle();
                8
            }
            0xE3 => 4, // ILLEGAL
            0xE4 => 4, // ILLEGAL
            0xE5 => {
                // PUSH HL
                let v = self.regs.hl();
                bus.trigger_oam_bug(self.regs.sp);
                bus.tick_mcycle();
                self.push(bus, v);
                16
            }
            0xE6 => {
                // AND A, n8
                let n = self.fetch_byte(bus);
                self.alu_and(n);
                8
            }
            0xE7 => {
                // RST 20H
                let pc = self.regs.pc;
                bus.tick_mcycle();
                bus.trigger_oam_bug(self.regs.sp);
                self.push(bus, pc);
                self.regs.pc = 0x0020;
                16
            }
            0xE8 => {
                // ADD SP, e8
                let e = self.fetch_byte(bus);
                bus.tick_mcycle();
                bus.tick_mcycle();
                self.regs.sp = self.add_sp_signed(e);
                16
            }
            0xE9 => {
                // JP HL
                self.regs.pc = self.regs.hl();
                4
            }
            0xEA => {
                // LD (a16), A
                let addr = self.fetch_word(bus);
                bus.write_byte(addr, self.regs.a);
                bus.tick_mcycle();
                16
            }
            0xEB => 4, // ILLEGAL
            0xEC => 4, // ILLEGAL
            0xED => 4, // ILLEGAL
            0xEE => {
                // XOR A, n8
                let n = self.fetch_byte(bus);
                self.alu_xor(n);
                8
            }
            0xEF => {
                // RST 28H
                let pc = self.regs.pc;
                bus.tick_mcycle();
                bus.trigger_oam_bug(self.regs.sp);
                self.push(bus, pc);
                self.regs.pc = 0x0028;
                16
            }

            0xF0 => {
                // LDH A, (a8)  — read from 0xFF00 + n into A
                let n = self.fetch_byte(bus);
                self.regs.a = bus.read_byte(0xFF00 | n as u16);
                bus.tick_mcycle();
                12
            }
            0xF1 => {
                // POP AF
                let v = self.pop(bus);
                self.regs.set_af(v);
                12
            }
            0xF2 => {
                // LD A, (C)  — read from 0xFF00 + C into A
                self.regs.a = bus.read_byte(0xFF00 | self.regs.c as u16);
                bus.tick_mcycle();
                8
            }
            0xF3 => {
                // DI: disable interrupts immediately
                self.ime = false;
                self.ime_pending = false;
                4
            }
            0xF4 => 4, // ILLEGAL
            0xF5 => {
                // PUSH AF
                let v = self.regs.af();
                bus.trigger_oam_bug(self.regs.sp);
                bus.tick_mcycle();
                self.push(bus, v);
                16
            }
            0xF6 => {
                // OR A, n8
                let n = self.fetch_byte(bus);
                self.alu_or(n);
                8
            }
            0xF7 => {
                // RST 30H
                let pc = self.regs.pc;
                bus.tick_mcycle();
                bus.trigger_oam_bug(self.regs.sp);
                self.push(bus, pc);
                self.regs.pc = 0x0030;
                16
            }
            0xF8 => {
                // LD HL, SP+e8
                let e = self.fetch_byte(bus);
                bus.tick_mcycle();
                let v = self.add_sp_signed(e);
                self.regs.set_hl(v);
                12
            }
            0xF9 => {
                // LD SP, HL
                self.regs.sp = self.regs.hl();
                bus.tick_mcycle();
                bus.trigger_oam_bug(self.regs.hl());
                8
            }
            0xFA => {
                // LD A, (a16)
                let addr = self.fetch_word(bus);
                self.regs.a = bus.read_byte(addr);
                bus.tick_mcycle();
                16
            }
            0xFB => {
                // EI: enable interrupts after the next instruction
                self.ime_pending = true;
                4
            }
            0xFC => 4, // ILLEGAL
            0xFD => 4, // ILLEGAL
            0xFE => {
                // CP A, n8
                let n = self.fetch_byte(bus);
                self.alu_sub(n, false); // sets flags, result discarded
                8
            }
            0xFF => {
                // RST 38H
                let pc = self.regs.pc;
                bus.tick_mcycle();
                bus.trigger_oam_bug(self.regs.sp);
                self.push(bus, pc);
                self.regs.pc = 0x0038;
                16
            }
        }
    }
}
