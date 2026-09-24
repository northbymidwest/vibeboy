mod registers;
pub use registers::Registers;

/// One M-cycle operation returned by `Cpu::mcycle()`.
/// The emulator loop services each op using the bus, then calls `mcycle()` again.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum McycleOp {
    /// Read a byte from `addr`. Emulator services it with `bus.tick_read(addr)`
    /// and stores the result in `cpu.data_latch`.
    Read { addr: u16 },
    /// Write `val` to `addr`. Emulator calls `bus.tick_write(addr, val)`.
    Write { addr: u16, val: u8 },
    /// Internal cycle (no memory access). Emulator calls `bus.tick_internal()`.
    Internal,
    /// HALT NOP cycle. Emulator does the split half-mcycle IF check.
    HaltNop,
    /// Speed switch idle cycle. Emulator calls `bus.tick_speed_switch_idle()`.
    SpeedSwitchIdle,
    /// Instruction complete — no M-cycle to tick for this call.
    Done,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Cpu {
    pub regs: Registers,
    pub ime: bool,
    pub ime_pending: bool,
    pub halted: bool,
    pub halt_bug: bool,
    /// Remaining M-cycles to idle during a CGB speed switch.
    pub speed_switch_remaining: u32,
    /// M-cycle count at which to toggle the speed (counted down from initial).
    pub speed_switch_toggle_at: u32,

    // ── M-cycle state machine fields ──────────────────────────────────────────
    /// Current opcode being executed.
    #[serde(default)]
    pub(crate) opcode: u8,
    /// CB-prefixed opcode (valid when opcode == 0xCB and phase >= 3).
    #[serde(default)]
    cb_opcode: u8,
    /// Phase counter within current instruction. 0 = fetch next opcode.
    #[serde(default)]
    pub(crate) phase: u8,
    /// Result of the last Read operation, set by the emulator loop.
    #[serde(default)]
    pub data_latch: u8,
    /// Inter-phase scratch byte.
    #[serde(default)]
    tmp8: u8,
    /// Inter-phase scratch word.
    #[serde(default)]
    pub(crate) tmp16: u16,
    /// True when executing an interrupt dispatch sequence.
    #[serde(default)]
    pub(crate) in_interrupt: bool,
    /// Phase counter within interrupt dispatch (0..=4).
    #[serde(default)]
    pub(crate) interrupt_phase: u8,
    /// Write-style OAM bug address to trigger (set by CPU, consumed by emulator).
    #[serde(default)]
    pub oam_bug_addr: Option<u16>,
    /// Read-style OAM bug address to trigger (set by CPU, consumed by emulator).
    #[serde(default)]
    pub oam_bug_read_addr: Option<u16>,
    /// Saved `ime_pending` state at instruction start for EI delay.
    #[serde(default)]
    pending_ime_at_start: bool,
    /// True when the last mcycle op was the final action of an instruction.
    /// The next mcycle() call will return Done.
    #[serde(default)]
    finishing: bool,
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Cpu {
    pub fn new() -> Self {
        Cpu {
            regs: Registers::new(),
            ime: false,
            ime_pending: false,
            halted: false,
            halt_bug: false,
            speed_switch_remaining: 0,
            speed_switch_toggle_at: 0,
            opcode: 0,
            cb_opcode: 0,
            phase: 0,
            data_latch: 0,
            tmp8: 0,
            tmp16: 0,
            in_interrupt: false,
            interrupt_phase: 0,
            oam_bug_addr: None,
            oam_bug_read_addr: None,
            pending_ime_at_start: false,
            finishing: false,
        }
    }

    pub fn halted(&self) -> bool {
        self.halted
    }

    // ── M-cycle state machine ─────────────────────────────────────────────────

    /// Return one M-cycle operation. The emulator loop must service the returned
    /// op (read/write/tick) and then call `mcycle()` again until `Done` is returned.
    pub fn mcycle(&mut self) -> McycleOp {
        // Terminal action was serviced — finish the instruction
        if self.finishing {
            self.finishing = false;
            return self.finish_instruction();
        }

        // Speed switch idle
        if self.speed_switch_remaining > 0 {
            return McycleOp::SpeedSwitchIdle;
        }

        // HALT NOP
        if self.halted && !self.in_interrupt {
            return McycleOp::HaltNop;
        }

        // Interrupt dispatch
        if self.in_interrupt {
            return self.interrupt_mcycle();
        }

        // Phase 0: start new instruction — emit opcode fetch
        if self.phase == 0 {
            // Save pending_ime state before anything (EI delay)
            self.pending_ime_at_start = self.ime_pending;

            let addr = self.regs.pc;
            self.regs.pc = self.regs.pc.wrapping_add(1);
            self.phase = 1;
            return McycleOp::Read { addr };
        }

        // Phase 1: opcode has been fetched into data_latch
        if self.phase == 1 {
            self.opcode = self.data_latch;
            if self.halt_bug {
                self.halt_bug = false;
                self.regs.pc = self.regs.pc.wrapping_sub(1);
            }
            // Advance to phase 2 for instructions that need more M-cycles,
            // or finish right here for 1-M-cycle instructions.
            self.phase = 2;
        }

        self.execute_phase()
    }

    /// Signal that an interrupt should be dispatched. Called by the emulator
    /// when it detects a pending interrupt (IME=true, IE & IF != 0).
    pub fn begin_interrupt_dispatch(&mut self) {
        self.in_interrupt = true;
        self.interrupt_phase = 0;
        self.ime = false;
    }

    fn interrupt_mcycle(&mut self) -> McycleOp {
        match self.interrupt_phase {
            0 => {
                // Internal cycle 1
                self.interrupt_phase = 1;
                McycleOp::Internal
            }
            1 => {
                // Internal cycle 2 + OAM bug triggers
                self.oam_bug_addr = Some(self.regs.pc);
                // Second OAM bug on SP
                self.tmp16 = self.regs.sp; // save for second OAM bug
                self.interrupt_phase = 2;
                McycleOp::Internal
            }
            2 => {
                // Push PC high byte
                self.regs.sp = self.regs.sp.wrapping_sub(1);
                let val = (self.regs.pc >> 8) as u8;
                self.interrupt_phase = 3;
                McycleOp::Write {
                    addr: self.regs.sp,
                    val,
                }
            }
            3 => {
                // Push PC low byte
                self.regs.sp = self.regs.sp.wrapping_sub(1);
                let val = (self.regs.pc & 0xFF) as u8;
                self.interrupt_phase = 4;
                McycleOp::Write {
                    addr: self.regs.sp,
                    val,
                }
            }
            4 => {
                // Vector fetch (internal) — emulator has set tmp16 to the vector address
                self.regs.pc = self.tmp16;
                self.in_interrupt = false;
                self.interrupt_phase = 0;
                self.phase = 0;
                McycleOp::Internal
            }
            _ => unreachable!(),
        }
    }

    /// Mark the current instruction as finishing with a terminal action.
    /// The next mcycle() call will return Done (via the `finishing` flag).
    fn finish_with(&mut self, op: McycleOp) -> McycleOp {
        self.finishing = true;
        op
    }

    /// Finish the current instruction and return Done, applying EI delay.
    fn finish_instruction(&mut self) -> McycleOp {
        self.phase = 0;
        // Apply EI delay: IME is enabled after the instruction following EI.
        // If DI executed this step it cleared ime_pending, so we skip the apply.
        if self.pending_ime_at_start && self.ime_pending {
            self.ime = true;
            self.ime_pending = false;
        }
        McycleOp::Done
    }

    // ── Register-pair helpers ─────────────────────────────────────────────────

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

    // ── ALU operations (pure CPU-internal, no bus access) ─────────────────────

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

    fn inc8(&mut self, val: u8) -> u8 {
        let result = val.wrapping_add(1);
        let z = result == 0;
        let h = (val & 0xF) == 0xF;
        self.regs.f &= 0x10;
        if z {
            self.regs.f |= 0x80;
        }
        if h {
            self.regs.f |= 0x20;
        }
        result
    }

    fn dec8(&mut self, val: u8) -> u8 {
        let result = val.wrapping_sub(1);
        let z = result == 0;
        let h = (val & 0xF) == 0x0;
        self.regs.f &= 0x10;
        self.regs.f |= 0x40;
        if z {
            self.regs.f |= 0x80;
        }
        if h {
            self.regs.f |= 0x20;
        }
        result
    }

    fn add_hl(&mut self, val: u16) {
        let hl = self.regs.hl() as u32;
        let v = val as u32;
        let result = hl + v;
        let h = (hl & 0xFFF) + (v & 0xFFF) > 0xFFF;
        let c = result > 0xFFFF;
        self.regs.f &= 0x80;
        if h {
            self.regs.f |= 0x20;
        }
        if c {
            self.regs.f |= 0x10;
        }
        self.regs.set_hl(result as u16);
    }

    fn add_sp_signed(&mut self, e: u8) -> u16 {
        let sp = self.regs.sp;
        let e16 = e as i8 as i16 as u16;
        let result = sp.wrapping_add(e16);
        let h = (sp & 0xF) + (e as u16 & 0xF) > 0xF;
        let c = (sp & 0xFF) + (e as u16 & 0xFF) > 0xFF;
        self.regs.set_flags(false, false, h, c);
        result
    }

    fn daa(&mut self) {
        let mut a = self.regs.a;
        if !self.regs.flag_n() {
            if self.regs.flag_c() || a > 0x99 {
                a = a.wrapping_add(0x60);
                self.regs.f |= 0x10;
            }
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
        self.regs.f &= 0x50;
        if a == 0 {
            self.regs.f |= 0x80;
        }
        self.regs.a = a;
    }

    /// Get register value by 3-bit register ID (for mcycle state machine).
    /// Returns the value directly for registers, or 0 for (HL) which needs a read.
    fn get_reg(&self, id: u8) -> u8 {
        match id {
            0 => self.regs.b,
            1 => self.regs.c,
            2 => self.regs.d,
            3 => self.regs.e,
            4 => self.regs.h,
            5 => self.regs.l,
            6 => 0, // (HL) — caller must handle via Read
            7 => self.regs.a,
            _ => unreachable!(),
        }
    }

    /// Set register value by 3-bit register ID (for mcycle state machine).
    /// Does nothing for (HL) id=6 — caller must handle via Write.
    fn set_reg(&mut self, id: u8, val: u8) {
        match id {
            0 => self.regs.b = val,
            1 => self.regs.c = val,
            2 => self.regs.d = val,
            3 => self.regs.e = val,
            4 => self.regs.h = val,
            5 => self.regs.l = val,
            6 => {} // (HL) — caller handles
            7 => self.regs.a = val,
            _ => unreachable!(),
        }
    }

    /// Evaluate condition code (2-bit cc field from opcodes).
    fn check_condition(&self, cc: u8) -> bool {
        match cc {
            0 => !self.regs.flag_z(), // NZ
            1 => self.regs.flag_z(),  // Z
            2 => !self.regs.flag_c(), // NC
            3 => self.regs.flag_c(),  // C
            _ => unreachable!(),
        }
    }

    // ── M-cycle phase execution ───────────────────────────────────────────────

    /// Execute one phase of the current instruction, returning the next McycleOp.
    /// Called when phase >= 2 (opcode already fetched and stored in self.opcode).
    fn execute_phase(&mut self) -> McycleOp {
        let op = self.opcode;
        match op {
            // ══════════════════════════════════════════════════════════════════
            // NOP (1 M-cycle: fetch only)
            // ══════════════════════════════════════════════════════════════════
            0x00 => self.finish_instruction(),

            // ══════════════════════════════════════════════════════════════════
            // LD r16, imm16 — 3 M-cycles: fetch, read lo, read hi
            // 0x01=BC, 0x11=DE, 0x21=HL, 0x31=SP
            // ══════════════════════════════════════════════════════════════════
            0x01 | 0x11 | 0x21 | 0x31 => {
                match self.phase {
                    2 => {
                        // Read low byte of immediate
                        let addr = self.regs.pc;
                        self.regs.pc = self.regs.pc.wrapping_add(1);
                        self.phase = 3;
                        McycleOp::Read { addr }
                    }
                    3 => {
                        self.tmp8 = self.data_latch; // lo
                        let addr = self.regs.pc;
                        self.regs.pc = self.regs.pc.wrapping_add(1);
                        self.phase = 4;
                        McycleOp::Read { addr }
                    }
                    4 => {
                        let hi = self.data_latch;
                        let val = (hi as u16) << 8 | self.tmp8 as u16;
                        let rp = (op >> 4) & 0x03;
                        self.set_r16(rp, val);
                        self.finish_instruction()
                    }
                    _ => unreachable!(),
                }
            }

            // ══════════════════════════════════════════════════════════════════
            // LD (BC), A / LD (DE), A — 2 M-cycles: fetch, write
            // ══════════════════════════════════════════════════════════════════
            0x02 => {
                // LD (BC), A
                self.finish_with(McycleOp::Write {
                    addr: self.regs.bc(),
                    val: self.regs.a,
                })
            }
            0x12 => {
                // LD (DE), A
                self.finish_with(McycleOp::Write {
                    addr: self.regs.de(),
                    val: self.regs.a,
                })
            }

            // ══════════════════════════════════════════════════════════════════
            // INC r16 / DEC r16 — 2 M-cycles: fetch, internal
            // 0x03/0x13/0x23/0x33 = INC, 0x0B/0x1B/0x2B/0x3B = DEC
            // ══════════════════════════════════════════════════════════════════
            0x03 | 0x13 | 0x23 | 0x33 => match self.phase {
                2 => {
                    let rp = (op >> 4) & 0x03;
                    let v = self.r16(rp);
                    self.oam_bug_addr = Some(v);
                    self.tmp16 = v.wrapping_add(1);
                    self.phase = 3;
                    McycleOp::Internal
                }
                3 => {
                    let rp = (op >> 4) & 0x03;
                    self.set_r16(rp, self.tmp16);
                    self.finish_instruction()
                }
                _ => unreachable!(),
            },
            0x0B | 0x1B | 0x2B | 0x3B => match self.phase {
                2 => {
                    let rp = (op >> 4) & 0x03;
                    let v = self.r16(rp);
                    self.oam_bug_addr = Some(v);
                    self.tmp16 = v.wrapping_sub(1);
                    self.phase = 3;
                    McycleOp::Internal
                }
                3 => {
                    let rp = (op >> 4) & 0x03;
                    self.set_r16(rp, self.tmp16);
                    self.finish_instruction()
                }
                _ => unreachable!(),
            },

            // ══════════════════════════════════════════════════════════════════
            // INC r8 / DEC r8 — 1 M-cycle for registers, 3 for (HL)
            // ══════════════════════════════════════════════════════════════════
            0x04 | 0x0C | 0x14 | 0x1C | 0x24 | 0x2C | 0x3C => {
                // INC r8
                let reg = (op >> 3) & 0x07;
                if reg == 6 {
                    // INC (HL): read, modify, write
                    match self.phase {
                        2 => {
                            self.phase = 3;
                            McycleOp::Read {
                                addr: self.regs.hl(),
                            }
                        }
                        3 => {
                            let r = self.inc8(self.data_latch);
                            self.phase = 4;
                            McycleOp::Write {
                                addr: self.regs.hl(),
                                val: r,
                            }
                        }
                        4 => self.finish_instruction(),
                        _ => unreachable!(),
                    }
                } else {
                    let v = self.get_reg(reg);
                    let r = self.inc8(v);
                    self.set_reg(reg, r);
                    self.finish_instruction()
                }
            }
            0x05 | 0x0D | 0x15 | 0x1D | 0x25 | 0x2D | 0x3D => {
                // DEC r8
                let reg = (op >> 3) & 0x07;
                if reg == 6 {
                    match self.phase {
                        2 => {
                            self.phase = 3;
                            McycleOp::Read {
                                addr: self.regs.hl(),
                            }
                        }
                        3 => {
                            let r = self.dec8(self.data_latch);
                            self.phase = 4;
                            McycleOp::Write {
                                addr: self.regs.hl(),
                                val: r,
                            }
                        }
                        4 => self.finish_instruction(),
                        _ => unreachable!(),
                    }
                } else {
                    let v = self.get_reg(reg);
                    let r = self.dec8(v);
                    self.set_reg(reg, r);
                    self.finish_instruction()
                }
            }

            // INC (HL) and DEC (HL) are handled by 0x34/0x35 explicitly
            0x34 => {
                // INC (HL) — same as above but explicit opcode
                match self.phase {
                    2 => {
                        self.phase = 3;
                        McycleOp::Read {
                            addr: self.regs.hl(),
                        }
                    }
                    3 => {
                        let r = self.inc8(self.data_latch);
                        self.phase = 4;
                        McycleOp::Write {
                            addr: self.regs.hl(),
                            val: r,
                        }
                    }
                    4 => self.finish_instruction(),
                    _ => unreachable!(),
                }
            }
            0x35 => {
                // DEC (HL)
                match self.phase {
                    2 => {
                        self.phase = 3;
                        McycleOp::Read {
                            addr: self.regs.hl(),
                        }
                    }
                    3 => {
                        let r = self.dec8(self.data_latch);
                        self.phase = 4;
                        McycleOp::Write {
                            addr: self.regs.hl(),
                            val: r,
                        }
                    }
                    4 => self.finish_instruction(),
                    _ => unreachable!(),
                }
            }

            // ══════════════════════════════════════════════════════════════════
            // LD r8, imm8 — 2 M-cycles: fetch, read imm
            // ══════════════════════════════════════════════════════════════════
            0x06 | 0x0E | 0x16 | 0x1E | 0x26 | 0x2E | 0x3E => {
                let reg = (op >> 3) & 0x07;
                if reg == 6 {
                    // LD (HL), imm8 — this is actually 0x36, handled below
                    unreachable!()
                }
                match self.phase {
                    2 => {
                        let addr = self.regs.pc;
                        self.regs.pc = self.regs.pc.wrapping_add(1);
                        self.phase = 3;
                        McycleOp::Read { addr }
                    }
                    3 => {
                        self.set_reg(reg, self.data_latch);
                        self.finish_instruction()
                    }
                    _ => unreachable!(),
                }
            }
            0x36 => {
                // LD (HL), imm8 — 3 M-cycles: fetch, read imm, write to (HL)
                match self.phase {
                    2 => {
                        let addr = self.regs.pc;
                        self.regs.pc = self.regs.pc.wrapping_add(1);
                        self.phase = 3;
                        McycleOp::Read { addr }
                    }
                    3 => {
                        self.tmp8 = self.data_latch;
                        self.phase = 4;
                        McycleOp::Write {
                            addr: self.regs.hl(),
                            val: self.tmp8,
                        }
                    }
                    4 => self.finish_instruction(),
                    _ => unreachable!(),
                }
            }

            // ══════════════════════════════════════════════════════════════════
            // Rotate A instructions — 1 M-cycle (fetch only)
            // ══════════════════════════════════════════════════════════════════
            0x07 => {
                // RLCA
                let c = self.regs.a >> 7;
                self.regs.a = (self.regs.a << 1) | c;
                self.regs.set_flags(false, false, false, c != 0);
                self.finish_instruction()
            }
            0x0F => {
                // RRCA
                let c = self.regs.a & 1;
                self.regs.a = (self.regs.a >> 1) | (c << 7);
                self.regs.set_flags(false, false, false, c != 0);
                self.finish_instruction()
            }
            0x17 => {
                // RLA
                let c = self.regs.a >> 7;
                let old_c = self.regs.flag_c() as u8;
                self.regs.a = (self.regs.a << 1) | old_c;
                self.regs.set_flags(false, false, false, c != 0);
                self.finish_instruction()
            }
            0x1F => {
                // RRA
                let c = self.regs.a & 1;
                let old_c = self.regs.flag_c() as u8;
                self.regs.a = (self.regs.a >> 1) | (old_c << 7);
                self.regs.set_flags(false, false, false, c != 0);
                self.finish_instruction()
            }

            // ══════════════════════════════════════════════════════════════════
            // LD (a16), SP — 5 M-cycles: fetch, read lo, read hi, write lo, write hi
            // ══════════════════════════════════════════════════════════════════
            0x08 => {
                match self.phase {
                    2 => {
                        let addr = self.regs.pc;
                        self.regs.pc = self.regs.pc.wrapping_add(1);
                        self.phase = 3;
                        McycleOp::Read { addr }
                    }
                    3 => {
                        self.tmp8 = self.data_latch; // addr lo
                        let addr = self.regs.pc;
                        self.regs.pc = self.regs.pc.wrapping_add(1);
                        self.phase = 4;
                        McycleOp::Read { addr }
                    }
                    4 => {
                        self.tmp16 = (self.data_latch as u16) << 8 | self.tmp8 as u16;
                        self.phase = 5;
                        McycleOp::Write {
                            addr: self.tmp16,
                            val: (self.regs.sp & 0xFF) as u8,
                        }
                    }
                    5 => {
                        self.phase = 6;
                        McycleOp::Write {
                            addr: self.tmp16.wrapping_add(1),
                            val: (self.regs.sp >> 8) as u8,
                        }
                    }
                    6 => self.finish_instruction(),
                    _ => unreachable!(),
                }
            }

            // ══════════════════════════════════════════════════════════════════
            // ADD HL, r16 — 2 M-cycles: fetch, internal
            // ══════════════════════════════════════════════════════════════════
            0x09 | 0x19 | 0x29 | 0x39 => match self.phase {
                2 => {
                    let rp = (op >> 4) & 0x03;
                    let v = self.r16(rp);
                    self.add_hl(v);
                    self.phase = 3;
                    McycleOp::Internal
                }
                3 => self.finish_instruction(),
                _ => unreachable!(),
            },

            // ══════════════════════════════════════════════════════════════════
            // LD A, (BC) / LD A, (DE) — 2 M-cycles: fetch, read
            // ══════════════════════════════════════════════════════════════════
            0x0A => match self.phase {
                2 => {
                    self.phase = 3;
                    McycleOp::Read {
                        addr: self.regs.bc(),
                    }
                }
                3 => {
                    self.regs.a = self.data_latch;
                    self.finish_instruction()
                }
                _ => unreachable!(),
            },
            0x1A => match self.phase {
                2 => {
                    self.phase = 3;
                    McycleOp::Read {
                        addr: self.regs.de(),
                    }
                }
                3 => {
                    self.regs.a = self.data_latch;
                    self.finish_instruction()
                }
                _ => unreachable!(),
            },

            // ══════════════════════════════════════════════════════════════════
            // STOP — 2 M-cycles: fetch, read 0x00 arg
            // ══════════════════════════════════════════════════════════════════
            0x10 => {
                match self.phase {
                    2 => {
                        let addr = self.regs.pc;
                        self.regs.pc = self.regs.pc.wrapping_add(1);
                        self.phase = 3;
                        McycleOp::Read { addr }
                    }
                    3 => {
                        // _next = self.data_latch (consumed but unused). The
                        // emulator sees opcode 0x10 and handles the speed
                        // switch / STOP, which needs bus state.
                        self.finish_instruction()
                    }
                    _ => unreachable!(),
                }
            }

            // ══════════════════════════════════════════════════════════════════
            // JR e8 (unconditional) — 3 M-cycles: fetch, read offset, internal
            // ══════════════════════════════════════════════════════════════════
            0x18 => match self.phase {
                2 => {
                    let addr = self.regs.pc;
                    self.regs.pc = self.regs.pc.wrapping_add(1);
                    self.phase = 3;
                    McycleOp::Read { addr }
                }
                3 => {
                    self.tmp8 = self.data_latch;
                    self.oam_bug_addr = Some(self.regs.pc);
                    self.regs.pc = self.regs.pc.wrapping_add(self.tmp8 as i8 as u16);
                    self.phase = 4;
                    McycleOp::Internal
                }
                4 => self.finish_instruction(),
                _ => unreachable!(),
            },

            // ══════════════════════════════════════════════════════════════════
            // JR cc, e8 — 2 M-cycles if not taken, 3 if taken
            // ══════════════════════════════════════════════════════════════════
            0x20 | 0x28 | 0x30 | 0x38 => {
                let cc = (op >> 3) & 0x03;
                match self.phase {
                    2 => {
                        let addr = self.regs.pc;
                        self.regs.pc = self.regs.pc.wrapping_add(1);
                        self.phase = 3;
                        McycleOp::Read { addr }
                    }
                    3 => {
                        self.tmp8 = self.data_latch;
                        if self.check_condition(cc) {
                            self.oam_bug_addr = Some(self.regs.pc);
                            self.regs.pc = self.regs.pc.wrapping_add(self.tmp8 as i8 as u16);
                            self.phase = 4;
                            McycleOp::Internal
                        } else {
                            self.finish_instruction()
                        }
                    }
                    4 => self.finish_instruction(),
                    _ => unreachable!(),
                }
            }

            // ══════════════════════════════════════════════════════════════════
            // LD (HL+), A / LD (HL-), A — 2 M-cycles: fetch, write
            // ══════════════════════════════════════════════════════════════════
            0x22 => {
                // LD (HL+), A
                let hl = self.regs.hl();
                self.regs.set_hl(hl.wrapping_add(1));
                self.finish_with(McycleOp::Write {
                    addr: hl,
                    val: self.regs.a,
                })
            }
            0x32 => {
                // LD (HL-), A
                let hl = self.regs.hl();
                self.regs.set_hl(hl.wrapping_sub(1));
                self.finish_with(McycleOp::Write {
                    addr: hl,
                    val: self.regs.a,
                })
            }

            // ══════════════════════════════════════════════════════════════════
            // LD A, (HL+) / LD A, (HL-) — 2 M-cycles: fetch, read
            // ══════════════════════════════════════════════════════════════════
            0x2A => match self.phase {
                2 => {
                    let hl = self.regs.hl();
                    self.oam_bug_read_addr = Some(hl);
                    self.regs.set_hl(hl.wrapping_add(1));
                    self.phase = 3;
                    McycleOp::Read { addr: hl }
                }
                3 => {
                    self.regs.a = self.data_latch;
                    self.finish_instruction()
                }
                _ => unreachable!(),
            },
            0x3A => match self.phase {
                2 => {
                    let hl = self.regs.hl();
                    self.oam_bug_read_addr = Some(hl);
                    self.regs.set_hl(hl.wrapping_sub(1));
                    self.phase = 3;
                    McycleOp::Read { addr: hl }
                }
                3 => {
                    self.regs.a = self.data_latch;
                    self.finish_instruction()
                }
                _ => unreachable!(),
            },

            // ══════════════════════════════════════════════════════════════════
            // DAA, CPL, SCF, CCF — 1 M-cycle (fetch only)
            // ══════════════════════════════════════════════════════════════════
            0x27 => {
                self.daa();
                self.finish_instruction()
            }
            0x2F => {
                self.regs.a = !self.regs.a;
                self.regs.f |= 0x60;
                self.finish_instruction()
            }
            0x37 => {
                self.regs.f &= 0x80;
                self.regs.f |= 0x10;
                self.finish_instruction()
            }
            0x3F => {
                let z = self.regs.flag_z();
                let c = !self.regs.flag_c();
                self.regs.set_flags(z, false, false, c);
                self.finish_instruction()
            }

            // ══════════════════════════════════════════════════════════════════
            // LD r8, r8 (0x40-0x7F except 0x76=HALT)
            // 1 M-cycle for reg-to-reg, 2 for (HL) source or dest
            // ══════════════════════════════════════════════════════════════════
            0x40..=0x75 | 0x77..=0x7F => {
                let dst = (op >> 3) & 0x07;
                let src = op & 0x07;
                if src == 6 {
                    // LD r8, (HL) — read
                    match self.phase {
                        2 => {
                            self.phase = 3;
                            McycleOp::Read {
                                addr: self.regs.hl(),
                            }
                        }
                        3 => {
                            self.set_reg(dst, self.data_latch);
                            self.finish_instruction()
                        }
                        _ => unreachable!(),
                    }
                } else if dst == 6 {
                    // LD (HL), r8 — write
                    match self.phase {
                        2 => {
                            let val = self.get_reg(src);
                            self.phase = 3;
                            McycleOp::Write {
                                addr: self.regs.hl(),
                                val,
                            }
                        }
                        3 => self.finish_instruction(),
                        _ => unreachable!(),
                    }
                } else {
                    // reg-to-reg
                    let val = self.get_reg(src);
                    self.set_reg(dst, val);
                    self.finish_instruction()
                }
            }

            // ══════════════════════════════════════════════════════════════════
            // HALT (0x76)
            // ══════════════════════════════════════════════════════════════════
            0x76 => {
                // EI→HALT: the EI delay completes during HALT
                if self.ime_pending {
                    self.ime = true;
                    self.ime_pending = false;
                    // Also update pending_ime_at_start so finish_instruction
                    // doesn't double-apply
                    self.pending_ime_at_start = false;
                }
                // HALT bug detection needs IE & IF (bus state), so the
                // emulator checks the halt_bug condition after Done.
                self.halted = true;
                self.finish_instruction()
            }

            // ══════════════════════════════════════════════════════════════════
            // ALU A, r8 (0x80-0xBF)
            // 1 M-cycle for registers, 2 for (HL) source
            // ══════════════════════════════════════════════════════════════════
            0x80..=0xBF => {
                let src = op & 0x07;
                let alu_op = (op >> 3) & 0x07;
                if src == 6 {
                    // (HL) source — need a read
                    match self.phase {
                        2 => {
                            self.phase = 3;
                            McycleOp::Read {
                                addr: self.regs.hl(),
                            }
                        }
                        3 => {
                            self.do_alu(alu_op, self.data_latch);
                            self.finish_instruction()
                        }
                        _ => unreachable!(),
                    }
                } else {
                    let val = self.get_reg(src);
                    self.do_alu(alu_op, val);
                    self.finish_instruction()
                }
            }

            // ══════════════════════════════════════════════════════════════════
            // RET cc — 2 M-cycles if not taken, 5 if taken
            // ══════════════════════════════════════════════════════════════════
            0xC0 | 0xC8 | 0xD0 | 0xD8 => {
                let cc = (op >> 3) & 0x03;
                match self.phase {
                    2 => {
                        // Internal cycle (condition check)
                        self.phase = 3;
                        McycleOp::Internal
                    }
                    3 => {
                        if self.check_condition(cc) {
                            // Pop lo
                            self.oam_bug_read_addr = Some(self.regs.sp);
                            self.phase = 4;
                            McycleOp::Read { addr: self.regs.sp }
                        } else {
                            self.finish_instruction()
                        }
                    }
                    4 => {
                        self.tmp8 = self.data_latch; // lo
                        self.regs.sp = self.regs.sp.wrapping_add(1);
                        self.oam_bug_read_addr = Some(self.regs.sp);
                        self.phase = 5;
                        McycleOp::Read { addr: self.regs.sp }
                    }
                    5 => {
                        let hi = self.data_latch;
                        self.regs.sp = self.regs.sp.wrapping_add(1);
                        self.tmp16 = (hi as u16) << 8 | self.tmp8 as u16;
                        self.phase = 6;
                        McycleOp::Internal // internal after pop
                    }
                    6 => {
                        self.regs.pc = self.tmp16;
                        self.finish_instruction()
                    }
                    _ => unreachable!(),
                }
            }

            // ══════════════════════════════════════════════════════════════════
            // POP r16 — 3 M-cycles: fetch, read lo, read hi
            // ══════════════════════════════════════════════════════════════════
            0xC1 | 0xD1 | 0xE1 | 0xF1 => {
                match self.phase {
                    2 => {
                        self.oam_bug_read_addr = Some(self.regs.sp);
                        self.phase = 3;
                        McycleOp::Read { addr: self.regs.sp }
                    }
                    3 => {
                        self.tmp8 = self.data_latch; // lo
                        self.regs.sp = self.regs.sp.wrapping_add(1);
                        self.oam_bug_read_addr = Some(self.regs.sp);
                        self.phase = 4;
                        McycleOp::Read { addr: self.regs.sp }
                    }
                    4 => {
                        let hi = self.data_latch;
                        self.regs.sp = self.regs.sp.wrapping_add(1);
                        let val = (hi as u16) << 8 | self.tmp8 as u16;
                        match (op >> 4) & 0x03 {
                            0 => self.regs.set_bc(val),
                            1 => self.regs.set_de(val),
                            2 => self.regs.set_hl(val),
                            3 => self.regs.set_af(val),
                            _ => unreachable!(),
                        }
                        self.finish_instruction()
                    }
                    _ => unreachable!(),
                }
            }

            // ══════════════════════════════════════════════════════════════════
            // JP cc, a16 — 3 M-cycles if not taken, 4 if taken
            // ══════════════════════════════════════════════════════════════════
            0xC2 | 0xCA | 0xD2 | 0xDA => {
                let cc = (op >> 3) & 0x03;
                match self.phase {
                    2 => {
                        let addr = self.regs.pc;
                        self.regs.pc = self.regs.pc.wrapping_add(1);
                        self.phase = 3;
                        McycleOp::Read { addr }
                    }
                    3 => {
                        self.tmp8 = self.data_latch; // lo
                        let addr = self.regs.pc;
                        self.regs.pc = self.regs.pc.wrapping_add(1);
                        self.phase = 4;
                        McycleOp::Read { addr }
                    }
                    4 => {
                        let hi = self.data_latch;
                        let target = (hi as u16) << 8 | self.tmp8 as u16;
                        if self.check_condition(cc) {
                            self.regs.pc = target;
                            self.phase = 5;
                            McycleOp::Internal
                        } else {
                            self.finish_instruction()
                        }
                    }
                    5 => self.finish_instruction(),
                    _ => unreachable!(),
                }
            }

            // ══════════════════════════════════════════════════════════════════
            // JP a16 — 4 M-cycles: fetch, read lo, read hi, internal
            // ══════════════════════════════════════════════════════════════════
            0xC3 => match self.phase {
                2 => {
                    let addr = self.regs.pc;
                    self.regs.pc = self.regs.pc.wrapping_add(1);
                    self.phase = 3;
                    McycleOp::Read { addr }
                }
                3 => {
                    self.tmp8 = self.data_latch;
                    let addr = self.regs.pc;
                    self.regs.pc = self.regs.pc.wrapping_add(1);
                    self.phase = 4;
                    McycleOp::Read { addr }
                }
                4 => {
                    let hi = self.data_latch;
                    self.regs.pc = (hi as u16) << 8 | self.tmp8 as u16;
                    self.phase = 5;
                    McycleOp::Internal
                }
                5 => self.finish_instruction(),
                _ => unreachable!(),
            },

            // ══════════════════════════════════════════════════════════════════
            // CALL cc, a16 — 3 M-cycles if not taken, 6 if taken
            // ══════════════════════════════════════════════════════════════════
            0xC4 | 0xCC | 0xD4 | 0xDC => {
                let cc = (op >> 3) & 0x03;
                match self.phase {
                    2 => {
                        let addr = self.regs.pc;
                        self.regs.pc = self.regs.pc.wrapping_add(1);
                        self.phase = 3;
                        McycleOp::Read { addr }
                    }
                    3 => {
                        self.tmp8 = self.data_latch; // lo
                        let addr = self.regs.pc;
                        self.regs.pc = self.regs.pc.wrapping_add(1);
                        self.phase = 4;
                        McycleOp::Read { addr }
                    }
                    4 => {
                        let hi = self.data_latch;
                        self.tmp16 = (hi as u16) << 8 | self.tmp8 as u16;
                        if self.check_condition(cc) {
                            self.oam_bug_addr = Some(self.regs.sp);
                            self.phase = 5;
                            McycleOp::Internal // internal before push
                        } else {
                            self.finish_instruction()
                        }
                    }
                    5 => {
                        // Push PC high byte
                        self.regs.sp = self.regs.sp.wrapping_sub(1);
                        let val = (self.regs.pc >> 8) as u8;
                        self.phase = 6;
                        McycleOp::Write {
                            addr: self.regs.sp,
                            val,
                        }
                    }
                    6 => {
                        // Push PC low byte
                        self.regs.sp = self.regs.sp.wrapping_sub(1);
                        let val = (self.regs.pc & 0xFF) as u8;
                        self.phase = 7;
                        McycleOp::Write {
                            addr: self.regs.sp,
                            val,
                        }
                    }
                    7 => {
                        self.regs.pc = self.tmp16;
                        self.finish_instruction()
                    }
                    _ => unreachable!(),
                }
            }

            // ══════════════════════════════════════════════════════════════════
            // PUSH r16 — 4 M-cycles: fetch, internal, write hi, write lo
            // ══════════════════════════════════════════════════════════════════
            0xC5 | 0xD5 | 0xE5 | 0xF5 => match self.phase {
                2 => {
                    self.oam_bug_addr = Some(self.regs.sp);
                    let rp = (op >> 4) & 0x03;
                    self.tmp16 = match rp {
                        0 => self.regs.bc(),
                        1 => self.regs.de(),
                        2 => self.regs.hl(),
                        3 => self.regs.af(),
                        _ => unreachable!(),
                    };
                    self.phase = 3;
                    McycleOp::Internal
                }
                3 => {
                    self.regs.sp = self.regs.sp.wrapping_sub(1);
                    let val = (self.tmp16 >> 8) as u8;
                    self.phase = 4;
                    McycleOp::Write {
                        addr: self.regs.sp,
                        val,
                    }
                }
                4 => {
                    self.regs.sp = self.regs.sp.wrapping_sub(1);
                    let val = (self.tmp16 & 0xFF) as u8;
                    self.phase = 5;
                    McycleOp::Write {
                        addr: self.regs.sp,
                        val,
                    }
                }
                5 => self.finish_instruction(),
                _ => unreachable!(),
            },

            // ══════════════════════════════════════════════════════════════════
            // ALU A, imm8 — 2 M-cycles: fetch, read imm
            // ══════════════════════════════════════════════════════════════════
            0xC6 | 0xCE | 0xD6 | 0xDE | 0xE6 | 0xEE | 0xF6 | 0xFE => match self.phase {
                2 => {
                    let addr = self.regs.pc;
                    self.regs.pc = self.regs.pc.wrapping_add(1);
                    self.phase = 3;
                    McycleOp::Read { addr }
                }
                3 => {
                    let alu_op = (op >> 3) & 0x07;
                    self.do_alu(alu_op, self.data_latch);
                    self.finish_instruction()
                }
                _ => unreachable!(),
            },

            // ══════════════════════════════════════════════════════════════════
            // RST vec — 4 M-cycles: fetch, internal, write hi, write lo
            // ══════════════════════════════════════════════════════════════════
            0xC7 | 0xCF | 0xD7 | 0xDF | 0xE7 | 0xEF | 0xF7 | 0xFF => match self.phase {
                2 => {
                    self.oam_bug_addr = Some(self.regs.sp);
                    self.phase = 3;
                    McycleOp::Internal
                }
                3 => {
                    self.regs.sp = self.regs.sp.wrapping_sub(1);
                    let val = (self.regs.pc >> 8) as u8;
                    self.phase = 4;
                    McycleOp::Write {
                        addr: self.regs.sp,
                        val,
                    }
                }
                4 => {
                    self.regs.sp = self.regs.sp.wrapping_sub(1);
                    let val = (self.regs.pc & 0xFF) as u8;
                    self.phase = 5;
                    McycleOp::Write {
                        addr: self.regs.sp,
                        val,
                    }
                }
                5 => {
                    let vec = (op & 0x38) as u16;
                    self.regs.pc = vec;
                    self.finish_instruction()
                }
                _ => unreachable!(),
            },

            // ══════════════════════════════════════════════════════════════════
            // RET — 4 M-cycles: fetch, read lo, read hi, internal
            // ══════════════════════════════════════════════════════════════════
            0xC9 => {
                match self.phase {
                    2 => {
                        self.oam_bug_read_addr = Some(self.regs.sp);
                        self.phase = 3;
                        McycleOp::Read { addr: self.regs.sp }
                    }
                    3 => {
                        self.tmp8 = self.data_latch; // lo
                        self.regs.sp = self.regs.sp.wrapping_add(1);
                        self.oam_bug_read_addr = Some(self.regs.sp);
                        self.phase = 4;
                        McycleOp::Read { addr: self.regs.sp }
                    }
                    4 => {
                        let hi = self.data_latch;
                        self.regs.sp = self.regs.sp.wrapping_add(1);
                        self.tmp16 = (hi as u16) << 8 | self.tmp8 as u16;
                        self.phase = 5;
                        McycleOp::Internal
                    }
                    5 => {
                        self.regs.pc = self.tmp16;
                        self.finish_instruction()
                    }
                    _ => unreachable!(),
                }
            }

            // ══════════════════════════════════════════════════════════════════
            // RETI — 4 M-cycles (same as RET but enables IME)
            // ══════════════════════════════════════════════════════════════════
            0xD9 => match self.phase {
                2 => {
                    self.oam_bug_read_addr = Some(self.regs.sp);
                    self.phase = 3;
                    McycleOp::Read { addr: self.regs.sp }
                }
                3 => {
                    self.tmp8 = self.data_latch;
                    self.regs.sp = self.regs.sp.wrapping_add(1);
                    self.oam_bug_read_addr = Some(self.regs.sp);
                    self.phase = 4;
                    McycleOp::Read { addr: self.regs.sp }
                }
                4 => {
                    let hi = self.data_latch;
                    self.regs.sp = self.regs.sp.wrapping_add(1);
                    self.tmp16 = (hi as u16) << 8 | self.tmp8 as u16;
                    self.phase = 5;
                    McycleOp::Internal
                }
                5 => {
                    self.regs.pc = self.tmp16;
                    self.ime = true;
                    self.finish_instruction()
                }
                _ => unreachable!(),
            },

            // ══════════════════════════════════════════════════════════════════
            // CB prefix — multi-phase
            // Phase 2: read CB opcode
            // Phase 3: opcode fetched, execute (may need (HL) read/write)
            // ══════════════════════════════════════════════════════════════════
            0xCB => {
                match self.phase {
                    2 => {
                        let addr = self.regs.pc;
                        self.regs.pc = self.regs.pc.wrapping_add(1);
                        self.phase = 3;
                        McycleOp::Read { addr }
                    }
                    3 => {
                        self.cb_opcode = self.data_latch;
                        let cb = self.cb_opcode;
                        let reg = cb & 0x07;
                        if reg == 6 {
                            // Need to read (HL) first
                            self.phase = 4;
                            McycleOp::Read {
                                addr: self.regs.hl(),
                            }
                        } else {
                            // Direct register — execute and done
                            let val = self.get_reg(reg);
                            let result = self.exec_cb_op(cb, val);
                            if (cb >> 6) != 1 {
                                // Not BIT — write result back
                                self.set_reg(reg, result);
                            }
                            self.finish_instruction()
                        }
                    }
                    4 => {
                        // (HL) value read into data_latch
                        let cb = self.cb_opcode;
                        let val = self.data_latch;
                        let result = self.exec_cb_op(cb, val);
                        if (cb >> 6) == 1 {
                            // BIT: no write-back, done
                            self.finish_instruction()
                        } else {
                            // Write result back to (HL)
                            self.phase = 5;
                            McycleOp::Write {
                                addr: self.regs.hl(),
                                val: result,
                            }
                        }
                    }
                    5 => self.finish_instruction(),
                    _ => unreachable!(),
                }
            }

            // ══════════════════════════════════════════════════════════════════
            // CALL a16 — 6 M-cycles: fetch, read lo, read hi, internal, push hi, push lo
            // ══════════════════════════════════════════════════════════════════
            0xCD => match self.phase {
                2 => {
                    let addr = self.regs.pc;
                    self.regs.pc = self.regs.pc.wrapping_add(1);
                    self.phase = 3;
                    McycleOp::Read { addr }
                }
                3 => {
                    self.tmp8 = self.data_latch;
                    let addr = self.regs.pc;
                    self.regs.pc = self.regs.pc.wrapping_add(1);
                    self.phase = 4;
                    McycleOp::Read { addr }
                }
                4 => {
                    let hi = self.data_latch;
                    self.tmp16 = (hi as u16) << 8 | self.tmp8 as u16;
                    self.oam_bug_addr = Some(self.regs.sp);
                    self.phase = 5;
                    McycleOp::Internal
                }
                5 => {
                    self.regs.sp = self.regs.sp.wrapping_sub(1);
                    let val = (self.regs.pc >> 8) as u8;
                    self.phase = 6;
                    McycleOp::Write {
                        addr: self.regs.sp,
                        val,
                    }
                }
                6 => {
                    self.regs.sp = self.regs.sp.wrapping_sub(1);
                    let val = (self.regs.pc & 0xFF) as u8;
                    self.phase = 7;
                    McycleOp::Write {
                        addr: self.regs.sp,
                        val,
                    }
                }
                7 => {
                    self.regs.pc = self.tmp16;
                    self.finish_instruction()
                }
                _ => unreachable!(),
            },

            // ══════════════════════════════════════════════════════════════════
            // LDH (a8), A — 3 M-cycles: fetch, read imm, write
            // ══════════════════════════════════════════════════════════════════
            0xE0 => match self.phase {
                2 => {
                    let addr = self.regs.pc;
                    self.regs.pc = self.regs.pc.wrapping_add(1);
                    self.phase = 3;
                    McycleOp::Read { addr }
                }
                3 => {
                    self.tmp8 = self.data_latch;
                    self.phase = 4;
                    McycleOp::Write {
                        addr: 0xFF00 | self.tmp8 as u16,
                        val: self.regs.a,
                    }
                }
                4 => self.finish_instruction(),
                _ => unreachable!(),
            },

            // ══════════════════════════════════════════════════════════════════
            // LD (C), A — 2 M-cycles: fetch, write
            // ══════════════════════════════════════════════════════════════════
            0xE2 => self.finish_with(McycleOp::Write {
                addr: 0xFF00 | self.regs.c as u16,
                val: self.regs.a,
            }),

            // ══════════════════════════════════════════════════════════════════
            // ADD SP, e8 — 4 M-cycles: fetch, read imm, internal, internal
            // ══════════════════════════════════════════════════════════════════
            0xE8 => match self.phase {
                2 => {
                    let addr = self.regs.pc;
                    self.regs.pc = self.regs.pc.wrapping_add(1);
                    self.phase = 3;
                    McycleOp::Read { addr }
                }
                3 => {
                    self.tmp8 = self.data_latch;
                    self.phase = 4;
                    McycleOp::Internal
                }
                4 => {
                    self.regs.sp = self.add_sp_signed(self.tmp8);
                    self.phase = 5;
                    McycleOp::Internal
                }
                5 => self.finish_instruction(),
                _ => unreachable!(),
            },

            // ══════════════════════════════════════════════════════════════════
            // JP HL — 1 M-cycle (fetch only)
            // ══════════════════════════════════════════════════════════════════
            0xE9 => {
                self.regs.pc = self.regs.hl();
                self.finish_instruction()
            }

            // ══════════════════════════════════════════════════════════════════
            // LD (a16), A — 4 M-cycles: fetch, read lo, read hi, write
            // ══════════════════════════════════════════════════════════════════
            0xEA => match self.phase {
                2 => {
                    let addr = self.regs.pc;
                    self.regs.pc = self.regs.pc.wrapping_add(1);
                    self.phase = 3;
                    McycleOp::Read { addr }
                }
                3 => {
                    self.tmp8 = self.data_latch;
                    let addr = self.regs.pc;
                    self.regs.pc = self.regs.pc.wrapping_add(1);
                    self.phase = 4;
                    McycleOp::Read { addr }
                }
                4 => {
                    let hi = self.data_latch;
                    self.tmp16 = (hi as u16) << 8 | self.tmp8 as u16;
                    self.phase = 5;
                    McycleOp::Write {
                        addr: self.tmp16,
                        val: self.regs.a,
                    }
                }
                5 => self.finish_instruction(),
                _ => unreachable!(),
            },

            // ══════════════════════════════════════════════════════════════════
            // LDH A, (a8) — 3 M-cycles: fetch, read imm, read from FF00+n
            // ══════════════════════════════════════════════════════════════════
            0xF0 => match self.phase {
                2 => {
                    let addr = self.regs.pc;
                    self.regs.pc = self.regs.pc.wrapping_add(1);
                    self.phase = 3;
                    McycleOp::Read { addr }
                }
                3 => {
                    self.tmp8 = self.data_latch;
                    self.phase = 4;
                    McycleOp::Read {
                        addr: 0xFF00 | self.tmp8 as u16,
                    }
                }
                4 => {
                    self.regs.a = self.data_latch;
                    self.finish_instruction()
                }
                _ => unreachable!(),
            },

            // ══════════════════════════════════════════════════════════════════
            // LD A, (C) — 2 M-cycles: fetch, read from FF00+C
            // ══════════════════════════════════════════════════════════════════
            0xF2 => match self.phase {
                2 => {
                    self.phase = 3;
                    McycleOp::Read {
                        addr: 0xFF00 | self.regs.c as u16,
                    }
                }
                3 => {
                    self.regs.a = self.data_latch;
                    self.finish_instruction()
                }
                _ => unreachable!(),
            },

            // ══════════════════════════════════════════════════════════════════
            // DI — 1 M-cycle
            // ══════════════════════════════════════════════════════════════════
            0xF3 => {
                self.ime = false;
                self.ime_pending = false;
                self.finish_instruction()
            }

            // ══════════════════════════════════════════════════════════════════
            // LD HL, SP+e8 — 3 M-cycles: fetch, read imm, internal
            // ══════════════════════════════════════════════════════════════════
            0xF8 => match self.phase {
                2 => {
                    let addr = self.regs.pc;
                    self.regs.pc = self.regs.pc.wrapping_add(1);
                    self.phase = 3;
                    McycleOp::Read { addr }
                }
                3 => {
                    self.tmp8 = self.data_latch;
                    self.phase = 4;
                    McycleOp::Internal
                }
                4 => {
                    let v = self.add_sp_signed(self.tmp8);
                    self.regs.set_hl(v);
                    self.finish_instruction()
                }
                _ => unreachable!(),
            },

            // ══════════════════════════════════════════════════════════════════
            // LD SP, HL — 2 M-cycles: fetch, internal
            // ══════════════════════════════════════════════════════════════════
            0xF9 => match self.phase {
                2 => {
                    self.regs.sp = self.regs.hl();
                    self.oam_bug_addr = Some(self.regs.hl());
                    self.phase = 3;
                    McycleOp::Internal
                }
                3 => self.finish_instruction(),
                _ => unreachable!(),
            },

            // ══════════════════════════════════════════════════════════════════
            // LD A, (a16) — 4 M-cycles: fetch, read lo, read hi, read from addr
            // ══════════════════════════════════════════════════════════════════
            0xFA => match self.phase {
                2 => {
                    let addr = self.regs.pc;
                    self.regs.pc = self.regs.pc.wrapping_add(1);
                    self.phase = 3;
                    McycleOp::Read { addr }
                }
                3 => {
                    self.tmp8 = self.data_latch;
                    let addr = self.regs.pc;
                    self.regs.pc = self.regs.pc.wrapping_add(1);
                    self.phase = 4;
                    McycleOp::Read { addr }
                }
                4 => {
                    let hi = self.data_latch;
                    self.tmp16 = (hi as u16) << 8 | self.tmp8 as u16;
                    self.phase = 5;
                    McycleOp::Read { addr: self.tmp16 }
                }
                5 => {
                    self.regs.a = self.data_latch;
                    self.finish_instruction()
                }
                _ => unreachable!(),
            },

            // ══════════════════════════════════════════════════════════════════
            // EI — 1 M-cycle
            // ══════════════════════════════════════════════════════════════════
            0xFB => {
                self.ime_pending = true;
                self.finish_instruction()
            }

            // ══════════════════════════════════════════════════════════════════
            // Undefined opcodes — lock CPU
            // ══════════════════════════════════════════════════════════════════
            0xD3 | 0xDB | 0xDD | 0xE3 | 0xE4 | 0xEB | 0xEC | 0xED | 0xF4 | 0xFC | 0xFD => {
                self.ime = false;
                self.ime_pending = false;
                self.halted = true;
                // The emulator sees the opcode and clears IE (bus state).
                self.finish_instruction()
            }
        }
    }

    /// Execute a CB-prefixed ALU operation on a value, returning the result.
    fn exec_cb_op(&mut self, cb: u8, val: u8) -> u8 {
        let bit = (cb >> 3) & 0x07;
        match cb >> 6 {
            0 => {
                // Rotate/shift group
                match (cb >> 3) & 0x07 {
                    0 => {
                        // RLC
                        let c = val >> 7;
                        let r = (val << 1) | c;
                        self.regs.set_flags(r == 0, false, false, c != 0);
                        r
                    }
                    1 => {
                        // RRC
                        let c = val & 1;
                        let r = (val >> 1) | (c << 7);
                        self.regs.set_flags(r == 0, false, false, c != 0);
                        r
                    }
                    2 => {
                        // RL
                        let old_c = self.regs.flag_c() as u8;
                        let c = val >> 7;
                        let r = (val << 1) | old_c;
                        self.regs.set_flags(r == 0, false, false, c != 0);
                        r
                    }
                    3 => {
                        // RR
                        let old_c = self.regs.flag_c() as u8;
                        let c = val & 1;
                        let r = (val >> 1) | (old_c << 7);
                        self.regs.set_flags(r == 0, false, false, c != 0);
                        r
                    }
                    4 => {
                        // SLA
                        let c = val >> 7;
                        let r = val << 1;
                        self.regs.set_flags(r == 0, false, false, c != 0);
                        r
                    }
                    5 => {
                        // SRA
                        let c = val & 1;
                        let r = (val >> 1) | (val & 0x80);
                        self.regs.set_flags(r == 0, false, false, c != 0);
                        r
                    }
                    6 => {
                        // SWAP
                        let r = val.rotate_left(4);
                        self.regs.set_flags(r == 0, false, false, false);
                        r
                    }
                    7 => {
                        // SRL
                        let c = val & 1;
                        let r = val >> 1;
                        self.regs.set_flags(r == 0, false, false, c != 0);
                        r
                    }
                    _ => unreachable!(),
                }
            }
            1 => {
                // BIT: test bit, Z = NOT(bit), N=0, H=1, C unchanged
                let b = (val >> bit) & 1;
                self.regs.f &= 0x10;
                self.regs.f |= 0x20;
                if b == 0 {
                    self.regs.f |= 0x80;
                }
                val // result not used for BIT
            }
            2 => {
                // RES
                val & !(1 << bit)
            }
            3 => {
                // SET
                val | (1 << bit)
            }
            _ => unreachable!(),
        }
    }

    /// Dispatch ALU operation by 3-bit operation ID.
    fn do_alu(&mut self, alu_op: u8, val: u8) {
        match alu_op {
            0 => self.alu_add(val, false), // ADD
            1 => {
                // ADC
                let c = self.regs.flag_c();
                self.alu_add(val, c);
            }
            2 => {
                let r = self.alu_sub(val, false);
                self.regs.a = r;
            } // SUB
            3 => {
                // SBC
                let c = self.regs.flag_c();
                let r = self.alu_sub(val, c);
                self.regs.a = r;
            }
            4 => self.alu_and(val), // AND
            5 => self.alu_xor(val), // XOR
            6 => self.alu_or(val),  // OR
            7 => {
                self.alu_sub(val, false);
            } // CP (discard result)
            _ => unreachable!(),
        }
    }
}
