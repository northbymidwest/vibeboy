//! Minimal SM83 assembler for generating boot ROMs at build time.
//!
//! Supports labels with forward references (resolved at finalization),
//! both absolute (JP/CALL) and relative (JR) addressing modes.

use std::collections::HashMap;

/// Fixup type for unresolved label references.
enum FixupKind {
    /// 16-bit absolute address (JP, CALL, LD rr,nn).
    Abs16,
    /// 8-bit signed relative offset (JR).
    Rel8,
}

struct Fixup {
    /// Byte offset in the code buffer where the address/offset goes.
    offset: usize,
    /// Label name to resolve.
    label: String,
    kind: FixupKind,
}

pub struct Asm {
    code: Vec<u8>,
    labels: HashMap<String, usize>,
    fixups: Vec<Fixup>,
}

impl Asm {
    pub fn new() -> Self {
        Self {
            code: Vec::new(),
            labels: HashMap::new(),
            fixups: Vec::new(),
        }
    }

    /// Current program counter (next byte offset).
    pub fn pc(&self) -> usize {
        self.code.len()
    }

    /// Define a label at the current PC.
    pub fn label(&mut self, name: &str) {
        self.labels.insert(name.to_string(), self.pc());
    }

    /// Emit raw bytes.
    fn emit(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }

    /// Record a fixup for a label reference.
    fn fixup(&mut self, offset: usize, label: &str, kind: FixupKind) {
        self.fixups.push(Fixup {
            offset,
            label: label.to_string(),
            kind,
        });
    }

    // ── Raw data ─────────────────────────────────────────────────────

    pub fn db(&mut self, bytes: &[u8]) {
        self.emit(bytes);
    }

    pub fn pad_to(&mut self, addr: usize) {
        while self.pc() < addr {
            self.code.push(0x00);
        }
    }

    // ── Basic instructions ───────────────────────────────────────────

    pub fn nop(&mut self) { self.emit(&[0x00]); }
    pub fn di(&mut self) { self.emit(&[0xF3]); }
    pub fn ret(&mut self) { self.emit(&[0xC9]); }
    pub fn xor_a(&mut self) { self.emit(&[0xAF]); }
    pub fn cpl(&mut self) { self.emit(&[0x2F]); }
    pub fn swap_a(&mut self) { self.emit(&[0xCB, 0x37]); }

    // ── ALU with immediate ───────────────────────────────────────────

    pub fn cp_a_n(&mut self, n: u8) { self.emit(&[0xFE, n]); }
    pub fn and_n(&mut self, n: u8) { self.emit(&[0xE6, n]); }
    pub fn add_a_n(&mut self, n: u8) { self.emit(&[0xC6, n]); }
    pub fn sub_n(&mut self, n: u8) { self.emit(&[0xD6, n]); }
    pub fn or_b(&mut self) { self.emit(&[0xB0]); }
    pub fn sub_hl_ind(&mut self) { self.emit(&[0x96]); }

    // ── Rotate/shift ─────────────────────────────────────────────────

    pub fn rla(&mut self) { self.emit(&[0x17]); }
    pub fn rl_c(&mut self) { self.emit(&[0xCB, 0x11]); }

    // ── ALU register ─────────────────────────────────────────────────

    pub fn cp_b(&mut self) { self.emit(&[0xB8]); }
    pub fn cp_c(&mut self) { self.emit(&[0xB9]); }
    pub fn cp_d(&mut self) { self.emit(&[0xBA]); }
    pub fn cp_hl_ind(&mut self) { self.emit(&[0xBE]); }
    pub fn add_a_a(&mut self) { self.emit(&[0x87]); }
    pub fn add_a_hl(&mut self) { self.emit(&[0x86]); }
    pub fn add_a_l(&mut self) { self.emit(&[0x85]); }
    pub fn add_a_e(&mut self) { self.emit(&[0x83]); }

    // ── 8-bit loads (immediate) ──────────────────────────────────────

    pub fn ld_a(&mut self, n: u8) { self.emit(&[0x3E, n]); }
    pub fn ld_b(&mut self, n: u8) { self.emit(&[0x06, n]); }
    pub fn ld_c(&mut self, n: u8) { self.emit(&[0x0E, n]); }
    pub fn ld_d(&mut self, n: u8) { self.emit(&[0x16, n]); }
    pub fn ld_e(&mut self, n: u8) { self.emit(&[0x1E, n]); }
    pub fn ld_h(&mut self, n: u8) { self.emit(&[0x26, n]); }
    pub fn ld_l(&mut self, n: u8) { self.emit(&[0x2E, n]); }

    // ── 8-bit register-to-register ───────────────────────────────────

    pub fn ld_a_b(&mut self) { self.emit(&[0x78]); }
    pub fn ld_a_c(&mut self) { self.emit(&[0x79]); }
    pub fn ld_a_d(&mut self) { self.emit(&[0x7A]); }
    pub fn ld_a_e(&mut self) { self.emit(&[0x7B]); }
    pub fn ld_a_h(&mut self) { self.emit(&[0x7C]); }
    pub fn ld_a_l(&mut self) { self.emit(&[0x7D]); }
    pub fn ld_a_hl_ind(&mut self) { self.emit(&[0x7E]); }
    pub fn ld_a_de_ind(&mut self) { self.emit(&[0x1A]); }
    pub fn ld_b_a(&mut self) { self.emit(&[0x47]); }
    pub fn ld_b_c(&mut self) { self.emit(&[0x41]); }
    pub fn ld_c_a(&mut self) { self.emit(&[0x4F]); }
    pub fn ld_d_a(&mut self) { self.emit(&[0x57]); }
    pub fn ld_e_a(&mut self) { self.emit(&[0x5F]); }
    pub fn ld_h_a(&mut self) { self.emit(&[0x67]); }
    pub fn ld_l_a(&mut self) { self.emit(&[0x6F]); }
    pub fn ld_l_e(&mut self) { self.emit(&[0x6B]); }
    pub fn ld_hl_ind_a(&mut self) { self.emit(&[0x77]); }

    // ── 16-bit loads ─────────────────────────────────────────────────

    pub fn ld_bc(&mut self, nn: u16) { self.emit(&[0x01, nn as u8, (nn >> 8) as u8]); }
    pub fn ld_sp(&mut self, nn: u16) { self.emit(&[0x31, nn as u8, (nn >> 8) as u8]); }

    pub fn ld_de_imm(&mut self, nn: u16) {
        self.emit(&[0x11, nn as u8, (nn >> 8) as u8]);
    }
    pub fn ld_de_label(&mut self, label: &str) {
        let off = self.pc() + 1;
        self.emit(&[0x11, 0, 0]);
        self.fixup(off, label, FixupKind::Abs16);
    }

    pub fn ld_hl_imm(&mut self, nn: u16) {
        self.emit(&[0x21, nn as u8, (nn >> 8) as u8]);
    }
    pub fn ld_hl_label(&mut self, label: &str) {
        let off = self.pc() + 1;
        self.emit(&[0x21, 0, 0]);
        self.fixup(off, label, FixupKind::Abs16);
    }

    // ── 16-bit arithmetic ────────────────────────────────────────────

    pub fn add_hl_bc(&mut self) { self.emit(&[0x09]); }
    pub fn add_hl_de(&mut self) { self.emit(&[0x19]); }

    // ── Memory ───────────────────────────────────────────────────────

    pub fn ldh_n_a(&mut self, n: u8) { self.emit(&[0xE0, n]); }
    pub fn ldh_a_n(&mut self, n: u8) { self.emit(&[0xF0, n]); }
    pub fn ld_hli_a(&mut self) { self.emit(&[0x22]); }
    pub fn ld_a_hli(&mut self) { self.emit(&[0x2A]); }
    pub fn ld_a_nn(&mut self, nn: u16) { self.emit(&[0xFA, nn as u8, (nn >> 8) as u8]); }
    pub fn ld_ffc_a(&mut self) { self.emit(&[0xE2]); }

    // ── Inc/Dec ──────────────────────────────────────────────────────

    pub fn inc_a(&mut self) { self.emit(&[0x3C]); }
    pub fn inc_c(&mut self) { self.emit(&[0x0C]); }
    pub fn inc_hl(&mut self) { self.emit(&[0x23]); }
    pub fn inc_de(&mut self) { self.emit(&[0x13]); }
    pub fn dec_a(&mut self) { self.emit(&[0x3D]); }
    pub fn dec_b(&mut self) { self.emit(&[0x05]); }
    pub fn dec_c(&mut self) { self.emit(&[0x0D]); }
    pub fn dec_e(&mut self) { self.emit(&[0x1D]); }

    // ── Stack ────────────────────────────────────────────────────────

    pub fn push_af(&mut self) { self.emit(&[0xF5]); }
    pub fn push_bc(&mut self) { self.emit(&[0xC5]); }
    pub fn push_de(&mut self) { self.emit(&[0xD5]); }
    pub fn push_hl(&mut self) { self.emit(&[0xE5]); }
    pub fn pop_af(&mut self) { self.emit(&[0xF1]); }
    pub fn pop_bc(&mut self) { self.emit(&[0xC1]); }
    pub fn pop_de(&mut self) { self.emit(&[0xD1]); }
    pub fn pop_hl(&mut self) { self.emit(&[0xE1]); }

    // ── Jumps (absolute) ─────────────────────────────────────────────

    pub fn jp(&mut self, label: &str) {
        let off = self.pc() + 1;
        self.emit(&[0xC3, 0, 0]);
        self.fixup(off, label, FixupKind::Abs16);
    }

    pub fn jp_nz(&mut self, label: &str) {
        let off = self.pc() + 1;
        self.emit(&[0xC2, 0, 0]);
        self.fixup(off, label, FixupKind::Abs16);
    }

    pub fn call(&mut self, label: &str) {
        let off = self.pc() + 1;
        self.emit(&[0xCD, 0, 0]);
        self.fixup(off, label, FixupKind::Abs16);
    }

    // ── Jumps (relative) ─────────────────────────────────────────────

    pub fn jr(&mut self, label: &str) {
        let off = self.pc() + 1;
        self.emit(&[0x18, 0]);
        self.fixup(off, label, FixupKind::Rel8);
    }

    pub fn jr_z(&mut self, label: &str) {
        let off = self.pc() + 1;
        self.emit(&[0x28, 0]);
        self.fixup(off, label, FixupKind::Rel8);
    }

    pub fn jr_nz(&mut self, label: &str) {
        let off = self.pc() + 1;
        self.emit(&[0x20, 0]);
        self.fixup(off, label, FixupKind::Rel8);
    }

    pub fn jr_c(&mut self, label: &str) {
        let off = self.pc() + 1;
        self.emit(&[0x38, 0]);
        self.fixup(off, label, FixupKind::Rel8);
    }

    // ── Resolve & finalize ───────────────────────────────────────────

    /// Resolve all label references. Panics on undefined labels or out-of-range JR.
    pub fn resolve(&mut self) {
        for fixup in &self.fixups {
            let target = *self.labels.get(&fixup.label)
                .unwrap_or_else(|| panic!("undefined label: {}", fixup.label));
            match fixup.kind {
                FixupKind::Abs16 => {
                    self.code[fixup.offset] = target as u8;
                    self.code[fixup.offset + 1] = (target >> 8) as u8;
                }
                FixupKind::Rel8 => {
                    let rel = target as isize - (fixup.offset as isize + 1);
                    assert!(
                        (-128..=127).contains(&rel),
                        "JR to '{}' out of range: {} (from offset {:#x})",
                        fixup.label, rel, fixup.offset
                    );
                    self.code[fixup.offset] = rel as u8;
                }
            }
        }
    }

    /// Pad to target size and return the final ROM bytes.
    pub fn into_rom(mut self, size: usize) -> Vec<u8> {
        self.resolve();
        assert!(
            self.code.len() <= size,
            "code is {:#x} bytes, exceeds {:#x}",
            self.code.len(), size
        );
        self.code.resize(size, 0x00);
        self.code
    }
}
