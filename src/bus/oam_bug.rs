/// OAM bug corruption (DMG only): write and read corruption during Mode 2.

use super::Bus;

impl Bus {
    /// Trigger OAM write corruption bug when a 16-bit register pointing to 0xFE00–0xFEFF
    /// is incremented/decremented during PPU Mode 2 (OAM scan). DMG only.
    pub fn trigger_oam_bug(&mut self, addr: u16) {
        if self.model.is_cgb() { return; }
        if addr < 0xFE00 || addr > 0xFEFF { return; }
        self.flush_ppu_deferred();
        self.trigger_oam_bug_inner();
    }

    pub fn trigger_oam_bug_from_write(&mut self, addr: u16) {
        if self.model.is_cgb() { return; }
        if addr < 0xFE00 || addr > 0xFEFF { return; }
        self.flush_ppu_deferred();
        self.trigger_oam_bug_inner();
    }

    fn trigger_oam_bug_inner(&mut self) {
        let row = self.ppu.oam_bug_row;

        // Row must be valid (not 0xFF) and >= 8 for corruption to occur.
        // No upper bound check — hardware allows corruption even at accessed_oam_row >= 160.
        if row == 0xFF || row < 8 {
            return;
        }
        let row = row as usize;
        // Bytes 0-1: bitwise glitch (operate as u16 little-endian)
        let a = u16::from_le_bytes([self.ppu.oam[row], self.ppu.oam[row + 1]]);
        let b = u16::from_le_bytes([self.ppu.oam[row - 8], self.ppu.oam[row - 7]]);
        let c = u16::from_le_bytes([self.ppu.oam[row - 4], self.ppu.oam[row - 3]]);
        let glitched = ((a ^ c) & (b ^ c)) ^ c;
        let bytes = glitched.to_le_bytes();
        self.ppu.oam[row] = bytes[0];
        self.ppu.oam[row + 1] = bytes[1];
        // Bytes 2-7: copy from previous row
        for i in 2..8 {
            self.ppu.oam[row + i] = self.ppu.oam[row - 8 + i];
        }
    }

    /// Trigger OAM read corruption bug when a memory read targets 0xFE00–0xFEFF
    /// during PPU Mode 2 (OAM scan). DMG only. Uses different formulas from write corruption.
    pub fn trigger_oam_bug_read(&mut self, addr: u16) {
        if self.model.is_cgb() { return; }
        if addr < 0xFE00 || addr > 0xFEFF { return; }
        self.flush_ppu_deferred();
        let row = self.ppu.oam_bug_row;
        // Row must be valid (not 0xFF) and >= 8 for corruption to occur.
        // No upper bound check — hardware allows corruption even at accessed_oam_row >= 160.
        if row == 0xFF || row < 8 { return; }
        let row = row as usize;

        if (row & 0x18) == 0x10 {
            // Secondary read corruption — affects row-8 word, then copies two rows back
            if row >= 0x18 { // need row-16 to exist
                let base_m8 = u16::from_le_bytes([self.ppu.oam[row - 16], self.ppu.oam[row - 15]]);
                let base_m4 = u16::from_le_bytes([self.ppu.oam[row - 8], self.ppu.oam[row - 7]]);
                let base_0  = u16::from_le_bytes([self.ppu.oam[row], self.ppu.oam[row + 1]]);
                let base_m2 = u16::from_le_bytes([self.ppu.oam[row - 4], self.ppu.oam[row - 3]]);
                // base[-4] = secondary(base[-8], base[-4], base[0], base[-2])
                let glitched = (base_m4 & (base_m8 | base_0 | base_m2)) | (base_m8 & base_0 & base_m2);
                let bytes = glitched.to_le_bytes();
                self.ppu.oam[row - 8] = bytes[0];
                self.ppu.oam[row - 7] = bytes[1];
                // Copy row-8 to row-16
                for i in 0..8 {
                    self.ppu.oam[row - 0x10 + i] = self.ppu.oam[row - 0x08 + i];
                }
            }
        } else if (row & 0x18) == 0x00 {
            // Tertiary/quaternary — model and row specific
            if row >= 0x20 { // need row-32 to exist for base[-16]
                let base_0   = u16::from_le_bytes([self.ppu.oam[row], self.ppu.oam[row + 1]]);
                let base_m2  = u16::from_le_bytes([self.ppu.oam[row - 4], self.ppu.oam[row - 3]]);
                let base_m4  = u16::from_le_bytes([self.ppu.oam[row - 8], self.ppu.oam[row - 7]]);
                let base_m8  = u16::from_le_bytes([self.ppu.oam[row - 16], self.ppu.oam[row - 15]]);
                let base_m16 = u16::from_le_bytes([self.ppu.oam[row - 32], self.ppu.oam[row - 31]]);

                let glitched = if row == 0x40 {
                    // Quaternary
                    let base_m3  = u16::from_le_bytes([self.ppu.oam[row - 6], self.ppu.oam[row - 5]]);
                    let base_m7  = u16::from_le_bytes([self.ppu.oam[row - 14], self.ppu.oam[row - 13]]);
                    let oam_0    = u16::from_le_bytes([self.ppu.oam[0], self.ppu.oam[1]]);
                    if self.model == crate::model::GbModel::Sgb2 {
                        // sgb2 variant
                        (base_m4 & (base_m16 | base_m8 | base_m2 | (oam_0 & base_0)))
                            | ((base_m2 & base_m8 & base_m16) & (base_0 | oam_0 | !base_m7))
                    } else {
                        // dmg variant (a unused)
                        (base_m4 & (base_m16 | base_m8 | (!base_m3 & base_m7) | base_m2 | base_0))
                            | (base_m2 & base_m8 & base_m16)
                    }
                } else if self.model == crate::model::GbModel::Mgb {
                    // MGB: tertiary_read_3
                    (base_m4 & (base_0 | base_m2 | base_m8 | base_m16)) | (base_m2 & base_m8 & base_m16)
                } else if self.model == crate::model::GbModel::Sgb2 {
                    // SGB2: tertiary_read_2
                    (base_m4 & (base_0 | base_m2 | base_m8 | base_m16)) | (base_0 & base_m2 & base_m8 & base_m16)
                } else if row == 0x20 {
                    // tertiary_read_2
                    (base_m4 & (base_0 | base_m2 | base_m8 | base_m16)) | (base_0 & base_m2 & base_m8 & base_m16)
                } else if row == 0x60 {
                    // tertiary_read_3
                    (base_m4 & (base_0 | base_m2 | base_m8 | base_m16)) | (base_m2 & base_m8 & base_m16)
                } else {
                    // tertiary_read_1 (default)
                    base_m4 | (base_0 & base_m2 & base_m8 & base_m16)
                };

                let bytes = glitched.to_le_bytes();
                self.ppu.oam[row - 8] = bytes[0];
                self.ppu.oam[row - 7] = bytes[1];
                // Copy two rows back
                for i in 0..8 {
                    let v = self.ppu.oam[row - 0x08 + i];
                    self.ppu.oam[row - 0x10 + i] = v;
                    self.ppu.oam[row - 0x20 + i] = v;
                }
            }
        } else {
            // Common case: (row & 0x18) == 0x08 or 0x18
            let a = u16::from_le_bytes([self.ppu.oam[row], self.ppu.oam[row + 1]]);
            let b = u16::from_le_bytes([self.ppu.oam[row - 8], self.ppu.oam[row - 7]]);
            let c = u16::from_le_bytes([self.ppu.oam[row - 4], self.ppu.oam[row - 3]]);
            let glitched = b | (a & c);
            let bytes = glitched.to_le_bytes();
            // Both base[-4] and base[0] get the glitched value
            self.ppu.oam[row - 8] = bytes[0];
            self.ppu.oam[row - 7] = bytes[1];
            self.ppu.oam[row] = bytes[0];
            self.ppu.oam[row + 1] = bytes[1];
        }

        // Copy previous row into current row (shared across all cases)
        for i in 0..8 {
            self.ppu.oam[row + i] = self.ppu.oam[row - 8 + i];
        }

        // Special: row 0x80 copies corruption to row 0
        if row == 0x80 || (self.model == crate::model::GbModel::Mgb && row == 0x40) {
            for i in 0..8 {
                self.ppu.oam[i] = self.ppu.oam[row + i];
            }
        }
    }
}
