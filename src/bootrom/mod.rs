//! Boot ROM generators for various Game Boy models.
//!
//! Each generator produces a byte-exact boot ROM that can be used as a
//! built-in fallback when no external boot ROM file is provided.

pub mod asm;
pub mod cgb;
pub mod dmg;
mod data;

/// Generate the built-in CGB boot ROM (2304 bytes).
pub fn cgb_boot_rom() -> Vec<u8> {
    cgb::build()
}

/// Generate the built-in DMG boot ROM (256 bytes).
pub fn dmg_boot_rom() -> Vec<u8> {
    dmg::build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cgb_boot_rom_size() {
        let rom = cgb_boot_rom();
        assert_eq!(rom.len(), 0x0900);
    }

    #[test]
    fn cgb_boot_rom_structure() {
        let rom = cgb_boot_rom();
        // Region 1 starts with DI (0xF3)
        assert_eq!(rom[0], 0xF3);
        // Handoff at 0xFC: LD A,$11
        assert_eq!(rom[0xFC], 0x3E);
        assert_eq!(rom[0xFD], 0x11);
        // Followed by LDH ($FF50),A
        assert_eq!(rom[0xFE], 0xE0);
        assert_eq!(rom[0xFF], 0x50);
        // Region 2 starts at 0x200 (should have code, not zeros)
        assert_ne!(rom[0x200], 0x00);
    }

    #[test]
    fn cgb_boot_rom_deterministic() {
        let rom1 = cgb_boot_rom();
        let rom2 = cgb_boot_rom();
        assert_eq!(rom1, rom2);
    }

}
