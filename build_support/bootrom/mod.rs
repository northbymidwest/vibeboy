//! Boot ROM generators (build-time only).
//!
//! Each generator assembles SM83 code into a byte-exact boot ROM.
//! Called from build.rs; output is embedded in the binary via include_bytes!.

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
