//! Built-in boot ROMs, generated at build time and embedded in the binary.

/// Built-in CGB boot ROM (2304 bytes).
pub const CGB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bootroms/cgb_boot.bin"));

/// Built-in DMG boot ROM (256 bytes).
pub const DMG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bootroms/dmg_boot.bin"));
