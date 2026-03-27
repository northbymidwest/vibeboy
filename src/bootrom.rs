//! Built-in boot ROMs, generated at build time and embedded in the binary.
//!
//! Use `builtin(model)` to get the built-in ROM for a given model, or `None`
//! for models that don't have a built-in (SGB, SGB2, DMG0, CGB0).

use crate::model::GbModel;

/// Built-in DMG boot ROM (256 bytes).
pub const DMG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bootroms/dmg_boot.bin"));

/// Built-in MGB boot ROM (256 bytes, Game Boy Pocket — A=$FF).
pub const MGB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bootroms/mgb_boot.bin"));

/// Built-in CGB boot ROM (2304 bytes).
pub const CGB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bootroms/cgb_boot.bin"));

/// Built-in AGB boot ROM (2304 bytes, GBA in GBC mode — B=0x01).
pub const AGB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bootroms/agb_boot.bin"));

/// Get the built-in boot ROM for a hardware model, if available.
pub fn builtin(model: GbModel) -> Option<&'static [u8]> {
    match model {
        GbModel::Dmg => Some(DMG),
        GbModel::Mgb => Some(MGB),
        GbModel::Cgb0 | GbModel::Cgb => Some(CGB),
        GbModel::Agb => Some(AGB),
        // No built-in for DMG0, SGB, SGB2 (need external files)
        _ => None,
    }
}
