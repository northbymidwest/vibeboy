use crate::model::GbModel;
use std::fs;
use std::path::Path;

pub fn detect_model_with_rom(path: &Path, rom: Option<&[u8]>) -> GbModel {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let path_str = path.to_str().unwrap_or("");
    // OAM bug tests are DMG-specific despite having CGB flag 0x80
    if path_str.contains("oam_bug") {
        return GbModel::Dmg;
    }
    // Check longer suffixes first to avoid partial matches
    if stem.ends_with("-dmgABCmgb") || stem.ends_with("-dmgABC") {
        GbModel::Dmg
    } else if stem.ends_with("-dmg0") {
        GbModel::Dmg0
    } else if stem.ends_with("-mgb") {
        GbModel::Mgb
    } else if stem.ends_with("-sgb2") {
        GbModel::Sgb2
    } else if stem.ends_with("-sgb") || stem.ends_with("-S") {
        GbModel::Sgb
    } else if stem.ends_with("-A") {
        GbModel::Agb
    } else if stem.ends_with("-cgb0") {
        GbModel::Cgb0
    } else if stem.ends_with("-C") || stem.ends_with("-cgb") || stem.ends_with("-cgbABCDE") {
        GbModel::Cgb
    } else if stem.ends_with("-GS") || stem.ends_with("-G") {
        GbModel::Dmg
    } else if let Some(data) = rom {
        // Auto-detect from cart header CGB flag
        let cgb_flag = data.get(0x0143).copied().unwrap_or(0);
        if cgb_flag == 0x80 || cgb_flag == 0xC0 {
            GbModel::Cgb
        } else {
            GbModel::Dmg
        }
    } else {
        GbModel::Cgb
    }
}

/// Load boot ROM for the given model from bootroms/ directory.
pub fn load_boot_rom(model: GbModel) -> Option<Vec<u8>> {
    let candidates: &[&str] = match model {
        GbModel::Dmg0 => &["bootroms/dmg0_boot.bin", "bootroms/dmg_boot.bin"],
        GbModel::Dmg => &["bootroms/dmg_boot.bin"],
        GbModel::Mgb => &["bootroms/mgb_boot.bin", "bootroms/dmg_boot.bin"],
        GbModel::Sgb => &["bootroms/sgb_boot.bin"],
        GbModel::Sgb2 => &["bootroms/sgb2_boot.bin"],
        GbModel::Cgb0 => &["bootroms/cgb0_boot.bin", "bootroms/cgb_boot.bin"],
        GbModel::Cgb => &["bootroms/cgb_boot.bin", "gbc_bios.bin"],
        GbModel::Agb => &["bootroms/cgb_agb_boot.bin", "bootroms/cgb_boot.bin"],
    };
    for path in candidates {
        if let Ok(data) = fs::read(path) {
            return Some(data);
        }
    }
    None
}

pub fn resolve_boot_rom(
    boot: bool,
    bootrom: Option<&Path>,
    model: GbModel,
) -> Option<Vec<u8>> {
    if let Some(p) = bootrom {
        Some(fs::read(p).unwrap_or_else(|e| {
            eprintln!("Failed to read boot ROM '{}': {}", p.display(), e);
            std::process::exit(1);
        }))
    } else if boot {
        load_boot_rom(model)
    } else {
        None
    }
}
