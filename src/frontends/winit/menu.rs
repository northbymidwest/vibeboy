use muda::{
    AboutMetadata, CheckMenuItem, Menu, MenuItem, PredefinedMenuItem, Submenu,
    accelerator::{Accelerator, Code, Modifiers},
};

use super::model::GbModel;
use super::scaling;

pub(super) const ID_OPEN: &str = "open_rom";
pub(super) const ID_QUIT: &str = "quit";
pub(super) const ID_PAUSE: &str = "pause";
pub(super) const ID_RESET: &str = "reset";
pub(super) const ID_PRINTER: &str = "toggle_printer";
pub(super) const ID_FORCE_CPU: &str = "force_cpu";

pub(super) const MODEL_IDS: &[(&str, &str)] = &[
    ("model_auto", "Auto"),
    ("model_dmg0", "DMG0"),
    ("model_dmg", "DMG"),
    ("model_mgb", "MGB"),
    ("model_sgb", "SGB"),
    ("model_sgb2", "SGB2"),
    ("model_cgb", "CGB"),
    ("model_agb", "AGB"),
];

pub(super) fn model_id_to_model(id: &str) -> Option<Option<GbModel>> {
    match id {
        "model_auto" => Some(None),
        "model_dmg0" => Some(Some(GbModel::Dmg0)),
        "model_dmg" => Some(Some(GbModel::Dmg)),
        "model_mgb" => Some(Some(GbModel::Mgb)),
        "model_sgb" => Some(Some(GbModel::Sgb)),
        "model_sgb2" => Some(Some(GbModel::Sgb2)),
        "model_cgb" => Some(Some(GbModel::Cgb)),
        "model_agb" => Some(Some(GbModel::Agb)),
        _ => None,
    }
}

pub(super) fn slot_save_id(n: usize) -> String {
    format!("slot_save_{}", n)
}

pub(super) fn slot_load_id(n: usize) -> String {
    format!("slot_load_{}", n)
}

/// All filter menu entries: (menu_id, display_name, ScaleFilter).
/// Derived from the central registry; menu_id is generated from the CLI name.
pub(super) fn filter_entries() -> Vec<(&'static str, &'static str, scaling::ScaleFilter)> {
    scaling::ScaleFilter::menu_entries()
        .map(|(display, filter)| {
            // Use cli_name as a stable menu ID (prefixed with "filter_")
            // We leak the string to get a &'static str since these are created once
            let id: &'static str = Box::leak(format!("filter_{}", filter.cli_name()).into_boxed_str());
            (id, display, filter)
        })
        .collect()
}

pub(super) fn filter_id_to_filter(id: &str) -> Option<scaling::ScaleFilter> {
    filter_entries().iter().find(|(mid, _, _)| *mid == id).map(|(_, _, f)| *f)
}

pub(super) fn build_menu(printer_on: bool) -> (Menu, Vec<(CheckMenuItem, scaling::ScaleFilter)>, CheckMenuItem, Vec<CheckMenuItem>, Vec<CheckMenuItem>, CheckMenuItem) {
    let menu = Menu::new();

    // File menu
    let file_menu = Submenu::new("File", true);
    file_menu
        .append_items(&[
            &MenuItem::with_id(
                ID_OPEN,
                "Open ROM...",
                true,
                Some(Accelerator::new(Some(Modifiers::SUPER), Code::KeyO)),
            ),
            &PredefinedMenuItem::separator(),
            &MenuItem::with_id(
                ID_QUIT,
                "Quit",
                true,
                Some(Accelerator::new(Some(Modifiers::SUPER), Code::KeyQ)),
            ),
        ])
        .unwrap();

    // Emulation menu
    let emu_menu = Submenu::new("Emulation", true);
    let printer_item = CheckMenuItem::with_id(
        ID_PRINTER, "Game Boy Printer", true, printer_on, None::<Accelerator>,
    );
    emu_menu
        .append_items(&[
            &MenuItem::with_id(
                ID_PAUSE,
                "Pause",
                true,
                Some(Accelerator::new(None, Code::F6)),
            ),
            &MenuItem::with_id(ID_RESET, "Reset", true, None::<Accelerator>),
            &PredefinedMenuItem::separator(),
            &printer_item,
        ])
        .unwrap();

    // Hardware model submenu
    let hw_menu = Submenu::new("Hardware", true);
    let mut model_items = Vec::new();
    for &(id, label) in MODEL_IDS {
        let checked = id == "model_auto";
        let item = CheckMenuItem::with_id(id, label, true, checked, None::<Accelerator>);
        hw_menu.append(&item).unwrap();
        model_items.push(item);
    }
    emu_menu.append(&PredefinedMenuItem::separator()).unwrap();
    emu_menu.append(&hw_menu).unwrap();

    // State menu with Save/Load submenus and slot selection
    let state_menu = Submenu::new("State", true);
    let save_sub = Submenu::new("Save State", true);
    let load_sub = Submenu::new("Load State", true);
    for i in 0..=9usize {
        save_sub
            .append(&MenuItem::with_id(
                slot_save_id(i),
                format!("Slot {}", i),
                true,
                if i == 0 { Some(Accelerator::new(None, Code::F5)) } else { None },
            ))
            .unwrap();
        load_sub
            .append(&MenuItem::with_id(
                slot_load_id(i),
                format!("Slot {}", i),
                true,
                if i == 0 { Some(Accelerator::new(None, Code::F7)) } else { None },
            ))
            .unwrap();
    }
    state_menu
        .append_items(&[&save_sub, &load_sub])
        .unwrap();
    state_menu.append(&PredefinedMenuItem::separator()).unwrap();
    let mut slot_items = Vec::new();
    for i in 0..=9usize {
        let item = CheckMenuItem::with_id(
            format!("select_slot_{}", i),
            format!("Slot {}", i),
            true,
            i == 0,
            None::<Accelerator>,
        );
        state_menu.append(&item).unwrap();
        slot_items.push(item);
    }

    // Filter menu -- use CheckMenuItem for checkmark support
    let filter_menu = Submenu::new("Filter", true);
    let force_cpu_item = CheckMenuItem::with_id(
        ID_FORCE_CPU, "Force CPU", true, false, None::<Accelerator>,
    );
    filter_menu.append(&force_cpu_item).unwrap();
    filter_menu.append(&PredefinedMenuItem::separator()).unwrap();
    let mut filter_items = Vec::new();
    {
        use scaling::FilterMenuGroup;
        let mut sub_menus = std::collections::BTreeMap::new();

        for (id, name, filter) in filter_entries() {
            let checked = filter == scaling::ScaleFilter::Nearest;
            let item = CheckMenuItem::with_id(id, name, true, checked, None::<Accelerator>);
            let group = filter.menu_group();
            if group == FilterMenuGroup::Main {
                filter_menu.append(&item).unwrap();
            } else {
                sub_menus.entry(group.label())
                    .or_insert_with(|| Submenu::new(group.label(), true))
                    .append(&item).unwrap();
            }
            filter_items.push((item, filter));
        }

        filter_menu.append(&PredefinedMenuItem::separator()).unwrap();
        for (_, sub) in &sub_menus {
            filter_menu.append(sub).unwrap();
        }
    }

    // Help menu
    let help_menu = Submenu::new("Help", true);
    help_menu
        .append(&PredefinedMenuItem::about(
            Some("About VibeBoy"),
            Some(AboutMetadata {
                name: Some("VibeBoy".into()),
                version: Some(env!("CARGO_PKG_VERSION").into()),
                ..Default::default()
            }),
        ))
        .unwrap();

    #[cfg(target_os = "macos")]
    {
        let app_menu = Submenu::new("VibeBoy", true);
        app_menu
            .append_items(&[
                &PredefinedMenuItem::about(
                    Some("About VibeBoy"),
                    Some(AboutMetadata {
                        name: Some("VibeBoy".into()),
                        version: Some(env!("CARGO_PKG_VERSION").into()),
                        ..Default::default()
                    }),
                ),
                &PredefinedMenuItem::separator(),
                &PredefinedMenuItem::quit(Some("Quit VibeBoy")),
            ])
            .unwrap();
        menu.append(&app_menu).unwrap();
    }

    menu.append_items(&[&file_menu, &emu_menu, &state_menu, &filter_menu, &help_menu])
        .unwrap();

    (menu, filter_items, printer_item, model_items, slot_items, force_cpu_item)
}
