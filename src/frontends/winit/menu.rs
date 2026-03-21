use muda::{
    AboutMetadata, CheckMenuItem, Menu, MenuItem, PredefinedMenuItem, Submenu,
    accelerator::{Accelerator, Code, Modifiers},
};

use super::scaling;

pub(super) const ID_OPEN: &str = "open_rom";
pub(super) const ID_QUIT: &str = "quit";
pub(super) const ID_PAUSE: &str = "pause";
pub(super) const ID_RESET: &str = "reset";

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

pub(super) fn build_menu() -> (Menu, Vec<(CheckMenuItem, scaling::ScaleFilter)>) {
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
    emu_menu
        .append_items(&[
            &MenuItem::with_id(
                ID_PAUSE,
                "Pause",
                true,
                Some(Accelerator::new(None, Code::F6)),
            ),
            &MenuItem::with_id(ID_RESET, "Reset", true, None::<Accelerator>),
        ])
        .unwrap();

    // State menu with Save/Load submenus
    let state_menu = Submenu::new("State", true);
    let save_sub = Submenu::new("Save State", true);
    let load_sub = Submenu::new("Load State", true);
    for i in 1..=9 {
        save_sub
            .append(&MenuItem::with_id(
                slot_save_id(i),
                format!("Slot {}", i),
                true,
                if i == 1 {
                    Some(Accelerator::new(None, Code::F5))
                } else {
                    None
                },
            ))
            .unwrap();
        load_sub
            .append(&MenuItem::with_id(
                slot_load_id(i),
                format!("Slot {}", i),
                true,
                if i == 1 {
                    Some(Accelerator::new(None, Code::F7))
                } else {
                    None
                },
            ))
            .unwrap();
    }
    state_menu
        .append_items(&[&save_sub, &load_sub])
        .unwrap();

    // Filter menu -- use CheckMenuItem for checkmark support
    let filter_menu = Submenu::new("Filter", true);
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

    (menu, filter_items)
}
