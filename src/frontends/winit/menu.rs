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
/// Filters that support arbitrary resolution (Bilinear, Bicubic, OmniScale,
/// AA Nearest, Vectorize) appear as a single entry -- no per-scale variants.
pub(super) fn filter_entries() -> Vec<(&'static str, &'static str, scaling::ScaleFilter)> {
    use scaling::*;
    vec![
        ("filter_nearest",     "Nearest",       ScaleFilter::Nearest),
        ("filter_bilinear",    "Bilinear",      ScaleFilter::Bilinear),
        ("filter_bicubic",     "Bicubic",       ScaleFilter::Bicubic),
        ("filter_epx",         "EPX / Scale2x", ScaleFilter::Epx),
        ("filter_scale3x",    "Scale3x",       ScaleFilter::Scale3x),
        ("filter_scale4x",    "Scale4x",       ScaleFilter::Scale4x),
        ("filter_eagle",       "Eagle",         ScaleFilter::Eagle),
        ("filter_2xsai",       "2xSaI",         ScaleFilter::Sai2x),
        ("filter_s2xsai",     "Super 2xSaI",   ScaleFilter::Super2xSai),
        ("filter_seagle",      "Super Eagle",   ScaleFilter::SuperEagle),
        ("filter_hq2x",       "HQ2x",          ScaleFilter::Hqx(HqxScale::Hq2x)),
        ("filter_hq3x",       "HQ3x",          ScaleFilter::Hqx(HqxScale::Hq3x)),
        ("filter_hq4x",       "HQ4x",          ScaleFilter::Hqx(HqxScale::Hq4x)),
        ("filter_xbr2x",      "xBR 2x",        ScaleFilter::Xbr(XbrScale::Xbr2x)),
        ("filter_xbr3x",      "xBR 3x",        ScaleFilter::Xbr(XbrScale::Xbr3x)),
        ("filter_xbr4x",      "xBR 4x",        ScaleFilter::Xbr(XbrScale::Xbr4x)),
        ("filter_super_xbr",  "Super xBR",     ScaleFilter::SuperXbr),
        ("filter_xbrz2x",     "xBRZ 2x",       ScaleFilter::Xbrz(XbrzScale::Xbrz2x)),
        ("filter_xbrz3x",     "xBRZ 3x",       ScaleFilter::Xbrz(XbrzScale::Xbrz3x)),
        ("filter_xbrz4x",     "xBRZ 4x",       ScaleFilter::Xbrz(XbrzScale::Xbrz4x)),
        ("filter_xbrz5x",     "xBRZ 5x",       ScaleFilter::Xbrz(XbrzScale::Xbrz5x)),
        ("filter_xbrz6x",     "xBRZ 6x",       ScaleFilter::Xbrz(XbrzScale::Xbrz6x)),
        ("filter_nedi",        "NEDI",          ScaleFilter::Nedi),
        ("filter_dcci",        "DCCI",          ScaleFilter::Dcci),
        ("filter_edi",         "EDI",           ScaleFilter::Edi),
        ("filter_omniscale",   "OmniScale",     ScaleFilter::OmniScale),
        ("filter_omniscale_l", "OmniScale Legacy", ScaleFilter::OmniScaleLegacy),
        ("filter_aanear",      "AA Nearest",    ScaleFilter::AaNearestNeighbor),
        ("filter_vectorize",   "Vectorize Legacy",          ScaleFilter::VectorizeLegacy),
        ("filter_vec_adapt",   "Vectorize Legacy Adaptive", ScaleFilter::VectorizeLegacyAdaptive),
        ("filter_vec_diff",    "Vectorize Diffusion", ScaleFilter::VectorizeDiffusion),
        ("filter_vec_sdiff",   "Vectorize Spline Diffusion", ScaleFilter::VectorizeSplineDiffusion),
        ("filter_vec_sdiffa",  "Vectorize Spline Diff Adaptive", ScaleFilter::VectorizeSplineDiffusionAdaptive),
        ("filter_vec_gpu",     "Vectorize GPU",  ScaleFilter::VectorizeGpu),
    ]
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
        let hqx_sub = Submenu::new("HQx", true);
        let xbr_sub = Submenu::new("xBR", true);
        let xbrz_sub = Submenu::new("xBRZ", true);
        let edge_sub = Submenu::new("Edge-Directed", true);

        for (id, name, filter) in filter_entries() {
            let checked = filter == scaling::ScaleFilter::Nearest;
            let item = CheckMenuItem::with_id(id, name, true, checked, None::<Accelerator>);
            match filter {
                scaling::ScaleFilter::Hqx(_) => { hqx_sub.append(&item).unwrap(); }
                scaling::ScaleFilter::Xbr(_)
                | scaling::ScaleFilter::SuperXbr => { xbr_sub.append(&item).unwrap(); }
                scaling::ScaleFilter::Xbrz(_) => { xbrz_sub.append(&item).unwrap(); }
                scaling::ScaleFilter::Nedi
                | scaling::ScaleFilter::Dcci
                | scaling::ScaleFilter::Edi => { edge_sub.append(&item).unwrap(); }
                _ => { filter_menu.append(&item).unwrap(); }
            }
            filter_items.push((item, filter));
        }

        filter_menu.append(&PredefinedMenuItem::separator()).unwrap();
        filter_menu.append_items(&[&hqx_sub, &xbr_sub, &xbrz_sub, &edge_sub]).unwrap();
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
