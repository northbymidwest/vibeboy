use std::path::PathBuf;

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Bool, Sel};
use objc2::{class, msg_send, sel, ClassType, MainThreadOnly};
use objc2_app_kit::{NSApplication, NSMenu, NSMenuItem};
use objc2_foundation::{NSString, MainThreadMarker};

use super::model::GbModel;
use super::scaling;
use super::persistence::load_recent_roms;

// Menu item tags for action detection
pub(super) const MENU_TAG_OPEN: isize = 100;
pub(super) const MENU_TAG_PAUSE: isize = 101;
pub(super) const MENU_TAG_RESET: isize = 102;
pub(super) const MENU_TAG_SAVE_STATE: isize = 103;
pub(super) const MENU_TAG_LOAD_STATE: isize = 104;
pub(super) const MENU_TAG_SLOT_BASE: isize = 200; // 200..208 for slots 1-9
pub(super) const MENU_TAG_MODEL_AUTO: isize = 300;
pub(super) const MENU_TAG_MODEL_DMG0: isize = 301;
pub(super) const MENU_TAG_MODEL_DMG: isize = 302;
pub(super) const MENU_TAG_MODEL_MGB: isize = 303;
pub(super) const MENU_TAG_MODEL_SGB: isize = 304;
pub(super) const MENU_TAG_MODEL_SGB2: isize = 305;
pub(super) const MENU_TAG_MODEL_CGB: isize = 306;
pub(super) const MENU_TAG_MODEL_AGB: isize = 307;
pub(super) const MENU_TAG_CONTROLS: isize = 400;
pub(super) const MENU_TAG_RECENT_BASE: isize = 500; // 500..509 for recent ROMs
pub(super) const MENU_TAG_CLEAR_RECENT: isize = 510;
pub(super) const MENU_TAG_FILTER_BASE: isize = 600; // 600..699 for filters
pub(super) const MENU_TAG_SHOW_FPS: isize = 700;

/// Filter entries from the central registry.
pub(super) fn filter_entries() -> Vec<(&'static str, scaling::ScaleFilter)> {
    scaling::ScaleFilter::menu_entries().collect()
}

pub(super) fn filter_tag_to_filter(tag: isize) -> Option<scaling::ScaleFilter> {
    let idx = (tag - MENU_TAG_FILTER_BASE) as usize;
    filter_entries().get(idx).map(|(_, f)| *f)
}

pub(super) fn update_filter_checkmarks(app: &NSApplication, selected_tag: isize) {
    unsafe {
        let main_menu = match app.mainMenu() {
            Some(m) => m,
            None => return,
        };
        // Filter menu is at index 4
        let filter_menu_item = main_menu.itemAtIndex(4);
        let filter_menu_item = match filter_menu_item {
            Some(item) => item,
            None => return,
        };
        let filter_submenu = match filter_menu_item.submenu() {
            Some(s) => s,
            None => return,
        };
        let count = filter_submenu.numberOfItems();
        for i in 0..count {
            if let Some(item) = filter_submenu.itemAtIndex(i) {
                let tag = item.tag();
                if tag >= MENU_TAG_FILTER_BASE {
                    let state: isize = if tag == selected_tag { 1 } else { 0 };
                    let _: () = msg_send![&item, setState: state];
                }
                // Check submenus too
                if let Some(sub) = item.submenu() {
                    let sub_count = sub.numberOfItems();
                    for j in 0..sub_count {
                        if let Some(sub_item) = sub.itemAtIndex(j) {
                            let sub_tag = sub_item.tag();
                            if sub_tag >= MENU_TAG_FILTER_BASE {
                                let state: isize = if sub_tag == selected_tag { 1 } else { 0 };
                                let _: () = msg_send![&sub_item, setState: state];
                            }
                        }
                    }
                }
            }
        }
    }
}

pub(super) fn model_tag_to_model(tag: isize) -> Option<Option<GbModel>> {
    match tag {
        MENU_TAG_MODEL_AUTO => Some(None), // Auto
        MENU_TAG_MODEL_DMG0 => Some(Some(GbModel::Dmg0)),
        MENU_TAG_MODEL_DMG => Some(Some(GbModel::Dmg)),
        MENU_TAG_MODEL_MGB => Some(Some(GbModel::Mgb)),
        MENU_TAG_MODEL_SGB => Some(Some(GbModel::Sgb)),
        MENU_TAG_MODEL_SGB2 => Some(Some(GbModel::Sgb2)),
        MENU_TAG_MODEL_CGB => Some(Some(GbModel::Cgb)),
        MENU_TAG_MODEL_AGB => Some(Some(GbModel::Agb)),
        _ => None,
    }
}

pub(super) fn update_model_checkmarks(app: &NSApplication, selected_tag: isize) {
    unsafe {
        let main_menu = match app.mainMenu() {
            Some(m) => m,
            None => return,
        };
        let emu_menu_item = match main_menu.itemAtIndex(3) {
            Some(i) => i,
            None => return,
        };
        let emu_submenu = match emu_menu_item.submenu() {
            Some(s) => s,
            None => return,
        };
        let hw_item = match emu_submenu.itemAtIndex(3) {
            Some(i) => i,
            None => return,
        };
        let hw_submenu = match hw_item.submenu() {
            Some(s) => s,
            None => return,
        };
        let count = hw_submenu.numberOfItems();
        for i in 0..count {
            if let Some(item) = hw_submenu.itemAtIndex(i) {
                let tag = item.tag();
                let state: isize = if tag == selected_tag { 1 } else { 0 };
                let _: () = msg_send![&item, setState: state];
            }
        }
    }
}

pub(super) fn rebuild_recent_menu(mtm: MainThreadMarker, app: &NSApplication, recents: &[String]) {
    unsafe {
        let main_menu = match app.mainMenu() {
            Some(m) => m,
            None => return,
        };
        let file_menu_item = match main_menu.itemAtIndex(1) {
            Some(i) => i,
            None => return,
        };
        let file_submenu = match file_menu_item.submenu() {
            Some(s) => s,
            None => return,
        };
        let recent_item = match file_submenu.itemAtIndex(2) {
            Some(i) => i,
            None => return,
        };
        let recent_submenu = match recent_item.submenu() {
            Some(s) => s,
            None => return,
        };
        recent_submenu.removeAllItems();
        for (i, path) in recents.iter().enumerate() {
            let display = PathBuf::from(path);
            let name = display.file_name().unwrap_or_default().to_string_lossy();
            let item = menu_item_with_tag(
                mtm, &name, sel!(menuAction:), "",
                MENU_TAG_RECENT_BASE + i as isize,
            );
            recent_submenu.addItem(&item);
        }
        if !recents.is_empty() {
            recent_submenu.addItem(&NSMenuItem::separatorItem(mtm));
        }
        let clear = menu_item_with_tag(mtm, "Clear Recent", sel!(menuAction:), "", MENU_TAG_CLEAR_RECENT);
        recent_submenu.addItem(&clear);
    }
}

// ── MenuActions ──────────────────────────────────────────────────────────────

pub(super) struct MenuActions {
    pub open_rom: bool,
    pub pause_toggle: bool,
    pub reset: bool,
    pub save_state: bool,
    pub load_state: bool,
    pub select_slot: Option<usize>,
    pub select_model: Option<isize>,  // tag of selected model
    pub select_filter: Option<isize>, // tag of selected filter
    pub toggle_fps: bool,
    pub open_controls: bool,
    pub open_recent: Option<usize>,   // index into recent ROMs list
    pub clear_recent: bool,
}

impl MenuActions {
    pub fn new() -> Self {
        MenuActions {
            open_rom: false,
            pause_toggle: false,
            reset: false,
            save_state: false,
            load_state: false,
            select_slot: None,
            select_model: None,
            select_filter: None,
            toggle_fps: false,
            open_controls: false,
            open_recent: None,
            clear_recent: false,
        }
    }

    pub fn take_all(&mut self) -> MenuActions {
        std::mem::replace(self, MenuActions::new())
    }
}

// ── ObjC class registration for menu handler ─────────────────────────────────

pub(super) mod menu_handler {
    use super::*;
    use objc2::runtime::{AnyClass, AnyObject, Sel};
    use std::os::raw::c_void;

    // We use raw msg_send for the menu handler since define_class! requires
    // significant ceremony. This module registers a custom ObjC class at runtime.

    pub unsafe fn create(app: &NSApplication) -> (*mut AnyObject, *mut MenuActions) {
        use objc2::runtime::ClassBuilder;

        unsafe {
            // Register class (only once)
            static mut CLASS_REGISTERED: bool = false;
            if !CLASS_REGISTERED {
                let superclass = AnyClass::get(c"NSObject").unwrap();
                let mut builder = ClassBuilder::new(c"VBMenuHandler", superclass).unwrap();

                builder.add_ivar::<*mut c_void>(c"_actions");

                builder.add_method(
                    sel!(menuAction:),
                    handle_menu_action as unsafe extern "C" fn(*mut AnyObject, Sel, *mut AnyObject),
                );
                builder.add_method(
                    sel!(validateMenuItem:),
                    validate_menu_item as unsafe extern "C" fn(*mut AnyObject, Sel, *mut AnyObject) -> Bool,
                );
                builder.add_method(
                    sel!(applicationDockMenu:),
                    application_dock_menu as unsafe extern "C" fn(*mut AnyObject, Sel, *mut AnyObject) -> *mut AnyObject,
                );

                let _cls = builder.register();
                CLASS_REGISTERED = true;
            }

            let class = AnyClass::get(c"VBMenuHandler").unwrap();
            let handler: *mut AnyObject = msg_send![class, new];

            let actions = Box::into_raw(Box::new(MenuActions::new()));
            let ivar = class.instance_variable(c"_actions").unwrap();
            let ptr = ivar.load_ptr::<*mut c_void>(&mut *handler);
            *ptr = actions as *mut c_void;

            // Set as app delegate
            let _: () = msg_send![app, setDelegate: handler];

            (handler, actions)
        }
    }

    unsafe extern "C" fn validate_menu_item(_this: *mut AnyObject, _sel: Sel, _item: *mut AnyObject) -> Bool {
        Bool::YES
    }

    unsafe extern "C" fn handle_menu_action(this: *mut AnyObject, _sel: Sel, sender: *mut AnyObject) {
        unsafe {
            let class = AnyClass::get(c"VBMenuHandler").unwrap();
            let ivar = class.instance_variable(c"_actions").unwrap();
            let ctx_ptr: *mut c_void = *ivar.load_ptr::<*mut c_void>(&*this);
            if ctx_ptr.is_null() {
                return;
            }
            let actions = &mut *(ctx_ptr as *mut MenuActions);

            let tag: isize = msg_send![sender, tag];
            match tag {
                MENU_TAG_OPEN => actions.open_rom = true,
                MENU_TAG_PAUSE => actions.pause_toggle = true,
                MENU_TAG_RESET => actions.reset = true,
                MENU_TAG_SAVE_STATE => actions.save_state = true,
                MENU_TAG_LOAD_STATE => actions.load_state = true,
                t if t >= MENU_TAG_SLOT_BASE && t < MENU_TAG_SLOT_BASE + 9 => {
                    actions.select_slot = Some((t - MENU_TAG_SLOT_BASE) as usize);
                }
                t if t >= MENU_TAG_MODEL_AUTO && t <= MENU_TAG_MODEL_AGB => {
                    actions.select_model = Some(t);
                }
                MENU_TAG_CONTROLS => actions.open_controls = true,
                t if t >= MENU_TAG_FILTER_BASE && t < MENU_TAG_FILTER_BASE + 100 => {
                    actions.select_filter = Some(t);
                }
                MENU_TAG_SHOW_FPS => actions.toggle_fps = true,
                t if t >= MENU_TAG_RECENT_BASE && t < MENU_TAG_RECENT_BASE + 10 => {
                    actions.open_recent = Some((t - MENU_TAG_RECENT_BASE) as usize);
                }
                MENU_TAG_CLEAR_RECENT => actions.clear_recent = true,
                _ => {}
            }
        }
    }

    unsafe extern "C" fn application_dock_menu(_this: *mut AnyObject, _sel: Sel, _app: *mut AnyObject) -> *mut AnyObject {
        unsafe {
            let mtm = MainThreadMarker::new_unchecked();
            let recents = load_recent_roms();
            if recents.is_empty() {
                return std::ptr::null_mut();
            }
            let menu = NSMenu::new(mtm);
            for (i, path) in recents.iter().enumerate() {
                let display = PathBuf::from(path);
                let name = display.file_name().unwrap_or_default().to_string_lossy();
                let item = menu_item_with_tag(
                    mtm, &name, sel!(menuAction:), "",
                    MENU_TAG_RECENT_BASE + i as isize,
                );
                menu.addItem(&item);
            }
            Retained::into_raw(menu) as *mut AnyObject
        }
    }
}

// ── Menu item helpers ────────────────────────────────────────────────────────

// Function key equivalents use Unicode private-use characters
pub(super) const K_F5_EQUIV: &str = "\u{F708}";  // NSF5FunctionKey
pub(super) const K_F7_EQUIV: &str = "\u{F70A}";  // NSF7FunctionKey

pub(super) fn menu_item(mtm: MainThreadMarker, title: &str, action: Sel, key: &str) -> Retained<NSMenuItem> {
    unsafe {
        let item = NSMenuItem::initWithTitle_action_keyEquivalent(
            NSMenuItem::alloc(mtm),
            &NSString::from_str(title),
            Some(action),
            &NSString::from_str(key),
        );
        item
    }
}

pub(super) fn menu_item_with_tag(mtm: MainThreadMarker, title: &str, action: Sel, key: &str, tag: isize) -> Retained<NSMenuItem> {
    unsafe {
        let item = NSMenuItem::initWithTitle_action_keyEquivalent(
            NSMenuItem::alloc(mtm),
            &NSString::from_str(title),
            Some(action),
            &NSString::from_str(key),
        );
        item.setTag(tag);
        item
    }
}

pub(super) fn menu_item_with_tag_and_key(mtm: MainThreadMarker, title: &str, action: Sel, tag: isize, key: &str) -> Retained<NSMenuItem> {
    unsafe {
        let item = NSMenuItem::initWithTitle_action_keyEquivalent(
            NSMenuItem::alloc(mtm),
            &NSString::from_str(title),
            Some(action),
            &NSString::from_str(key),
        );
        item.setTag(tag);
        // Function keys need NSFunctionKeyMask
        let ns_function_key_mask: usize = 1 << 23;
        let _: () = msg_send![&item, setKeyEquivalentModifierMask: ns_function_key_mask];
        item
    }
}

// ── create_menu_bar ──────────────────────────────────────────────────────────

pub(super) fn create_menu_bar(mtm: MainThreadMarker, app: &NSApplication) {
    unsafe {
        let main_menu = NSMenu::new(mtm);

        // -- VibeBoy menu --
        let app_menu = NSMenu::new(mtm);
        app_menu.setTitle(&NSString::from_str("VibeBoy"));

        let about_item = menu_item(mtm, "About VibeBoy", sel!(orderFrontStandardAboutPanel:), "");
        app_menu.addItem(&about_item);
        app_menu.addItem(&NSMenuItem::separatorItem(mtm));

        let quit_item = menu_item(mtm, "Quit VibeBoy", sel!(terminate:), "q");
        app_menu.addItem(&quit_item);

        let app_menu_item = NSMenuItem::new(mtm);
        app_menu_item.setSubmenu(Some(&app_menu));
        main_menu.addItem(&app_menu_item);

        // -- File menu --
        let file_menu = NSMenu::new(mtm);
        file_menu.setTitle(&NSString::from_str("File"));

        let open_item = menu_item_with_tag(mtm, "Open ROM\u{2026}", sel!(menuAction:), "o", MENU_TAG_OPEN);
        file_menu.addItem(&open_item);
        file_menu.addItem(&NSMenuItem::separatorItem(mtm));

        // Recent ROMs submenu
        let recent_submenu = NSMenu::new(mtm);
        recent_submenu.setTitle(&NSString::from_str("Recent ROMs"));
        let clear_item = menu_item_with_tag(mtm, "Clear Recent", sel!(menuAction:), "", MENU_TAG_CLEAR_RECENT);
        recent_submenu.addItem(&clear_item);
        let recent_menu_item = NSMenuItem::new(mtm);
        recent_menu_item.setTitle(&NSString::from_str("Recent ROMs"));
        recent_menu_item.setSubmenu(Some(&recent_submenu));
        file_menu.addItem(&recent_menu_item);
        file_menu.addItem(&NSMenuItem::separatorItem(mtm));

        let close_item = menu_item(mtm, "Close Window", sel!(performClose:), "w");
        file_menu.addItem(&close_item);

        let file_menu_item = NSMenuItem::new(mtm);
        file_menu_item.setSubmenu(Some(&file_menu));
        main_menu.addItem(&file_menu_item);

        // -- View menu --
        let view_menu = NSMenu::new(mtm);
        view_menu.setTitle(&NSString::from_str("View"));

        let fps_item = menu_item_with_tag(mtm, "Show FPS Overlay", sel!(menuAction:), "f", MENU_TAG_SHOW_FPS);
        view_menu.addItem(&fps_item);

        let view_menu_item = NSMenuItem::new(mtm);
        view_menu_item.setSubmenu(Some(&view_menu));
        main_menu.addItem(&view_menu_item);

        // -- Emulation menu --
        let emu_menu = NSMenu::new(mtm);
        emu_menu.setTitle(&NSString::from_str("Emulation"));

        let pause_item = menu_item_with_tag(mtm, "Pause", sel!(menuAction:), "p", MENU_TAG_PAUSE);
        emu_menu.addItem(&pause_item);

        let reset_item = menu_item_with_tag(mtm, "Reset", sel!(menuAction:), "r", MENU_TAG_RESET);
        emu_menu.addItem(&reset_item);
        emu_menu.addItem(&NSMenuItem::separatorItem(mtm));

        // Hardware submenu
        let hw_submenu = NSMenu::new(mtm);
        hw_submenu.setTitle(&NSString::from_str("Hardware"));
        let models = [
            ("Auto", MENU_TAG_MODEL_AUTO),
            ("DMG0", MENU_TAG_MODEL_DMG0),
            ("DMG", MENU_TAG_MODEL_DMG),
            ("MGB", MENU_TAG_MODEL_MGB),
            ("SGB", MENU_TAG_MODEL_SGB),
            ("SGB2", MENU_TAG_MODEL_SGB2),
            ("CGB", MENU_TAG_MODEL_CGB),
            ("AGB", MENU_TAG_MODEL_AGB),
        ];
        for (name, tag) in &models {
            let item = menu_item_with_tag(mtm, name, sel!(menuAction:), "", *tag);
            if *tag == MENU_TAG_MODEL_AUTO {
                let _: () = msg_send![&item, setState: 1isize]; // NSOnState
            }
            hw_submenu.addItem(&item);
        }
        let hw_menu_item = NSMenuItem::new(mtm);
        hw_menu_item.setTitle(&NSString::from_str("Hardware"));
        hw_menu_item.setSubmenu(Some(&hw_submenu));
        emu_menu.addItem(&hw_menu_item);

        emu_menu.addItem(&NSMenuItem::separatorItem(mtm));
        let controls_item = menu_item_with_tag(mtm, "Controls\u{2026}", sel!(menuAction:), "", MENU_TAG_CONTROLS);
        emu_menu.addItem(&controls_item);

        let emu_menu_item = NSMenuItem::new(mtm);
        emu_menu_item.setSubmenu(Some(&emu_menu));
        main_menu.addItem(&emu_menu_item);

        // -- Filter menu --
        let filter_menu = NSMenu::new(mtm);
        filter_menu.setTitle(&NSString::from_str("Filter"));

        let entries = filter_entries();
        let hqx_sub = NSMenu::new(mtm);
        hqx_sub.setTitle(&NSString::from_str("HQx"));
        let xbr_sub = NSMenu::new(mtm);
        xbr_sub.setTitle(&NSString::from_str("xBR"));
        let xbrz_sub = NSMenu::new(mtm);
        xbrz_sub.setTitle(&NSString::from_str("xBRZ"));
        let edge_sub = NSMenu::new(mtm);
        edge_sub.setTitle(&NSString::from_str("Edge Detect"));

        for (i, (name, f)) in entries.iter().enumerate() {
            let tag = MENU_TAG_FILTER_BASE + i as isize;
            let item = menu_item_with_tag(mtm, name, sel!(menuAction:), "", tag);

            match f {
                scaling::ScaleFilter::Hqx(_) => {
                    hqx_sub.addItem(&item);
                }
                scaling::ScaleFilter::Xbr(_) | scaling::ScaleFilter::SuperXbr => {
                    xbr_sub.addItem(&item);
                }
                scaling::ScaleFilter::Xbrz(_) => {
                    xbrz_sub.addItem(&item);
                }
                scaling::ScaleFilter::Nedi | scaling::ScaleFilter::Dcci | scaling::ScaleFilter::Edi => {
                    edge_sub.addItem(&item);
                }
                _ => {
                    filter_menu.addItem(&item);
                }
            }
        }

        // Add submenus
        filter_menu.addItem(&NSMenuItem::separatorItem(mtm));
        for (sub_menu, sub_title) in [
            (&*hqx_sub, "HQx"), (&*xbr_sub, "xBR"), (&*xbrz_sub, "xBRZ"), (&*edge_sub, "Edge Detect"),
        ] {
            let sub_item = NSMenuItem::new(mtm);
            sub_item.setTitle(&NSString::from_str(sub_title));
            sub_item.setSubmenu(Some(sub_menu));
            filter_menu.addItem(&sub_item);
        }

        let filter_menu_item = NSMenuItem::new(mtm);
        filter_menu_item.setSubmenu(Some(&filter_menu));
        main_menu.addItem(&filter_menu_item);

        // -- State menu --
        let state_menu = NSMenu::new(mtm);
        state_menu.setTitle(&NSString::from_str("State"));

        let save_item = menu_item_with_tag(mtm, "Save State", sel!(menuAction:), "", MENU_TAG_SAVE_STATE);
        state_menu.addItem(&save_item);
        let load_item = menu_item_with_tag(mtm, "Load State", sel!(menuAction:), "", MENU_TAG_LOAD_STATE);
        state_menu.addItem(&load_item);
        state_menu.addItem(&NSMenuItem::separatorItem(mtm));

        for slot in 1..=9usize {
            let title = format!("Slot {}", slot);
            let key = format!("{}", slot);
            let item = menu_item_with_tag(mtm, &title, sel!(menuAction:), &key, MENU_TAG_SLOT_BASE + slot as isize - 1);
            let _: () = msg_send![&item, setKeyEquivalentModifierMask: 0usize];
            state_menu.addItem(&item);
        }

        let state_menu_item = NSMenuItem::new(mtm);
        state_menu_item.setSubmenu(Some(&state_menu));
        main_menu.addItem(&state_menu_item);

        // -- Window menu --
        let window_menu = NSMenu::new(mtm);
        window_menu.setTitle(&NSString::from_str("Window"));

        let minimize_item = menu_item(mtm, "Minimize", sel!(performMiniaturize:), "m");
        window_menu.addItem(&minimize_item);
        let zoom_item = menu_item(mtm, "Zoom", sel!(performZoom:), "");
        window_menu.addItem(&zoom_item);
        window_menu.addItem(&NSMenuItem::separatorItem(mtm));
        let front_item = menu_item(mtm, "Bring All to Front", sel!(arrangeInFront:), "");
        window_menu.addItem(&front_item);

        let window_menu_item = NSMenuItem::new(mtm);
        window_menu_item.setSubmenu(Some(&window_menu));
        main_menu.addItem(&window_menu_item);

        app.setMainMenu(Some(&main_menu));
        let _: () = msg_send![app, setWindowsMenu: &*window_menu];
    }
}
