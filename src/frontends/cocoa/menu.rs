use std::path::PathBuf;

use cocoa::appkit::{NSApp, NSApplication, NSMenu, NSMenuItem};
use cocoa::base::{id, nil, SEL};
use cocoa::foundation::{NSAutoreleasePool, NSString};
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};

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

/// Filter entries: (display_name, ScaleFilter, is_submenu_item).
/// Items marked as submenu items go into HQx/xBR/xBRZ/Edge submenus.
pub(super) fn filter_entries() -> Vec<(&'static str, scaling::ScaleFilter)> {
    use scaling::*;
    vec![
        ("Nearest",            ScaleFilter::Nearest),
        ("Bilinear",           ScaleFilter::Bilinear),
        ("Bicubic",            ScaleFilter::Bicubic),
        ("EPX / Scale2x",     ScaleFilter::Epx),
        ("Scale3x",           ScaleFilter::Scale3x),
        ("Scale4x",           ScaleFilter::Scale4x),
        ("Eagle",             ScaleFilter::Eagle),
        ("2xSaI",             ScaleFilter::Sai2x),
        ("Super 2xSaI",      ScaleFilter::Super2xSai),
        ("Super Eagle",      ScaleFilter::SuperEagle),
        ("HQ2x",             ScaleFilter::Hqx(HqxScale::Hq2x)),
        ("HQ3x",             ScaleFilter::Hqx(HqxScale::Hq3x)),
        ("HQ4x",             ScaleFilter::Hqx(HqxScale::Hq4x)),
        ("xBR 2x",           ScaleFilter::Xbr(XbrScale::Xbr2x)),
        ("xBR 3x",           ScaleFilter::Xbr(XbrScale::Xbr3x)),
        ("xBR 4x",           ScaleFilter::Xbr(XbrScale::Xbr4x)),
        ("Super xBR",        ScaleFilter::SuperXbr),
        ("xBRZ 2x",          ScaleFilter::Xbrz(XbrzScale::Xbrz2x)),
        ("xBRZ 3x",          ScaleFilter::Xbrz(XbrzScale::Xbrz3x)),
        ("xBRZ 4x",          ScaleFilter::Xbrz(XbrzScale::Xbrz4x)),
        ("xBRZ 5x",          ScaleFilter::Xbrz(XbrzScale::Xbrz5x)),
        ("xBRZ 6x",          ScaleFilter::Xbrz(XbrzScale::Xbrz6x)),
        ("NEDI",             ScaleFilter::Nedi),
        ("DCCI",             ScaleFilter::Dcci),
        ("EDI",              ScaleFilter::Edi),
        ("OmniScale",        ScaleFilter::OmniScale),
        ("OmniScale Legacy", ScaleFilter::OmniScaleLegacy),
        ("AA Nearest",       ScaleFilter::AaNearestNeighbor),
        ("Vectorize Legacy",  ScaleFilter::VectorizeLegacy),
        ("Vectorize Legacy Adaptive", ScaleFilter::VectorizeLegacyAdaptive),
        ("Vectorize Diffusion", ScaleFilter::VectorizeDiffusion),
        ("Vectorize Spline Diffusion", ScaleFilter::VectorizeSplineDiffusion),
        ("Vectorize Spline Diff Adaptive", ScaleFilter::VectorizeSplineDiffusionAdaptive),
        ("Vectorize", ScaleFilter::Vectorize),
        ("Vectorize Adaptive", ScaleFilter::VectorizeAdaptive),
        ("Vectorize GPU", ScaleFilter::VectorizeGpu),
    ]
}

pub(super) fn filter_tag_to_filter(tag: isize) -> Option<scaling::ScaleFilter> {
    let idx = (tag - MENU_TAG_FILTER_BASE) as usize;
    filter_entries().get(idx).map(|(_, f)| *f)
}

pub(super) fn update_filter_checkmarks(app: id, selected_tag: isize) {
    unsafe {
        // Filter menu is at index 4 (VibeBoy, File, View, Emulation, Filter, ...)
        let filter_menu_item: id = msg_send![app.mainMenu(), itemAtIndex: 4isize];
        let filter_submenu: id = msg_send![filter_menu_item, submenu];
        if filter_submenu == nil { return; }
        let count: isize = msg_send![filter_submenu, numberOfItems];
        for i in 0..count {
            let item: id = msg_send![filter_submenu, itemAtIndex: i];
            let tag: isize = msg_send![item, tag];
            if tag >= MENU_TAG_FILTER_BASE {
                let state: isize = if tag == selected_tag { 1 } else { 0 };
                let _: () = msg_send![item, setState: state];
            }
            // Check submenus too
            let sub: id = msg_send![item, submenu];
            if sub != nil {
                let sub_count: isize = msg_send![sub, numberOfItems];
                for j in 0..sub_count {
                    let sub_item: id = msg_send![sub, itemAtIndex: j];
                    let sub_tag: isize = msg_send![sub_item, tag];
                    if sub_tag >= MENU_TAG_FILTER_BASE {
                        let state: isize = if sub_tag == selected_tag { 1 } else { 0 };
                        let _: () = msg_send![sub_item, setState: state];
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

pub(super) fn update_model_checkmarks(app: id, selected_tag: isize) {
    unsafe {
        // Emulation menu is at index 2, Hardware submenu is item 2 (0-indexed: after Pause, Reset)
        let emu_menu_item: id = msg_send![app.mainMenu(), itemAtIndex: 3isize];
        let emu_submenu: id = msg_send![emu_menu_item, submenu];
        let hw_item: id = msg_send![emu_submenu, itemAtIndex: 3isize]; // after pause, reset, separator
        let hw_submenu: id = msg_send![hw_item, submenu];
        if hw_submenu == nil { return; }
        let count: isize = msg_send![hw_submenu, numberOfItems];
        for i in 0..count {
            let item: id = msg_send![hw_submenu, itemAtIndex: i];
            let tag: isize = msg_send![item, tag];
            let state: isize = if tag == selected_tag { 1 } else { 0 }; // NSOnState=1, NSOffState=0
            let _: () = msg_send![item, setState: state];
        }
    }
}

pub(super) fn rebuild_recent_menu(app: id, recents: &[String]) {
    unsafe {
        // File menu is at index 1, "Recent ROMs" submenu is item at index 2 (after Open, separator)
        let file_menu_item: id = msg_send![app.mainMenu(), itemAtIndex: 1isize];
        let file_submenu: id = msg_send![file_menu_item, submenu];
        let recent_item: id = msg_send![file_submenu, itemAtIndex: 2isize];
        let recent_submenu: id = msg_send![recent_item, submenu];
        if recent_submenu == nil { return; }
        let _: () = msg_send![recent_submenu, removeAllItems];
        for (i, path) in recents.iter().enumerate() {
            let display = PathBuf::from(path);
            let name = display.file_name().unwrap_or_default().to_string_lossy();
            let item = menu_item_with_tag(
                &name, sel!(menuAction:), "",
                MENU_TAG_RECENT_BASE + i as isize,
            );
            let _: () = msg_send![recent_submenu, addItem: item];
        }
        if !recents.is_empty() {
            let _: () = msg_send![recent_submenu, addItem: NSMenuItem::separatorItem(nil)];
        }
        let clear = menu_item_with_tag("Clear Recent", sel!(menuAction:), "", MENU_TAG_CLEAR_RECENT);
        let _: () = msg_send![recent_submenu, addItem: clear];
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
    use objc::declare::ClassDecl;
    use objc::runtime::{Class, Object, Sel};
    use std::os::raw::c_void;
    use std::sync::Once;

    static REGISTER: Once = Once::new();

    pub fn register_class() -> &'static Class {
        REGISTER.call_once(|| {
            let superclass = Class::get("NSObject").unwrap();
            let mut decl = ClassDecl::new("VBMenuHandler", superclass).unwrap();

            decl.add_ivar::<*mut c_void>("_actions");

            unsafe {
                decl.add_method(
                    sel!(menuAction:),
                    handle_menu_action as extern "C" fn(&Object, Sel, id),
                );
                // Respond YES to validateMenuItem: so our items are always enabled
                decl.add_method(
                    sel!(validateMenuItem:),
                    validate_menu_item as extern "C" fn(&Object, Sel, id) -> bool,
                );
                decl.add_method(
                    sel!(applicationDockMenu:),
                    application_dock_menu as extern "C" fn(&Object, Sel, id) -> id,
                );
            }

            decl.register();
        });
        Class::get("VBMenuHandler").unwrap()
    }

    extern "C" fn validate_menu_item(_this: &Object, _sel: Sel, _item: id) -> bool {
        true
    }

    extern "C" fn handle_menu_action(this: &Object, _sel: Sel, sender: id) {
        unsafe {
            let ctx_ptr: *mut c_void = *this.get_ivar("_actions");
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

    extern "C" fn application_dock_menu(_this: &Object, _sel: Sel, _app: id) -> id {
        unsafe {
            let recents = load_recent_roms();
            if recents.is_empty() {
                return nil;
            }
            let menu = NSMenu::new(nil).autorelease();
            for (i, path) in recents.iter().enumerate() {
                let display = PathBuf::from(path);
                let name = display.file_name().unwrap_or_default().to_string_lossy();
                let item = menu_item_with_tag(
                    &name, sel!(menuAction:), "",
                    MENU_TAG_RECENT_BASE + i as isize,
                );
                let _: () = msg_send![menu, addItem: item];
            }
            menu
        }
    }

    /// Create a menu handler instance and wire it up. Returns (handler_id, actions_ptr).
    pub unsafe fn create(app: id) -> (id, *mut MenuActions) {
        let class = register_class();
        let handler: id = msg_send![class, new];

        let actions = Box::into_raw(Box::new(MenuActions::new()));
        (*handler).set_ivar("_actions", actions as *mut c_void);

        // Set as first responder target for menu items that use menuAction: selector.
        // We do this by making the handler the app's delegate — the responder chain
        // sends unhandled actions up to the app delegate.
        let _: () = msg_send![app, setDelegate: handler];

        (handler, actions)
    }
}

// ── Menu item helpers ────────────────────────────────────────────────────────

// Function key equivalents use Unicode private-use characters
pub(super) const K_F5_EQUIV: &str = "\u{F708}";  // NSF5FunctionKey
pub(super) const K_F7_EQUIV: &str = "\u{F70A}";  // NSF7FunctionKey

pub(super) unsafe fn menu_item(title: &str, action: SEL, key: &str) -> id {
    NSMenuItem::alloc(nil).initWithTitle_action_keyEquivalent_(
        NSString::alloc(nil).init_str(title),
        action,
        NSString::alloc(nil).init_str(key),
    ).autorelease()
}

pub(super) unsafe fn menu_item_with_tag(title: &str, action: SEL, key: &str, tag: isize) -> id {
    let item = NSMenuItem::alloc(nil).initWithTitle_action_keyEquivalent_(
        NSString::alloc(nil).init_str(title),
        action,
        NSString::alloc(nil).init_str(key),
    ).autorelease();
    let _: () = msg_send![item, setTag: tag];
    // Target the app delegate (first responder chain will route to us)
    item
}

pub(super) unsafe fn menu_item_with_tag_and_key(title: &str, action: SEL, tag: isize, key: &str) -> id {
    let item = NSMenuItem::alloc(nil).initWithTitle_action_keyEquivalent_(
        NSString::alloc(nil).init_str(title),
        action,
        NSString::alloc(nil).init_str(key),
    ).autorelease();
    let _: () = msg_send![item, setTag: tag];
    // Function keys need NSFunctionKeyMask
    let ns_function_key_mask: u64 = 1 << 23;
    let _: () = msg_send![item, setKeyEquivalentModifierMask: ns_function_key_mask];
    item
}

// ── create_menu_bar ──────────────────────────────────────────────────────────

pub(super) unsafe fn create_menu_bar(app: id) {
    let main_menu = NSMenu::new(nil).autorelease();

    // -- VibeBoy menu --
    let app_menu = NSMenu::new(nil).autorelease();
    let _: () = msg_send![app_menu, setTitle: NSString::alloc(nil).init_str("VibeBoy")];

    let about_item = menu_item("About VibeBoy", sel!(orderFrontStandardAboutPanel:), "");
    let _: () = msg_send![app_menu, addItem: about_item];
    let _: () = msg_send![app_menu, addItem: NSMenuItem::separatorItem(nil)];

    let quit_item = menu_item("Quit VibeBoy", sel!(terminate:), "q");
    let _: () = msg_send![app_menu, addItem: quit_item];

    let app_menu_item = NSMenuItem::new(nil).autorelease();
    let _: () = msg_send![app_menu_item, setSubmenu: app_menu];
    let _: () = msg_send![main_menu, addItem: app_menu_item];

    // -- File menu --
    let file_menu = NSMenu::new(nil).autorelease();
    let _: () = msg_send![file_menu, setTitle: NSString::alloc(nil).init_str("File")];

    let open_item = menu_item_with_tag("Open ROM\u{2026}", sel!(menuAction:), "o", MENU_TAG_OPEN);
    let _: () = msg_send![file_menu, addItem: open_item];
    let _: () = msg_send![file_menu, addItem: NSMenuItem::separatorItem(nil)];

    // Recent ROMs submenu
    let recent_submenu = NSMenu::new(nil).autorelease();
    let _: () = msg_send![recent_submenu, setTitle: NSString::alloc(nil).init_str("Recent ROMs")];
    let clear_item = menu_item_with_tag("Clear Recent", sel!(menuAction:), "", MENU_TAG_CLEAR_RECENT);
    let _: () = msg_send![recent_submenu, addItem: clear_item];
    let recent_menu_item = NSMenuItem::new(nil).autorelease();
    let _: () = msg_send![recent_menu_item, setTitle: NSString::alloc(nil).init_str("Recent ROMs")];
    let _: () = msg_send![recent_menu_item, setSubmenu: recent_submenu];
    let _: () = msg_send![file_menu, addItem: recent_menu_item];
    let _: () = msg_send![file_menu, addItem: NSMenuItem::separatorItem(nil)];

    let close_item = menu_item("Close Window", sel!(performClose:), "w");
    let _: () = msg_send![file_menu, addItem: close_item];

    let file_menu_item = NSMenuItem::new(nil).autorelease();
    let _: () = msg_send![file_menu_item, setSubmenu: file_menu];
    let _: () = msg_send![main_menu, addItem: file_menu_item];

    // -- View menu --
    let view_menu = NSMenu::new(nil).autorelease();
    let _: () = msg_send![view_menu, setTitle: NSString::alloc(nil).init_str("View")];

    let fps_item = menu_item_with_tag("Show FPS Overlay", sel!(menuAction:), "f", MENU_TAG_SHOW_FPS);
    let _: () = msg_send![view_menu, addItem: fps_item];

    let view_menu_item = NSMenuItem::new(nil).autorelease();
    let _: () = msg_send![view_menu_item, setSubmenu: view_menu];
    let _: () = msg_send![main_menu, addItem: view_menu_item];

    // -- Emulation menu --
    let emu_menu = NSMenu::new(nil).autorelease();
    let _: () = msg_send![emu_menu, setTitle: NSString::alloc(nil).init_str("Emulation")];

    let pause_item = menu_item_with_tag("Pause", sel!(menuAction:), "p", MENU_TAG_PAUSE);
    let _: () = msg_send![emu_menu, addItem: pause_item];

    let reset_item = menu_item_with_tag("Reset", sel!(menuAction:), "r", MENU_TAG_RESET);
    let _: () = msg_send![emu_menu, addItem: reset_item];
    let _: () = msg_send![emu_menu, addItem: NSMenuItem::separatorItem(nil)];

    // Hardware submenu
    let hw_submenu = NSMenu::new(nil).autorelease();
    let _: () = msg_send![hw_submenu, setTitle: NSString::alloc(nil).init_str("Hardware")];
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
        let item = menu_item_with_tag(name, sel!(menuAction:), "", *tag);
        if *tag == MENU_TAG_MODEL_AUTO {
            let _: () = msg_send![item, setState: 1isize]; // NSOnState — checked by default
        }
        let _: () = msg_send![hw_submenu, addItem: item];
    }
    let hw_menu_item = NSMenuItem::new(nil).autorelease();
    let _: () = msg_send![hw_menu_item, setTitle: NSString::alloc(nil).init_str("Hardware")];
    let _: () = msg_send![hw_menu_item, setSubmenu: hw_submenu];
    let _: () = msg_send![emu_menu, addItem: hw_menu_item];

    let _: () = msg_send![emu_menu, addItem: NSMenuItem::separatorItem(nil)];
    let controls_item = menu_item_with_tag("Controls\u{2026}", sel!(menuAction:), "", MENU_TAG_CONTROLS);
    let _: () = msg_send![emu_menu, addItem: controls_item];

    let emu_menu_item = NSMenuItem::new(nil).autorelease();
    let _: () = msg_send![emu_menu_item, setSubmenu: emu_menu];
    let _: () = msg_send![main_menu, addItem: emu_menu_item];

    // -- Filter menu --
    let filter_menu = NSMenu::new(nil).autorelease();
    let _: () = msg_send![filter_menu, setTitle: NSString::alloc(nil).init_str("Filter")];

    // Group filters into submenus for organization
    let entries = filter_entries();
    let mut hqx_sub = NSMenu::new(nil).autorelease();
    let _: () = msg_send![hqx_sub, setTitle: NSString::alloc(nil).init_str("HQx")];
    let mut xbr_sub = NSMenu::new(nil).autorelease();
    let _: () = msg_send![xbr_sub, setTitle: NSString::alloc(nil).init_str("xBR")];
    let mut xbrz_sub = NSMenu::new(nil).autorelease();
    let _: () = msg_send![xbrz_sub, setTitle: NSString::alloc(nil).init_str("xBRZ")];
    let mut edge_sub = NSMenu::new(nil).autorelease();
    let _: () = msg_send![edge_sub, setTitle: NSString::alloc(nil).init_str("Edge Detect")];

    for (i, (name, f)) in entries.iter().enumerate() {
        let tag = MENU_TAG_FILTER_BASE + i as isize;
        let item = menu_item_with_tag(name, sel!(menuAction:), "", tag);

        // Determine which submenu (if any) this filter belongs to
        match f {
            scaling::ScaleFilter::Hqx(_) => {
                let _: () = msg_send![hqx_sub, addItem: item];
            }
            scaling::ScaleFilter::Xbr(_) | scaling::ScaleFilter::SuperXbr => {
                let _: () = msg_send![xbr_sub, addItem: item];
            }
            scaling::ScaleFilter::Xbrz(_) => {
                let _: () = msg_send![xbrz_sub, addItem: item];
            }
            scaling::ScaleFilter::Nedi | scaling::ScaleFilter::Dcci | scaling::ScaleFilter::Edi => {
                let _: () = msg_send![edge_sub, addItem: item];
            }
            _ => {
                let _: () = msg_send![filter_menu, addItem: item];
            }
        }
    }

    // Add submenus
    let _: () = msg_send![filter_menu, addItem: NSMenuItem::separatorItem(nil)];
    for (sub_menu, sub_title) in [
        (hqx_sub, "HQx"), (xbr_sub, "xBR"), (xbrz_sub, "xBRZ"), (edge_sub, "Edge Detect"),
    ] {
        let sub_item = NSMenuItem::new(nil).autorelease();
        let _: () = msg_send![sub_item, setTitle: NSString::alloc(nil).init_str(sub_title)];
        let _: () = msg_send![sub_item, setSubmenu: sub_menu];
        let _: () = msg_send![filter_menu, addItem: sub_item];
    }

    let filter_menu_item = NSMenuItem::new(nil).autorelease();
    let _: () = msg_send![filter_menu_item, setSubmenu: filter_menu];
    let _: () = msg_send![main_menu, addItem: filter_menu_item];

    // -- State menu --
    let state_menu = NSMenu::new(nil).autorelease();
    let _: () = msg_send![state_menu, setTitle: NSString::alloc(nil).init_str("State")];

    let save_item = menu_item_with_tag("Save State", sel!(menuAction:), "", MENU_TAG_SAVE_STATE);
    let _: () = msg_send![state_menu, addItem: save_item];
    let load_item = menu_item_with_tag("Load State", sel!(menuAction:), "", MENU_TAG_LOAD_STATE);
    let _: () = msg_send![state_menu, addItem: load_item];
    let _: () = msg_send![state_menu, addItem: NSMenuItem::separatorItem(nil)];

    for slot in 1..=9usize {
        let title = format!("Slot {}", slot);
        let key = format!("{}", slot);
        let item = menu_item_with_tag(&title, sel!(menuAction:), &key, MENU_TAG_SLOT_BASE + slot as isize - 1);
        let _: () = msg_send![item, setKeyEquivalentModifierMask: 0u64]; // no modifier
        let _: () = msg_send![state_menu, addItem: item];
    }

    let state_menu_item = NSMenuItem::new(nil).autorelease();
    let _: () = msg_send![state_menu_item, setSubmenu: state_menu];
    let _: () = msg_send![main_menu, addItem: state_menu_item];

    // -- Window menu --
    let window_menu = NSMenu::new(nil).autorelease();
    let _: () = msg_send![window_menu, setTitle: NSString::alloc(nil).init_str("Window")];

    let minimize_item = menu_item("Minimize", sel!(performMiniaturize:), "m");
    let _: () = msg_send![window_menu, addItem: minimize_item];
    let zoom_item = menu_item("Zoom", sel!(performZoom:), "");
    let _: () = msg_send![window_menu, addItem: zoom_item];
    let _: () = msg_send![window_menu, addItem: NSMenuItem::separatorItem(nil)];
    let front_item = menu_item("Bring All to Front", sel!(arrangeInFront:), "");
    let _: () = msg_send![window_menu, addItem: front_item];

    let window_menu_item = NSMenuItem::new(nil).autorelease();
    let _: () = msg_send![window_menu_item, setSubmenu: window_menu];
    let _: () = msg_send![main_menu, addItem: window_menu_item];

    app.setMainMenu_(main_menu);
    let _: () = msg_send![app, setWindowsMenu: window_menu];
}
