use std::collections::HashMap;

use cocoa::base::{id, nil};
use cocoa::foundation::NSString;
use objc::{class, msg_send, sel, sel_impl};

use super::emulator::Emulator;

pub(super) fn default_key_map() -> HashMap<u16, u8> {
    let mut m = HashMap::new();
    m.insert(6, Emulator::BTN_B);       // Z
    m.insert(7, Emulator::BTN_A);       // X
    m.insert(36, Emulator::BTN_START);   // Return
    m.insert(60, Emulator::BTN_SELECT);  // Right Shift
    m.insert(124, Emulator::BTN_RIGHT);  // Right arrow
    m.insert(123, Emulator::BTN_LEFT);   // Left arrow
    m.insert(126, Emulator::BTN_UP);     // Up arrow
    m.insert(125, Emulator::BTN_DOWN);   // Down arrow
    m
}


pub(super) fn keycode_name(code: u16) -> &'static str {
    match code {
        0 => "A", 1 => "S", 2 => "D", 3 => "F", 4 => "H", 5 => "G",
        6 => "Z", 7 => "X", 8 => "C", 9 => "V", 11 => "B", 12 => "Q",
        13 => "W", 14 => "E", 15 => "R", 16 => "Y", 17 => "T",
        31 => "O", 32 => "U", 34 => "I", 35 => "P", 37 => "L",
        38 => "J", 40 => "K", 41 => ";", 45 => "N", 46 => "M",
        36 => "Return", 48 => "Tab", 49 => "Space", 51 => "Delete",
        53 => "Escape", 56 => "LShift", 60 => "RShift",
        123 => "Left", 124 => "Right", 125 => "Down", 126 => "Up",
        96 => "F5", 97 => "F6", 98 => "F7", 99 => "F3",
        _ => "?",
    }
}


pub(super) fn load_key_map() -> HashMap<u16, u8> {
    unsafe {
        let defaults: id = msg_send![class!(NSUserDefaults), standardUserDefaults];
        let key = NSString::alloc(nil).init_str("ControlMappings");
        let dict: id = msg_send![defaults, dictionaryForKey: key];
        if dict == nil {
            return default_key_map();
        }
        let mut map = HashMap::new();
        let keys: id = msg_send![dict, allKeys];
        let count: usize = msg_send![keys, count];
        for i in 0..count {
            let k: id = msg_send![keys, objectAtIndex: i];
            let v: id = msg_send![dict, objectForKey: k];
            let k_str: *const i8 = msg_send![k, UTF8String];
            let v_int: i64 = msg_send![v, integerValue];
            let k_val: u16 = std::ffi::CStr::from_ptr(k_str)
                .to_str().unwrap_or("0").parse().unwrap_or(0);
            map.insert(k_val, v_int as u8);
        }
        if map.is_empty() { default_key_map() } else { map }
    }
}

pub(super) fn save_key_map(map: &HashMap<u16, u8>) {
    unsafe {
        let dict: id = msg_send![class!(NSMutableDictionary), new];
        for (&keycode, &btn) in map {
            let k = NSString::alloc(nil).init_str(&keycode.to_string());
            let v: id = msg_send![class!(NSNumber), numberWithInteger: btn as isize];
            let _: () = msg_send![dict, setObject: v forKey: k];
        }
        let defaults: id = msg_send![class!(NSUserDefaults), standardUserDefaults];
        let key = NSString::alloc(nil).init_str("ControlMappings");
        let _: () = msg_send![defaults, setObject: dict forKey: key];
        let _: () = msg_send![dict, release];
    }
}

// ── Recent ROMs ─────────────────────────────────────────────────────────────

pub(super) fn load_recent_roms() -> Vec<String> {
    unsafe {
        let defaults: id = msg_send![class!(NSUserDefaults), standardUserDefaults];
        let key = NSString::alloc(nil).init_str("RecentROMs");
        let arr: id = msg_send![defaults, arrayForKey: key];
        if arr == nil {
            return Vec::new();
        }
        let count: usize = msg_send![arr, count];
        let mut result = Vec::new();
        for i in 0..count {
            let s: id = msg_send![arr, objectAtIndex: i];
            let cstr: *const i8 = msg_send![s, UTF8String];
            let path = std::ffi::CStr::from_ptr(cstr).to_str().unwrap_or("").to_string();
            if !path.is_empty() {
                result.push(path);
            }
        }
        result
    }
}

pub(super) fn save_recent_roms(roms: &[String]) {
    unsafe {
        let arr: id = msg_send![class!(NSMutableArray), arrayWithCapacity: roms.len()];
        for path in roms {
            let s = NSString::alloc(nil).init_str(path);
            let _: () = msg_send![arr, addObject: s];
        }
        let defaults: id = msg_send![class!(NSUserDefaults), standardUserDefaults];
        let key = NSString::alloc(nil).init_str("RecentROMs");
        let _: () = msg_send![defaults, setObject: arr forKey: key];
    }
}

pub(super) fn add_recent_rom(path: &str) {
    let mut recents = load_recent_roms();
    recents.retain(|p| p != path);
    recents.insert(0, path.to_string());
    recents.truncate(10);
    save_recent_roms(&recents);
}
