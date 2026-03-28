use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{
    NSMutableArray, NSMutableDictionary, NSNumber, NSString, NSUserDefaults,
};

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
    let defaults = NSUserDefaults::standardUserDefaults();
    let key = NSString::from_str("ControlMappings");
    let Some(dict) = defaults.dictionaryForKey(&key) else {
        return default_key_map();
    };
    let mut map = HashMap::new();
    let keys = dict.allKeys();
    let count = keys.count();
    for i in 0..count {
        let k: Retained<NSString> = keys.objectAtIndex(i);
        if let Some(v) = dict.objectForKey(&k) {
            // The value is an NSNumber stored via numberWithInteger:
            if let Ok(num) = Retained::downcast::<NSNumber>(v) {
                let k_val: u16 = k.to_string().parse().unwrap_or(0);
                map.insert(k_val, num.integerValue() as u8);
            }
        }
    }
    if map.is_empty() { default_key_map() } else { map }
}

pub(super) fn save_key_map(map: &HashMap<u16, u8>) {
    let dict = NSMutableDictionary::<NSString, NSNumber>::new();
    for (&keycode, &btn) in map {
        let k = NSString::from_str(&keycode.to_string());
        let v = NSNumber::numberWithInteger(btn as isize);
        unsafe {
            dict.setObject_forKey(
                &v,
                ProtocolObject::from_ref(&*k),
            );
        }
    }
    let defaults = NSUserDefaults::standardUserDefaults();
    let key = NSString::from_str("ControlMappings");
    // setObject_forKey expects Option<&AnyObject>; upcast the dict reference
    unsafe {
        defaults.setObject_forKey(Some(&dict as &objc2::runtime::AnyObject), &key);
    }
}

// ── Recent ROMs ─────────────────────────────────────────────────────────────

pub(super) fn load_recent_roms() -> Vec<String> {
    let defaults = NSUserDefaults::standardUserDefaults();
    let key = NSString::from_str("RecentROMs");
    let Some(arr) = defaults.arrayForKey(&key) else {
        return Vec::new();
    };
    let count = arr.count();
    let mut result = Vec::new();
    for i in 0..count {
        let obj = arr.objectAtIndex(i);
        if let Ok(s) = Retained::downcast::<NSString>(obj) {
            let path = s.to_string();
            if !path.is_empty() {
                result.push(path);
            }
        }
    }
    result
}

pub(super) fn save_recent_roms(roms: &[String]) {
    let arr = NSMutableArray::<NSString>::arrayWithCapacity(roms.len());
    for path in roms {
        let s = NSString::from_str(path);
        arr.addObject(&s);
    }
    let defaults = NSUserDefaults::standardUserDefaults();
    let key = NSString::from_str("RecentROMs");
    unsafe {
        defaults.setObject_forKey(Some(&arr as &objc2::runtime::AnyObject), &key);
    }
}

pub(super) fn add_recent_rom(path: &str) {
    let mut recents = load_recent_roms();
    recents.retain(|p| p != path);
    recents.insert(0, path.to_string());
    recents.truncate(10);
    save_recent_roms(&recents);
}
