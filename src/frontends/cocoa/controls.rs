use std::collections::HashMap;
use std::path::PathBuf;

use objc2::{class, msg_send};
use objc2::runtime::AnyObject;
use objc2_app_kit::{NSApplication, NSControl, NSEvent, NSFont, NSWindow};
use objc2_foundation::{NSPoint, NSRect, NSSize, NSString};

use super::emulator::Emulator;
use super::persistence::{keycode_name, default_key_map, save_key_map};
use super::K_ESCAPE;

pub(super) fn show_controls_panel(key_map: &mut HashMap<u16, u8>) {
    unsafe {
        let panel_rect = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(360.0, 340.0));
        // NSWindowStyleMask values: Titled=1, Closable=2
        let style: usize = 1 | 2;
        let panel: *mut AnyObject = msg_send![class!(NSPanel), alloc];
        let panel: *mut AnyObject = msg_send![panel,
            initWithContentRect: panel_rect
            styleMask: style
            backing: 2usize  // NSBackingStoreBuffered
            defer: false
        ];
        let title = NSString::from_str("Controls");
        let _: () = msg_send![panel, setTitle: &*title];
        let _: () = msg_send![panel, center];

        let content: *mut AnyObject = msg_send![panel, contentView];

        // Build reverse map: button -> keycode
        let mut btn_to_key: HashMap<u8, u16> = HashMap::new();
        for (&keycode, &btn) in key_map.iter() {
            btn_to_key.insert(btn, keycode);
        }

        // Create labels for each button
        let button_order = [
            (Emulator::BTN_UP, "Up"),
            (Emulator::BTN_DOWN, "Down"),
            (Emulator::BTN_LEFT, "Left"),
            (Emulator::BTN_RIGHT, "Right"),
            (Emulator::BTN_A, "A"),
            (Emulator::BTN_B, "B"),
            (Emulator::BTN_START, "Start"),
            (Emulator::BTN_SELECT, "Select"),
        ];

        let header_str = NSString::from_str("Click a key binding, then press a new key to reassign.\nPress Escape to cancel.");
        let header_frame = NSRect::new(NSPoint::new(20.0, 290.0), NSSize::new(320.0, 40.0));
        let header_label: *mut AnyObject = msg_send![class!(NSTextField), alloc];
        let header_label: *mut AnyObject = msg_send![header_label, initWithFrame: header_frame];
        let _: () = msg_send![header_label, setStringValue: &*header_str];
        let _: () = msg_send![header_label, setBezeled: false];
        let _: () = msg_send![header_label, setDrawsBackground: false];
        let _: () = msg_send![header_label, setEditable: false];
        let _: () = msg_send![header_label, setSelectable: false];
        let font = NSFont::systemFontOfSize(11.0);
        (*(header_label as *const NSControl)).setFont(Some(&font));
        let _: () = msg_send![content, addSubview: header_label];

        let mut key_labels: Vec<(u8, *mut AnyObject)> = Vec::new();

        for (i, &(btn, name)) in button_order.iter().enumerate() {
            let y = 250.0 - (i as f64 * 30.0);

            // Action name label
            let name_frame = NSRect::new(NSPoint::new(30.0, y), NSSize::new(100.0, 24.0));
            let name_label: *mut AnyObject = msg_send![class!(NSTextField), alloc];
            let name_label: *mut AnyObject = msg_send![name_label, initWithFrame: name_frame];
            let name_ns = NSString::from_str(name);
            let _: () = msg_send![name_label, setStringValue: &*name_ns];
            let _: () = msg_send![name_label, setBezeled: false];
            let _: () = msg_send![name_label, setDrawsBackground: false];
            let _: () = msg_send![name_label, setEditable: false];
            let _: () = msg_send![name_label, setSelectable: false];
            let bold_font = NSFont::boldSystemFontOfSize(13.0);
            (*(name_label as *const NSControl)).setFont(Some(&bold_font));
            let _: () = msg_send![content, addSubview: name_label];

            // Key binding button
            let key_name = btn_to_key.get(&btn)
                .map(|&k| keycode_name(k))
                .unwrap_or("(none)");
            let btn_frame = NSRect::new(NSPoint::new(150.0, y), NSSize::new(120.0, 24.0));
            let btn_view: *mut AnyObject = msg_send![class!(NSButton), alloc];
            let btn_view: *mut AnyObject = msg_send![btn_view, initWithFrame: btn_frame];
            let key_ns = NSString::from_str(key_name);
            let _: () = msg_send![btn_view, setTitle: &*key_ns];
            let _: () = msg_send![btn_view, setBezelStyle: 1isize]; // NSRoundedBezelStyle
            let _: () = msg_send![btn_view, setTag: btn as isize];
            let _: () = msg_send![content, addSubview: btn_view];

            key_labels.push((btn, btn_view));
        }

        // Reset to Defaults button
        let reset_frame = NSRect::new(NSPoint::new(115.0, 10.0), NSSize::new(130.0, 30.0));
        let reset_btn: *mut AnyObject = msg_send![class!(NSButton), alloc];
        let reset_btn: *mut AnyObject = msg_send![reset_btn, initWithFrame: reset_frame];
        let reset_title = NSString::from_str("Reset to Defaults");
        let _: () = msg_send![reset_btn, setTitle: &*reset_title];
        let _: () = msg_send![reset_btn, setBezelStyle: 1isize];
        let _: () = msg_send![content, addSubview: reset_btn];

        // Run as modal, handle key presses for remapping
        let _: () = msg_send![panel, makeKeyAndOrderFront: std::ptr::null::<AnyObject>()];

        let app: *mut AnyObject = msg_send![class!(NSApplication), sharedApplication];

        // Simple modal loop: click a button, then press a key
        let mut waiting_for_key: Option<u8> = None;

        let mode = NSString::from_str("kCFRunLoopDefaultMode");

        loop {
            let distant_future: *mut AnyObject = msg_send![class!(NSDate), distantFuture];
            let event: *mut AnyObject = msg_send![app,
                nextEventMatchingMask: u64::MAX
                untilDate: distant_future
                inMode: &*mode
                dequeue: true
            ];

            if event.is_null() { continue; }

            let event_type: u64 = msg_send![event, type];

            // Check if panel was closed
            let visible: bool = msg_send![panel, isVisible];
            if !visible { break; }

            // NSKeyDown = 10
            if event_type == 10 {
                let keycode: u16 = msg_send![event, keyCode];

                if let Some(btn) = waiting_for_key {
                    if keycode == K_ESCAPE {
                        // Cancel remapping
                        waiting_for_key = None;
                        for &(b, label) in &key_labels {
                            if b == btn {
                                let cur_name = btn_to_key.get(&b)
                                    .map(|&k| keycode_name(k))
                                    .unwrap_or("(none)");
                                let ns = NSString::from_str(cur_name);
                                let _: () = msg_send![label, setTitle: &*ns];
                            }
                        }
                        continue;
                    }

                    // Remove old mapping for this button
                    key_map.retain(|_, &mut v| v != btn);
                    key_map.remove(&keycode);
                    key_map.insert(keycode, btn);
                    btn_to_key.insert(btn, keycode);
                    save_key_map(key_map);

                    for &(b, label) in &key_labels {
                        if b == btn {
                            let ns = NSString::from_str(keycode_name(keycode));
                            let _: () = msg_send![label, setTitle: &*ns];
                        }
                    }
                    waiting_for_key = None;
                    continue;
                }

                if keycode == K_ESCAPE {
                    break;
                }
            // NSLeftMouseUp = 2
            } else if event_type == 2 {
                let location: NSPoint = msg_send![event, locationInWindow];
                for &(btn, label) in &key_labels {
                    let frame: NSRect = msg_send![label, frame];
                    if location.x >= frame.origin.x
                        && location.x <= frame.origin.x + frame.size.width
                        && location.y >= frame.origin.y
                        && location.y <= frame.origin.y + frame.size.height
                    {
                        waiting_for_key = Some(btn);
                        let ns = NSString::from_str("Press a key...");
                        let _: () = msg_send![label, setTitle: &*ns];
                        break;
                    }
                }

                // Check if Reset to Defaults was clicked
                let reset_frame: NSRect = msg_send![reset_btn, frame];
                if location.x >= reset_frame.origin.x
                    && location.x <= reset_frame.origin.x + reset_frame.size.width
                    && location.y >= reset_frame.origin.y
                    && location.y <= reset_frame.origin.y + reset_frame.size.height
                {
                    *key_map = default_key_map();
                    save_key_map(key_map);
                    btn_to_key.clear();
                    for (&keycode, &btn) in key_map.iter() {
                        btn_to_key.insert(btn, keycode);
                    }
                    for &(btn, label) in &key_labels {
                        let name = btn_to_key.get(&btn)
                            .map(|&k| keycode_name(k))
                            .unwrap_or("(none)");
                        let ns = NSString::from_str(name);
                        let _: () = msg_send![label, setTitle: &*ns];
                    }
                    waiting_for_key = None;
                }
            }

            let _: () = msg_send![app, sendEvent: event];
        }

        let _: () = msg_send![panel, close];
    }
}

pub(super) fn open_rom_dialog() -> Option<PathBuf> {
    unsafe {
        let panel: *mut AnyObject = msg_send![class!(NSOpenPanel), openPanel];
        let _: () = msg_send![panel, setCanChooseFiles: true];
        let _: () = msg_send![panel, setCanChooseDirectories: false];
        let _: () = msg_send![panel, setAllowsMultipleSelection: false];

        let gb = NSString::from_str("gb");
        let gbc = NSString::from_str("gbc");
        let types: *mut AnyObject = msg_send![class!(NSMutableArray), arrayWithCapacity: 2usize];
        let _: () = msg_send![types, addObject: &*gb];
        let _: () = msg_send![types, addObject: &*gbc];
        let _: () = msg_send![panel, setAllowedFileTypes: types];

        let response: isize = msg_send![panel, runModal];
        if response != 1 {
            return None;
        }

        let url: *mut AnyObject = msg_send![panel, URL];
        let path: *mut AnyObject = msg_send![url, path];
        let path_str: *const i8 = msg_send![path, UTF8String];
        let path = std::ffi::CStr::from_ptr(path_str)
            .to_str()
            .ok()?
            .to_string();
        Some(PathBuf::from(path))
    }
}
