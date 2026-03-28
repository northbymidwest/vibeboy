use std::collections::HashMap;
use std::path::PathBuf;

use objc2::MainThreadOnly;
use objc2::rc::Retained;
use objc2_app_kit::{
    NSApplication, NSBackingStoreType, NSBezelStyle, NSButton, NSControl, NSEventMask,
    NSEventType, NSFont, NSModalResponseOK, NSOpenPanel, NSPanel, NSTextField, NSView, NSWindow,
    NSWindowStyleMask,
};
use objc2_foundation::{
    MainThreadMarker, NSArray, NSAutoreleasePool, NSDate, NSDefaultRunLoopMode, NSPoint, NSRect,
    NSSize, NSString,
};

use super::emulator::Emulator;
use super::persistence::{default_key_map, keycode_name, save_key_map};
use super::K_ESCAPE;

pub(super) fn show_controls_panel(key_map: &mut HashMap<u16, u8>) {
    let mtm = unsafe { MainThreadMarker::new_unchecked() };

    let panel_rect = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(360.0, 340.0));
    let style = NSWindowStyleMask::Titled | NSWindowStyleMask::Closable;

    let panel = NSPanel::initWithContentRect_styleMask_backing_defer(
        NSPanel::alloc(mtm),
        panel_rect,
        style,
        NSBackingStoreType::Buffered,
        false,
    );
    let title = NSString::from_str("Controls");
    panel.setTitle(&title);
    panel.center();

    let content_view = panel.contentView().expect("panel must have content view");

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

    let header_str =
        NSString::from_str("Click a key binding, then press a new key to reassign.\nPress Escape to cancel.");
    let header_frame = NSRect::new(NSPoint::new(20.0, 290.0), NSSize::new(320.0, 40.0));
    let header_label = NSTextField::initWithFrame(NSTextField::alloc(mtm), header_frame);
    header_label.setStringValue(&header_str);
    header_label.setBezeled(false);
    header_label.setDrawsBackground(false);
    header_label.setEditable(false);
    header_label.setSelectable(false);
    let font = NSFont::systemFontOfSize(11.0);
    header_label.setFont(Some(&font));
    content_view.addSubview(&header_label);

    let mut key_labels: Vec<(u8, Retained<NSButton>)> = Vec::new();

    for (i, &(btn, name)) in button_order.iter().enumerate() {
        let y = 250.0 - (i as f64 * 30.0);

        // Action name label
        let name_frame = NSRect::new(NSPoint::new(30.0, y), NSSize::new(100.0, 24.0));
        let name_label = NSTextField::initWithFrame(NSTextField::alloc(mtm), name_frame);
        let name_ns = NSString::from_str(name);
        name_label.setStringValue(&name_ns);
        name_label.setBezeled(false);
        name_label.setDrawsBackground(false);
        name_label.setEditable(false);
        name_label.setSelectable(false);
        let bold_font = NSFont::boldSystemFontOfSize(13.0);
        name_label.setFont(Some(&bold_font));
        content_view.addSubview(&name_label);

        // Key binding button
        let key_name = btn_to_key
            .get(&btn)
            .map(|&k| keycode_name(k))
            .unwrap_or("(none)");
        let btn_frame = NSRect::new(NSPoint::new(150.0, y), NSSize::new(120.0, 24.0));
        let btn_view = NSButton::initWithFrame(NSButton::alloc(mtm), btn_frame);
        let key_ns = NSString::from_str(key_name);
        btn_view.setTitle(&key_ns);
        btn_view.setBezelStyle(NSBezelStyle::Push);
        btn_view.setTag(btn as isize);
        content_view.addSubview(&btn_view);

        key_labels.push((btn, btn_view));
    }

    // Reset to Defaults button
    let reset_frame = NSRect::new(NSPoint::new(115.0, 10.0), NSSize::new(130.0, 30.0));
    let reset_btn = NSButton::initWithFrame(NSButton::alloc(mtm), reset_frame);
    let reset_title = NSString::from_str("Reset to Defaults");
    reset_btn.setTitle(&reset_title);
    reset_btn.setBezelStyle(NSBezelStyle::Push);
    content_view.addSubview(&reset_btn);

    // Run as modal, handle key presses for remapping
    panel.makeKeyAndOrderFront(None);

    let app = NSApplication::sharedApplication(mtm);

    // Simple modal loop: click a button, then press a key
    let mut waiting_for_key: Option<u8> = None;

    loop {
        let _pool = unsafe { NSAutoreleasePool::new() };

        let distant_future = NSDate::distantFuture();
        let Some(event) = app.nextEventMatchingMask_untilDate_inMode_dequeue(
            NSEventMask::Any,
            Some(&distant_future),
            unsafe { NSDefaultRunLoopMode },
            true,
        ) else {
            continue;
        };

        let event_type = event.r#type();

        // Check if panel was closed
        if !panel.isVisible() {
            break;
        }

        if event_type == NSEventType::KeyDown {
            let keycode = event.keyCode();

            if let Some(btn) = waiting_for_key {
                if keycode == K_ESCAPE {
                    // Cancel remapping
                    waiting_for_key = None;
                    for &(b, ref label) in &key_labels {
                        if b == btn {
                            let cur_name = btn_to_key
                                .get(&b)
                                .map(|&k| keycode_name(k))
                                .unwrap_or("(none)");
                            let ns = NSString::from_str(cur_name);
                            label.setTitle(&ns);
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

                for &(b, ref label) in &key_labels {
                    if b == btn {
                        let ns = NSString::from_str(keycode_name(keycode));
                        label.setTitle(&ns);
                    }
                }
                waiting_for_key = None;
                continue;
            }

            if keycode == K_ESCAPE {
                break;
            }
        } else if event_type == NSEventType::LeftMouseUp {
            let location = unsafe { event.locationInWindow() };
            for &(btn, ref label) in &key_labels {
                let frame: NSRect = label.frame();
                if location.x >= frame.origin.x
                    && location.x <= frame.origin.x + frame.size.width
                    && location.y >= frame.origin.y
                    && location.y <= frame.origin.y + frame.size.height
                {
                    waiting_for_key = Some(btn);
                    let ns = NSString::from_str("Press a key...");
                    label.setTitle(&ns);
                    break;
                }
            }

            // Check if Reset to Defaults was clicked
            let reset_frame: NSRect = reset_btn.frame();
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
                for &(btn, ref label) in &key_labels {
                    let name = btn_to_key
                        .get(&btn)
                        .map(|&k| keycode_name(k))
                        .unwrap_or("(none)");
                    let ns = NSString::from_str(name);
                    label.setTitle(&ns);
                }
                waiting_for_key = None;
            }
        }

        app.sendEvent(&event);
    }

    panel.close();
}

#[allow(deprecated)]
pub(super) fn open_rom_dialog() -> Option<PathBuf> {
    let mtm = unsafe { MainThreadMarker::new_unchecked() };
    let panel = NSOpenPanel::openPanel(mtm);
    panel.setCanChooseFiles(true);
    panel.setCanChooseDirectories(false);
    panel.setAllowsMultipleSelection(false);

    let gb = NSString::from_str("gb");
    let gbc = NSString::from_str("gbc");
    let types = NSArray::from_retained_slice(&[gb, gbc]);
    panel.setAllowedFileTypes(Some(&types));

    let response = panel.runModal();
    if response != NSModalResponseOK {
        return None;
    }

    let url = panel.URL()?;
    let path_ns = url.path()?;
    Some(PathBuf::from(path_ns.to_string()))
}
