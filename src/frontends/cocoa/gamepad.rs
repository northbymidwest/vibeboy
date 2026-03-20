use std::collections::{HashMap, HashSet};

use cocoa::base::id;
use objc::{class, msg_send, sel, sel_impl};

use super::emulator::Emulator;

#[link(name = "GameController", kind = "framework")]
unsafe extern "C" {}

pub(super) struct GamepadState {
    // Button state from the most recently polled controller
    pub dpad_up: bool,
    pub dpad_down: bool,
    pub dpad_left: bool,
    pub dpad_right: bool,
    pub btn_a: bool,
    pub btn_b: bool,
    pub btn_start: bool,
    pub btn_select: bool,
    // Analog stick -> d-pad
    pub stick_up: bool,
    pub stick_down: bool,
    pub stick_left: bool,
    pub stick_right: bool,
    // Shoulder buttons for rewind/fast-forward
    pub l_shoulder: bool,
    pub r_shoulder: bool,
    // Accelerometer data (x, y, z in g)
    pub accel: Option<(f32, f32, f32)>,
}

impl GamepadState {
    pub fn new() -> Self {
        Self {
            dpad_up: false, dpad_down: false, dpad_left: false, dpad_right: false,
            btn_a: false, btn_b: false, btn_start: false, btn_select: false,
            stick_up: false, stick_down: false, stick_left: false, stick_right: false,
            l_shoulder: false, r_shoulder: false,
            accel: None,
        }
    }

    /// Poll the current GCController state. Returns true if a controller is connected.
    pub fn poll(&mut self) -> bool {
        unsafe {
            let gc_class = class!(GCController);
            let controllers: id = msg_send![gc_class, controllers];
            let count: usize = msg_send![controllers, count];
            if count == 0 {
                *self = Self::new();
                return false;
            }

            let controller: id = msg_send![controllers, objectAtIndex: 0usize];
            let gamepad: id = msg_send![controller, extendedGamepad];
            if gamepad.is_null() {
                *self = Self::new();
                return false;
            }

            // D-pad
            let dpad: id = msg_send![gamepad, dpad];
            let up_btn: id = msg_send![dpad, up];
            let down_btn: id = msg_send![dpad, down];
            let left_btn: id = msg_send![dpad, left];
            let right_btn: id = msg_send![dpad, right];
            self.dpad_up = msg_send![up_btn, isPressed];
            self.dpad_down = msg_send![down_btn, isPressed];
            self.dpad_left = msg_send![left_btn, isPressed];
            self.dpad_right = msg_send![right_btn, isPressed];

            // Face buttons — East=A, South=B (matching SDL3 layout)
            let a_btn: id = msg_send![gamepad, buttonA]; // South (cross/B)
            let b_btn: id = msg_send![gamepad, buttonB]; // East (circle/A)
            self.btn_b = msg_send![a_btn, isPressed]; // GC buttonA = GB B
            self.btn_a = msg_send![b_btn, isPressed]; // GC buttonB = GB A

            // Menu buttons
            let menu: id = msg_send![gamepad, buttonMenu];
            let options: id = msg_send![gamepad, buttonOptions];
            self.btn_start = if !menu.is_null() { msg_send![menu, isPressed] } else { false };
            self.btn_select = if !options.is_null() { msg_send![options, isPressed] } else { false };

            // Shoulders
            let l_shoulder: id = msg_send![gamepad, leftShoulder];
            let r_shoulder: id = msg_send![gamepad, rightShoulder];
            self.l_shoulder = msg_send![l_shoulder, isPressed];
            self.r_shoulder = msg_send![r_shoulder, isPressed];

            // Left analog stick -> d-pad (deadzone 0.3)
            let left_stick: id = msg_send![gamepad, leftThumbstick];
            let x_axis: id = msg_send![left_stick, xAxis];
            let y_axis: id = msg_send![left_stick, yAxis];
            let stick_x: f32 = msg_send![x_axis, value];
            let stick_y: f32 = msg_send![y_axis, value];
            const DEADZONE: f32 = 0.3;
            self.stick_right = stick_x > DEADZONE;
            self.stick_left = stick_x < -DEADZONE;
            self.stick_up = stick_y > DEADZONE;
            self.stick_down = stick_y < -DEADZONE;

            // Motion (accelerometer)
            let motion: id = msg_send![controller, motion];
            if !motion.is_null() {
                #[repr(C)]
                struct GCAcceleration {
                    x: f64,
                    y: f64,
                    z: f64,
                }
                let accel: GCAcceleration = msg_send![motion, acceleration];
                self.accel = Some((accel.x as f32, accel.y as f32, accel.z as f32));
            } else {
                self.accel = None;
            }

            true
        }
    }

    pub fn apply_to_emu(&self, emu: &mut Emulator, key_map: &HashMap<u16, u8>, keys_down: &HashSet<u16>) {
        // For each GB button, OR keyboard + gamepad state
        let kb_state = |btn: u8| -> bool {
            key_map.iter().any(|(k, b)| *b == btn && keys_down.contains(k))
        };

        let btns: &[(u8, bool)] = &[
            (Emulator::BTN_UP,     self.dpad_up || self.stick_up),
            (Emulator::BTN_DOWN,   self.dpad_down || self.stick_down),
            (Emulator::BTN_LEFT,   self.dpad_left || self.stick_left),
            (Emulator::BTN_RIGHT,  self.dpad_right || self.stick_right),
            (Emulator::BTN_A,      self.btn_a),
            (Emulator::BTN_B,      self.btn_b),
            (Emulator::BTN_START,  self.btn_start),
            (Emulator::BTN_SELECT, self.btn_select),
        ];

        for &(btn, gp_pressed) in btns {
            emu.set_button(btn, kb_state(btn) || gp_pressed);
        }
    }
}
