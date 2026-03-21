use std::collections::{HashMap, HashSet};

use objc2::{class, msg_send, sel};
use objc2::encode::{Encode, Encoding, RefEncode};
use objc2::runtime::AnyObject;
use objc2_foundation::NSString;

use super::emulator::Emulator;

#[link(name = "GameController", kind = "framework")]
unsafe extern "C" {}

#[link(name = "CoreHaptics", kind = "framework")]
unsafe extern "C" {
    static CHHapticEventTypeHapticContinuous: *mut AnyObject;
    static CHHapticEventParameterIDHapticIntensity: *mut AnyObject;
    static CHHapticEventParameterIDHapticSharpness: *mut AnyObject;
}

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
    // Rumble: CoreHaptics engine + pattern player, retained per-controller
    haptic_engine: *mut AnyObject,
    haptic_player: *mut AnyObject,
    haptic_controller: *mut AnyObject, // track which controller owns the engine
    rumble_on: bool,
}

impl GamepadState {
    pub fn new() -> Self {
        Self {
            dpad_up: false, dpad_down: false, dpad_left: false, dpad_right: false,
            btn_a: false, btn_b: false, btn_start: false, btn_select: false,
            stick_up: false, stick_down: false, stick_left: false, stick_right: false,
            l_shoulder: false, r_shoulder: false,
            accel: None,
            haptic_engine: std::ptr::null_mut(),
            haptic_player: std::ptr::null_mut(),
            haptic_controller: std::ptr::null_mut(),
            rumble_on: false,
        }
    }

    /// Set up CoreHaptics engine and continuous rumble player for the given controller.
    unsafe fn ensure_haptics(&mut self, controller: *mut AnyObject) {
        if controller == self.haptic_controller && !self.haptic_engine.is_null() {
            return;
        }
        self.teardown_haptics();

        let haptics: *mut AnyObject = msg_send![controller, haptics];
        if haptics.is_null() {
            return;
        }

        let locality = NSString::from_str("Default");
        let engine: *mut AnyObject = msg_send![haptics, createEngineWithLocality: &*locality];
        if engine.is_null() {
            return;
        }

        let engine: *mut AnyObject = msg_send![engine, retain];

        let mut error: *mut AnyObject = std::ptr::null_mut();
        let ok: bool = msg_send![engine, startAndReturnError: &mut error];
        if !ok {
            let _: () = msg_send![engine, release];
            return;
        }

        let intensity_param: *mut AnyObject = msg_send![class!(CHHapticEventParameter), alloc];
        let intensity_param: *mut AnyObject = msg_send![intensity_param,
            initWithParameterID: CHHapticEventParameterIDHapticIntensity
            value: 0.6f32];

        let sharpness_param: *mut AnyObject = msg_send![class!(CHHapticEventParameter), alloc];
        let sharpness_param: *mut AnyObject = msg_send![sharpness_param,
            initWithParameterID: CHHapticEventParameterIDHapticSharpness
            value: 0.4f32];

        let params_arr: [*mut AnyObject; 2] = [intensity_param, sharpness_param];
        let params: *mut AnyObject = msg_send![class!(NSArray), arrayWithObjects: params_arr.as_ptr()
            count: 2usize];

        let event: *mut AnyObject = msg_send![class!(CHHapticEvent), alloc];
        let event: *mut AnyObject = msg_send![event,
            initWithEventType: CHHapticEventTypeHapticContinuous
            parameters: params
            relativeTime: 0.0f64
            duration: 30.0f64];

        let events: *mut AnyObject = msg_send![class!(NSArray), arrayWithObject: event];
        let empty_params: *mut AnyObject = msg_send![class!(NSArray), array];

        let pattern: *mut AnyObject = msg_send![class!(CHHapticPattern), alloc];
        let pattern: *mut AnyObject = msg_send![pattern,
            initWithEvents: events
            parameters: empty_params
            error: &mut error];
        if pattern.is_null() {
            let _: () = msg_send![engine, release];
            return;
        }

        let player: *mut AnyObject = msg_send![engine,
            createPlayerWithPattern: pattern
            error: &mut error];
        if player.is_null() {
            let _: () = msg_send![pattern, release];
            let _: () = msg_send![engine, release];
            return;
        }

        let player: *mut AnyObject = msg_send![player, retain];

        let _: () = msg_send![event, release];
        let _: () = msg_send![intensity_param, release];
        let _: () = msg_send![sharpness_param, release];
        let _: () = msg_send![pattern, release];

        self.haptic_engine = engine;
        self.haptic_player = player;
        self.haptic_controller = controller;
    }

    unsafe fn teardown_haptics(&mut self) {
        if !self.haptic_player.is_null() {
            if self.rumble_on {
                let _: () = msg_send![self.haptic_player, stopAtTime: 0.0f64 error: std::ptr::null_mut::<*mut AnyObject>()];
            }
            let _: () = msg_send![self.haptic_player, release];
            self.haptic_player = std::ptr::null_mut();
        }
        if !self.haptic_engine.is_null() {
            let _: () = msg_send![self.haptic_engine, stopWithCompletionHandler: std::ptr::null::<AnyObject>()];
            let _: () = msg_send![self.haptic_engine, release];
            self.haptic_engine = std::ptr::null_mut();
        }
        self.haptic_controller = std::ptr::null_mut();
        self.rumble_on = false;
    }

    /// Start or stop rumble. Only sends commands on state transitions.
    pub fn set_rumble(&mut self, on: bool) {
        if on == self.rumble_on {
            return;
        }
        self.rumble_on = on;
        if self.haptic_player.is_null() {
            return;
        }
        unsafe {
            let mut error: *mut AnyObject = std::ptr::null_mut();
            if on {
                let _: bool = msg_send![self.haptic_player,
                    startAtTime: 0.0f64
                    error: &mut error];
            } else {
                let _: bool = msg_send![self.haptic_player,
                    stopAtTime: 0.0f64
                    error: &mut error];
            }
        }
    }

    /// Poll the current GCController state. Returns true if a controller is connected.
    pub fn poll(&mut self) -> bool {
        unsafe {
            let gc_class = class!(GCController);
            let controllers: *mut AnyObject = msg_send![gc_class, controllers];
            let count: usize = msg_send![controllers, count];
            if count == 0 {
                self.teardown_haptics();
                self.clear_buttons();
                return false;
            }

            let controller: *mut AnyObject = msg_send![controllers, objectAtIndex: 0usize];
            let gamepad: *mut AnyObject = msg_send![controller, extendedGamepad];
            if gamepad.is_null() {
                self.teardown_haptics();
                self.clear_buttons();
                return false;
            }

            // D-pad
            let dpad: *mut AnyObject = msg_send![gamepad, dpad];
            let up_btn: *mut AnyObject = msg_send![dpad, up];
            let down_btn: *mut AnyObject = msg_send![dpad, down];
            let left_btn: *mut AnyObject = msg_send![dpad, left];
            let right_btn: *mut AnyObject = msg_send![dpad, right];
            self.dpad_up = msg_send![up_btn, isPressed];
            self.dpad_down = msg_send![down_btn, isPressed];
            self.dpad_left = msg_send![left_btn, isPressed];
            self.dpad_right = msg_send![right_btn, isPressed];

            // Face buttons
            let a_btn: *mut AnyObject = msg_send![gamepad, buttonA];
            let b_btn: *mut AnyObject = msg_send![gamepad, buttonB];
            self.btn_b = msg_send![a_btn, isPressed];
            self.btn_a = msg_send![b_btn, isPressed];

            // Menu buttons
            let menu: *mut AnyObject = msg_send![gamepad, buttonMenu];
            let options: *mut AnyObject = msg_send![gamepad, buttonOptions];
            self.btn_start = if !menu.is_null() { msg_send![menu, isPressed] } else { false };
            self.btn_select = if !options.is_null() { msg_send![options, isPressed] } else { false };

            // Shoulders
            let l_shoulder: *mut AnyObject = msg_send![gamepad, leftShoulder];
            let r_shoulder: *mut AnyObject = msg_send![gamepad, rightShoulder];
            self.l_shoulder = msg_send![l_shoulder, isPressed];
            self.r_shoulder = msg_send![r_shoulder, isPressed];

            // Left analog stick -> d-pad (deadzone 0.3)
            let left_stick: *mut AnyObject = msg_send![gamepad, leftThumbstick];
            let x_axis: *mut AnyObject = msg_send![left_stick, xAxis];
            let y_axis: *mut AnyObject = msg_send![left_stick, yAxis];
            let stick_x: f32 = msg_send![x_axis, value];
            let stick_y: f32 = msg_send![y_axis, value];
            const DEADZONE: f32 = 0.3;
            self.stick_right = stick_x > DEADZONE;
            self.stick_left = stick_x < -DEADZONE;
            self.stick_up = stick_y > DEADZONE;
            self.stick_down = stick_y < -DEADZONE;

            // Motion (accelerometer)
            let motion: *mut AnyObject = msg_send![controller, motion];
            if !motion.is_null() {
                #[repr(C)]
                #[derive(Copy, Clone)]
                struct GCAcceleration {
                    x: f64,
                    y: f64,
                    z: f64,
                }
                unsafe impl Encode for GCAcceleration {
                    const ENCODING: Encoding = Encoding::Struct(
                        "GCAcceleration",
                        &[Encoding::Double, Encoding::Double, Encoding::Double],
                    );
                }
                unsafe impl RefEncode for GCAcceleration {
                    const ENCODING_REF: Encoding = Encoding::Pointer(&GCAcceleration::ENCODING);
                }
                let accel: GCAcceleration = msg_send![motion, acceleration];
                self.accel = Some((accel.x as f32, accel.y as f32, accel.z as f32));
            } else {
                self.accel = None;
            }

            true
        }
    }

    /// Lazily initialize haptics for the current controller (call after poll()).
    pub fn ensure_haptics_ready(&mut self) {
        if !self.haptic_controller.is_null() {
            return;
        }
        unsafe {
            let gc_class = class!(GCController);
            let controllers: *mut AnyObject = msg_send![gc_class, controllers];
            let count: usize = msg_send![controllers, count];
            if count == 0 {
                return;
            }
            let controller: *mut AnyObject = msg_send![controllers, objectAtIndex: 0usize];
            self.ensure_haptics(controller);
        }
    }

    fn clear_buttons(&mut self) {
        self.dpad_up = false; self.dpad_down = false;
        self.dpad_left = false; self.dpad_right = false;
        self.btn_a = false; self.btn_b = false;
        self.btn_start = false; self.btn_select = false;
        self.stick_up = false; self.stick_down = false;
        self.stick_left = false; self.stick_right = false;
        self.l_shoulder = false; self.r_shoulder = false;
        self.accel = None;
    }

    pub fn apply_to_emu(&self, emu: &mut Emulator, key_map: &HashMap<u16, u8>, keys_down: &HashSet<u16>) {
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

impl Drop for GamepadState {
    fn drop(&mut self) {
        unsafe { self.teardown_haptics(); }
    }
}
