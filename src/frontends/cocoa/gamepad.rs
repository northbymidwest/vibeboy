use std::collections::{HashMap, HashSet};

use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{AllocAnyThread, Message};
use objc2_core_haptics::{
    CHHapticEngine, CHHapticEvent, CHHapticEventParameter,
    CHHapticEventParameterIDHapticIntensity, CHHapticEventParameterIDHapticSharpness,
    CHHapticEventTypeHapticContinuous, CHHapticPattern, CHHapticPatternPlayer,
};
use objc2_foundation::NSArray;
use objc2_game_controller::{
    GCAcceleration, GCController, GCHapticsLocalityDefault,
};

use super::emulator::Emulator;

#[link(name = "GameController", kind = "framework")]
unsafe extern "C" {}

#[link(name = "CoreHaptics", kind = "framework")]
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
    // Rumble: CoreHaptics engine + pattern player, retained per-controller
    haptic_engine: Option<Retained<CHHapticEngine>>,
    haptic_player: Option<Retained<objc2::runtime::ProtocolObject<dyn CHHapticPatternPlayer>>>,
    haptic_controller: Option<Retained<GCController>>,
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
            haptic_engine: None,
            haptic_player: None,
            haptic_controller: None,
            rumble_on: false,
        }
    }

    /// Set up CoreHaptics engine and continuous rumble player for the given controller.
    unsafe fn ensure_haptics(&mut self, controller: &GCController) {
        if let Some(ref hc) = self.haptic_controller {
            if std::ptr::eq(hc.as_ref() as *const GCController, controller as *const GCController)
                && self.haptic_engine.is_some()
            {
                return;
            }
        }
        self.teardown_haptics();

        let Some(haptics) = controller.haptics() else {
            return;
        };

        // GCDeviceHaptics::createEngineWithLocality is not available on macOS in the typed
        // bindings (gated behind iOS/tvOS/visionOS cfg). Use msg_send! for this one call.
        let engine_ptr: *mut AnyObject = msg_send![&*haptics, createEngineWithLocality: &**GCHapticsLocalityDefault];
        if engine_ptr.is_null() {
            return;
        }
        let engine: Retained<CHHapticEngine> =
            unsafe { Retained::retain(engine_ptr as *mut CHHapticEngine).unwrap() };

        if engine.startAndReturnError().is_err() {
            return;
        }

        let intensity_param = unsafe {
            CHHapticEventParameter::initWithParameterID_value(
                CHHapticEventParameter::alloc(),
                CHHapticEventParameterIDHapticIntensity,
                0.6,
            )
        };

        let sharpness_param = unsafe {
            CHHapticEventParameter::initWithParameterID_value(
                CHHapticEventParameter::alloc(),
                CHHapticEventParameterIDHapticSharpness,
                0.4,
            )
        };

        let params = NSArray::from_retained_slice(&[intensity_param, sharpness_param]);

        let event = unsafe {
            CHHapticEvent::initWithEventType_parameters_relativeTime_duration(
                CHHapticEvent::alloc(),
                CHHapticEventTypeHapticContinuous,
                &params,
                0.0,
                30.0,
            )
        };

        let events: Retained<NSArray<CHHapticEvent>> = NSArray::from_retained_slice(&[event]);
        let empty_params = NSArray::new();

        let Ok(pattern) = (unsafe {
            CHHapticPattern::initWithEvents_parameters_error(
                CHHapticPattern::alloc(),
                &events,
                &empty_params,
            )
        }) else {
            return;
        };

        let Ok(player) = (unsafe { engine.createPlayerWithPattern_error(&pattern) }) else {
            return;
        };

        self.haptic_engine = Some(engine);
        self.haptic_player = Some(player);
        self.haptic_controller = Some(controller.retain());
    }

    unsafe fn teardown_haptics(&mut self) {
        if let Some(ref player) = self.haptic_player {
            if self.rumble_on {
                let _ = player.stopAtTime_error(0.0);
            }
        }
        self.haptic_player = None;
        if let Some(ref engine) = self.haptic_engine {
            engine.stopWithCompletionHandler(std::ptr::null_mut());
        }
        self.haptic_engine = None;
        self.haptic_controller = None;
        self.rumble_on = false;
    }

    /// Start or stop rumble. Only sends commands on state transitions.
    pub fn set_rumble(&mut self, on: bool) {
        if on == self.rumble_on {
            return;
        }
        self.rumble_on = on;
        let Some(ref player) = self.haptic_player else {
            return;
        };
        unsafe {
            if on {
                let _ = player.startAtTime_error(0.0);
            } else {
                let _ = player.stopAtTime_error(0.0);
            }
        }
    }

    /// Poll the current GCController state. Returns true if a controller is connected.
    pub fn poll(&mut self) -> bool {
        unsafe {
            let controllers = GCController::controllers();
            if controllers.count() == 0 {
                self.teardown_haptics();
                self.clear_buttons();
                return false;
            }

            let controller = controllers.objectAtIndex(0);
            let Some(gamepad) = controller.extendedGamepad() else {
                self.teardown_haptics();
                self.clear_buttons();
                return false;
            };

            // D-pad
            let dpad = gamepad.dpad();
            self.dpad_up = dpad.up().isPressed();
            self.dpad_down = dpad.down().isPressed();
            self.dpad_left = dpad.left().isPressed();
            self.dpad_right = dpad.right().isPressed();

            // Face buttons
            let a_btn = gamepad.buttonA();
            let b_btn = gamepad.buttonB();
            self.btn_b = a_btn.isPressed();
            self.btn_a = b_btn.isPressed();

            // Menu buttons
            let menu = gamepad.buttonMenu();
            self.btn_start = menu.isPressed();
            self.btn_select = if let Some(options) = gamepad.buttonOptions() {
                options.isPressed()
            } else {
                false
            };

            // Shoulders
            self.l_shoulder = gamepad.leftShoulder().isPressed();
            self.r_shoulder = gamepad.rightShoulder().isPressed();

            // Left analog stick -> d-pad (deadzone 0.3)
            let left_stick = gamepad.leftThumbstick();
            let stick_x = left_stick.xAxis().value();
            let stick_y = left_stick.yAxis().value();
            const DEADZONE: f32 = 0.3;
            self.stick_right = stick_x > DEADZONE;
            self.stick_left = stick_x < -DEADZONE;
            self.stick_up = stick_y > DEADZONE;
            self.stick_down = stick_y < -DEADZONE;

            // Motion (accelerometer) — must activate sensors first
            if let Some(motion) = controller.motion() {
                if !motion.sensorsActive() {
                    motion.setSensorsActive(true);
                }
                let accel: GCAcceleration = motion.acceleration();
                self.accel = Some((accel.x as f32, accel.y as f32, accel.z as f32));
            } else {
                self.accel = None;
            }

            true
        }
    }

    /// Lazily initialize haptics for the current controller (call after poll()).
    pub fn ensure_haptics_ready(&mut self) {
        if self.haptic_controller.is_some() {
            return;
        }
        unsafe {
            let controllers = GCController::controllers();
            if controllers.count() == 0 {
                return;
            }
            let controller = controllers.objectAtIndex(0);
            self.ensure_haptics(&controller);
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
