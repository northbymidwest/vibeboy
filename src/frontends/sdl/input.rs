use super::*;

pub(super) fn handle_input(
    emu: &mut Emulator,
    ks: &sdl3::keyboard::KeyboardState,
    gp: Option<&sdl3::gamepad::Gamepad>,
) {
    const STICK_DEADZONE: i16 = 8000;

    let map: &[(Scancode, u8)] = &[
        (Scancode::Z,      Emulator::BTN_B),
        (Scancode::X,      Emulator::BTN_A),
        (Scancode::Return, Emulator::BTN_START),
        (Scancode::RShift, Emulator::BTN_SELECT),
        (Scancode::Right,  Emulator::BTN_RIGHT),
        (Scancode::Left,   Emulator::BTN_LEFT),
        (Scancode::Up,     Emulator::BTN_UP),
        (Scancode::Down,   Emulator::BTN_DOWN),
    ];

    if let Some(gp) = gp {
        // Gamepad buttons (OR'd with keyboard — either source can press)
        let gp_map: &[(GpButton, u8)] = &[
            (GpButton::East,      Emulator::BTN_A),
            (GpButton::South,     Emulator::BTN_B),
            (GpButton::Start,     Emulator::BTN_START),
            (GpButton::Back,      Emulator::BTN_SELECT),
            (GpButton::DPadRight, Emulator::BTN_RIGHT),
            (GpButton::DPadLeft,  Emulator::BTN_LEFT),
            (GpButton::DPadUp,    Emulator::BTN_UP),
            (GpButton::DPadDown,  Emulator::BTN_DOWN),
        ];

        // Left analog stick → d-pad
        let lx = gp.axis(GpAxis::LeftX);
        let ly = gp.axis(GpAxis::LeftY);

        for (sc, btn) in map {
            let kb = ks.is_scancode_pressed(*sc);
            let gp_btn = gp_map.iter().find(|(_, b)| b == btn).map_or(false, |(gb, _)| gp.button(*gb));
            let stick = match *btn {
                b if b == Emulator::BTN_RIGHT => lx > STICK_DEADZONE,
                b if b == Emulator::BTN_LEFT  => lx < -STICK_DEADZONE,
                b if b == Emulator::BTN_DOWN  => ly > STICK_DEADZONE,
                b if b == Emulator::BTN_UP    => ly < -STICK_DEADZONE,
                _ => false,
            };
            emu.set_button(*btn, kb || gp_btn || stick);
        }
    } else {
        for (sc, btn) in map {
            emu.set_button(*btn, ks.is_scancode_pressed(*sc));
        }
    }
}
