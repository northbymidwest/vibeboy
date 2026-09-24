/// Joypad / P1 register (0xFF00).
///
/// Button bit masks (internal, set = pressed):
///   Bit 0: Right   Bit 1: Left   Bit 2: Up    Bit 3: Down
///   Bit 4: A       Bit 5: B      Bit 6: Select Bit 7: Start
pub const BTN_RIGHT: u8 = 0x01;
pub const BTN_LEFT: u8 = 0x02;
pub const BTN_UP: u8 = 0x04;
pub const BTN_DOWN: u8 = 0x08;
pub const BTN_A: u8 = 0x10;
pub const BTN_B: u8 = 0x20;
pub const BTN_SELECT: u8 = 0x40;
pub const BTN_START: u8 = 0x80;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Joypad {
    /// Which button groups to read: bit5=actions, bit4=dpad (0=selected)
    p1_select: u8,
    /// Pressed buttons, internal mask (bit set = pressed)
    buttons: u8,
    /// Set when a P10-P13 input line fell (requests the joypad interrupt)
    pub interrupt: bool,
}

impl Default for Joypad {
    fn default() -> Self {
        Self::new()
    }
}

impl Joypad {
    pub fn new() -> Self {
        Joypad {
            p1_select: 0x00, // hardware reset: both select lines LOW (active)
            buttons: 0,
            interrupt: false,
        }
    }

    pub fn read(&self) -> u8 {
        let mut result = 0xCF | self.p1_select; // bits 7-6 always 1, bits 5-4 from select
        if self.p1_select & 0x10 == 0 {
            // Direction buttons selected (right=bit0, left=bit1, up=bit2, down=bit3)
            if self.buttons & BTN_RIGHT != 0 {
                result &= !0x01;
            }
            if self.buttons & BTN_LEFT != 0 {
                result &= !0x02;
            }
            if self.buttons & BTN_UP != 0 {
                result &= !0x04;
            }
            if self.buttons & BTN_DOWN != 0 {
                result &= !0x08;
            }
        }
        if self.p1_select & 0x20 == 0 {
            // Action buttons selected (a=bit0, b=bit1, select=bit2, start=bit3)
            if self.buttons & BTN_A != 0 {
                result &= !0x01;
            }
            if self.buttons & BTN_B != 0 {
                result &= !0x02;
            }
            if self.buttons & BTN_SELECT != 0 {
                result &= !0x04;
            }
            if self.buttons & BTN_START != 0 {
                result &= !0x08;
            }
        }
        result
    }

    pub fn write(&mut self, val: u8) {
        let before = self.input_lines();
        self.p1_select = val & 0x30;
        self.detect_falling_edge(before);
    }

    /// Set or clear a button. `button` is one of the BTN_* constants.
    pub fn set_button(&mut self, button: u8, pressed: bool) {
        let before = self.input_lines();
        if pressed {
            self.buttons |= button;
        } else {
            self.buttons &= !button;
        }
        self.detect_falling_edge(before);
    }

    /// The P10-P13 input lines (bits 0-3, 0 = low) as the CPU sees them:
    /// a pressed button pulls its line low only while its group is selected.
    fn input_lines(&self) -> u8 {
        self.read() & 0x0F
    }

    /// The joypad interrupt is requested when any of P10-P13 goes from high
    /// to low. That happens on a new press in a selected group, and also on
    /// a P1 write that selects a group in which a button is already held.
    fn detect_falling_edge(&mut self, before: u8) {
        if before & !self.input_lines() != 0 {
            self.interrupt = true;
        }
    }

    pub fn clear_interrupt(&mut self) {
        self.interrupt = false;
    }

    /// Get the raw button state (for preserving across rewind).
    pub fn buttons(&self) -> u8 {
        self.buttons
    }

    /// Restore raw button state (after rewind snapshot restore).
    pub fn set_buttons_raw(&mut self, buttons: u8) {
        self.buttons = buttons;
    }

    /// True when a held button in a selected group pulls one of P10-P13 low.
    pub fn any_selected_line_low(&self) -> bool {
        self.input_lines() != 0x0F
    }
}
