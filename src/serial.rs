/// Serial port emulation with pluggable device interface.
///
/// The serial clock is derived from the DIV counter (bit 7 normal speed,
/// bit 1 CGB fast mode). Each falling edge on that bit calls
/// `serial_master_edge()`, which toggles an internal clock and, on
/// the falling half, shifts one bit through SB.

/// Trait for devices attached to the serial port (printer, link cable, etc.).
pub trait SerialDevice {
    /// Called when the Game Boy sends a bit (MSB of SB). Accumulate it.
    fn bit_start(&mut self, bit: bool);
    /// Called to read the device's response bit. Returns the bit value.
    fn bit_end(&mut self) -> bool;
    /// Advance device timers by `ticks` clock cycles.
    fn tick(&mut self, ticks: u32);
}

/// Default device: no cable connected. Returns 1 (pull-up) for every bit.
pub struct Disconnected;

impl SerialDevice for Disconnected {
    fn bit_start(&mut self, _bit: bool) {}
    fn bit_end(&mut self) -> bool { true }
    fn tick(&mut self, _ticks: u32) {}
}

pub struct Serial {
    /// SB register (0xFF01)
    pub sb: u8,
    /// SC register (0xFF02)
    pub sc: u8,
    /// Serial interrupt pending
    pub interrupt: bool,
    /// Bits shifted so far in current transfer (0–8)
    serial_count: u8,
    /// Internal master clock toggle (edges on each DIV trigger)
    master_clock: bool,
    /// Which DIV bit drives the serial clock (0x80 normal, 0x04 CGB fast)
    serial_mask: u16,
    /// CGB game mode flag (false for DMG games on CGB hardware)
    pub cgb_mode: bool,
    /// Attached serial device
    pub device: Box<dyn SerialDevice>,
    /// Buffer of serial output bytes (for Blargg test detection)
    pub serial_output: Vec<u8>,
}

#[derive(Clone)]
pub struct SerialSnapshot {
    pub sb: u8,
    pub sc: u8,
    pub interrupt: bool,
    pub serial_count: u8,
    pub master_clock: bool,
    pub serial_mask: u16,
    pub cgb_mode: bool,
}

impl Serial {
    pub fn take_snapshot(&self) -> SerialSnapshot {
        SerialSnapshot {
            sb: self.sb,
            sc: self.sc,
            interrupt: self.interrupt,
            serial_count: self.serial_count,
            master_clock: self.master_clock,
            serial_mask: self.serial_mask,
            cgb_mode: self.cgb_mode,
        }
    }

    pub fn apply_snapshot(&mut self, s: &SerialSnapshot) {
        self.sb = s.sb;
        self.sc = s.sc;
        self.interrupt = s.interrupt;
        self.serial_count = s.serial_count;
        self.master_clock = s.master_clock;
        self.serial_mask = s.serial_mask;
        self.cgb_mode = s.cgb_mode;
        // device and serial_output are transient — not restored
    }

    pub fn new(cgb_mode: bool) -> Self {
        Serial {
            sb: 0x00,
            sc: 0x7E,
            interrupt: false,
            serial_count: 0,
            master_clock: false,
            serial_mask: 0x80,
            cgb_mode,
            device: Box::new(Disconnected),
            serial_output: Vec::new(),
        }
    }

    /// Called when the DIV counter changes. Detects falling edges on the
    /// serial clock bit and drives the shift register.
    pub fn step(&mut self, old_div: u16, new_div: u16) {
        // Falling edge: bit was 1 in old, 0 in new
        let triggers = old_div & !new_div;
        if triggers & self.serial_mask != 0 {
            self.serial_master_edge();
        }
    }

    /// One edge of the serial master clock (called on falling edge of DIV bit).
    fn serial_master_edge(&mut self) {
        // Tick device timers
        let ticks = if self.serial_mask == 0x04 { 4 } else { 128 };
        self.device.tick(ticks);

        // Toggle master clock
        self.master_clock = !self.master_clock;

        // On falling edge of master clock, shift data if transfer active
        if !self.master_clock && (self.sc & 0x81) == 0x81 {
            self.serial_count += 1;
            if self.serial_count == 8 {
                self.serial_count = 0;
                self.sc &= !0x80;
                self.interrupt = true;
            }

            // Shift SB left and receive bit from device
            self.sb <<= 1;
            self.sb |= if self.device.bit_end() { 1 } else { 0 };

            // Send next bit to device (if more bits to go)
            if self.serial_count > 0 {
                self.device.bit_start(self.sb & 0x80 != 0);
            }
        }
    }

    /// Write to SC register (0xFF02).
    pub fn write_sc(&mut self, val: u8) {
        self.serial_count = 0;

        // In DMG mode, bit 1 is always set
        let val = if !self.cgb_mode { val | 0x02 } else { val };

        // If master clock is high when SC is written, fire an edge
        if self.master_clock {
            self.serial_master_edge();
        }

        self.sc = val;
        self.serial_mask = if self.cgb_mode && (val & 0x02) != 0 { 0x04 } else { 0x80 };

        // If transfer starting with internal clock, send first bit to device
        if (val & 0x80) != 0 && (val & 0x01) != 0 {
            // Also push to serial_output for test ROM detection
            self.serial_output.push(self.sb);

            self.device.bit_start(self.sb & 0x80 != 0);
        }
    }

    /// Read SC register (0xFF02).
    pub fn read_sc(&self) -> u8 {
        if self.cgb_mode {
            self.sc | 0x7C
        } else {
            self.sc | 0x7E
        }
    }
}
