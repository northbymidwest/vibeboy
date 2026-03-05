/// Timer — DIV (0xFF04), TIMA (0xFF05), TMA (0xFF06), TAC (0xFF07).
///
/// DIV: Upper byte of an internal 16-bit counter that increments every T-cycle.
///      Writing any value resets the full counter to 0.
/// TIMA: Incremented at a rate determined by TAC bits 0-1.
///       On overflow, reloaded from TMA and Timer interrupt is requested.
/// TMA: Reload value for TIMA.
/// TAC: Bit 2 = timer enable, bits 1-0 = clock select.
pub struct Timer {
    /// Internal 16-bit counter (DIV = counter >> 8)
    counter: u16,
    pub tima: u8,
    pub tma: u8,
    pub tac: u8,
    /// Set when TIMA overflows — signals interrupt request
    pub interrupt: bool,
    /// Overflow delay: TIMA is reloaded 4 cycles after overflow
    overflow_delay: u8,
}

impl Timer {
    pub fn new() -> Self {
        Timer {
            counter: 0xAB00, // GBC post-boot value
            tima: 0,
            tma: 0,
            tac: 0xF8,
            interrupt: false,
            overflow_delay: 0,
        }
    }

    /// Advance timer by `cycles` T-cycles.
    pub fn step(&mut self, cycles: u32) {
        for _ in 0..cycles {
            self.tick_once();
        }
    }

    fn tick_once(&mut self) {
        let old_counter = self.counter;
        self.counter = self.counter.wrapping_add(1);

        // Handle overflow delay
        if self.overflow_delay > 0 {
            self.overflow_delay -= 1;
            if self.overflow_delay == 0 {
                self.tima = self.tma;
                self.interrupt = true;
            }
        }

        // Check if TIMA should increment (falling edge detection on relevant bit)
        if self.tac & 0x04 != 0 {
            let bit = self.timer_bit();
            let old_bit = (old_counter >> bit) & 1;
            let new_bit = (self.counter >> bit) & 1;
            if old_bit == 1 && new_bit == 0 {
                self.tima = self.tima.wrapping_add(1);
                if self.tima == 0 {
                    self.overflow_delay = 4;
                }
            }
        }
    }

    /// Returns the counter bit that drives TIMA based on TAC clock select.
    fn timer_bit(&self) -> u16 {
        match self.tac & 0x03 {
            0 => 9,  // 4096 Hz   (every 1024 T-cycles)
            1 => 3,  // 262144 Hz (every 16 T-cycles)
            2 => 5,  // 65536 Hz  (every 64 T-cycles)
            3 => 7,  // 16384 Hz  (every 256 T-cycles)
            _ => unreachable!(),
        }
    }

    pub fn read(&self, addr: u16) -> u8 {
        match addr {
            0xFF03 => 0xFF,
            0xFF04 => (self.counter >> 8) as u8,
            0xFF05 => self.tima,
            0xFF06 => self.tma,
            0xFF07 => self.tac | 0xF8,
            _ => 0xFF,
        }
    }

    pub fn write(&mut self, addr: u16, val: u8) {
        match addr {
            0xFF04 => self.counter = 0, // Any write resets DIV
            0xFF05 => {
                self.tima = val;
                self.overflow_delay = 0;
            }
            0xFF06 => self.tma = val,
            0xFF07 => self.tac = val & 0x07,
            _ => {}
        }
    }

    pub fn clear_interrupt(&mut self) {
        self.interrupt = false;
    }
}
