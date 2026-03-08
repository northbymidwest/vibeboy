use crate::model::GbModel;

/// Timer — DIV (0xFF04), TIMA (0xFF05), TMA (0xFF06), TAC (0xFF07).
///
/// DIV: Upper byte of an internal 16-bit counter that increments every T-cycle.
///      Writing any value resets the full counter to 0.
/// TIMA: Incremented at a rate determined by TAC bits 0-1.
///       On overflow, reloaded from TMA and Timer interrupt is requested.
/// TMA: Reload value for TIMA.
/// TAC: Bit 2 = timer enable, bits 1-0 = clock select.
#[derive(Clone)]
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
    /// Set when the reload (TMA→TIMA copy) fired during the current M-cycle's step().
    /// Used to resolve write conflicts: TIMA write in same M-cycle as reload is ignored;
    /// TMA write in same M-cycle as reload updates the just-reloaded TIMA value.
    reload_fired: bool,
}

impl Timer {
    pub fn new() -> Self {
        Self::post_boot(GbModel::Cgb, true, &[])
    }

    /// Hardware reset state: counter starts at 8 to account for the two internal
    /// startup M-cycles before the CPU begins executing the first instruction.
    pub fn reset() -> Self {
        Timer {
            counter: 8,
            tima: 0,
            tma: 0,
            tac: 0xF8,
            interrupt: false,
            overflow_delay: 0,
            reload_fired: false,
        }
    }

    /// Post-boot timer state for the given hardware model.
    /// For CGB/AGB, timing differs between CGB-native games and DMG compat mode.
    /// For SGB/SGB2, the timer counter depends on ROM header data (the boot ROM
    /// sends header bytes as SGB packets, and bit values affect branch timing).
    pub fn post_boot(model: GbModel, is_cgb_game: bool, rom: &[u8]) -> Self {
        let counter = match model {
            GbModel::Cgb if is_cgb_game => 0x1EA4,
            GbModel::Cgb => 0x267C,
            GbModel::Agb if is_cgb_game => 0x1EA8,
            GbModel::Agb => 0x2680,
            GbModel::Dmg0 => 0x1830,
            GbModel::Dmg | GbModel::Mgb => 0xABCC,
            GbModel::Sgb | GbModel::Sgb2 => {
                // The SGB boot ROM sends 6 packets (96 bytes) built from ROM header
                // bytes $0104-$014F. Each 1-bit in the packet data takes 4 fewer
                // T-cycles than a 0-bit (JR NZ taken vs not-taken + extra LD).
                // Only the last packet's timing affects the final counter because
                // VBlank waits between packets absorb timing differences.
                // Formula: counter = BASE - 4 * popcount(all_packet_bytes)
                let popcount = sgb_packet_popcount(rom);
                0xDD74u16.wrapping_sub(4 * popcount as u16)
            }
        };
        Timer {
            counter,
            tima: 0,
            tma: 0,
            tac: 0xF8,
            interrupt: false,
            overflow_delay: 0,
            reload_fired: false,
        }
    }

    /// Advance timer by `cycles` T-cycles (one M-cycle = 4 T-cycles).
    pub fn step(&mut self, cycles: u32) {
        self.reload_fired = false;
        for _ in 0..cycles {
            self.tick_once();
        }
    }

    fn tick_once(&mut self) {
        let old_counter = self.counter;
        self.counter = self.counter.wrapping_add(1);

        // Handle overflow delay: TIMA is reloaded from TMA 4 T-cycles after overflow.
        if self.overflow_delay > 0 {
            self.overflow_delay -= 1;
            if self.overflow_delay == 0 {
                self.tima = self.tma;
                self.interrupt = true;
                self.reload_fired = true;
            }
        }

        // Check if TIMA should increment (falling edge on the counter bit driving TIMA).
        if self.tac & 0x04 != 0 {
            let bit = self.timer_bit();
            let old_bit = (old_counter >> bit) & 1;
            let new_bit = (self.counter >> bit) & 1;
            if old_bit == 1 && new_bit == 0 {
                self.increment_tima();
            }
        }
    }

    fn increment_tima(&mut self) {
        self.tima = self.tima.wrapping_add(1);
        if self.tima == 0 {
            self.overflow_delay = 4;
        }
    }

    /// Glitch-triggered TIMA increment (from DIV or TAC writes).
    /// On real hardware, the write occurs mid-M-cycle (T2). Since our model
    /// applies the full M-cycle tick before the write, and the 4-cycle overflow
    /// delay would start counting from the next tick_mcycle (too late), we fire
    /// the interrupt immediately for write-triggered overflows.
    fn increment_tima_glitch(&mut self) {
        self.tima = self.tima.wrapping_add(1);
        if self.tima == 0 {
            self.tima = self.tma;
            self.interrupt = true;
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

    /// The "mux output": 1 when the timer is enabled and the selected counter bit is high.
    /// TIMA increments on a falling edge of this signal.
    fn mux_output(&self) -> bool {
        self.tac & 0x04 != 0 && (self.counter >> self.timer_bit()) & 1 == 1
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
            0xFF04 => {
                // Writing DIV resets the counter to 0.
                // If the currently-selected timer bit was 1, the reset creates a falling edge
                // on that bit → TIMA increments (even with timer disabled via TAC bit 2,
                // because the falling edge comes from the counter directly).
                if self.mux_output() {
                    self.increment_tima_glitch();
                }
                self.counter = 0;
            }
            0xFF05 => {
                // Writing TIMA during the reload M-cycle is ignored — the reload wins.
                // Otherwise, the write cancels any pending reload.
                if !self.reload_fired {
                    self.tima = val;
                    self.overflow_delay = 0;
                }
            }
            0xFF06 => {
                self.tma = val;
                // If the TMA→TIMA reload just fired this M-cycle, the new TMA value is
                // used retroactively (hardware reads TMA after the CPU write completes).
                if self.reload_fired {
                    self.tima = val;
                }
            }
            0xFF07 => {
                // TAC write: if the mux output falls from 1 to 0 (timer disabled or bit
                // changes to a currently-low bit), a falling edge triggers TIMA increment.
                let old_mux = self.mux_output();
                self.tac = val & 0x07;
                let new_mux = self.mux_output();
                if old_mux && !new_mux {
                    self.increment_tima_glitch();
                }
            }
            _ => {}
        }
    }

    pub fn clear_interrupt(&mut self) {
        self.interrupt = false;
    }

    /// Expose the internal 16-bit DIV counter for serial clock edge detection.
    pub fn counter(&self) -> u16 {
        self.counter
    }
}

/// Compute the popcount of the 96-byte SGB packet buffer that the SGB boot ROM
/// builds from ROM header bytes $0104-$014F. This mirrors the boot ROM code at
/// $002F-$004A which reads header data into 6 groups with running checksums.
pub fn sgb_packet_popcount(rom: &[u8]) -> u32 {
    let mut packet_data = [0u8; 96];
    let mut de: usize = 0x14F;
    let mut hl: usize = 0x57;
    let mut a: u8 = 0xFB;
    let mut c: u8 = 6;
    loop {
        let mut b: u8 = 0;
        for _ in 0..c {
            let byte = if de < rom.len() { rom[de] } else { 0 };
            packet_data[hl] = byte;
            b = b.wrapping_add(byte);
            de = de.wrapping_sub(1);
            hl = hl.wrapping_sub(1);
        }
        packet_data[hl] = b;
        hl = hl.wrapping_sub(1);
        packet_data[hl] = a;
        hl = hl.wrapping_sub(1);
        c = 14;
        a = a.wrapping_sub(2);
        if a == 0xEF {
            break;
        }
    }
    packet_data.iter().map(|b| b.count_ones()).sum()
}
