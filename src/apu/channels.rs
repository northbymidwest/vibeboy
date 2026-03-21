/// Channel structs: SquareCh (CH1/CH2), Sweep (CH1), WaveCh (CH3), NoiseCh (CH4).

pub(super) const DUTY_TABLE: [[u8; 8]; 4] = [
    [0, 0, 0, 0, 0, 0, 0, 1], // 12.5%
    [1, 0, 0, 0, 0, 0, 0, 1], // 25%
    [1, 0, 0, 0, 0, 1, 1, 1], // 50%
    [0, 1, 1, 1, 1, 1, 1, 0], // 75%
];

// Noise base divisor in T-cycles for the counter increment rate.
pub(super) const NOISE_DIVISORS: [u32; 8] = [4, 8, 16, 24, 32, 40, 48, 56];

// ── Envelope clock tracking ───────────────────────────────────────────────

#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub(super) struct EnvelopeClock {
    pub clock: bool,
    pub locked: bool,
    pub should_lock: bool,
}

/// Update envelope clock state machine.
pub(super) fn set_envelope_clock(ec: &mut EnvelopeClock, value: bool, direction: bool, volume: u8) {
    if ec.clock == value { return; }
    if value {
        ec.clock = true;
        ec.should_lock = (volume == 0xF && direction) || (volume == 0x0 && !direction);
    } else {
        ec.clock = false;
        ec.locked |= ec.should_lock;
    }
}

// ── Square channel ─────────────────────────────────────────────────────────

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct SquareCh {
    // NRx1
    pub duty: u8,
    // NRx2
    pub env_init_vol: u8,
    pub env_add: bool,
    pub env_period: u8,
    // NRx3/NRx4
    pub freq: u16,         // 11-bit
    pub len_enable: bool,

    // Internal
    pub enabled: bool,
    pub dac_on: bool,
    pub freq_timer: u32,
    pub duty_pos: u8,
    pub length_counter: u16,
    pub volume: u8,
    pub env_timer: u8,
    pub volume_countdown: u8,
    pub current_sample: u8,
    pub sample_suppressed: bool,
    pub just_reloaded: bool,
    pub did_tick: bool,
    pub delay: u32,

    // Envelope clock tracking
    pub envelope_clock: EnvelopeClock,

    // Raw NRx2 value for glitch detection
    pub nrx2_raw: u8,
}

impl SquareCh {
    pub fn new() -> Self {
        SquareCh {
            duty: 0,
            env_init_vol: 0,
            env_add: false,
            env_period: 0,
            freq: 0,
            len_enable: false,
            enabled: false,
            dac_on: false,
            freq_timer: 8,
            duty_pos: 0,
            length_counter: 64,
            volume: 0,
            env_timer: 0,
            volume_countdown: 0,
            current_sample: 0,
            sample_suppressed: false,
            just_reloaded: false,
            did_tick: false,
            delay: 0,
            envelope_clock: EnvelopeClock::default(),
            nrx2_raw: 0,
        }
    }

    pub fn reload_period(&self) -> u32 {
        (2048 - self.freq as u32) * 4
    }

    pub fn tick_freq(&mut self, mut cycles: u32) {
        self.just_reloaded = false;
        self.delay = self.delay.saturating_sub(cycles);
        loop {
            if self.freq_timer > cycles {
                self.freq_timer -= cycles;
                return;
            }
            cycles -= self.freq_timer;
            self.duty_pos = (self.duty_pos + 1) & 7;
            self.did_tick = true;
            self.sample_suppressed = false;
            self.update_sample();
            self.freq_timer = self.reload_period();
            if self.freq_timer == 0 {
                self.freq_timer = 1;
            }
            if cycles == 0 {
                self.just_reloaded = true;
                return;
            }
        }
    }

    pub fn update_sample(&mut self) {
        if !self.enabled || !self.dac_on || self.sample_suppressed {
            self.current_sample = 0;
        } else {
            self.current_sample = DUTY_TABLE[self.duty as usize][self.duty_pos as usize] * self.volume;
        }
    }

    pub fn output(&self) -> u8 {
        self.current_sample
    }

    pub fn clock_length(&mut self) {
        if self.len_enable && self.length_counter > 0 {
            self.length_counter -= 1;
            if self.length_counter == 0 {
                self.enabled = false;
            }
        }
    }

    /// Tick envelope: adjust volume unconditionally (after checking locked and env_period).
    pub fn tick_envelope(&mut self) {
        set_envelope_clock(&mut self.envelope_clock, false, false, 0);
        if self.envelope_clock.locked { return; }
        if self.env_period == 0 { return; }
        set_envelope_clock(&mut self.envelope_clock, false, false, 0);
        if self.env_add {
            self.volume = (self.volume + 1) & 0xF;
        } else {
            self.volume = self.volume.wrapping_sub(1) & 0xF;
        }
        self.update_sample();
    }
}

// ── CH1 sweep ──────────────────────────────────────────────────────────────
//
// The sweep operates in two domains:
//   128 Hz (period timer): clocked by the frame sequencer when div_divider & 3 == 3.
//     When the period timer expires, a new sweep step executes: compute delta,
//     apply to frequency, then schedule an overflow check.
//   1 MHz (calculation): the overflow check runs after a short delay (reload_timer)
//     followed by a countdown equal to the shift value. Both tick in the 1MHz
//     domain (one step per lf_div toggle).
//
// On trigger:
//   shadow = current frequency
//   addend = shadow >> shift (or 0 if shift == 0)
//   schedule overflow check via reload_timer + calc_countdown
//   period timer reloaded
//
// Overflow check:
//   sum = shadow + addend (with negate: shadow - addend via XOR trick)
//   if sum > 0x7FF and not negating → disable channel
//   update shadow from current frequency (unless in restart hold window)
//
// NR10 write:
//   if negate was used and negate bit is now clear → check overflow with
//   old negate state and completed addend, disable if overflow

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct Sweep {
    // NR10 register fields
    pub period: u8,
    pub negate: bool,
    pub shift: u8,

    // Core state
    pub enabled: bool,
    pub shadow: u16,
    pub neg_used: bool,

    /// Period timer: counts down from period (or 8 if period==0). When it
    /// reaches 0, a sweep step executes and the timer reloads.
    timer: u8,

    /// Current sweep addend (shadow >> shift, possibly negated)
    addend: u16,

    /// Delay before overflow check fires (counts down in 1MHz domain)
    reload_timer: u8,

    /// Countdown from shift value in 1MHz domain
    calc_countdown: u8,

    /// True when an overflow check is scheduled
    pub calc_pending: bool,

    /// Completed addend from last overflow check (for NR10 negate-clear checks)
    completed_addend: u16,

    /// Restart hold: prevents shadow update on rapid retrigger (counts down in 1MHz)
    restart_hold: u8,
}

impl Sweep {
    pub fn new() -> Self {
        Sweep {
            period: 0, negate: false, shift: 0,
            enabled: false, shadow: 0, neg_used: false,
            timer: 0, addend: 0,
            reload_timer: 0, calc_countdown: 0,
            calc_pending: false, completed_addend: 0,
            restart_hold: 0,
        }
    }

    /// Initialize sweep on CH1 trigger. May disable the channel via `ch`.
    pub fn trigger(&mut self, ch: &mut SquareCh, was_active: bool, lf_div: u32) {
        let ch1_freq = ch.freq;
        self.shadow = ch1_freq;
        self.completed_addend = 0;
        self.neg_used = false;
        self.restart_hold = 2;

        self.enabled = self.period != 0 || self.shift != 0;
        self.timer = if self.period == 0 { 8 } else { self.period };

        if self.shift != 0 {
            self.addend = ch1_freq >> self.shift;

            // Immediate overflow check at trigger (required by blargg test 06).
            // If shadow + delta > 2047, disable immediately.
            let delta = self.shadow >> self.shift;
            let check = if self.negate {
                self.neg_used = true;
                self.shadow.wrapping_sub(delta)
            } else {
                self.shadow + delta
            };
            if check > 2047 {
                ch.enabled = false;
            }

            // Deferred overflow check disabled for now — the immediate check
            // above handles the blargg test 06 case. The sample-accurate deferred
            // model needs more work to not break tests 04/05/07.
            self.calc_pending = false;
        } else {
            self.addend = 0;
            self.calc_pending = false;
        }
    }

    /// Called from the frame sequencer at 128Hz (div_divider & 3 == 3).
    pub fn clock_period(&mut self, ch: &mut SquareCh) {
        if self.timer > 0 {
            self.timer -= 1;
        }
        if self.timer > 0 { return; }
        self.timer = if self.period == 0 { 8 } else { self.period };
        if !self.enabled { return; }
        if self.period == 0 { return; }

        // Sweep step: compute hypothetical new frequency.
        let delta = self.shadow >> self.shift;
        let new_freq = if self.negate {
            self.neg_used = true;
            self.shadow.wrapping_sub(delta)
        } else {
            self.shadow + delta
        };

        // First overflow check
        if new_freq > 2047 {
            ch.enabled = false;
            return;
        }

        // Only update shadow/freq when shift != 0
        if self.shift != 0 {
            self.shadow = new_freq;
            ch.freq = new_freq;
            self.addend = new_freq >> self.shift;
            ch.update_sample();
        }

        // Second overflow check (always, even if shift == 0)
        let delta2 = self.shadow >> self.shift;
        let next_freq = if self.negate {
            self.shadow.wrapping_sub(delta2)
        } else {
            self.shadow + delta2
        };
        if next_freq > 2047 {
            ch.enabled = false;
        }
    }

    /// Perform the overflow check. Computes the hypothetical next frequency
    /// and disables CH1 if it would overflow (> 2047) in non-negate mode.
    fn overflow_check(&mut self, ch: &mut SquareCh) {
        let delta = self.shadow >> self.shift;
        let next_freq = if self.negate {
            self.neg_used = true;
            self.shadow.wrapping_sub(delta)
        } else {
            self.shadow + delta
        };
        if next_freq > 2047 {
            ch.enabled = false;
        }
        self.completed_addend = delta;
    }

    /// Step the sweep calculation in the 1MHz domain.
    /// Called once per lf_div toggle (every 2 T-cycles).
    pub fn step_1mhz(&mut self, ch: &mut SquareCh) {
        if self.restart_hold > 0 {
            self.restart_hold -= 1;
        }

        if !self.calc_pending { return; }

        // Reload timer must expire first
        if self.reload_timer > 0 {
            self.reload_timer -= 1;
            if self.reload_timer == 0 && self.calc_countdown == 0 {
                self.overflow_check(ch);
                self.calc_pending = false;
            }
            return;
        }

        // Calculation countdown
        if self.calc_countdown > 0 {
            self.calc_countdown -= 1;
        }
        if self.calc_countdown == 0 {
            self.overflow_check(ch);
            self.calc_pending = false;
        }
    }

    /// Handle NR10 write.
    pub fn write_nr10(&mut self, val: u8, ch: &mut SquareCh) {
        let old_neg = self.negate;
        self.period = (val >> 4) & 0x07;
        self.negate = val & 0x08 != 0;
        self.shift = val & 0x07;

        // Negate-clear check: if negate was used and is now cleared,
        // disable CH1. The old negate state determines the check.
        if self.neg_used && old_neg && !self.negate {
            ch.enabled = false;
        }
    }
}

// ── Wave channel (CH3) ────────────────────────────────────────────────────

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct WaveCh {
    pub dac_on: bool,
    pub vol_code: u8,
    pub freq: u16,
    pub len_enable: bool,
    pub enabled: bool,
    pub freq_timer: u32,
    pub sample_pos: u8,
    pub length_counter: u16,
    pub last_nibble: u8,
    pub wave_form_just_read: bool,
    pub wave_ram: [u8; 16],
}

impl WaveCh {
    pub fn new() -> Self {
        WaveCh {
            dac_on: false,
            vol_code: 0,
            freq: 0,
            len_enable: false,
            enabled: false,
            freq_timer: 0,
            sample_pos: 0,
            length_counter: 256,
            last_nibble: 0,
            wave_form_just_read: false,
            wave_ram: [0; 16],
        }
    }

    pub fn reload_period(&self) -> u32 {
        (2048 - self.freq as u32) * 2
    }

    pub fn tick_freq(&mut self, mut cycles: u32) {
        if !self.enabled {
            return;
        }
        self.wave_form_just_read = false;
        loop {
            if self.freq_timer > cycles {
                self.freq_timer -= cycles;
                return;
            }
            cycles -= self.freq_timer;
            self.sample_pos = (self.sample_pos + 1) & 31;
            let byte = self.wave_ram[(self.sample_pos >> 1) as usize];
            self.last_nibble = if self.sample_pos & 1 == 0 { byte >> 4 } else { byte & 0xF };
            self.wave_form_just_read = cycles == 0;
            self.freq_timer = self.reload_period();
            if self.freq_timer == 0 {
                self.freq_timer = 1;
            }
        }
    }

    pub fn output(&self) -> u8 {
        if !self.enabled || !self.dac_on {
            return 0;
        }
        let sample = self.last_nibble;
        match self.vol_code {
            0 => 0,
            1 => sample,
            2 => sample >> 1,
            3 => sample >> 2,
            _ => 0,
        }
    }

    pub fn clock_length(&mut self) {
        if self.len_enable && self.length_counter > 0 {
            self.length_counter -= 1;
            if self.length_counter == 0 {
                self.enabled = false;
            }
        }
    }
}

// ── Noise channel (CH4) ──────────────────────────────────────────────────

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct NoiseCh {
    pub env_init_vol: u8,
    pub env_add: bool,
    pub env_period: u8,
    pub clock_shift: u8,
    pub lfsr_narrow: bool,
    pub divisor_code: u8,
    pub len_enable: bool,

    pub enabled: bool,
    pub dac_on: bool,
    pub length_counter: u16,
    pub volume: u8,
    pub env_timer: u8,
    pub volume_countdown: u8,

    // Counter-based frequency timing
    pub counter: u32,
    pub counter_countdown: u32,
    pub counter_active: bool,
    pub countdown_reloaded: bool,
    pub alignment: u32,

    // LFSR
    pub lfsr: u16,
    pub lfsr_sample: bool,

    // Envelope clock tracking
    pub envelope_clock: EnvelopeClock,

    // Raw NRx2 value for glitch detection
    pub nrx2_raw: u8,
}

impl NoiseCh {
    pub fn new() -> Self {
        NoiseCh {
            env_init_vol: 0,
            env_add: false,
            env_period: 0,
            clock_shift: 0,
            lfsr_narrow: false,
            divisor_code: 0,
            len_enable: false,
            enabled: false,
            dac_on: false,
            length_counter: 64,
            volume: 0,
            env_timer: 0,
            volume_countdown: 0,
            counter: 0,
            counter_countdown: 0,
            counter_active: false,
            countdown_reloaded: false,
            alignment: 0,
            lfsr: 0,
            lfsr_sample: false,
            envelope_clock: EnvelopeClock::default(),
            nrx2_raw: 0,
        }
    }

    pub fn base_divisor(&self) -> u32 {
        NOISE_DIVISORS[self.divisor_code as usize]
    }

    /// Perform one LFSR step: XOR bits 0 and 1, shift right, put result
    /// in the high bit (14). In narrow mode, also write to bit 6.
    /// Output is the inverse of bit 0 after the shift.
    fn step_lfsr(&mut self) {
        let bit0 = self.lfsr & 1;
        let bit1 = (self.lfsr >> 1) & 1;
        let feedback = bit0 ^ bit1;
        self.lfsr >>= 1;
        self.lfsr |= feedback << 14;
        if self.lfsr_narrow {
            self.lfsr = (self.lfsr & !0x40) | (feedback << 6);
        }
        // Output is inverted bit 0 (Pan Docs)
        self.lfsr_sample = self.lfsr & 1 == 0;
    }

    /// Advance the 14-bit hardware counter by `cycles` T-cycles.
    pub fn tick_counter(&mut self, mut cycles: u32) {
        if !self.counter_active { return; }
        self.countdown_reloaded = false;
        while cycles > 0 {
            if self.counter_countdown > cycles {
                self.counter_countdown -= cycles;
                return;
            }
            cycles -= self.counter_countdown;

            // counter increments (one step of the internal 14-bit counter)
            let old_bit = (self.counter >> self.clock_shift) & 1;
            self.counter = (self.counter + 1) & 0x3FFF;
            let new_bit = (self.counter >> self.clock_shift) & 1;
            // LFSR steps on 0→1 transition of the selected bit
            if old_bit == 0 && new_bit == 1 {
                self.step_lfsr();
            }

            self.counter_countdown = self.base_divisor();
            self.countdown_reloaded = true;
        }
    }

    pub fn output(&self) -> u8 {
        if !self.enabled || !self.dac_on {
            return 0;
        }
        if self.lfsr_sample { self.volume } else { 0 }
    }

    pub fn clock_length(&mut self) {
        if self.len_enable && self.length_counter > 0 {
            self.length_counter -= 1;
            if self.length_counter == 0 {
                self.enabled = false;
            }
        }
    }

    pub fn tick_envelope(&mut self) {
        set_envelope_clock(&mut self.envelope_clock, false, false, 0);
        if self.envelope_clock.locked { return; }
        if self.env_period == 0 { return; }
        set_envelope_clock(&mut self.envelope_clock, false, false, 0);
        if self.env_add {
            self.volume = (self.volume + 1) & 0xF;
        } else {
            self.volume = self.volume.wrapping_sub(1) & 0xF;
        }
    }
}

// ── NRx2 glitch helper (CGB-D/E behavior) ────────────────────────────────

pub(super) fn nrx2_glitch(volume: &mut u8, value: u8, old_value: u8, countdown: &mut u8, lock: &mut EnvelopeClock) {
    if lock.clock {
        *countdown = value & 7;
    }
    let mut should_tick = (value & 7) != 0 && (old_value & 7) == 0 && !lock.locked;
    let should_invert = (value & 8) ^ (old_value & 8) != 0;

    if (value & 0xF) == 8 && (old_value & 0xF) == 8 && !lock.locked {
        should_tick = true;
    }

    if should_invert {
        if value & 8 != 0 {
            if (old_value & 7) == 0 && !lock.locked {
                *volume ^= 0xF;
            } else {
                *volume = (0xE_u8.wrapping_sub(*volume)) & 0xF;
            }
            should_tick = false;
        } else {
            *volume = (0x10_u8.wrapping_sub(*volume)) & 0xF;
        }
    }
    if should_tick {
        if value & 8 != 0 {
            *volume = (*volume + 1) & 0xF;
        } else {
            *volume = volume.wrapping_sub(1) & 0xF;
        }
    } else if (value & 7) == 0 && lock.clock {
        set_envelope_clock(lock, false, false, 0);
    }
}
