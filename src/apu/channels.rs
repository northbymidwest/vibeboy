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

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct Sweep {
    pub period: u8,
    pub negate: bool,
    pub shift: u8,
    pub enabled: bool,
    pub timer: u8,
    pub shadow: u16,
    pub neg_used: bool,
}

impl Sweep {
    pub fn new() -> Self {
        Sweep { period: 0, negate: false, shift: 0, enabled: false, timer: 0, shadow: 0, neg_used: false }
    }

    /// Calculate new frequency; returns None if overflow (channel should disable).
    pub fn calc(&mut self) -> Option<u16> {
        let delta = self.shadow >> self.shift;
        let new_freq = if self.negate {
            self.neg_used = true;
            self.shadow.wrapping_sub(delta)
        } else {
            self.shadow + delta
        };
        if new_freq > 2047 {
            None
        } else {
            Some(new_freq)
        }
    }

    pub fn clock(&mut self, ch: &mut SquareCh) {
        if self.timer > 0 {
            self.timer -= 1;
        }
        if self.timer > 0 { return; }
        self.timer = if self.period == 0 { 8 } else { self.period };
        if !self.enabled { return; }
        if self.period == 0 { return; }
        if let Some(new_freq) = self.calc() {
            if self.shift != 0 {
                self.shadow = new_freq;
                ch.freq = new_freq;
                ch.update_sample();
            }
            // Overflow check again after update
            if self.calc().is_none() {
                ch.enabled = false;
            }
        } else {
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
