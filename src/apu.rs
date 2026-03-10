/// Game Boy Color APU — 4-channel audio synthesis.
///
/// Channels:
///   CH1 — Square wave with frequency sweep (NR10-NR14, 0xFF10-0xFF14)
///   CH2 — Square wave                      (NR21-NR24, 0xFF16-0xFF19)
///   CH3 — Wave output (custom waveform)    (NR30-NR34, 0xFF1A-0xFF1E)
///   CH4 — Noise (LFSR)                     (NR41-NR44, 0xFF20-0xFF23)
///
/// Frame sequencer is clocked by falling edge of DIV bit 12 (normal speed)
/// or bit 13 (double speed). The div_divider counter maps to FS steps:
///   div_divider & 1 == 1 → length counters
///   div_divider & 3 == 3 → CH1 sweep
///   div_divider & 7 == 7 → volume envelopes

use std::sync::Arc;

const DUTY_TABLE: [[u8; 8]; 4] = [
    [0, 0, 0, 0, 0, 0, 0, 1], // 12.5%
    [1, 0, 0, 0, 0, 0, 0, 1], // 25%
    [1, 0, 0, 0, 0, 1, 1, 1], // 50%
    [0, 1, 1, 1, 1, 1, 1, 0], // 75%
];

// Noise divisor table (for NR43 bits 2-0)
const NOISE_DIVISORS: [u32; 8] = [8, 16, 32, 48, 64, 80, 96, 112];

// ── BLIP buffer (band-limited synthesis) ──────────────────────────────────

const BLIP_WIDTH: usize = 64;
const BLIP_PHASES: usize = 256;
const BLIP_BUF_SIZE: usize = BLIP_WIDTH * 2; // 128
const BLIP_ONE: i32 = 0x10000; // 65536

#[derive(Clone)]
struct BlipBuf {
    steps: Arc<[[i32; BLIP_WIDTH]; BLIP_PHASES]>,
    buf_l: [i32; BLIP_BUF_SIZE],
    buf_r: [i32; BLIP_BUF_SIZE],
    pos: usize,
    out_l: i32,
    out_r: i32,
}

impl BlipBuf {
    fn new() -> Self {
        // Compute Blackman-windowed sinc filter
        let n = BLIP_WIDTH * BLIP_PHASES; // 16384 points
        let lowpass = 15.0_f64 / 16.0;
        let mut master = vec![0.0_f64; n];

        for i in 0..n {
            // Center the sinc at n/2
            let x = (i as f64 - n as f64 / 2.0) * std::f64::consts::PI * 2.0 * lowpass / BLIP_PHASES as f64;
            let sinc = if x.abs() < 1e-12 { 1.0 } else { x.sin() / x };

            // Blackman window
            let theta = 2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64;
            let a0 = 7938.0 / 18608.0;
            let a1 = 9240.0 / 18608.0;
            let a2 = 1430.0 / 18608.0;
            let window = a0 - a1 * theta.cos() + a2 * (2.0 * theta).cos();

            master[i] = sinc * window;
        }

        // Normalize so each phase's taps sum to ~1.0 (total master sums to BLIP_PHASES)
        let sum: f64 = master.iter().sum();
        let inv = BLIP_PHASES as f64 / sum;
        for v in &mut master {
            *v *= inv;
        }

        // Extract phases: phase p, tap t → master[t * BLIP_PHASES + p]
        let mut steps = Box::new([[0i32; BLIP_WIDTH]; BLIP_PHASES]);
        for p in 0..BLIP_PHASES {
            for t in 0..BLIP_WIDTH {
                steps[p][t] = (master[t * BLIP_PHASES + p] * BLIP_ONE as f64) as i32;
            }
        }

        BlipBuf {
            steps: Arc::from(steps),
            buf_l: [0; BLIP_BUF_SIZE],
            buf_r: [0; BLIP_BUF_SIZE],
            pos: 0,
            out_l: 0,
            out_r: 0,
        }
    }

    /// Record an amplitude transition (delta) at a fractional output position.
    fn update(&mut self, delta_l: i32, delta_r: i32, phase: usize) {
        let phase_idx = phase & (BLIP_PHASES - 1);
        let delay = phase / BLIP_PHASES;
        let coeffs = &self.steps[phase_idx];
        for i in 0..BLIP_WIDTH {
            let offset = (self.pos + i + delay) & (BLIP_BUF_SIZE - 1);
            self.buf_l[offset] += delta_l * coeffs[i];
            self.buf_r[offset] += delta_r * coeffs[i];
        }
    }

    /// Read one output sample from the circular buffer.
    fn read(&mut self) -> (f32, f32) {
        self.out_l += self.buf_l[self.pos];
        self.buf_l[self.pos] = 0;
        self.out_r += self.buf_r[self.pos];
        self.buf_r[self.pos] = 0;
        self.pos = (self.pos + 1) & (BLIP_BUF_SIZE - 1);
        (
            self.out_l as f32 / BLIP_ONE as f32,
            self.out_r as f32 / BLIP_ONE as f32,
        )
    }
}

// ── Envelope clock tracking ───────────────────────────────────────────────

#[derive(Clone, Default)]
struct EnvelopeClock {
    clock: bool,
    locked: bool,
    should_lock: bool,
}

/// Update envelope clock state machine.
fn set_envelope_clock(ec: &mut EnvelopeClock, value: bool, direction: bool, volume: u8) {
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

#[derive(Clone)]
struct SquareCh {
    // NRx1
    duty: u8,
    // NRx2
    env_init_vol: u8,
    env_add: bool,
    env_period: u8,
    // NRx3/NRx4
    freq: u16,         // 11-bit
    len_enable: bool,

    // Internal
    enabled: bool,
    dac_on: bool,
    freq_timer: u32,   // T-cycle countdown; reload = (freq ^ 0x7FF) * 4 + 2
    duty_pos: u8,
    length_counter: u16,
    volume: u8,
    env_timer: u8,
    volume_countdown: u8, // separate from env_timer, decremented when envelope_clock is not set
    sample_suppressed: bool, // first sample after trigger is suppressed (outputs 0)
    just_reloaded: bool,     // true when last tick consumed all cycles exactly
    did_tick: bool,          // true when duty_pos advanced, cleared on trigger
    delay: u32,              // startup delay metadata (not consumed from cycles)

    // Envelope clock tracking
    envelope_clock: EnvelopeClock,

    // Raw NRx2 value for glitch detection
    nrx2_raw: u8,
}

impl SquareCh {
    fn new() -> Self {
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
            sample_suppressed: false,
            just_reloaded: false,
            did_tick: false,
            delay: 0,
            envelope_clock: EnvelopeClock::default(),
            nrx2_raw: 0,
        }
    }

    fn reload_period(&self) -> u32 {
        (2048 - self.freq as u32) * 4
    }

    fn tick_freq(&mut self, mut cycles: u32) {
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

    fn output(&self) -> u8 {
        if !self.enabled || !self.dac_on || self.sample_suppressed {
            return 0;
        }
        DUTY_TABLE[self.duty as usize][self.duty_pos as usize] * self.volume
    }

    fn clock_length(&mut self) {
        if self.len_enable && self.length_counter > 0 {
            self.length_counter -= 1;
            if self.length_counter == 0 {
                self.enabled = false;
            }
        }
    }

    /// Tick envelope: adjust volume unconditionally (after checking locked and env_period).
    /// Called when envelope_clock is set during div_event envelope step.
    /// Does NOT touch volume_countdown — that's handled separately in div_event.
    fn tick_envelope(&mut self) {
        set_envelope_clock(&mut self.envelope_clock, false, false, 0);
        if self.envelope_clock.locked { return; }
        if self.env_period == 0 { return; }
        // Second set_envelope_clock call (needed for PCM mask timing in double speed)
        set_envelope_clock(&mut self.envelope_clock, false, false, 0);
        if self.env_add {
            self.volume = (self.volume + 1) & 0xF;
        } else {
            self.volume = self.volume.wrapping_sub(1) & 0xF;
        }
    }
}

// ── CH1 sweep ──────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Sweep {
    period: u8,
    negate: bool,
    shift: u8,
    enabled: bool,
    timer: u8,
    shadow: u16, // shadow copy of CH1 freq
    neg_used: bool,
}

impl Sweep {
    fn new() -> Self {
        Sweep { period: 0, negate: false, shift: 0, enabled: false, timer: 0, shadow: 0, neg_used: false }
    }

    /// Calculate new frequency; returns None if overflow (channel should disable).
    fn calc(&mut self) -> Option<u16> {
        let delta = self.shadow >> self.shift;
        let new_freq = if self.negate {
            self.neg_used = true;
            self.shadow.wrapping_sub(delta)
        } else {
            self.shadow + delta
        };
        if new_freq > 2047 { None } else { Some(new_freq) }
    }

    fn clock(&mut self, ch: &mut SquareCh) {
        if self.timer > 0 {
            self.timer -= 1;
        }
        if self.timer == 0 {
            self.timer = if self.period == 0 { 8 } else { self.period };
            if self.enabled && self.period != 0 {
                match self.calc() {
                    None => ch.enabled = false,
                    Some(f) => {
                        // Only write back frequency if shift != 0
                        if self.shift != 0 {
                            self.shadow = f;
                            ch.freq = f;
                            // overflow check with new shadow
                            if self.calc().is_none() {
                                ch.enabled = false;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── Wave channel (CH3) ─────────────────────────────────────────────────────

#[derive(Clone)]
struct WaveCh {
    dac_on: bool,
    vol_code: u8,  // 0=mute, 1=100%, 2=50%, 3=25%
    freq: u16,
    len_enable: bool,

    enabled: bool,
    freq_timer: u32,  // reload = (2048 - freq) * 2
    sample_pos: u8,   // 0-31
    length_counter: u16,
    last_nibble: u8,
    wave_form_just_read: bool,
    pub wave_ram: [u8; 16],
}

impl WaveCh {
    fn new() -> Self {
        WaveCh {
            dac_on: false,
            vol_code: 0,
            freq: 0,
            len_enable: false,
            enabled: false,
            freq_timer: 8,
            sample_pos: 0,
            length_counter: 256,
            last_nibble: 0,
            wave_form_just_read: false,
            wave_ram: [0u8; 16],
        }
    }

    fn reload_period(&self) -> u32 {
        (2048 - self.freq as u32) * 2
    }

    fn tick_freq(&mut self, mut cycles: u32) {
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

    fn output(&self) -> u8 {
        if !self.enabled || !self.dac_on {
            return 0;
        }
        match self.vol_code {
            0 => 0,
            1 => self.last_nibble,
            2 => self.last_nibble >> 1,
            3 => self.last_nibble >> 2,
            _ => 0,
        }
    }


    fn clock_length(&mut self) {
        if self.len_enable && self.length_counter > 0 {
            self.length_counter -= 1;
            if self.length_counter == 0 {
                self.enabled = false;
            }
        }
    }
}

// ── Noise channel (CH4) ────────────────────────────────────────────────────

#[derive(Clone)]
struct NoiseCh {
    env_init_vol: u8,
    env_add: bool,
    env_period: u8,
    clock_shift: u8,
    lfsr_narrow: bool, // true = 7-bit LFSR
    divisor_code: u8,
    len_enable: bool,

    enabled: bool,
    dac_on: bool,
    freq_timer: u32,
    lfsr: u16,
    length_counter: u16,
    volume: u8,
    env_timer: u8,
    volume_countdown: u8,

    // Envelope clock tracking
    envelope_clock: EnvelopeClock,

    // Raw NR42 value for glitch detection
    nrx2_raw: u8,
}

impl NoiseCh {
    fn new() -> Self {
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
            freq_timer: 8,
            lfsr: 0x7FFF,
            length_counter: 64,
            volume: 0,
            env_timer: 0,
            volume_countdown: 0,
            envelope_clock: EnvelopeClock::default(),
            nrx2_raw: 0,
        }
    }

    fn reload_period(&self) -> u32 {
        let div = NOISE_DIVISORS[self.divisor_code as usize];
        if self.clock_shift < 14 {
            div << self.clock_shift
        } else {
            div << 13 // cap to avoid overflow
        }
    }

    fn tick_freq(&mut self, mut cycles: u32) {
        if !self.enabled {
            return;
        }
        loop {
            if self.freq_timer > cycles {
                self.freq_timer -= cycles;
                return;
            }
            cycles -= self.freq_timer;
            let xor = (self.lfsr & 1) ^ ((self.lfsr >> 1) & 1);
            self.lfsr = (self.lfsr >> 1) | (xor << 14);
            if self.lfsr_narrow {
                self.lfsr = (self.lfsr & !(1 << 6)) | (xor << 6);
            }
            self.freq_timer = self.reload_period();
            if self.freq_timer == 0 {
                self.freq_timer = 8;
            }
        }
    }

    fn output(&self) -> u8 {
        if !self.enabled || !self.dac_on {
            return 0;
        }
        // Output is high when LFSR bit0 is 0
        if self.lfsr & 1 == 0 { self.volume } else { 0 }
    }


    fn clock_length(&mut self) {
        if self.len_enable && self.length_counter > 0 {
            self.length_counter -= 1;
            if self.length_counter == 0 {
                self.enabled = false;
            }
        }
    }

    /// Tick envelope: adjust volume unconditionally (after checking locked and env_period).
    fn tick_envelope(&mut self) {
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

fn nrx2_glitch(volume: &mut u8, value: u8, old_value: u8, countdown: &mut u8, lock: &mut EnvelopeClock) {
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

// ── APU top level ──────────────────────────────────────────────────────────

const SAMPLE_RATE: u32 = 96_000;

#[derive(Clone)]
pub struct Apu {
    ch1: SquareCh,
    sweep: Sweep,
    ch2: SquareCh,
    ch3: WaveCh,
    ch4: NoiseCh,

    nr50: u8, // master volume
    nr51: u8, // panning
    power: bool,

    // DIV-coupled frame sequencer
    div_divider: u8,
    skip_div_event: u8, // 0=inactive, 1=skip next, 2=skipped (resume normal)

    // Current DIV counter (updated from bus each tick)
    div_counter: u16,
    double_speed: bool,

    // Sub-2MHz phase tracker (toggles each T-cycle)
    lf_div: bool,

    // PCM masking (for envelope glitch behavior)
    pcm_mask: [u8; 2],

    // Sample timing
    sample_accum: u64,

    // Output buffer (interleaved L/R f32 pairs)
    pub sample_buf: Vec<f32>,

    // BLIP buffer for band-limited resampling
    blip: BlipBuf,
    prev_left: i32,
    prev_right: i32,

    // High-pass filter state (models the coupling capacitor on real hardware)
    hpf_left: f32,
    hpf_right: f32,
    hpf_prev_in_l: f32,
    hpf_prev_in_r: f32,

    // Model-dependent clock rate for sample timing
    sample_accum_tick: u64,
    sample_accum_thresh: u64,

    // CGB mode flag (affects power-off behavior)
    is_cgb: bool,
}

impl Apu {
    pub fn new(cpu_clock_rate: u32, is_cgb: bool, is_sgb: bool) -> Self {
        let mut apu = Apu {
            ch1: SquareCh::new(),
            sweep: Sweep::new(),
            ch2: SquareCh::new(),
            ch3: WaveCh::new(),
            ch4: NoiseCh::new(),
            nr50: 0x77,
            nr51: 0xF3,
            power: true,
            div_divider: 0,
            skip_div_event: 0,
            div_counter: 0,
            double_speed: false,
            lf_div: true,
            pcm_mask: [0xFF, 0xFF],
            sample_accum: 0,
            sample_buf: Vec::with_capacity(1024),
            blip: BlipBuf::new(),
            prev_left: 0,
            prev_right: 0,
            hpf_left: 0.0,
            hpf_right: 0.0,
            hpf_prev_in_l: 0.0,
            hpf_prev_in_r: 0.0,
            sample_accum_tick: SAMPLE_RATE as u64,
            sample_accum_thresh: cpu_clock_rate as u64,
            is_cgb,
        };
        // Post-boot CH1 state: registers match what boot ROM left behind,
        // but the envelope has decayed volume to 0 during the boot animation.
        // All boot ROMs (DMG/SGB) write the same NR11/NR12 values, but
        // SGB never triggers CH1, so DAC and channel stay disabled.
        apu.ch1.duty = 2;
        apu.ch1.env_init_vol = 0xF;
        apu.ch1.env_period = 3;
        apu.ch1.nrx2_raw = 0xF3;
        apu.ch1.volume = 0;
        if !is_sgb {
            apu.ch1.dac_on = true;
            apu.ch1.enabled = true;
        }
        apu
    }

    /// Hardware reset state for use with boot ROM execution.
    pub fn reset(cpu_clock_rate: u32, is_cgb: bool) -> Self {
        Apu {
            ch1: SquareCh::new(),
            sweep: Sweep::new(),
            ch2: SquareCh::new(),
            ch3: WaveCh::new(),
            ch4: NoiseCh::new(),
            nr50: 0x00,
            nr51: 0x00,
            power: false,
            div_divider: 0,
            skip_div_event: 0,
            div_counter: 0,
            double_speed: false,
            lf_div: true,
            pcm_mask: [0xFF, 0xFF],
            sample_accum: 0,
            sample_buf: Vec::with_capacity(1024),
            blip: BlipBuf::new(),
            prev_left: 0,
            prev_right: 0,
            hpf_left: 0.0,
            hpf_right: 0.0,
            hpf_prev_in_l: 0.0,
            hpf_prev_in_r: 0.0,
            sample_accum_tick: SAMPLE_RATE as u64,
            sample_accum_thresh: cpu_clock_rate as u64,
            is_cgb,
        }
    }

    /// Update the DIV counter from the bus (called each tick).
    pub fn set_div_counter(&mut self, counter: u16) {
        self.div_counter = counter;
    }

    /// Update double-speed mode from the bus.
    pub fn set_double_speed(&mut self, ds: bool) {
        self.double_speed = ds;
    }

    fn apu_bit(&self) -> u16 {
        if self.double_speed { 0x2000 } else { 0x1000 }
    }

    /// Get effective lf_div value for write-time calculations (trigger delays).
    /// Since lf_div now toggles during power-off M-cycles (matching hardware),
    /// the phase relationship between our write-then-tick model and the
    /// advance-then-write reference model is maintained: at write time, both
    /// models see the same lf_div value after power on initialization.
    fn write_lf(&self) -> u32 {
        if self.lf_div { 1 } else { 0 }
    }

    // ── Register read ──────────────────────────────────────────────────────

    pub fn read(&self, addr: u16) -> u8 {
        match addr {
            // CH1
            0xFF10 => 0x80 | (self.sweep.period << 4) | (if self.sweep.negate { 0x08 } else { 0 }) | self.sweep.shift,
            0xFF11 => 0x3F | (self.ch1.duty << 6),
            0xFF12 => self.ch1.nrx2_raw,
            0xFF13 => 0xFF,
            0xFF14 => 0xBF | (if self.ch1.len_enable { 0x40 } else { 0 }),
            0xFF15 => 0xFF,
            // CH2
            0xFF16 => 0x3F | (self.ch2.duty << 6),
            0xFF17 => self.ch2.nrx2_raw,
            0xFF18 => 0xFF,
            0xFF19 => 0xBF | (if self.ch2.len_enable { 0x40 } else { 0 }),
            // CH3
            0xFF1A => if self.ch3.dac_on { 0xFF } else { 0x7F },
            0xFF1B => 0xFF,
            0xFF1C => 0x9F | (self.ch3.vol_code << 5),
            0xFF1D => 0xFF,
            0xFF1E => 0xBF | (if self.ch3.len_enable { 0x40 } else { 0 }),
            0xFF1F => 0xFF,
            // CH4
            0xFF20 => 0xFF,
            0xFF21 => self.ch4.nrx2_raw,
            0xFF22 => (self.ch4.clock_shift << 4) | (if self.ch4.lfsr_narrow { 0x08 } else { 0 }) | self.ch4.divisor_code,
            0xFF23 => 0xBF | (if self.ch4.len_enable { 0x40 } else { 0 }),
            // Global
            0xFF24 => self.nr50,
            0xFF25 => self.nr51,
            0xFF26 => {
                let mut v = if self.power { 0x80 } else { 0x00 } | 0x70;
                if self.ch1.enabled { v |= 0x01; }
                if self.ch2.enabled { v |= 0x02; }
                if self.ch3.enabled { v |= 0x04; }
                if self.ch4.enabled { v |= 0x08; }
                v
            }
            // Wave RAM: when CH3 is active, redirect to current sample position
            0xFF30..=0xFF3F => {
                if self.ch3.enabled {
                    if !self.is_cgb && !self.ch3.wave_form_just_read {
                        return 0xFF;
                    }
                    self.ch3.wave_ram[(self.ch3.sample_pos >> 1) as usize]
                } else {
                    self.ch3.wave_ram[(addr - 0xFF30) as usize]
                }
            }
            _ => 0xFF,
        }
    }

    // ── Register write ─────────────────────────────────────────────────────

    pub fn write(&mut self, addr: u16, val: u8) {
        // Wave RAM: when CH3 is active, redirect write to current sample position
        if (0xFF30..=0xFF3F).contains(&addr) {
            if self.ch3.enabled {
                if !self.is_cgb && !self.ch3.wave_form_just_read {
                    return;
                }
                self.ch3.wave_ram[(self.ch3.sample_pos >> 1) as usize] = val;
            } else {
                self.ch3.wave_ram[(addr - 0xFF30) as usize] = val;
            }
            return;
        }

        // NR52: power control — always writable
        if addr == 0xFF26 {
            let was_on = self.power;
            self.power = val & 0x80 != 0;
            if was_on && !self.power {
                self.power_off();
            } else if !was_on && self.power {
                // Power on: reset lf_div to 1 (hardware re-initializes on enable)
                self.lf_div = true;
                // Check if DIV APU bit is high → skip first div event
                let apu_bit = self.apu_bit();
                if self.div_counter & apu_bit != 0 {
                    self.skip_div_event = 1;
                    self.div_divider = 1;
                }
            }
            return;
        }

        // On DMG, length counters are writable even when powered off.
        // On CGB, all writes (except NR52 and wave RAM) are ignored when off.
        if !self.power {
            if !self.is_cgb {
                match addr {
                    0xFF11 => self.ch1.length_counter = 64 - (val & 0x3F) as u16,
                    0xFF16 => self.ch2.length_counter = 64 - (val & 0x3F) as u16,
                    0xFF1B => self.ch3.length_counter = 256 - val as u16,
                    0xFF20 => self.ch4.length_counter = 64 - (val & 0x3F) as u16,
                    _ => {}
                }
            }
            return;
        }

        match addr {
            // ── CH1 ────────────────────────────────────────────────────────
            0xFF10 => {
                let old_neg = self.sweep.negate;
                self.sweep.period = (val >> 4) & 0x07;
                self.sweep.negate = val & 0x08 != 0;
                self.sweep.shift  = val & 0x07;
                // If negate was used and negate bit is cleared, disable CH1
                if self.sweep.neg_used && old_neg && !self.sweep.negate {
                    self.ch1.enabled = false;
                }
            }
            0xFF11 => {
                self.ch1.duty = (val >> 6) & 0x03;
                self.ch1.length_counter = 64 - (val & 0x3F) as u16;
            }
            0xFF12 => {
                let old_value = self.ch1.nrx2_raw;
                self.ch1.nrx2_raw = val;
                self.ch1.env_init_vol = val >> 4;
                self.ch1.env_add     = val & 0x08 != 0;
                self.ch1.env_period  = val & 0x07;
                self.ch1.dac_on = val & 0xF8 != 0;
                if !self.ch1.dac_on {
                    self.ch1.enabled = false;
                } else if self.ch1.enabled {
                    nrx2_glitch(&mut self.ch1.volume, val, old_value, &mut self.ch1.volume_countdown, &mut self.ch1.envelope_clock);
                    // PCM mask: CH1 is low nibble of pcm_mask[0]
                    self.pcm_mask[0] &= self.ch1.volume | 0xF0;
                }
            }
            0xFF13 => {
                self.ch1.freq = (self.ch1.freq & 0x700) | val as u16;
                if self.ch1.just_reloaded {
                    self.ch1.freq_timer = self.ch1.reload_period();
                }
            }
            0xFF14 => {
                let old_freq = self.ch1.freq;
                // did_tick frequency change edge case (must check BEFORE freq update)
                if val & 0x80 == 0 && self.ch1.enabled {
                    let old_hi = (old_freq >> 8) & 7;
                    let new_hi = ((val & 0x07) as u16) as u32;
                    if old_hi == 7 && new_hi != 7 && self.ch1.did_tick {
                        if (self.ch1.freq_timer.wrapping_sub(2)) / 4 == (old_freq ^ 0x7FF) as u32 {
                            self.ch1.duty_pos = self.ch1.duty_pos.wrapping_sub(1) & 7;
                            self.ch1.sample_suppressed = false;
                        }
                    }
                }
                self.ch1.freq = (self.ch1.freq & 0x0FF) | (((val & 0x07) as u16) << 8);
                if self.ch1.just_reloaded {
                    self.ch1.freq_timer = self.ch1.reload_period();
                }
                let was_enabled = self.ch1.len_enable;
                self.ch1.len_enable = val & 0x40 != 0;
                if !was_enabled && val & 0x40 != 0 && self.div_divider & 1 != 0 {
                    self.ch1.clock_length();
                }
                if val & 0x80 != 0 {
                    self.trigger_ch1(val);
                }
            }
            // ── CH2 ────────────────────────────────────────────────────────
            0xFF16 => {
                self.ch2.duty = (val >> 6) & 0x03;
                self.ch2.length_counter = 64 - (val & 0x3F) as u16;
            }
            0xFF17 => {
                let old_value = self.ch2.nrx2_raw;
                self.ch2.nrx2_raw = val;
                self.ch2.env_init_vol = val >> 4;
                self.ch2.env_add     = val & 0x08 != 0;
                self.ch2.env_period  = val & 0x07;
                self.ch2.dac_on = val & 0xF8 != 0;
                if !self.ch2.dac_on {
                    self.ch2.enabled = false;
                } else if self.ch2.enabled {
                    nrx2_glitch(&mut self.ch2.volume, val, old_value, &mut self.ch2.volume_countdown, &mut self.ch2.envelope_clock);
                    // PCM mask: CH2 is high nibble of pcm_mask[0]
                    self.pcm_mask[0] &= (self.ch2.volume << 4) | 0x0F;
                }
            }
            0xFF18 => {
                self.ch2.freq = (self.ch2.freq & 0x700) | val as u16;
                if self.ch2.just_reloaded {
                    self.ch2.freq_timer = self.ch2.reload_period();
                }
            }
            0xFF19 => {
                let old_freq = self.ch2.freq;
                // did_tick frequency change edge case (must check BEFORE freq update)
                if val & 0x80 == 0 && self.ch2.enabled {
                    let old_hi = (old_freq >> 8) & 7;
                    let new_hi = ((val & 0x07) as u16) as u32;
                    if old_hi == 7 && new_hi != 7 && self.ch2.did_tick {
                        if (self.ch2.freq_timer.wrapping_sub(2)) / 4 == (old_freq ^ 0x7FF) as u32 {
                            self.ch2.duty_pos = self.ch2.duty_pos.wrapping_sub(1) & 7;
                            self.ch2.sample_suppressed = false;
                        }
                    }
                }
                self.ch2.freq = (self.ch2.freq & 0x0FF) | (((val & 0x07) as u16) << 8);
                if self.ch2.just_reloaded {
                    self.ch2.freq_timer = self.ch2.reload_period();
                }
                let was_enabled = self.ch2.len_enable;
                self.ch2.len_enable = val & 0x40 != 0;
                if !was_enabled && val & 0x40 != 0 && self.div_divider & 1 != 0 {
                    self.ch2.clock_length();
                }
                if val & 0x80 != 0 {
                    self.trigger_ch2(val);
                }
            }
            // ── CH3 ────────────────────────────────────────────────────────
            0xFF1A => {
                self.ch3.dac_on = val & 0x80 != 0;
                if !self.ch3.dac_on { self.ch3.enabled = false; }
            }
            0xFF1B => self.ch3.length_counter = 256 - val as u16,
            0xFF1C => self.ch3.vol_code = (val >> 5) & 0x03,
            0xFF1D => self.ch3.freq = (self.ch3.freq & 0x700) | val as u16,
            0xFF1E => {
                self.ch3.freq = (self.ch3.freq & 0x0FF) | (((val & 0x07) as u16) << 8);
                let was_enabled = self.ch3.len_enable;
                self.ch3.len_enable = val & 0x40 != 0;
                if !was_enabled && val & 0x40 != 0 && self.div_divider & 1 != 0 {
                    self.ch3.clock_length();
                }
                if val & 0x80 != 0 {
                    self.trigger_ch3();
                }
            }
            // ── CH4 ────────────────────────────────────────────────────────
            0xFF20 => self.ch4.length_counter = 64 - (val & 0x3F) as u16,
            0xFF21 => {
                let old_value = self.ch4.nrx2_raw;
                self.ch4.nrx2_raw = val;
                self.ch4.env_init_vol = val >> 4;
                self.ch4.env_add     = val & 0x08 != 0;
                self.ch4.env_period  = val & 0x07;
                self.ch4.dac_on = val & 0xF8 != 0;
                if !self.ch4.dac_on {
                    self.ch4.enabled = false;
                } else if self.ch4.enabled {
                    nrx2_glitch(&mut self.ch4.volume, val, old_value, &mut self.ch4.volume_countdown, &mut self.ch4.envelope_clock);
                    // PCM mask: CH4 is high nibble of pcm_mask[1]
                    self.pcm_mask[1] &= (self.ch4.volume << 4) | 0x0F;
                }
            }
            0xFF22 => {
                self.ch4.clock_shift  = val >> 4;
                self.ch4.lfsr_narrow  = val & 0x08 != 0;
                self.ch4.divisor_code = val & 0x07;
            }
            0xFF23 => {
                let was_enabled = self.ch4.len_enable;
                self.ch4.len_enable = val & 0x40 != 0;
                if !was_enabled && val & 0x40 != 0 && self.div_divider & 1 != 0 {
                    self.ch4.clock_length();
                }
                if val & 0x80 != 0 {
                    self.trigger_ch4();
                }
            }
            // ── Global ─────────────────────────────────────────────────────
            0xFF24 => {
                self.nr50 = val;
                self.record_mix_delta();
            }
            0xFF25 => {
                self.nr51 = val;
                self.record_mix_delta();
            }
            _ => {}
        }
    }

    // ── Channel triggers ───────────────────────────────────────────────────

    fn trigger_ch1(&mut self, val: u8) {
        let was_active = self.ch1.enabled;
        self.ch1.enabled = true;
        if self.ch1.length_counter == 0 {
            self.ch1.length_counter = 64;
            if self.ch1.len_enable && self.div_divider & 1 == 1 {
                self.ch1.length_counter -= 1;
            }
        }
        self.ch1.volume    = self.ch1.env_init_vol;
        self.ch1.env_timer = self.ch1.env_period;
        self.ch1.volume_countdown = self.ch1.env_period;
        self.ch1.envelope_clock.locked = false;
        self.ch1.envelope_clock.clock = false;
        // Note: should_lock is NOT reset on trigger (hardware behavior)

        let lf = self.write_lf();
        let base = (self.ch1.freq ^ 0x7FF) as u32 * 4;
        let mut force_unsuppressed = false;
        if !was_active {
            // Retrigger duty advance (not active → active)
            if val & 4 == 0
                && ((self.ch1.freq_timer.wrapping_sub(self.ch1.delay).wrapping_sub(2)) / 4) & 0x400 == 0
            {
                self.ch1.duty_pos = (self.ch1.duty_pos + 1) & 7;
                force_unsuppressed = true;
            }
            self.ch1.delay = (7 - lf) * 2;
            self.ch1.freq_timer = base + self.ch1.delay;
            self.ch1.sample_suppressed = !force_unsuppressed;
        } else {
            // Retrigger duty advance (already active)
            let old_freq = self.ch1.freq;
            if !self.ch1.just_reloaded
                && val & 4 == 0
                && ((self.ch1.freq_timer.wrapping_sub(self.ch1.delay).wrapping_sub(4)) / 4) & 0x400 == 0
            {
                self.ch1.duty_pos = (self.ch1.duty_pos + 1) & 7;
                self.ch1.sample_suppressed = false;
            }
            let mut extra_delay = 0u32;
            if self.ch1.freq == 0x7FF && old_freq != 0x7FF && self.ch1.sample_suppressed {
                extra_delay = 4;
            }
            self.ch1.delay = (5 - lf + extra_delay) * 2;
            self.ch1.freq_timer = base + self.ch1.delay;
        }
        self.ch1.did_tick = false;

        self.sweep.shadow   = self.ch1.freq;
        self.sweep.neg_used = false;
        self.sweep.timer    = if self.sweep.period == 0 { 8 } else { self.sweep.period };
        self.sweep.enabled  = self.sweep.period != 0 || self.sweep.shift != 0;
        if self.sweep.shift != 0 {
            if self.sweep.calc().is_none() {
                self.ch1.enabled = false;
            }
        }
        if !self.ch1.dac_on {
            self.ch1.enabled = false;
        }
    }

    fn trigger_ch2(&mut self, val: u8) {
        let was_active = self.ch2.enabled;
        self.ch2.enabled = true;
        if self.ch2.length_counter == 0 {
            self.ch2.length_counter = 64;
            if self.ch2.len_enable && self.div_divider & 1 == 1 {
                self.ch2.length_counter -= 1;
            }
        }
        self.ch2.volume    = self.ch2.env_init_vol;
        self.ch2.env_timer = self.ch2.env_period;
        self.ch2.volume_countdown = self.ch2.env_period;
        self.ch2.envelope_clock.locked = false;
        self.ch2.envelope_clock.clock = false;

        let lf = self.write_lf();
        let base = (self.ch2.freq ^ 0x7FF) as u32 * 4;
        let mut force_unsuppressed = false;
        if !was_active {
            // Retrigger duty advance (not active → active)
            if val & 4 == 0
                && ((self.ch2.freq_timer.wrapping_sub(self.ch2.delay).wrapping_sub(2)) / 4) & 0x400 == 0
            {
                self.ch2.duty_pos = (self.ch2.duty_pos + 1) & 7;
                force_unsuppressed = true;
            }
            self.ch2.delay = (7 - lf) * 2;
            self.ch2.freq_timer = base + self.ch2.delay;
            self.ch2.sample_suppressed = !force_unsuppressed;
        } else {
            // Retrigger duty advance (already active)
            let old_freq = self.ch2.freq;
            if !self.ch2.just_reloaded
                && val & 4 == 0
                && ((self.ch2.freq_timer.wrapping_sub(self.ch2.delay).wrapping_sub(4)) / 4) & 0x400 == 0
            {
                self.ch2.duty_pos = (self.ch2.duty_pos + 1) & 7;
                self.ch2.sample_suppressed = false;
            }
            let mut extra_delay = 0u32;
            if self.ch2.freq == 0x7FF && old_freq != 0x7FF && self.ch2.sample_suppressed {
                extra_delay = 4;
            }
            self.ch2.delay = (5 - lf + extra_delay) * 2;
            self.ch2.freq_timer = base + self.ch2.delay;
        }
        self.ch2.did_tick = false;

        if !self.ch2.dac_on {
            self.ch2.enabled = false;
        }
    }

    fn trigger_ch3(&mut self) {
        let was_enabled = self.ch3.enabled;
        self.ch3.enabled = true;
        if self.ch3.length_counter == 0 {
            self.ch3.length_counter = 256;
            if self.ch3.len_enable && self.div_divider & 1 == 1 {
                self.ch3.length_counter -= 1;
            }
        }
        // DMG wave RAM corruption on retrigger when channel is active
        // and sample_countdown == 0 (our freq_timer == 1)
        if !self.is_cgb && was_enabled && self.ch3.freq_timer <= 2 {
            let offset = (((self.ch3.sample_pos + 1) >> 1) & 0xF) as usize;
            if offset < 4 {
                self.ch3.wave_ram[0] = self.ch3.wave_ram[offset];
            } else {
                let base = offset & !3;
                let src = [
                    self.ch3.wave_ram[base],
                    self.ch3.wave_ram[base + 1],
                    self.ch3.wave_ram[base + 2],
                    self.ch3.wave_ram[base + 3],
                ];
                self.ch3.wave_ram[0] = src[0];
                self.ch3.wave_ram[1] = src[1];
                self.ch3.wave_ram[2] = src[2];
                self.ch3.wave_ram[3] = src[3];
            }
        }
        self.ch3.freq_timer  = self.ch3.reload_period() + 6;
        self.ch3.sample_pos  = 0;
        // Note: do NOT reset last_nibble — hardware keeps the last-read sample byte
        if !self.ch3.dac_on {
            self.ch3.enabled = false;
        }
    }

    fn trigger_ch4(&mut self) {
        self.ch4.enabled = true;
        if self.ch4.length_counter == 0 {
            self.ch4.length_counter = 64;
            if self.ch4.len_enable && self.div_divider & 1 == 1 {
                self.ch4.length_counter -= 1;
            }
        }
        self.ch4.volume    = self.ch4.env_init_vol;
        self.ch4.env_timer = self.ch4.env_period;
        self.ch4.volume_countdown = self.ch4.env_period;
        self.ch4.envelope_clock.locked = false;
        self.ch4.envelope_clock.clock = false;
        self.ch4.lfsr      = 0x7FFF;
        self.ch4.freq_timer = self.ch4.reload_period();
        if !self.ch4.dac_on {
            self.ch4.enabled = false;
        }
    }

    // ── Power off ──────────────────────────────────────────────────────────

    fn power_off(&mut self) {
        // Record delta to silence before resetting channels
        let phase = self.current_blip_phase();
        let dl = -self.prev_left;
        let dr = -self.prev_right;
        if dl != 0 || dr != 0 {
            self.blip.update(dl, dr, phase);
        }
        self.prev_left = 0;
        self.prev_right = 0;

        // Hardware preserves wave RAM across power cycles.
        // On DMG, length counters are also preserved; on CGB they are reset.
        let wave_ram = self.ch3.wave_ram;
        let len1 = self.ch1.length_counter;
        let len2 = self.ch2.length_counter;
        let len3 = self.ch3.length_counter;
        let len4 = self.ch4.length_counter;

        self.ch1 = SquareCh::new();
        self.sweep = Sweep::new();
        self.ch2 = SquareCh::new();
        self.ch3 = WaveCh::new();
        self.ch4 = NoiseCh::new();

        self.ch3.wave_ram = wave_ram;
        if !self.is_cgb {
            self.ch1.length_counter = len1;
            self.ch2.length_counter = len2;
            self.ch3.length_counter = len3;
            self.ch4.length_counter = len4;
        }

        self.nr50 = 0;
        self.nr51 = 0;
        self.div_divider = 0;
        self.skip_div_event = 0;
        self.lf_div = false; // zeroed on power off (re-initialized to 1 on power on)
    }

    // ── DIV-coupled frame sequencer ─────────────────────────────────────────

    /// Called by Bus when DIV bit 12 (or 13 in double speed) has a falling edge.
    pub fn div_event(&mut self) {
        if !self.power { return; }

        // Handle skip_div_event state machine
        if self.skip_div_event == 1 {
            self.skip_div_event = 2;
            return;
        }
        if self.skip_div_event == 2 {
            self.skip_div_event = 0;
            // Don't increment div_divider on the event after skip
        } else {
            self.div_divider = self.div_divider.wrapping_add(1);
        }

        // Reset PCM mask each div event
        self.pcm_mask = [0xFF, 0xFF];

        // Length counters: when div_divider & 1 == 1 (every other event)
        if self.div_divider & 1 == 1 {
            self.ch1.clock_length();
            self.ch2.clock_length();
            self.ch3.clock_length();
            self.ch4.clock_length();
        }

        // Sweep: when div_divider & 3 == 3 (every 4th event)
        if self.div_divider & 3 == 3 {
            let ch1 = &mut self.ch1;
            self.sweep.clock(ch1);
        }

        // Envelope volume_countdown: only on envelope steps (div_divider & 7 == 7)
        // Decrement for channels where clock is NOT set
        if self.div_divider & 7 == 7 {
            if !self.ch1.envelope_clock.clock {
                self.ch1.volume_countdown = self.ch1.volume_countdown.wrapping_sub(1) & 7;
            }
            if !self.ch2.envelope_clock.clock {
                self.ch2.volume_countdown = self.ch2.volume_countdown.wrapping_sub(1) & 7;
            }
            if !self.ch4.envelope_clock.clock {
                self.ch4.volume_countdown = self.ch4.volume_countdown.wrapping_sub(1) & 7;
            }
        }

        // Envelope tick: on EVERY div event, tick channels where clock IS set
        if self.ch1.envelope_clock.clock {
            self.ch1.tick_envelope();
        }
        if self.ch2.envelope_clock.clock {
            self.ch2.tick_envelope();
        }
        if self.ch4.envelope_clock.clock {
            self.ch4.tick_envelope();
        }
    }

    /// Called by Bus when DIV bit 12 (or 13 in double speed) has a rising edge.
    pub fn div_secondary_event(&mut self) {
        if !self.power { return; }

        // Reset PCM mask
        self.pcm_mask = [0xFF, 0xFF];

        // On secondary event: for active channels with volume_countdown == 0,
        // reload countdown from env_period and set envelope clock
        if self.ch1.enabled && self.ch1.volume_countdown == 0 {
            self.ch1.volume_countdown = self.ch1.env_period;
            set_envelope_clock(&mut self.ch1.envelope_clock, self.ch1.env_period != 0, self.ch1.env_add, self.ch1.volume);
        }
        if self.ch2.enabled && self.ch2.volume_countdown == 0 {
            self.ch2.volume_countdown = self.ch2.env_period;
            set_envelope_clock(&mut self.ch2.envelope_clock, self.ch2.env_period != 0, self.ch2.env_add, self.ch2.volume);
        }
        if self.ch4.enabled && self.ch4.volume_countdown == 0 {
            self.ch4.volume_countdown = self.ch4.env_period;
            set_envelope_clock(&mut self.ch4.envelope_clock, self.ch4.env_period != 0, self.ch4.env_add, self.ch4.volume);
        }
    }

    // ── Integer mixing (for BLIP delta detection) ─────────────────────────

    /// Compute current mixed L/R as integers.
    /// DAC: digital 0-15 → bipolar -15..+15 (×2-15), DAC off → 0.
    /// After panning sum: range ±60. After NR50 volume (×1-8): range ±480.
    fn mix_integer(&self) -> (i32, i32) {
        let dac = |digital: u8, dac_on: bool| -> i32 {
            if dac_on { digital as i32 * 2 - 15 } else { 0 }
        };

        let o1 = dac(self.ch1.output(), self.ch1.dac_on);
        let o2 = dac(self.ch2.output(), self.ch2.dac_on);
        let o3 = dac(self.ch3.output(), self.ch3.dac_on);
        let o4 = dac(self.ch4.output(), self.ch4.dac_on);

        let mut left = 0i32;
        let mut right = 0i32;

        if self.nr51 & 0x10 != 0 { left  += o1; }
        if self.nr51 & 0x20 != 0 { left  += o2; }
        if self.nr51 & 0x40 != 0 { left  += o3; }
        if self.nr51 & 0x80 != 0 { left  += o4; }
        if self.nr51 & 0x01 != 0 { right += o1; }
        if self.nr51 & 0x02 != 0 { right += o2; }
        if self.nr51 & 0x04 != 0 { right += o3; }
        if self.nr51 & 0x08 != 0 { right += o4; }

        let lvol = ((self.nr50 >> 4) & 0x07) as i32 + 1;
        let rvol = (self.nr50 & 0x07) as i32 + 1;

        (left * lvol, right * rvol)
    }

    /// Compute fractional BLIP phase from sample accumulator position.
    #[inline]
    fn current_blip_phase(&self) -> usize {
        let thresh = self.sample_accum_thresh as usize;
        (self.sample_accum as usize * BLIP_PHASES / thresh)
            .min(BLIP_PHASES * 2 - 1)
    }

    /// Record a mix delta to the BLIP buffer if the mixed output has changed.
    fn record_mix_delta(&mut self) {
        let (new_l, new_r) = self.mix_integer();
        let dl = new_l - self.prev_left;
        let dr = new_r - self.prev_right;
        if dl != 0 || dr != 0 {
            let phase = self.current_blip_phase();
            self.blip.update(dl, dr, phase);
            self.prev_left = new_l;
            self.prev_right = new_r;
        }
    }

    /// Emit one 96 kHz sample from the BLIP buffer, then apply HPF.
    fn emit_sample(&mut self) {
        let (l, r) = self.blip.read();
        // Normalize ±480 → ±1.0
        let l = l / 480.0;
        let r = r / 480.0;

        // High-pass filter (coupling capacitor): removes DC offset from bipolar DAC.
        // Cutoff ~10 Hz at 96 kHz: alpha = 1 - (2π × 10 / 96000) ≈ 0.9993
        const HPF_ALPHA: f32 = 0.9993;
        self.hpf_left  = HPF_ALPHA * (self.hpf_left + l - self.hpf_prev_in_l);
        self.hpf_right = HPF_ALPHA * (self.hpf_right + r - self.hpf_prev_in_r);
        self.hpf_prev_in_l = l;
        self.hpf_prev_in_r = r;

        self.sample_buf.push(self.hpf_left);
        self.sample_buf.push(self.hpf_right);
    }

    // ── Main step ──────────────────────────────────────────────────────────

    pub fn step(&mut self, cycles: u32) {
        let tick = self.sample_accum_tick;
        let thresh = self.sample_accum_thresh;

        // Toggle lf_div (sub-2MHz phase) even when APU is off — hardware
        // continues toggling this counter regardless of NR52 power state.
        // Flips when APU cycles (cycles/2) is odd.
        // Normal speed: bus_cycles=4, apu_cycles=2 (even) → no toggle.
        // Double speed: bus_cycles=2, apu_cycles=1 (odd) → toggles every M-cycle.
        if (cycles >> 1) & 1 != 0 {
            self.lf_div = !self.lf_div;
        }

        if !self.power {
            // Still need to emit silence at the correct rate
            self.sample_accum += cycles as u64 * tick;
            while self.sample_accum >= thresh {
                self.sample_accum -= thresh;
                self.emit_sample();
            }
            return;
        }

        // Advance all frequency timers
        if self.ch1.enabled { self.ch1.tick_freq(cycles); }
        if self.ch2.enabled { self.ch2.tick_freq(cycles); }
        self.ch3.tick_freq(cycles);
        self.ch4.tick_freq(cycles);

        // Frame sequencer is now driven by DIV events from Bus, not by counting cycles here.

        // Check for amplitude change and record delta to BLIP buffer
        self.record_mix_delta();

        // Sample generation: emit one BLIP sample every CPU_RATE/SAMPLE_RATE T-cycles
        self.sample_accum += cycles as u64 * tick;
        while self.sample_accum >= thresh {
            self.sample_accum -= thresh;
            self.emit_sample();
        }
    }

    // ── Public interface ───────────────────────────────────────────────────

    /// Drain and return all pending audio samples (interleaved L/R f32).
    pub fn drain_samples(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.sample_buf)
    }

    /// Set CH1 volume to 0 after post-boot trigger. On real hardware, the boot
    /// ROM's startup sound has fully decayed by the time the game starts.
    pub fn set_ch1_post_boot_volume(&mut self) {
        self.ch1.volume = 0;
    }

    /// CGB PCM12 register (0xFF76): CH1 output in low nibble, CH2 in high nibble.
    /// Inactive channels read as 0. PCM mask only applies to CGB<=C (not our target CGB-D/E).
    pub fn pcm12(&self) -> u8 {
        let ch1 = if self.ch1.enabled { self.ch1.output() & 0x0F } else { 0 };
        let ch2 = if self.ch2.enabled { self.ch2.output() & 0x0F } else { 0 };
        ch1 | (ch2 << 4)
    }

    /// CGB PCM34 register (0xFF77): CH3 output in low nibble, CH4 in high nibble.
    /// Inactive channels read as 0. PCM mask only applies to CGB<=C (not our target CGB-D/E).
    pub fn pcm34(&self) -> u8 {
        let ch3 = if self.ch3.enabled { self.ch3.output() & 0x0F } else { 0 };
        let ch4 = if self.ch4.enabled { self.ch4.output() & 0x0F } else { 0 };
        ch3 | (ch4 << 4)
    }
}
