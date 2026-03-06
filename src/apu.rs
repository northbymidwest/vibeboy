/// Game Boy Color APU — 4-channel audio synthesis.
///
/// Channels:
///   CH1 — Square wave with frequency sweep (NR10-NR14, 0xFF10-0xFF14)
///   CH2 — Square wave                      (NR21-NR24, 0xFF16-0xFF19)
///   CH3 — Wave output (custom waveform)    (NR30-NR34, 0xFF1A-0xFF1E)
///   CH4 — Noise (LFSR)                     (NR41-NR44, 0xFF20-0xFF23)
///
/// Frame sequencer runs at 512 Hz (every 8192 T-cycles), 8-step sequence:
///   Steps 0,2,4,6 → length counters
///   Steps 2,6     → CH1 sweep
///   Step  7       → volume envelopes

use std::sync::Arc;

const DUTY_TABLE: [[u8; 8]; 4] = [
    [0, 0, 0, 0, 0, 0, 0, 1], // 12.5%
    [1, 0, 0, 0, 0, 0, 0, 1], // 25%
    [1, 0, 0, 0, 1, 1, 1, 1], // 50%
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
    freq_timer: u32,   // T-cycle countdown; reload = (2048 - freq) * 4
    duty_pos: u8,
    length_counter: u16,
    volume: u8,
    env_timer: u8,
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
        }
    }

    fn reload_period(&self) -> u32 {
        (2048 - self.freq as u32) * 4
    }

    fn tick_freq(&mut self, mut cycles: u32) {
        loop {
            if self.freq_timer > cycles {
                self.freq_timer -= cycles;
                return;
            }
            cycles -= self.freq_timer;
            self.duty_pos = (self.duty_pos + 1) & 7;
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

    fn clock_envelope(&mut self) {
        if self.env_period == 0 {
            return;
        }
        if self.env_timer > 0 {
            self.env_timer -= 1;
        }
        if self.env_timer == 0 {
            self.env_timer = self.env_period;
            if self.env_add && self.volume < 15 {
                self.volume += 1;
            } else if !self.env_add && self.volume > 0 {
                self.volume -= 1;
            }
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
        loop {
            if self.freq_timer > cycles {
                self.freq_timer -= cycles;
                return;
            }
            cycles -= self.freq_timer;
            self.sample_pos = (self.sample_pos + 1) & 31;
            let byte = self.wave_ram[(self.sample_pos >> 1) as usize];
            self.last_nibble = if self.sample_pos & 1 == 0 { byte >> 4 } else { byte & 0xF };
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

    fn clock_envelope(&mut self) {
        if self.env_period == 0 {
            return;
        }
        if self.env_timer > 0 {
            self.env_timer -= 1;
        }
        if self.env_timer == 0 {
            self.env_timer = self.env_period;
            if self.env_add && self.volume < 15 {
                self.volume += 1;
            } else if !self.env_add && self.volume > 0 {
                self.volume -= 1;
            }
        }
    }
}

// ── APU top level ──────────────────────────────────────────────────────────

const SAMPLE_RATE: u32 = 96_000;
const CPU_RATE: u32 = 4_194_304;
// Cycles per sample as a fixed-point ratio: emit one sample every CPU_RATE/SAMPLE_RATE cycles.
// Use u64 accumulator: add SAMPLE_RATE each T-cycle; emit when >= CPU_RATE.
const SAMPLE_ACCUM_TICK: u64 = SAMPLE_RATE as u64;
const SAMPLE_ACCUM_THRESH: u64 = CPU_RATE as u64;

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

    // Frame sequencer
    fs_counter: u32,
    fs_step: u8,

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
}

impl Apu {
    pub fn new() -> Self {
        let mut apu = Apu {
            ch1: SquareCh::new(),
            sweep: Sweep::new(),
            ch2: SquareCh::new(),
            ch3: WaveCh::new(),
            ch4: NoiseCh::new(),
            nr50: 0x77,
            nr51: 0xF3,
            power: true,
            fs_counter: 0,
            fs_step: 0,
            sample_accum: 0,
            sample_buf: Vec::with_capacity(1024),
            blip: BlipBuf::new(),
            prev_left: 0,
            prev_right: 0,
            hpf_left: 0.0,
            hpf_right: 0.0,
            hpf_prev_in_l: 0.0,
            hpf_prev_in_r: 0.0,
        };
        // Post-boot CH1 state: registers match what boot ROM left behind,
        // but the envelope has decayed volume to 0 during the boot animation.
        apu.ch1.duty = 2;
        apu.ch1.env_init_vol = 0xF;
        apu.ch1.env_period = 3;
        apu.ch1.dac_on = true;
        apu.ch1.enabled = true;
        apu.ch1.volume = 0;
        apu
    }

    // ── Register read ──────────────────────────────────────────────────────

    pub fn read(&self, addr: u16) -> u8 {
        match addr {
            // CH1
            0xFF10 => 0x80 | (self.sweep.period << 4) | (if self.sweep.negate { 0x08 } else { 0 }) | self.sweep.shift,
            0xFF11 => 0x3F | (self.ch1.duty << 6),
            0xFF12 => (self.ch1.env_init_vol << 4) | (if self.ch1.env_add { 0x08 } else { 0 }) | self.ch1.env_period,
            0xFF13 => 0xFF,
            0xFF14 => 0xBF | (if self.ch1.len_enable { 0x40 } else { 0 }),
            0xFF15 => 0xFF,
            // CH2
            0xFF16 => 0x3F | (self.ch2.duty << 6),
            0xFF17 => (self.ch2.env_init_vol << 4) | (if self.ch2.env_add { 0x08 } else { 0 }) | self.ch2.env_period,
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
            0xFF21 => (self.ch4.env_init_vol << 4) | (if self.ch4.env_add { 0x08 } else { 0 }) | self.ch4.env_period,
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
            // Wave RAM
            0xFF30..=0xFF3F => self.ch3.wave_ram[(addr - 0xFF30) as usize],
            _ => 0xFF,
        }
    }

    // ── Register write ─────────────────────────────────────────────────────

    pub fn write(&mut self, addr: u16, val: u8) {
        // Wave RAM always accessible
        if (0xFF30..=0xFF3F).contains(&addr) {
            self.ch3.wave_ram[(addr - 0xFF30) as usize] = val;
            return;
        }

        // NR52: power control — always writable
        if addr == 0xFF26 {
            let was_on = self.power;
            self.power = val & 0x80 != 0;
            if was_on && !self.power {
                self.power_off();
            }
            return;
        }

        // Length counters writable even when powered off
        if !self.power {
            match addr {
                0xFF11 => self.ch1.length_counter = 64 - (val & 0x3F) as u16,
                0xFF16 => self.ch2.length_counter = 64 - (val & 0x3F) as u16,
                0xFF1B => self.ch3.length_counter = 256 - val as u16,
                0xFF20 => self.ch4.length_counter = 64 - (val & 0x3F) as u16,
                _ => {}
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
                self.ch1.env_init_vol = val >> 4;
                self.ch1.env_add     = val & 0x08 != 0;
                self.ch1.env_period  = val & 0x07;
                self.ch1.dac_on = val & 0xF8 != 0;
                if !self.ch1.dac_on { self.ch1.enabled = false; }
            }
            0xFF13 => self.ch1.freq = (self.ch1.freq & 0x700) | val as u16,
            0xFF14 => {
                self.ch1.freq = (self.ch1.freq & 0x0FF) | (((val & 0x07) as u16) << 8);
                let was_enabled = self.ch1.len_enable;
                self.ch1.len_enable = val & 0x40 != 0;
                // Extra length clock on 0→1 transition of length enable on odd fs step
                if !was_enabled && val & 0x40 != 0 && self.fs_step & 1 != 0 {
                    self.ch1.clock_length();
                }
                if val & 0x80 != 0 {
                    self.trigger_ch1();
                }
            }
            // ── CH2 ────────────────────────────────────────────────────────
            0xFF16 => {
                self.ch2.duty = (val >> 6) & 0x03;
                self.ch2.length_counter = 64 - (val & 0x3F) as u16;
            }
            0xFF17 => {
                self.ch2.env_init_vol = val >> 4;
                self.ch2.env_add     = val & 0x08 != 0;
                self.ch2.env_period  = val & 0x07;
                self.ch2.dac_on = val & 0xF8 != 0;
                if !self.ch2.dac_on { self.ch2.enabled = false; }
            }
            0xFF18 => self.ch2.freq = (self.ch2.freq & 0x700) | val as u16,
            0xFF19 => {
                self.ch2.freq = (self.ch2.freq & 0x0FF) | (((val & 0x07) as u16) << 8);
                let was_enabled = self.ch2.len_enable;
                self.ch2.len_enable = val & 0x40 != 0;
                if !was_enabled && val & 0x40 != 0 && self.fs_step & 1 != 0 {
                    self.ch2.clock_length();
                }
                if val & 0x80 != 0 {
                    self.trigger_ch2();
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
                if !was_enabled && val & 0x40 != 0 && self.fs_step & 1 != 0 {
                    self.ch3.clock_length();
                }
                if val & 0x80 != 0 && self.ch3.dac_on {
                    self.trigger_ch3();
                }
            }
            // ── CH4 ────────────────────────────────────────────────────────
            0xFF20 => self.ch4.length_counter = 64 - (val & 0x3F) as u16,
            0xFF21 => {
                self.ch4.env_init_vol = val >> 4;
                self.ch4.env_add     = val & 0x08 != 0;
                self.ch4.env_period  = val & 0x07;
                self.ch4.dac_on = val & 0xF8 != 0;
                if !self.ch4.dac_on { self.ch4.enabled = false; }
            }
            0xFF22 => {
                self.ch4.clock_shift  = val >> 4;
                self.ch4.lfsr_narrow  = val & 0x08 != 0;
                self.ch4.divisor_code = val & 0x07;
            }
            0xFF23 => {
                let was_enabled = self.ch4.len_enable;
                self.ch4.len_enable = val & 0x40 != 0;
                if !was_enabled && val & 0x40 != 0 && self.fs_step & 1 != 0 {
                    self.ch4.clock_length();
                }
                if val & 0x80 != 0 && self.ch4.dac_on {
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

    fn trigger_ch1(&mut self) {
        self.ch1.enabled = true;
        if self.ch1.length_counter == 0 {
            self.ch1.length_counter = 64;
        }
        self.ch1.freq_timer = self.ch1.reload_period();
        self.ch1.volume    = self.ch1.env_init_vol;
        self.ch1.env_timer = self.ch1.env_period;

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

    fn trigger_ch2(&mut self) {
        self.ch2.enabled = true;
        if self.ch2.length_counter == 0 {
            self.ch2.length_counter = 64;
        }
        self.ch2.freq_timer = self.ch2.reload_period();
        self.ch2.volume    = self.ch2.env_init_vol;
        self.ch2.env_timer = self.ch2.env_period;
        if !self.ch2.dac_on {
            self.ch2.enabled = false;
        }
    }

    fn trigger_ch3(&mut self) {
        self.ch3.enabled = true;
        if self.ch3.length_counter == 0 {
            self.ch3.length_counter = 256;
        }
        self.ch3.freq_timer  = self.ch3.reload_period() + 6;
        self.ch3.sample_pos  = 0;
        self.ch3.last_nibble = 0;
    }

    fn trigger_ch4(&mut self) {
        self.ch4.enabled = true;
        if self.ch4.length_counter == 0 {
            self.ch4.length_counter = 64;
        }
        self.ch4.volume    = self.ch4.env_init_vol;
        self.ch4.env_timer = self.ch4.env_period;
        self.ch4.lfsr      = 0x7FFF;
        self.ch4.freq_timer = self.ch4.reload_period();
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

        self.ch1 = SquareCh::new();
        self.sweep = Sweep::new();
        self.ch2 = SquareCh::new();
        self.ch3 = WaveCh::new();
        self.ch4 = NoiseCh::new();
        self.nr50 = 0;
        self.nr51 = 0;
        self.fs_step = 0;
    }

    // ── Frame sequencer ────────────────────────────────────────────────────

    fn clock_frame_seq(&mut self) {
        match self.fs_step {
            0 | 4 => {
                self.ch1.clock_length();
                self.ch2.clock_length();
                self.ch3.clock_length();
                self.ch4.clock_length();
            }
            2 | 6 => {
                self.ch1.clock_length();
                self.ch2.clock_length();
                self.ch3.clock_length();
                self.ch4.clock_length();
                let ch1 = &mut self.ch1;
                self.sweep.clock(ch1);
            }
            7 => {
                self.ch1.clock_envelope();
                self.ch2.clock_envelope();
                self.ch4.clock_envelope();
            }
            _ => {}
        }
        self.fs_step = (self.fs_step + 1) & 7;
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
    fn current_blip_phase(&self) -> usize {
        (self.sample_accum as usize * BLIP_PHASES / SAMPLE_ACCUM_THRESH as usize)
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
        if !self.power {
            // Still need to emit silence at the correct rate
            self.sample_accum += cycles as u64 * SAMPLE_ACCUM_TICK;
            while self.sample_accum >= SAMPLE_ACCUM_THRESH {
                self.sample_accum -= SAMPLE_ACCUM_THRESH;
                self.emit_sample();
            }
            return;
        }

        // Advance all frequency timers
        self.ch1.tick_freq(cycles);
        self.ch2.tick_freq(cycles);
        self.ch3.tick_freq(cycles);
        self.ch4.tick_freq(cycles);

        // Frame sequencer (512 Hz = every 8192 T-cycles)
        self.fs_counter += cycles;
        while self.fs_counter >= 8192 {
            self.fs_counter -= 8192;
            self.clock_frame_seq();
        }

        // Check for amplitude change and record delta to BLIP buffer
        self.record_mix_delta();

        // Sample generation: emit one BLIP sample every CPU_RATE/SAMPLE_RATE T-cycles
        self.sample_accum += cycles as u64 * SAMPLE_ACCUM_TICK;
        while self.sample_accum >= SAMPLE_ACCUM_THRESH {
            self.sample_accum -= SAMPLE_ACCUM_THRESH;
            self.emit_sample();
        }
    }

    // ── Public interface ───────────────────────────────────────────────────

    /// Drain and return all pending audio samples (interleaved L/R f32).
    pub fn drain_samples(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.sample_buf)
    }

    /// CGB PCM12 register (0xFF76): CH1 output in low nibble, CH2 in high nibble.
    pub fn pcm12(&self) -> u8 {
        (self.ch1.output() & 0x0F) | ((self.ch2.output() & 0x0F) << 4)
    }

    /// CGB PCM34 register (0xFF77): CH3 output in low nibble, CH4 in high nibble.
    pub fn pcm34(&self) -> u8 {
        (self.ch3.output() & 0x0F) | ((self.ch4.output() & 0x0F) << 4)
    }
}
