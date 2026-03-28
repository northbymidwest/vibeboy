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

mod channels;
mod registers;
mod sequencer;

use channels::{SquareCh, Sweep, WaveCh, NoiseCh};

use std::sync::Arc;

// ── BLIP buffer (band-limited synthesis) ──────────────────────────────────

const BLIP_WIDTH: usize = 64;
const BLIP_PHASES: usize = 256;
const BLIP_BUF_SIZE: usize = BLIP_WIDTH * 2; // 128
const BLIP_ONE: i32 = 0x10000; // 65536

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct BlipBuf {
    #[serde(skip, default = "blip_sinc_table")]
    steps: Arc<[[i32; BLIP_WIDTH]; BLIP_PHASES]>,
    #[serde(with = "serde_big_array::BigArray")]
    buf_l: [i32; BLIP_BUF_SIZE],
    #[serde(with = "serde_big_array::BigArray")]
    buf_r: [i32; BLIP_BUF_SIZE],
    pos: usize,
    out_l: i32,
    out_r: i32,
}

fn blip_sinc_table() -> Arc<[[i32; BLIP_WIDTH]; BLIP_PHASES]> {
    use std::sync::LazyLock;
    static TABLE: LazyLock<Arc<[[i32; BLIP_WIDTH]; BLIP_PHASES]>> = LazyLock::new(|| {
        let n = BLIP_WIDTH * BLIP_PHASES;
        let lowpass = 15.0_f64 / 16.0;
        let mut master = vec![0.0_f64; n];

        for i in 0..n {
            let x = (i as f64 - n as f64 / 2.0) * std::f64::consts::PI * 2.0 * lowpass / BLIP_PHASES as f64;
            let sinc = if x.abs() < 1e-12 { 1.0 } else { x.sin() / x };
            let theta = 2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64;
            let a0 = 7938.0 / 18608.0;
            let a1 = 9240.0 / 18608.0;
            let a2 = 1430.0 / 18608.0;
            let window = a0 - a1 * theta.cos() + a2 * (2.0 * theta).cos();
            master[i] = sinc * window;
        }

        let sum: f64 = master.iter().sum();
        let inv = BLIP_PHASES as f64 / sum;
        for v in &mut master {
            *v *= inv;
        }

        let mut steps = Box::new([[0i32; BLIP_WIDTH]; BLIP_PHASES]);
        for p in 0..BLIP_PHASES {
            for t in 0..BLIP_WIDTH {
                steps[p][t] = (master[t * BLIP_PHASES + p] * BLIP_ONE as f64) as i32;
            }
        }
        Arc::from(steps)
    });
    TABLE.clone()
}

fn default_blip_buf() -> BlipBuf {
    BlipBuf::new()
}

impl BlipBuf {
    fn new() -> Self {
        BlipBuf {
            steps: blip_sinc_table(),
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

// ── APU top level ──────────────────────────────────────────────────────────

/// Default sample rate when not specified by a frontend.
pub const DEFAULT_SAMPLE_RATE: u32 = 96_000;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    skip_div_event: u8,

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
    #[serde(skip)]
    pub sample_buf: Vec<f32>,

    // When true, skip sample_buf accumulation (headless / test runner mode)
    #[serde(skip)]
    pub headless: bool,

    // BLIP buffer for band-limited resampling
    #[serde(skip, default = "default_blip_buf")]
    blip: BlipBuf,
    #[serde(skip)]
    prev_left: i32,
    #[serde(skip)]
    prev_right: i32,
    #[serde(skip)]
    prev_channel_outputs: u32,

    // High-pass filter state (models the coupling capacitor on real hardware)
    #[serde(skip)]
    hpf_left: f32,
    #[serde(skip)]
    hpf_right: f32,
    #[serde(skip)]
    hpf_prev_in_l: f32,
    #[serde(skip)]
    hpf_prev_in_r: f32,

    // Model-dependent clock rate for sample timing
    sample_accum_tick: u64,
    sample_accum_thresh: u64,

    // CGB mode flag (affects power-off behavior)
    is_cgb: bool,

    // Output sample rate (e.g. 96_000 or 48_000)
    sample_rate: u32,
}

/// Transient APU filter state not included in serde snapshots.
/// Used by runahead to preserve audio continuity across save/restore.
pub struct ApuFilterState {
    pub prev_left: i32,
    pub prev_right: i32,
    pub hpf_left: f32,
    pub hpf_right: f32,
    pub hpf_prev_in_l: f32,
    pub hpf_prev_in_r: f32,
}

impl Apu {
    /// Save transient filter state (not captured by serde).
    pub fn save_filter_state(&self) -> ApuFilterState {
        ApuFilterState {
            prev_left: self.prev_left,
            prev_right: self.prev_right,
            hpf_left: self.hpf_left,
            hpf_right: self.hpf_right,
            hpf_prev_in_l: self.hpf_prev_in_l,
            hpf_prev_in_r: self.hpf_prev_in_r,
        }
    }

    /// Restore transient filter state.
    pub fn restore_filter_state(&mut self, state: &ApuFilterState) {
        self.prev_left = state.prev_left;
        self.prev_right = state.prev_right;
        self.hpf_left = state.hpf_left;
        self.hpf_right = state.hpf_right;
        self.hpf_prev_in_l = state.hpf_prev_in_l;
        self.hpf_prev_in_r = state.hpf_prev_in_r;
    }

    pub fn new(cpu_clock_rate: u32, is_cgb: bool, is_sgb: bool, sample_rate: u32) -> Self {
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
            headless: false,
            blip: BlipBuf::new(),
            prev_left: 0,
            prev_right: 0,
            prev_channel_outputs: u32::MAX,
            hpf_left: 0.0,
            hpf_right: 0.0,
            hpf_prev_in_l: 0.0,
            hpf_prev_in_r: 0.0,
            sample_accum_tick: sample_rate as u64,
            sample_accum_thresh: cpu_clock_rate as u64,
            is_cgb,
            sample_rate,
        };
        // Post-boot CH1 state: registers match what boot ROM left behind
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
    pub fn reset(cpu_clock_rate: u32, is_cgb: bool, sample_rate: u32) -> Self {
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
            headless: false,
            blip: BlipBuf::new(),
            prev_left: 0,
            prev_right: 0,
            prev_channel_outputs: u32::MAX,
            hpf_left: 0.0,
            hpf_right: 0.0,
            hpf_prev_in_l: 0.0,
            hpf_prev_in_r: 0.0,
            sample_accum_tick: sample_rate as u64,
            sample_accum_thresh: cpu_clock_rate as u64,
            is_cgb,
            sample_rate,
        }
    }

    /// Current output sample rate.
    pub fn sample_rate(&self) -> u32 { self.sample_rate }

    /// Change the output sample rate (used when restoring save states across
    /// frontends that may run at different rates).
    pub fn set_sample_rate(&mut self, rate: u32) {
        self.sample_rate = rate;
        self.sample_accum_tick = rate as u64;
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
    fn write_lf(&self) -> u32 {
        if self.lf_div { 1 } else { 0 }
    }

    // ── Integer mixing (for BLIP delta detection) ─────────────────────────

    /// Compute current mixed L/R as integers.
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
    #[inline]
    fn record_mix_delta(&mut self) {
        // Quick check: skip mix_integer if no channel output changed
        let o = (self.ch1.output() as u32)
            | ((self.ch2.output() as u32) << 8)
            | ((self.ch3.output() as u32) << 16)
            | ((self.ch4.output() as u32) << 24);
        if o == self.prev_channel_outputs {
            return;
        }
        self.prev_channel_outputs = o;

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
        let l = l / 480.0;
        let r = r / 480.0;

        const HPF_ALPHA: f32 = 0.9993;
        self.hpf_left  = HPF_ALPHA * (self.hpf_left + l - self.hpf_prev_in_l);
        self.hpf_right = HPF_ALPHA * (self.hpf_right + r - self.hpf_prev_in_r);
        self.hpf_prev_in_l = l;
        self.hpf_prev_in_r = r;

        if !self.headless {
            // Clamp to [-1, 1] — the BLIP sinc filter overshoots on sharp
            // transitions (Gibbs phenomenon), which would cause hard digital
            // clipping in audio backends.
            self.sample_buf.push(self.hpf_left.clamp(-1.0, 1.0));
            self.sample_buf.push(self.hpf_right.clamp(-1.0, 1.0));
        }
    }

    // ── Main step ──────────────────────────────────────────────────────────

    pub fn step(&mut self, cycles: u32) {
        let tick = self.sample_accum_tick;
        let thresh = self.sample_accum_thresh;

        let lf_toggled = (cycles >> 1) & 1 != 0;
        if lf_toggled {
            self.lf_div = !self.lf_div;
        }

        if !self.power {
            self.sample_accum += cycles as u64 * tick;
            while self.sample_accum >= thresh {
                self.sample_accum -= thresh;
                self.emit_sample();
            }
            return;
        }

        if self.ch1.enabled { self.ch1.tick_freq(cycles); }
        if self.ch2.enabled { self.ch2.tick_freq(cycles); }
        self.ch3.tick_freq(cycles);
        self.ch4.alignment += cycles;
        self.ch4.tick_counter(cycles);

        // Sweep calculation runs in 1MHz domain (one step per 2 T-cycles).
        // Only step when there's a pending calculation (avoids unnecessary work
        // and prevents restart_hold from counting down prematurely).
        if self.sweep.calc_pending {
            let sweep_steps = cycles / 2;
            for _ in 0..sweep_steps {
                self.sweep.step_1mhz(&mut self.ch1);
            }
        }

        self.record_mix_delta();

        self.sample_accum += cycles as u64 * tick;
        while self.sample_accum >= thresh {
            self.sample_accum -= thresh;
            self.emit_sample();
        }
    }

    // ── Public interface ───────────────────────────────────────────────────

    pub fn drain_samples(&mut self) -> Vec<f32> {
        std::mem::replace(&mut self.sample_buf, Vec::with_capacity(3200))
    }

    pub fn set_ch1_post_boot_volume(&mut self) {
        self.ch1.volume = 0;
    }

    pub fn pcm12(&self) -> u8 {
        let ch1 = if self.ch1.enabled { self.ch1.output() & 0x0F } else { 0 };
        let ch2 = if self.ch2.enabled { self.ch2.output() & 0x0F } else { 0 };
        ch1 | (ch2 << 4)
    }

    pub fn pcm34(&self) -> u8 {
        let ch3 = if self.ch3.enabled { self.ch3.output() & 0x0F } else { 0 };
        let ch4 = if self.ch4.enabled { self.ch4.output() & 0x0F } else { 0 };
        ch3 | (ch4 << 4)
    }
}
