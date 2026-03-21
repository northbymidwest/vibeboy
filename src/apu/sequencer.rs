/// Frame sequencer (DIV-coupled) and channel trigger logic.

use super::Apu;
use super::channels::{SquareCh, Sweep, WaveCh, NoiseCh, set_envelope_clock};

impl Apu {
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
            self.sweep.clock(&mut self.ch1);
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

    // ── Channel triggers ───────────────────────────────────────────────────

    pub(super) fn trigger_ch1(&mut self, val: u8) {
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
        self.ch1.update_sample();
    }

    pub(super) fn trigger_ch2(&mut self, val: u8) {
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
        self.ch2.update_sample();
    }

    pub(super) fn trigger_ch3(&mut self) {
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
        if !self.ch3.dac_on {
            self.ch3.enabled = false;
        }
    }

    pub(super) fn trigger_ch4(&mut self) {
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

        // Reset LFSR to all 1s on trigger (Pan Docs: "all bits are set to 1")
        self.ch4.lfsr = 0x7FFF;
        self.ch4.lfsr_sample = true;

        let raw_div = self.ch4.divisor_code as u32;
        let was_active = self.ch4.counter_active;
        let base = if raw_div == 0 { 12 } else { raw_div * 8 + 12 };
        let align = self.ch4.alignment & 7;
        let adj: i32 = if was_active && raw_div == 0 && (align & 2) == 0 {
            8
        } else if !was_active && raw_div > 1 && (align & 6) == 0 {
            -8
        } else {
            0
        };
        self.ch4.counter_countdown = ((base as i32) + adj).max(1) as u32;
        self.ch4.counter = 0;
        self.ch4.counter_active = true;

        if !self.ch4.dac_on {
            self.ch4.enabled = false;
        }
    }

    // ── Power off ──────────────────────────────────────────────────────────

    pub(super) fn power_off(&mut self) {
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
}
