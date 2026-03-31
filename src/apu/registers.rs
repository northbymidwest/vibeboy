/// APU register read/write dispatch (0xFF10-0xFF3F).

use super::Apu;
use super::channels::{NOISE_DIVISORS, nrx2_glitch};

impl Apu {
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
            // Wave RAM: when CH3 is active, access redirects to current sample position.
            // CGB: always accessible (no timing restriction).
            // DMG: only accessible on the same cycle CH3 reads (wave_form_just_read).
            0xFF30..=0xFF3F => {
                if self.ch3.enabled {
                    if self.is_cgb || self.ch3.wave_form_just_read {
                        self.ch3.wave_ram[(self.ch3.sample_pos >> 1) as usize]
                    } else {
                        0xFF
                    }
                } else {
                    self.ch3.wave_ram[(addr - 0xFF30) as usize]
                }
            }
            _ => 0xFF,
        }
    }

    pub fn write(&mut self, addr: u16, val: u8) {
        // Wave RAM: when CH3 is active, access redirects to current sample position.
        // CGB: always accessible. DMG: only when wave_form_just_read.
        if (0xFF30..=0xFF3F).contains(&addr) {
            if self.ch3.enabled {
                if self.is_cgb || self.ch3.wave_form_just_read {
                    self.ch3.wave_ram[(self.ch3.sample_pos >> 1) as usize] = val;
                }
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
                self.sweep.write_nr10(val, &mut self.ch1);
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
                    self.ch1.update_sample();
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
                    let new_hi = (val & 0x07) as u32;
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
                    self.ch2.update_sample();
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
                    let new_hi = (val & 0x07) as u32;
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
                // NR43 write: update frequency parameters.
                if self.ch4.countdown_reloaded {
                    let new_div_code = val & 0x07;
                    let new_divisor = NOISE_DIVISORS[new_div_code as usize];
                    self.ch4.counter_countdown = new_divisor;
                }
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
                self.record_mix_delta_inner(true);
            }
            0xFF25 => {
                self.nr51 = val;
                self.record_mix_delta_inner(true);
            }
            _ => {}
        }
    }
}
