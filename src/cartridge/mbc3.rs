use super::Cartridge;
use crate::clock::Clock;
use std::sync::Arc;

/// Standard RTC save footer: five live registers (S, M, H, DL, DH) as u32 LE at
/// 0..20, the five latched registers at 20..40, and a u64 LE unix timestamp at
/// 40..48. This is the layout shared by most other emulators.
const RTC_FOOTER_LEN: usize = 48;
/// Variant of the standard footer with a u32 LE timestamp at 40..44.
const RTC_FOOTER_LEN_TS32: usize = 44;
/// Implemented bits of each RTC register: S/M 6 bits, H 5 bits, DL 8 bits,
/// DH bit 0 (day bit 8), bit 6 (halt) and bit 7 (day carry).
const RTC_MASKS: [u8; 5] = [0x3F, 0x3F, 0x1F, 0xFF, 0xC1];

/// RTC state decoded from a battery save footer.
struct RtcFooter {
    regs: [u8; 5],
    latched: [u8; 5],
    /// Unix timestamp of the save, or 0 if unknown.
    saved_ts: u64,
}

impl RtcFooter {
    fn parse(f: &[u8]) -> Option<Self> {
        let u32_at = |o: usize| u32::from_le_bytes([f[o], f[o + 1], f[o + 2], f[o + 3]]);
        let mut regs = [0u8; 5];
        let mut latched = [0u8; 5];
        let saved_ts = match f.len() {
            RTC_FOOTER_LEN | RTC_FOOTER_LEN_TS32 => {
                for i in 0..5 {
                    regs[i] = u32_at(i * 4) as u8;
                    latched[i] = u32_at(20 + i * 4) as u8;
                }
                if f.len() == RTC_FOOTER_LEN {
                    let mut ts = [0u8; 8];
                    ts.copy_from_slice(&f[40..48]);
                    u64::from_le_bytes(ts)
                } else {
                    u32_at(40) as u64
                }
            }
            _ => return None,
        };
        for i in 0..5 {
            regs[i] &= RTC_MASKS[i];
            latched[i] &= RTC_MASKS[i];
        }
        Some(RtcFooter {
            regs,
            latched,
            saved_ts,
        })
    }
}

pub struct Mbc3 {
    rom: Arc<[u8]>,
    ram: Vec<u8>,
    rom_bank: usize,
    ram_bank: usize, // 0-3 (or 0-7 for MBC30) = RAM, 0x08-0x0C = RTC registers
    ram_enabled: bool,
    battery: bool,
    has_rtc: bool,
    mbc30: bool,           // MBC30: 8-bit ROM bank, 8 RAM banks (Pokemon Crystal JP)
    rtc_regs: [u8; 5],     // S, M, H, DL, DH (live counters)
    rtc_latched: [u8; 5],  // latched snapshot for reading
    rtc_latch_ready: bool, // true after writing 0x00, arms latch
    rtc_last_secs: u64,    // last time RTC was updated (seconds from clock)
    clock: Arc<dyn Clock>,
}

impl Mbc3 {
    pub(super) fn new(
        rom: Arc<[u8]>,
        ram_size: usize,
        battery: bool,
        has_rtc: bool,
        clock: Arc<dyn Clock>,
    ) -> Self {
        // MBC30 detection: ROM > 2MB or RAM > 32KB (only Pokemon Crystal JP)
        let mbc30 = rom.len() > 0x200000 || ram_size > 0x8000;
        if mbc30 {
            log::info!(
                "MBC30 detected (ROM={}KB, RAM={}KB)",
                rom.len() / 1024,
                ram_size / 1024
            );
        }
        let rtc_last_secs = clock.now_secs();
        Mbc3 {
            rom,
            ram: vec![0u8; ram_size],
            rom_bank: 1,
            ram_bank: 0,
            ram_enabled: false,
            battery,
            has_rtc,
            mbc30,
            rtc_regs: [0; 5],
            rtc_latched: [0; 5],
            rtc_latch_ready: false,
            rtc_last_secs,
            clock,
        }
    }

    /// Advance RTC registers by elapsed wall-clock time since last update.
    fn advance_rtc(&mut self) {
        // Don't advance if halted (DH bit 6)
        if self.rtc_regs[4] & 0x40 != 0 {
            return;
        }

        let now = self.clock.now_secs();
        let elapsed = now.saturating_sub(self.rtc_last_secs);
        self.rtc_last_secs = now;
        if elapsed == 0 {
            return;
        }

        self.add_seconds_to_rtc(elapsed);
    }

    fn add_seconds_to_rtc(&mut self, seconds: u64) {
        let mut secs = (self.rtc_regs[0] as u64).saturating_add(seconds);
        let mut mins = self.rtc_regs[1] as u64 + secs / 60;
        secs %= 60;
        let mut hrs = self.rtc_regs[2] as u64 + mins / 60;
        mins %= 60;
        let day_lo = self.rtc_regs[3] as u64;
        let day_hi_bit = (self.rtc_regs[4] & 0x01) as u64;
        let mut days = day_lo | (day_hi_bit << 8);
        days += hrs / 24;
        hrs %= 24;

        self.rtc_regs[0] = secs as u8;
        self.rtc_regs[1] = mins as u8;
        self.rtc_regs[2] = hrs as u8;
        self.rtc_regs[3] = days as u8; // low 8 bits
        // Preserve halt bit, set day bit 8. The carry bit is sticky: it is set
        // when the day counter overflows 511 and only cleared by a DH write.
        let carry = if days > 511 { 0x80 } else { 0 };
        self.rtc_regs[4] = (self.rtc_regs[4] & 0xC0) | ((days >> 8) as u8 & 0x01) | carry;
    }
}

impl Cartridge for Mbc3 {
    fn read_rom(&self, addr: u16) -> u8 {
        let idx = match addr {
            0x0000..=0x3FFF => addr as usize,
            0x4000..=0x7FFF => self.rom_bank * 0x4000 + (addr as usize - 0x4000),
            _ => return 0xFF,
        };
        self.rom
            .get(idx % self.rom.len().max(1))
            .copied()
            .unwrap_or(0xFF)
    }

    fn write_rom(&mut self, addr: u16, val: u8) {
        match addr {
            0x0000..=0x1FFF => self.ram_enabled = val & 0x0F == 0x0A,
            0x2000..=0x3FFF => {
                let mask = if self.mbc30 { 0xFF } else { 0x7F };
                let b = (val & mask) as usize;
                self.rom_bank = if b == 0 { 1 } else { b };
            }
            0x4000..=0x5FFF => {
                let v = val as usize;
                // 0x00-0x03 = RAM bank, 0x08-0x0C = RTC register select
                self.ram_bank = v;
            }
            0x6000..=0x7FFF if self.has_rtc => {
                if val == 0x00 {
                    self.rtc_latch_ready = true;
                } else if val == 0x01 && self.rtc_latch_ready {
                    self.advance_rtc();
                    self.rtc_latched = self.rtc_regs;
                    self.rtc_latch_ready = false;
                } else {
                    self.rtc_latch_ready = false;
                }
            }
            _ => {}
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        if !self.ram_enabled {
            return 0xFF;
        }
        let max_ram_bank = if self.mbc30 { 0x07 } else { 0x03 };
        match self.ram_bank {
            b if b <= max_ram_bank => {
                let idx = b * 0x2000 + (addr as usize - 0xA000);
                self.ram
                    .get(idx % self.ram.len().max(1))
                    .copied()
                    .unwrap_or(0xFF)
            }
            0x08..=0x0C if self.has_rtc => self.rtc_latched[self.ram_bank - 0x08],
            _ => 0xFF,
        }
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        if !self.ram_enabled {
            return;
        }
        let max_ram_bank = if self.mbc30 { 0x07 } else { 0x03 };
        match self.ram_bank {
            b if b <= max_ram_bank => {
                let idx = b * 0x2000 + (addr as usize - 0xA000);
                let len = self.ram.len().max(1);
                if let Some(b) = self.ram.get_mut(idx % len) {
                    *b = val;
                }
            }
            0x08..=0x0C if self.has_rtc => {
                let reg = self.ram_bank - 0x08;
                // If unhalting (clearing halt bit), reset base so time counts from now
                if reg == 4 && self.rtc_regs[4] & 0x40 != 0 && val & 0x40 == 0 {
                    self.rtc_last_secs = self.clock.now_secs();
                }
                // Advance RTC before overwriting registers so accumulated time isn't lost
                self.advance_rtc();
                self.rtc_regs[reg] = val & RTC_MASKS[reg];
            }
            _ => {}
        }
    }

    fn has_battery(&self) -> bool {
        self.battery
    }

    fn ram_data(&self) -> &[u8] {
        &self.ram
    }

    fn save_data(&self) -> Vec<u8> {
        let mut data = self.ram.clone();
        if self.has_rtc {
            let mut footer = [0u8; RTC_FOOTER_LEN];
            for i in 0..5 {
                footer[i * 4] = self.rtc_regs[i];
                footer[20 + i * 4] = self.rtc_latched[i];
            }
            let ts = self.clock.unix_timestamp_secs();
            footer[40..48].copy_from_slice(&ts.to_le_bytes());
            data.extend_from_slice(&footer);
        }
        data
    }

    fn load_ram(&mut self, data: &[u8]) {
        let ram_len = self.ram.len();
        // The RTC footer follows the header-sized RAM.
        let footer_start = (self.has_rtc
            && matches!(
                data.len().checked_sub(ram_len),
                Some(RTC_FOOTER_LEN | RTC_FOOTER_LEN_TS32)
            ))
        .then_some(ram_len);
        let ram_part = &data[..footer_start.unwrap_or(data.len())];
        let copy_len = ram_len.min(ram_part.len());
        self.ram[..copy_len].copy_from_slice(&ram_part[..copy_len]);

        let Some(footer) = footer_start.and_then(|start| RtcFooter::parse(&data[start..])) else {
            return;
        };
        self.rtc_regs = footer.regs;
        self.rtc_latched = footer.latched;
        if footer.saved_ts != 0 && self.rtc_regs[4] & 0x40 == 0 {
            let elapsed = self
                .clock
                .unix_timestamp_secs()
                .saturating_sub(footer.saved_ts);
            if elapsed > 0 {
                self.add_seconds_to_rtc(elapsed);
            }
        }
        self.rtc_last_secs = self.clock.now_secs();
    }
    fn snapshot_state(&self) -> Vec<u8> {
        let mut s = Vec::new();
        s.extend_from_slice(&(self.rom_bank as u32).to_le_bytes());
        s.extend_from_slice(&(self.ram_bank as u32).to_le_bytes());
        s.push(self.ram_enabled as u8);
        s.push(self.rtc_latch_ready as u8);
        s.extend_from_slice(&self.rtc_regs);
        s.extend_from_slice(&self.rtc_latched);
        s.extend_from_slice(&self.ram);
        s
    }
    fn restore_state(&mut self, d: &[u8]) {
        if d.len() < 20 {
            return;
        }
        self.rom_bank = u32::from_le_bytes([d[0], d[1], d[2], d[3]]) as usize;
        self.ram_bank = u32::from_le_bytes([d[4], d[5], d[6], d[7]]) as usize;
        self.ram_enabled = d[8] != 0;
        self.rtc_latch_ready = d[9] != 0;
        self.rtc_regs.copy_from_slice(&d[10..15]);
        self.rtc_latched.copy_from_slice(&d[15..20]);
        self.rtc_last_secs = self.clock.now_secs();
        let ram = &d[20..];
        let len = self.ram.len().min(ram.len());
        self.ram[..len].copy_from_slice(&ram[..len]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedClock(u64);

    impl Clock for FixedClock {
        fn now_secs(&self) -> u64 {
            self.0
        }
        fn unix_timestamp_secs(&self) -> u64 {
            self.0
        }
    }

    const NOW: u64 = 1_750_000_000;

    fn cart(ram_size: usize) -> Mbc3 {
        let rom: Arc<[u8]> = vec![0u8; 0x8000].into();
        Mbc3::new(rom, ram_size, true, true, Arc::new(FixedClock(NOW)))
    }

    #[test]
    fn standard_footer_round_trip() {
        let mut c = cart(0x2000);
        c.rtc_regs = [5, 6, 7, 8, 0x01];
        c.rtc_latched = [1, 2, 3, 4, 0x00];
        c.ram[0] = 0xAB;
        let save = c.save_data();
        assert_eq!(save.len(), 0x2000 + 48);
        let f = &save[0x2000..];
        assert_eq!(&f[..8], &[5, 0, 0, 0, 6, 0, 0, 0]);
        assert_eq!(u64::from_le_bytes(f[40..48].try_into().unwrap()), NOW);

        let mut d = cart(0x2000);
        d.load_ram(&save);
        assert_eq!(d.ram[0], 0xAB);
        assert_eq!(d.rtc_regs, [5, 6, 7, 8, 0x01]);
        assert_eq!(d.rtc_latched, [1, 2, 3, 4, 0x00]);
    }

    #[test]
    fn footer_with_u32_timestamp_advances_time() {
        let mut data = vec![0u8; 44];
        data[0] = 10; // S
        data[40..44].copy_from_slice(&((NOW - 65) as u32).to_le_bytes());
        let mut c = cart(0);
        c.load_ram(&data);
        assert_eq!(c.rtc_regs[..2], [15, 1]);
    }

    #[test]
    fn day_carry_is_sticky_and_writes_are_masked() {
        let mut c = cart(0);
        c.rtc_regs = [0, 0, 0, 0xFF, 0x01];
        c.add_seconds_to_rtc(86_400);
        assert_eq!(c.rtc_regs[3..], [0x00, 0x80]);
        c.add_seconds_to_rtc(86_400);
        assert_eq!(c.rtc_regs[3..], [0x01, 0x80]);

        c.write_rom(0x0000, 0x0A);
        c.write_rom(0x4000, 0x0C);
        c.write_ram(0xA000, 0xFF);
        assert_eq!(c.rtc_regs[4], 0xC1);
        c.write_ram(0xA000, 0x40);
        assert_eq!(c.rtc_regs[4], 0x40);
        c.write_rom(0x4000, 0x08);
        c.write_ram(0xA000, 0xFF);
        assert_eq!(c.rtc_regs[0], 0x3F);
    }
}
