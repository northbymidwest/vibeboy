use std::sync::Arc;
use super::Cartridge;
use crate::clock::Clock;

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
    pub(super) fn new(rom: Arc<[u8]>, ram_size: usize, battery: bool, has_rtc: bool, clock: Arc<dyn Clock>) -> Self {
        // MBC30 detection: ROM > 2MB or RAM > 32KB (only Pokemon Crystal JP)
        let mbc30 = rom.len() > 0x200000 || ram_size > 0x8000;
        if mbc30 {
            log::info!("MBC30 detected (ROM={}KB, RAM={}KB)", rom.len() / 1024, ram_size / 1024);
        }
        let rtc_last_secs = clock.now_secs();
        Mbc3 {
            rom,
            ram: vec![0u8; ram_size.max(0x2000)],
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
        if self.rtc_regs[4] & 0x40 != 0 { return; }

        let now = self.clock.now_secs();
        let elapsed = now.saturating_sub(self.rtc_last_secs);
        self.rtc_last_secs = now;
        if elapsed == 0 { return; }

        self.add_seconds_to_rtc(elapsed);
    }

    fn add_seconds_to_rtc(&mut self, seconds: u64) {
        let mut secs = self.rtc_regs[0] as u64 + seconds;
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
        // Preserve halt bit, set day bit 8, set carry if >511
        let carry = if days > 511 { 0x80 } else { 0 };
        self.rtc_regs[4] = (self.rtc_regs[4] & 0x40) | ((days >> 8) as u8 & 0x01) | carry;
    }
}

impl Cartridge for Mbc3 {
    fn read_rom(&self, addr: u16) -> u8 {
        let idx = match addr {
            0x0000..=0x3FFF => addr as usize,
            0x4000..=0x7FFF => self.rom_bank * 0x4000 + (addr as usize - 0x4000),
            _ => return 0xFF,
        };
        self.rom.get(idx % self.rom.len().max(1)).copied().unwrap_or(0xFF)
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
            0x6000..=0x7FFF => {
                if self.has_rtc {
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
            }
            _ => {}
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        if !self.ram_enabled { return 0xFF; }
        let max_ram_bank = if self.mbc30 { 0x07 } else { 0x03 };
        match self.ram_bank {
            b if b <= max_ram_bank => {
                let idx = b * 0x2000 + (addr as usize - 0xA000);
                self.ram.get(idx % self.ram.len().max(1)).copied().unwrap_or(0xFF)
            }
            0x08..=0x0C if self.has_rtc => self.rtc_latched[self.ram_bank - 0x08],
            _ => 0xFF,
        }
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        if !self.ram_enabled { return; }
        let max_ram_bank = if self.mbc30 { 0x07 } else { 0x03 };
        match self.ram_bank {
            b if b <= max_ram_bank => {
                let idx = b * 0x2000 + (addr as usize - 0xA000);
                let len = self.ram.len().max(1);
                self.ram[idx % len] = val;
            }
            0x08..=0x0C if self.has_rtc => {
                let reg = self.ram_bank - 0x08;
                // If unhalting (clearing halt bit), reset base so time counts from now
                if reg == 4 && self.rtc_regs[4] & 0x40 != 0 && val & 0x40 == 0 {
                    self.rtc_last_secs = self.clock.now_secs();
                }
                // Advance RTC before overwriting registers so accumulated time isn't lost
                self.advance_rtc();
                self.rtc_regs[reg] = val;
            }
            _ => {}
        }
    }

    fn has_battery(&self) -> bool { self.battery }

    fn ram_data(&self) -> &[u8] { &self.ram }

    fn save_data(&self) -> Vec<u8> {
        let mut data = self.ram.clone();
        if self.has_rtc {
            let mut footer = [0u8; 48];
            footer[..5].copy_from_slice(&self.rtc_regs);
            footer[5..10].copy_from_slice(&self.rtc_latched);
            let ts = self.clock.unix_timestamp_secs() as i64;
            footer[20..28].copy_from_slice(&ts.to_le_bytes());
            data.extend_from_slice(&footer);
        }
        data
    }

    fn load_ram(&mut self, data: &[u8]) {
        let ram_len = self.ram.len();
        let copy_len = ram_len.min(data.len());
        self.ram[..copy_len].copy_from_slice(&data[..copy_len]);

        // Load RTC state from 48-byte footer after RAM data
        if self.has_rtc && data.len() >= ram_len + 48 {
            let rtc = &data[ram_len..];
            for i in 0..5 {
                self.rtc_regs[i] = rtc[i];
            }
            for i in 0..5 {
                self.rtc_latched[i] = rtc[5 + i];
            }
            // Bytes 20-27: unix timestamp of last save (i64 LE)
            if rtc.len() >= 28 {
                let mut ts_bytes = [0u8; 8];
                ts_bytes.copy_from_slice(&rtc[20..28]);
                let saved_ts = i64::from_le_bytes(ts_bytes);
                let now_ts = self.clock.unix_timestamp_secs() as i64;
                let elapsed = (now_ts - saved_ts).max(0) as u64;
                if elapsed > 0 && self.rtc_regs[4] & 0x40 == 0 {
                    self.add_seconds_to_rtc(elapsed);
                }
            }
            self.rtc_last_secs = self.clock.now_secs();
        }
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
        if d.len() < 20 { return; }
        self.rom_bank = u32::from_le_bytes([d[0],d[1],d[2],d[3]]) as usize;
        self.ram_bank = u32::from_le_bytes([d[4],d[5],d[6],d[7]]) as usize;
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
