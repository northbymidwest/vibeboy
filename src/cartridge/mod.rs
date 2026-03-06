/// Cartridge abstraction — ROM-only, MBC1, MBC2, MBC3 (with RTC), MBC5.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub trait Cartridge: Send {
    fn read_rom(&self, addr: u16) -> u8;
    fn write_rom(&mut self, addr: u16, val: u8);
    fn read_ram(&self, addr: u16) -> u8;
    fn write_ram(&mut self, addr: u16, val: u8);
    fn has_battery(&self) -> bool { false }
    fn ram_data(&self) -> &[u8] { &[] }
    /// Returns save data (may include extra metadata like RTC state).
    fn save_data(&self) -> Vec<u8> { self.ram_data().to_vec() }
    fn load_ram(&mut self, _data: &[u8]) {}
}

// ── ROM-only ──────────────────────────────────────────────────────────────────

pub struct RomOnly {
    rom: Vec<u8>,
}

impl RomOnly {
    fn new(rom: Vec<u8>) -> Self { RomOnly { rom } }
}

impl Cartridge for RomOnly {
    fn read_rom(&self, addr: u16) -> u8 {
        self.rom.get(addr as usize).copied().unwrap_or(0xFF)
    }
    fn write_rom(&mut self, _addr: u16, _val: u8) {}
    fn read_ram(&self, _addr: u16) -> u8 { 0xFF }
    fn write_ram(&mut self, _addr: u16, _val: u8) {}
}

// ── MBC1 ──────────────────────────────────────────────────────────────────────

pub struct Mbc1 {
    rom: Vec<u8>,
    ram: Vec<u8>,
    rom_bank: usize,
    ram_bank: usize,
    ram_enabled: bool,
    /// 0 = ROM banking mode, 1 = RAM banking mode
    banking_mode: u8,
    /// Upper 2 bits (affect bank 0 in mode 1, and RAM bank)
    upper: usize,
    battery: bool,
}

impl Mbc1 {
    fn new(rom: Vec<u8>, ram_size: usize, battery: bool) -> Self {
        Mbc1 {
            rom,
            ram: vec![0u8; ram_size.max(0x2000)],
            rom_bank: 1,
            ram_bank: 0,
            ram_enabled: false,
            banking_mode: 0,
            upper: 0,
            battery,
        }
    }
}

impl Cartridge for Mbc1 {
    fn read_rom(&self, addr: u16) -> u8 {
        let idx = match addr {
            0x0000..=0x3FFF => {
                if self.banking_mode == 1 {
                    (self.upper << 19) | (addr as usize)
                } else {
                    addr as usize
                }
            }
            0x4000..=0x7FFF => {
                let bank = (self.upper << 5) | self.rom_bank;
                bank * 0x4000 + (addr as usize - 0x4000)
            }
            _ => return 0xFF,
        };
        self.rom.get(idx % self.rom.len().max(1)).copied().unwrap_or(0xFF)
    }

    fn write_rom(&mut self, addr: u16, val: u8) {
        match addr {
            0x0000..=0x1FFF => self.ram_enabled = val & 0x0F == 0x0A,
            0x2000..=0x3FFF => {
                let b = (val & 0x1F) as usize;
                self.rom_bank = if b == 0 { 1 } else { b };
            }
            0x4000..=0x5FFF => {
                self.upper = (val & 0x03) as usize;
                if self.banking_mode == 1 {
                    self.ram_bank = self.upper;
                }
            }
            0x6000..=0x7FFF => {
                self.banking_mode = val & 0x01;
                if self.banking_mode == 0 {
                    self.ram_bank = 0;
                } else {
                    self.ram_bank = self.upper;
                }
            }
            _ => {}
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        if !self.ram_enabled { return 0xFF; }
        let idx = self.ram_bank * 0x2000 + (addr as usize - 0xA000);
        self.ram.get(idx % self.ram.len().max(1)).copied().unwrap_or(0xFF)
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        if !self.ram_enabled { return; }
        let idx = self.ram_bank * 0x2000 + (addr as usize - 0xA000);
        let len = self.ram.len().max(1);
        let i = idx % len;
        self.ram[i] = val;
    }

    fn has_battery(&self) -> bool { self.battery }
    fn ram_data(&self) -> &[u8] { &self.ram }
    fn load_ram(&mut self, data: &[u8]) {
        let len = self.ram.len().min(data.len());
        self.ram[..len].copy_from_slice(&data[..len]);
    }
}

// ── MBC2 ──────────────────────────────────────────────────────────────────────

pub struct Mbc2 {
    rom: Vec<u8>,
    ram: [u8; 512], // 512 × 4-bit values
    rom_bank: usize, // 1-15
    ram_enabled: bool,
    battery: bool,
}

impl Mbc2 {
    fn new(rom: Vec<u8>, battery: bool) -> Self {
        Mbc2 { rom, ram: [0u8; 512], rom_bank: 1, ram_enabled: false, battery }
    }
}

impl Cartridge for Mbc2 {
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
            0x0000..=0x3FFF => {
                if addr & 0x0100 == 0 {
                    self.ram_enabled = (val & 0x0F) == 0x0A;
                } else {
                    let b = (val & 0x0F) as usize;
                    self.rom_bank = if b == 0 { 1 } else { b };
                }
            }
            _ => {}
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        if !self.ram_enabled { return 0xFF; }
        self.ram[(addr as usize) & 0x1FF] | 0xF0
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        if !self.ram_enabled { return; }
        self.ram[(addr as usize) & 0x1FF] = val & 0x0F;
    }

    fn has_battery(&self) -> bool { self.battery }
    fn ram_data(&self) -> &[u8] { &self.ram }
    fn load_ram(&mut self, data: &[u8]) {
        let len = self.ram.len().min(data.len());
        self.ram[..len].copy_from_slice(&data[..len]);
    }
}

// ── MBC3 ──────────────────────────────────────────────────────────────────────

pub struct Mbc3 {
    rom: Vec<u8>,
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
    rtc_base: Instant,     // last time RTC was updated
}

impl Mbc3 {
    fn new(rom: Vec<u8>, ram_size: usize, battery: bool, has_rtc: bool) -> Self {
        // MBC30 detection: ROM > 2MB or RAM > 32KB (only Pokemon Crystal JP)
        let mbc30 = rom.len() > 0x200000 || ram_size > 0x8000;
        if mbc30 {
            log::info!("MBC30 detected (ROM={}KB, RAM={}KB)", rom.len() / 1024, ram_size / 1024);
        }
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
            rtc_base: Instant::now(),
        }
    }

    /// Advance RTC registers by elapsed wall-clock time since last update.
    fn advance_rtc(&mut self) {
        // Don't advance if halted (DH bit 6)
        if self.rtc_regs[4] & 0x40 != 0 { return; }

        let elapsed = self.rtc_base.elapsed().as_secs();
        self.rtc_base = Instant::now();
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
                    self.rtc_base = Instant::now();
                }
                self.rtc_regs[reg] = val;
                self.rtc_base = Instant::now();
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
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;
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
                let now_ts = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64;
                let elapsed = (now_ts - saved_ts).max(0) as u64;
                if elapsed > 0 && self.rtc_regs[4] & 0x40 == 0 {
                    self.add_seconds_to_rtc(elapsed);
                }
            }
            self.rtc_base = Instant::now();
        }
    }
}

// ── MBC5 ──────────────────────────────────────────────────────────────────────

pub struct Mbc5 {
    rom: Vec<u8>,
    ram: Vec<u8>,
    rom_bank: usize, // 9-bit bank number
    ram_bank: usize,
    ram_enabled: bool,
    battery: bool,
}

impl Mbc5 {
    fn new(rom: Vec<u8>, ram_size: usize, battery: bool) -> Self {
        Mbc5 {
            rom,
            ram: vec![0u8; ram_size.max(0x2000)],
            rom_bank: 1,
            ram_bank: 0,
            ram_enabled: false,
            battery,
        }
    }
}

impl Cartridge for Mbc5 {
    fn read_rom(&self, addr: u16) -> u8 {
        let idx = match addr {
            0x0000..=0x3FFF => addr as usize,
            0x4000..=0x7FFF => self.rom_bank * 0x4000 + (addr as usize - 0x4000),
            _ => return 0xFF,
        };
        self.rom.get(idx).copied().unwrap_or(0xFF)
    }

    fn write_rom(&mut self, addr: u16, val: u8) {
        match addr {
            0x0000..=0x1FFF => self.ram_enabled = val & 0x0F == 0x0A,
            0x2000..=0x2FFF => self.rom_bank = (self.rom_bank & 0x100) | (val as usize),
            0x3000..=0x3FFF => self.rom_bank = (self.rom_bank & 0xFF) | (((val & 0x01) as usize) << 8),
            0x4000..=0x5FFF => self.ram_bank = (val & 0x0F) as usize,
            _ => {}
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        if !self.ram_enabled { return 0xFF; }
        let idx = self.ram_bank * 0x2000 + (addr as usize - 0xA000);
        self.ram.get(idx).copied().unwrap_or(0xFF)
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        if !self.ram_enabled { return; }
        let idx = self.ram_bank * 0x2000 + (addr as usize - 0xA000);
        if let Some(b) = self.ram.get_mut(idx) { *b = val; }
    }

    fn has_battery(&self) -> bool { self.battery }
    fn ram_data(&self) -> &[u8] { &self.ram }
    fn load_ram(&mut self, data: &[u8]) {
        let len = self.ram.len().min(data.len());
        self.ram[..len].copy_from_slice(&data[..len]);
    }
}

// ── Factory ───────────────────────────────────────────────────────────────────

/// Construct the appropriate cartridge from a ROM image.
pub fn make_cartridge(rom: Vec<u8>) -> Box<dyn Cartridge> {
    let cart_type = rom.get(0x0147).copied().unwrap_or(0);
    let ram_size: usize = match rom.get(0x0149).copied().unwrap_or(0) {
        0x01 => 0x0800,
        0x02 => 0x2000,
        0x03 => 0x8000,
        0x04 => 0x20000,
        0x05 => 0x10000,
        _    => 0,
    };

    log::info!(
        "Cart type={:#04X} title={} ram_size={:#X}",
        cart_type,
        rom.get(0x0134..0x0143)
            .and_then(|s| std::str::from_utf8(s).ok())
            .unwrap_or("?")
            .trim_matches('\0'),
        ram_size
    );

    match cart_type {
        0x00 => Box::new(RomOnly::new(rom)),
        0x01..=0x03 => {
            let battery = cart_type == 0x03;
            Box::new(Mbc1::new(rom, ram_size, battery))
        }
        0x05 | 0x06 => {
            let battery = cart_type == 0x06;
            Box::new(Mbc2::new(rom, battery))
        }
        0x0F..=0x13 => {
            let battery = matches!(cart_type, 0x0F | 0x10 | 0x13);
            let has_rtc = matches!(cart_type, 0x0F | 0x10);
            Box::new(Mbc3::new(rom, ram_size, battery, has_rtc))
        }
        0x19..=0x1E => {
            let battery = matches!(cart_type, 0x1B | 0x1E);
            Box::new(Mbc5::new(rom, ram_size, battery))
        }
        other => {
            log::warn!("Unsupported cart type {:#04X}, using ROM-only", other);
            Box::new(RomOnly::new(rom))
        }
    }
}
