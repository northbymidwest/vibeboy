/// Cartridge abstraction — all known Game Boy mappers.

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
    /// Returns true if this cartridge has a camera sensor (Pocket Camera).
    fn has_camera(&self) -> bool { false }
    /// Feed a 128×112 grayscale image from a webcam into the camera sensor.
    fn set_camera_image(&mut self, _grayscale: &[u8; 128 * 112]) {}
    /// Returns true if this cartridge has an accelerometer (MBC7).
    fn has_accelerometer(&self) -> bool { false }
    /// Feed accelerometer values in MBC7 u16 format (center = 0x81D0).
    fn set_accelerometer(&mut self, _x: u16, _y: u16) {}
    /// Snapshot mutable cartridge state (registers + RAM, not ROM) for save states / rewind.
    fn snapshot_state(&self) -> Vec<u8> { Vec::new() }
    /// Restore mutable cartridge state from a previous snapshot.
    fn restore_state(&mut self, _data: &[u8]) {}
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
    fn snapshot_state(&self) -> Vec<u8> {
        let mut s = Vec::new();
        s.extend_from_slice(&(self.rom_bank as u32).to_le_bytes());
        s.extend_from_slice(&(self.ram_bank as u32).to_le_bytes());
        s.push(self.ram_enabled as u8);
        s.push(self.banking_mode);
        s.extend_from_slice(&self.ram);
        s
    }
    fn restore_state(&mut self, d: &[u8]) {
        if d.len() < 10 { return; }
        self.rom_bank = u32::from_le_bytes([d[0],d[1],d[2],d[3]]) as usize;
        self.ram_bank = u32::from_le_bytes([d[4],d[5],d[6],d[7]]) as usize;
        self.ram_enabled = d[8] != 0;
        self.banking_mode = d[9];
        let ram = &d[10..];
        let len = self.ram.len().min(ram.len());
        self.ram[..len].copy_from_slice(&ram[..len]);
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
    fn snapshot_state(&self) -> Vec<u8> {
        let mut s = Vec::new();
        s.extend_from_slice(&(self.rom_bank as u32).to_le_bytes());
        s.push(self.ram_enabled as u8);
        s.extend_from_slice(&self.ram);
        s
    }
    fn restore_state(&mut self, d: &[u8]) {
        if d.len() < 5 { return; }
        self.rom_bank = u32::from_le_bytes([d[0],d[1],d[2],d[3]]) as usize;
        self.ram_enabled = d[4] != 0;
        let ram = &d[5..];
        let len = self.ram.len().min(ram.len());
        self.ram[..len].copy_from_slice(&ram[..len]);
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
        self.rtc_base = Instant::now();
        let ram = &d[20..];
        let len = self.ram.len().min(ram.len());
        self.ram[..len].copy_from_slice(&ram[..len]);
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
    fn snapshot_state(&self) -> Vec<u8> {
        let mut s = Vec::new();
        s.extend_from_slice(&(self.rom_bank as u32).to_le_bytes());
        s.extend_from_slice(&(self.ram_bank as u32).to_le_bytes());
        s.push(self.ram_enabled as u8);
        s.extend_from_slice(&self.ram);
        s
    }
    fn restore_state(&mut self, d: &[u8]) {
        if d.len() < 9 { return; }
        self.rom_bank = u32::from_le_bytes([d[0],d[1],d[2],d[3]]) as usize;
        self.ram_bank = u32::from_le_bytes([d[4],d[5],d[6],d[7]]) as usize;
        self.ram_enabled = d[8] != 0;
        let ram = &d[9..];
        let len = self.ram.len().min(ram.len());
        self.ram[..len].copy_from_slice(&ram[..len]);
    }
}

// ── ROM+RAM ──────────────────────────────────────────────────────────────────

pub struct RomRam {
    rom: Vec<u8>,
    ram: Vec<u8>,
    battery: bool,
}

impl RomRam {
    fn new(rom: Vec<u8>, battery: bool) -> Self {
        RomRam { rom, ram: vec![0u8; 0x2000], battery }
    }
}

impl Cartridge for RomRam {
    fn read_rom(&self, addr: u16) -> u8 {
        self.rom.get(addr as usize).copied().unwrap_or(0xFF)
    }
    fn write_rom(&mut self, _addr: u16, _val: u8) {}
    fn read_ram(&self, addr: u16) -> u8 {
        self.ram[(addr as usize - 0xA000) & 0x1FFF]
    }
    fn write_ram(&mut self, addr: u16, val: u8) {
        self.ram[(addr as usize - 0xA000) & 0x1FFF] = val;
    }
    fn has_battery(&self) -> bool { self.battery }
    fn ram_data(&self) -> &[u8] { &self.ram }
    fn load_ram(&mut self, data: &[u8]) {
        let len = self.ram.len().min(data.len());
        self.ram[..len].copy_from_slice(&data[..len]);
    }
    fn snapshot_state(&self) -> Vec<u8> { self.ram.clone() }
    fn restore_state(&mut self, d: &[u8]) {
        let len = self.ram.len().min(d.len());
        self.ram[..len].copy_from_slice(&d[..len]);
    }
}

// ── MBC6 ─────────────────────────────────────────────────────────────────────

pub struct Mbc6 {
    rom: Vec<u8>,
    ram: Vec<u8>,
    flash: Vec<u8>,
    rom_bank_a: usize,
    rom_bank_b: usize,
    ram_bank_a: usize,
    ram_bank_b: usize,
    ram_enabled_a: bool,
    ram_enabled_b: bool,
    flash_enabled: bool,
    flash_write_enabled: bool,
    bank_a_is_flash: bool,
    bank_b_is_flash: bool,
}

impl Mbc6 {
    fn new(rom: Vec<u8>, ram_size: usize) -> Self {
        Mbc6 {
            flash: vec![0xFF; 0x100000], // 1MB MX29F008TC
            rom,
            ram: vec![0u8; ram_size.max(0x2000)],
            rom_bank_a: 2,
            rom_bank_b: 3,
            ram_bank_a: 0,
            ram_bank_b: 0,
            ram_enabled_a: false,
            ram_enabled_b: false,
            flash_enabled: false,
            flash_write_enabled: false,
            bank_a_is_flash: false,
            bank_b_is_flash: false,
        }
    }
}

impl Cartridge for Mbc6 {
    fn read_rom(&self, addr: u16) -> u8 {
        match addr {
            0x0000..=0x3FFF => self.rom.get(addr as usize).copied().unwrap_or(0xFF),
            0x4000..=0x5FFF => {
                if self.bank_a_is_flash {
                    let idx = self.rom_bank_a * 0x2000 + (addr as usize - 0x4000);
                    self.flash.get(idx % self.flash.len()).copied().unwrap_or(0xFF)
                } else {
                    let idx = self.rom_bank_a * 0x2000 + (addr as usize - 0x4000);
                    self.rom.get(idx % self.rom.len().max(1)).copied().unwrap_or(0xFF)
                }
            }
            0x6000..=0x7FFF => {
                if self.bank_b_is_flash {
                    let idx = self.rom_bank_b * 0x2000 + (addr as usize - 0x6000);
                    self.flash.get(idx % self.flash.len()).copied().unwrap_or(0xFF)
                } else {
                    let idx = self.rom_bank_b * 0x2000 + (addr as usize - 0x6000);
                    self.rom.get(idx % self.rom.len().max(1)).copied().unwrap_or(0xFF)
                }
            }
            _ => 0xFF,
        }
    }

    fn write_rom(&mut self, addr: u16, val: u8) {
        match addr {
            0x0000..=0x03FF => self.ram_enabled_a = val == 0x0A,
            0x0400..=0x07FF => self.ram_enabled_b = val == 0x0A,
            0x0800..=0x0BFF => self.ram_bank_a = (val & 0x07) as usize,
            0x0C00..=0x0FFF => self.ram_bank_b = (val & 0x07) as usize,
            0x1000 => self.flash_enabled = val == 0x01,
            0x1001 => self.flash_write_enabled = val == 0x01,
            0x2000..=0x27FF => self.rom_bank_a = (val as usize) & 0x7F,
            0x2800..=0x2FFF => self.bank_a_is_flash = val == 0x08,
            0x3000..=0x37FF => self.rom_bank_b = (val as usize) & 0x7F,
            0x3800..=0x3FFF => self.bank_b_is_flash = val == 0x08,
            _ => {}
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        match addr {
            0xA000..=0xAFFF => {
                if !self.ram_enabled_a { return 0xFF; }
                let idx = self.ram_bank_a * 0x1000 + (addr as usize - 0xA000);
                self.ram.get(idx % self.ram.len().max(1)).copied().unwrap_or(0xFF)
            }
            0xB000..=0xBFFF => {
                if !self.ram_enabled_b { return 0xFF; }
                let idx = self.ram_bank_b * 0x1000 + (addr as usize - 0xB000);
                self.ram.get(idx % self.ram.len().max(1)).copied().unwrap_or(0xFF)
            }
            _ => 0xFF,
        }
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        match addr {
            0xA000..=0xAFFF => {
                if !self.ram_enabled_a { return; }
                let idx = self.ram_bank_a * 0x1000 + (addr as usize - 0xA000);
                let len = self.ram.len().max(1);
                self.ram[idx % len] = val;
            }
            0xB000..=0xBFFF => {
                if !self.ram_enabled_b { return; }
                let idx = self.ram_bank_b * 0x1000 + (addr as usize - 0xB000);
                let len = self.ram.len().max(1);
                self.ram[idx % len] = val;
            }
            _ => {}
        }
    }

    fn has_battery(&self) -> bool { true }
    fn ram_data(&self) -> &[u8] { &self.ram }

    fn save_data(&self) -> Vec<u8> {
        let mut data = self.ram.clone();
        data.extend_from_slice(&self.flash);
        data
    }

    fn load_ram(&mut self, data: &[u8]) {
        let ram_len = self.ram.len();
        let copy_len = ram_len.min(data.len());
        self.ram[..copy_len].copy_from_slice(&data[..copy_len]);
        if data.len() > ram_len {
            let flash_len = self.flash.len().min(data.len() - ram_len);
            self.flash[..flash_len].copy_from_slice(&data[ram_len..ram_len + flash_len]);
        }
    }
    fn snapshot_state(&self) -> Vec<u8> {
        let mut s = Vec::new();
        s.extend_from_slice(&(self.rom_bank_a as u32).to_le_bytes());
        s.extend_from_slice(&(self.rom_bank_b as u32).to_le_bytes());
        s.extend_from_slice(&(self.ram_bank_a as u32).to_le_bytes());
        s.extend_from_slice(&(self.ram_bank_b as u32).to_le_bytes());
        s.push(self.ram_enabled_a as u8);
        s.push(self.ram_enabled_b as u8);
        s.push(self.flash_enabled as u8);
        s.push(self.flash_write_enabled as u8);
        s.push(self.bank_a_is_flash as u8);
        s.push(self.bank_b_is_flash as u8);
        s.extend_from_slice(&self.ram);
        s.extend_from_slice(&self.flash);
        s
    }
    fn restore_state(&mut self, d: &[u8]) {
        if d.len() < 22 { return; }
        self.rom_bank_a = u32::from_le_bytes([d[0],d[1],d[2],d[3]]) as usize;
        self.rom_bank_b = u32::from_le_bytes([d[4],d[5],d[6],d[7]]) as usize;
        self.ram_bank_a = u32::from_le_bytes([d[8],d[9],d[10],d[11]]) as usize;
        self.ram_bank_b = u32::from_le_bytes([d[12],d[13],d[14],d[15]]) as usize;
        self.ram_enabled_a = d[16] != 0;
        self.ram_enabled_b = d[17] != 0;
        self.flash_enabled = d[18] != 0;
        self.flash_write_enabled = d[19] != 0;
        self.bank_a_is_flash = d[20] != 0;
        self.bank_b_is_flash = d[21] != 0;
        let rest = &d[22..];
        let ram_len = self.ram.len().min(rest.len());
        self.ram[..ram_len].copy_from_slice(&rest[..ram_len]);
        if rest.len() > ram_len {
            let flash = &rest[ram_len..];
            let flash_len = self.flash.len().min(flash.len());
            self.flash[..flash_len].copy_from_slice(&flash[..flash_len]);
        }
    }
}

// ── MBC7 ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum EepromState {
    Idle,
    Command,
    Address,
    ReadData,
    WriteData,
}

pub struct Mbc7 {
    rom: Vec<u8>,
    rom_bank: usize,
    enable_a: bool,
    enable_b: bool,
    accel_latched: bool,
    accel_x: u16,
    accel_y: u16,
    /// Live accelerometer values set by external sensor; latched on write 0xAA.
    sensor_x: u16,
    sensor_y: u16,
    // EEPROM 93LC56: 128 × 16-bit words
    eeprom: [u16; 128],
    eeprom_cs: bool,
    eeprom_clk: bool,
    eeprom_di: bool,
    eeprom_do: bool,
    eeprom_state: EepromState,
    eeprom_cmd: u8,
    eeprom_addr: u8,
    eeprom_shift: u16,
    eeprom_bit_count: u8,
    eeprom_write_enable: bool,
}

impl Mbc7 {
    fn new(rom: Vec<u8>) -> Self {
        Mbc7 {
            rom,
            rom_bank: 1,
            enable_a: false,
            enable_b: false,
            accel_latched: false,
            accel_x: 0x8000,
            accel_y: 0x8000,
            sensor_x: 0x81D0,
            sensor_y: 0x81D0,
            eeprom: [0xFFFF; 128],
            eeprom_cs: false,
            eeprom_clk: false,
            eeprom_di: false,
            eeprom_do: true,
            eeprom_state: EepromState::Idle,
            eeprom_cmd: 0,
            eeprom_addr: 0,
            eeprom_shift: 0,
            eeprom_bit_count: 0,
            eeprom_write_enable: false,
        }
    }

    fn eeprom_clock_rise(&mut self) {
        match self.eeprom_state {
            EepromState::Idle => {
                if self.eeprom_di {
                    // Start bit received
                    self.eeprom_state = EepromState::Command;
                    self.eeprom_cmd = 0;
                    self.eeprom_bit_count = 0;
                }
            }
            EepromState::Command => {
                self.eeprom_cmd = (self.eeprom_cmd << 1) | (self.eeprom_di as u8);
                self.eeprom_bit_count += 1;
                if self.eeprom_bit_count == 2 {
                    self.eeprom_state = EepromState::Address;
                    self.eeprom_addr = 0;
                    self.eeprom_bit_count = 0;
                }
            }
            EepromState::Address => {
                self.eeprom_addr = (self.eeprom_addr << 1) | (self.eeprom_di as u8);
                self.eeprom_bit_count += 1;
                if self.eeprom_bit_count == 7 {
                    self.eeprom_bit_count = 0;
                    match self.eeprom_cmd {
                        0b10 => {
                            // READ
                            self.eeprom_shift = self.eeprom[self.eeprom_addr as usize & 0x7F];
                            self.eeprom_state = EepromState::ReadData;
                            self.eeprom_do = false; // dummy 0 bit before data
                        }
                        0b01 => {
                            // WRITE
                            if self.eeprom_write_enable {
                                self.eeprom_shift = 0;
                                self.eeprom_state = EepromState::WriteData;
                            } else {
                                self.eeprom_state = EepromState::Idle;
                            }
                        }
                        0b11 => {
                            // ERASE
                            if self.eeprom_write_enable {
                                self.eeprom[self.eeprom_addr as usize & 0x7F] = 0xFFFF;
                            }
                            self.eeprom_do = true;
                            self.eeprom_state = EepromState::Idle;
                        }
                        0b00 => {
                            // Special: top 2 bits of address select sub-command
                            match self.eeprom_addr >> 5 {
                                0b00 => self.eeprom_write_enable = false, // EWDS
                                0b01 => {
                                    // WRAL
                                    if self.eeprom_write_enable {
                                        self.eeprom_shift = 0;
                                        self.eeprom_state = EepromState::WriteData;
                                    } else {
                                        self.eeprom_state = EepromState::Idle;
                                    }
                                }
                                0b10 => {
                                    // ERAL
                                    if self.eeprom_write_enable {
                                        self.eeprom = [0xFFFF; 128];
                                    }
                                    self.eeprom_state = EepromState::Idle;
                                }
                                0b11 | _ => {
                                    self.eeprom_write_enable = true; // EWEN
                                    self.eeprom_state = EepromState::Idle;
                                }
                            }
                        }
                        _ => self.eeprom_state = EepromState::Idle,
                    }
                }
            }
            EepromState::ReadData => {
                self.eeprom_do = (self.eeprom_shift >> 15) != 0;
                self.eeprom_shift <<= 1;
                self.eeprom_bit_count += 1;
                if self.eeprom_bit_count == 16 {
                    self.eeprom_bit_count = 0;
                    // Auto-increment address for sequential read
                    self.eeprom_addr = (self.eeprom_addr + 1) & 0x7F;
                    self.eeprom_shift = self.eeprom[self.eeprom_addr as usize];
                }
            }
            EepromState::WriteData => {
                self.eeprom_shift = (self.eeprom_shift << 1) | (self.eeprom_di as u16);
                self.eeprom_bit_count += 1;
                if self.eeprom_bit_count == 16 {
                    if self.eeprom_cmd == 0b01 {
                        // WRITE single
                        self.eeprom[self.eeprom_addr as usize & 0x7F] = self.eeprom_shift;
                    } else {
                        // WRAL
                        self.eeprom = [self.eeprom_shift; 128];
                    }
                    self.eeprom_do = true;
                    self.eeprom_state = EepromState::Idle;
                    self.eeprom_bit_count = 0;
                }
            }
        }
    }
}

impl Cartridge for Mbc7 {
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
            0x0000..=0x1FFF => self.enable_a = val == 0x0A,
            0x2000..=0x3FFF => {
                self.rom_bank = (val as usize) & 0xFF;
                if self.rom_bank == 0 { self.rom_bank = 1; }
            }
            0x4000..=0x5FFF => self.enable_b = val == 0x40,
            _ => {}
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        if !self.enable_a || !self.enable_b { return 0xFF; }
        match (addr >> 4) & 0x0F {
            0x0 | 0x1 => 0, // write-only
            0x2 => self.accel_x as u8,
            0x3 => (self.accel_x >> 8) as u8,
            0x4 => self.accel_y as u8,
            0x5 => (self.accel_y >> 8) as u8,
            0x6 => 0x00,
            0x7 => 0xFF,
            0x8 => (self.eeprom_do as u8) | ((self.eeprom_di as u8) << 1) |
                   ((self.eeprom_clk as u8) << 6) | ((self.eeprom_cs as u8) << 7),
            _ => 0xFF,
        }
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        if !self.enable_a || !self.enable_b { return; }
        match (addr >> 4) & 0x0F {
            0x0 => {
                if val == 0x55 {
                    self.accel_x = 0x8000;
                    self.accel_y = 0x8000;
                    self.accel_latched = false;
                }
            }
            0x1 => {
                if val == 0xAA && !self.accel_latched {
                    self.accel_x = self.sensor_x;
                    self.accel_y = self.sensor_y;
                    self.accel_latched = true;
                }
            }
            0x8 => {
                let new_cs = val & 0x80 != 0;
                let new_clk = val & 0x40 != 0;
                self.eeprom_di = val & 0x02 != 0;

                if !new_cs {
                    // CS low: reset
                    self.eeprom_state = EepromState::Idle;
                    self.eeprom_do = true;
                    self.eeprom_bit_count = 0;
                } else if new_clk && !self.eeprom_clk {
                    // Rising clock edge
                    self.eeprom_clock_rise();
                }

                self.eeprom_cs = new_cs;
                self.eeprom_clk = new_clk;
            }
            _ => {}
        }
    }

    fn has_battery(&self) -> bool { true }
    fn has_accelerometer(&self) -> bool { true }
    fn set_accelerometer(&mut self, x: u16, y: u16) {
        self.sensor_x = x;
        self.sensor_y = y;
    }

    fn save_data(&self) -> Vec<u8> {
        // Save EEPROM as 256 bytes (128 × 16-bit LE words)
        let mut data = vec![0u8; 256];
        for (i, &word) in self.eeprom.iter().enumerate() {
            data[i * 2] = word as u8;
            data[i * 2 + 1] = (word >> 8) as u8;
        }
        data
    }

    fn load_ram(&mut self, data: &[u8]) {
        let words = data.len().min(256) / 2;
        for i in 0..words {
            self.eeprom[i] = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        }
    }
    fn snapshot_state(&self) -> Vec<u8> {
        let mut s = Vec::new();
        s.extend_from_slice(&(self.rom_bank as u32).to_le_bytes());
        s.push(self.enable_a as u8);
        s.push(self.enable_b as u8);
        s.push(self.accel_latched as u8);
        s.extend_from_slice(&self.accel_x.to_le_bytes());
        s.extend_from_slice(&self.accel_y.to_le_bytes());
        // EEPROM: 128 words × 2 bytes
        for &w in &self.eeprom {
            s.extend_from_slice(&w.to_le_bytes());
        }
        s.push(self.eeprom_cs as u8);
        s.push(self.eeprom_clk as u8);
        s.push(self.eeprom_di as u8);
        s.push(self.eeprom_do as u8);
        s.push(self.eeprom_state as u8);
        s.push(self.eeprom_cmd);
        s.push(self.eeprom_addr);
        s.extend_from_slice(&self.eeprom_shift.to_le_bytes());
        s.push(self.eeprom_bit_count);
        s.push(self.eeprom_write_enable as u8);
        s
    }
    fn restore_state(&mut self, d: &[u8]) {
        if d.len() < 7 + 256 + 10 { return; }
        self.rom_bank = u32::from_le_bytes([d[0],d[1],d[2],d[3]]) as usize;
        self.enable_a = d[4] != 0;
        self.enable_b = d[5] != 0;
        self.accel_latched = d[6] != 0;
        self.accel_x = u16::from_le_bytes([d[7],d[8]]);
        self.accel_y = u16::from_le_bytes([d[9],d[10]]);
        let ee = &d[11..];
        for i in 0..128 {
            self.eeprom[i] = u16::from_le_bytes([ee[i*2], ee[i*2+1]]);
        }
        let r = &ee[256..];
        if r.len() >= 10 {
            self.eeprom_cs = r[0] != 0;
            self.eeprom_clk = r[1] != 0;
            self.eeprom_di = r[2] != 0;
            self.eeprom_do = r[3] != 0;
            self.eeprom_state = match r[4] {
                1 => EepromState::Command,
                2 => EepromState::Address,
                3 => EepromState::ReadData,
                4 => EepromState::WriteData,
                _ => EepromState::Idle,
            };
            self.eeprom_cmd = r[5];
            self.eeprom_addr = r[6];
            self.eeprom_shift = u16::from_le_bytes([r[7],r[8]]);
            self.eeprom_bit_count = r[9];
            if r.len() > 10 { self.eeprom_write_enable = r[10] != 0; }
        }
    }
}

// ── MMM01 ────────────────────────────────────────────────────────────────────

pub struct Mmm01 {
    rom: Vec<u8>,
    ram: Vec<u8>,
    battery: bool,
    mapped: bool,
    // Unmapped mode captures
    rom_base: usize,     // base ROM bank (from $2000/$4000 in unmapped mode)
    rom_bank_mask: usize, // per-game ROM bank mask
    ram_bank_mask: usize,
    // Mapped mode registers
    rom_bank: usize,
    ram_bank: usize,
    ram_enabled: bool,
    banking_mode: u8,
    upper: usize,
}

impl Mmm01 {
    fn new(rom: Vec<u8>, ram_size: usize, battery: bool) -> Self {
        Mmm01 {
            rom,
            ram: vec![0u8; ram_size.max(0x2000)],
            battery,
            mapped: false,
            rom_base: 0,
            rom_bank_mask: 0x1FF,
            ram_bank_mask: 0x03,
            rom_bank: 1,
            ram_bank: 0,
            ram_enabled: false,
            banking_mode: 0,
            upper: 0,
        }
    }

}

impl Cartridge for Mmm01 {
    fn read_rom(&self, addr: u16) -> u8 {
        if !self.mapped {
            // Unmapped: last 32KB of ROM
            let base = self.rom.len().saturating_sub(0x8000);
            let idx = base + (addr as usize);
            return self.rom.get(idx).copied().unwrap_or(0xFF);
        }
        let idx = match addr {
            0x0000..=0x3FFF => {
                let bank = self.rom_base;
                bank * 0x4000 + (addr as usize)
            }
            0x4000..=0x7FFF => {
                let bank = self.rom_base + (self.rom_bank & self.rom_bank_mask);
                bank * 0x4000 + (addr as usize - 0x4000)
            }
            _ => return 0xFF,
        };
        self.rom.get(idx % self.rom.len().max(1)).copied().unwrap_or(0xFF)
    }

    fn write_rom(&mut self, addr: u16, val: u8) {
        if !self.mapped {
            match addr {
                0x0000..=0x1FFF => {
                    if val & 0x40 != 0 {
                        // Lock into mapped mode
                        self.mapped = true;
                        log::info!("MMM01: locked into mapped mode, rom_base={}", self.rom_base);
                    }
                    self.ram_bank_mask = ((val >> 4) & 0x03) as usize;
                }
                0x2000..=0x3FFF => {
                    self.rom_base = (val as usize) & 0x7F;
                }
                0x4000..=0x5FFF => {
                    self.rom_bank_mask = ((val >> 1) & 0x1F) as usize;
                    if self.rom_bank_mask == 0 { self.rom_bank_mask = 0x1F; }
                }
                0x6000..=0x7FFF => {
                    self.banking_mode = val & 0x01;
                }
                _ => {}
            }
            return;
        }
        // Mapped mode: MBC1-like
        match addr {
            0x0000..=0x1FFF => self.ram_enabled = val & 0x0F == 0x0A,
            0x2000..=0x3FFF => {
                let b = (val & 0x1F) as usize;
                self.rom_bank = if b == 0 { 1 } else { b };
            }
            0x4000..=0x5FFF => {
                self.upper = (val & 0x03) as usize;
                if self.banking_mode == 1 {
                    self.ram_bank = self.upper & self.ram_bank_mask;
                }
            }
            0x6000..=0x7FFF => {
                self.banking_mode = val & 0x01;
                if self.banking_mode == 0 {
                    self.ram_bank = 0;
                } else {
                    self.ram_bank = self.upper & self.ram_bank_mask;
                }
            }
            _ => {}
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        if !self.ram_enabled || !self.mapped { return 0xFF; }
        let idx = self.ram_bank * 0x2000 + (addr as usize - 0xA000);
        self.ram.get(idx % self.ram.len().max(1)).copied().unwrap_or(0xFF)
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        if !self.ram_enabled || !self.mapped { return; }
        let idx = self.ram_bank * 0x2000 + (addr as usize - 0xA000);
        let len = self.ram.len().max(1);
        self.ram[idx % len] = val;
    }

    fn has_battery(&self) -> bool { self.battery }
    fn ram_data(&self) -> &[u8] { &self.ram }
    fn load_ram(&mut self, data: &[u8]) {
        let len = self.ram.len().min(data.len());
        self.ram[..len].copy_from_slice(&data[..len]);
    }
    fn snapshot_state(&self) -> Vec<u8> {
        let mut s = Vec::new();
        s.push(self.mapped as u8);
        s.extend_from_slice(&(self.rom_base as u32).to_le_bytes());
        s.extend_from_slice(&(self.rom_bank_mask as u32).to_le_bytes());
        s.extend_from_slice(&(self.ram_bank_mask as u32).to_le_bytes());
        s.extend_from_slice(&(self.rom_bank as u32).to_le_bytes());
        s.extend_from_slice(&(self.ram_bank as u32).to_le_bytes());
        s.push(self.ram_enabled as u8);
        s.push(self.banking_mode);
        s.extend_from_slice(&(self.upper as u32).to_le_bytes());
        s.extend_from_slice(&self.ram);
        s
    }
    fn restore_state(&mut self, d: &[u8]) {
        if d.len() < 27 { return; }
        self.mapped = d[0] != 0;
        self.rom_base = u32::from_le_bytes([d[1],d[2],d[3],d[4]]) as usize;
        self.rom_bank_mask = u32::from_le_bytes([d[5],d[6],d[7],d[8]]) as usize;
        self.ram_bank_mask = u32::from_le_bytes([d[9],d[10],d[11],d[12]]) as usize;
        self.rom_bank = u32::from_le_bytes([d[13],d[14],d[15],d[16]]) as usize;
        self.ram_bank = u32::from_le_bytes([d[17],d[18],d[19],d[20]]) as usize;
        self.ram_enabled = d[21] != 0;
        self.banking_mode = d[22];
        self.upper = u32::from_le_bytes([d[23],d[24],d[25],d[26]]) as usize;
        let ram = &d[27..];
        let len = self.ram.len().min(ram.len());
        self.ram[..len].copy_from_slice(&ram[..len]);
    }
}

// ── HuC1 ─────────────────────────────────────────────────────────────────────

pub struct HuC1 {
    rom: Vec<u8>,
    ram: Vec<u8>,
    rom_bank: usize,
    ram_bank: usize,
    ir_mode: bool,
    ir_led: bool,
}

impl HuC1 {
    fn new(rom: Vec<u8>, ram_size: usize) -> Self {
        HuC1 {
            rom,
            ram: vec![0u8; ram_size.max(0x2000)],
            rom_bank: 1,
            ram_bank: 0,
            ir_mode: false,
            ir_led: false,
        }
    }
}

impl Cartridge for HuC1 {
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
            0x0000..=0x1FFF => self.ir_mode = val == 0x0E,
            0x2000..=0x3FFF => {
                let b = (val & 0x3F) as usize;
                self.rom_bank = if b == 0 { 1 } else { b };
            }
            0x4000..=0x5FFF => self.ram_bank = (val & 0x03) as usize,
            _ => {}
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        if self.ir_mode {
            return 0xC0; // No IR signal received (stub)
        }
        let idx = self.ram_bank * 0x2000 + (addr as usize - 0xA000);
        self.ram.get(idx % self.ram.len().max(1)).copied().unwrap_or(0xFF)
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        if self.ir_mode {
            self.ir_led = val & 0x01 != 0;
            return;
        }
        let idx = self.ram_bank * 0x2000 + (addr as usize - 0xA000);
        let len = self.ram.len().max(1);
        self.ram[idx % len] = val;
    }

    fn has_battery(&self) -> bool { true }
    fn ram_data(&self) -> &[u8] { &self.ram }
    fn load_ram(&mut self, data: &[u8]) {
        let len = self.ram.len().min(data.len());
        self.ram[..len].copy_from_slice(&data[..len]);
    }
    fn snapshot_state(&self) -> Vec<u8> {
        let mut s = Vec::new();
        s.extend_from_slice(&(self.rom_bank as u32).to_le_bytes());
        s.extend_from_slice(&(self.ram_bank as u32).to_le_bytes());
        s.push(self.ir_mode as u8);
        s.push(self.ir_led as u8);
        s.extend_from_slice(&self.ram);
        s
    }
    fn restore_state(&mut self, d: &[u8]) {
        if d.len() < 10 { return; }
        self.rom_bank = u32::from_le_bytes([d[0],d[1],d[2],d[3]]) as usize;
        self.ram_bank = u32::from_le_bytes([d[4],d[5],d[6],d[7]]) as usize;
        self.ir_mode = d[8] != 0;
        self.ir_led = d[9] != 0;
        let ram = &d[10..];
        let len = self.ram.len().min(ram.len());
        self.ram[..len].copy_from_slice(&ram[..len]);
    }
}

// ── HuC3 ─────────────────────────────────────────────────────────────────────

pub struct HuC3 {
    rom: Vec<u8>,
    ram: Vec<u8>,
    rom_bank: usize,
    ram_bank: usize,
    mode: u8,           // $0000-$1FFF write selects mode
    rtc_mem: [u8; 128], // nybble-addressed RTC/internal memory
    rtc_addr: u8,       // current RTC address pointer
    rtc_data_out: u8,   // response nybble for reads
    rtc_minutes: u32,   // minutes in day (0-1439)
    rtc_days: u32,      // day counter
    rtc_base: Instant,
}

impl HuC3 {
    fn new(rom: Vec<u8>, ram_size: usize) -> Self {
        HuC3 {
            rom,
            ram: vec![0u8; ram_size.max(0x2000)],
            rom_bank: 1,
            ram_bank: 0,
            mode: 0,
            rtc_mem: [0; 128],
            rtc_addr: 0,
            rtc_data_out: 0,
            rtc_minutes: 0,
            rtc_days: 0,
            rtc_base: Instant::now(),
        }
    }

    fn advance_rtc(&mut self) {
        let elapsed = self.rtc_base.elapsed().as_secs();
        self.rtc_base = Instant::now();
        let elapsed_mins = elapsed / 60;
        if elapsed_mins == 0 { return; }
        self.rtc_minutes += elapsed_mins as u32;
        self.rtc_days += self.rtc_minutes / 1440;
        self.rtc_minutes %= 1440;
    }

    fn latch_time_to_mem(&mut self) {
        self.advance_rtc();
        let mins = self.rtc_minutes;
        let days = self.rtc_days;
        // Minutes at addrs $10-$12 (nybble-addressed, 12-bit BCD-ish)
        self.rtc_mem[0x10] = (mins & 0x0F) as u8;
        self.rtc_mem[0x11] = ((mins >> 4) & 0x0F) as u8;
        self.rtc_mem[0x12] = ((mins >> 8) & 0x0F) as u8;
        // Days at addrs $13-$15
        self.rtc_mem[0x13] = (days & 0x0F) as u8;
        self.rtc_mem[0x14] = ((days >> 4) & 0x0F) as u8;
        self.rtc_mem[0x15] = ((days >> 8) & 0x0F) as u8;
    }

    fn set_time_from_mem(&mut self) {
        self.rtc_minutes = (self.rtc_mem[0x10] as u32)
            | ((self.rtc_mem[0x11] as u32) << 4)
            | ((self.rtc_mem[0x12] as u32) << 8);
        self.rtc_days = (self.rtc_mem[0x13] as u32)
            | ((self.rtc_mem[0x14] as u32) << 4)
            | ((self.rtc_mem[0x15] as u32) << 8);
        self.rtc_base = Instant::now();
    }

    fn handle_rtc_cmd(&mut self, cmd_byte: u8) {
        let cmd = (cmd_byte >> 4) & 0x07;
        let arg = cmd_byte & 0x0F;
        match cmd {
            0x1 => {
                // Read + increment address
                let addr = (self.rtc_addr as usize) & 0x7F;
                self.rtc_data_out = self.rtc_mem[addr] & 0x0F;
                self.rtc_addr = self.rtc_addr.wrapping_add(1) & 0x7F;
            }
            0x3 => {
                // Write + increment address
                let addr = (self.rtc_addr as usize) & 0x7F;
                self.rtc_mem[addr] = arg & 0x0F;
                self.rtc_addr = self.rtc_addr.wrapping_add(1) & 0x7F;
            }
            0x4 => {
                // Set address low nybble
                self.rtc_addr = (self.rtc_addr & 0x70) | (arg & 0x0F);
            }
            0x5 => {
                // Set address high nybble
                self.rtc_addr = (self.rtc_addr & 0x0F) | ((arg & 0x07) << 4);
            }
            0x6 => {
                // Extended command
                match arg {
                    0x0 => self.latch_time_to_mem(),
                    0x1 => self.set_time_from_mem(),
                    0x2 => self.rtc_data_out = 0x01, // status: ready
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

impl Cartridge for HuC3 {
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
            0x0000..=0x1FFF => self.mode = val,
            0x2000..=0x3FFF => {
                let b = (val & 0x7F) as usize;
                self.rom_bank = if b == 0 { 1 } else { b };
            }
            0x4000..=0x5FFF => self.ram_bank = (val & 0x03) as usize,
            _ => {}
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        match self.mode {
            0x00 | 0x0A => {
                let idx = self.ram_bank * 0x2000 + (addr as usize - 0xA000);
                self.ram.get(idx % self.ram.len().max(1)).copied().unwrap_or(0xFF)
            }
            0x0C => {
                // RTC response read
                0x80 | (self.rtc_data_out & 0x0F)
            }
            0x0D => 0x01, // Semaphore: always ready
            0x0E => 0xC0, // IR stub: no signal
            _ => 0xFF,
        }
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        match self.mode {
            0x0A => {
                let idx = self.ram_bank * 0x2000 + (addr as usize - 0xA000);
                let len = self.ram.len().max(1);
                self.ram[idx % len] = val;
            }
            0x0B => {
                // RTC command write
                self.handle_rtc_cmd(val);
            }
            _ => {}
        }
    }

    fn has_battery(&self) -> bool { true }
    fn ram_data(&self) -> &[u8] { &self.ram }

    fn save_data(&self) -> Vec<u8> {
        let mut data = self.ram.clone();
        // Append RTC state: rtc_mem (128 bytes) + minutes (4 bytes LE) + days (4 bytes LE) + timestamp (8 bytes LE)
        data.extend_from_slice(&self.rtc_mem);
        data.extend_from_slice(&self.rtc_minutes.to_le_bytes());
        data.extend_from_slice(&self.rtc_days.to_le_bytes());
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        data.extend_from_slice(&ts.to_le_bytes());
        data
    }

    fn load_ram(&mut self, data: &[u8]) {
        let ram_len = self.ram.len();
        let copy_len = ram_len.min(data.len());
        self.ram[..copy_len].copy_from_slice(&data[..copy_len]);

        // Load RTC state
        if data.len() >= ram_len + 128 + 4 + 4 + 8 {
            let rtc_start = ram_len;
            self.rtc_mem.copy_from_slice(&data[rtc_start..rtc_start + 128]);
            let mut buf4 = [0u8; 4];
            buf4.copy_from_slice(&data[rtc_start + 128..rtc_start + 132]);
            self.rtc_minutes = u32::from_le_bytes(buf4);
            buf4.copy_from_slice(&data[rtc_start + 132..rtc_start + 136]);
            self.rtc_days = u32::from_le_bytes(buf4);
            let mut buf8 = [0u8; 8];
            buf8.copy_from_slice(&data[rtc_start + 136..rtc_start + 144]);
            let saved_ts = i64::from_le_bytes(buf8);
            let now_ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;
            let elapsed_mins = ((now_ts - saved_ts).max(0) as u64) / 60;
            self.rtc_minutes += elapsed_mins as u32;
            self.rtc_days += self.rtc_minutes / 1440;
            self.rtc_minutes %= 1440;
            self.rtc_base = Instant::now();
        }
    }
    fn snapshot_state(&self) -> Vec<u8> {
        let mut s = Vec::new();
        s.extend_from_slice(&(self.rom_bank as u32).to_le_bytes());
        s.extend_from_slice(&(self.ram_bank as u32).to_le_bytes());
        s.push(self.mode);
        s.extend_from_slice(&self.rtc_mem);
        s.push(self.rtc_addr);
        s.push(self.rtc_data_out);
        s.extend_from_slice(&self.rtc_minutes.to_le_bytes());
        s.extend_from_slice(&self.rtc_days.to_le_bytes());
        s.extend_from_slice(&self.ram);
        s
    }
    fn restore_state(&mut self, d: &[u8]) {
        if d.len() < 9 + 128 + 2 + 8 { return; }
        self.rom_bank = u32::from_le_bytes([d[0],d[1],d[2],d[3]]) as usize;
        self.ram_bank = u32::from_le_bytes([d[4],d[5],d[6],d[7]]) as usize;
        self.mode = d[8];
        self.rtc_mem.copy_from_slice(&d[9..137]);
        self.rtc_addr = d[137];
        self.rtc_data_out = d[138];
        self.rtc_minutes = u32::from_le_bytes([d[139],d[140],d[141],d[142]]);
        self.rtc_days = u32::from_le_bytes([d[143],d[144],d[145],d[146]]);
        self.rtc_base = Instant::now();
        let ram = &d[147..];
        let len = self.ram.len().min(ram.len());
        self.ram[..len].copy_from_slice(&ram[..len]);
    }
}

// ── TAMA5 ────────────────────────────────────────────────────────────────────

pub struct Tama5 {
    rom: Vec<u8>,
    rom_bank: usize,
    tama_ram: [u8; 32], // 32 nybbles internal RAM
    reg_select: u8,
    data_in_lo: u8,
    data_in_hi: u8,
    data_out_lo: u8,
    data_out_hi: u8,
    // RTC: TC8521AM — simplified
    rtc_regs: [u8; 52], // 4 pages × 13 nybble registers
    rtc_base: Instant,
    rtc_seconds: u32,
}

impl Tama5 {
    fn new(rom: Vec<u8>) -> Self {
        Tama5 {
            rom,
            rom_bank: 1,
            tama_ram: [0; 32],
            reg_select: 0,
            data_in_lo: 0,
            data_in_hi: 0,
            data_out_lo: 0,
            data_out_hi: 0,
            rtc_regs: [0; 52],
            rtc_base: Instant::now(),
            rtc_seconds: 0,
        }
    }

    fn advance_rtc(&mut self) {
        let elapsed = self.rtc_base.elapsed().as_secs() as u32;
        self.rtc_base = Instant::now();
        self.rtc_seconds += elapsed;
    }

    fn execute_command(&mut self) {
        let cmd = (self.data_in_hi as u16) << 4 | (self.data_in_lo as u16);
        let addr = (cmd & 0x1F) as usize;
        let cmd_type = cmd >> 5;

        match cmd_type {
            0x00 => {
                // RAM write: uses data from regs $04/$05
                let val = self.data_in_lo; // value was written to $04 before command
                if addr < 32 {
                    self.tama_ram[addr] = val & 0x0F;
                }
            }
            0x01 => {
                // RAM read
                if addr < 32 {
                    let val = self.tama_ram[addr] & 0x0F;
                    self.data_out_lo = val & 0x0F;
                    self.data_out_hi = 0;
                }
            }
            0x02..=0x03 => {
                // MCU commands (time get/set) — simplified
                self.advance_rtc();
                self.data_out_lo = 0;
                self.data_out_hi = 0;
            }
            0x04..=0x05 => {
                // RTC register access
                self.data_out_lo = 0;
                self.data_out_hi = 0;
            }
            _ => {}
        }
    }
}

impl Cartridge for Tama5 {
    fn read_rom(&self, addr: u16) -> u8 {
        let idx = match addr {
            0x0000..=0x3FFF => addr as usize,
            0x4000..=0x7FFF => self.rom_bank * 0x4000 + (addr as usize - 0x4000),
            _ => return 0xFF,
        };
        self.rom.get(idx % self.rom.len().max(1)).copied().unwrap_or(0xFF)
    }

    fn write_rom(&mut self, _addr: u16, _val: u8) {}

    fn read_ram(&self, addr: u16) -> u8 {
        if addr == 0xA000 {
            // Data port read — return data out based on reg_select
            match self.reg_select {
                0x0C => self.data_out_lo & 0x0F,
                0x0D => self.data_out_hi & 0x0F,
                _ => 0x01, // ready flag
            }
        } else {
            0xFF
        }
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        let val = val & 0x0F;
        if addr == 0xA001 {
            // Register select
            self.reg_select = val;
        } else if addr == 0xA000 {
            match self.reg_select {
                0x00 => {
                    // ROM bank low nybble
                    self.rom_bank = (self.rom_bank & 0xF0) | (val as usize);
                    if self.rom_bank == 0 { self.rom_bank = 1; }
                }
                0x01 => {
                    // ROM bank high nybble
                    self.rom_bank = (self.rom_bank & 0x0F) | ((val as usize) << 4);
                    if self.rom_bank == 0 { self.rom_bank = 1; }
                }
                0x04 => self.data_in_lo = val,
                0x05 => self.data_in_hi = val,
                0x06 => {
                    // Command low
                    self.data_in_lo = val;
                }
                0x07 => {
                    // Command high — triggers execution
                    self.data_in_hi = val;
                    self.execute_command();
                }
                _ => {}
            }
        }
    }

    fn has_battery(&self) -> bool { true }

    fn save_data(&self) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&self.tama_ram);
        data.extend_from_slice(&self.rtc_regs);
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        data.extend_from_slice(&ts.to_le_bytes());
        data
    }

    fn load_ram(&mut self, data: &[u8]) {
        if data.len() >= 32 {
            self.tama_ram.copy_from_slice(&data[..32]);
        }
        if data.len() >= 32 + 52 {
            self.rtc_regs.copy_from_slice(&data[32..84]);
        }
        if data.len() >= 32 + 52 + 8 {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&data[84..92]);
            let saved_ts = i64::from_le_bytes(buf);
            let now_ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;
            let elapsed = (now_ts - saved_ts).max(0) as u32;
            self.rtc_seconds += elapsed;
        }
        self.rtc_base = Instant::now();
    }
    fn snapshot_state(&self) -> Vec<u8> {
        let mut s = Vec::new();
        s.extend_from_slice(&(self.rom_bank as u32).to_le_bytes());
        s.extend_from_slice(&self.tama_ram);
        s.push(self.reg_select);
        s.push(self.data_in_lo);
        s.push(self.data_in_hi);
        s.push(self.data_out_lo);
        s.push(self.data_out_hi);
        s.extend_from_slice(&self.rtc_regs);
        s.extend_from_slice(&self.rtc_seconds.to_le_bytes());
        s
    }
    fn restore_state(&mut self, d: &[u8]) {
        if d.len() < 4 + 32 + 5 + 52 + 4 { return; }
        self.rom_bank = u32::from_le_bytes([d[0],d[1],d[2],d[3]]) as usize;
        self.tama_ram.copy_from_slice(&d[4..36]);
        self.reg_select = d[36];
        self.data_in_lo = d[37];
        self.data_in_hi = d[38];
        self.data_out_lo = d[39];
        self.data_out_hi = d[40];
        self.rtc_regs.copy_from_slice(&d[41..93]);
        self.rtc_seconds = u32::from_le_bytes([d[93],d[94],d[95],d[96]]);
        self.rtc_base = Instant::now();
    }
}

// ── Pocket Camera ────────────────────────────────────────────────────────────

pub struct PocketCamera {
    rom: Vec<u8>,
    ram: Vec<u8>,
    rom_bank: usize,
    ram_bank: usize,
    camera_regs_mapped: bool,
    camera_regs: [u8; 0x36],
    image_ready: bool,       // set after capture completes, image can be read
    noise_seed: u32,
    camera_image: Option<Box<[u8; 128 * 112]>>,
}

impl PocketCamera {
    fn new(rom: Vec<u8>) -> Self {
        PocketCamera {
            rom,
            ram: vec![0u8; 0x20000], // 128KB SRAM
            rom_bank: 1,
            ram_bank: 0,
            camera_regs_mapped: false,
            camera_regs: [0; 0x36],
            image_ready: false,
            noise_seed: 0x1234,
            camera_image: None,
        }
    }

    /// Generate a noise value for a pixel coordinate (deterministic hash).
    fn noise(&self, x: u8, y: u8) -> u8 {
        let value = (x as u32).wrapping_mul(151)
            .wrapping_add((y as u32).wrapping_mul(149))
            ^ self.noise_seed;
        let mut hash: u32 = 0;
        let mut v = value;
        for _ in 0..32 {
            hash <<= 1;
            if hash & 0x100 != 0 {
                hash ^= 0x101;
            }
            if v & 0x8000_0000 != 0 {
                hash ^= 0xA1;
            }
            v <<= 1;
        }
        hash as u8
    }

    /// Compute the processed color for pixel (x,y) using gain + exposure.
    fn processed_color(&self, x: u8, y: u8) -> i32 {
        let px = x.min(127);
        let py = y.min(111);
        let raw = if let Some(ref img) = self.camera_image {
            img[py as usize * 128 + px as usize] as i32
        } else {
            self.noise(px, py) as i32
        };

        // Apply gain
        let gain_idx = (self.camera_regs[4] & 0x1F) as usize;
        #[allow(clippy::excessive_precision)]
        const GAIN: [f64; 32] = [
            0.881, 0.915, 0.946, 0.974, 1.000, 1.024, 1.047, 1.068,
            1.088, 1.124, 1.157, 1.187, 1.214, 1.240, 1.274, 1.316,
            1.353, 1.386, 1.416, 1.443, 1.469, 1.493, 1.515, 1.536,
            1.555, 1.574, 1.591, 1.608, 1.624, 1.639, 1.653, 1.667,
        ];
        let color = (raw as f64 * GAIN[gain_idx]) as i32;

        // Apply exposure
        let exposure = ((self.camera_regs[2] as i32) << 8) | (self.camera_regs[3] as i32);
        color * exposure / 0x1000
    }

    /// Generate one byte of the 128×112 captured image on-the-fly.
    /// `offset` is relative to $A100, range 0..0xE00 (3584 bytes = 224 tiles).
    fn read_image_byte(&self, offset: u16) -> u8 {
        let tile_x = ((offset / 16) % 16) as u8;
        let tile_y = ((offset / 16) / 16) as u8;
        let row = ((offset >> 1) & 7) as u8;
        let bit = (offset & 1) as u8; // 0=low bitplane, 1=high bitplane
        let y = tile_y * 8 + row;

        let mut result: u8 = 0;
        for dx in 0..8u8 {
            let x = tile_x * 8 + dx;
            let color = self.processed_color(x, y);

            // Dither using the 4×4×3 threshold matrix in registers $06-$35
            let pat_idx = ((x & 3) + (y & 3) * 4) as usize;
            let pat_base = 6 + pat_idx * 3; // register offset

            let pixel = if pat_base + 2 < 0x36 {
                if color < self.camera_regs[pat_base] as i32 {
                    3
                } else if color < self.camera_regs[pat_base + 1] as i32 {
                    2
                } else if color < self.camera_regs[pat_base + 2] as i32 {
                    1
                } else {
                    0
                }
            } else {
                0
            };

            result <<= 1;
            result |= (pixel >> bit) & 1;
        }
        result
    }
}

impl Cartridge for PocketCamera {
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
            0x0000..=0x1FFF => {} // ram_enable accepted but camera ignores it
            0x2000..=0x3FFF => {
                // Full 8-bit ROM bank select
                self.rom_bank = val as usize;
                if self.rom_bank == 0 { self.rom_bank = 1; }
            }
            0x4000..=0x5FFF => {
                self.ram_bank = val as usize;
                self.camera_regs_mapped = val & 0x10 != 0;
            }
            _ => {}
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        // Camera register reads: only register 0 returns data, rest return 0
        if self.camera_regs_mapped {
            if (addr & 0x7F) == 0 {
                return self.camera_regs[0];
            }
            return 0;
        }

        // Camera busy: all RAM reads return 0
        if self.camera_regs[0] & 1 != 0 {
            return 0;
        }

        // Bank 0, $A100-$AEFF: generate image on the fly
        let ram_bank = self.ram_bank & 0x0F;
        if self.image_ready && ram_bank == 0 && addr >= 0xA100 && addr < 0xAF00 {
            return self.read_image_byte(addr - 0xA100);
        }

        // Normal RAM read (camera bypasses ram_enable)
        let idx = ram_bank * 0x2000 + (addr as usize - 0xA000);
        self.ram.get(idx % self.ram.len()).copied().unwrap_or(0xFF)
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        if self.camera_regs_mapped {
            let reg = (addr as usize) & 0x7F;
            if reg == 0 {
                let old = self.camera_regs[0];
                let new_val = val & 0x07;
                self.camera_regs[0] = new_val;
                // Trigger capture on 0→1 transition of bit 0
                if new_val & 1 != 0 && old & 1 == 0 {
                    // Randomize noise seed each capture
                    self.noise_seed = self.noise_seed.wrapping_mul(1103515245).wrapping_add(12345);
                    self.image_ready = true;
                    // Immediately mark capture complete (clear busy bit)
                    self.camera_regs[0] &= !1;
                }
            } else if reg < 0x36 {
                self.camera_regs[reg] = val;
            }
            return;
        }

        // Camera busy: forbid RAM writes
        if self.camera_regs[0] & 1 != 0 { return; }

        // Normal RAM write (camera bypasses ram_enable)
        let ram_bank = self.ram_bank & 0x0F;
        let idx = ram_bank * 0x2000 + (addr as usize - 0xA000);
        if idx < self.ram.len() {
            self.ram[idx] = val;
        }
    }

    fn has_battery(&self) -> bool { true }
    fn ram_data(&self) -> &[u8] { &self.ram }
    fn load_ram(&mut self, data: &[u8]) {
        let len = self.ram.len().min(data.len());
        self.ram[..len].copy_from_slice(&data[..len]);
    }
    fn has_camera(&self) -> bool { true }
    fn set_camera_image(&mut self, grayscale: &[u8; 128 * 112]) {
        let img = self.camera_image.get_or_insert_with(|| Box::new([0u8; 128 * 112]));
        img.copy_from_slice(grayscale);
    }
    fn snapshot_state(&self) -> Vec<u8> {
        let mut s = Vec::new();
        s.extend_from_slice(&(self.rom_bank as u32).to_le_bytes());
        s.extend_from_slice(&(self.ram_bank as u32).to_le_bytes());
        s.push(self.camera_regs_mapped as u8);
        s.extend_from_slice(&self.camera_regs);
        s.push(self.image_ready as u8);
        s.extend_from_slice(&self.noise_seed.to_le_bytes());
        s.extend_from_slice(&self.ram);
        s
    }
    fn restore_state(&mut self, d: &[u8]) {
        if d.len() < 9 + 0x36 + 1 + 4 { return; }
        self.rom_bank = u32::from_le_bytes([d[0],d[1],d[2],d[3]]) as usize;
        self.ram_bank = u32::from_le_bytes([d[4],d[5],d[6],d[7]]) as usize;
        self.camera_regs_mapped = d[8] != 0;
        self.camera_regs.copy_from_slice(&d[9..9+0x36]);
        let o = 9 + 0x36;
        self.image_ready = d[o] != 0;
        self.noise_seed = u32::from_le_bytes([d[o+1],d[o+2],d[o+3],d[o+4]]);
        let ram = &d[o+5..];
        let len = self.ram.len().min(ram.len());
        self.ram[..len].copy_from_slice(&ram[..len]);
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
        0x08 | 0x09 => {
            let battery = cart_type == 0x09;
            Box::new(RomRam::new(rom, battery))
        }
        0x0B..=0x0D => {
            let battery = cart_type == 0x0D;
            Box::new(Mmm01::new(rom, ram_size, battery))
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
        0x20 => Box::new(Mbc6::new(rom, ram_size)),
        0x22 => Box::new(Mbc7::new(rom)),
        0xFC => Box::new(PocketCamera::new(rom)),
        0xFD => Box::new(Tama5::new(rom)),
        0xFE => Box::new(HuC3::new(rom, ram_size)),
        0xFF => Box::new(HuC1::new(rom, ram_size)),
        other => {
            log::warn!("Unsupported cart type {:#04X}, using ROM-only", other);
            Box::new(RomOnly::new(rom))
        }
    }
}
