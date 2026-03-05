/// Cartridge abstraction — ROM-only, MBC1, MBC3, MBC5.

pub trait Cartridge: Send {
    fn read_rom(&self, addr: u16) -> u8;
    fn write_rom(&mut self, addr: u16, val: u8);
    fn read_ram(&self, addr: u16) -> u8;
    fn write_ram(&mut self, addr: u16, val: u8);
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
}

impl Mbc1 {
    fn new(rom: Vec<u8>, ram_size: usize) -> Self {
        Mbc1 {
            rom,
            ram: vec![0u8; ram_size.max(0x2000)],
            rom_bank: 1,
            ram_bank: 0,
            ram_enabled: false,
            banking_mode: 0,
            upper: 0,
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
}

// ── MBC3 ──────────────────────────────────────────────────────────────────────

pub struct Mbc3 {
    rom: Vec<u8>,
    ram: Vec<u8>,
    rom_bank: usize,
    ram_bank: usize,
    ram_enabled: bool,
}

impl Mbc3 {
    fn new(rom: Vec<u8>, ram_size: usize) -> Self {
        Mbc3 {
            rom,
            ram: vec![0u8; ram_size.max(0x2000)],
            rom_bank: 1,
            ram_bank: 0,
            ram_enabled: false,
        }
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
                let b = (val & 0x7F) as usize;
                self.rom_bank = if b == 0 { 1 } else { b };
            }
            0x4000..=0x5FFF => self.ram_bank = (val & 0x07) as usize,
            0x6000..=0x7FFF => {} // RTC latch — ignored
            _ => {}
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        if !self.ram_enabled || self.ram_bank > 3 { return 0xFF; }
        let idx = self.ram_bank * 0x2000 + (addr as usize - 0xA000);
        self.ram.get(idx % self.ram.len().max(1)).copied().unwrap_or(0xFF)
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        if !self.ram_enabled || self.ram_bank > 3 { return; }
        let idx = self.ram_bank * 0x2000 + (addr as usize - 0xA000);
        let len = self.ram.len().max(1);
        self.ram[idx % len] = val;
    }
}

// ── MBC5 ──────────────────────────────────────────────────────────────────────

pub struct Mbc5 {
    rom: Vec<u8>,
    ram: Vec<u8>,
    rom_bank: usize, // 9-bit bank number
    ram_bank: usize,
    ram_enabled: bool,
}

impl Mbc5 {
    fn new(rom: Vec<u8>, ram_size: usize) -> Self {
        Mbc5 {
            rom,
            ram: vec![0u8; ram_size.max(0x2000)],
            rom_bank: 1,
            ram_bank: 0,
            ram_enabled: false,
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
        0x01..=0x03 => Box::new(Mbc1::new(rom, ram_size)),
        0x0F..=0x13 => Box::new(Mbc3::new(rom, ram_size)),
        0x19..=0x1E => Box::new(Mbc5::new(rom, ram_size)),
        other => {
            log::warn!("Unsupported cart type {:#04X}, using ROM-only", other);
            Box::new(RomOnly::new(rom))
        }
    }
}
