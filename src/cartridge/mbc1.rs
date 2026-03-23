use std::sync::Arc;
use super::Cartridge;

pub struct Mbc1 {
    rom: Arc<[u8]>,
    ram: Vec<u8>,
    rom_bank: usize,
    ram_bank: usize,
    ram_enabled: bool,
    /// 0 = ROM banking mode, 1 = RAM banking mode
    banking_mode: u8,
    /// Upper 2 bits (affect bank 0 in mode 1, and RAM bank)
    upper: usize,
    battery: bool,
    /// MBC1M multicart wiring: upper bits shift by 4 instead of 5
    multicart: bool,
}

impl Mbc1 {
    pub(super) fn new(rom: Arc<[u8]>, ram_size: usize, battery: bool) -> Self {
        // Detect MBC1M multicart: Nintendo logo at 0x104 also appears at 0x40104
        let multicart = rom.len() >= 0x44000
            && rom.len() >= 0x40134
            && rom[0x104..0x134] == rom[0x40104..0x40134];
        if multicart {
            log::info!("MBC1M multicart detected (ROM size: {}KB)", rom.len() / 1024);
        }
        Mbc1 {
            rom,
            ram: vec![0u8; ram_size.max(0x2000)],
            rom_bank: 1,
            ram_bank: 0,
            ram_enabled: false,
            banking_mode: 0,
            upper: 0,
            battery,
            multicart,
        }
    }
}

impl Cartridge for Mbc1 {
    fn read_rom(&self, addr: u16) -> u8 {
        let shift = if self.multicart { 4 } else { 5 };
        let mask = (1usize << shift) - 1; // 0x0F for multicart, 0x1F for standard
        let idx = match addr {
            0x0000..=0x3FFF => {
                if self.banking_mode == 1 {
                    (self.upper << (shift + 14)) | (addr as usize)
                } else {
                    addr as usize
                }
            }
            0x4000..=0x7FFF => {
                let mut bank = (self.upper << shift) | (self.rom_bank & mask);
                // Zero-adjust: if the full 5-bit register is 0, increment
                if self.rom_bank & 0x1F == 0 {
                    bank += 1;
                }
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
                // Store raw 5-bit value; masking happens at address time
                self.rom_bank = (val & 0x1F) as usize;
            }
            0x4000..=0x5FFF => {
                self.upper = (val & 0x03) as usize;
                if self.banking_mode == 1 {
                    self.ram_bank = if self.multicart { 0 } else { self.upper };
                }
            }
            0x6000..=0x7FFF => {
                self.banking_mode = val & 0x01;
                if self.banking_mode == 0 {
                    self.ram_bank = 0;
                } else {
                    self.ram_bank = if self.multicart { 0 } else { self.upper };
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
