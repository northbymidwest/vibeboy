use std::sync::Arc;
use super::Cartridge;

pub struct Mbc6 {
    rom: Arc<[u8]>,
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
    pub(super) fn new(rom: Arc<[u8]>, ram_size: usize) -> Self {
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
