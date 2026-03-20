use super::Cartridge;

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
    pub(super) fn new(rom: Vec<u8>, ram_size: usize, battery: bool) -> Self {
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
