use super::Cartridge;

pub struct Mbc2 {
    rom: Vec<u8>,
    ram: [u8; 512], // 512 × 4-bit values
    rom_bank: usize, // 1-15
    ram_enabled: bool,
    battery: bool,
}

impl Mbc2 {
    pub(super) fn new(rom: Vec<u8>, battery: bool) -> Self {
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
