use super::Cartridge;

pub struct Mbc5 {
    rom: Vec<u8>,
    ram: Vec<u8>,
    rom_bank: usize, // 9-bit bank number
    ram_bank: usize,
    ram_enabled: bool,
    battery: bool,
    has_rumble: bool,
    rumble_on: bool,
    /// Sticky flag: set whenever rumble_on transitions to true, cleared by the frontend.
    rumble_latched: bool,
}

impl Mbc5 {
    pub(super) fn new(rom: Vec<u8>, ram_size: usize, battery: bool, has_rumble: bool) -> Self {
        Mbc5 {
            rom,
            ram: vec![0u8; ram_size.max(0x2000)],
            rom_bank: 1,
            ram_bank: 0,
            ram_enabled: false,
            battery,
            has_rumble,
            rumble_on: false,
            rumble_latched: false,
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
        self.rom.get(idx % self.rom.len().max(1)).copied().unwrap_or(0xFF)
    }

    fn write_rom(&mut self, addr: u16, val: u8) {
        match addr {
            0x0000..=0x1FFF => self.ram_enabled = val & 0x0F == 0x0A,
            0x2000..=0x2FFF => self.rom_bank = (self.rom_bank & 0x100) | (val as usize),
            0x3000..=0x3FFF => self.rom_bank = (self.rom_bank & 0xFF) | (((val & 0x01) as usize) << 8),
            0x4000..=0x5FFF => {
                if self.has_rumble {
                    let on = val & 0x08 != 0;
                    if on { self.rumble_latched = true; }
                    self.rumble_on = on;
                    self.ram_bank = (val & 0x07) as usize;
                } else {
                    self.ram_bank = (val & 0x0F) as usize;
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
        if let Some(b) = self.ram.get_mut(idx) { *b = val; }
    }

    fn has_battery(&self) -> bool { self.battery }
    fn has_rumble(&self) -> bool { self.has_rumble }
    fn rumble_active(&self) -> bool { self.rumble_on }
    fn drain_rumble(&mut self) -> bool {
        let active = self.rumble_on || self.rumble_latched;
        self.rumble_latched = false;
        active
    }
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
        s.push(self.rumble_on as u8);
        s.extend_from_slice(&self.ram);
        s
    }
    fn restore_state(&mut self, d: &[u8]) {
        if d.len() < 10 { return; }
        self.rom_bank = u32::from_le_bytes([d[0],d[1],d[2],d[3]]) as usize;
        self.ram_bank = u32::from_le_bytes([d[4],d[5],d[6],d[7]]) as usize;
        self.ram_enabled = d[8] != 0;
        self.rumble_on = d[9] != 0;
        let ram = &d[10..];
        let len = self.ram.len().min(ram.len());
        self.ram[..len].copy_from_slice(&ram[..len]);
    }
}
