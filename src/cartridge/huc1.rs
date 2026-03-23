use std::sync::Arc;
use super::Cartridge;

pub struct HuC1 {
    rom: Arc<[u8]>,
    ram: Vec<u8>,
    rom_bank: usize,
    ram_bank: usize,
    ir_mode: bool,
    ir_led: bool,
}

impl HuC1 {
    pub(super) fn new(rom: Arc<[u8]>, ram_size: usize) -> Self {
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
