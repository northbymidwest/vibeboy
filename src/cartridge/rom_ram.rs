use std::sync::Arc;
use super::Cartridge;

pub struct RomRam {
    rom: Arc<[u8]>,
    ram: Vec<u8>,
    battery: bool,
}

impl RomRam {
    pub(super) fn new(rom: Arc<[u8]>, battery: bool) -> Self {
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
