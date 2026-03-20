use super::Cartridge;

pub struct RomOnly {
    rom: Vec<u8>,
}

impl RomOnly {
    pub(super) fn new(rom: Vec<u8>) -> Self { RomOnly { rom } }
}

impl Cartridge for RomOnly {
    fn read_rom(&self, addr: u16) -> u8 {
        self.rom.get(addr as usize).copied().unwrap_or(0xFF)
    }
    fn write_rom(&mut self, _addr: u16, _val: u8) {}
    fn read_ram(&self, _addr: u16) -> u8 { 0xFF }
    fn write_ram(&mut self, _addr: u16, _val: u8) {}
}
