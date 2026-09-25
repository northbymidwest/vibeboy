use super::{CartState, Cartridge, WRONG_MAPPER};
use std::sync::Arc;

pub struct RomOnly {
    rom: Arc<[u8]>,
}

impl RomOnly {
    pub(super) fn new(rom: Arc<[u8]>) -> Self {
        RomOnly { rom }
    }
}

impl Cartridge for RomOnly {
    fn read_rom(&self, addr: u16) -> u8 {
        self.rom.get(addr as usize).copied().unwrap_or(0xFF)
    }
    fn write_rom(&mut self, _addr: u16, _val: u8) {}
    fn read_ram(&self, _addr: u16) -> u8 {
        0xFF
    }
    fn write_ram(&mut self, _addr: u16, _val: u8) {}
    fn snapshot_state(&self) -> CartState {
        CartState::RomOnly
    }
    fn validate_state(&self, state: &CartState) -> Result<(), &'static str> {
        match state {
            CartState::RomOnly => Ok(()),
            _ => Err(WRONG_MAPPER),
        }
    }
    fn restore_state(&mut self, _state: &CartState) {}
}
