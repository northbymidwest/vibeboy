use super::{CartState, Cartridge, WRONG_MAPPER, ensure_ram_len};
use std::sync::Arc;

pub struct RomRam {
    rom: Arc<[u8]>,
    battery: bool,
    state: RomRamState,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct RomRamState {
    ram: Vec<u8>,
}

impl RomRam {
    pub(super) fn new(rom: Arc<[u8]>, battery: bool) -> Self {
        RomRam {
            rom,
            battery,
            state: RomRamState {
                ram: vec![0u8; 0x2000],
            },
        }
    }
}

impl Cartridge for RomRam {
    fn read_rom(&self, addr: u16) -> u8 {
        self.rom.get(addr as usize).copied().unwrap_or(0xFF)
    }
    fn write_rom(&mut self, _addr: u16, _val: u8) {}
    fn read_ram(&self, addr: u16) -> u8 {
        self.state.ram[(addr as usize - 0xA000) & 0x1FFF]
    }
    fn write_ram(&mut self, addr: u16, val: u8) {
        self.state.ram[(addr as usize - 0xA000) & 0x1FFF] = val;
    }
    fn has_battery(&self) -> bool {
        self.battery
    }
    fn ram_data(&self) -> &[u8] {
        &self.state.ram
    }
    fn load_ram(&mut self, data: &[u8]) {
        let len = self.state.ram.len().min(data.len());
        self.state.ram[..len].copy_from_slice(&data[..len]);
    }
    fn snapshot_state(&self) -> CartState {
        CartState::RomRam(self.state.clone())
    }
    fn validate_state(&self, state: &CartState) -> Result<(), &'static str> {
        let CartState::RomRam(s) = state else {
            return Err(WRONG_MAPPER);
        };
        ensure_ram_len(&s.ram, &self.state.ram)
    }
    fn restore_state(&mut self, state: &CartState) {
        let CartState::RomRam(s) = state else {
            panic!("{WRONG_MAPPER}");
        };
        self.state.clone_from(s);
    }
}
