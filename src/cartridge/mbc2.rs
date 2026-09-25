use super::{BAD_REGISTERS, CartState, Cartridge, WRONG_MAPPER, ensure, ensure_ram_len};
use std::sync::Arc;

pub struct Mbc2 {
    rom: Arc<[u8]>,
    battery: bool,
    state: Mbc2State,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Mbc2State {
    ram: Vec<u8>,    // 512 × 4-bit values
    rom_bank: usize, // 1-15
    ram_enabled: bool,
}

impl Mbc2 {
    pub(super) fn new(rom: Arc<[u8]>, battery: bool) -> Self {
        Mbc2 {
            rom,
            battery,
            state: Mbc2State {
                ram: vec![0u8; 512],
                rom_bank: 1,
                ram_enabled: false,
            },
        }
    }
}

impl Cartridge for Mbc2 {
    fn read_rom(&self, addr: u16) -> u8 {
        let idx = match addr {
            0x0000..=0x3FFF => addr as usize,
            0x4000..=0x7FFF => self.state.rom_bank * 0x4000 + (addr as usize - 0x4000),
            _ => return 0xFF,
        };
        self.rom
            .get(idx % self.rom.len().max(1))
            .copied()
            .unwrap_or(0xFF)
    }

    fn write_rom(&mut self, addr: u16, val: u8) {
        if let 0x0000..=0x3FFF = addr {
            if addr & 0x0100 == 0 {
                self.state.ram_enabled = (val & 0x0F) == 0x0A;
            } else {
                let b = (val & 0x0F) as usize;
                self.state.rom_bank = if b == 0 { 1 } else { b };
            }
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        if !self.state.ram_enabled {
            return 0xFF;
        }
        self.state.ram[(addr as usize) & 0x1FF] | 0xF0
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        if !self.state.ram_enabled {
            return;
        }
        self.state.ram[(addr as usize) & 0x1FF] = val & 0x0F;
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
        CartState::Mbc2(self.state.clone())
    }
    fn validate_state(&self, state: &CartState) -> Result<(), &'static str> {
        let CartState::Mbc2(s) = state else {
            return Err(WRONG_MAPPER);
        };
        ensure_ram_len(&s.ram, &self.state.ram)?;
        ensure((1..=0x0F).contains(&s.rom_bank), BAD_REGISTERS)
    }
    fn restore_state(&mut self, state: &CartState) {
        let CartState::Mbc2(s) = state else {
            panic!("{WRONG_MAPPER}");
        };
        self.state.clone_from(s);
    }
}
