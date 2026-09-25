use super::{BAD_REGISTERS, CartState, Cartridge, WRONG_MAPPER, ensure, ensure_ram_len};
use std::sync::Arc;

pub struct HuC1 {
    rom: Arc<[u8]>,
    state: HuC1State,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct HuC1State {
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
            state: HuC1State {
                ram: vec![0u8; ram_size.max(0x2000)],
                rom_bank: 1,
                ram_bank: 0,
                ir_mode: false,
                ir_led: false,
            },
        }
    }
}

impl Cartridge for HuC1 {
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
        match addr {
            0x0000..=0x1FFF => self.state.ir_mode = val == 0x0E,
            0x2000..=0x3FFF => {
                let b = (val & 0x3F) as usize;
                self.state.rom_bank = if b == 0 { 1 } else { b };
            }
            0x4000..=0x5FFF => self.state.ram_bank = (val & 0x03) as usize,
            _ => {}
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        if self.state.ir_mode {
            return 0xC0; // No IR signal received (stub)
        }
        let idx = self.state.ram_bank * 0x2000 + (addr as usize - 0xA000);
        self.state
            .ram
            .get(idx % self.state.ram.len().max(1))
            .copied()
            .unwrap_or(0xFF)
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        if self.state.ir_mode {
            self.state.ir_led = val & 0x01 != 0;
            return;
        }
        let idx = self.state.ram_bank * 0x2000 + (addr as usize - 0xA000);
        let len = self.state.ram.len().max(1);
        self.state.ram[idx % len] = val;
    }

    fn has_battery(&self) -> bool {
        true
    }
    fn ram_data(&self) -> &[u8] {
        &self.state.ram
    }
    fn load_ram(&mut self, data: &[u8]) {
        let len = self.state.ram.len().min(data.len());
        self.state.ram[..len].copy_from_slice(&data[..len]);
    }
    fn snapshot_state(&self) -> CartState {
        CartState::HuC1(self.state.clone())
    }
    fn validate_state(&self, state: &CartState) -> Result<(), &'static str> {
        let CartState::HuC1(s) = state else {
            return Err(WRONG_MAPPER);
        };
        ensure_ram_len(&s.ram, &self.state.ram)?;
        ensure(s.rom_bank <= 0x3F && s.ram_bank <= 3, BAD_REGISTERS)
    }
    fn restore_state(&mut self, state: &CartState) {
        let CartState::HuC1(s) = state else {
            panic!("{WRONG_MAPPER}");
        };
        self.state.clone_from(s);
    }
}
