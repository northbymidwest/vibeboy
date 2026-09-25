use super::{BAD_REGISTERS, CartState, Cartridge, WRONG_MAPPER, ensure, ensure_ram_len};
use std::sync::Arc;

pub struct Mbc5 {
    rom: Arc<[u8]>,
    battery: bool,
    has_rumble: bool,
    state: Mbc5State,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Mbc5State {
    ram: Vec<u8>,
    rom_bank: usize, // 9-bit bank number
    ram_bank: usize,
    ram_enabled: bool,
    rumble_on: bool,
    /// Sticky flag: set whenever rumble_on transitions to true, cleared by the frontend.
    rumble_latched: bool,
}

impl Mbc5 {
    pub(super) fn new(rom: Arc<[u8]>, ram_size: usize, battery: bool, has_rumble: bool) -> Self {
        Mbc5 {
            rom,
            battery,
            has_rumble,
            state: Mbc5State {
                ram: vec![0u8; ram_size.max(0x2000)],
                rom_bank: 1,
                ram_bank: 0,
                ram_enabled: false,
                rumble_on: false,
                rumble_latched: false,
            },
        }
    }
}

impl Cartridge for Mbc5 {
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
            0x0000..=0x1FFF => self.state.ram_enabled = val & 0x0F == 0x0A,
            0x2000..=0x2FFF => self.state.rom_bank = (self.state.rom_bank & 0x100) | (val as usize),
            0x3000..=0x3FFF => {
                self.state.rom_bank = (self.state.rom_bank & 0xFF) | (((val & 0x01) as usize) << 8)
            }
            0x4000..=0x5FFF => {
                if self.has_rumble {
                    let on = val & 0x08 != 0;
                    if on {
                        self.state.rumble_latched = true;
                    }
                    self.state.rumble_on = on;
                    self.state.ram_bank = (val & 0x07) as usize;
                } else {
                    self.state.ram_bank = (val & 0x0F) as usize;
                }
            }
            _ => {}
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        if !self.state.ram_enabled {
            return 0xFF;
        }
        let idx = self.state.ram_bank * 0x2000 + (addr as usize - 0xA000);
        self.state
            .ram
            .get(idx % self.state.ram.len().max(1))
            .copied()
            .unwrap_or(0xFF)
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        if !self.state.ram_enabled {
            return;
        }
        let idx = self.state.ram_bank * 0x2000 + (addr as usize - 0xA000);
        // Mirror the same way reads do so small RAMs alias consistently.
        let len = self.state.ram.len().max(1);
        if let Some(b) = self.state.ram.get_mut(idx % len) {
            *b = val;
        }
    }

    fn has_battery(&self) -> bool {
        self.battery
    }
    fn has_rumble(&self) -> bool {
        self.has_rumble
    }
    fn rumble_active(&self) -> bool {
        self.state.rumble_on
    }
    fn drain_rumble(&mut self) -> bool {
        let active = self.state.rumble_on || self.state.rumble_latched;
        self.state.rumble_latched = false;
        active
    }
    fn ram_data(&self) -> &[u8] {
        &self.state.ram
    }
    fn load_ram(&mut self, data: &[u8]) {
        let len = self.state.ram.len().min(data.len());
        self.state.ram[..len].copy_from_slice(&data[..len]);
    }
    fn snapshot_state(&self) -> CartState {
        CartState::Mbc5(self.state.clone())
    }
    fn validate_state(&self, state: &CartState) -> Result<(), &'static str> {
        let CartState::Mbc5(s) = state else {
            return Err(WRONG_MAPPER);
        };
        ensure_ram_len(&s.ram, &self.state.ram)?;
        ensure(s.rom_bank <= 0x1FF && s.ram_bank <= 0x0F, BAD_REGISTERS)
    }
    fn restore_state(&mut self, state: &CartState) {
        let CartState::Mbc5(s) = state else {
            panic!("{WRONG_MAPPER}");
        };
        self.state.clone_from(s);
    }
}
