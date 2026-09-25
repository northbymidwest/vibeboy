use super::{BAD_REGISTERS, CartState, Cartridge, WRONG_MAPPER, ensure, ensure_ram_len};
use std::sync::Arc;

pub struct Mbc6 {
    rom: Arc<[u8]>,
    state: Mbc6State,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Mbc6State {
    ram: Vec<u8>,
    flash: Vec<u8>,
    rom_bank_a: usize,
    rom_bank_b: usize,
    ram_bank_a: usize,
    ram_bank_b: usize,
    ram_enabled_a: bool,
    ram_enabled_b: bool,
    flash_enabled: bool,
    flash_write_enabled: bool,
    bank_a_is_flash: bool,
    bank_b_is_flash: bool,
}

impl Mbc6 {
    pub(super) fn new(rom: Arc<[u8]>, ram_size: usize) -> Self {
        Mbc6 {
            rom,
            state: Mbc6State {
                ram: vec![0u8; ram_size.max(0x2000)],
                flash: vec![0xFF; 0x100000], // 1MB MX29F008TC
                rom_bank_a: 2,
                rom_bank_b: 3,
                ram_bank_a: 0,
                ram_bank_b: 0,
                ram_enabled_a: false,
                ram_enabled_b: false,
                flash_enabled: false,
                flash_write_enabled: false,
                bank_a_is_flash: false,
                bank_b_is_flash: false,
            },
        }
    }
}

impl Cartridge for Mbc6 {
    fn read_rom(&self, addr: u16) -> u8 {
        match addr {
            0x0000..=0x3FFF => self.rom.get(addr as usize).copied().unwrap_or(0xFF),
            0x4000..=0x5FFF => {
                if self.state.bank_a_is_flash {
                    let idx = self.state.rom_bank_a * 0x2000 + (addr as usize - 0x4000);
                    self.state
                        .flash
                        .get(idx % self.state.flash.len())
                        .copied()
                        .unwrap_or(0xFF)
                } else {
                    let idx = self.state.rom_bank_a * 0x2000 + (addr as usize - 0x4000);
                    self.rom
                        .get(idx % self.rom.len().max(1))
                        .copied()
                        .unwrap_or(0xFF)
                }
            }
            0x6000..=0x7FFF => {
                if self.state.bank_b_is_flash {
                    let idx = self.state.rom_bank_b * 0x2000 + (addr as usize - 0x6000);
                    self.state
                        .flash
                        .get(idx % self.state.flash.len())
                        .copied()
                        .unwrap_or(0xFF)
                } else {
                    let idx = self.state.rom_bank_b * 0x2000 + (addr as usize - 0x6000);
                    self.rom
                        .get(idx % self.rom.len().max(1))
                        .copied()
                        .unwrap_or(0xFF)
                }
            }
            _ => 0xFF,
        }
    }

    fn write_rom(&mut self, addr: u16, val: u8) {
        match addr {
            0x0000..=0x03FF => self.state.ram_enabled_a = val == 0x0A,
            0x0400..=0x07FF => self.state.ram_enabled_b = val == 0x0A,
            0x0800..=0x0BFF => self.state.ram_bank_a = (val & 0x07) as usize,
            0x0C00..=0x0FFF => self.state.ram_bank_b = (val & 0x07) as usize,
            0x1000 => self.state.flash_enabled = val == 0x01,
            0x1001 => self.state.flash_write_enabled = val == 0x01,
            0x2000..=0x27FF => self.state.rom_bank_a = (val as usize) & 0x7F,
            0x2800..=0x2FFF => self.state.bank_a_is_flash = val == 0x08,
            0x3000..=0x37FF => self.state.rom_bank_b = (val as usize) & 0x7F,
            0x3800..=0x3FFF => self.state.bank_b_is_flash = val == 0x08,
            _ => {}
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        match addr {
            0xA000..=0xAFFF => {
                if !self.state.ram_enabled_a {
                    return 0xFF;
                }
                let idx = self.state.ram_bank_a * 0x1000 + (addr as usize - 0xA000);
                self.state
                    .ram
                    .get(idx % self.state.ram.len().max(1))
                    .copied()
                    .unwrap_or(0xFF)
            }
            0xB000..=0xBFFF => {
                if !self.state.ram_enabled_b {
                    return 0xFF;
                }
                let idx = self.state.ram_bank_b * 0x1000 + (addr as usize - 0xB000);
                self.state
                    .ram
                    .get(idx % self.state.ram.len().max(1))
                    .copied()
                    .unwrap_or(0xFF)
            }
            _ => 0xFF,
        }
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        match addr {
            0xA000..=0xAFFF => {
                if !self.state.ram_enabled_a {
                    return;
                }
                let idx = self.state.ram_bank_a * 0x1000 + (addr as usize - 0xA000);
                let len = self.state.ram.len().max(1);
                self.state.ram[idx % len] = val;
            }
            0xB000..=0xBFFF => {
                if !self.state.ram_enabled_b {
                    return;
                }
                let idx = self.state.ram_bank_b * 0x1000 + (addr as usize - 0xB000);
                let len = self.state.ram.len().max(1);
                self.state.ram[idx % len] = val;
            }
            _ => {}
        }
    }

    fn has_battery(&self) -> bool {
        true
    }
    fn ram_data(&self) -> &[u8] {
        &self.state.ram
    }

    fn save_data(&self) -> Vec<u8> {
        let mut data = self.state.ram.clone();
        data.extend_from_slice(&self.state.flash);
        data
    }

    fn load_ram(&mut self, data: &[u8]) {
        let ram_len = self.state.ram.len();
        let copy_len = ram_len.min(data.len());
        self.state.ram[..copy_len].copy_from_slice(&data[..copy_len]);
        if data.len() > ram_len {
            let flash_len = self.state.flash.len().min(data.len() - ram_len);
            self.state.flash[..flash_len].copy_from_slice(&data[ram_len..ram_len + flash_len]);
        }
    }
    fn snapshot_state(&self) -> CartState {
        CartState::Mbc6(self.state.clone())
    }
    fn validate_state(&self, state: &CartState) -> Result<(), &'static str> {
        let CartState::Mbc6(s) = state else {
            return Err(WRONG_MAPPER);
        };
        ensure_ram_len(&s.ram, &self.state.ram)?;
        ensure_ram_len(&s.flash, &self.state.flash)?;
        ensure(
            s.rom_bank_a <= 0x7F && s.rom_bank_b <= 0x7F && s.ram_bank_a <= 7 && s.ram_bank_b <= 7,
            BAD_REGISTERS,
        )
    }
    fn restore_state(&mut self, state: &CartState) {
        let CartState::Mbc6(s) = state else {
            panic!("{WRONG_MAPPER}");
        };
        self.state.clone_from(s);
    }
}
