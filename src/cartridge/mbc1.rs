use super::{BAD_REGISTERS, CartState, Cartridge, WRONG_MAPPER, ensure, ensure_ram_len};
use std::sync::Arc;

pub struct Mbc1 {
    rom: Arc<[u8]>,
    battery: bool,
    /// MBC1M multicart wiring: upper bits shift by 4 instead of 5
    multicart: bool,
    state: Mbc1State,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Mbc1State {
    ram: Vec<u8>,
    rom_bank: usize,
    ram_bank: usize,
    ram_enabled: bool,
    /// 0 = ROM banking mode, 1 = RAM banking mode
    banking_mode: u8,
    /// Upper 2 bits (affect bank 0 in mode 1, and RAM bank)
    upper: usize,
}

impl Mbc1 {
    pub(super) fn new(rom: Arc<[u8]>, ram_size: usize, battery: bool) -> Self {
        // Detect MBC1M multicart: Nintendo logo at 0x104 also appears at 0x40104
        let multicart = rom.len() >= 0x44000
            && rom.len() >= 0x40134
            && rom[0x104..0x134] == rom[0x40104..0x40134];
        if multicart {
            log::info!(
                "MBC1M multicart detected (ROM size: {}KB)",
                rom.len() / 1024
            );
        }
        Mbc1 {
            rom,
            battery,
            multicart,
            state: Mbc1State {
                ram: vec![0u8; ram_size.max(0x2000)],
                rom_bank: 1,
                ram_bank: 0,
                ram_enabled: false,
                banking_mode: 0,
                upper: 0,
            },
        }
    }
}

impl Cartridge for Mbc1 {
    fn read_rom(&self, addr: u16) -> u8 {
        let shift = if self.multicart { 4 } else { 5 };
        let mask = (1usize << shift) - 1; // 0x0F for multicart, 0x1F for standard
        let idx = match addr {
            0x0000..=0x3FFF => {
                if self.state.banking_mode == 1 {
                    (self.state.upper << (shift + 14)) | (addr as usize)
                } else {
                    addr as usize
                }
            }
            0x4000..=0x7FFF => {
                let mut bank = (self.state.upper << shift) | (self.state.rom_bank & mask);
                // Zero-adjust: if the full 5-bit register is 0, increment
                if self.state.rom_bank & 0x1F == 0 {
                    bank += 1;
                }
                bank * 0x4000 + (addr as usize - 0x4000)
            }
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
            0x2000..=0x3FFF => {
                // Store raw 5-bit value; masking happens at address time
                self.state.rom_bank = (val & 0x1F) as usize;
            }
            0x4000..=0x5FFF => {
                self.state.upper = (val & 0x03) as usize;
                if self.state.banking_mode == 1 {
                    self.state.ram_bank = if self.multicart { 0 } else { self.state.upper };
                }
            }
            0x6000..=0x7FFF => {
                self.state.banking_mode = val & 0x01;
                if self.state.banking_mode == 0 {
                    self.state.ram_bank = 0;
                } else {
                    self.state.ram_bank = if self.multicart { 0 } else { self.state.upper };
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
        let len = self.state.ram.len().max(1);
        let i = idx % len;
        self.state.ram[i] = val;
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
        CartState::Mbc1(self.state.clone())
    }
    fn validate_state(&self, state: &CartState) -> Result<(), &'static str> {
        let CartState::Mbc1(s) = state else {
            return Err(WRONG_MAPPER);
        };
        ensure_ram_len(&s.ram, &self.state.ram)?;
        ensure(
            s.rom_bank <= 0x1F && s.ram_bank <= 3 && s.banking_mode <= 1 && s.upper <= 3,
            BAD_REGISTERS,
        )
    }
    fn restore_state(&mut self, state: &CartState) {
        let CartState::Mbc1(s) = state else {
            panic!("{WRONG_MAPPER}");
        };
        self.state.clone_from(s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_registers_the_mapper_cannot_produce() {
        let rom: Arc<[u8]> = vec![0u8; 0x8000].into();
        let c = Mbc1::new(rom, 0x2000, false);
        let mut s = c.state.clone();
        assert!(c.validate_state(&CartState::Mbc1(s.clone())).is_ok());
        s.upper = 4;
        assert_eq!(c.validate_state(&CartState::Mbc1(s)), Err(BAD_REGISTERS));
    }
}
