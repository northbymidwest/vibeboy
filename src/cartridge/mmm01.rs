use super::{BAD_REGISTERS, CartState, Cartridge, WRONG_MAPPER, ensure, ensure_ram_len};
use std::sync::Arc;

pub struct Mmm01 {
    rom: Arc<[u8]>,
    battery: bool,
    state: Mmm01State,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Mmm01State {
    ram: Vec<u8>,
    mapped: bool,
    // Unmapped mode captures
    rom_base: usize,      // base ROM bank (from $2000/$4000 in unmapped mode)
    rom_bank_mask: usize, // per-game ROM bank mask
    ram_bank_mask: usize,
    // Mapped mode registers
    rom_bank: usize,
    ram_bank: usize,
    ram_enabled: bool,
    banking_mode: u8,
    upper: usize,
}

impl Mmm01 {
    pub(super) fn new(rom: Arc<[u8]>, ram_size: usize, battery: bool) -> Self {
        Mmm01 {
            rom,
            battery,
            state: Mmm01State {
                ram: vec![0u8; ram_size.max(0x2000)],
                mapped: false,
                rom_base: 0,
                rom_bank_mask: 0x1FF,
                ram_bank_mask: 0x03,
                rom_bank: 1,
                ram_bank: 0,
                ram_enabled: false,
                banking_mode: 0,
                upper: 0,
            },
        }
    }
}

impl Cartridge for Mmm01 {
    fn read_rom(&self, addr: u16) -> u8 {
        if !self.state.mapped {
            // Unmapped: last 32KB of ROM
            let base = self.rom.len().saturating_sub(0x8000);
            let idx = base + (addr as usize);
            return self.rom.get(idx).copied().unwrap_or(0xFF);
        }
        let idx = match addr {
            0x0000..=0x3FFF => {
                let bank = self.state.rom_base;
                bank * 0x4000 + (addr as usize)
            }
            0x4000..=0x7FFF => {
                let bank = self.state.rom_base + (self.state.rom_bank & self.state.rom_bank_mask);
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
        if !self.state.mapped {
            match addr {
                0x0000..=0x1FFF => {
                    if val & 0x40 != 0 {
                        // Lock into mapped mode
                        self.state.mapped = true;
                        log::info!(
                            "MMM01: locked into mapped mode, rom_base={}",
                            self.state.rom_base
                        );
                    }
                    self.state.ram_bank_mask = ((val >> 4) & 0x03) as usize;
                }
                0x2000..=0x3FFF => {
                    self.state.rom_base = (val as usize) & 0x7F;
                }
                0x4000..=0x5FFF => {
                    self.state.rom_bank_mask = ((val >> 1) & 0x1F) as usize;
                    if self.state.rom_bank_mask == 0 {
                        self.state.rom_bank_mask = 0x1F;
                    }
                }
                0x6000..=0x7FFF => {
                    self.state.banking_mode = val & 0x01;
                }
                _ => {}
            }
            return;
        }
        // Mapped mode: MBC1-like
        match addr {
            0x0000..=0x1FFF => self.state.ram_enabled = val & 0x0F == 0x0A,
            0x2000..=0x3FFF => {
                let b = (val & 0x1F) as usize;
                self.state.rom_bank = if b == 0 { 1 } else { b };
            }
            0x4000..=0x5FFF => {
                self.state.upper = (val & 0x03) as usize;
                if self.state.banking_mode == 1 {
                    self.state.ram_bank = self.state.upper & self.state.ram_bank_mask;
                }
            }
            0x6000..=0x7FFF => {
                self.state.banking_mode = val & 0x01;
                if self.state.banking_mode == 0 {
                    self.state.ram_bank = 0;
                } else {
                    self.state.ram_bank = self.state.upper & self.state.ram_bank_mask;
                }
            }
            _ => {}
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        if !self.state.ram_enabled || !self.state.mapped {
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
        if !self.state.ram_enabled || !self.state.mapped {
            return;
        }
        let idx = self.state.ram_bank * 0x2000 + (addr as usize - 0xA000);
        let len = self.state.ram.len().max(1);
        self.state.ram[idx % len] = val;
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
        CartState::Mmm01(self.state.clone())
    }
    fn validate_state(&self, state: &CartState) -> Result<(), &'static str> {
        let CartState::Mmm01(s) = state else {
            return Err(WRONG_MAPPER);
        };
        ensure_ram_len(&s.ram, &self.state.ram)?;
        ensure(
            s.rom_base <= 0x7F
                && s.rom_bank_mask <= 0x1FF
                && s.ram_bank_mask <= 3
                && s.rom_bank <= 0x1F
                && s.ram_bank <= 3
                && s.banking_mode <= 1
                && s.upper <= 3,
            BAD_REGISTERS,
        )
    }
    fn restore_state(&mut self, state: &CartState) {
        let CartState::Mmm01(s) = state else {
            panic!("{WRONG_MAPPER}");
        };
        self.state.clone_from(s);
    }
}
