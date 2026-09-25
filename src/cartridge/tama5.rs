use super::{BAD_REGISTERS, CartState, Cartridge, WRONG_MAPPER, ensure};
use crate::clock::Clock;
use std::sync::Arc;

pub struct Tama5 {
    rom: Arc<[u8]>,
    rtc_last_secs: u64,
    clock: Arc<dyn Clock>,
    state: Tama5State,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Tama5State {
    rom_bank: usize,
    tama_ram: [u8; 32], // 32 bytes internal RAM
    reg_select: u8,
    data_in_lo: u8, // reg $04: write value low nybble
    data_in_hi: u8, // reg $05: write value high nybble
    addr_hi: u8,    // reg $06: bit 0 = address bit 4, bits 1-3 = command
    addr_lo: u8,    // reg $07: address low nybble, write executes the command
    data_out_lo: u8,
    data_out_hi: u8,
    // RTC: TC8521AM (simplified)
    #[serde(with = "serde_big_array::BigArray")]
    rtc_regs: [u8; 52], // 4 pages x 13 nybble registers
    rtc_seconds: u32,
}

impl Tama5 {
    pub(super) fn new(rom: Arc<[u8]>, clock: Arc<dyn Clock>) -> Self {
        let rtc_last_secs = clock.now_secs();
        Tama5 {
            rom,
            rtc_last_secs,
            clock,
            state: Tama5State {
                rom_bank: 1,
                tama_ram: [0; 32],
                reg_select: 0,
                data_in_lo: 0,
                data_in_hi: 0,
                addr_hi: 0,
                addr_lo: 0,
                data_out_lo: 0,
                data_out_hi: 0,
                rtc_regs: [0; 52],
                rtc_seconds: 0,
            },
        }
    }

    fn advance_rtc(&mut self) {
        let now = self.clock.now_secs();
        let elapsed = now.saturating_sub(self.rtc_last_secs);
        self.rtc_last_secs = now;
        self.add_seconds(elapsed);
    }

    fn add_seconds(&mut self, secs: u64) {
        let secs = u32::try_from(secs).unwrap_or(u32::MAX);
        self.state.rtc_seconds = self.state.rtc_seconds.saturating_add(secs);
    }

    fn execute_command(&mut self) {
        let addr = (((self.state.addr_hi & 0x01) << 4) | self.state.addr_lo) as usize;
        let cmd_type = self.state.addr_hi >> 1;

        match cmd_type {
            0x00 => {
                // RAM write: value comes from regs $04/$05
                self.state.tama_ram[addr] = (self.state.data_in_hi << 4) | self.state.data_in_lo;
            }
            0x01 => {
                // RAM read
                let val = self.state.tama_ram[addr];
                self.state.data_out_lo = val & 0x0F;
                self.state.data_out_hi = val >> 4;
            }
            0x02 => {
                // MCU commands (time get/set), simplified
                self.advance_rtc();
                self.state.data_out_lo = 0;
                self.state.data_out_hi = 0;
            }
            0x04 => {
                // RTC register access
                self.state.data_out_lo = 0;
                self.state.data_out_hi = 0;
            }
            _ => {}
        }
    }
}

impl Cartridge for Tama5 {
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

    fn write_rom(&mut self, _addr: u16, _val: u8) {}

    fn read_ram(&self, addr: u16) -> u8 {
        if addr == 0xA000 {
            // Data port read: return data out based on reg_select
            match self.state.reg_select {
                0x0C => self.state.data_out_lo & 0x0F,
                0x0D => self.state.data_out_hi & 0x0F,
                _ => 0x01, // ready flag
            }
        } else {
            0xFF
        }
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        let val = val & 0x0F;
        if addr == 0xA001 {
            // Register select
            self.state.reg_select = val;
        } else if addr == 0xA000 {
            match self.state.reg_select {
                0x00 => {
                    // ROM bank low nybble
                    self.state.rom_bank = (self.state.rom_bank & 0xF0) | (val as usize);
                    if self.state.rom_bank == 0 {
                        self.state.rom_bank = 1;
                    }
                }
                0x01 => {
                    // ROM bank high nybble
                    self.state.rom_bank = (self.state.rom_bank & 0x0F) | ((val as usize) << 4);
                    if self.state.rom_bank == 0 {
                        self.state.rom_bank = 1;
                    }
                }
                0x04 => self.state.data_in_lo = val,
                0x05 => self.state.data_in_hi = val,
                0x06 => self.state.addr_hi = val,
                0x07 => {
                    // Address low: triggers execution
                    self.state.addr_lo = val;
                    self.execute_command();
                }
                _ => {}
            }
        }
    }

    fn has_battery(&self) -> bool {
        true
    }

    fn save_data(&self) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&self.state.tama_ram);
        data.extend_from_slice(&self.state.rtc_regs);
        let ts = self.clock.unix_timestamp_secs() as i64;
        data.extend_from_slice(&ts.to_le_bytes());
        data
    }

    fn load_ram(&mut self, data: &[u8]) {
        if data.len() >= 32 {
            self.state.tama_ram.copy_from_slice(&data[..32]);
        }
        if data.len() >= 32 + 52 {
            self.state.rtc_regs.copy_from_slice(&data[32..84]);
        }
        if data.len() >= 32 + 52 + 8 {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&data[84..92]);
            let saved_ts = u64::try_from(i64::from_le_bytes(buf)).unwrap_or(0);
            if saved_ts != 0 {
                let elapsed = self.clock.unix_timestamp_secs().saturating_sub(saved_ts);
                self.add_seconds(elapsed);
            }
        }
        self.rtc_last_secs = self.clock.now_secs();
    }
    fn snapshot_state(&self) -> CartState {
        CartState::Tama5(self.state.clone())
    }
    fn validate_state(&self, state: &CartState) -> Result<(), &'static str> {
        let CartState::Tama5(s) = state else {
            return Err(WRONG_MAPPER);
        };
        // Every register write is masked to a nybble.
        let nybbles = [
            s.reg_select,
            s.data_in_lo,
            s.data_in_hi,
            s.addr_hi,
            s.addr_lo,
            s.data_out_lo,
            s.data_out_hi,
        ];
        ensure(
            s.rom_bank <= 0xFF && nybbles.iter().all(|&n| n <= 0x0F),
            BAD_REGISTERS,
        )
    }
    fn restore_state(&mut self, state: &CartState) {
        let CartState::Tama5(s) = state else {
            panic!("{WRONG_MAPPER}");
        };
        self.state.clone_from(s);
        // The RTC counts on from the restored registers.
        self.rtc_last_secs = self.clock.now_secs();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedClock;

    impl Clock for FixedClock {
        fn now_secs(&self) -> u64 {
            0
        }
        fn unix_timestamp_secs(&self) -> u64 {
            0
        }
    }

    #[test]
    fn validate_rejects_registers_wider_than_a_nybble() {
        let rom: Arc<[u8]> = vec![0u8; 0x8000].into();
        let c = Tama5::new(rom, Arc::new(FixedClock));
        let mut s = c.state.clone();
        assert!(c.validate_state(&CartState::Tama5(s.clone())).is_ok());
        // The command address indexes the 32-byte RAM, so it must stay in range.
        s.addr_lo = 0x10;
        assert_eq!(c.validate_state(&CartState::Tama5(s)), Err(BAD_REGISTERS));
    }
}
