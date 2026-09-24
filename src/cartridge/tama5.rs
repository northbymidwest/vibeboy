use super::Cartridge;
use crate::clock::Clock;
use std::sync::Arc;

pub struct Tama5 {
    rom: Arc<[u8]>,
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
    rtc_regs: [u8; 52], // 4 pages x 13 nybble registers
    rtc_last_secs: u64,
    rtc_seconds: u32,
    clock: Arc<dyn Clock>,
}

impl Tama5 {
    pub(super) fn new(rom: Arc<[u8]>, clock: Arc<dyn Clock>) -> Self {
        let rtc_last_secs = clock.now_secs();
        Tama5 {
            rom,
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
            rtc_last_secs,
            rtc_seconds: 0,
            clock,
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
        self.rtc_seconds = self.rtc_seconds.saturating_add(secs);
    }

    fn execute_command(&mut self) {
        let addr = (((self.addr_hi & 0x01) << 4) | self.addr_lo) as usize;
        let cmd_type = self.addr_hi >> 1;

        match cmd_type {
            0x00 => {
                // RAM write: value comes from regs $04/$05
                self.tama_ram[addr] = (self.data_in_hi << 4) | self.data_in_lo;
            }
            0x01 => {
                // RAM read
                let val = self.tama_ram[addr];
                self.data_out_lo = val & 0x0F;
                self.data_out_hi = val >> 4;
            }
            0x02 => {
                // MCU commands (time get/set), simplified
                self.advance_rtc();
                self.data_out_lo = 0;
                self.data_out_hi = 0;
            }
            0x04 => {
                // RTC register access
                self.data_out_lo = 0;
                self.data_out_hi = 0;
            }
            _ => {}
        }
    }
}

impl Cartridge for Tama5 {
    fn read_rom(&self, addr: u16) -> u8 {
        let idx = match addr {
            0x0000..=0x3FFF => addr as usize,
            0x4000..=0x7FFF => self.rom_bank * 0x4000 + (addr as usize - 0x4000),
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
            match self.reg_select {
                0x0C => self.data_out_lo & 0x0F,
                0x0D => self.data_out_hi & 0x0F,
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
            self.reg_select = val;
        } else if addr == 0xA000 {
            match self.reg_select {
                0x00 => {
                    // ROM bank low nybble
                    self.rom_bank = (self.rom_bank & 0xF0) | (val as usize);
                    if self.rom_bank == 0 {
                        self.rom_bank = 1;
                    }
                }
                0x01 => {
                    // ROM bank high nybble
                    self.rom_bank = (self.rom_bank & 0x0F) | ((val as usize) << 4);
                    if self.rom_bank == 0 {
                        self.rom_bank = 1;
                    }
                }
                0x04 => self.data_in_lo = val,
                0x05 => self.data_in_hi = val,
                0x06 => self.addr_hi = val,
                0x07 => {
                    // Address low: triggers execution
                    self.addr_lo = val;
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
        data.extend_from_slice(&self.tama_ram);
        data.extend_from_slice(&self.rtc_regs);
        let ts = self.clock.unix_timestamp_secs() as i64;
        data.extend_from_slice(&ts.to_le_bytes());
        data
    }

    fn load_ram(&mut self, data: &[u8]) {
        if data.len() >= 32 {
            self.tama_ram.copy_from_slice(&data[..32]);
        }
        if data.len() >= 32 + 52 {
            self.rtc_regs.copy_from_slice(&data[32..84]);
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
    fn snapshot_state(&self) -> Vec<u8> {
        let mut s = Vec::new();
        s.extend_from_slice(&(self.rom_bank as u32).to_le_bytes());
        s.extend_from_slice(&self.tama_ram);
        s.push(self.reg_select);
        s.push(self.data_in_lo);
        s.push(self.data_in_hi);
        s.push(self.addr_hi);
        s.push(self.addr_lo);
        s.push(self.data_out_lo);
        s.push(self.data_out_hi);
        s.extend_from_slice(&self.rtc_regs);
        s.extend_from_slice(&self.rtc_seconds.to_le_bytes());
        s
    }
    fn restore_state(&mut self, d: &[u8]) {
        if d.len() < 4 + 32 + 7 + 52 + 4 {
            return;
        }
        self.rom_bank = u32::from_le_bytes([d[0], d[1], d[2], d[3]]) as usize;
        self.tama_ram.copy_from_slice(&d[4..36]);
        self.reg_select = d[36];
        self.data_in_lo = d[37];
        self.data_in_hi = d[38];
        self.addr_hi = d[39];
        self.addr_lo = d[40];
        self.data_out_lo = d[41];
        self.data_out_hi = d[42];
        self.rtc_regs.copy_from_slice(&d[43..95]);
        self.rtc_seconds = u32::from_le_bytes([d[95], d[96], d[97], d[98]]);
        self.rtc_last_secs = self.clock.now_secs();
    }
}
