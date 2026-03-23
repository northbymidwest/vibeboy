use std::sync::Arc;
use super::{Cartridge, Instant, unix_timestamp_secs};

pub struct Tama5 {
    rom: Arc<[u8]>,
    rom_bank: usize,
    tama_ram: [u8; 32], // 32 nybbles internal RAM
    reg_select: u8,
    data_in_lo: u8,
    data_in_hi: u8,
    data_out_lo: u8,
    data_out_hi: u8,
    // RTC: TC8521AM — simplified
    rtc_regs: [u8; 52], // 4 pages × 13 nybble registers
    rtc_base: Instant,
    rtc_seconds: u32,
}

impl Tama5 {
    pub(super) fn new(rom: Arc<[u8]>) -> Self {
        Tama5 {
            rom,
            rom_bank: 1,
            tama_ram: [0; 32],
            reg_select: 0,
            data_in_lo: 0,
            data_in_hi: 0,
            data_out_lo: 0,
            data_out_hi: 0,
            rtc_regs: [0; 52],
            rtc_base: Instant::now(),
            rtc_seconds: 0,
        }
    }

    fn advance_rtc(&mut self) {
        let elapsed = self.rtc_base.elapsed().as_secs() as u32;
        self.rtc_base = Instant::now();
        self.rtc_seconds += elapsed;
    }

    fn execute_command(&mut self) {
        let cmd = (self.data_in_hi as u16) << 4 | (self.data_in_lo as u16);
        let addr = (cmd & 0x1F) as usize;
        let cmd_type = cmd >> 5;

        match cmd_type {
            0x00 => {
                // RAM write: uses data from regs $04/$05
                let val = self.data_in_lo; // value was written to $04 before command
                if addr < 32 {
                    self.tama_ram[addr] = val & 0x0F;
                }
            }
            0x01 => {
                // RAM read
                if addr < 32 {
                    let val = self.tama_ram[addr] & 0x0F;
                    self.data_out_lo = val & 0x0F;
                    self.data_out_hi = 0;
                }
            }
            0x02..=0x03 => {
                // MCU commands (time get/set) — simplified
                self.advance_rtc();
                self.data_out_lo = 0;
                self.data_out_hi = 0;
            }
            0x04..=0x05 => {
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
        self.rom.get(idx % self.rom.len().max(1)).copied().unwrap_or(0xFF)
    }

    fn write_rom(&mut self, _addr: u16, _val: u8) {}

    fn read_ram(&self, addr: u16) -> u8 {
        if addr == 0xA000 {
            // Data port read — return data out based on reg_select
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
                    if self.rom_bank == 0 { self.rom_bank = 1; }
                }
                0x01 => {
                    // ROM bank high nybble
                    self.rom_bank = (self.rom_bank & 0x0F) | ((val as usize) << 4);
                    if self.rom_bank == 0 { self.rom_bank = 1; }
                }
                0x04 => self.data_in_lo = val,
                0x05 => self.data_in_hi = val,
                0x06 => {
                    // Command low
                    self.data_in_lo = val;
                }
                0x07 => {
                    // Command high — triggers execution
                    self.data_in_hi = val;
                    self.execute_command();
                }
                _ => {}
            }
        }
    }

    fn has_battery(&self) -> bool { true }

    fn save_data(&self) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&self.tama_ram);
        data.extend_from_slice(&self.rtc_regs);
        let ts = unix_timestamp_secs() as i64;
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
            let saved_ts = i64::from_le_bytes(buf);
            let now_ts = unix_timestamp_secs() as i64;
            let elapsed = (now_ts - saved_ts).max(0) as u32;
            self.rtc_seconds += elapsed;
        }
        self.rtc_base = Instant::now();
    }
    fn snapshot_state(&self) -> Vec<u8> {
        let mut s = Vec::new();
        s.extend_from_slice(&(self.rom_bank as u32).to_le_bytes());
        s.extend_from_slice(&self.tama_ram);
        s.push(self.reg_select);
        s.push(self.data_in_lo);
        s.push(self.data_in_hi);
        s.push(self.data_out_lo);
        s.push(self.data_out_hi);
        s.extend_from_slice(&self.rtc_regs);
        s.extend_from_slice(&self.rtc_seconds.to_le_bytes());
        s
    }
    fn restore_state(&mut self, d: &[u8]) {
        if d.len() < 4 + 32 + 5 + 52 + 4 { return; }
        self.rom_bank = u32::from_le_bytes([d[0],d[1],d[2],d[3]]) as usize;
        self.tama_ram.copy_from_slice(&d[4..36]);
        self.reg_select = d[36];
        self.data_in_lo = d[37];
        self.data_in_hi = d[38];
        self.data_out_lo = d[39];
        self.data_out_hi = d[40];
        self.rtc_regs.copy_from_slice(&d[41..93]);
        self.rtc_seconds = u32::from_le_bytes([d[93],d[94],d[95],d[96]]);
        self.rtc_base = Instant::now();
    }
}
