use std::sync::Arc;
use super::Cartridge;
use crate::clock::Clock;

pub struct HuC3 {
    rom: Arc<[u8]>,
    ram: Vec<u8>,
    rom_bank: usize,
    ram_bank: usize,
    mode: u8,           // $0000-$1FFF write selects mode
    rtc_mem: [u8; 128], // nybble-addressed RTC/internal memory
    rtc_addr: u8,       // current RTC address pointer
    rtc_data_out: u8,   // response nybble for reads
    rtc_minutes: u32,   // minutes in day (0-1439)
    rtc_days: u32,      // day counter
    rtc_last_secs: u64,
    clock: Arc<dyn Clock>,
}

impl HuC3 {
    pub(super) fn new(rom: Arc<[u8]>, ram_size: usize, clock: Arc<dyn Clock>) -> Self {
        let rtc_last_secs = clock.now_secs();
        HuC3 {
            rom,
            ram: vec![0u8; ram_size.max(0x2000)],
            rom_bank: 1,
            ram_bank: 0,
            mode: 0,
            rtc_mem: [0; 128],
            rtc_addr: 0,
            rtc_data_out: 0,
            rtc_minutes: 0,
            rtc_days: 0,
            rtc_last_secs,
            clock,
        }
    }

    fn advance_rtc(&mut self) {
        let now = self.clock.now_secs();
        let elapsed = now.saturating_sub(self.rtc_last_secs);
        self.rtc_last_secs = now;
        let elapsed_mins = elapsed / 60;
        if elapsed_mins == 0 { return; }
        self.rtc_minutes += elapsed_mins as u32;
        self.rtc_days += self.rtc_minutes / 1440;
        self.rtc_minutes %= 1440;
    }

    fn latch_time_to_mem(&mut self) {
        self.advance_rtc();
        let mins = self.rtc_minutes;
        let days = self.rtc_days;
        // Minutes at addrs $10-$12 (nybble-addressed, 12-bit BCD-ish)
        self.rtc_mem[0x10] = (mins & 0x0F) as u8;
        self.rtc_mem[0x11] = ((mins >> 4) & 0x0F) as u8;
        self.rtc_mem[0x12] = ((mins >> 8) & 0x0F) as u8;
        // Days at addrs $13-$15
        self.rtc_mem[0x13] = (days & 0x0F) as u8;
        self.rtc_mem[0x14] = ((days >> 4) & 0x0F) as u8;
        self.rtc_mem[0x15] = ((days >> 8) & 0x0F) as u8;
    }

    fn set_time_from_mem(&mut self) {
        self.rtc_minutes = (self.rtc_mem[0x10] as u32)
            | ((self.rtc_mem[0x11] as u32) << 4)
            | ((self.rtc_mem[0x12] as u32) << 8);
        self.rtc_days = (self.rtc_mem[0x13] as u32)
            | ((self.rtc_mem[0x14] as u32) << 4)
            | ((self.rtc_mem[0x15] as u32) << 8);
        self.rtc_last_secs = self.clock.now_secs();
    }

    fn handle_rtc_cmd(&mut self, cmd_byte: u8) {
        let cmd = (cmd_byte >> 4) & 0x07;
        let arg = cmd_byte & 0x0F;
        match cmd {
            0x1 => {
                // Read + increment address
                let addr = (self.rtc_addr as usize) & 0x7F;
                self.rtc_data_out = self.rtc_mem[addr] & 0x0F;
                self.rtc_addr = self.rtc_addr.wrapping_add(1) & 0x7F;
            }
            0x3 => {
                // Write + increment address
                let addr = (self.rtc_addr as usize) & 0x7F;
                self.rtc_mem[addr] = arg & 0x0F;
                self.rtc_addr = self.rtc_addr.wrapping_add(1) & 0x7F;
            }
            0x4 => {
                // Set address low nybble
                self.rtc_addr = (self.rtc_addr & 0x70) | (arg & 0x0F);
            }
            0x5 => {
                // Set address high nybble
                self.rtc_addr = (self.rtc_addr & 0x0F) | ((arg & 0x07) << 4);
            }
            0x6 => {
                // Extended command
                match arg {
                    0x0 => self.latch_time_to_mem(),
                    0x1 => self.set_time_from_mem(),
                    0x2 => self.rtc_data_out = 0x01, // status: ready
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

impl Cartridge for HuC3 {
    fn read_rom(&self, addr: u16) -> u8 {
        let idx = match addr {
            0x0000..=0x3FFF => addr as usize,
            0x4000..=0x7FFF => self.rom_bank * 0x4000 + (addr as usize - 0x4000),
            _ => return 0xFF,
        };
        self.rom.get(idx % self.rom.len().max(1)).copied().unwrap_or(0xFF)
    }

    fn write_rom(&mut self, addr: u16, val: u8) {
        match addr {
            0x0000..=0x1FFF => self.mode = val,
            0x2000..=0x3FFF => {
                let b = (val & 0x7F) as usize;
                self.rom_bank = if b == 0 { 1 } else { b };
            }
            0x4000..=0x5FFF => self.ram_bank = (val & 0x03) as usize,
            _ => {}
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        match self.mode {
            0x00 | 0x0A => {
                let idx = self.ram_bank * 0x2000 + (addr as usize - 0xA000);
                self.ram.get(idx % self.ram.len().max(1)).copied().unwrap_or(0xFF)
            }
            0x0C => {
                // RTC response read
                0x80 | (self.rtc_data_out & 0x0F)
            }
            0x0D => 0x01, // Semaphore: always ready
            0x0E => 0xC0, // IR stub: no signal
            _ => 0xFF,
        }
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        match self.mode {
            0x0A => {
                let idx = self.ram_bank * 0x2000 + (addr as usize - 0xA000);
                let len = self.ram.len().max(1);
                self.ram[idx % len] = val;
            }
            0x0B => {
                // RTC command write
                self.handle_rtc_cmd(val);
            }
            _ => {}
        }
    }

    fn has_battery(&self) -> bool { true }
    fn ram_data(&self) -> &[u8] { &self.ram }

    fn save_data(&self) -> Vec<u8> {
        let mut data = self.ram.clone();
        // Append RTC state: rtc_mem (128 bytes) + minutes (4 bytes LE) + days (4 bytes LE) + timestamp (8 bytes LE)
        data.extend_from_slice(&self.rtc_mem);
        data.extend_from_slice(&self.rtc_minutes.to_le_bytes());
        data.extend_from_slice(&self.rtc_days.to_le_bytes());
        let ts = self.clock.unix_timestamp_secs() as i64;
        data.extend_from_slice(&ts.to_le_bytes());
        data
    }

    fn load_ram(&mut self, data: &[u8]) {
        let ram_len = self.ram.len();
        let copy_len = ram_len.min(data.len());
        self.ram[..copy_len].copy_from_slice(&data[..copy_len]);

        // Load RTC state
        if data.len() >= ram_len + 128 + 4 + 4 + 8 {
            let rtc_start = ram_len;
            self.rtc_mem.copy_from_slice(&data[rtc_start..rtc_start + 128]);
            let mut buf4 = [0u8; 4];
            buf4.copy_from_slice(&data[rtc_start + 128..rtc_start + 132]);
            self.rtc_minutes = u32::from_le_bytes(buf4);
            buf4.copy_from_slice(&data[rtc_start + 132..rtc_start + 136]);
            self.rtc_days = u32::from_le_bytes(buf4);
            let mut buf8 = [0u8; 8];
            buf8.copy_from_slice(&data[rtc_start + 136..rtc_start + 144]);
            let saved_ts = i64::from_le_bytes(buf8);
            let now_ts = self.clock.unix_timestamp_secs() as i64;
            let elapsed_mins = ((now_ts - saved_ts).max(0) as u64) / 60;
            self.rtc_minutes += elapsed_mins as u32;
            self.rtc_days += self.rtc_minutes / 1440;
            self.rtc_minutes %= 1440;
            self.rtc_last_secs = self.clock.now_secs();
        }
    }
    fn snapshot_state(&self) -> Vec<u8> {
        let mut s = Vec::new();
        s.extend_from_slice(&(self.rom_bank as u32).to_le_bytes());
        s.extend_from_slice(&(self.ram_bank as u32).to_le_bytes());
        s.push(self.mode);
        s.extend_from_slice(&self.rtc_mem);
        s.push(self.rtc_addr);
        s.push(self.rtc_data_out);
        s.extend_from_slice(&self.rtc_minutes.to_le_bytes());
        s.extend_from_slice(&self.rtc_days.to_le_bytes());
        s.extend_from_slice(&self.ram);
        s
    }
    fn restore_state(&mut self, d: &[u8]) {
        if d.len() < 9 + 128 + 2 + 8 { return; }
        self.rom_bank = u32::from_le_bytes([d[0],d[1],d[2],d[3]]) as usize;
        self.ram_bank = u32::from_le_bytes([d[4],d[5],d[6],d[7]]) as usize;
        self.mode = d[8];
        self.rtc_mem.copy_from_slice(&d[9..137]);
        self.rtc_addr = d[137];
        self.rtc_data_out = d[138];
        self.rtc_minutes = u32::from_le_bytes([d[139],d[140],d[141],d[142]]);
        self.rtc_days = u32::from_le_bytes([d[143],d[144],d[145],d[146]]);
        self.rtc_last_secs = self.clock.now_secs();
        let ram = &d[147..];
        let len = self.ram.len().min(ram.len());
        self.ram[..len].copy_from_slice(&ram[..len]);
    }
}
