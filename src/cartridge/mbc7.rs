use std::sync::Arc;
use super::Cartridge;

#[derive(Clone, Copy, PartialEq)]
enum EepromState {
    Idle,
    Command,
    Address,
    ReadData,
    WriteData,
}

pub struct Mbc7 {
    rom: Arc<[u8]>,
    rom_bank: usize,
    enable_a: bool,
    enable_b: bool,
    accel_latched: bool,
    accel_x: u16,
    accel_y: u16,
    /// Live accelerometer values set by external sensor; latched on write 0xAA.
    sensor_x: u16,
    sensor_y: u16,
    // EEPROM 93LC56: 128 × 16-bit words
    eeprom: [u16; 128],
    eeprom_cs: bool,
    eeprom_clk: bool,
    eeprom_di: bool,
    eeprom_do: bool,
    eeprom_state: EepromState,
    eeprom_cmd: u8,
    eeprom_addr: u8,
    eeprom_shift: u16,
    eeprom_bit_count: u8,
    eeprom_write_enable: bool,
}

impl Mbc7 {
    pub(super) fn new(rom: Arc<[u8]>) -> Self {
        Mbc7 {
            rom,
            rom_bank: 1,
            enable_a: false,
            enable_b: false,
            accel_latched: false,
            accel_x: 0x8000,
            accel_y: 0x8000,
            sensor_x: 0x81D0,
            sensor_y: 0x81D0,
            eeprom: [0xFFFF; 128],
            eeprom_cs: false,
            eeprom_clk: false,
            eeprom_di: false,
            eeprom_do: true,
            eeprom_state: EepromState::Idle,
            eeprom_cmd: 0,
            eeprom_addr: 0,
            eeprom_shift: 0,
            eeprom_bit_count: 0,
            eeprom_write_enable: false,
        }
    }

    fn eeprom_clock_rise(&mut self) {
        match self.eeprom_state {
            EepromState::Idle => {
                if self.eeprom_di {
                    // Start bit received
                    self.eeprom_state = EepromState::Command;
                    self.eeprom_cmd = 0;
                    self.eeprom_bit_count = 0;
                }
            }
            EepromState::Command => {
                self.eeprom_cmd = (self.eeprom_cmd << 1) | (self.eeprom_di as u8);
                self.eeprom_bit_count += 1;
                if self.eeprom_bit_count == 2 {
                    self.eeprom_state = EepromState::Address;
                    self.eeprom_addr = 0;
                    self.eeprom_bit_count = 0;
                }
            }
            EepromState::Address => {
                self.eeprom_addr = (self.eeprom_addr << 1) | (self.eeprom_di as u8);
                self.eeprom_bit_count += 1;
                if self.eeprom_bit_count == 7 {
                    self.eeprom_bit_count = 0;
                    match self.eeprom_cmd {
                        0b10 => {
                            // READ
                            self.eeprom_shift = self.eeprom[self.eeprom_addr as usize & 0x7F];
                            self.eeprom_state = EepromState::ReadData;
                            self.eeprom_do = false; // dummy 0 bit before data
                        }
                        0b01 => {
                            // WRITE
                            if self.eeprom_write_enable {
                                self.eeprom_shift = 0;
                                self.eeprom_state = EepromState::WriteData;
                            } else {
                                self.eeprom_state = EepromState::Idle;
                            }
                        }
                        0b11 => {
                            // ERASE
                            if self.eeprom_write_enable {
                                self.eeprom[self.eeprom_addr as usize & 0x7F] = 0xFFFF;
                            }
                            self.eeprom_do = true;
                            self.eeprom_state = EepromState::Idle;
                        }
                        0b00 => {
                            // Special: top 2 bits of address select sub-command
                            match self.eeprom_addr >> 5 {
                                0b00 => self.eeprom_write_enable = false, // EWDS
                                0b01 => {
                                    // WRAL
                                    if self.eeprom_write_enable {
                                        self.eeprom_shift = 0;
                                        self.eeprom_state = EepromState::WriteData;
                                    } else {
                                        self.eeprom_state = EepromState::Idle;
                                    }
                                }
                                0b10 => {
                                    // ERAL
                                    if self.eeprom_write_enable {
                                        self.eeprom = [0xFFFF; 128];
                                    }
                                    self.eeprom_state = EepromState::Idle;
                                }
                                0b11 | _ => {
                                    self.eeprom_write_enable = true; // EWEN
                                    self.eeprom_state = EepromState::Idle;
                                }
                            }
                        }
                        _ => self.eeprom_state = EepromState::Idle,
                    }
                }
            }
            EepromState::ReadData => {
                self.eeprom_do = (self.eeprom_shift >> 15) != 0;
                self.eeprom_shift <<= 1;
                self.eeprom_bit_count += 1;
                if self.eeprom_bit_count == 16 {
                    self.eeprom_bit_count = 0;
                    // Auto-increment address for sequential read
                    self.eeprom_addr = (self.eeprom_addr + 1) & 0x7F;
                    self.eeprom_shift = self.eeprom[self.eeprom_addr as usize];
                }
            }
            EepromState::WriteData => {
                self.eeprom_shift = (self.eeprom_shift << 1) | (self.eeprom_di as u16);
                self.eeprom_bit_count += 1;
                if self.eeprom_bit_count == 16 {
                    if self.eeprom_cmd == 0b01 {
                        // WRITE single
                        self.eeprom[self.eeprom_addr as usize & 0x7F] = self.eeprom_shift;
                    } else {
                        // WRAL
                        self.eeprom = [self.eeprom_shift; 128];
                    }
                    self.eeprom_do = true;
                    self.eeprom_state = EepromState::Idle;
                    self.eeprom_bit_count = 0;
                }
            }
        }
    }
}

impl Cartridge for Mbc7 {
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
            0x0000..=0x1FFF => self.enable_a = val == 0x0A,
            0x2000..=0x3FFF => {
                self.rom_bank = (val as usize) & 0xFF;
                if self.rom_bank == 0 { self.rom_bank = 1; }
            }
            0x4000..=0x5FFF => self.enable_b = val == 0x40,
            _ => {}
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        if !self.enable_a || !self.enable_b { return 0xFF; }
        match (addr >> 4) & 0x0F {
            0x0 | 0x1 => 0, // write-only
            0x2 => self.accel_x as u8,
            0x3 => (self.accel_x >> 8) as u8,
            0x4 => self.accel_y as u8,
            0x5 => (self.accel_y >> 8) as u8,
            0x6 => 0x00,
            0x7 => 0xFF,
            0x8 => (self.eeprom_do as u8) | ((self.eeprom_di as u8) << 1) |
                   ((self.eeprom_clk as u8) << 6) | ((self.eeprom_cs as u8) << 7),
            _ => 0xFF,
        }
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        if !self.enable_a || !self.enable_b { return; }
        match (addr >> 4) & 0x0F {
            0x0 => {
                if val == 0x55 {
                    self.accel_x = 0x8000;
                    self.accel_y = 0x8000;
                    self.accel_latched = false;
                }
            }
            0x1 => {
                if val == 0xAA && !self.accel_latched {
                    self.accel_x = self.sensor_x;
                    self.accel_y = self.sensor_y;
                    self.accel_latched = true;
                }
            }
            0x8 => {
                let new_cs = val & 0x80 != 0;
                let new_clk = val & 0x40 != 0;
                self.eeprom_di = val & 0x02 != 0;

                if !new_cs {
                    // CS low: reset
                    self.eeprom_state = EepromState::Idle;
                    self.eeprom_do = true;
                    self.eeprom_bit_count = 0;
                } else if new_clk && !self.eeprom_clk {
                    // Rising clock edge
                    self.eeprom_clock_rise();
                }

                self.eeprom_cs = new_cs;
                self.eeprom_clk = new_clk;
            }
            _ => {}
        }
    }

    fn has_battery(&self) -> bool { true }
    fn has_accelerometer(&self) -> bool { true }
    fn set_accelerometer(&mut self, x: u16, y: u16) {
        self.sensor_x = x;
        self.sensor_y = y;
    }

    fn save_data(&self) -> Vec<u8> {
        // Save EEPROM as 256 bytes (128 × 16-bit LE words)
        let mut data = vec![0u8; 256];
        for (i, &word) in self.eeprom.iter().enumerate() {
            data[i * 2] = word as u8;
            data[i * 2 + 1] = (word >> 8) as u8;
        }
        data
    }

    fn load_ram(&mut self, data: &[u8]) {
        let words = data.len().min(256) / 2;
        for i in 0..words {
            self.eeprom[i] = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        }
    }
    fn snapshot_state(&self) -> Vec<u8> {
        let mut s = Vec::new();
        s.extend_from_slice(&(self.rom_bank as u32).to_le_bytes());
        s.push(self.enable_a as u8);
        s.push(self.enable_b as u8);
        s.push(self.accel_latched as u8);
        s.extend_from_slice(&self.accel_x.to_le_bytes());
        s.extend_from_slice(&self.accel_y.to_le_bytes());
        // EEPROM: 128 words × 2 bytes
        for &w in &self.eeprom {
            s.extend_from_slice(&w.to_le_bytes());
        }
        s.push(self.eeprom_cs as u8);
        s.push(self.eeprom_clk as u8);
        s.push(self.eeprom_di as u8);
        s.push(self.eeprom_do as u8);
        s.push(self.eeprom_state as u8);
        s.push(self.eeprom_cmd);
        s.push(self.eeprom_addr);
        s.extend_from_slice(&self.eeprom_shift.to_le_bytes());
        s.push(self.eeprom_bit_count);
        s.push(self.eeprom_write_enable as u8);
        s
    }
    fn restore_state(&mut self, d: &[u8]) {
        if d.len() < 7 + 256 + 10 { return; }
        self.rom_bank = u32::from_le_bytes([d[0],d[1],d[2],d[3]]) as usize;
        self.enable_a = d[4] != 0;
        self.enable_b = d[5] != 0;
        self.accel_latched = d[6] != 0;
        self.accel_x = u16::from_le_bytes([d[7],d[8]]);
        self.accel_y = u16::from_le_bytes([d[9],d[10]]);
        let ee = &d[11..];
        for i in 0..128 {
            self.eeprom[i] = u16::from_le_bytes([ee[i*2], ee[i*2+1]]);
        }
        let r = &ee[256..];
        if r.len() >= 10 {
            self.eeprom_cs = r[0] != 0;
            self.eeprom_clk = r[1] != 0;
            self.eeprom_di = r[2] != 0;
            self.eeprom_do = r[3] != 0;
            self.eeprom_state = match r[4] {
                1 => EepromState::Command,
                2 => EepromState::Address,
                3 => EepromState::ReadData,
                4 => EepromState::WriteData,
                _ => EepromState::Idle,
            };
            self.eeprom_cmd = r[5];
            self.eeprom_addr = r[6];
            self.eeprom_shift = u16::from_le_bytes([r[7],r[8]]);
            self.eeprom_bit_count = r[9];
            if r.len() > 10 { self.eeprom_write_enable = r[10] != 0; }
        }
    }
}
