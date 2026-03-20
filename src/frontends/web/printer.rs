/// Web-compatible Game Boy Printer — matches native printer protocol exactly,
/// but stores completed prints as RGBA pixel data for JS download instead of
/// writing PNG files to disk.

use crate::serial::SerialDevice;

const PRINTER_DATA_SIZE: usize = 0x280;
const PRINTER_MAX_COMMAND_LENGTH: usize = 0x280;
const PRINTER_IMAGE_SIZE: usize = 160 * 200;

const COMMAND_INIT: u8 = 0x01;
const COMMAND_START: u8 = 0x02;
const COMMAND_DATA: u8 = 0x04;

#[derive(Clone, Copy, PartialEq, PartialOrd)]
enum CommandState {
    Magic1,
    Magic2,
    CommandId,
    Compression,
    LengthLow,
    LengthHigh,
    Data,
    ChecksumLow,
    ChecksumHigh,
    Active,
    Status,
}

impl CommandState {
    fn next(self) -> Self {
        match self {
            Self::Magic1 => Self::Magic2,
            Self::Magic2 => Self::CommandId,
            Self::CommandId => Self::Compression,
            Self::Compression => Self::LengthLow,
            Self::LengthLow => Self::LengthHigh,
            Self::LengthHigh => Self::Data,
            Self::Data => Self::ChecksumLow,
            Self::ChecksumLow => Self::ChecksumHigh,
            Self::ChecksumHigh => Self::Active,
            Self::Active => Self::Status,
            Self::Status => Self::Magic1,
        }
    }
}

pub struct WebPrinter {
    command_state: CommandState,
    command_id: u8,
    compression: bool,
    length_left: u16,
    command_length: usize,
    command_data: [u8; PRINTER_MAX_COMMAND_LENGTH],
    checksum: u16,
    status: u8,
    image: [u8; PRINTER_IMAGE_SIZE],
    image_offset: usize,
    byte_being_received: u8,
    bits_received: u8,
    byte_to_send: u8,
    bit_to_send: bool,
    compression_run_length: u8,
    compression_run_is_compressed: bool,
    idle_time: u32,
    time_remaining: u32,
    clock_rate: u32,
    pub pending_prints: Vec<(Vec<u8>, u32, u32)>,
}

impl WebPrinter {
    pub fn new(clock_rate: u32) -> Self {
        WebPrinter {
            command_state: CommandState::Magic1,
            command_id: 0,
            compression: false,
            length_left: 0,
            command_length: 0,
            command_data: [0; PRINTER_MAX_COMMAND_LENGTH],
            checksum: 0,
            status: 0,
            image: [0; PRINTER_IMAGE_SIZE],
            image_offset: 0,
            byte_being_received: 0,
            bits_received: 0,
            byte_to_send: 0,
            bit_to_send: false,
            compression_run_length: 0,
            compression_run_is_compressed: false,
            idle_time: 0,
            time_remaining: 0,
            clock_rate,
            pending_prints: Vec::new(),
        }
    }

    pub fn take_print(&mut self) -> Option<(Vec<u8>, u32, u32)> {
        if self.pending_prints.is_empty() { None }
        else { Some(self.pending_prints.remove(0)) }
    }

    pub fn has_pending_print(&self) -> bool {
        !self.pending_prints.is_empty()
    }

    // Mirrors native printer's byte_receive_completed exactly
    fn byte_receive_completed(&mut self, byte_received: u8) {
        self.byte_to_send = 0;

        match self.command_state {
            CommandState::Magic1 => {
                if byte_received != 0x88 { return; }
                self.status &= !1;
                self.command_length = 0;
                self.checksum = 0;
            }
            CommandState::Magic2 => {
                if byte_received != 0x33 {
                    if byte_received != 0x88 {
                        self.command_state = CommandState::Magic1;
                    }
                    return;
                }
            }
            CommandState::CommandId => {
                self.command_id = byte_received & 0x0F;
            }
            CommandState::Compression => {
                self.compression = byte_received & 1 != 0;
            }
            CommandState::LengthLow => {
                self.length_left = byte_received as u16;
            }
            CommandState::LengthHigh => {
                self.length_left |= ((byte_received & 3) as u16) << 8;
            }
            CommandState::Data => {
                if self.command_length < PRINTER_MAX_COMMAND_LENGTH {
                    if self.compression {
                        if self.compression_run_length == 0 {
                            self.compression_run_is_compressed = byte_received & 0x80 != 0;
                            self.compression_run_length = (byte_received & 0x7F) + 1
                                + if self.compression_run_is_compressed { 1 } else { 0 };
                        } else if self.compression_run_is_compressed {
                            while self.compression_run_length > 0 {
                                if self.command_length < PRINTER_MAX_COMMAND_LENGTH {
                                    self.command_data[self.command_length] = byte_received;
                                    self.command_length += 1;
                                }
                                self.compression_run_length -= 1;
                            }
                        } else {
                            self.command_data[self.command_length] = byte_received;
                            self.command_length += 1;
                            self.compression_run_length -= 1;
                        }
                    } else {
                        self.command_data[self.command_length] = byte_received;
                        self.command_length += 1;
                    }
                }
                self.length_left -= 1;
            }
            CommandState::ChecksumLow => {
                self.checksum ^= byte_received as u16;
            }
            CommandState::ChecksumHigh => {
                self.checksum ^= (byte_received as u16) << 8;
                if self.checksum != 0 {
                    self.status |= 1;
                    self.command_state = CommandState::Magic1;
                    return;
                }
                self.byte_to_send = 0x81;
            }
            CommandState::Active => {
                if self.command_id == COMMAND_INIT {
                    self.byte_to_send = 0;
                } else {
                    if self.status == 6 && self.time_remaining == 0 {
                        self.status = 4; // Done printing
                    }
                    self.byte_to_send = self.status;
                }
            }
            CommandState::Status => {
                self.command_state = CommandState::Magic1;
                self.handle_command();
                return;
            }
        }

        // Accumulate checksum for ID through Data
        if self.command_state >= CommandState::CommandId
            && self.command_state < CommandState::ChecksumLow
        {
            self.checksum = self.checksum.wrapping_add(byte_received as u16);
        }

        // Advance state
        if self.command_state != CommandState::Data {
            self.command_state = self.command_state.next();
        }

        // Skip Data state if length is 0
        if self.command_state == CommandState::Data && self.length_left == 0 {
            self.command_state = self.command_state.next();
        }
    }

    fn handle_command(&mut self) {
        match self.command_id {
            COMMAND_INIT => {
                self.status = 0;
                self.image_offset = 0;
            }
            COMMAND_START => {
                if self.command_length == 4 {
                    self.status = 6; // Printing
                    let rows = self.image_offset / 160;
                    self.time_remaining = (rows as u32) * self.clock_rate / 256 / 8;
                    self.save_image();
                    self.image_offset = 0;
                }
            }
            COMMAND_DATA => {
                if self.command_length == PRINTER_DATA_SIZE {
                    self.image_offset %= PRINTER_IMAGE_SIZE;
                    self.status = 8;

                    // Decode 2bpp tile data into image buffer
                    // 0x280 bytes = 2 rows of 20 tiles (each tile 8×8, 16 bytes)
                    let mut byte_idx = 0usize;
                    for _row in 0..2 {
                        for tile_x in 0..20 {
                            for y in 0..8 {
                                let lo = self.command_data[byte_idx];
                                let hi = self.command_data[byte_idx + 1];
                                byte_idx += 2;
                                for x_pixel in 0..8 {
                                    let shift = 7 - x_pixel;
                                    let color = ((lo >> shift) & 1)
                                        | (((hi >> shift) & 1) << 1);
                                    let idx = self.image_offset
                                        + tile_x * 8
                                        + x_pixel
                                        + y * 160;
                                    if idx < PRINTER_IMAGE_SIZE {
                                        self.image[idx] = color;
                                    }
                                }
                            }
                        }
                        self.image_offset += 8 * 160;
                    }
                }
            }
            _ => {}
        }
        self.command_length = 0;
    }

    fn save_image(&mut self) {
        let height = self.image_offset / 160;
        if height == 0 { return; }

        let palette = if self.command_length >= 3 {
            self.command_data[2]
        } else {
            0xE4
        };

        let gray_levels: [u8; 4] = [0xFF, 0xAA, 0x55, 0x00];
        let width = 160u32;
        let h = height as u32;

        let mut rgba = Vec::with_capacity((width * h * 4) as usize);
        for i in 0..(width * h) as usize {
            let px = self.image[i];
            let mapped = (palette >> (px * 2)) & 3;
            let g = gray_levels[mapped as usize];
            rgba.push(g);
            rgba.push(g);
            rgba.push(g);
            rgba.push(0xFF);
        }

        self.pending_prints.push((rgba, width, h));
    }
}

impl SerialDevice for WebPrinter {
    fn bit_start(&mut self, bit: bool) {
        if self.idle_time > self.clock_rate {
            self.command_state = CommandState::Magic1;
            self.bits_received = 0;
        }
        self.idle_time = 0;

        self.byte_being_received <<= 1;
        self.byte_being_received |= if bit { 1 } else { 0 };
        self.bits_received += 1;

        if self.bits_received == 8 {
            let byte = self.byte_being_received;
            self.byte_being_received = 0;
            self.bits_received = 0;
            self.byte_receive_completed(byte);
        }
    }

    fn bit_end(&mut self) -> bool {
        let ret = self.bit_to_send;
        self.bit_to_send = self.byte_to_send & 0x80 != 0;
        self.byte_to_send <<= 1;
        ret
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn tick(&mut self, ticks: u32) {
        if self.command_state != CommandState::Magic1 || self.bits_received > 0 {
            self.idle_time += ticks;
        }
        if self.time_remaining > 0 {
            if self.time_remaining <= ticks {
                self.time_remaining = 0;
            } else {
                self.time_remaining -= ticks;
            }
        }
    }
}
