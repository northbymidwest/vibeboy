/// Game Boy Printer emulation.
///
/// Implements the SerialDevice trait to receive print data over the
/// serial port and render completed prints. Output mode determines
/// whether images are saved as PNG files (native) or queued as RGBA
/// pixel data in memory (web).

use crate::serial::SerialDevice;

#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

const PRINTER_DATA_SIZE: usize = 0x280;
const PRINTER_MAX_COMMAND_LENGTH: usize = 0x280;
/// Maximum image size: 160 pixels wide x 200 pixels tall (25 strips of 2 tile rows)
const PRINTER_IMAGE_SIZE: usize = 160 * 200;

const COMMAND_INIT: u8 = 0x01;
const COMMAND_START: u8 = 0x02;
const COMMAND_DATA: u8 = 0x04;

#[derive(Debug, Clone, Copy, PartialEq)]
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

impl PartialOrd for CommandState {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some((*self as u8).cmp(&(*other as u8)))
    }
}

/// Determines where completed prints are sent.
pub enum PrintOutput {
    /// Save PNGs to disk (native only).
    #[cfg(not(target_arch = "wasm32"))]
    File { output_dir: PathBuf },
    /// Queue completed prints as (RGBA pixels, width, height) tuples in memory.
    Memory,
}

pub struct Printer {
    command_state: CommandState,
    command_id: u8,
    compression: bool,
    length_left: u16,
    command_length: usize,
    command_data: [u8; PRINTER_MAX_COMMAND_LENGTH],
    checksum: u16,
    status: u8,

    /// 2-bit pixel image buffer (160 x up to 200)
    image: [u8; PRINTER_IMAGE_SIZE],
    image_offset: usize,

    /// Bit accumulation for incoming serial byte
    byte_being_received: u8,
    bits_received: u8,

    /// Response byte and current bit to send back
    byte_to_send: u8,
    bit_to_send: bool,

    /// RLE decompression state
    compression_run_length: u8,
    compression_run_is_compressed: bool,

    /// Idle time counter (in clock ticks) for protocol timeout
    idle_time: u32,
    /// Print timer (clock ticks remaining)
    time_remaining: u32,

    /// Output mode
    output: PrintOutput,
    /// Counter for naming output files (File mode only)
    print_count: u32,
    /// CPU clock rate for timing calculations
    clock_rate: u32,

    /// Completed prints waiting for retrieval (Memory mode only)
    pending_prints: Vec<(Vec<u8>, u32, u32)>,
}

impl Printer {
    /// Create a printer that saves PNGs to the given directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(output_dir: &Path, clock_rate: u32) -> Self {
        Self::with_output(PrintOutput::File { output_dir: output_dir.to_path_buf() }, clock_rate)
    }

    /// Create a printer that queues completed prints in memory.
    pub fn new_memory(clock_rate: u32) -> Self {
        Self::with_output(PrintOutput::Memory, clock_rate)
    }

    fn with_output(output: PrintOutput, clock_rate: u32) -> Self {
        Printer {
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
            output,
            print_count: 0,
            clock_rate,
            pending_prints: Vec::new(),
        }
    }

    /// Returns true if there are completed prints waiting for retrieval (Memory mode).
    pub fn has_pending_print(&self) -> bool {
        !self.pending_prints.is_empty()
    }

    /// Take the next completed print as (RGBA pixels, width, height).
    pub fn take_print(&mut self) -> Option<(Vec<u8>, u32, u32)> {
        if self.pending_prints.is_empty() {
            None
        } else {
            Some(self.pending_prints.remove(0))
        }
    }

    fn byte_receive_completed(&mut self, byte_received: u8) {
        self.byte_to_send = 0;

        match self.command_state {
            CommandState::Magic1 => {
                if byte_received != 0x88 {
                    return;
                }
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
                    self.status |= 1; // Checksum error
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
                        self.status = 4; // Done
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

                    // Calculate print time: 1 second per 8-pixel row
                    let rows = self.image_offset / 160;
                    self.time_remaining = (rows as u32) * self.clock_rate / 256 / 8;

                    self.output_image();
                    self.image_offset = 0;
                }
            }
            COMMAND_DATA => {
                if self.command_length == PRINTER_DATA_SIZE {
                    self.image_offset %= PRINTER_IMAGE_SIZE;
                    self.status = 8; // Received full data block

                    // Decode 2bpp tile data into image buffer
                    // 0x280 bytes = 2 rows of 20 tiles (each tile 8x8, 16 bytes)
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
            _ => {} // NOP and unknown
        }
        self.command_length = 0;
    }

    /// Render the 2-bit image buffer to RGBA pixels using the print palette.
    fn render_rgba(&self) -> Option<(Vec<u8>, u32, u32)> {
        let height = self.image_offset / 160;
        if height == 0 {
            return None;
        }

        // Get palette from PRINT command data byte 2
        let palette = if self.command_length >= 3 {
            self.command_data[2]
        } else {
            0xE4 // default: 0,1,2,3
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

        Some((rgba, width, h))
    }

    /// Output the completed print image according to the configured output mode.
    fn output_image(&mut self) {
        #[cfg(all(not(target_arch = "wasm32"), feature = "native"))]
        if let PrintOutput::File { ref output_dir } = self.output {
            let dir = output_dir.clone();
            self.save_image_to_file(&dir);
            return;
        }

        if let PrintOutput::Memory = self.output {
            if let Some(print) = self.render_rgba() {
                self.pending_prints.push(print);
            }
        }
    }

    /// Save the image as a grayscale PNG file (native only).
    #[cfg(all(not(target_arch = "wasm32"), feature = "native"))]
    fn save_image_to_file(&mut self, output_dir: &std::path::Path) {
        let height = self.image_offset / 160;
        if height == 0 {
            return;
        }

        // Get palette from PRINT command data byte 2
        let palette = if self.command_length >= 3 {
            self.command_data[2]
        } else {
            0xE4 // default: 0,1,2,3
        };

        // Map 2-bit values through palette to grayscale
        let gray_levels: [u8; 4] = [0xFF, 0xAA, 0x55, 0x00];
        let width = 160;
        let pixels: Vec<u8> = self.image[..self.image_offset]
            .iter()
            .map(|&px| {
                let mapped = (palette >> (px * 2)) & 3;
                gray_levels[mapped as usize]
            })
            .collect();

        // Ensure output directory exists
        if let Err(e) = std::fs::create_dir_all(output_dir) {
            log::error!("Failed to create printer output dir: {}", e);
            return;
        }

        let path = output_dir.join(format!("print_{:04}.png", self.print_count));
        self.print_count += 1;

        // Create grayscale PNG using the image crate
        let img = image::GrayImage::from_raw(width as u32, height as u32, pixels);
        match img {
            Some(img) => {
                if let Err(e) = img.save(&path) {
                    log::error!("Failed to save printer image: {}", e);
                } else {
                    log::info!("Printer: saved {}x{} image to {}", width, height, path.display());
                }
            }
            None => {
                log::error!("Printer: failed to create image buffer");
            }
        }
    }
}

impl SerialDevice for Printer {
    fn bit_start(&mut self, bit: bool) {
        // Reset protocol if idle too long (> 1 second)
        if self.idle_time > self.clock_rate {
            self.command_state = CommandState::Magic1;
            self.bits_received = 0;
        }
        self.idle_time = 0;

        // Accumulate bit into byte (MSB first)
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

    fn tick(&mut self, ticks: u32) {
        // Accumulate idle time when protocol is active
        if self.command_state != CommandState::Magic1 || self.bits_received > 0 {
            self.idle_time += ticks;
        }

        // Decrement print timer
        if self.time_remaining > 0 {
            if self.time_remaining <= ticks {
                self.time_remaining = 0;
            } else {
                self.time_remaining -= ticks;
            }
        }
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}
