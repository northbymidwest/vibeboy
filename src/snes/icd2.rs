/// ICD2 bridge: connects the Game Boy side to the SNES side.
///
/// The ICD2 chip in the real SGB cartridge captures the GB's LCD output as
/// 2bpp tile rows, buffers command packets from the GB, and relays joypad
/// data from the SNES controllers to the GB.
///
/// SNES-side registers:
///   $6000 (R)   — row status / rows ready
///   $6001 (W)   — select which row to read
///   $6002 (R)   — packet ready flag
///   $6003 (W)   — control (reset/players/speed)
///   $6004-$6007 (W) — joypad data for players 1-4
///   $7000-$700F (R) — last command packet (16 bytes)
///   $7800-$7FFF (R) — pixel data for selected row (320 bytes used)

pub struct Icd2 {
    /// Pixel rows: each row is 160 pixels (2bpp shade values) × 2 for double-width = 320
    /// Ring buffer of 4 tile rows (each 8 scanlines = 160 tiles × 8 rows)
    /// Actually stores 320 bytes per row-group (20 tiles × 16 bytes per tile)
    pixel_buffer: Vec<u8>,
    /// How many complete 8-row tile-rows have been fed
    rows_fed: usize,
    /// Which row the SNES is currently reading ($6001)
    read_row_select: u8,
    /// Rows ready bitmask
    pub rows_ready: u8,

    /// Last command packet from GB (16 bytes)
    packet_data: [u8; 16],
    /// True when a new packet is available
    pub packet_ready: bool,

    /// Full scanline data: 144 lines × 160 pixels of 2-bit shades
    /// Fed from GB PPU shade buffer each frame
    scanline_data: Vec<u8>,
    /// Whether scanline data has been refreshed this frame
    scanlines_fresh: bool,

    /// Joypad data (SNES→GB), 4 players
    pub joy_data: [u16; 4],

    /// Control register ($6003)
    pub control: u8,
}

impl Icd2 {
    pub fn new() -> Self {
        Icd2 {
            // 4 row-groups × 320 bytes
            pixel_buffer: vec![0u8; 4 * 320],
            rows_fed: 0,
            read_row_select: 0,
            rows_ready: 0,
            packet_data: [0; 16],
            packet_ready: false,
            scanline_data: vec![0u8; 160 * 144],
            scanlines_fresh: false,
            joy_data: [0; 4],
            control: 0,
        }
    }

    /// Mutable access to packet data for initialization.
    pub fn packet_data_mut(&mut self) -> &mut [u8; 16] {
        &mut self.packet_data
    }

    /// Feed a 16-byte command packet from the GB side.
    pub fn feed_packet(&mut self, data: &[u8; 16]) {
        self.packet_data = *data;
        self.packet_ready = true;
    }

    /// Feed a full frame of scanline data (160×144 2-bit shades).
    /// This converts the shade buffer into the ICD2's tile-row format.
    pub fn feed_scanlines(&mut self, shade_buf: &[u8]) {
        if shade_buf.len() >= 160 * 144 {
            self.scanline_data.copy_from_slice(&shade_buf[..160 * 144]);
        }
        self.scanlines_fresh = true;

        // Convert scanlines into 4 tile-row groups for the ring buffer
        // Each tile row = 8 scanlines × 20 tiles × 16 bytes/tile (2bpp planar)
        // Total = 18 tile rows, but we present them 4 at a time via the ring buffer
        self.rows_fed = 0;
        self.rows_ready = 0;
        self.refresh_pixel_buffer();
    }

    /// Refresh the pixel buffer with the next 4 tile-row groups.
    fn refresh_pixel_buffer(&mut self) {
        // Fill all 4 row-group slots from scanline data
        for slot in 0..4 {
            let tile_row = self.rows_fed + slot;
            if tile_row >= 18 { break; }
            let buf_offset = slot * 320;
            let y_start = tile_row * 8;
            for tile_x in 0..20 {
                let tile_offset = buf_offset + tile_x * 16;
                for row in 0..8 {
                    let y = y_start + row;
                    if y >= 144 {
                        self.pixel_buffer[tile_offset + row * 2] = 0;
                        self.pixel_buffer[tile_offset + row * 2 + 1] = 0;
                        continue;
                    }
                    let mut bp0 = 0u8;
                    let mut bp1 = 0u8;
                    for px in 0..8 {
                        let x = tile_x * 8 + px;
                        let shade = if x < 160 {
                            self.scanline_data[y * 160 + x] & 3
                        } else {
                            0
                        };
                        let bit = 7 - px;
                        bp0 |= (shade & 1) << bit;
                        bp1 |= ((shade >> 1) & 1) << bit;
                    }
                    self.pixel_buffer[tile_offset + row * 2] = bp0;
                    self.pixel_buffer[tile_offset + row * 2 + 1] = bp1;
                }
            }
            self.rows_ready |= 1 << slot;
        }
    }

    /// Read an ICD2 register (SNES side).
    pub fn read(&mut self, addr: u16) -> u8 {
        match addr {
            0x6000 => {
                // Row status: bits 0-3 = rows ready
                self.rows_ready
            }
            0x6002 => {
                // Packet ready — bit 0 indicates a command packet is available
                if self.packet_ready { 0x01 } else { 0x00 }
            }
            0x6003 => self.control,
            0x7000..=0x700F => {
                // Packet data
                let idx = (addr - 0x7000) as usize;
                let val = self.packet_data[idx];
                // Clear packet_ready after reading last byte
                if idx == 15 {
                    self.packet_ready = false;
                }
                val
            }
            0x7800..=0x7FFF => {
                // Pixel data for selected row
                let offset = (self.read_row_select as usize & 3) * 320;
                let idx = (addr - 0x7800) as usize;
                if idx < 320 {
                    self.pixel_buffer[offset + idx]
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    /// Write an ICD2 register (SNES side).
    pub fn write(&mut self, addr: u16, val: u8) {
        match addr {
            0x6001 => {
                // Row select + advance
                self.read_row_select = val & 3;
                // Mark this row as consumed
                self.rows_ready &= !(1 << (val & 3));
                // If all rows consumed, advance to next batch of 4
                if self.rows_ready == 0 {
                    self.rows_fed += 4;
                    if self.rows_fed < 18 {
                        self.refresh_pixel_buffer();
                    }
                }
            }
            0x6003 => {
                self.control = val;
                // Bit 0: reset GB CPU (not implemented — we drive the GB separately)
            }
            0x6004 => self.joy_data[0] = (self.joy_data[0] & 0xFF00) | val as u16,
            0x6005 => self.joy_data[0] = (self.joy_data[0] & 0x00FF) | ((val as u16) << 8),
            0x6006 => self.joy_data[1] = (self.joy_data[1] & 0xFF00) | val as u16,
            0x6007 => self.joy_data[1] = (self.joy_data[1] & 0x00FF) | ((val as u16) << 8),
            _ => {}
        }
    }
}
