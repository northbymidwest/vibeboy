/// ICD2 bridge: connects the Game Boy side to the SNES side.
///
/// The ICD2 chip in the real SGB cartridge captures the GB's LCD output as
/// 2bpp tile rows, buffers command packets from the GB, and relays joypad
/// data from the SNES controllers to the GB.
///
/// Real hardware:
///   $6000 (R)   — bits 7-3: current GB tile row ($11 during VBlank)
///                  bits 1-0: writeBank (slot currently being filled)
///   $6001 (W)   — set readBank (bits 1-0), reset readAddress to 0
///   $6002 (R)   — bit 0: command packet available
///   $6003 (W)   — control (reset/players/speed)
///   $6004-$6007 (W) — joypad data for players 1-4
///   $7000-$700F (R) — last command packet (16 bytes)
///   $7800       (R) — pixel data stream (auto-incrementing readAddress)
///
/// Ring buffer: 4 slots × 512 bytes each. writeBank advances every 8 GB
/// scanlines (one tile row). The SNES reads the 3 completed slots, avoiding
/// the one being written. Over 6 NMIs, all 18 tile rows are transferred.

#[derive(Clone)]
pub struct Icd2 {
    /// Ring buffer: 4 slots × 512 bytes (20 tiles × 16 bytes/tile = 320 used per slot)
    output: Vec<u8>,
    /// Which slot is "being written" (SNES avoids reading this one)
    write_bank: u8,
    /// Which slot the SNES is reading from ($6001)
    read_bank: u8,
    /// Auto-incrementing read pointer within current slot (0-511)
    read_address: u16,

    /// All 18 tile rows pre-computed from shade buffer (18 × 320 bytes)
    all_rows: Vec<u8>,
    /// Next tile row to serve when the BIOS reads a slot
    next_row: usize,
    /// How many slots have been read since last refill
    slots_read: u8,

    /// Last command packet from GB (16 bytes)
    packet_data: [u8; 16],
    /// True when a new packet is available
    pub packet_ready: bool,

    /// Joypad data (SNES→GB), 4 players
    pub joy_data: [u16; 4],

    /// Control register ($6003)
    pub control: u8,

    /// Simulated current GB tile row (0-17 active, 0x11=VBlank)
    /// Updated by the bus based on SNES cycle timing
    pub current_tile_row: u8,
}

impl Icd2 {
    pub fn new() -> Self {
        Icd2 {
            output: vec![0u8; 4 * 512],
            write_bank: 0,
            read_bank: 0,
            read_address: 0,
            all_rows: vec![0u8; 18 * 320],
            next_row: 0,
            slots_read: 0,
            packet_data: [0; 16],
            packet_ready: false,
            joy_data: [0; 4],
            control: 0,
            current_tile_row: 0x11, // Start in VBlank
        }
    }

    /// Feed a 16-byte command packet from the GB side.
    pub fn feed_packet(&mut self, data: &[u8; 16]) {
        self.packet_data = *data;
        self.packet_ready = true;
    }

    /// Feed a full frame of scanline data (160×144 2-bit shades).
    /// Converts shade buffer into 18 tile rows of 2bpp planar data,
    /// pre-loaded into the ring buffer for progressive reading.
    pub fn feed_scanlines(&mut self, shade_buf: &[u8]) {
        // Convert all 18 tile rows to 2bpp planar format
        for tile_row in 0..18 {
            let y_start = tile_row * 8;
            let row_base = tile_row * 320;
            for tile_x in 0..20 {
                let tile_offset = row_base + tile_x * 16;
                for row in 0..8 {
                    let y = y_start + row;
                    let mut bp0 = 0u8;
                    let mut bp1 = 0u8;
                    if y < 144 {
                        for px in 0..8 {
                            let x = tile_x * 8 + px;
                            let shade = if x < 160 && y * 160 + x < shade_buf.len() {
                                shade_buf[y * 160 + x] & 3
                            } else {
                                0
                            };
                            let bit = 7 - px;
                            bp0 |= (shade & 1) << bit;
                            bp1 |= ((shade >> 1) & 1) << bit;
                        }
                    }
                    self.all_rows[tile_offset + row * 2] = bp0;
                    self.all_rows[tile_offset + row * 2 + 1] = bp1;
                }
            }
        }

        // Reset ring buffer state — start at tile row 0
        self.next_row = 0;
        self.write_bank = 0;
        self.slots_read = 0;
        self.current_tile_row = 0x11; // VBlank initially; updated per frame phase
        // Pre-fill all 4 slots with tile rows 0-3
        for slot in 0..4 {
            self.fill_slot(slot);
        }
    }

    /// Set the current tile row visible in $6000.
    /// Called by the SNES frame runner to simulate GB LCD progress.
    pub fn set_tile_row(&mut self, row: u8) {
        let old = self.current_tile_row;
        self.current_tile_row = row;

        // When tile row advances, update the ring buffer:
        // fill the write_bank slot and advance write_bank
        if row < 18 && row != old && row > 0 {
            // The just-completed row fills the current write_bank slot
            let completed_row = (row as usize).saturating_sub(1);
            if completed_row < 18 {
                let slot = self.write_bank as usize;
                self.fill_slot_with(slot, completed_row);
                self.write_bank = (self.write_bank + 1) & 3;
            }
        }
    }

    /// Fill a ring buffer slot with a specific tile row.
    fn fill_slot_with(&mut self, slot: usize, tile_row: usize) {
        if tile_row >= 18 { return; }
        let src_base = tile_row * 320;
        let dst_base = slot * 512;
        self.output[dst_base..dst_base + 512].fill(0);
        self.output[dst_base..dst_base + 320]
            .copy_from_slice(&self.all_rows[src_base..src_base + 320]);
    }

    /// Fill a ring buffer slot with the next available tile row.
    fn fill_slot(&mut self, slot: usize) {
        if self.next_row >= 18 { return; }
        let src_base = self.next_row * 320;
        let dst_base = slot * 512;
        // Clear the slot first (512 bytes, 320 used + 192 padding)
        self.output[dst_base..dst_base + 512].fill(0);
        // Copy 320 bytes of tile data
        self.output[dst_base..dst_base + 320]
            .copy_from_slice(&self.all_rows[src_base..src_base + 320]);
        self.next_row += 1;
    }

    /// Read an ICD2 register (SNES side).
    pub fn read(&mut self, addr: u16) -> u8 {
        match addr {
            0x6000 => {
                // Bits 7-3: current GB tile row (0-17 during active, $11 during VBlank)
                // Bits 1-0: writeBank (slot currently being filled)
                let tile_row = self.current_tile_row;
                let wb = (tile_row as u8) & 3; // writeBank tracks which slot is being filled
                (tile_row << 3) | wb
            }
            0x6002 => {
                let val = if self.packet_ready { 0x01 } else { 0x00 };
                log::trace!("ICD2 $6002 read: packet_ready={}", self.packet_ready);
                val
            }
            0x6003 => self.control,
            0x7000..=0x700F => {
                let idx = (addr - 0x7000) as usize;
                let val = self.packet_data[idx];
                if idx == 15 {
                    self.packet_ready = false;
                }
                val
            }
            0x7800 => {
                // Pixel data stream — auto-incrementing within selected slot
                let offset = (self.read_bank as usize & 3) * 512 + self.read_address as usize;
                let val = if offset < self.output.len() {
                    self.output[offset]
                } else {
                    0
                };
                self.read_address = (self.read_address + 1) & 0x1FF; // wrap at 512
                val
            }
            // All reads in $7800-$7FFF map to the same auto-incrementing port
            0x7801..=0x7FFF => {
                let offset = (self.read_bank as usize & 3) * 512 + self.read_address as usize;
                let val = if offset < self.output.len() {
                    self.output[offset]
                } else {
                    0
                };
                self.read_address = (self.read_address + 1) & 0x1FF;
                val
            }
            _ => 0,
        }
    }

    /// Write an ICD2 register (SNES side).
    pub fn write(&mut self, addr: u16, val: u8) {
        match addr {
            0x6001 => {
                // Select readBank and reset readAddress (per bsnes)
                self.read_bank = val & 3;
                self.read_address = 0;
            }
            0x6003 => {
                self.control = val;
            }
            0x6004 => self.joy_data[0] = (self.joy_data[0] & 0xFF00) | val as u16,
            0x6005 => self.joy_data[0] = (self.joy_data[0] & 0x00FF) | ((val as u16) << 8),
            0x6006 => self.joy_data[1] = (self.joy_data[1] & 0xFF00) | val as u16,
            0x6007 => self.joy_data[1] = (self.joy_data[1] & 0x00FF) | ((val as u16) << 8),
            _ => {}
        }
    }
}
