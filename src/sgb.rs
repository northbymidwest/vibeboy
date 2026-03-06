/// Super Game Boy command protocol, palettes, attributes, border & VRAM transfers.
///
/// The SGB receives commands through the P1 register (0xFF00) using a serial
/// bit-bang protocol: RESET(0x00) → 128 data bits → STOP(0x30), repeated for
/// multi-packet commands.

/// Default DMG grayscale palette (RGB555).
const DEFAULT_PALETTE: [u16; 4] = [0x7FFF, 0x56B5, 0x294A, 0x0000];

#[derive(Clone, Copy, PartialEq)]
enum PacketState {
    Idle,
    /// Waiting for data bits after RESET pulse
    ReceivingBits,
}

#[derive(Clone)]
pub struct Sgb {
    // ── Packet protocol state ──
    state: PacketState,
    /// Last value written to P1 (bits 4-5)
    last_p1: u8,
    /// Protocol active: disabled during boot ROM, enabled after boot ROM unmaps
    pub protocol_active: bool,
    /// Current packet buffer (up to 7 packets × 16 bytes)
    packet_buf: Vec<u8>,
    /// Bits received in current packet
    bit_count: u16,
    /// Current byte being assembled
    current_byte: u8,
    /// Number of packets expected (from header byte)
    expected_packets: u8,
    /// Packets received so far
    packets_received: u8,

    /// LLE mode: when true, packets are relayed to the SNES CPU instead of
    /// being processed by the HLE command handlers.
    pub lle_mode: bool,
    /// Queue of individual 16-byte packets to relay to the SNES subsystem.
    /// Each packet is relayed as it completes (not accumulated for multi-packet commands).
    pub pending_packets: Vec<[u8; 16]>,

    // ── Palettes ──
    /// 4 active SGB palettes (RGB555), color 0 is shared across all
    pub palettes: [[u16; 4]; 4],

    // ── Attribute map ──
    /// Palette index (0-3) per 8×8 tile (20×18 = 360 tiles)
    pub attr_map: [[u8; 20]; 18],

    // ── System palettes (from PAL_TRN) ──
    /// 512 system palettes × 4 colors each
    sys_palettes: Vec<u16>,

    // ── Attribute files (from ATTR_TRN) ──
    /// Up to 45 attribute files, each 90 bytes (20×18 / 4, packed 2 bits)
    attr_files: Vec<[u8; 90]>,

    // ── Border ──
    /// Border tile data: 256 tiles × 32 bytes (SNES 4bpp planar)
    pub border_tiles: Vec<u8>,
    /// Border tilemap: 32×28 = 896 entries (u16 each)
    pub border_map: Vec<u16>,
    /// Border palettes: 4 palettes × 16 colors (RGB555)
    pub border_palettes: [[u16; 16]; 4],
    /// True when border data changed and needs re-render
    pub border_dirty: bool,

    // ── Screen masking ──
    /// 0=off, 1=freeze, 2=black, 3=color0
    pub mask_mode: u8,
    /// Frozen frame buffer (captured when MASK_EN(1) is sent)
    pub frozen_buffer: Vec<u32>,

    // ── Multiplayer ──
    /// Number of active players (1, 2, or 4)
    pub player_count: u8,
    /// Current player index (0-3), cycles on P1 reads
    pub current_player: u8,
    // ── Deferred VRAM transfer ──
    /// Command ID of a pending VRAM transfer (PAL_TRN, CHR_TRN, PCT_TRN, ATTR_TRN)
    pub pending_transfer: Option<u8>,
    /// Extra data for pending transfer (e.g. CHR_TRN half select)
    pending_transfer_data: u8,
    /// Frames remaining before executing the pending transfer (SGB hardware delays ~2 frames)
    transfer_countdown: u8,
}

impl Sgb {
    pub fn new() -> Self {
        Sgb {
            state: PacketState::Idle,
            last_p1: 0x30,
            protocol_active: false, // activated after boot ROM finishes
            packet_buf: Vec::with_capacity(112),
            bit_count: 0,
            current_byte: 0,
            expected_packets: 0,
            packets_received: 0,

            lle_mode: false,
            pending_packets: Vec::new(),

            palettes: [DEFAULT_PALETTE; 4],

            attr_map: [[0u8; 20]; 18],

            sys_palettes: vec![0u16; 512 * 4],
            attr_files: vec![[0u8; 90]; 45],

            border_tiles: vec![0u8; 256 * 32],
            border_map: vec![0u16; 32 * 28],
            border_palettes: [[0u16; 16]; 4],
            border_dirty: false,

            mask_mode: 0,
            frozen_buffer: vec![0u32; 160 * 144],

            player_count: 1,
            current_player: 0,
            pending_transfer: None,
            pending_transfer_data: 0,
            transfer_countdown: 0,
        }
    }

    /// Called when the game writes to P1 (0xFF00).
    ///
    /// SGB packet protocol:
    ///   RESET: write 0x00 then 0x30 (both low, then release)
    ///   Data:  128 × (0x10 or 0x20, then 0x30) — bit on transition from 0x30
    ///   STOP:  write 0x00 (both low again, signals end of packet)
    ///
    /// Normal joypad polling writes 0x20 or 0x10 directly (without 0x30 between),
    /// so we only recognize RESET when 0x00 is followed by 0x30.
    pub fn write_p1(&mut self, val: u8) {
        let p1 = val & 0x30;
        let old = self.last_p1;
        self.last_p1 = p1;

        if !self.protocol_active {
            return;
        }

        // MLT_REQ player cycling: increment on P15 rising edge (bit5: 0→1)
        // Only cycle when not receiving command data
        if self.player_count > 1 && self.state == PacketState::Idle {
            let old_p15 = old & 0x20;
            let new_p15 = p1 & 0x20;
            if old_p15 == 0 && new_p15 != 0 {
                self.current_player = (self.current_player + 1) % self.player_count;
            }
        }

        match self.state {
            PacketState::Idle => {
                // Look for RESET sequence: 0x00 followed by 0x30
                if old == 0x00 && p1 == 0x30 {
                    self.state = PacketState::ReceivingBits;
                    self.bit_count = 0;
                    self.current_byte = 0;
                    if self.packets_received == 0 {
                        self.packet_buf.clear();
                    }
                }
            }
            PacketState::ReceivingBits => {
                if p1 == 0x00 {
                    // 0x00 during data reception = STOP (end of packet)
                    self.finish_packet();
                    return;
                }

                // Data bit on transition: 0x30 → 0x10/0x20
                if old == 0x30 && (p1 == 0x10 || p1 == 0x20) {
                    // Bit4=0 (0x20) = data "0"; Bit5=0 (0x10) = data "1"
                    let bit = if p1 == 0x10 { 1u8 } else { 0u8 };

                    if self.bit_count < 128 {
                        let bit_in_byte = (self.bit_count % 8) as u8;
                        self.current_byte |= bit << bit_in_byte;

                        self.bit_count += 1;

                        if self.bit_count % 8 == 0 {
                            self.packet_buf.push(self.current_byte);
                            self.current_byte = 0;
                        }
                    } else {
                        // This is the STOP bit (bit 129) — end this packet
                        self.finish_packet();
                    }
                }
                // 0x30 (release) and 0x10→0x20/0x20→0x10 are ignored
            }
        }
    }

    /// Read P1 with multiplayer encoding.
    /// Returns bits 1-0 to OR into the P1 result when both select lines are high (0x30).
    pub fn read_p1_id(&self) -> u8 {
        if self.player_count <= 1 {
            return 0x0F; // no multiplayer: all bits high (no buttons pressed)
        }
        // When game reads P1 with both select lines high, return player ID
        // Player 0: 0x0F, Player 1: 0x0E, Player 2: 0x0D, Player 3: 0x0C
        let id = 0x0F - self.current_player;
        id
    }

    /// Finish the current packet and either execute the command or wait for more packets.
    fn finish_packet(&mut self) {
        // Truncate to 16 bytes per packet (discard STOP bit / extra bits)
        let target = (self.packets_received as usize + 1) * 16;
        self.packet_buf.truncate(target);
        while self.packet_buf.len() < target {
            self.packet_buf.push(0);
        }

        if self.packets_received == 0 {
            let header = self.packet_buf[0];
            self.expected_packets = header & 0x07;
            if self.expected_packets == 0 {
                self.expected_packets = 1;
            }
        }

        self.packets_received += 1;

        // In LLE mode, relay each individual 16-byte packet to the SNES as it
        // completes. The SNES BIOS receives one packet per NMI and handles
        // multi-packet command accumulation internally.
        if self.lle_mode {
            let mut pkt = [0u8; 16];
            let start = (self.packets_received as usize - 1) * 16;
            let end = std::cmp::min(start + 16, self.packet_buf.len());
            pkt[..end - start].copy_from_slice(&self.packet_buf[start..end]);
            self.pending_packets.push(pkt);
        }

        if self.packets_received >= self.expected_packets {
            if !self.lle_mode {
                self.execute_command();
            } else {
                // Also run HLE for palette/mask commands to keep SGB state in sync
                self.execute_command();
            }
            self.packets_received = 0;
            self.expected_packets = 0;
            self.packet_buf.clear();
            self.state = PacketState::Idle;
        } else {
            // More packets expected — go to Idle and wait for next RESET (0x00→0x30)
            self.state = PacketState::Idle;
        }
    }

    /// Execute a completed SGB command.
    fn execute_command(&mut self) {
        if self.packet_buf.is_empty() {
            return;
        }
        let cmd = (self.packet_buf[0] >> 3) & 0x1F;
        log::debug!("SGB command 0x{:02X} ({} packets, {} bytes)", cmd, self.expected_packets, self.packet_buf.len());

        if cmd > 0x17 {
            // Invalid command — likely false positive from joypad polling
            return;
        }

        match cmd {
            0x00 => self.cmd_pal01(),
            0x01 => self.cmd_pal23(),
            0x02 => self.cmd_pal03(),
            0x03 => self.cmd_pal12(),
            0x04 => self.cmd_attr_blk(),
            0x05 => self.cmd_attr_lin(),
            0x06 => self.cmd_attr_div(),
            0x07 => self.cmd_attr_chr(),
            0x0A => self.cmd_pal_set(),
            0x0B => self.cmd_pal_trn(),
            0x11 => self.cmd_mlt_req(),
            0x13 => self.cmd_chr_trn(),
            0x14 => self.cmd_pct_trn(),
            0x15 => self.cmd_attr_trn(),
            0x16 => self.cmd_attr_set(),
            0x17 => self.cmd_mask_en(),
            0x0F => self.cmd_data_snd(),
            0x08 | 0x09 | 0x0C | 0x0D | 0x0E | 0x19 => {
                // OBJ_TRN, ICON_EN, ATRC_EN, TEST_EN, ICON_TRN, SOUND
                // These commands control SNES-side features we don't emulate
            }
            _ => {
                log::debug!("SGB: unhandled command 0x{:02X}", cmd);
            }
        }
    }

    // ── Palette commands ──

    fn read_color(&self, offset: usize) -> u16 {
        if offset + 1 >= self.packet_buf.len() { return 0; }
        self.packet_buf[offset] as u16 | ((self.packet_buf[offset + 1] as u16) << 8)
    }

    /// PAL01: Set palettes 0 and 1
    fn cmd_pal01(&mut self) {
        let color0 = self.read_color(1);
        self.palettes[0][0] = color0;
        self.palettes[0][1] = self.read_color(3);
        self.palettes[0][2] = self.read_color(5);
        self.palettes[0][3] = self.read_color(7);
        self.palettes[1][0] = color0;
        self.palettes[1][1] = self.read_color(9);
        self.palettes[1][2] = self.read_color(11);
        self.palettes[1][3] = self.read_color(13);
        // Color 0 shared across all palettes
        self.palettes[2][0] = color0;
        self.palettes[3][0] = color0;
    }

    /// PAL23: Set palettes 2 and 3
    fn cmd_pal23(&mut self) {
        let color0 = self.read_color(1);
        self.palettes[2][0] = color0;
        self.palettes[2][1] = self.read_color(3);
        self.palettes[2][2] = self.read_color(5);
        self.palettes[2][3] = self.read_color(7);
        self.palettes[3][0] = color0;
        self.palettes[3][1] = self.read_color(9);
        self.palettes[3][2] = self.read_color(11);
        self.palettes[3][3] = self.read_color(13);
        // Color 0 shared across all palettes
        self.palettes[0][0] = color0;
        self.palettes[1][0] = color0;
    }

    /// PAL03: Set palettes 0 and 3
    fn cmd_pal03(&mut self) {
        let color0 = self.read_color(1);
        self.palettes[0][0] = color0;
        self.palettes[0][1] = self.read_color(3);
        self.palettes[0][2] = self.read_color(5);
        self.palettes[0][3] = self.read_color(7);
        self.palettes[3][0] = color0;
        self.palettes[3][1] = self.read_color(9);
        self.palettes[3][2] = self.read_color(11);
        self.palettes[3][3] = self.read_color(13);
        self.palettes[1][0] = color0;
        self.palettes[2][0] = color0;
    }

    /// PAL12: Set palettes 1 and 2
    fn cmd_pal12(&mut self) {
        let color0 = self.read_color(1);
        self.palettes[1][0] = color0;
        self.palettes[1][1] = self.read_color(3);
        self.palettes[1][2] = self.read_color(5);
        self.palettes[1][3] = self.read_color(7);
        self.palettes[2][0] = color0;
        self.palettes[2][1] = self.read_color(9);
        self.palettes[2][2] = self.read_color(11);
        self.palettes[2][3] = self.read_color(13);
        self.palettes[0][0] = color0;
        self.palettes[3][0] = color0;
    }

    // ── Attribute commands ──

    /// ATTR_BLK: Set attributes for rectangular regions
    fn cmd_attr_blk(&mut self) {
        let count = self.packet_buf[1] as usize;
        for i in 0..count {
            let base = 2 + i * 6;
            if base + 5 >= self.packet_buf.len() { break; }

            let ctrl = self.packet_buf[base];
            let pal_data = self.packet_buf[base + 1];
            let x1 = self.packet_buf[base + 2] as usize;
            let y1 = self.packet_buf[base + 3] as usize;
            let x2 = self.packet_buf[base + 4] as usize;
            let y2 = self.packet_buf[base + 5] as usize;

            let pal_inside = pal_data & 0x03;
            let pal_border = (pal_data >> 2) & 0x03;
            let pal_outside = (pal_data >> 4) & 0x03;

            let mut set_inside = ctrl & 0x01 != 0;
            let mut set_border = ctrl & 0x02 != 0;
            let set_outside = ctrl & 0x04 != 0;

            // SameBoy behavior: when only inside is set (no border/outside),
            // automatically extend inside palette to the border.
            // When only outside is set (no border/inside), extend to border.
            let mut pal_border_eff = pal_border;
            if set_inside && !set_border && !set_outside {
                set_border = true;
                pal_border_eff = pal_inside;
            } else if set_outside && !set_border && !set_inside {
                set_border = true;
                pal_border_eff = pal_outside;
            }

            let x1 = x1 & 0x1F;
            let y1 = y1 & 0x1F;
            let x2 = x2 & 0x1F;
            let y2 = y2 & 0x1F;

            for ty in 0..18usize {
                for tx in 0..20usize {
                    if tx < x1 || tx > x2 || ty < y1 || ty > y2 {
                        // Outside the rectangle
                        if set_outside {
                            self.attr_map[ty][tx] = pal_outside;
                        }
                    } else if tx > x1 && tx < x2 && ty > y1 && ty < y2 {
                        // Inside the rectangle (strictly)
                        if set_inside {
                            self.attr_map[ty][tx] = pal_inside;
                        }
                    } else {
                        // Border (on the edges)
                        if set_border {
                            self.attr_map[ty][tx] = pal_border_eff;
                        }
                    }
                }
            }
        }
    }

    /// ATTR_LIN: Set attributes for horizontal/vertical lines
    fn cmd_attr_lin(&mut self) {
        let count = self.packet_buf[1] as usize;
        for i in 0..count {
            let idx = 2 + i;
            if idx >= self.packet_buf.len() { break; }

            let data = self.packet_buf[idx];
            let line = (data & 0x1F) as usize;
            let pal = (data >> 5) & 0x03;
            let horizontal = data & 0x80 != 0;

            if horizontal {
                if line < 18 {
                    for tx in 0..20 {
                        self.attr_map[line][tx] = pal;
                    }
                }
            } else {
                if line < 20 {
                    for ty in 0..18 {
                        self.attr_map[ty][line] = pal;
                    }
                }
            }
        }
    }

    /// ATTR_DIV: Divide screen horizontally or vertically
    fn cmd_attr_div(&mut self) {
        let data = self.packet_buf[1];
        // Bits 0-1: palette below/right of line (high coords)
        // Bits 2-3: palette above/left of line (low coords)
        // Bits 4-5: palette on the dividing line itself
        let pal_below_right = data & 0x03;
        let pal_above_left = (data >> 2) & 0x03;
        let pal_line = (data >> 4) & 0x03;
        let horizontal = data & 0x40 != 0;
        let line = (self.packet_buf[2] & 0x1F) as usize;

        for ty in 0..18usize {
            for tx in 0..20usize {
                let coord = if horizontal { ty } else { tx };
                let pal = if coord < line {
                    pal_above_left
                } else if coord == line {
                    pal_line
                } else {
                    pal_below_right
                };
                self.attr_map[ty][tx] = pal;
            }
        }
    }

    /// ATTR_CHR: Set attributes for tiles sequentially
    fn cmd_attr_chr(&mut self) {
        let start_x = self.packet_buf[1] as usize;
        let start_y = self.packet_buf[2] as usize;
        let count = self.packet_buf[3] as u16 | ((self.packet_buf[4] as u16) << 8);
        let horizontal = self.packet_buf[5] & 0x01 == 0;

        let mut tx = start_x;
        let mut ty = start_y;

        // Data starts at byte 6, packed 2 bits per tile (4 tiles per byte)
        for i in 0..count as usize {
            if ty >= 18 || tx >= 20 { break; }
            let byte_idx = 6 + i / 4;
            if byte_idx >= self.packet_buf.len() { break; }
            let shift = (3 - (i % 4)) * 2;
            let pal = (self.packet_buf[byte_idx] >> shift) & 0x03;
            self.attr_map[ty][tx] = pal;

            if horizontal {
                tx += 1;
                if tx >= 20 { tx = 0; ty += 1; }
            } else {
                ty += 1;
                if ty >= 18 { ty = 0; tx += 1; }
            }
        }
    }

    /// PAL_SET: Select system palettes + optional attribute file
    fn cmd_pal_set(&mut self) {
        for i in 0..4 {
            let idx = self.packet_buf[1 + i * 2] as usize
                | ((self.packet_buf[2 + i * 2] as usize) << 8);
            let pal_idx = idx & 0x1FF;
            let base = pal_idx * 4;
            for c in 0..4 {
                if base + c < self.sys_palettes.len() {
                    self.palettes[i][c] = self.sys_palettes[base + c];
                }
            }
            log::debug!("SGB PAL_SET: pal[{}] = sys[{}] = [{:04X},{:04X},{:04X},{:04X}]",
                i, pal_idx,
                self.palettes[i][0], self.palettes[i][1],
                self.palettes[i][2], self.palettes[i][3]);
        }
        // Byte 9 bit 7: also apply attribute file
        let attr_byte = self.packet_buf[9];
        if attr_byte & 0x80 != 0 {
            let file_idx = (attr_byte & 0x3F) as usize;
            log::debug!("SGB PAL_SET: also applying attr file {}", file_idx);
            self.apply_attr_file(file_idx);
        }
        if attr_byte & 0x40 != 0 {
            self.mask_mode = 0;
        }
        // Ensure color 0 shared
        let c0 = self.palettes[0][0];
        self.palettes[1][0] = c0;
        self.palettes[2][0] = c0;
        self.palettes[3][0] = c0;
    }

    /// PAL_TRN: Transfer system palettes from VRAM
    fn cmd_pal_trn(&mut self) {
        self.pending_transfer = Some(0x0B);
        self.pending_transfer_data = 0;
        self.transfer_countdown = 2;
    }

    /// MLT_REQ: Multiplayer request
    fn cmd_mlt_req(&mut self) {
        let n = self.packet_buf[1] & 0x03;
        self.player_count = match n {
            0 => 1,
            1 => 2,
            3 => 4,
            _ => 1,
        };
        self.current_player = 0;
        log::debug!("SGB MLT_REQ: {} players", self.player_count);
    }

    /// CHR_TRN: Transfer border tiles from VRAM
    fn cmd_chr_trn(&mut self) {
        let half = self.packet_buf[1] & 0x01;
        self.pending_transfer = Some(0x13);
        self.pending_transfer_data = half;
        self.transfer_countdown = 2;
    }

    /// PCT_TRN: Transfer border map + palettes from VRAM
    fn cmd_pct_trn(&mut self) {
        self.pending_transfer = Some(0x14);
        self.pending_transfer_data = 0;
        self.transfer_countdown = 2;
    }

    /// ATTR_SET: Apply an attribute file
    fn cmd_attr_set(&mut self) {
        let file_idx = (self.packet_buf[1] & 0x3F) as usize;
        let cancel_mask = self.packet_buf[1] & 0x40 != 0;
        log::debug!("SGB ATTR_SET: file={} cancel_mask={}", file_idx, cancel_mask);
        self.apply_attr_file(file_idx);
        if cancel_mask {
            self.mask_mode = 0;
        }
    }

    /// ATTR_TRN: Transfer attribute files from VRAM
    fn cmd_attr_trn(&mut self) {
        self.pending_transfer = Some(0x15);
        self.pending_transfer_data = 0;
        self.transfer_countdown = 2;
    }

    /// DATA_SND: Write data to SNES WRAM (used for BIOS hotpatching)
    fn cmd_data_snd(&mut self) {
        let addr_lo = self.packet_buf[1] as u16;
        let addr_hi = self.packet_buf[2] as u16;
        let bank = self.packet_buf[3];
        let snes_addr = (addr_hi << 8) | addr_lo;
        log::debug!("SGB DATA_SND: bank=${:02X} addr=${:04X}", bank, snes_addr);
    }

    /// MASK_EN: Screen masking
    fn cmd_mask_en(&mut self) {
        self.mask_mode = self.packet_buf[1] & 0x03;
        log::debug!("SGB MASK_EN: mode {}", self.mask_mode);
    }

    // ── Attribute file helpers ──

    fn apply_attr_file(&mut self, file_idx: usize) {
        if file_idx >= 45 { return; }
        let file = self.attr_files[file_idx];
        for i in 0..360 {
            let byte_idx = i / 4;
            let bit_shift = (3 - (i % 4)) * 2;
            let pal = (file[byte_idx] >> bit_shift) & 0x03;
            let tx = i % 20;
            let ty = i / 20;
            if ty < 18 && tx < 20 {
                self.attr_map[ty][tx] = pal;
            }
        }
    }

    // ── VRAM transfer execution ──

    /// Execute a pending VRAM transfer using the provided VRAM data.
    /// `vram_data` should be the first 4096 bytes of VRAM bank 0.
    pub fn execute_transfer(&mut self, vram_data: &[u8]) {
        let cmd = match self.pending_transfer.take() {
            Some(c) => c,
            None => return,
        };
        let extra = self.pending_transfer_data;

        match cmd {
            0x0B => {
                // PAL_TRN: 512 palettes × 4 colors × 2 bytes = 4096 bytes
                for i in 0..2048 {
                    if i * 2 + 1 < vram_data.len() {
                        self.sys_palettes[i] = vram_data[i * 2] as u16
                            | ((vram_data[i * 2 + 1] as u16) << 8);
                    }
                }
                log::debug!("SGB PAL_TRN: loaded system palettes");
            }
            0x13 => {
                // CHR_TRN: 128 tiles × 32 bytes = 4096 bytes per half
                let offset = if extra != 0 { 4096 } else { 0 };
                let len = std::cmp::min(4096, vram_data.len());
                if offset + len <= self.border_tiles.len() {
                    self.border_tiles[offset..offset + len].copy_from_slice(&vram_data[..len]);
                }
                self.border_dirty = true;
                log::debug!("SGB CHR_TRN: loaded border tiles (half {})", extra);
            }
            0x14 => {
                // PCT_TRN: tilemap (32×28 × 2 = 1792 bytes) + palettes (4 × 16 × 2 = 128 bytes)
                // Tilemap at offset 0
                for i in 0..896 {
                    if i * 2 + 1 < vram_data.len() {
                        self.border_map[i] = vram_data[i * 2] as u16
                            | ((vram_data[i * 2 + 1] as u16) << 8);
                    }
                }
                // Palettes at offset 0x800 (2048)
                for pal in 0..4 {
                    for col in 0..16 {
                        let off = 0x800 + pal * 32 + col * 2;
                        if off + 1 < vram_data.len() {
                            self.border_palettes[pal][col] = vram_data[off] as u16
                                | ((vram_data[off + 1] as u16) << 8);
                        }
                    }
                }
                self.border_dirty = true;
                log::debug!("SGB PCT_TRN: loaded border map + palettes");
            }
            0x15 => {
                // ATTR_TRN: 45 attribute files × 90 bytes = 4050 bytes
                for f in 0..45 {
                    let base = f * 90;
                    for b in 0..90 {
                        if base + b < vram_data.len() {
                            self.attr_files[f][b] = vram_data[base + b];
                        }
                    }
                }
                log::debug!("SGB ATTR_TRN: loaded attribute files");
            }
            _ => {}
        }
    }

    /// Check if a VRAM transfer is ready to execute (countdown expired).
    pub fn has_pending_transfer(&self) -> bool {
        self.pending_transfer.is_some() && self.transfer_countdown == 0
    }

    /// Tick the transfer countdown (call once per frame).
    pub fn tick_transfer(&mut self) {
        if self.transfer_countdown > 0 {
            self.transfer_countdown -= 1;
        }
    }

    // ── Rendering ──

    /// Convert RGB555 to 0x00RRGGBB.
    pub fn rgb555_to_rgb32(c: u16) -> u32 {
        let r5 = (c & 0x1F) as u32;
        let g5 = ((c >> 5) & 0x1F) as u32;
        let b5 = ((c >> 10) & 0x1F) as u32;
        let r8 = (r5 << 3) | (r5 >> 2);
        let g8 = (g5 << 3) | (g5 >> 2);
        let b8 = (b5 << 3) | (b5 >> 2);
        (r8 << 16) | (g8 << 8) | b8
    }

    /// Apply SGB palettes to a shade buffer, producing a colorized frame.
    /// `shade_buf`: 160×144 array of 2-bit shade indices.
    /// `frame_out`: 160×144 output in 0x00RRGGBB format.
    pub fn apply_palettes(&self, shade_buf: &[u8], frame_out: &mut [u32]) {
        for ty in 0..18usize {
            for tx in 0..20usize {
                let pal_idx = self.attr_map[ty][tx] as usize;
                let pal = &self.palettes[pal_idx.min(3)];
                for py in 0..8usize {
                    let y = ty * 8 + py;
                    if y >= 144 { break; }
                    for px in 0..8usize {
                        let x = tx * 8 + px;
                        if x >= 160 { break; }
                        let idx = y * 160 + x;
                        let shade = shade_buf[idx] as usize;
                        frame_out[idx] = Self::rgb555_to_rgb32(pal[shade & 3]);
                    }
                }
            }
        }
    }

    /// Render the 256×224 border into `border_buf`.
    /// Uses SNES 4bpp planar tile format.
    pub fn render_border(&self, border_buf: &mut [u32]) {
        for map_y in 0..28usize {
            for map_x in 0..32usize {
                let entry = self.border_map[map_y * 32 + map_x];
                let tile_idx = (entry & 0xFF) as usize;
                let pal_idx = ((entry >> 10) & 0x07) as usize;
                let x_flip = entry & 0x4000 != 0;
                let y_flip = entry & 0x8000 != 0;
                // Priority bit (0x2000) ignored for border

                // The SGB BIOS uses SNES palettes 4-7 for border tiles.
                // border_palettes[0-3] maps to SNES palettes 4-7.
                // Remap tilemap palette index: 4→0, 5→1, 6→2, 7→3, 0-3→as-is.
                let local_pal = if pal_idx >= 4 { pal_idx - 4 } else { pal_idx };
                let pal = &self.border_palettes[local_pal.min(3)];

                for ty in 0..8usize {
                    let row = if y_flip { 7 - ty } else { ty };
                    let tile_base = tile_idx * 32;
                    // SNES 4bpp planar: bytes 0-15 = low 2 planes, bytes 16-31 = high 2 planes
                    let bp0 = self.border_tiles.get(tile_base + row * 2).copied().unwrap_or(0);
                    let bp1 = self.border_tiles.get(tile_base + row * 2 + 1).copied().unwrap_or(0);
                    let bp2 = self.border_tiles.get(tile_base + 16 + row * 2).copied().unwrap_or(0);
                    let bp3 = self.border_tiles.get(tile_base + 16 + row * 2 + 1).copied().unwrap_or(0);

                    for tx_px in 0..8usize {
                        let bit = if x_flip { tx_px } else { 7 - tx_px };
                        let color_idx = ((bp0 >> bit) & 1)
                            | (((bp1 >> bit) & 1) << 1)
                            | (((bp2 >> bit) & 1) << 2)
                            | (((bp3 >> bit) & 1) << 3);

                        let screen_x = map_x * 8 + tx_px;
                        let screen_y = map_y * 8 + ty;
                        if screen_x < 256 && screen_y < 224 {
                            let pixel = if color_idx == 0 {
                                // Color 0 = transparent (show game area or black)
                                0x00000000
                            } else {
                                Self::rgb555_to_rgb32(pal[color_idx as usize])
                            };
                            border_buf[screen_y * 256 + screen_x] = pixel;
                        }
                    }
                }
            }
        }
    }

    /// Composite game frame into the 256×224 border buffer at offset (48, 40).
    pub fn composite_frame(border_buf: &mut [u32], game_buf: &[u32]) {
        for y in 0..144usize {
            for x in 0..160usize {
                let bx = x + 48;
                let by = y + 40;
                if bx < 256 && by < 224 {
                    border_buf[by * 256 + bx] = game_buf[y * 160 + x];
                }
            }
        }
    }
}
