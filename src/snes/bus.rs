/// SNES memory map (LoROM layout for SGB cartridge).
///
/// | Bank     | Address       | Maps To                          |
/// |----------|---------------|----------------------------------|
/// | $00-$3F  | $0000-$1FFF  | WRAM (first 8KB, mirrored)       |
/// | $00-$3F  | $2100-$21FF  | PPU registers                    |
/// | $00-$3F  | $4000-$41FF  | CPU I/O (joypad, old-style)      |
/// | $00-$3F  | $4200-$42FF  | CPU I/O (NMI, multiply, divide)  |
/// | $00-$3F  | $4300-$43FF  | DMA registers                    |
/// | $00-$3F  | $6000-$7FFF  | ICD2 registers (SGB-specific)    |
/// | $00-$7D  | $8000-$FFFF  | ROM                              |
/// | $7E       | $0000-$FFFF | WRAM first 64KB                  |
/// | $7F       | $0000-$FFFF | WRAM second 64KB                 |
/// | $80-$FF  | *            | Mirror of $00-$7F                |

use super::dma::DmaController;
use super::icd2::Icd2;
use super::ppu_regs::SnesPpuRegs;

pub struct SnesBus {
    pub rom: Vec<u8>,
    pub wram: Vec<u8>,          // 128KB
    pub ppu: SnesPpuRegs,
    pub dma: DmaController,
    pub icd2: Icd2,

    // CPU I/O registers
    pub nmitimen: u8,           // $4200: NMI/IRQ enable
    pub rdnmi: u8,              // $4210: NMI flag (bit 7) + version
    pub timeup: u8,             // $4211: IRQ flag
    pub hvbjoy: u8,             // $4212: H/V blank + joypad busy
    pub joy1: u16,              // $4218-19: Joypad 1
    pub joy2: u16,              // $421A-1B: Joypad 2

    // Multiply/divide hardware
    pub wrmpya: u8,             // $4202
    pub wrmpyb: u8,             // $4203
    pub wrdiv: u16,             // $4204-05
    pub wrdivb: u8,             // $4206
    pub rddiv: u16,             // $4214-15: quotient
    pub rdmpy: u16,             // $4216-17: product/remainder

    // Memory-mapped I/O latches
    pub wrio: u8,               // $4201
    pub htime: u16,             // $4207-08
    pub vtime: u16,             // $4209-0A
    pub mdmaen: u8,             // $420B: DMA enable
    pub hdmaen: u8,             // $420C: HDMA enable
    pub memsel: u8,             // $420D: ROM speed

    // APU I/O ports ($2140-$2143) — SNES→SPC writes, SPC→SNES reads
    pub apu_out: [u8; 4],       // What the SNES writes (CPU→APU)
    pub apu_in: [u8; 4],        // What the SNES reads (APU→CPU)
    /// SPC700 upload protocol state:
    /// 0 = waiting for handshake ($AA/$BB ready)
    /// 1 = uploading (echoing counter on port 0)
    /// 2 = executed (SPC program "running")
    pub apu_state: u8,
    /// Last counter value written to port 0 during upload
    apu_last_counter: u8,
    /// Track port 1 value to detect execute command (port1=0 → execute)
    apu_port1_val: u8,
    /// After execute, echo the final counter once, then switch to $AA/$BB
    pub apu_echo_pending: bool,
    /// NMI fire count (diagnostic)
    pub nmi_fire_count: u64,

    /// WMADD — 17-bit WRAM address for $2180 access
    pub wmadd: u32,

    /// Current frame phase cycle tracking for ICD2 tile row simulation
    pub active_display_start: u64,
    pub current_cpu_cycles: u64,
    pub in_vblank: bool,

    /// Set when TIMEUP ($4211) is read, signaling IRQ was acknowledged.
    pub irq_ack: bool,
}

impl SnesBus {
    pub fn new(rom: Vec<u8>) -> Self {
        SnesBus {
            rom,
            wram: vec![0u8; 128 * 1024],
            ppu: SnesPpuRegs::new(),
            dma: DmaController::new(),
            icd2: Icd2::new(),
            nmitimen: 0,
            rdnmi: 0x02,  // Version 2, NMI not pending
            timeup: 0,
            hvbjoy: 0,
            joy1: 0,
            joy2: 0,
            wrmpya: 0xFF,
            wrmpyb: 0,
            wrdiv: 0xFFFF,
            wrdivb: 0,
            rddiv: 0,
            rdmpy: 0,
            wrio: 0xFF,
            htime: 0x1FF,
            vtime: 0x1FF,
            mdmaen: 0,
            hdmaen: 0,
            memsel: 0,
            apu_out: [0; 4],
            apu_in: [0xAA, 0xBB, 0x00, 0x00], // SPC700 IPL ready handshake
            apu_state: 0, // Waiting for handshake
            apu_last_counter: 0,
            apu_port1_val: 0,
            apu_echo_pending: false,
            nmi_fire_count: 0,
            wmadd: 0,
            active_display_start: 0,
            current_cpu_cycles: 0,
            in_vblank: true,
            irq_ack: false,
        }
    }

    /// Compute the current scanline (VCOUNT) from CPU cycle position.
    /// Frame layout: VBlank starts at scanline 225, active display at scanline 0.
    /// Our frame starts at VBlank entry, so:
    ///   cycles 0..(37*1364) → scanlines 225-261
    ///   cycles (37*1364)..(262*1364) → scanlines 0-224
    fn current_vcount(&self) -> u16 {
        const CYCLES_PER_LINE: u64 = 1364;
        if self.in_vblank {
            // VBlank phase: scanlines 225-261
            let elapsed = self.current_cpu_cycles.saturating_sub(
                self.active_display_start.saturating_sub(37 * CYCLES_PER_LINE)
            );
            let line = elapsed / CYCLES_PER_LINE;
            (225 + line).min(261) as u16
        } else {
            // Active display: scanlines 0-224
            let elapsed = self.current_cpu_cycles.saturating_sub(self.active_display_start);
            let line = elapsed / CYCLES_PER_LINE;
            line.min(224) as u16
        }
    }

    /// Read a byte from the SNES address space (24-bit address).
    pub fn read(&mut self, addr: u32) -> u8 {
        let bank = ((addr >> 16) & 0xFF) as u8;
        let offset = (addr & 0xFFFF) as u16;
        let effective_bank = bank & 0x7F; // Mirror $80-$FF → $00-$7F

        // Banks $7E-$7F: WRAM
        if bank == 0x7E || bank == 0x7F {
            let wram_addr = ((bank as usize & 1) << 16) | offset as usize;
            return self.wram[wram_addr];
        }

        match offset {
            0x0000..=0x1FFF if effective_bank <= 0x3F => {
                // WRAM mirror (first 8KB)
                self.wram[offset as usize]
            }
            0x2140..=0x2143 if effective_bank <= 0x3F => {
                // APU I/O ports — read from SPC700 side
                let port = (offset - 0x2140) as usize;
                let val = self.apu_in[port];
                // After execute command: first read returns the echo,
                // then switch to $AA/$BB (SPC program "ready" signal)
                if port == 0 && self.apu_echo_pending {
                    self.apu_echo_pending = false;
                    // After this read (the echo), set $AA/$BB for next read
                    self.apu_in = [0xAA, 0xBB, 0x00, 0x00];
                }
                val
            }
            0x2180 if effective_bank <= 0x3F => {
                // WMDATA read — read WRAM at WMADD, then increment
                let wram_addr = self.wmadd as usize;
                let val = if wram_addr < self.wram.len() { self.wram[wram_addr] } else { 0 };
                self.wmadd = (self.wmadd + 1) & 0x01FFFF;
                val
            }
            0x2100..=0x21FF if effective_bank <= 0x3F => {
                match offset {
                    0x213D => {
                        // OPVCT — return current scanline based on cycle position
                        self.current_vcount() as u8
                    }
                    0x213F => {
                        // STAT78 — bit 7 = field (interlace), bits 3-0 = version
                        // Also resets the OPHCT/OPVCT flipflop
                        0x01
                    }
                    _ => self.ppu.read(offset)
                }
            }
            0x4016 if effective_bank <= 0x3F => {
                // Old-style joypad read (not used by SGB BIOS normally)
                0
            }
            0x4017 if effective_bank <= 0x3F => 0,
            0x4200..=0x42FF if effective_bank <= 0x3F => {
                self.read_cpu_io(offset)
            }
            0x4300..=0x43FF if effective_bank <= 0x3F => {
                self.dma.read(offset - 0x4300)
            }
            0x6000..=0x7FFF if effective_bank <= 0x3F => {
                self.icd2.read(offset)
            }
            0x8000..=0xFFFF if effective_bank <= 0x7D => {
                self.read_rom(effective_bank, offset)
            }
            _ => {
                // Unmapped — open bus
                0
            }
        }
    }

    /// Write a byte to the SNES address space.
    pub fn write(&mut self, addr: u32, val: u8) {
        let bank = ((addr >> 16) & 0xFF) as u8;
        let offset = (addr & 0xFFFF) as u16;
        let effective_bank = bank & 0x7F;

        if bank == 0x7E || bank == 0x7F {
            let wram_addr = ((bank as usize & 1) << 16) | offset as usize;
            self.wram[wram_addr] = val;
            return;
        }

        match offset {
            0x0000..=0x1FFF if effective_bank <= 0x3F => {
                self.wram[offset as usize] = val;
            }
            0x2140..=0x2143 if effective_bank <= 0x3F => {
                // APU I/O ports — SNES→SPC700 write
                // Simulates the SPC700 IPL upload protocol:
                //   State 0 (WaitHandshake): ports return $AA/$BB, wait for $CC
                //   State 1 (Uploading): echo counter on port 0, track port 1
                //   State 2 (Executed): SPC program "running", ports return last values
                let port = (offset - 0x2140) as usize;
                self.apu_out[port] = val;

                match self.apu_state {
                    0 => {
                        // Waiting for handshake — BIOS writes $CC to port 0
                        if port == 0 && val == 0xCC {
                            log::debug!("APU: handshake $CC received, entering upload state");
                            self.apu_state = 1;
                            self.apu_in[0] = 0xCC; // Echo $CC
                            // Set to $FF so first data counter ($00) = $FF+1 is expected
                            self.apu_last_counter = 0xFF;
                            self.apu_port1_val = 1; // Non-zero default (not execute)
                        }
                    }
                    1 => {
                        // Uploading — echo port 0 writes (counter), track port 1
                        if port == 1 {
                            self.apu_port1_val = val;
                        }
                        if port == 0 {
                            // Check if this is an execute or new-block command:
                            // After a block, counter jumps by +2 (not +1).
                            // If port 1 was 0 → execute. If port 1 != 0 → new block.
                            let is_transition = val != self.apu_last_counter.wrapping_add(1);
                            if is_transition && self.apu_port1_val == 0 {
                                // Execute command — SPC program starts running
                                log::debug!("APU: execute command (counter=${:02X}), SPC program running", val);
                                self.apu_in[0] = val; // Echo final counter first
                                self.apu_state = 2;
                                // After echo is read, the SPC program "initializes"
                                // and signals ready with $AA/$BB. We set a flag to
                                // switch to $AA after the echo read.
                                self.apu_echo_pending = true;
                            } else if is_transition {
                                // New block start — echo counter, reset port1 tracking
                                log::debug!("APU: new block (counter=${:02X})", val);
                                self.apu_in[0] = val;
                                self.apu_last_counter = val;
                                self.apu_port1_val = 1; // Reset for next transition
                            } else {
                                // Normal data byte — echo counter
                                self.apu_in[0] = val;
                                self.apu_last_counter = val;
                            }
                        }
                    }
                    2 => {
                        // SPC program "running" — the SPC program always presents
                        // $AA on port 0 when ready. Only $CC starts a new upload.
                        if port == 0 {
                            if val == 0xCC {
                                log::debug!("APU: new upload to SPC program ($CC)");
                                self.apu_state = 1;
                                self.apu_in[0] = 0xCC;
                                self.apu_last_counter = 0xFF;
                                self.apu_port1_val = 1;
                            }
                            // Don't echo other port 0 writes — keep $AA
                        }
                    }
                    _ => {}
                }
            }
            0x2100..=0x21FF if effective_bank <= 0x3F => {
                self.ppu.write(offset, val);
            }
            0x2180 if effective_bank <= 0x3F => {
                // WMDATA — write to WRAM at WMADD, then increment
                let wram_addr = self.wmadd as usize;
                if wram_addr < self.wram.len() {
                    self.wram[wram_addr] = val;
                }
                self.wmadd = (self.wmadd + 1) & 0x01FFFF; // 17-bit wrap
            }
            0x2181 if effective_bank <= 0x3F => {
                self.wmadd = (self.wmadd & 0x01FF00) | val as u32;
            }
            0x2182 if effective_bank <= 0x3F => {
                self.wmadd = (self.wmadd & 0x0100FF) | ((val as u32) << 8);
            }
            0x2183 if effective_bank <= 0x3F => {
                self.wmadd = (self.wmadd & 0x00FFFF) | (((val as u32) & 1) << 16);
            }
            0x4200..=0x42FF if effective_bank <= 0x3F => {
                self.write_cpu_io(offset, val);
            }
            0x4300..=0x43FF if effective_bank <= 0x3F => {
                self.dma.write(offset - 0x4300, val);
            }
            0x6000..=0x7FFF if effective_bank <= 0x3F => {
                self.icd2.write(offset, val);
            }
            _ => {
                // ROM writes / unmapped — ignore
            }
        }
    }

    fn read_rom(&self, bank: u8, offset: u16) -> u8 {
        // LoROM: each bank maps $8000-$FFFF to ROM
        let rom_addr = (bank as usize) * 0x8000 + (offset as usize - 0x8000);
        self.rom.get(rom_addr).copied().unwrap_or(0)
    }

    fn read_cpu_io(&mut self, addr: u16) -> u8 {
        match addr {
            0x4210 => {
                let v = self.rdnmi;
                self.rdnmi &= 0x7F; // Clear NMI flag on read
                v
            }
            0x4211 => {
                let v = self.timeup;
                self.timeup = 0;
                self.irq_ack = true; // Signal IRQ line should be deasserted
                v
            }
            0x4212 => self.hvbjoy,
            0x4214 => self.rddiv as u8,
            0x4215 => (self.rddiv >> 8) as u8,
            0x4216 => self.rdmpy as u8,
            0x4217 => (self.rdmpy >> 8) as u8,
            0x4218 => self.joy1 as u8,
            0x4219 => (self.joy1 >> 8) as u8,
            0x421A => self.joy2 as u8,
            0x421B => (self.joy2 >> 8) as u8,
            _ => 0,
        }
    }

    fn write_cpu_io(&mut self, addr: u16, val: u8) {
        match addr {
            0x4200 => {
                let old = self.nmitimen;
                self.nmitimen = val;
                if val & 0x80 != 0 && old & 0x80 == 0 {
                    log::debug!("SNES: NMI enabled (NMITIMEN=${:02X}), nmi_fire_count={}", val, self.nmi_fire_count);
                }
            }
            0x4201 => self.wrio = val,
            0x4202 => self.wrmpya = val,
            0x4203 => {
                self.wrmpyb = val;
                // Multiply: result = wrmpya × wrmpyb
                self.rdmpy = self.wrmpya as u16 * val as u16;
                self.rddiv = self.rdmpy; // Shares register
            }
            0x4204 => self.wrdiv = (self.wrdiv & 0xFF00) | val as u16,
            0x4205 => self.wrdiv = (self.wrdiv & 0x00FF) | ((val as u16) << 8),
            0x4206 => {
                self.wrdivb = val;
                // Divide: rddiv = wrdiv / wrdivb, rdmpy = wrdiv % wrdivb
                if val == 0 {
                    self.rddiv = 0xFFFF;
                    self.rdmpy = self.wrdiv;
                } else {
                    self.rddiv = self.wrdiv / val as u16;
                    self.rdmpy = self.wrdiv % val as u16;
                }
            }
            0x4207 => self.htime = (self.htime & 0x100) | val as u16,
            0x4208 => self.htime = (self.htime & 0x0FF) | ((val as u16 & 1) << 8),
            0x4209 => self.vtime = (self.vtime & 0x100) | val as u16,
            0x420A => self.vtime = (self.vtime & 0x0FF) | ((val as u16 & 1) << 8),
            0x420B => {
                // General DMA enable — execute immediately
                self.mdmaen = val;
                if val != 0 {
                    self.execute_dma(val);
                }
            }
            0x420C => self.hdmaen = val,
            0x420D => self.memsel = val,
            _ => {}
        }
    }

    /// Execute general-purpose DMA. We need to handle the read/write through
    /// our own bus, so we use a slightly different approach than passing closures.
    fn execute_dma(&mut self, enable: u8) {
        for ch in 0..8u8 {
            if enable & (1 << ch) == 0 { continue; }
            let c = self.dma.channels[ch as usize];
            let direction = c.dmap & 0x80;
            let mode = c.dmap & 0x07;
            let fixed = c.dmap & 0x08 != 0;
            let decrement = c.dmap & 0x10 != 0;
            let mut a_addr = c.a1t;
            let mut remaining = if c.das == 0 { 0x10000u32 } else { c.das as u32 };
            let b_base = 0x2100u16 | c.bbad as u16;

            let offsets: &[u16] = match mode {
                0 => &[0],
                1 => &[0, 1],
                2 | 6 => &[0, 0],
                3 | 7 => &[0, 0, 1, 1],
                4 => &[0, 1, 2, 3],
                5 => &[0, 1, 0, 1],
                _ => &[0],
            };

            let mut offset_idx = 0usize;
            while remaining > 0 {
                let b_addr = b_base.wrapping_add(offsets[offset_idx % offsets.len()]);
                if direction == 0 {
                    // A→B: read from A-bus, write to B-bus (PPU)
                    let val = self.dma_read_a(a_addr);
                    self.ppu.write(b_addr, val);
                } else {
                    // B→A: read from B-bus, write to A-bus
                    let val = self.ppu.read(b_addr);
                    self.dma_write_a(a_addr, val);
                }

                if !fixed {
                    if decrement {
                        a_addr = (a_addr & 0xFF0000) | ((a_addr as u16).wrapping_sub(1) as u32 & 0xFFFF);
                    } else {
                        a_addr = (a_addr & 0xFF0000) | ((a_addr as u16).wrapping_add(1) as u32 & 0xFFFF);
                    }
                }

                offset_idx += 1;
                remaining -= 1;
            }

            self.dma.channels[ch as usize].a1t = a_addr;
            self.dma.channels[ch as usize].das = 0;
        }
    }

    /// DMA A-bus read (ROM, WRAM, ICD2, etc.)
    fn dma_read_a(&mut self, addr: u32) -> u8 {
        let bank = ((addr >> 16) & 0xFF) as u8;
        let offset = (addr & 0xFFFF) as u16;
        let effective_bank = bank & 0x7F;

        if bank == 0x7E || bank == 0x7F {
            let wram_addr = ((bank as usize & 1) << 16) | offset as usize;
            return self.wram[wram_addr];
        }

        match offset {
            0x0000..=0x1FFF if effective_bank <= 0x3F => self.wram[offset as usize],
            0x6000..=0x7FFF if effective_bank <= 0x3F => {
                // ICD2 read during DMA — the BIOS DMAs pixel data from $7800+
                self.icd2.read(offset)
            }
            0x8000..=0xFFFF if effective_bank <= 0x7D => self.read_rom(effective_bank, offset),
            _ => 0,
        }
    }

    /// DMA A-bus write
    fn dma_write_a(&mut self, addr: u32, val: u8) {
        let bank = ((addr >> 16) & 0xFF) as u8;
        let offset = (addr & 0xFFFF) as u16;

        if bank == 0x7E || bank == 0x7F {
            let wram_addr = ((bank as usize & 1) << 16) | offset as usize;
            self.wram[wram_addr] = val;
            return;
        }

        let effective_bank = bank & 0x7F;
        match offset {
            0x0000..=0x1FFF if effective_bank <= 0x3F => {
                self.wram[offset as usize] = val;
            }
            _ => {}
        }
    }

    /// Set NMI flag (called at VBlank start).
    pub fn set_nmi_flag(&mut self) {
        self.rdnmi = 0x82; // bit7=NMI pending, bits 0-3=version
    }

    /// Check if NMI is enabled and pending.
    pub fn nmi_enabled(&self) -> bool {
        self.nmitimen & 0x80 != 0
    }
}
