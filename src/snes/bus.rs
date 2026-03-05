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
            0x2100..=0x21FF if effective_bank <= 0x3F => {
                self.ppu.read(offset)
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
            0x2100..=0x21FF if effective_bank <= 0x3F => {
                self.ppu.write(offset, val);
            }
            0x2180 if effective_bank <= 0x3F => {
                // WMDATA — write to WRAM at WMADD
                // We skip tracking WMADD for now (SGB BIOS uses DMA for WRAM writes)
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
            0x4200 => self.nmitimen = val,
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

    /// DMA A-bus read (ROM, WRAM, etc.)
    fn dma_read_a(&self, addr: u32) -> u8 {
        let bank = ((addr >> 16) & 0xFF) as u8;
        let offset = (addr & 0xFFFF) as u16;
        let effective_bank = bank & 0x7F;

        if bank == 0x7E || bank == 0x7F {
            let wram_addr = ((bank as usize & 1) << 16) | offset as usize;
            return self.wram[wram_addr];
        }

        match offset {
            0x0000..=0x1FFF if effective_bank <= 0x3F => self.wram[offset as usize],
            0x6000..=0x7FFF if effective_bank <= 0x3F => 0, // ICD2 read during DMA — not typical
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
