/// SNES PPU register file — no rendering, just captures writes and stores VRAM/CGRAM/OAM.
/// This is sufficient for SGB emulation where we extract palette/tile data from SNES memory.

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SnesPpuRegs {
    pub vram: Vec<u8>,        // 64KB (word-addressed, stored as bytes)
    #[serde(with = "serde_big_array::BigArray")]
    pub cgram: [u8; 512],     // 256 colors × 2 bytes (RGB555)
    #[serde(with = "serde_big_array::BigArray")]
    pub oam: [u8; 544],       // 512 + 32 bytes

    // VRAM access
    pub vmain: u8,            // $2115: increment mode
    pub vmadd: u16,           // $2116-17: VRAM word address
    vram_prefetch: u16,       // Prefetch latch

    // CGRAM access
    pub cgadd: u16,           // $2121: CGRAM byte address (9 bits, auto-increments)
    cg_latch: u8,             // Low byte latch for $2122
    cg_flipflop: bool,        // false=low byte, true=high byte

    // OAM access
    pub oamadd: u16,          // $2102-03: OAM word address
    oam_latch: u8,            // Low byte latch for $2104
    oam_addr_internal: u16,   // Internal byte address

    // Display registers (stored but not rendered)
    pub inidisp: u8,          // $2100
    pub bgmode: u8,           // $2105
    pub mosaic: u8,           // $2106
    pub bg_sc: [u8; 4],       // $2107-210A: BG1-4 tilemap base
    pub bg_chr: [u8; 2],      // $210B-210C: BG1-4 chr base
    pub setini: u8,           // $2133
}

impl SnesPpuRegs {
    pub fn new() -> Self {
        SnesPpuRegs {
            vram: vec![0u8; 0x10000],
            cgram: [0u8; 512],
            oam: [0u8; 544],
            vmain: 0,
            vmadd: 0,
            vram_prefetch: 0,
            cgadd: 0,
            cg_latch: 0,
            cg_flipflop: false,
            oamadd: 0,
            oam_latch: 0,
            oam_addr_internal: 0,
            inidisp: 0x80, // force blank on reset
            bgmode: 0,
            mosaic: 0,
            bg_sc: [0; 4],
            bg_chr: [0; 2],
            setini: 0,
        }
    }

    /// Translate VRAM word address to byte offset using VMAIN increment mapping.
    fn vram_byte_addr(&self, word_addr: u16) -> usize {
        let mapping = (self.vmain >> 2) & 3;
        let addr = match mapping {
            0 => word_addr,
            1 => {
                // 8-bit rotation: aaaaaaaaBBBccccc -> aaaaaaaacccccBBB
                let a = word_addr & 0xFF00;
                let b = (word_addr >> 5) & 0x07;
                let c = word_addr & 0x1F;
                a | (c << 3) | b
            }
            2 => {
                // 9-bit rotation: aaaaaaaBBBcccccc -> aaaaaaaccccccBBB
                let a = word_addr & 0xFE00;
                let b = (word_addr >> 6) & 0x07;
                let c = word_addr & 0x3F;
                a | (c << 3) | b
            }
            3 => {
                // 10-bit rotation: aaaaaaBBBccccccc -> aaaaaacccccccBBB
                let a = word_addr & 0xFC00;
                let b = (word_addr >> 7) & 0x07;
                let c = word_addr & 0x7F;
                a | (c << 3) | b
            }
            _ => unreachable!(),
        };
        ((addr as usize) * 2) & 0xFFFF
    }

    fn vram_increment(&self) -> u16 {
        match self.vmain & 3 {
            0 => 1,
            1 => 32,
            2 | 3 => 128,
            _ => unreachable!(),
        }
    }

    pub fn write(&mut self, addr: u16, val: u8) {
        match addr {
            0x2100 => self.inidisp = val,
            0x2101 => {} // OBJ size/base — ignored
            0x2102 => {
                self.oamadd = (self.oamadd & 0x100) | val as u16;
                self.oam_addr_internal = self.oamadd * 2;
            }
            0x2103 => {
                self.oamadd = (self.oamadd & 0xFF) | ((val as u16 & 1) << 8);
                self.oam_addr_internal = self.oamadd * 2;
            }
            0x2104 => { // OAMDATA
                let a = self.oam_addr_internal as usize;
                if a < 512 {
                    if a & 1 == 0 {
                        self.oam_latch = val;
                    } else {
                        self.oam[a - 1] = self.oam_latch;
                        self.oam[a] = val;
                    }
                } else if a < 544 {
                    self.oam[a] = val;
                }
                self.oam_addr_internal = self.oam_addr_internal.wrapping_add(1);
                if self.oam_addr_internal >= 544 {
                    self.oam_addr_internal = 0;
                }
            }
            0x2105 => self.bgmode = val,
            0x2106 => self.mosaic = val,
            0x2107 => self.bg_sc[0] = val,
            0x2108 => self.bg_sc[1] = val,
            0x2109 => self.bg_sc[2] = val,
            0x210A => self.bg_sc[3] = val,
            0x210B => self.bg_chr[0] = val,
            0x210C => self.bg_chr[1] = val,
            0x210D..=0x2114 => {} // BG scroll registers — ignored for SGB
            0x2115 => self.vmain = val,
            0x2116 => {
                self.vmadd = (self.vmadd & 0xFF00) | val as u16;
                // Prefetch on address write
                let byte_addr = self.vram_byte_addr(self.vmadd);
                self.vram_prefetch = self.vram[byte_addr] as u16
                    | ((self.vram.get(byte_addr + 1).copied().unwrap_or(0) as u16) << 8);
            }
            0x2117 => {
                self.vmadd = (self.vmadd & 0x00FF) | ((val as u16) << 8);
                let byte_addr = self.vram_byte_addr(self.vmadd);
                self.vram_prefetch = self.vram[byte_addr] as u16
                    | ((self.vram.get(byte_addr + 1).copied().unwrap_or(0) as u16) << 8);
            }
            0x2118 => { // VMDATAL — write low byte
                let byte_addr = self.vram_byte_addr(self.vmadd);
                if byte_addr < self.vram.len() {
                    self.vram[byte_addr] = val;
                }
                if self.vmain & 0x80 == 0 {
                    self.vmadd = self.vmadd.wrapping_add(self.vram_increment());
                }
            }
            0x2119 => { // VMDATAH — write high byte
                let byte_addr = self.vram_byte_addr(self.vmadd);
                if byte_addr + 1 < self.vram.len() {
                    self.vram[byte_addr + 1] = val;
                }
                if self.vmain & 0x80 != 0 {
                    self.vmadd = self.vmadd.wrapping_add(self.vram_increment());
                }
            }
            0x2121 => { // CGADD
                self.cgadd = (val as u16) * 2;
                self.cg_flipflop = false;
            }
            0x2122 => { // CGDATA
                if !self.cg_flipflop {
                    self.cg_latch = val;
                    self.cg_flipflop = true;
                } else {
                    let idx = (self.cgadd as usize) & 0x1FE;
                    if idx + 1 < 512 {
                        self.cgram[idx] = self.cg_latch;
                        self.cgram[idx + 1] = val & 0x7F; // Mask bit 15
                    }
                    self.cgadd = self.cgadd.wrapping_add(2) & 0x1FF;
                    self.cg_flipflop = false;
                }
            }
            0x2123..=0x2133 => {} // Window, color math, etc — ignored for SGB
            _ => {}
        }
    }

    pub fn read(&self, addr: u16) -> u8 {
        match addr {
            0x2134..=0x2136 => 0, // Multiplication result (stubbed)
            0x2137 => 0, // SLHV latch
            0x2138 => { // OAMDATAREAD — stub
                0
            }
            0x2139 => { // VMDATALREAD
                self.vram_prefetch as u8
            }
            0x213A => { // VMDATAHREAD
                (self.vram_prefetch >> 8) as u8
            }
            0x213B => 0, // CGDATAREAD — stub
            0x213C => 0, // OPHCT
            0x213D => 0, // OPVCT
            0x213E => 0x01, // STAT77 — version
            0x213F => 0x01, // STAT78 — version
            _ => 0,
        }
    }
}
