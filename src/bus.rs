use crate::apu::Apu;
use crate::cartridge::{make_cartridge, Cartridge};
use crate::joypad::Joypad;
use crate::ppu::Ppu;
use crate::timer::Timer;

/// HDMA state (0xFF51–0xFF55).
struct Hdma {
    src: u16,
    dst: u16,
    /// Remaining blocks (each block = 16 bytes). 0 = inactive.
    blocks: u8,
    /// 0 = General Purpose DMA, 1 = H-Blank DMA
    mode: u8,
    active: bool,
}

impl Hdma {
    fn new() -> Self {
        Hdma { src: 0, dst: 0x8000, blocks: 0, mode: 0, active: false }
    }
}

pub struct Bus {
    pub cart: Box<dyn Cartridge>,
    pub ppu: Ppu,
    pub timer: Timer,
    pub joypad: Joypad,
    pub apu: Apu,

    /// WRAM: bank 0 (0xC000–0xCFFF) + banks 1-7 (0xD000–0xDFFF)
    wram: [[u8; 0x1000]; 8],
    wram_bank: usize, // SVBK register (0xFF70), bank 1-7

    hram: [u8; 0x7F],

    /// Interrupt Flags (0xFF0F)
    pub if_: u8,
    /// Interrupt Enable (0xFFFF)
    pub ie: u8,

    /// Serial registers (stub)
    sb: u8,
    sc: u8,

    /// KEY1 — speed switch (0xFF4D)
    pub key1: u8,
    /// Double-speed mode active
    pub double_speed: bool,

    hdma: Hdma,
}

impl Bus {
    pub fn new(rom: Vec<u8>) -> Self {
        Bus {
            cart: make_cartridge(rom),
            ppu: Ppu::new(),
            timer: Timer::new(),
            joypad: Joypad::new(),
            apu: Apu::new(),
            wram: [[0u8; 0x1000]; 8],
            wram_bank: 1,
            hram: [0u8; 0x7F],
            if_: 0xE1,
            ie: 0x00,
            sb: 0x00,
            sc: 0x7E,
            key1: 0xFF,
            double_speed: false,
            hdma: Hdma::new(),
        }
    }

    // ── Public accessors for Cpu ───────────────────────────────────────────────

    pub fn ie(&self) -> u8 { self.ie }
    pub fn if_reg(&self) -> u8 { self.if_ }
    pub fn if_mut(&mut self) -> &mut u8 { &mut self.if_ }

    // ── Memory read ───────────────────────────────────────────────────────────

    pub fn read_byte(&self, addr: u16) -> u8 {
        match addr {
            0x0000..=0x7FFF => self.cart.read_rom(addr),
            0x8000..=0x9FFF => self.ppu.read_vram(addr),
            0xA000..=0xBFFF => self.cart.read_ram(addr),
            0xC000..=0xCFFF => self.wram[0][(addr - 0xC000) as usize],
            0xD000..=0xDFFF => self.wram[self.wram_bank][(addr - 0xD000) as usize],
            0xE000..=0xEFFF => self.wram[0][(addr - 0xE000) as usize], // echo
            0xF000..=0xFDFF => self.wram[self.wram_bank][(addr - 0xF000) as usize], // echo
            0xFE00..=0xFE9F => self.ppu.read_oam(addr),
            0xFEA0..=0xFEFF => 0xFF, // unusable
            0xFF00..=0xFF7F => self.read_io(addr),
            0xFF80..=0xFFFE => self.hram[(addr - 0xFF80) as usize],
            0xFFFF => self.ie,
        }
    }

    pub fn read_word(&self, addr: u16) -> u16 {
        let lo = self.read_byte(addr) as u16;
        let hi = self.read_byte(addr.wrapping_add(1)) as u16;
        (hi << 8) | lo
    }

    // ── Memory write ──────────────────────────────────────────────────────────

    pub fn write_byte(&mut self, addr: u16, val: u8) {
        match addr {
            0x0000..=0x7FFF => self.cart.write_rom(addr, val),
            0x8000..=0x9FFF => self.ppu.write_vram(addr, val),
            0xA000..=0xBFFF => self.cart.write_ram(addr, val),
            0xC000..=0xCFFF => self.wram[0][(addr - 0xC000) as usize] = val,
            0xD000..=0xDFFF => self.wram[self.wram_bank][(addr - 0xD000) as usize] = val,
            0xE000..=0xEFFF => self.wram[0][(addr - 0xE000) as usize] = val,
            0xF000..=0xFDFF => self.wram[self.wram_bank][(addr - 0xF000) as usize] = val,
            0xFE00..=0xFE9F => self.ppu.write_oam(addr, val),
            0xFEA0..=0xFEFF => {} // unusable
            0xFF00..=0xFF7F => self.write_io(addr, val),
            0xFF80..=0xFFFE => self.hram[(addr - 0xFF80) as usize] = val,
            0xFFFF => self.ie = val,
        }
    }

    pub fn write_word(&mut self, addr: u16, val: u16) {
        self.write_byte(addr, (val & 0xFF) as u8);
        self.write_byte(addr.wrapping_add(1), (val >> 8) as u8);
    }

    // ── I/O register read ─────────────────────────────────────────────────────

    fn read_io(&self, addr: u16) -> u8 {
        match addr {
            0xFF00 => self.joypad.read(),
            0xFF01 => self.sb,
            0xFF02 => self.sc | 0x7E,
            0xFF03 => 0xFF,
            0xFF04..=0xFF07 => self.timer.read(addr),
            0xFF0F => self.if_ | 0xE0,
            0xFF10..=0xFF3F => self.apu.read(addr),
            0xFF40..=0xFF4B | 0xFF4F | 0xFF68..=0xFF6B => self.ppu.read(addr),
            0xFF4D => self.key1,
            0xFF51 => (self.hdma.src >> 8) as u8,
            0xFF52 => (self.hdma.src & 0xFF) as u8,
            0xFF53 => (self.hdma.dst >> 8) as u8,
            0xFF54 => (self.hdma.dst & 0xFF) as u8,
            0xFF55 => {
                if self.hdma.active {
                    self.hdma.blocks.wrapping_sub(1) & 0x7F
                } else {
                    0xFF
                }
            }
            0xFF70 => self.wram_bank as u8 | 0xF8,
            _ => 0xFF,
        }
    }

    // ── I/O register write ────────────────────────────────────────────────────

    fn write_io(&mut self, addr: u16, val: u8) {
        match addr {
            0xFF00 => self.joypad.write(val),
            0xFF01 => self.sb = val,
            0xFF02 => {
                self.sc = val;
                // Serial transfer start (bit7=1): print byte to stdout for test ROMs
                if val & 0x80 != 0 {
                    print!("{}", self.sb as char);
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                }
            }
            0xFF04..=0xFF07 => self.timer.write(addr, val),
            0xFF0F => self.if_ = val | 0xE0,
            0xFF10..=0xFF3F => self.apu.write(addr, val),
            0xFF46 => self.oam_dma(val),
            0xFF40..=0xFF45 | 0xFF47..=0xFF4B | 0xFF4F | 0xFF68..=0xFF6B => self.ppu.write(addr, val),
            0xFF4D => {
                // KEY1: prepare speed switch (bit 0 = switch request)
                self.key1 = (self.key1 & 0x80) | (val & 0x01);
            }
            0xFF51 => self.hdma.src = (self.hdma.src & 0x00FF) | ((val as u16) << 8),
            0xFF52 => self.hdma.src = (self.hdma.src & 0xFF00) | ((val & 0xF0) as u16),
            0xFF53 => self.hdma.dst = (self.hdma.dst & 0x00FF) | (((val & 0x1F) as u16) << 8) | 0x8000,
            0xFF54 => self.hdma.dst = (self.hdma.dst & 0xFF00) | ((val & 0xF0) as u16),
            0xFF55 => self.start_hdma(val),
            0xFF70 => {
                let bank = (val & 0x07) as usize;
                self.wram_bank = if bank == 0 { 1 } else { bank };
            }
            _ => {}
        }
    }

    // ── OAM DMA ───────────────────────────────────────────────────────────────

    fn oam_dma(&mut self, source_page: u8) {
        // Instant OAM DMA: copy 0xA0 bytes from (source_page << 8) to OAM
        let base = (source_page as u16) << 8;
        for i in 0..0xA0u16 {
            let byte = self.read_byte(base + i);
            self.ppu.write_oam(0xFE00 + i, byte);
        }
        self.ppu.write(0xFF46, source_page);
    }

    // ── HDMA ──────────────────────────────────────────────────────────────────

    fn start_hdma(&mut self, val: u8) {
        if val == 0xFF && self.hdma.active {
            // Terminate H-Blank DMA
            self.hdma.active = false;
            return;
        }
        let blocks = (val & 0x7F) + 1;
        let mode = (val >> 7) & 1;
        self.hdma.blocks = blocks;
        self.hdma.mode = mode;
        self.hdma.active = true;

        if mode == 0 {
            // General purpose DMA: transfer immediately
            self.do_hdma_transfer(blocks);
            self.hdma.active = false;
        }
        // H-Blank DMA: transfer one block per HBlank, handled in tick()
    }

    fn do_hdma_transfer(&mut self, blocks: u8) {
        for _ in 0..blocks {
            for byte_off in 0..16u16 {
                let src_addr = self.hdma.src + byte_off;
                let dst_addr = self.hdma.dst + byte_off;
                let byte = self.read_byte(src_addr);
                self.ppu.write_vram(dst_addr, byte);
            }
            self.hdma.src = self.hdma.src.wrapping_add(16);
            self.hdma.dst = self.hdma.dst.wrapping_add(16);
            // Wrap dst within VRAM
            let dst_off = (self.hdma.dst - 0x8000) % 0x2000;
            self.hdma.dst = 0x8000 + dst_off;
        }
    }

    // ── Speed switch (called by CPU on STOP) ──────────────────────────────────

    pub fn do_speed_switch(&mut self) {
        if self.key1 & 0x01 != 0 {
            self.double_speed = !self.double_speed;
            if self.double_speed {
                self.key1 = 0x80; // bit7=1: double speed active, bit0=0: no pending switch
            } else {
                self.key1 = 0x00;
            }
        }
    }

    // ── Tick: advance all components by T-cycles ──────────────────────────────

    /// Tick the bus by one M-cycle (4 T-cycles normal speed, 2 in double-speed).
    /// Call this once per CPU M-cycle (memory access or internal cycle).
    pub fn tick_mcycle(&mut self) {
        let cycles = if self.double_speed { 2 } else { 4 };
        self.tick(cycles);
    }

    /// Advance all bus components by `cycles` T-cycles (at normal 4MHz rate).
    /// The caller (emulator) divides by speed factor before calling for PPU/timer.
    pub fn tick(&mut self, cycles: u32) {
        self.timer.step(cycles);
        if self.timer.interrupt {
            self.if_ |= 0x04;
            self.timer.clear_interrupt();
        }

        let ppu_flags = self.ppu.step(cycles);
        self.if_ |= ppu_flags;

        self.apu.step(cycles);

        if self.joypad.interrupt {
            self.if_ |= 0x10;
            self.joypad.clear_interrupt();
        }

        // H-Blank HDMA: transfer one block each time the PPU enters Mode 0
        if self.ppu.hblank_entered {
            self.ppu.hblank_entered = false;
            if self.hdma.active && self.hdma.mode == 1 {
                self.do_hdma_transfer(1);
                if self.hdma.blocks > 0 {
                    self.hdma.blocks -= 1;
                }
                if self.hdma.blocks == 0 {
                    self.hdma.active = false;
                }
            }
        }
    }

    pub fn frame_ready(&self) -> bool {
        self.ppu.frame_ready
    }

    pub fn clear_frame_ready(&mut self) {
        self.ppu.frame_ready = false;
    }
}
