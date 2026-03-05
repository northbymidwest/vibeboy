use crate::apu::Apu;
use crate::cartridge::{make_cartridge, Cartridge};
use crate::joypad::Joypad;
use crate::model::GbModel;
use crate::ppu::Ppu;
use crate::timer::Timer;

use std::path::{Path, PathBuf};

/// OAM DMA state.
///
/// Timing (from hardware):
///   M=0: write to 0xFF46
///   M=1: delay — OAM accessible for fresh start, still blocked for restart
///   M=2..161: DMA copies 1 byte per M-cycle, OAM inaccessible
///
/// Blocking rule:
///   - Before first byte is copied (progress==0): use `was_blocking`
///     (false for fresh start → accessible; true for restart → still blocked)
///   - After first byte (progress>0 while active): always blocking
///   - After DMA finishes (active==false): never blocking
struct OamDma {
    active: bool,
    source: u16,        // source base address (source_page << 8)
    progress: u8,       // bytes copied so far (0–159)
    delay: u8,          // M-cycles remaining before first byte copy
    was_blocking: bool, // OAM was blocked when DMA was (re)started
    /// Blocking state captured at the *start* of the current M-cycle (before step_oam_dma
    /// runs). CPU reads/writes check this so that the M-cycle where DMA copies its last
    /// byte still blocks OAM, even though `active` becomes false during that same step.
    blocking: bool,
}

impl OamDma {
    fn new() -> Self {
        OamDma { active: false, source: 0, progress: 0, delay: 0, was_blocking: false, blocking: false }
    }
    fn is_blocking(&self) -> bool {
        self.blocking
    }
    /// Compute whether OAM should be blocked this M-cycle, based on current DMA state
    /// (called before step_oam_dma so the result reflects the start of the M-cycle).
    fn compute_blocking(&self) -> bool {
        if !self.active { return false; }
        if self.delay > 0 { return self.was_blocking; }
        true // actively copying
    }
}

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
    /// Buffer of serial output bytes (for test ROM detection)
    pub serial_output: Vec<u8>,

    /// KEY1 — speed switch (0xFF4D)
    pub key1: u8,
    /// Double-speed mode active
    pub double_speed: bool,

    oam_dma: OamDma,
    hdma: Hdma,

    /// Boot ROM bytes (up to 0x900 for CGB). None = skip boot ROM.
    boot_rom: Option<Vec<u8>>,
    /// Whether the boot ROM is still mapped (cleared by writing 0xFF50).
    boot_rom_active: bool,

    /// Path to .sav file (set when cartridge has battery).
    save_path: Option<PathBuf>,

    /// Hardware model (DMG, CGB, etc.)
    model: GbModel,
}

impl Bus {
    pub fn new(rom: Vec<u8>, boot_rom: Option<Vec<u8>>, rom_path: Option<&Path>, model: GbModel) -> Self {
        let boot_rom_active = boot_rom.is_some();

        let mut ppu = Ppu::new();
        ppu.cgb_mode = model.is_cgb();
        if boot_rom_active {
            ppu.reset();
        }

        let mut joypad = Joypad::new();
        if !boot_rom_active {
            // Post-boot P1: boot ROM writes 0x00, clearing both select bits.
            // This makes P1 read as 0xCF (both groups selected, no buttons pressed).
            joypad.write(0x00);
        }

        let mut cart = make_cartridge(rom);

        // Compute .sav path and load existing save data
        let save_path = rom_path
            .filter(|_| cart.has_battery())
            .map(|p| p.with_extension("sav"));
        if let Some(ref sp) = save_path {
            if let Ok(data) = std::fs::read(sp) {
                log::info!("Loaded save from {}", sp.display());
                cart.load_ram(&data);
            }
        }

        Bus {
            cart,
            ppu,
            timer: Timer::post_boot(model),
            joypad,
            apu: Apu::new(),
            wram: [[0u8; 0x1000]; 8],
            wram_bank: 1,
            hram: [0u8; 0x7F],
            if_: 0xE1,
            ie: 0x00,
            sb: 0x00,
            sc: 0x7E,
            serial_output: Vec::new(),
            key1: 0x00,
            double_speed: false,
            oam_dma: OamDma::new(),
            hdma: Hdma::new(),
            boot_rom,
            boot_rom_active,
            save_path,
            model,
        }
    }

    pub fn has_boot_rom(&self) -> bool {
        self.boot_rom.is_some()
    }

    /// Write cartridge RAM to .sav file if battery-backed.
    pub fn save_to_disk(&self) {
        if let Some(ref path) = self.save_path {
            let data = self.cart.save_data();
            if !data.is_empty() {
                if let Err(e) = std::fs::write(path, data) {
                    log::error!("Failed to write save file '{}': {}", path.display(), e);
                } else {
                    log::info!("Saved to {}", path.display());
                }
            }
        }
    }

    // ── Public accessors for Cpu ───────────────────────────────────────────────

    pub fn ie(&self) -> u8 { self.ie }
    pub fn if_reg(&self) -> u8 { self.if_ }
    pub fn if_mut(&mut self) -> &mut u8 { &mut self.if_ }

    // ── Memory read ───────────────────────────────────────────────────────────

    /// CPU memory read. On CGB, OAM DMA only blocks OAM (0xFE00–0xFE9F);
    /// all other memory (ROM, WRAM, VRAM, I/O, HRAM) remains accessible.
    pub fn read_byte(&self, addr: u16) -> u8 {
        if self.oam_dma.is_blocking() {
            if matches!(addr, 0xFE00..=0xFE9F) {
                return 0xFF;
            }
        }
        self.read_byte_raw(addr)
    }

    /// Raw read bypassing DMA bus-conflict logic (for DMA/HDMA controllers).
    fn read_byte_raw(&self, addr: u16) -> u8 {
        // Boot ROM overlay: covers 0x0000-0x00FF and (for CGB) 0x0200-0x08FF.
        if self.boot_rom_active {
            if let Some(ref brom) = self.boot_rom {
                let idx = match addr {
                    0x0000..=0x00FF => Some(addr as usize),
                    0x0200..=0x08FF if self.model.is_cgb() => Some(addr as usize),
                    _ => None,
                };
                if let Some(i) = idx {
                    if i < brom.len() {
                        return brom[i];
                    }
                }
            }
        }
        match addr {
            0x0000..=0x7FFF => self.cart.read_rom(addr),
            0x8000..=0x9FFF => {
                if !self.ppu.vram_accessible { 0xFF } else { self.ppu.read_vram(addr) }
            }
            0xA000..=0xBFFF => self.cart.read_ram(addr),
            0xC000..=0xCFFF => self.wram[0][(addr - 0xC000) as usize],
            0xD000..=0xDFFF => self.wram[self.wram_bank][(addr - 0xD000) as usize],
            0xE000..=0xEFFF => self.wram[0][(addr - 0xE000) as usize], // echo
            0xF000..=0xFDFF => self.wram[self.wram_bank][(addr - 0xF000) as usize], // echo
            0xFE00..=0xFE9F => {
                if !self.ppu.oam_accessible { 0xFF } else { self.ppu.read_oam(addr) }
            }
            0xFEA0..=0xFEFF => 0xFF, // unusable
            0xFF00..=0xFF7F => self.read_io(addr),
            0xFF80..=0xFFFE => self.hram[(addr - 0xFF80) as usize],
            0xFFFF          => self.ie,
        }
    }

    pub fn read_word(&self, addr: u16) -> u16 {
        let lo = self.read_byte(addr) as u16;
        let hi = self.read_byte(addr.wrapping_add(1)) as u16;
        (hi << 8) | lo
    }

    // ── Memory write ──────────────────────────────────────────────────────────

    /// CPU memory write. On CGB, OAM DMA only blocks OAM writes;
    /// all other writes (ROM, WRAM, I/O, HRAM) proceed normally.
    pub fn write_byte(&mut self, addr: u16, val: u8) {
        if self.oam_dma.is_blocking() {
            if matches!(addr, 0xFE00..=0xFE9F) {
                return; // OAM writes ignored during DMA
            }
        }
        self.write_byte_raw(addr, val);
    }

    fn write_byte_raw(&mut self, addr: u16, val: u8) {
        match addr {
            0x0000..=0x7FFF => self.cart.write_rom(addr, val),
            0x8000..=0x9FFF => {
                if self.ppu.vram_accessible { self.ppu.write_vram(addr, val); }
            }
            0xA000..=0xBFFF => self.cart.write_ram(addr, val),
            0xC000..=0xCFFF => self.wram[0][(addr - 0xC000) as usize] = val,
            0xD000..=0xDFFF => self.wram[self.wram_bank][(addr - 0xD000) as usize] = val,
            0xE000..=0xEFFF => self.wram[0][(addr - 0xE000) as usize] = val,
            0xF000..=0xFDFF => self.wram[self.wram_bank][(addr - 0xF000) as usize] = val,
            0xFE00..=0xFE9F => {
                if self.ppu.oam_accessible { self.ppu.write_oam(addr, val); }
            }
            0xFEA0..=0xFEFF => {} // unusable
            0xFF00..=0xFF7F => self.write_io(addr, val),
            0xFF80..=0xFFFE => self.hram[(addr - 0xFF80) as usize] = val,
            0xFFFF          => self.ie = val,
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
            0xFF40..=0xFF4B => self.ppu.read(addr),
            0xFF4D => {
                if !self.model.is_cgb() { return 0xFF; }
                self.key1 | 0x7E
            }
            0xFF4F | 0xFF68..=0xFF6B => {
                if !self.model.is_cgb() { return 0xFF; }
                self.ppu.read(addr)
            }
            0xFF51..=0xFF55 => {
                if !self.model.is_cgb() { return 0xFF; }
                match addr {
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
                    _ => 0xFF,
                }
            }
            0xFF50 => if self.boot_rom_active { 0xFE } else { 0xFF },
            0xFF70 => {
                if !self.model.is_cgb() { return 0xFF; }
                self.wram_bank as u8 | 0xF8
            }
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
                    self.serial_output.push(self.sb);
                    print!("{}", self.sb as char);
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                }
            }
            0xFF04..=0xFF07 => {
                self.timer.write(addr, val);
                if self.timer.interrupt {
                    self.if_ |= 0x04;
                    self.timer.clear_interrupt();
                }
            }
            0xFF0F => self.if_ = val | 0xE0,
            0xFF10..=0xFF3F => self.apu.write(addr, val),
            0xFF46 => self.start_oam_dma(val),
            0xFF4F | 0xFF68..=0xFF6B if !self.model.is_cgb() => {} // ignore on DMG
            0xFF40..=0xFF45 | 0xFF47..=0xFF4B | 0xFF4F | 0xFF68..=0xFF6B => {
                self.ppu.write(addr, val);
                // Immediately transfer any interrupt flags from register writes
                // (e.g., LCD enable triggering STAT, LYC write causing coincidence)
                if self.ppu.if_flags != 0 {
                    self.if_ |= self.ppu.if_flags;
                    self.ppu.if_flags = 0;
                }
            }
            0xFF4D if !self.model.is_cgb() => {} // ignore on DMG
            0xFF4D => {
                // KEY1: prepare speed switch (bit 0 = switch request)
                self.key1 = (self.key1 & 0x80) | (val & 0x01);
            }
            0xFF51..=0xFF55 if !self.model.is_cgb() => {} // ignore on DMG
            0xFF51 => self.hdma.src = (self.hdma.src & 0x00FF) | ((val as u16) << 8),
            0xFF52 => self.hdma.src = (self.hdma.src & 0xFF00) | ((val & 0xF0) as u16),
            0xFF53 => self.hdma.dst = (self.hdma.dst & 0x00FF) | (((val & 0x1F) as u16) << 8) | 0x8000,
            0xFF54 => self.hdma.dst = (self.hdma.dst & 0xFF00) | ((val & 0xF0) as u16),
            0xFF55 => self.start_hdma(val),
            0xFF50 => {
                // Writing any non-zero value permanently disables the boot ROM.
                if val != 0 {
                    self.boot_rom_active = false;
                    // Keep cgb_mode=true: the boot ROM programs CGB palettes
                    // for both CGB and DMG games (DMG compat palette).
                }
            }
            0xFF70 if !self.model.is_cgb() => {} // ignore on DMG
            0xFF70 => {
                let bank = (val & 0x07) as usize;
                self.wram_bank = if bank == 0 { 1 } else { bank };
            }
            _ => {}
        }
    }

    // ── OAM DMA ───────────────────────────────────────────────────────────────

    fn start_oam_dma(&mut self, source_page: u8) {
        // Store source page in PPU register so 0xFF46 reads back correctly.
        self.ppu.write(0xFF46, source_page);
        // Use the pre-step blocking state captured at the start of this M-cycle.
        let was_blocking = self.oam_dma.blocking;
        // Schedule DMA: 1 M-cycle delay before blocking starts, then 160 transfers.
        self.oam_dma = OamDma {
            active:       true,
            source:       (source_page as u16) << 8,
            progress:     0,
            delay:        1,
            was_blocking,
            blocking:     was_blocking,
        };
    }

    /// Advance OAM DMA by one M-cycle. Called from tick_mcycle().
    fn step_oam_dma(&mut self) {
        if !self.oam_dma.active { return; }
        if self.oam_dma.delay > 0 {
            self.oam_dma.delay -= 1;
            return;
        }
        // Copy one byte from source to OAM.
        let src = self.oam_dma.source + self.oam_dma.progress as u16;
        let byte = self.read_byte_raw(src);
        self.ppu.oam[self.oam_dma.progress as usize] = byte;
        self.oam_dma.progress += 1;
        if self.oam_dma.progress >= 160 {
            self.oam_dma.active = false;
        }
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
                let byte = self.read_byte_raw(src_addr);
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
        // Capture blocking state BEFORE advancing DMA so CPU accesses in this M-cycle
        // see the correct blocking state (e.g. last DMA copy still blocks OAM).
        self.oam_dma.blocking = self.oam_dma.compute_blocking();
        // Timer is clocked by the CPU, so always 4 T-cycles per M-cycle.
        // PPU/APU run at fixed 4MHz, so 2 T-cycles per M-cycle in double-speed.
        let bus_cycles = if self.double_speed { 2 } else { 4 };
        self.tick(4, bus_cycles);
        self.step_oam_dma();
    }

    /// Advance all bus components. `timer_cycles` is CPU-clock T-cycles (always 4 per M-cycle).
    /// `bus_cycles` is 4MHz-rate T-cycles (4 normal, 2 double-speed).
    pub fn tick(&mut self, timer_cycles: u32, bus_cycles: u32) {
        self.timer.step(timer_cycles);
        if self.timer.interrupt {
            self.if_ |= 0x04;
            self.timer.clear_interrupt();
        }

        let ppu_flags = self.ppu.step(bus_cycles);
        self.if_ |= ppu_flags;

        self.apu.step(bus_cycles);

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
