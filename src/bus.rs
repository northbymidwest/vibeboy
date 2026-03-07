use crate::apu::Apu;
use crate::cartridge::{make_cartridge, Cartridge};
use crate::joypad::Joypad;
use crate::model::GbModel;
use crate::ppu::Ppu;
use crate::serial::Serial;
use crate::sgb::Sgb;
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
#[derive(Clone)]
pub(crate) struct OamDma {
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
#[derive(Clone)]
pub(crate) struct Hdma {
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
    pub(crate) wram: [[u8; 0x1000]; 8],
    pub(crate) wram_bank: usize, // SVBK register (0xFF70), bank 1-7

    pub(crate) hram: [u8; 0x7F],

    /// Interrupt Flags (0xFF0F)
    pub if_: u8,
    /// Interrupt Enable (0xFFFF)
    pub ie: u8,

    /// Serial port (SB/SC registers + attached device)
    pub serial: Serial,

    /// KEY1 — speed switch (0xFF4D)
    pub key1: u8,
    /// Double-speed mode active
    pub double_speed: bool,

    pub(crate) oam_dma: OamDma,
    pub(crate) hdma: Hdma,

    /// Boot ROM bytes (up to 0x900 for CGB). None = skip boot ROM.
    boot_rom: Option<Vec<u8>>,
    /// Whether the boot ROM is still mapped (cleared by writing 0xFF50).
    pub(crate) boot_rom_active: bool,

    /// Path to .sav file (set when cartridge has battery).
    save_path: Option<PathBuf>,

    /// Hardware model (DMG, CGB, etc.)
    pub(crate) model: GbModel,

    /// Debug: last CPU PC that triggered OAM bug
    pub debug_oam_pc: u16,

    /// SGB command processor (only present for SGB/SGB2 models)
    pub sgb: Option<Sgb>,
}

impl Bus {
    pub fn new(rom: Vec<u8>, boot_rom: Option<Vec<u8>>, rom_path: Option<&Path>, model: GbModel) -> Self {
        let boot_rom_active = boot_rom.is_some();

        let mut ppu = Ppu::new();
        ppu.cgb_mode = model.is_cgb();
        ppu.sgb_mode = model.is_sgb();
        if model.is_sgb() {
            ppu.cgb_mode = false;
        }
        if boot_rom_active {
            ppu.reset();
        }

        // Detect DMG game running on CGB hardware (no boot ROM)
        let cgb_flag = rom.get(0x0143).copied().unwrap_or(0);
        let is_dmg_game = cgb_flag != 0x80 && cgb_flag != 0xC0;
        if model.is_cgb() && is_dmg_game && !boot_rom_active {
            ppu.dmg_compat = true;
            // Default grayscale reference colors (no boot ROM to set real ones)
            let grays: [u16; 4] = [0x7FFF, 0x56B5, 0x294A, 0x0000];
            ppu.dmg_bg_ref = grays;
            ppu.dmg_obj_ref = [grays, grays];
            // Sync initial post-boot palette values to CGB palette RAM
            ppu.sync_dmg_palette_to_cgb(0xFC, false, 0);
            ppu.sync_dmg_palette_to_cgb(0xFF, true, 0);
            ppu.sync_dmg_palette_to_cgb(0xFF, true, 1);
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
            timer: if boot_rom_active { Timer::reset() } else { Timer::post_boot(model) },
            joypad,
            apu: Apu::new(model.cpu_clock_rate(), model.is_cgb()),
            wram: [[0u8; 0x1000]; 8],
            wram_bank: 1,
            hram: [0u8; 0x7F],
            if_: if boot_rom_active { 0x00 } else { 0xE1 },
            ie: 0x00,
            serial: Serial::new(model.is_cgb()),
            key1: 0x00,
            double_speed: false,
            oam_dma: OamDma::new(),
            hdma: Hdma::new(),
            boot_rom,
            boot_rom_active,
            save_path,
            model,
            debug_oam_pc: 0,
            sgb: if model.is_sgb() {
                let mut sgb = Sgb::new();
                if !boot_rom_active {
                    sgb.protocol_active = true;
                }
                Some(sgb)
            } else {
                None
            },
        }
    }

    pub fn has_boot_rom(&self) -> bool {
        self.boot_rom.is_some()
    }

    /// Create a snapshot of the bus state for rewind / save states.
    pub fn take_snapshot(&self) -> crate::snapshot::BusSnapshot {
        let mut apu_clone = self.apu.clone();
        apu_clone.sample_buf.clear();
        crate::snapshot::BusSnapshot {
            ppu: self.ppu.clone(),
            timer: self.timer.clone(),
            joypad: self.joypad.clone(),
            apu: apu_clone,
            wram: self.wram,
            wram_bank: self.wram_bank,
            hram: self.hram,
            if_: self.if_,
            ie: self.ie,
            serial: self.serial.take_snapshot(),
            key1: self.key1,
            double_speed: self.double_speed,
            oam_dma: self.oam_dma.clone(),
            hdma: self.hdma.clone(),
            boot_rom_active: self.boot_rom_active,
            model: self.model,
            sgb: self.sgb.clone(),
            cart_state: self.cart.snapshot_state(),
        }
    }

    /// Restore the bus state from a snapshot.
    pub fn apply_snapshot(&mut self, s: &crate::snapshot::BusSnapshot) {
        self.ppu = s.ppu.clone();
        self.timer = s.timer.clone();
        self.joypad = s.joypad.clone();
        self.apu = s.apu.clone();
        self.apu.sample_buf.clear();
        self.wram = s.wram;
        self.wram_bank = s.wram_bank;
        self.hram = s.hram;
        self.if_ = s.if_;
        self.ie = s.ie;
        self.serial.apply_snapshot(&s.serial);
        self.key1 = s.key1;
        self.double_speed = s.double_speed;
        self.oam_dma = s.oam_dma.clone();
        self.hdma = s.hdma.clone();
        self.boot_rom_active = s.boot_rom_active;
        self.sgb = s.sgb.clone();
        self.cart.restore_state(&s.cart_state);
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
                let v = if !self.ppu.oam_accessible { 0xFF } else { self.ppu.read_oam(addr) };
                // Debug: log reads in corrupted region when LCD is off (cp_oam context)
                if addr >= 0xFE18 && addr <= 0xFE1F && self.ppu.lcdc & 0x80 == 0 {
                    let idx = (addr - 0xFE00) as usize;
                    let expected = 0x0Cu8.wrapping_add(idx as u8);
                    log::warn!("OAM READ ${:04X}: got={:02X} expected={:02X} match={} oam_accessible={} pc={:04X}",
                        addr, v, expected, v == expected, self.ppu.oam_accessible, self.debug_oam_pc);
                }
                v
            }
            0xFEA0..=0xFEFF => {
                // DMG/MGB/SGB: reads return 0x00
                // CGB: model-dependent patterns (simplified to 0x00)
                0x00
            }
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
                log::trace!("OAM write BLOCKED by DMA: addr={:04X} val={:02X}", addr, val);
                return; // OAM writes ignored during DMA
            }
        }
        // DMG OAM bug: writes to OAM range during Mode 2 trigger corruption
        if !self.ppu.oam_accessible && addr >= 0xFE00 && addr < 0xFF00 {
            self.trigger_oam_bug_from_write(addr);
        }
        // Debug: log OAM writes
        if addr >= 0xFE00 && addr <= 0xFE9F && self.ppu.oam_accessible {
            log::trace!("OAM write OK: addr={:04X} val={:02X}", addr, val);
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
            0xFF00 => {
                if let Some(ref sgb) = self.sgb {
                    if sgb.player_count > 1 {
                        // When both select lines high, return player ID
                        let p1_select = self.joypad.read() & 0x30;
                        if p1_select == 0x30 {
                            return 0xC0 | 0x30 | sgb.read_p1_id();
                        }
                    }
                }
                self.joypad.read()
            }
            0xFF01 => self.serial.sb,
            0xFF02 => self.serial.read_sc(),
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
            0xFF76 => if self.model.is_cgb() { self.apu.pcm12() } else { 0xFF },
            0xFF77 => if self.model.is_cgb() { self.apu.pcm34() } else { 0xFF },
            _ => 0xFF,
        }
    }

    // ── I/O register write ────────────────────────────────────────────────────

    fn write_io(&mut self, addr: u16, val: u8) {
        match addr {
            0xFF00 => {
                self.joypad.write(val);
                if let Some(ref mut sgb) = self.sgb {
                    sgb.write_p1(val);
                }
            }
            0xFF01 => self.serial.sb = val,
            0xFF02 => {
                self.serial.write_sc(val);
                // Print to stdout for test ROMs (write_sc pushes to serial_output)
                if val & 0x80 != 0 {
                    print!("{}", self.serial.sb as char);
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                }
            }
            0xFF04..=0xFF07 => {
                let old_div = self.timer.counter();
                self.timer.write(addr, val);
                let new_div = self.timer.counter();
                if self.timer.interrupt {
                    self.if_ |= 0x04;
                    self.timer.clear_interrupt();
                }
                // DIV reset can create falling edge for serial clock
                self.serial.step(old_div, new_div);
                if self.serial.interrupt {
                    self.if_ |= 0x08;
                    self.serial.interrupt = false;
                }
                // APU: DIV write can create falling/rising edge on APU bit
                if addr == 0xFF04 {
                    let apu_bit: u16 = if self.double_speed { 0x2000 } else { 0x1000 };
                    let triggers = old_div & !new_div;
                    if triggers & apu_bit != 0 {
                        self.apu.div_event();
                    } else {
                        let secondary = !old_div & new_div;
                        if secondary & apu_bit != 0 {
                            self.apu.div_secondary_event();
                        }
                    }
                    self.apu.set_div_counter(new_div);
                }
            }
            0xFF0F => self.if_ = val | 0xE0,
            0xFF10..=0xFF3F => {
                self.apu.set_div_counter(self.timer.counter());
                self.apu.set_double_speed(self.double_speed);
                self.apu.write(addr, val);
            }
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
                    // Activate SGB protocol now that boot ROM is done
                    if let Some(ref mut sgb) = self.sgb {
                        sgb.protocol_active = true;
                    }
                    // Detect DMG compat: CGB hardware running DMG game
                    let cgb_flag = self.cart.read_rom(0x0143);
                    if self.model.is_cgb() && cgb_flag != 0x80 && cgb_flag != 0xC0 {
                        self.ppu.dmg_compat = true;
                        // Capture current CGB palette RAM as reference colors
                        // (the boot ROM has programmed these)
                        for i in 0..4 {
                            let off = i * 2;
                            self.ppu.dmg_bg_ref[i] = self.ppu.bcpd[off] as u16
                                | ((self.ppu.bcpd[off + 1] as u16) << 8);
                            self.ppu.dmg_obj_ref[0][i] = self.ppu.ocpd[off] as u16
                                | ((self.ppu.ocpd[off + 1] as u16) << 8);
                            self.ppu.dmg_obj_ref[1][i] = self.ppu.ocpd[8 + off] as u16
                                | ((self.ppu.ocpd[8 + off + 1] as u16) << 8);
                        }
                    }
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

    // ── OAM bug (DMG only) ────────────────────────────────────────────────────

    /// Trigger OAM write corruption bug when a 16-bit register pointing to 0xFE00–0xFEFF
    /// is incremented/decremented during PPU Mode 2 (OAM scan). DMG only.
    pub fn trigger_oam_bug(&mut self, addr: u16) {
        self.trigger_oam_bug_inner(addr, "INSTR");
    }

    pub fn trigger_oam_bug_from_write(&mut self, addr: u16) {
        self.trigger_oam_bug_inner(addr, "WRITE");
    }

    fn trigger_oam_bug_inner(&mut self, addr: u16, source: &str) {
        if self.model.is_cgb() { return; }
        if addr < 0xFE00 || addr > 0xFEFF { return; }
        let row = self.ppu.oam_bug_row;

        // SameBoy only checks: row != 0xFF && row >= 8
        // No upper bound check (hardware allows corruption even at accessed_oam_row >= 160)
        if row == 0xFF || row < 8 {
            return;
        }
        let row = row as usize;
        let variant = if source == "INSTR" { "W" } else { "R" };
        log::debug!("OAM_BUG_{}: addr={:04X} row={:02X} ly={} dot={} tt={} pc={:04X}",
            variant, addr, row, self.ppu.ly, self.ppu.dot, self.ppu.total_ticks, self.debug_oam_pc);
        // Bytes 0-1: bitwise glitch (operate as u16 little-endian)
        let a = u16::from_le_bytes([self.ppu.oam[row], self.ppu.oam[row + 1]]);
        let b = u16::from_le_bytes([self.ppu.oam[row - 8], self.ppu.oam[row - 7]]);
        let c = u16::from_le_bytes([self.ppu.oam[row - 4], self.ppu.oam[row - 3]]);
        let glitched = ((a ^ c) & (b ^ c)) ^ c;
        let bytes = glitched.to_le_bytes();
        self.ppu.oam[row] = bytes[0];
        self.ppu.oam[row + 1] = bytes[1];
        // Bytes 2-7: copy from previous row
        for i in 2..8 {
            self.ppu.oam[row + i] = self.ppu.oam[row - 8 + i];
        }
        log::warn!("  AFTER: oam[{}..{}]={:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X}",
            row, row+7,
            self.ppu.oam[row], self.ppu.oam[row+1], self.ppu.oam[row+2], self.ppu.oam[row+3],
            self.ppu.oam[row+4], self.ppu.oam[row+5], self.ppu.oam[row+6], self.ppu.oam[row+7]);
    }

    /// Trigger OAM read corruption bug when a memory read targets 0xFE00–0xFEFF
    /// during PPU Mode 2 (OAM scan). DMG only. Uses different formulas from write corruption.
    pub fn trigger_oam_bug_read(&mut self, addr: u16) {
        if self.model.is_cgb() { return; }
        if addr < 0xFE00 || addr > 0xFEFF { return; }
        let row = self.ppu.oam_bug_row;
        // SameBoy only checks: row != 0xFF && row >= 8
        // No upper bound check (hardware allows corruption even at accessed_oam_row >= 160)
        if row == 0xFF || row < 8 { return; }
        let row = row as usize;
        log::debug!("OAM_BUG_R: addr={:04X} row={:02X} ly={} pc={:04X}",
            addr, row, self.ppu.ly, self.debug_oam_pc);

        if (row & 0x18) == 0x10 {
            // Secondary read corruption — affects row-8 word, then copies two rows back
            if row >= 0x18 { // need row-16 to exist
                let base_m8 = u16::from_le_bytes([self.ppu.oam[row - 16], self.ppu.oam[row - 15]]);
                let base_m4 = u16::from_le_bytes([self.ppu.oam[row - 8], self.ppu.oam[row - 7]]);
                let base_0  = u16::from_le_bytes([self.ppu.oam[row], self.ppu.oam[row + 1]]);
                let base_m2 = u16::from_le_bytes([self.ppu.oam[row - 4], self.ppu.oam[row - 3]]);
                // base[-4] = secondary(base[-8], base[-4], base[0], base[-2])
                let glitched = (base_m4 & (base_m8 | base_0 | base_m2)) | (base_m8 & base_0 & base_m2);
                let bytes = glitched.to_le_bytes();
                self.ppu.oam[row - 8] = bytes[0];
                self.ppu.oam[row - 7] = bytes[1];
                // Copy row-8 to row-16
                for i in 0..8 {
                    self.ppu.oam[row - 0x10 + i] = self.ppu.oam[row - 0x08 + i];
                }
            }
        } else if (row & 0x18) == 0x00 {
            // Tertiary/quaternary — model and row specific
            if row >= 0x20 { // need row-32 to exist for base[-16]
                let base_0   = u16::from_le_bytes([self.ppu.oam[row], self.ppu.oam[row + 1]]);
                let base_m2  = u16::from_le_bytes([self.ppu.oam[row - 4], self.ppu.oam[row - 3]]);
                let base_m4  = u16::from_le_bytes([self.ppu.oam[row - 8], self.ppu.oam[row - 7]]);
                let base_m8  = u16::from_le_bytes([self.ppu.oam[row - 16], self.ppu.oam[row - 15]]);
                let base_m16 = u16::from_le_bytes([self.ppu.oam[row - 32], self.ppu.oam[row - 31]]);

                let glitched = if row == 0x40 {
                    // Quaternary
                    let base_m3  = u16::from_le_bytes([self.ppu.oam[row - 6], self.ppu.oam[row - 5]]);
                    let base_m7  = u16::from_le_bytes([self.ppu.oam[row - 14], self.ppu.oam[row - 13]]);
                    let oam_0    = u16::from_le_bytes([self.ppu.oam[0], self.ppu.oam[1]]);
                    if self.model == crate::model::GbModel::Sgb2 {
                        // sgb2 variant
                        (base_m4 & (base_m16 | base_m8 | base_m2 | (oam_0 & base_0)))
                            | ((base_m2 & base_m8 & base_m16) & (base_0 | oam_0 | !base_m7))
                    } else {
                        // dmg variant (a unused)
                        (base_m4 & (base_m16 | base_m8 | (!base_m3 & base_m7) | base_m2 | base_0))
                            | (base_m2 & base_m8 & base_m16)
                    }
                } else if self.model == crate::model::GbModel::Mgb {
                    // MGB: tertiary_read_3
                    (base_m4 & (base_0 | base_m2 | base_m8 | base_m16)) | (base_m2 & base_m8 & base_m16)
                } else if self.model == crate::model::GbModel::Sgb2 {
                    // SGB2: tertiary_read_2
                    (base_m4 & (base_0 | base_m2 | base_m8 | base_m16)) | (base_0 & base_m2 & base_m8 & base_m16)
                } else if row == 0x20 {
                    // tertiary_read_2
                    (base_m4 & (base_0 | base_m2 | base_m8 | base_m16)) | (base_0 & base_m2 & base_m8 & base_m16)
                } else if row == 0x60 {
                    // tertiary_read_3
                    (base_m4 & (base_0 | base_m2 | base_m8 | base_m16)) | (base_m2 & base_m8 & base_m16)
                } else {
                    // tertiary_read_1 (default)
                    base_m4 | (base_0 & base_m2 & base_m8 & base_m16)
                };

                let bytes = glitched.to_le_bytes();
                self.ppu.oam[row - 8] = bytes[0];
                self.ppu.oam[row - 7] = bytes[1];
                // Copy two rows back
                for i in 0..8 {
                    let v = self.ppu.oam[row - 0x08 + i];
                    self.ppu.oam[row - 0x10 + i] = v;
                    self.ppu.oam[row - 0x20 + i] = v;
                }
            }
        } else {
            // Common case: (row & 0x18) == 0x08 or 0x18
            let a = u16::from_le_bytes([self.ppu.oam[row], self.ppu.oam[row + 1]]);
            let b = u16::from_le_bytes([self.ppu.oam[row - 8], self.ppu.oam[row - 7]]);
            let c = u16::from_le_bytes([self.ppu.oam[row - 4], self.ppu.oam[row - 3]]);
            let glitched = b | (a & c);
            let bytes = glitched.to_le_bytes();
            // Both base[-4] and base[0] get the glitched value
            self.ppu.oam[row - 8] = bytes[0];
            self.ppu.oam[row - 7] = bytes[1];
            self.ppu.oam[row] = bytes[0];
            self.ppu.oam[row + 1] = bytes[1];
        }

        // Copy previous row into current row (shared across all cases)
        for i in 0..8 {
            self.ppu.oam[row + i] = self.ppu.oam[row - 8 + i];
        }

        // Special: row 0x80 copies corruption to row 0
        if row == 0x80 || (self.model == crate::model::GbModel::Mgb && row == 0x40) {
            for i in 0..8 {
                self.ppu.oam[i] = self.ppu.oam[row + i];
            }
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
        // Capture DIV counter before and after timer step for serial/APU edge detection
        let old_div = self.timer.counter();
        self.timer.step(timer_cycles);
        let new_div = self.timer.counter();
        if self.timer.interrupt {
            self.if_ |= 0x04;
            self.timer.clear_interrupt();
        }

        // Serial clock is derived from DIV counter
        self.serial.step(old_div, new_div);
        if self.serial.interrupt {
            self.if_ |= 0x08;
            self.serial.interrupt = false;
        }

        // APU frame sequencer is clocked by DIV bit 12 (or 13 in double speed)
        let apu_bit: u16 = if self.double_speed { 0x2000 } else { 0x1000 };
        let triggers = old_div & !new_div; // bits that fell
        if triggers & apu_bit != 0 {
            self.apu.div_event();
        } else {
            let secondary = !old_div & new_div; // bits that rose
            if secondary & apu_bit != 0 {
                self.apu.div_secondary_event();
            }
        }
        self.apu.set_div_counter(new_div);
        self.apu.set_double_speed(self.double_speed);

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

    // ── SGB ──────────────────────────────────────────────────────────────────

    /// Apply SGB palettes to the frame buffer using the shade buffer.
    /// Call after PPU renders a complete frame.
    pub fn apply_sgb_palettes(&mut self) {
        if let Some(ref sgb) = self.sgb {
            match sgb.mask_mode {
                0 => {
                    // Normal: remap using shade buffer
                    sgb.apply_palettes(&self.ppu.shade_buffer, &mut self.ppu.frame_buffer);
                }
                1 => {
                    // Freeze: show frozen buffer
                    // (frozen_buffer captured at mask time — nothing to do here,
                    //  the emulator will use frozen_buffer directly)
                }
                2 => {
                    // Black screen
                    for p in self.ppu.frame_buffer.iter_mut() {
                        *p = 0x00000000;
                    }
                }
                3 => {
                    // Color 0 of palette 0
                    let c = Sgb::rgb555_to_rgb32(sgb.palettes[0][0]);
                    for p in self.ppu.frame_buffer.iter_mut() {
                        *p = c;
                    }
                }
                _ => {}
            }
        }
    }

    /// Check for pending SGB VRAM transfers and execute them.
    /// Call after each frame.
    pub fn check_sgb_transfer(&mut self) {
        // Tick the transfer countdown
        if let Some(ref mut sgb) = self.sgb {
            sgb.tick_transfer();
        }
        let has_transfer = self.sgb.as_ref().map_or(false, |s: &Sgb| s.has_pending_transfer());
        if !has_transfer { return; }

        // Read VRAM tiles directly to reconstruct the 4096-byte transfer data.
        // The game writes data as tiles at $8000-$8FFF, sets up the tilemap with
        // tiles $00-$FF in order, then sends the TRN command.
        let lcdc = self.ppu.read(0xFF40);
        let tile_data_base: usize = if lcdc & 0x10 != 0 { 0x0000 } else { 0x0800 };
        let tile_map_base: usize = if lcdc & 0x08 != 0 { 0x1C00 } else { 0x1800 };
        let signed_addr = lcdc & 0x10 == 0;

        let mut vram_data = vec![0u8; 4096];
        for tile_idx in 0..256usize {
            let map_x = tile_idx % 20;
            let map_y = tile_idx / 20;
            let map_addr = tile_map_base + map_y * 32 + map_x;
            let raw_tile = self.ppu.vram[0][map_addr];

            let tile_offset = if signed_addr {
                let signed_idx = raw_tile as i8 as i16;
                ((0x1000 + signed_idx * 16) as usize) & 0x1FFF
            } else {
                raw_tile as usize * 16
            };

            let dst_base = tile_idx * 16;
            for b in 0..16 {
                if tile_offset + b < self.ppu.vram[0].len() && dst_base + b < 4096 {
                    vram_data[dst_base + b] = self.ppu.vram[0][tile_offset + b];
                }
            }
        }

        if let Some(ref mut sgb) = self.sgb {
            sgb.execute_transfer(&vram_data);
        }
    }

    /// Capture the current frame for MASK_EN(1) freeze mode.
    pub fn capture_sgb_freeze(&mut self) {
        if let Some(ref mut sgb) = self.sgb {
            if sgb.mask_mode == 1 {
                sgb.frozen_buffer.copy_from_slice(&self.ppu.frame_buffer);
            }
        }
    }
}
