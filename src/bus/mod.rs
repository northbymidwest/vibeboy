use crate::apu::Apu;
use crate::cartridge::{make_cartridge, Cartridge};
use crate::clock::Clock;
use crate::joypad::Joypad;
use crate::model::GbModel;
use crate::ppu::Ppu;
use crate::serial::Serial;
use crate::sgb::Sgb;
use crate::timer::Timer;


mod dma;
mod io;
mod oam_bug;

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
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct OamDma {
    active: bool,
    source: u16,        // source base address (source_page << 8)
    progress: u8,       // M-cycles elapsed (0–161)
    delay: u8,          // M-cycles remaining before first bus read
    was_blocking: bool, // OAM was blocked when DMA was (re)started
    /// Blocking state captured at the *start* of the current M-cycle (before step_oam_dma
    /// runs). CPU reads/writes check this so that the M-cycle where DMA copies its last
    /// byte still blocks OAM, even though `active` becomes false during that same step.
    blocking: bool,
    /// Pipelined OAM write: DMA reads a byte from the bus in one M-cycle
    /// and writes it to OAM in the NEXT. This models the 1-cycle delay
    /// between bus read and OAM latch observed on hardware.
    pending_write: Option<(usize, u8)>,
    /// Bus conflict latch: when the CPU writes to the same bus as DMA,
    /// the write value appears on the data bus and DMA copies it instead
    /// of the source byte. Set by write_byte, consumed by step_oam_dma.
    pub(super) bus_conflict_value: Option<u8>,
    /// Last byte read from the DMA source, used for PPU bus-byte override.
    last_bus_byte: u8,
}

impl OamDma {
    fn new() -> Self {
        OamDma {
            active: false, source: 0, progress: 0, delay: 0,
            was_blocking: false, blocking: false,
            pending_write: None, bus_conflict_value: None,
            last_bus_byte: 0xFF,
        }
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
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Hdma {
    src: u16,
    dst: u16,
    /// Remaining blocks (each block = 16 bytes). 0 = inactive.
    blocks: u8,
    /// 0 = General Purpose DMA, 1 = H-Blank DMA
    mode: u8,
    active: bool,
    /// True while a DMA block is being transferred (prevents re-entrant hblank trigger)
    in_transfer: bool,
}

impl Hdma {
    fn new() -> Self {
        Hdma { src: 0, dst: 0x8000, blocks: 0, mode: 0, active: false, in_transfer: false }
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
    pub boot_rom_active: bool,

    /// Hardware model (DMG, CGB, etc.)
    pub(crate) model: GbModel,

    /// SGB command processor (only present for SGB/SGB2 models)
    pub sgb: Option<Sgb>,

    /// Undocumented CGB registers
    ff72: u8,
    ff73: u8,
    ff74: u8,
    ff75: u8,

    /// True when CGB hardware runs a DMG game (locks CGB-only IO registers)
    pub(crate) dmg_compat: bool,

    /// PPU T-cycle debt from mid-M-cycle glitch handling (e.g. tile_sel_glitch).
    /// Next tick_mcycle advances PPU by (4 - debt) T-cycles instead of 4.
    ppu_tick_debt: u32,

    /// Deferred PPU T-cycles from the lazy tick model.
    /// Each M-cycle ticks PPU immediately for the first half, deferring the
    /// second half. Deferred ticks are flushed before any PPU-state-sensitive
    /// read or write, or at the start of the next tick_mcycle.
    ppu_deferred: u32,

    /// Extra T-cycles consumed by GDMA/HDMA that the CPU must account for.
    /// Set during DMA transfers, read and cleared by CPU after each instruction.
    pub(crate) dma_halt_cycles: u32,

}

/// Simple LCG PRNG for RAM initialization.
fn ram_random(state: &mut u64) -> u8 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 56) as u8
}

/// Initialize WRAM with hardware-realistic patterns.
/// DMG: even 256-byte pages biased toward 0xFF (rand|rand), odd pages toward 0x00 (rand&rand).
/// CGB/AGB: random fill.
fn init_wram(model: GbModel) -> [[u8; 0x1000]; 8] {
    let mut wram = [[0u8; 0x1000]; 8];
    let mut rng: u64 = 0x5A6B7C8D9E0F1A2B;
    let bank_count = if model.is_cgb() { 8 } else { 2 };
    for bank in 0..bank_count {
        for i in 0..0x1000 {
            let byte = if model.is_cgb() {
                ram_random(&mut rng)
            } else {
                // DMG: alternating 256-byte pages
                let even_page = (i & 0x100) == 0;
                if even_page {
                    ram_random(&mut rng) | ram_random(&mut rng)
                } else {
                    ram_random(&mut rng) & ram_random(&mut rng)
                }
            };
            wram[bank][i] = byte;
        }
    }
    wram
}

/// Initialize HRAM with hardware-realistic patterns.
/// DMG: odd bytes biased toward 0xFF, even bytes toward 0x00.
/// CGB/AGB: random fill.
fn init_hram(model: GbModel) -> [u8; 0x7F] {
    let mut hram = [0u8; 0x7F];
    let mut rng: u64 = 0x5A6B7C8D9E0F1A2B;
    for i in 0..0x7F {
        hram[i] = if model.is_cgb() {
            ram_random(&mut rng)
        } else if (i & 1) != 0 {
            ram_random(&mut rng) | ram_random(&mut rng) | ram_random(&mut rng)
        } else {
            ram_random(&mut rng) & ram_random(&mut rng) & ram_random(&mut rng)
        };
    }
    hram
}

impl Bus {
    pub fn new(rom: std::sync::Arc<[u8]>, boot_rom: Option<Vec<u8>>, model: GbModel, clock: std::sync::Arc<dyn Clock>) -> Self {
        let boot_rom_active = boot_rom.is_some();

        let mut ppu = Ppu::new();
        ppu.cgb_mode = model.is_cgb();
        ppu.mgb_mode = model == GbModel::Mgb;
        ppu.sgb_mode = model.is_sgb();
        if model.is_sgb() {
            ppu.cgb_mode = false;
        }
        // Detect DMG game running on CGB hardware
        let cgb_flag = rom.get(0x0143).copied().unwrap_or(0);
        let is_cgb_game = cgb_flag == 0x80 || cgb_flag == 0xC0;
        let is_dmg_game = !is_cgb_game;

        if boot_rom_active {
            ppu.reset();
        } else {
            // Set post-boot PPU position (LY, dot, mode) for all models.
            // SGB/SGB2 timing depends on ROM header data; CGB/AGB differs for native vs compat.
            ppu.set_post_boot(model, is_cgb_game, &rom);
        }
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
            if model.is_sgb() {
                // SGB boot ROM finishes with P1=0x30 (both select lines deselected),
                // so P1 reads as 0xFF (no buttons visible).
                joypad.write(0x30);
            } else if model.is_cgb() {
                // CGB boot ROM finishes with P1 unselected (reads as $FF).
                joypad.write(0x30);
            } else {
                // DMG boot ROM writes 0x00, clearing both select bits.
                // This makes P1 read as 0xCF (both groups selected, no buttons pressed).
                joypad.write(0x00);
            }
        }

        // Compute timer before rom is moved into cartridge
        let timer = if boot_rom_active { Timer::reset(model) } else { Timer::post_boot(model, is_cgb_game, &rom) };

        let mut cart = make_cartridge(rom, clock);

        Bus {
            cart,
            ppu,
            timer,
            joypad,
            apu: if boot_rom_active { Apu::reset(model.cpu_clock_rate(), model.is_cgb()) } else { Apu::new(model.cpu_clock_rate(), model.is_cgb(), model.is_sgb()) },
            wram: init_wram(model),
            wram_bank: 1,
            hram: init_hram(model),
            if_: if boot_rom_active { 0x00 } else { 0xE1 },
            ie: 0x00,
            serial: Serial::new(model.is_cgb() && is_cgb_game),
            key1: 0x00,
            double_speed: false,
            oam_dma: OamDma::new(),
            hdma: Hdma::new(),
            boot_rom,
            boot_rom_active,
            model,
            sgb: if model.is_sgb() {
                let mut sgb = Sgb::new();
                if !boot_rom_active {
                    sgb.protocol_active = true;
                }
                Some(sgb)
            } else {
                None
            },
            ff72: 0,
            ff73: 0,
            ff74: 0,
            ff75: 0,
            dmg_compat: model.is_cgb() && is_dmg_game && !boot_rom_active,
            ppu_tick_debt: 0,
            ppu_deferred: 0,
            dma_halt_cycles: 0,
        }
    }

    pub fn has_boot_rom(&self) -> bool {
        self.boot_rom.is_some()
    }

    /// Create a snapshot of the bus state for rewind / save states.
    pub fn take_snapshot(&mut self) -> crate::snapshot::BusSnapshot {
        // Temporarily take sample_buf to avoid cloning it (it can be large)
        let saved_buf = std::mem::take(&mut self.apu.sample_buf);
        let apu_clone = self.apu.clone();
        self.apu.sample_buf = saved_buf;
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
            ff72: self.ff72,
            ff73: self.ff73,
            ff74: self.ff74,
            ff75: self.ff75,
            dmg_compat: self.dmg_compat,
            ppu_tick_debt: self.ppu_tick_debt,
            ppu_deferred: self.ppu_deferred,
            dma_halt_cycles: self.dma_halt_cycles,
        }
    }

    /// Restore the bus state from a snapshot.
    pub fn apply_snapshot(&mut self, s: &crate::snapshot::BusSnapshot) {
        self.ppu = s.ppu.clone();
        self.timer = s.timer.clone();
        self.joypad = s.joypad.clone();
        // Preserve the existing sample_buf allocation when restoring APU state
        let saved_buf = std::mem::take(&mut self.apu.sample_buf);
        self.apu = s.apu.clone();
        self.apu.sample_buf = saved_buf;
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
        self.ff72 = s.ff72;
        self.ff73 = s.ff73;
        self.ff74 = s.ff74;
        self.ff75 = s.ff75;
        self.dmg_compat = s.dmg_compat;
        self.ppu_tick_debt = s.ppu_tick_debt;
        self.ppu_deferred = s.ppu_deferred;
        self.dma_halt_cycles = s.dma_halt_cycles;
    }

    // ── Public accessors for Cpu ───────────────────────────────────────────────

    pub fn ie(&self) -> u8 { self.ie }
    pub fn if_reg(&mut self) -> u8 {
        self.flush_ppu_deferred();
        self.if_
    }
    pub fn if_mut(&mut self) -> &mut u8 { &mut self.if_ }

    // ── Memory read ───────────────────────────────────────────────────────────

    /// CPU memory read with DMA bus conflict handling.
    /// - DMG: OAM reads return 0xFF during active DMA transfer.
    /// - CGB: DMA blocks reads from the same bus as the source (including during delay).
    ///   Cart bus (ROM + SRAM) and WRAM bus are separate.
    pub fn read_byte(&mut self, addr: u16) -> u8 {
        // Flush all deferred PPU ticks so CPU sees correct PPU state
        // (mode bits, LY, VRAM/OAM accessibility, IF flags).
        self.flush_ppu_deferred();
        // OAM bus reads return 0xFF during active DMA (both DMG and CGB)
        // The entire $FE00-$FEFF range is on the OAM bus
        if self.oam_dma.is_blocking() && matches!(addr, 0xFE00..=0xFEFF) {
            return 0xFF;
        }
        if self.model.is_cgb() {
            // CGB: same-bus conflict after 2 M-cycle warm-up (delay + first copy at progress=0)
            if self.oam_dma.active && self.oam_dma.delay == 0 && self.oam_dma.progress > 0
                && self.oam_dma_same_bus(addr)
            {
                return self.oam_dma_conflict_byte();
            }
        } else {
            // DMG: during active DMA transfer (after warm-up), reads from the same bus
            // as the DMA source return the previous byte transferred.
            if self.oam_dma.active && self.oam_dma.delay == 0 && self.oam_dma.progress > 0
                && self.oam_dma_same_bus(addr)
            {
                return self.oam_dma_conflict_byte();
            }
        }
        self.read_byte_raw(addr)
    }

    /// Check if addr is on the same bus as the OAM DMA source.
    /// Addresses >= 0xFE00 (internal bus) are never in conflict.
    fn oam_dma_same_bus(&self, addr: u16) -> bool {
        if addr >= 0xFE00 {
            return false;
        }
        let src = self.oam_dma.source;
        if self.model.is_cgb() {
            // CGB has 3 buses: MAIN (cart ROM+SRAM), VRAM, RAM (WRAM)
            // WRAM reads conflict unless DMA source is VRAM
            if addr >= 0xC000 {
                return !matches!(src, 0x8000..=0x9FFF);
            }
            // Echo WRAM source (>= 0xE000) conflicts with everything except VRAM
            if src >= 0xE000 {
                return !matches!(addr, 0x8000..=0x9FFF);
            }
            // Default: same bus check
            match src {
                0x0000..=0x7FFF | 0xA000..=0xBFFF => matches!(addr, 0x0000..=0x7FFF | 0xA000..=0xBFFF),
                0x8000..=0x9FFF => matches!(addr, 0x8000..=0x9FFF),
                0xC000..=0xDFFF => matches!(addr, 0xC000..=0xDFFF),
                _ => false,
            }
        } else {
            // DMG: only VRAM is a separate bus; everything else is one shared bus
            match src {
                0x8000..=0x9FFF => matches!(addr, 0x8000..=0x9FFF),
                _ => !matches!(addr, 0x8000..=0x9FFF),
            }
        }
    }

    /// Returns the bus conflict byte during OAM DMA.
    /// - CGB: returns the byte at the current DMA source address.
    /// - DMG: returns the *previous* byte transferred (source + progress - 1),
    ///   i.e. reading from (dma_source + progress - 1).
    fn oam_dma_conflict_byte(&self) -> u8 {
        if !self.oam_dma.active || self.oam_dma.progress == 0 {
            return 0xFF;
        }
        // Both DMG and CGB return the last byte transferred (source + progress - 1)
        let mut src = self.oam_dma.source.wrapping_add(self.oam_dma.progress as u16 - 1);
        if !self.model.is_cgb() && src >= 0xFE00 {
            src -= 0x2000;
        }
        self.read_byte_raw(src)
    }

    /// Raw read bypassing DMA bus-conflict logic (for DMA/HDMA controllers).
    pub(super) fn read_byte_raw(&self, addr: u16) -> u8 {
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
                if !self.ppu.vram_accessible {
                    0xFF
                } else {
                    self.ppu.read_vram(addr)
                }
            }
            0xA000..=0xBFFF => self.cart.read_ram(addr),
            0xC000..=0xCFFF => self.wram[0][(addr - 0xC000) as usize],
            0xD000..=0xDFFF => self.wram[self.wram_bank][(addr - 0xD000) as usize],
            0xE000..=0xEFFF => self.wram[0][(addr - 0xE000) as usize], // echo
            0xF000..=0xFDFF => self.wram[self.wram_bank][(addr - 0xF000) as usize], // echo
            0xFE00..=0xFE9F => {
                if !self.ppu.oam_accessible { 0xFF } else { self.ppu.read_oam(addr) }
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

    pub fn read_word(&mut self, addr: u16) -> u16 {
        let lo = self.read_byte(addr) as u16;
        let hi = self.read_byte(addr.wrapping_add(1)) as u16;
        (hi << 8) | lo
    }

    // ── Memory write ──────────────────────────────────────────────────────────

    /// CPU memory write with DMA bus conflict handling.
    /// - DMG: OAM writes blocked during DMA; external bus writes are ignored.
    /// - CGB: writes to the same bus as DMA source are blocked + OAM.
    pub fn write_byte(&mut self, addr: u16, val: u8) {
        // Flush deferred PPU ticks so the write sees correct PPU state
        // (mode, accessibility). PPU register writes (0xFF40-0xFF6B) handle
        // flushing in write_byte_raw with conflict-specific timing.
        if !matches!(addr, 0xFF40..=0xFF6B) {
            self.flush_ppu_deferred();
        }
        // OAM bus writes blocked during active DMA (both DMG and CGB)
        // The entire $FE00-$FEFF range is on the OAM bus
        if self.oam_dma.is_blocking() && matches!(addr, 0xFE00..=0xFEFF) {
            return;
        }
        // Write during active DMA bus conflict: the CPU's write value appears
        // on the data bus. The DMA pipeline is holding a byte that was read
        // in the previous M-cycle — the CPU's write replaces that byte's
        // value before it gets flushed to OAM.
        if self.oam_dma.active && self.oam_dma.delay == 0 && self.oam_dma.progress > 0
            && self.oam_dma_same_bus(addr)
        {
            // Replace the pending pipeline byte (from previous step's read)
            // with the CPU's write value. This models the bus conflict where
            // the CPU's data appears on the shared bus during the DMA cycle.
            if let Some((idx, _)) = self.oam_dma.pending_write {
                self.oam_dma.pending_write = Some((idx, val));
            }
            return;
        }
        // DMG OAM bug: writes to OAM range during Mode 2 trigger corruption
        if !self.ppu.oam_accessible && addr >= 0xFE00 && addr < 0xFF00 {
            self.trigger_oam_bug_from_write(addr);
        }
        self.write_byte_raw(addr, val);
    }

    fn write_byte_raw(&mut self, addr: u16, val: u8) {
        match addr {
            0x0000..=0x7FFF => self.cart.write_rom(addr, val),
            0x8000..=0x9FFF => {
                if self.ppu.vram_write_accessible {
                    self.ppu.write_vram(addr, val);
                }
            }
            0xA000..=0xBFFF => self.cart.write_ram(addr, val),
            0xC000..=0xCFFF => self.wram[0][(addr - 0xC000) as usize] = val,
            0xD000..=0xDFFF => self.wram[self.wram_bank][(addr - 0xD000) as usize] = val,
            0xE000..=0xEFFF => self.wram[0][(addr - 0xE000) as usize] = val,
            0xF000..=0xFDFF => self.wram[self.wram_bank][(addr - 0xF000) as usize] = val,
            0xFE00..=0xFE9F => {
                if self.ppu.oam_write_accessible { self.ppu.write_oam(addr, val); }
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

    // ── Eager M-cycle tick methods (used by emulator mcycle loop) ─────────────

    /// Read, then tick. Matches old model exactly:
    /// 1. read_byte (which flushes deferred PPU internally)
    /// 2. tick_mcycle (compute blocking, tick timer/serial/APU, defer PPU, hdma, oam_dma)
    pub fn tick_read(&mut self, addr: u16) -> u8 {
        // Read first (read_byte handles flush_ppu_deferred internally)
        let val = self.read_byte(addr);

        // Then tick_mcycle equivalent
        self.oam_dma.blocking = self.oam_dma.compute_blocking();
        let bus_cycles = if self.double_speed { 2u32 } else { 4 };
        let debt = self.ppu_tick_debt;
        self.ppu_tick_debt = 0;
        let ppu_cycles = bus_cycles.saturating_sub(debt);
        // During DMA, tick PPU eagerly with bus byte so OAM scan sees
        // the DMA transfer byte instead of stored OAM values.
        if self.oam_dma.blocking {
            self.sync_ppu_dma_bus_byte();
            let flags = self.ppu.step(ppu_cycles);
            self.if_ |= flags;
        } else {
            self.ppu_deferred += ppu_cycles;
        }
        self.tick_split(4, bus_cycles, 0);
        self.check_hdma_hblank();
        self.step_oam_dma();
        val
    }

    /// Write, then tick. For PPU conflict registers, the write_io conflict
    /// handlers in io.rs handle the PPU timing internally (using ppu_deferred
    /// from the PREVIOUS tick). For non-conflict addresses, write_byte handles
    /// flush_ppu_deferred internally.
    pub fn tick_write(&mut self, addr: u16, val: u8) {
        // Write first (write_byte handles flush_ppu_deferred internally,
        // and write_io handles PPU conflict timing for PPU registers)
        self.write_byte(addr, val);

        // Then tick_mcycle equivalent
        self.oam_dma.blocking = self.oam_dma.compute_blocking();
        let bus_cycles = if self.double_speed { 2u32 } else { 4 };
        let debt = self.ppu_tick_debt;
        self.ppu_tick_debt = 0;
        let ppu_cycles = bus_cycles.saturating_sub(debt);
        if self.oam_dma.blocking {
            self.sync_ppu_dma_bus_byte();
            let flags = self.ppu.step(ppu_cycles);
            self.if_ |= flags;
        } else {
            self.ppu_deferred += ppu_cycles;
        }
        self.tick_split(4, bus_cycles, 0);
        self.check_hdma_hblank();
        self.step_oam_dma();
    }

    /// Tick one internal M-cycle (no bus access).
    /// Equivalent to tick_mcycle without any memory access.
    pub fn tick_internal(&mut self) {
        self.oam_dma.blocking = self.oam_dma.compute_blocking();
        let bus_cycles = if self.double_speed { 2u32 } else { 4 };
        let debt = self.ppu_tick_debt;
        self.ppu_tick_debt = 0;
        let ppu_cycles = bus_cycles.saturating_sub(debt);
        if self.oam_dma.blocking {
            self.sync_ppu_dma_bus_byte();
            let flags = self.ppu.step(ppu_cycles);
            self.if_ |= flags;
        } else {
            self.ppu_deferred += ppu_cycles;
        }
        self.tick_split(4, bus_cycles, 0);
        self.check_hdma_hblank();
        self.step_oam_dma();
    }

    /// Tick half an M-cycle (for HALT NOP split IF check).
    pub fn tick_half(&mut self) {
        self.oam_dma.blocking = self.oam_dma.compute_blocking();
        let bus_cycles = if self.double_speed { 1u32 } else { 2 };
        if self.oam_dma.blocking {
            self.sync_ppu_dma_bus_byte();
            let flags = self.ppu.step(bus_cycles);
            self.if_ |= flags;
        } else {
            self.ppu_deferred += bus_cycles;
        }
        self.tick_split(2, bus_cycles, 0);
    }

    /// Tick second half of M-cycle + post-processing (for HALT NOP).
    pub fn tick_half_post(&mut self) {
        let bus_cycles = if self.double_speed { 1u32 } else { 2 };
        if self.oam_dma.blocking {
            self.sync_ppu_dma_bus_byte();
            let flags = self.ppu.step(bus_cycles);
            self.if_ |= flags;
        } else {
            self.ppu_deferred += bus_cycles;
        }
        self.tick_split(2, bus_cycles, 0);
        self.check_hdma_hblank();
        self.step_oam_dma();
    }

    /// Check if this address is a PPU conflict register needing special write timing.
    /// Reserved for future eager PPU ticking model.
    #[allow(dead_code)]
    fn is_ppu_conflict_register(&self, addr: u16) -> bool {
        match addr {
            // CGB LCDC tile_sel_glitch
            0xFF40 if self.model.is_cgb() && !self.double_speed => true,
            // DMG LCDC complex glitch
            0xFF40 if !self.model.is_cgb() => true,
            // DMG SCY READ_NEW
            0xFF42 if !self.model.is_cgb() => true,
            // DMG/CGB-double SCX
            0xFF43 if !self.model.is_cgb() || self.double_speed => true,
            // DMG palette glitch
            0xFF47..=0xFF49 if !self.model.is_cgb() => true,
            // WY READ_NEW
            0xFF4A => true,
            // DMG WX
            0xFF4B if !self.model.is_cgb() => true,
            _ => false,
        }
    }

    /// Handle PPU timing around a write for PPU conflict registers.
    /// Ticks PPU with per-register split, then performs the write.
    /// `ppu_cycles` is the total PPU budget for this M-cycle (after debt).
    /// Reserved for future eager PPU ticking model.
    #[allow(dead_code)]
    fn tick_write_ppu_conflict(&mut self, addr: u16, val: u8, ppu_cycles: u32) {
        match addr {
            // DMG palette writes: 2T old, 1T (old|new) glitch, write, 1T new
            0xFF47..=0xFF49 if !self.model.is_cgb() && ppu_cycles >= 4 => {
                // 2T with old palette value
                let flags = self.ppu.step(2);
                self.if_ |= flags;
                // 1T with glitch palette (old | new)
                let old_val = match addr {
                    0xFF47 => self.ppu.bgp_rendering,
                    0xFF48 => self.ppu.obp0_rendering,
                    _ => self.ppu.obp1_rendering,
                };
                let glitch = old_val | val;
                match addr {
                    0xFF47 => self.ppu.bgp_rendering = glitch,
                    0xFF48 => self.ppu.obp0_rendering = glitch,
                    _ => self.ppu.obp1_rendering = glitch,
                }
                let flags = self.ppu.step(1);
                self.if_ |= flags;
                // Write real value, remaining 1T uses it
                self.ppu.write(addr, val);
                if self.ppu.if_flags != 0 { self.if_ |= self.ppu.if_flags; self.ppu.if_flags = 0; }
                let flags = self.ppu.step(1);
                self.if_ |= flags;
            }
            // DMG palette: not enough PPU budget for full conflict
            0xFF47..=0xFF49 if !self.model.is_cgb() => {
                if ppu_cycles > 0 {
                    let flags = self.ppu.step(ppu_cycles);
                    self.if_ |= flags;
                }
                self.ppu.write(addr, val);
                if self.ppu.if_flags != 0 { self.if_ |= self.ppu.if_flags; self.ppu.if_flags = 0; }
            }
            // DMG SCY: READ_NEW — (ppu_cycles-1)T old, write, 1T new
            0xFF42 if !self.model.is_cgb() => {
                if ppu_cycles > 1 {
                    let flags = self.ppu.step(ppu_cycles - 1);
                    self.if_ |= flags;
                }
                self.ppu.write(addr, val);
                if self.ppu.if_flags != 0 { self.if_ |= self.ppu.if_flags; self.ppu.if_flags = 0; }
                if ppu_cycles >= 1 {
                    let flags = self.ppu.step(1);
                    self.if_ |= flags;
                }
            }
            // DMG/CGB-double SCX: write takes effect 2T early
            0xFF43 if !self.model.is_cgb() || self.double_speed => {
                if ppu_cycles > 2 {
                    let flags = self.ppu.step(ppu_cycles - 2);
                    self.if_ |= flags;
                }
                self.ppu.write(addr, val);
                if self.ppu.if_flags != 0 { self.if_ |= self.ppu.if_flags; self.ppu.if_flags = 0; }
                let remaining = ppu_cycles.min(2);
                if remaining > 0 {
                    let flags = self.ppu.step(remaining);
                    self.if_ |= flags;
                }
            }
            // CGB LCDC: tile_sel_glitch when TILE_SEL transitions 1→0
            0xFF40 if self.model.is_cgb() && !self.double_speed => {
                if ppu_cycles > 0 {
                    let flags = self.ppu.step(ppu_cycles);
                    self.if_ |= flags;
                }
                let old_lcdc = self.ppu.lcdc;
                self.ppu.write(addr, val);
                if self.ppu.if_flags != 0 { self.if_ |= self.ppu.if_flags; self.ppu.if_flags = 0; }
                // TILE_SEL (bit 4) transition 1→0: 1T glitch window
                if (old_lcdc & 0x10) != 0 && (val & 0x10) == 0 {
                    self.ppu.tile_sel_glitch = true;
                    let flags = self.ppu.step(1);
                    self.if_ |= flags;
                    self.ppu.tile_sel_glitch = false;
                    self.ppu_tick_debt = 1; // borrow 1T from next M-cycle
                    if self.ppu.if_flags != 0 { self.if_ |= self.ppu.if_flags; self.ppu.if_flags = 0; }
                }
            }
            // DMG LCDC: complex glitch handler
            0xFF40 if !self.model.is_cgb() => {
                let in_mode3 = self.ppu.mode == 3;
                if in_mode3 && ppu_cycles >= 4 {
                    // OBJ_EN takes effect immediately when cleared
                    if (val & 0x02) == 0 {
                        self.ppu.lcdc &= !0x02;
                    }
                    // 2T old
                    let flags = self.ppu.step(2);
                    self.if_ |= flags;
                    let old_lcdc = self.ppu.lcdc;
                    // 1T glitch: old | (new & BG_EN)
                    let glitch = old_lcdc | (val & 0x01);
                    let saved_lcdc = self.ppu.lcdc;
                    self.ppu.lcdc = glitch;
                    let flags = self.ppu.step(1);
                    self.if_ |= flags;
                    self.ppu.lcdc = saved_lcdc;
                    // Window disable glitch
                    if (saved_lcdc & 0x20) != 0 && (val & 0x20) == 0
                        && self.ppu.fetcher_is_window()
                    {
                        self.ppu.disable_window_pixel_insertion_glitch = true;
                    }
                    // Write real value, remaining 1T
                    self.ppu.write(addr, val);
                    if self.ppu.if_flags != 0 { self.if_ |= self.ppu.if_flags; self.ppu.if_flags = 0; }
                    let flags = self.ppu.step(1);
                    self.if_ |= flags;
                } else {
                    if ppu_cycles > 0 {
                        let flags = self.ppu.step(ppu_cycles);
                        self.if_ |= flags;
                    }
                    self.ppu.write(addr, val);
                    if self.ppu.if_flags != 0 { self.if_ |= self.ppu.if_flags; self.ppu.if_flags = 0; }
                }
            }
            // WY: READ_NEW — write takes effect 1T before end of M-cycle
            0xFF4A => {
                if ppu_cycles > 1 {
                    let flags = self.ppu.step(ppu_cycles - 1);
                    self.if_ |= flags;
                }
                self.ppu.write(addr, val);
                if self.ppu.if_flags != 0 { self.if_ |= self.ppu.if_flags; self.ppu.if_flags = 0; }
                if ppu_cycles >= 1 {
                    let flags = self.ppu.step(1);
                    self.if_ |= flags;
                }
            }
            // DMG WX: READ_OLD + wx_just_changed flag for 1T after write
            0xFF4B if !self.model.is_cgb() => {
                if ppu_cycles > 0 {
                    let flags = self.ppu.step(ppu_cycles);
                    self.if_ |= flags;
                }
                self.ppu.write(addr, val);
                if self.ppu.if_flags != 0 { self.if_ |= self.ppu.if_flags; self.ppu.if_flags = 0; }
                // 1T with wx_just_changed to suppress window trigger
                self.ppu.wx_just_changed = true;
                let flags = self.ppu.step(1);
                self.if_ |= flags;
                self.ppu.wx_just_changed = false;
                self.ppu_tick_debt = 1; // compensate for the extra 1T
                if self.ppu.if_flags != 0 { self.if_ |= self.ppu.if_flags; self.ppu.if_flags = 0; }
            }
            // Fallback (should not be reached if is_ppu_conflict_register is correct)
            _ => {
                if ppu_cycles > 0 {
                    let flags = self.ppu.step(ppu_cycles);
                    self.if_ |= flags;
                }
                self.write_byte(addr, val);
            }
        }
    }

    // ── PPU deferred tick flush ───────────────────────────────────────────────

    /// Update the PPU's DMA bus byte state before stepping the PPU.
    /// During OAM DMA, the PPU reads whatever byte is on the OAM data bus
    /// (the current DMA transfer byte) instead of stored OAM values.
    fn sync_ppu_dma_bus_byte(&mut self) {
        self.ppu.dma_bus_byte = if self.oam_dma.is_blocking() {
            Some(self.oam_dma.last_bus_byte)
        } else {
            None
        };
    }

    /// Flush deferred PPU ticks from the lazy tick model.
    pub fn flush_ppu_deferred(&mut self) {
        if self.ppu_deferred > 0 {
            let d = self.ppu_deferred;
            self.ppu_deferred = 0;
            self.sync_ppu_dma_bus_byte();
            let flags = self.ppu.step(d);
            self.if_ |= flags;

            // Don't trigger HDMA during flush — defer to tick_mcycle.
            // Hardware detects mode 0 during tick (after CPU read), pauses
            // the CPU, and transfers during the pause. Data only becomes
            // visible at the next CPU read after the pause completes.
        }
    }

    // ── Speed switch (called by CPU on STOP) ──────────────────────────────────

    /// Check if a speed switch is armed (KEY1 bit 0 set).
    pub fn speed_switch_armed(&self) -> bool {
        self.model.is_cgb() && (self.key1 & 0x01 != 0)
    }

    /// Prepare for speed switch: reset DIV counter with falling edge detection.
    /// Called at the start of the STOP instruction before entering idle state.
    pub fn do_speed_switch_prepare(&mut self) {
        // Reset DIV counter (triggers falling-edge effects on timer/APU)
        let old_counter = self.timer.counter();
        if self.timer.mux_output() {
            self.timer.increment_tima_glitch();
        }
        self.timer.set_counter(0);
        // Detect DIV-driven APU falling edge from the reset
        let apu_bit: u16 = if self.double_speed { 0x2000 } else { 0x1000 };
        if old_counter & apu_bit != 0 {
            self.apu.div_event();
        }
        self.apu.set_div_counter(0);
    }

    /// Toggle the actual speed (called partway through speed switch idle).
    pub fn do_speed_toggle(&mut self) {
        self.double_speed = !self.double_speed;
        self.ppu.double_speed = self.double_speed;
        self.apu.set_double_speed(self.double_speed);
        if self.double_speed {
            self.key1 = 0x80;
        } else {
            self.key1 = 0x00;
        }
    }

    pub fn is_double_speed(&self) -> bool {
        self.double_speed
    }

    // ── Frame & SGB ──────────────────────────────────────────────────────────

    pub fn frame_ready(&self) -> bool {
        self.ppu.frame_ready
    }

    pub fn clear_frame_ready(&mut self) {
        self.ppu.frame_ready = false;
    }

    /// Apply SGB palettes to the frame buffer using the shade buffer.
    /// Call after PPU renders a complete frame.
    pub fn apply_sgb_palettes(&mut self) {
        if let Some(ref sgb) = self.sgb {
            if sgb.boot_pending {
                // Hide uninitialized game output until first SGB command arrives
                // (real hardware shows the SNES boot animation during this period)
                for p in self.ppu.frame_buffer.iter_mut() {
                    *p = 0x00000000;
                }
                return;
            }
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
        let _tile_data_base: usize = if lcdc & 0x10 != 0 { 0x0000 } else { 0x0800 };
        let tile_map_base: usize = if lcdc & 0x08 != 0 { 0x1C00 } else { 0x1800 };
        let signed_addr = lcdc & 0x10 == 0;

        let mut vram_data = [0u8; 4096];
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
                let buf = sgb.frozen_buffer.get_or_insert_with(|| vec![0u32; 160 * 144]);
                buf.copy_from_slice(&self.ppu.frame_buffer);
            }
        }
    }
}
