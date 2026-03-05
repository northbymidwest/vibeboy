/// SNES subsystem for SGB low-level emulation.
///
/// Runs the SGB BIOS ROM on a minimal 65C816 CPU to process SGB commands.
/// This enables games that use DATA_SND to hotpatch the BIOS (e.g. Kirby's
/// Dream Land 2) to get correct per-tile palette attributes.

pub mod bus;
pub mod cpu;
pub mod dma;
pub mod icd2;
pub mod ppu_regs;

use bus::SnesBus;
use cpu::Cpu65816;

/// Master cycles per SNES frame (~21.477 MHz / 60 fps ≈ 357,366).
const SNES_CYCLES_PER_FRAME: u64 = 357_366;

pub struct SnesSys {
    pub cpu: Cpu65816,
    pub bus: SnesBus,
}

impl SnesSys {
    /// Create a new SNES system with the given program ROM.
    pub fn new(rom: Vec<u8>) -> Self {
        let bus = SnesBus::new(rom);
        let mut cpu = Cpu65816::new();
        // Reset the CPU — reads reset vector from ROM
        let bus_ref = &bus;
        cpu.reset(&|addr| {
            // Read from bus for reset vector (can't use &mut during reset)
            let bank = ((addr >> 16) & 0xFF) as u8;
            let offset = (addr & 0xFFFF) as u16;
            let effective_bank = bank & 0x7F;
            if bank == 0x7E || bank == 0x7F {
                let wram_addr = ((bank as usize & 1) << 16) | offset as usize;
                return bus_ref.wram[wram_addr];
            }
            if offset >= 0x8000 && effective_bank <= 0x7D {
                let rom_addr = (effective_bank as usize) * 0x8000 + (offset as usize - 0x8000);
                return bus_ref.rom.get(rom_addr).copied().unwrap_or(0);
            }
            0
        });
        SnesSys { cpu, bus }
    }

    /// Run one SNES frame (~357,366 master cycles).
    /// Fires NMI at the start (VBlank) and runs the CPU until the cycle budget expires.
    pub fn run_frame(&mut self) {
        // Signal VBlank / NMI
        self.bus.set_nmi_flag();
        self.bus.hvbjoy = 0x81; // VBlank active

        if self.bus.nmi_enabled() {
            self.cpu.set_nmi(true);
        }

        let target = self.cpu.cycles + SNES_CYCLES_PER_FRAME;

        // Run CPU
        while self.cpu.cycles < target {
            if self.cpu.waiting {
                // WAI: skip to target (NMI already handled above)
                self.cpu.cycles = target;
                break;
            }
            if self.cpu.stopped {
                self.cpu.cycles = target;
                break;
            }
            self.step_cpu();
        }

        // Clear NMI line after frame
        self.cpu.set_nmi(false);
        self.bus.hvbjoy = 0x01; // VBlank over, not in HBlank
    }

    /// Step the CPU one instruction using raw pointer bus access.
    fn step_cpu(&mut self) {
        let bus_ptr = &mut self.bus as *mut SnesBus;
        let read_fn = move |addr: u32| -> u8 {
            unsafe { (*bus_ptr).read(addr) }
        };
        let mut write_fn = move |addr: u32, val: u8| {
            unsafe { (*bus_ptr).write(addr, val) }
        };
        self.cpu.step(&read_fn, &mut write_fn);
    }

    /// Feed a command packet from the GB to the SNES.
    pub fn feed_packet(&mut self, data: &[u8; 16]) {
        self.bus.icd2.feed_packet(data);
    }

    /// Feed scanline shade data from the GB PPU.
    pub fn feed_scanlines(&mut self, shade_buf: &[u8]) {
        self.bus.icd2.feed_scanlines(shade_buf);
    }

    /// Extract the 4 SGB palettes from SNES CGRAM.
    /// SGB BIOS stores palettes at CGRAM offsets 0-31 (4 palettes × 4 colors × 2 bytes).
    pub fn extract_palettes(&self) -> [[u16; 4]; 4] {
        let mut pals = [[0u16; 4]; 4];
        for pal in 0..4 {
            for col in 0..4 {
                let off = (pal * 4 + col) * 2;
                pals[pal][col] = self.bus.ppu.cgram[off] as u16
                    | ((self.bus.ppu.cgram[off + 1] as u16) << 8);
            }
        }
        pals
    }

    /// Extract border tile data from SNES VRAM.
    /// The SGB BIOS stores border tiles at a known VRAM location.
    /// Returns (tiles, tilemap, palettes).
    pub fn extract_border(&self) -> (Vec<u8>, Vec<u16>, [[u16; 16]; 4]) {
        // Border tile data: typically at VRAM $4000-$5FFF (word addresses $2000-$2FFF)
        // = byte offsets $4000-$5FFF in our VRAM array
        // 256 tiles × 32 bytes = 8192 bytes
        let tile_base = 0x4000usize; // Byte offset in VRAM
        let mut tiles = vec![0u8; 256 * 32];
        let len = std::cmp::min(tiles.len(), self.bus.ppu.vram.len() - tile_base);
        tiles[..len].copy_from_slice(&self.bus.ppu.vram[tile_base..tile_base + len]);

        // Border tilemap: typically at VRAM $3000-$3FFF (word $1800-$1FFF)
        // = byte offsets $3000-$37FF (32×28×2 = 1792 bytes)
        let map_base = 0x3000usize;
        let mut tilemap = vec![0u16; 32 * 28];
        for i in 0..896 {
            let off = map_base + i * 2;
            if off + 1 < self.bus.ppu.vram.len() {
                tilemap[i] = self.bus.ppu.vram[off] as u16
                    | ((self.bus.ppu.vram[off + 1] as u16) << 8);
            }
        }

        // Border palettes: CGRAM offsets 32-95 (4 palettes × 16 colors × 2 bytes)
        let mut palettes = [[0u16; 16]; 4];
        for pal in 0..4 {
            for col in 0..16 {
                let off = (32 + pal * 16 + col) * 2;
                if off + 1 < self.bus.ppu.cgram.len() {
                    palettes[pal][col] = self.bus.ppu.cgram[off] as u16
                        | ((self.bus.ppu.cgram[off + 1] as u16) << 8);
                }
            }
        }

        (tiles, tilemap, palettes)
    }

    /// Extract the attribute map from SNES WRAM.
    /// The SGB BIOS maintains a 20×18 attribute map in WRAM.
    /// The exact address depends on the BIOS version; we search for it
    /// or use a known offset.
    pub fn extract_attr_map(&self) -> Option<[[u8; 20]; 18]> {
        // The SGB BIOS stores the attribute map at a known WRAM address.
        // For SGB1 BIOS: $7F:0000 area or $00:0800 area
        // For SGB2 BIOS: similar location
        // We'll read from the BG1 tilemap in SNES VRAM to extract per-tile palettes.

        // The BIOS renders the GB screen as BG1. Each BG tile entry has palette bits
        // in bits 10-12 of the tilemap word. The BG1 tilemap base is in bg_sc[0].
        let bg1_sc = self.bus.ppu.bg_sc[0];
        let tilemap_base = ((bg1_sc as usize & 0xFC) >> 2) * 0x800; // Word address → byte offset is ×2
        let tilemap_byte_base = tilemap_base * 2;

        // BG1 is 32×32 tiles. The GB screen occupies 20×18 tiles starting from top-left.
        let mut attr_map = [[0u8; 20]; 18];
        for ty in 0..18 {
            for tx in 0..20 {
                let map_idx = ty * 32 + tx;
                let byte_off = tilemap_byte_base + map_idx * 2;
                if byte_off + 1 < self.bus.ppu.vram.len() {
                    let entry = self.bus.ppu.vram[byte_off] as u16
                        | ((self.bus.ppu.vram[byte_off + 1] as u16) << 8);
                    // Palette is bits 10-12
                    attr_map[ty][tx] = ((entry >> 10) & 0x07) as u8;
                }
            }
        }
        Some(attr_map)
    }
}
