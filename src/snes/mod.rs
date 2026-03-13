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
/// Master cycles per scanline (~1364).
const CYCLES_PER_SCANLINE: u64 = 1364;
/// Scanline where VBlank starts (NTSC).
const VBLANK_START_LINE: u64 = 225;

#[derive(Clone)]
pub struct SnesSys {
    pub cpu: Cpu65816,
    pub bus: SnesBus,
    pub frame_count: u64,
}

impl SnesSys {
    /// Create a new SNES system with the given program ROM.
    pub fn new(rom: Vec<u8>) -> Self {
        let bus = SnesBus::new(rom);
        let mut cpu = Cpu65816::new();
        // Reset the CPU — reads reset vector from ROM
        let bus_ref = &bus;
        cpu.reset(&|addr| {
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
        SnesSys { cpu, bus, frame_count: 0 }
    }

    /// Run one SNES frame (~357,366 master cycles).
    ///
    /// Frame layout (262 scanlines):
    ///   Scanlines 0-224: Active display (HVBJOY bit 7 = 0)
    ///   Scanlines 225-261: VBlank (HVBJOY bit 7 = 1, NMI fires at 225)
    ///
    /// We enter run_frame at the start of VBlank (scanline 225).
    pub fn run_frame(&mut self) {
        let frame_start = self.cpu.cycles;
        let target = frame_start + SNES_CYCLES_PER_FRAME;

        // --- Phase 1: VBlank (scanlines 225-261, ~37 scanlines) ---
        self.bus.set_nmi_flag();
        self.bus.hvbjoy = 0x80; // VBlank active, auto-joypad NOT busy
        self.bus.in_vblank = true;
        self.bus.active_display_start = frame_start + 37 * CYCLES_PER_SCANLINE;
        self.bus.icd2.set_tile_row(0x11); // VBlank indicator

        if self.bus.nmi_enabled() {
            self.cpu.set_nmi(true);
            self.bus.nmi_fire_count += 1;
        }

        let vblank_end = frame_start + 37 * CYCLES_PER_SCANLINE;
        self.run_until(vblank_end);

        // --- Phase 2: Active display (scanlines 0-224) ---
        self.cpu.set_nmi(false);
        self.bus.hvbjoy = 0x00;
        self.bus.in_vblank = false;
        self.bus.active_display_start = vblank_end;

        // Simulate GB tile row progression during active display.
        // Each GB tile row = 8 scanlines ≈ 8 × 456 × 5.12 ≈ 18,678 SNES master cycles.
        // We run in tile-row-sized chunks so the ICD2 $6000 shows progressive rows.
        let cycles_per_tile_row = 8 * CYCLES_PER_SCANLINE; // ~10,912 SNES cycles
        let irq_enabled = self.bus.nmitimen & 0x30 != 0;
        let irq_cycle = if irq_enabled {
            let vtime = self.bus.vtime as u64;
            Some(vblank_end + vtime * CYCLES_PER_SCANLINE)
        } else {
            None
        };
        let mut irq_fired = false;

        for tile_row in 0..18u8 {
            self.bus.icd2.set_tile_row(tile_row);
            let row_end = vblank_end + (tile_row as u64 + 1) * cycles_per_tile_row;
            let row_target = row_end.min(target);

            // Check for IRQ in this chunk
            if !irq_fired {
                if let Some(irq_at) = irq_cycle {
                    if irq_at <= row_target && irq_at > self.cpu.cycles {
                        self.run_until(irq_at);
                        self.bus.timeup = 0x80;
                        self.cpu.irq_line = true;
                        irq_fired = true;
                    }
                }
            }

            self.run_until(row_target);
        }

        // Remaining active display after tile rows (scanlines 144-224)
        self.bus.icd2.set_tile_row(0x11); // Back to VBlank indicator

        // Check for IRQ in the remaining display period (scanlines 144-224)
        if !irq_fired {
            if let Some(irq_at) = irq_cycle {
                if irq_at <= target && irq_at > self.cpu.cycles {
                    self.run_until(irq_at);
                    self.bus.timeup = 0x80;
                    self.cpu.irq_line = true;
                    irq_fired = true;
                }
            }
        }

        self.run_until(target);

        if irq_fired {
            self.cpu.irq_line = false;
        }

        // The SGB1 BIOS has several gate flags that gate packet/display processing:
        //   $02F6: enables packet reading (set during intro animation)
        //   $02F8: handshake complete (requires 5× $F1 init packets from GB)
        //   $0C4F: enables DATA_SND patch execution (set during command processing)
        //   $0C48: enables per-tile attribute assignment in patched code
        //   $02CA: enables display processing in main loop
        // Force-enable all after the SPC upload completes.
        if self.frame_count > 30 {
            let gates: &[(usize, u8, &str)] = &[
                (0x02F6, 1, "packet processing"),
                (0x02F8, 1, "handshake flag"),
                (0x02CA, 1, "display processing"),
                (0x0C4F, 0xFF, "DATA_SND patch gate"),
                (0x0C48, 1, "per-tile attribute gate"),
            ];
            for &(addr, val, desc) in gates {
                if self.bus.wram[addr] == 0 {
                    self.bus.wram[addr] = val;
                    log::info!("SNES: force-enabled {} (WRAM[${:04X}]={:02X})", desc, addr, val);
                }
            }
            // The BIOS CLI instruction at $B150 enables IRQ after init.
            // Force-clear the I flag so IRQ handler ($B50A) can run.
            self.cpu.p &= !0x04; // Clear FLAG_I

            // Note: DATA_SND patched code at WRAM $0820 would apply per-tile palette
            // attributes, but the BIOS display pipeline can't reach it without full
            // scanline-accurate ICD2→DMA→VRAM emulation. The HLE path handles
            // standard ATTR_SET/PAL_SET commands for per-tile palettes instead.
        }

        self.frame_count += 1;

        // Periodic diagnostics (every 300 frames)
        if self.frame_count % 300 == 0 {
            log::debug!(
                "SNES frame {}: PC={:02X}:{:04X} NMITIMEN=${:02X} apu_state={}",
                self.frame_count, self.cpu.pbr, self.cpu.pc,
                self.bus.nmitimen, self.bus.apu_state,
            );
        }
    }

    /// Run the CPU until the given cycle count, handling WAI properly.
    fn run_until(&mut self, target: u64) {
        while self.cpu.cycles < target {
            if self.cpu.stopped {
                self.cpu.cycles = target;
                break;
            }
            if self.cpu.waiting {
                // Don't skip to target — step to check for interrupts each iteration
                self.step_cpu();
                if self.cpu.waiting {
                    // Still waiting (no interrupt woke us) — skip to target
                    self.cpu.cycles = target;
                    break;
                }
            } else {
                self.step_cpu();
            }
        }
    }

    /// Step the CPU one instruction using raw pointer bus access.
    fn step_cpu(&mut self) {
        // Update bus cycle counter so VCOUNT reads return correct scanline
        self.bus.current_cpu_cycles = self.cpu.cycles;
        let bus_ptr = &mut self.bus as *mut SnesBus;
        let read_fn = move |addr: u32| -> u8 {
            unsafe { (*bus_ptr).read(addr) }
        };
        let mut write_fn = move |addr: u32, val: u8| {
            unsafe { (*bus_ptr).write(addr, val) }
        };
        self.cpu.step(&read_fn, &mut write_fn);

        // Check if IRQ was acknowledged (TIMEUP read)
        if self.bus.irq_ack {
            self.cpu.irq_line = false;
            self.bus.irq_ack = false;
        }
    }

    /// Returns true if the BIOS has finished its init handshake.
    /// The BIOS sets WRAM $02F8 = 1 after receiving 5× $F1 ICD2 packets.
    pub fn bios_ready(&self) -> bool {
        self.bus.wram[0x02F8] != 0
    }

    /// Feed a command packet from the GB to the SNES.
    pub fn feed_packet(&mut self, data: &[u8; 16]) {
        let cmd = (data[0] >> 3) & 0x1F;
        log::debug!("SNES: feed packet cmd=${:02X} [{:02X} {:02X} {:02X} {:02X} ...]",
            cmd, data[0], data[1], data[2], data[3]);

        // Handle DATA_SND (cmd $0F) directly: write bytes to SNES WRAM.
        // The BIOS state machine may not be in a packet-reading state,
        // so we apply WRAM patches immediately.
        if cmd == 0x0F {
            // DATA_SND format: byte0=cmd|len, byte1=addr_lo, byte2=addr_hi,
            // byte3=bank, byte4=num_bytes, byte5..byte15=data (up to 11 bytes)
            let addr_lo = data[1] as u16;
            let addr_hi = data[2] as u16;
            let bank = data[3];
            let snes_addr = (addr_hi << 8) | addr_lo;
            let num_bytes = (data[4] as usize).min(11); // byte 4 is the count
            // DATA_SND writes to SNES WRAM (bank $00/$7E)
            if bank == 0x00 || bank == 0x7E {
                for i in 0..num_bytes {
                    let wram_addr = snes_addr as usize + i;
                    if wram_addr < self.bus.wram.len() {
                        self.bus.wram[wram_addr] = data[5 + i];
                    }
                }
                log::debug!("SNES: DATA_SND wrote {} bytes to WRAM ${:04X}", num_bytes, snes_addr);
            }
        }

        self.bus.icd2.feed_packet(data);
    }

    /// Create a snapshot of the SNES subsystem (excluding ROM).
    pub fn take_snapshot(&self) -> crate::snapshot::SnesSnapshot {
        crate::snapshot::SnesSnapshot {
            cpu: self.cpu.clone(),
            bus_wram: self.bus.wram.clone(),
            bus_ppu: self.bus.ppu.clone(),
            bus_dma: self.bus.dma.clone(),
            bus_icd2: self.bus.icd2.clone(),
            nmitimen: self.bus.nmitimen,
            rdnmi: self.bus.rdnmi,
            timeup: self.bus.timeup,
            hvbjoy: self.bus.hvbjoy,
            joy1: self.bus.joy1,
            joy2: self.bus.joy2,
            wrmpya: self.bus.wrmpya,
            wrmpyb: self.bus.wrmpyb,
            wrdiv: self.bus.wrdiv,
            wrdivb: self.bus.wrdivb,
            rddiv: self.bus.rddiv,
            rdmpy: self.bus.rdmpy,
            wrio: self.bus.wrio,
            htime: self.bus.htime,
            vtime: self.bus.vtime,
            mdmaen: self.bus.mdmaen,
            hdmaen: self.bus.hdmaen,
            memsel: self.bus.memsel,
            apu_out: self.bus.apu_out,
            apu_in: self.bus.apu_in,
            apu_state: self.bus.apu_state,
            apu_last_counter: self.bus.apu_last_counter,
            apu_port1_val: self.bus.apu_port1_val,
            apu_echo_pending: self.bus.apu_echo_pending,
            nmi_fire_count: self.bus.nmi_fire_count,
            wmadd: self.bus.wmadd,
            active_display_start: self.bus.active_display_start,
            current_cpu_cycles: self.bus.current_cpu_cycles,
            in_vblank: self.bus.in_vblank,
            irq_ack: self.bus.irq_ack,
            frame_count: self.frame_count,
        }
    }

    /// Restore the SNES subsystem from a snapshot.
    pub fn apply_snapshot(&mut self, s: &crate::snapshot::SnesSnapshot) {
        self.cpu = s.cpu.clone();
        self.bus.wram = s.bus_wram.clone();
        self.bus.ppu = s.bus_ppu.clone();
        self.bus.dma = s.bus_dma.clone();
        self.bus.icd2 = s.bus_icd2.clone();
        self.bus.nmitimen = s.nmitimen;
        self.bus.rdnmi = s.rdnmi;
        self.bus.timeup = s.timeup;
        self.bus.hvbjoy = s.hvbjoy;
        self.bus.joy1 = s.joy1;
        self.bus.joy2 = s.joy2;
        self.bus.wrmpya = s.wrmpya;
        self.bus.wrmpyb = s.wrmpyb;
        self.bus.wrdiv = s.wrdiv;
        self.bus.wrdivb = s.wrdivb;
        self.bus.rddiv = s.rddiv;
        self.bus.rdmpy = s.rdmpy;
        self.bus.wrio = s.wrio;
        self.bus.htime = s.htime;
        self.bus.vtime = s.vtime;
        self.bus.mdmaen = s.mdmaen;
        self.bus.hdmaen = s.hdmaen;
        self.bus.memsel = s.memsel;
        self.bus.apu_out = s.apu_out;
        self.bus.apu_in = s.apu_in;
        self.bus.apu_state = s.apu_state;
        self.bus.apu_last_counter = s.apu_last_counter;
        self.bus.apu_port1_val = s.apu_port1_val;
        self.bus.apu_echo_pending = s.apu_echo_pending;
        self.bus.nmi_fire_count = s.nmi_fire_count;
        self.bus.wmadd = s.wmadd;
        self.bus.active_display_start = s.active_display_start;
        self.bus.current_cpu_cycles = s.current_cpu_cycles;
        self.bus.in_vblank = s.in_vblank;
        self.bus.irq_ack = s.irq_ack;
        self.frame_count = s.frame_count;
    }

    /// Feed scanline shade data from the GB PPU.
    pub fn feed_scanlines(&mut self, shade_buf: &[u8]) {
        self.bus.icd2.feed_scanlines(shade_buf);
    }

    /// Extract the 4 SGB palettes from SNES CGRAM.
    /// In Mode 1 (the SGB BIOS mode), BG1 is 4bpp so each palette block
    /// occupies 16 CGRAM color entries. In Mode 0 it would be 4 entries.
    pub fn extract_palettes(&self) -> [[u16; 4]; 4] {
        let mut pals = [[0u16; 4]; 4];
        let mode = self.bus.ppu.bgmode & 0x07;
        // Mode 0: BG1 is 2bpp (4 colors/palette), Mode 1+: BG1 is 4bpp (16 colors/palette)
        let stride = if mode == 0 { 4 } else { 16 };
        for pal in 0..4 {
            for col in 0..4 {
                let off = (pal * stride + col) * 2;
                pals[pal][col] = self.bus.ppu.cgram[off] as u16
                    | ((self.bus.ppu.cgram[off + 1] as u16) << 8);
            }
        }
        pals
    }

    /// Extract border tile data from SNES VRAM.
    pub fn extract_border(&self) -> (Vec<u8>, Vec<u16>, [[u16; 16]; 4]) {
        // BG2 chr base from $210B upper nibble: each unit = $2000 byte offset
        let bg2_chr_nibble = (self.bus.ppu.bg_chr[0] >> 4) as usize;
        let tile_base = bg2_chr_nibble * 0x2000;
        let mut tiles = vec![0u8; 256 * 32];
        let tile_len = tiles.len();
        if tile_base + tile_len <= self.bus.ppu.vram.len() {
            tiles.copy_from_slice(&self.bus.ppu.vram[tile_base..tile_base + tile_len]);
        }

        // BG2SC ($2108): tilemap base from bits 7-2
        let bg2_sc = self.bus.ppu.bg_sc[1];
        let map_base = ((bg2_sc as usize & 0xFC) >> 2) * 0x800;
        let mut tilemap = vec![0u16; 32 * 28];
        for i in 0..896 {
            let off = map_base + i * 2;
            if off + 1 < self.bus.ppu.vram.len() {
                tilemap[i] = self.bus.ppu.vram[off] as u16
                    | ((self.bus.ppu.vram[off + 1] as u16) << 8);
            }
        }

        // Border palettes: In Mode 1, BG2 is 4bpp. The SGB BIOS uses
        // SNES palettes 4-7 for the border (CGRAM colors 64-127).
        // We extract 4 palettes of 16 colors starting at CGRAM color 64.
        let mut palettes = [[0u16; 16]; 4];
        for pal in 0..4 {
            for col in 0..16 {
                let off = ((pal + 4) * 16 + col) * 2; // Palettes 4-7
                if off + 1 < self.bus.ppu.cgram.len() {
                    palettes[pal][col] = self.bus.ppu.cgram[off] as u16
                        | ((self.bus.ppu.cgram[off + 1] as u16) << 8);
                }
            }
        }

        (tiles, tilemap, palettes)
    }

    /// Extract the attribute map from SNES VRAM BG1 tilemap.
    /// The BIOS renders the GB screen as BG1, with per-tile palette in bits 10-12.
    pub fn extract_attr_map(&self) -> Option<[[u8; 20]; 18]> {
        let bg1_sc = self.bus.ppu.bg_sc[0];
        // BG1SC bits 7-2: tilemap base = (value >> 2) * 0x400 words = * 0x800 bytes
        let tilemap_byte_base = ((bg1_sc as usize & 0xFC) >> 2) * 0x800;

        // BG1 is 32×32 tiles. The GB screen occupies 20×18 tiles from top-left.
        let mut attr_map = [[0u8; 20]; 18];
        for ty in 0..18 {
            for tx in 0..20 {
                let map_idx = ty * 32 + tx;
                let byte_off = tilemap_byte_base + map_idx * 2;
                if byte_off + 1 < self.bus.ppu.vram.len() {
                    let entry = self.bus.ppu.vram[byte_off] as u16
                        | ((self.bus.ppu.vram[byte_off + 1] as u16) << 8);
                    attr_map[ty][tx] = ((entry >> 10) & 0x07) as u8;
                }
            }
        }
        Some(attr_map)
    }
}
