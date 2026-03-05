use crate::bus::Bus;
use crate::cpu::{Cpu, Registers};
use crate::joypad::{
    BTN_A, BTN_B, BTN_DOWN, BTN_LEFT, BTN_RIGHT, BTN_SELECT, BTN_START, BTN_UP,
};
use crate::model::GbModel;
use crate::sgb::Sgb;
use crate::snes::SnesSys;
use std::path::Path;

/// T-cycles per frame at normal speed (70224 = 456 × 154).
pub const CYCLES_PER_FRAME: u32 = 70_224;

pub struct Emulator {
    pub cpu: Cpu,
    pub bus: Bus,
    model: GbModel,
    /// Cached border buffer (256×224) for SGB
    border_buffer: Vec<u32>,
    /// Composited output buffer (256×224) for SGB
    sgb_output: Vec<u32>,
    /// SNES subsystem for SGB LLE (None = HLE fallback)
    snes: Option<SnesSys>,
}

impl Emulator {
    /// Create a new emulator. If `boot_rom` is Some, the CPU starts at PC=0x0000
    /// with hardware reset registers and executes the boot ROM. Otherwise the CPU
    /// starts at PC=0x0100 with post-boot register values for the given model.
    pub fn new(
        rom: Vec<u8>,
        boot_rom: Option<Vec<u8>>,
        rom_path: Option<&Path>,
        model: GbModel,
        snes_rom: Option<Vec<u8>>,
    ) -> Self {
        let has_boot = boot_rom.is_some();
        let mut cpu = Cpu::new();
        if has_boot {
            cpu.regs = Registers::reset();
        } else {
            cpu.regs = Registers::post_boot(model);
        }
        let mut bus = Bus::new(rom, boot_rom, rom_path, model);

        // Enable LLE mode if we have a SNES program ROM
        let snes = if model.is_sgb() && snes_rom.is_some() {
            if let Some(ref mut sgb) = bus.sgb {
                sgb.lle_mode = true;
            }
            let snes_sys = SnesSys::new(snes_rom.unwrap());
            log::info!("SGB LLE: SNES CPU active (PC=${:04X})", snes_sys.cpu.pc);
            Some(snes_sys)
        } else {
            None
        };

        Emulator {
            cpu,
            bus,
            model,
            border_buffer: vec![0u32; 256 * 224],
            sgb_output: vec![0u32; 256 * 224],
            snes,
        }
    }

    /// Persist battery-backed cartridge RAM to disk.
    pub fn save(&self) {
        self.bus.save_to_disk();
    }

    /// Run until one full frame has been rendered (VBlank).
    pub fn step_frame(&mut self) {
        self.bus.clear_frame_ready();
        while !self.bus.frame_ready() {
            self.step();
        }

        // SGB post-processing
        if self.model.is_sgb() {
            if self.snes.is_some() {
                self.step_snes_frame();
            } else {
                self.bus.apply_sgb_palettes();
                self.bus.check_sgb_transfer();
                self.bus.capture_sgb_freeze();
            }
        }
    }

    /// Run the SNES subsystem for one frame (LLE mode).
    fn step_snes_frame(&mut self) {
        let snes = self.snes.as_mut().unwrap();

        // 1. Feed any pending SGB packet to SNES
        if let Some(ref sgb) = self.bus.sgb {
            if let Some(pkt) = sgb.pending_packet {
                snes.feed_packet(&pkt);
            }
        }
        if let Some(ref mut sgb) = self.bus.sgb {
            sgb.pending_packet = None;
        }

        // 2. Feed shade buffer (GB LCD output) to SNES ICD2
        snes.feed_scanlines(&self.bus.ppu.shade_buffer);

        // 3. Run SNES CPU for one frame
        snes.run_frame();

        // 4. Extract palettes from SNES CGRAM and update SGB state
        let lle_palettes = snes.extract_palettes();
        // Check if SNES has non-zero palettes before overwriting
        let has_pal_data = lle_palettes.iter().any(|p| p.iter().any(|&c| c != 0));
        if has_pal_data {
            if let Some(ref mut sgb) = self.bus.sgb {
                sgb.palettes = lle_palettes;
            }
        }

        // 5. Extract attribute map from SNES VRAM tilemap
        if let Some(attr_map) = snes.extract_attr_map() {
            // Only update if there's variation (not all zeros)
            let has_variation = attr_map.iter().any(|row| row.iter().any(|&v| v != 0));
            if has_variation {
                if let Some(ref mut sgb) = self.bus.sgb {
                    sgb.attr_map = attr_map;
                }
            }
        }

        // 6. Extract border data from SNES VRAM/CGRAM
        let (tiles, tilemap, border_pals) = snes.extract_border();
        let has_border = tiles.iter().any(|&b| b != 0);
        if has_border {
            if let Some(ref mut sgb) = self.bus.sgb {
                let len = std::cmp::min(tiles.len(), sgb.border_tiles.len());
                sgb.border_tiles[..len].copy_from_slice(&tiles[..len]);
                let mlen = std::cmp::min(tilemap.len(), sgb.border_map.len());
                sgb.border_map[..mlen].copy_from_slice(&tilemap[..mlen]);
                sgb.border_palettes = border_pals;
                sgb.border_dirty = true;
            }
        }

        // 7. Apply palettes and handle masking (same as HLE path)
        self.bus.apply_sgb_palettes();
        // Also handle VRAM transfers for compatibility
        self.bus.check_sgb_transfer();
        self.bus.capture_sgb_freeze();
    }

    /// Execute one CPU instruction and advance bus components.
    fn step(&mut self) {
        self.cpu.step(&mut self.bus);
        // Ticking is now done inline by CPU during each M-cycle
    }

    /// Run until the Mooneye LD B,B breakpoint (opcode 0x40 at PC) or
    /// `max_frames` frames elapse. Returns Some((b,c,d,e,h,l)) on breakpoint.
    pub fn run_until_breakpoint(&mut self, max_frames: u32) -> Option<[u8; 6]> {
        let cycles_per_frame = 70_224u32;
        let mut cycles = 0u32;
        let limit = cycles_per_frame * max_frames;
        loop {
            // Check for Mooneye LD B,B breakpoint before executing
            if self.bus.read_byte(self.cpu.regs.pc) == 0x40 {
                let r = &self.cpu.regs;
                return Some([r.b, r.c, r.d, r.e, r.h, r.l]);
            }
            self.cpu.step(&mut self.bus);
            cycles += 4; // approximate; good enough for frame counting
            if cycles >= limit {
                return None;
            }
        }
    }

    /// Run until serial output contains "Passed" or "Failed", or the Blargg
    /// done loop (JR -2) is hit, or max_frames elapse.
    /// Returns the serial output as a string.
    pub fn run_until_serial_result(&mut self, max_frames: u32) -> String {
        let cycles_per_frame = 70_224u32;
        let mut cycles = 0u32;
        let limit = cycles_per_frame * max_frames;
        loop {
            let pc = self.cpu.regs.pc;
            self.cpu.step(&mut self.bus);
            cycles += 4;

            // Detect Blargg done loop: JR -2 (opcode 18 FE)
            if self.bus.read_byte(pc) == 0x18 && self.bus.read_byte(pc.wrapping_add(1)) == 0xFE {
                let result = self.bus.read_byte(0xA000);
                if result == 0 {
                    return "Passed".to_string();
                } else {
                    return format!("Failed #{}", result);
                }
            }

            // Check serial output periodically
            if cycles % 1024 == 0 || cycles >= limit {
                let output = String::from_utf8_lossy(&self.bus.serial_output);
                if output.contains("Passed") || output.contains("Failed") || cycles >= limit {
                    return output.into_owned();
                }
            }
        }
    }

    /// Return the current frame buffer (160 × 144 pixels, 0x00RRGGBB).
    pub fn frame_buffer(&self) -> &[u32] {
        // For SGB with freeze mask, return frozen buffer
        if let Some(ref sgb) = self.bus.sgb {
            if sgb.mask_mode == 1 {
                return &sgb.frozen_buffer;
            }
        }
        self.bus.ppu.frame_buffer()
    }

    pub fn is_sgb(&self) -> bool {
        self.model.is_sgb()
    }

    /// Get the composited 256×224 SGB frame (border + game).
    pub fn sgb_composited_frame(&mut self) -> &[u32] {
        if let Some(ref sgb) = self.bus.sgb {
            if sgb.border_dirty {
                sgb.render_border(&mut self.border_buffer);
                // border_dirty will be cleared below
            }
        }
        if let Some(ref mut sgb) = self.bus.sgb {
            sgb.border_dirty = false;
        }

        // Copy border into output
        self.sgb_output.copy_from_slice(&self.border_buffer);

        // Composite game frame
        let game_buf = if let Some(ref sgb) = self.bus.sgb {
            if sgb.mask_mode == 1 {
                &sgb.frozen_buffer
            } else {
                self.bus.ppu.frame_buffer()
            }
        } else {
            self.bus.ppu.frame_buffer()
        };
        Sgb::composite_frame(&mut self.sgb_output, game_buf);

        &self.sgb_output
    }

    // ── Input handling ────────────────────────────────────────────────────────

    pub fn set_button(&mut self, button: u8, pressed: bool) {
        self.bus.joypad.set_button(button, pressed);
    }

    // Expose button constants for main.rs
    pub const BTN_RIGHT:  u8 = BTN_RIGHT;
    pub const BTN_LEFT:   u8 = BTN_LEFT;
    pub const BTN_UP:     u8 = BTN_UP;
    pub const BTN_DOWN:   u8 = BTN_DOWN;
    pub const BTN_A:      u8 = BTN_A;
    pub const BTN_B:      u8 = BTN_B;
    pub const BTN_SELECT: u8 = BTN_SELECT;
    pub const BTN_START:  u8 = BTN_START;
}
