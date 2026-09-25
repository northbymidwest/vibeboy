use super::{BAD_REGISTERS, CartState, Cartridge, WRONG_MAPPER, ensure, ensure_ram_len};
use std::sync::Arc;

/// Captured image location in SRAM bank 0 ($A100) and size (224 tiles).
const IMAGE_SRAM_OFFSET: usize = 0x100;
const IMAGE_BYTES: usize = 0xE00;

pub struct PocketCamera {
    rom: Arc<[u8]>,
    camera_image: Option<Box<[u8; 128 * 112]>>,
    state: PocketCameraState,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct PocketCameraState {
    ram: Vec<u8>,
    rom_bank: usize,
    ram_bank: usize,
    camera_regs_mapped: bool,
    #[serde(with = "serde_big_array::BigArray")]
    camera_regs: [u8; 0x36],
    noise_seed: u32,
}

impl PocketCamera {
    pub(super) fn new(rom: Arc<[u8]>) -> Self {
        PocketCamera {
            rom,
            camera_image: None,
            state: PocketCameraState {
                ram: vec![0u8; 0x20000], // 128KB SRAM
                rom_bank: 1,
                ram_bank: 0,
                camera_regs_mapped: false,
                camera_regs: [0; 0x36],
                noise_seed: 0x1234,
            },
        }
    }

    /// Generate a noise value for a pixel coordinate (deterministic hash).
    fn noise(&self, x: u8, y: u8) -> u8 {
        let value = (x as u32)
            .wrapping_mul(151)
            .wrapping_add((y as u32).wrapping_mul(149))
            ^ self.state.noise_seed;
        let mut hash: u32 = 0;
        let mut v = value;
        for _ in 0..32 {
            hash <<= 1;
            if hash & 0x100 != 0 {
                hash ^= 0x101;
            }
            if v & 0x8000_0000 != 0 {
                hash ^= 0xA1;
            }
            v <<= 1;
        }
        hash as u8
    }

    /// Compute the processed color for pixel (x,y) using gain + exposure.
    fn processed_color(&self, x: u8, y: u8) -> i32 {
        let px = x.min(127);
        let py = y.min(111);
        let raw = if let Some(ref img) = self.camera_image {
            img[py as usize * 128 + px as usize] as i32
        } else {
            self.noise(px, py) as i32
        };

        // Apply gain
        let gain_idx = (self.state.camera_regs[4] & 0x1F) as usize;
        const GAIN: [f64; 32] = [
            0.881, 0.915, 0.946, 0.974, 1.000, 1.024, 1.047, 1.068, 1.088, 1.124, 1.157, 1.187,
            1.214, 1.240, 1.274, 1.316, 1.353, 1.386, 1.416, 1.443, 1.469, 1.493, 1.515, 1.536,
            1.555, 1.574, 1.591, 1.608, 1.624, 1.639, 1.653, 1.667,
        ];
        let color = (raw as f64 * GAIN[gain_idx]) as i32;

        // Apply exposure
        let exposure =
            ((self.state.camera_regs[2] as i32) << 8) | (self.state.camera_regs[3] as i32);
        color * exposure / 0x1000
    }

    /// Generate one byte of the 128×112 captured image from the current
    /// sensor input and registers.
    /// `offset` is relative to $A100, range 0..0xE00 (3584 bytes = 224 tiles).
    fn image_byte(&self, offset: u16) -> u8 {
        let tile_x = ((offset / 16) % 16) as u8;
        let tile_y = ((offset / 16) / 16) as u8;
        let row = ((offset >> 1) & 7) as u8;
        let bit = (offset & 1) as u8; // 0=low bitplane, 1=high bitplane
        let y = tile_y * 8 + row;

        let mut result: u8 = 0;
        for dx in 0..8u8 {
            let x = tile_x * 8 + dx;
            let color = self.processed_color(x, y);

            // Dither using the 4×4×3 threshold matrix in registers $06-$35
            let pat_idx = ((x & 3) + (y & 3) * 4) as usize;
            let pat_base = 6 + pat_idx * 3; // register offset

            let pixel = if pat_base + 2 < 0x36 {
                if color < self.state.camera_regs[pat_base] as i32 {
                    3
                } else if color < self.state.camera_regs[pat_base + 1] as i32 {
                    2
                } else if color < self.state.camera_regs[pat_base + 2] as i32 {
                    1
                } else {
                    0
                }
            } else {
                0
            };

            result <<= 1;
            result |= (pixel >> bit) & 1;
        }
        result
    }

    /// Process the sensor image with the current gain, exposure and dither
    /// registers and store it in SRAM bank 0 at $A100-$AEFF, as the camera
    /// does at the end of a capture.
    fn capture_to_sram(&mut self) {
        let image: Vec<u8> = (0..IMAGE_BYTES as u16)
            .map(|offset| self.image_byte(offset))
            .collect();
        self.state.ram[IMAGE_SRAM_OFFSET..IMAGE_SRAM_OFFSET + IMAGE_BYTES].copy_from_slice(&image);
    }
}

impl Cartridge for PocketCamera {
    fn read_rom(&self, addr: u16) -> u8 {
        let idx = match addr {
            0x0000..=0x3FFF => addr as usize,
            0x4000..=0x7FFF => self.state.rom_bank * 0x4000 + (addr as usize - 0x4000),
            _ => return 0xFF,
        };
        self.rom
            .get(idx % self.rom.len().max(1))
            .copied()
            .unwrap_or(0xFF)
    }

    fn write_rom(&mut self, addr: u16, val: u8) {
        match addr {
            0x0000..=0x1FFF => {} // ram_enable accepted but camera ignores it
            0x2000..=0x3FFF => {
                // Full 8-bit ROM bank select
                self.state.rom_bank = val as usize;
                if self.state.rom_bank == 0 {
                    self.state.rom_bank = 1;
                }
            }
            0x4000..=0x5FFF => {
                self.state.ram_bank = val as usize;
                self.state.camera_regs_mapped = val & 0x10 != 0;
            }
            _ => {}
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        // Camera register reads: only register 0 returns data, rest return 0
        if self.state.camera_regs_mapped {
            if (addr & 0x7F) == 0 {
                return self.state.camera_regs[0];
            }
            return 0;
        }

        // Camera busy: all RAM reads return 0
        if self.state.camera_regs[0] & 1 != 0 {
            return 0;
        }

        // Normal RAM read (camera bypasses ram_enable)
        let ram_bank = self.state.ram_bank & 0x0F;
        let idx = ram_bank * 0x2000 + (addr as usize - 0xA000);
        self.state
            .ram
            .get(idx % self.state.ram.len())
            .copied()
            .unwrap_or(0xFF)
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        if self.state.camera_regs_mapped {
            let reg = (addr as usize) & 0x7F;
            if reg == 0 {
                let old = self.state.camera_regs[0];
                let new_val = val & 0x07;
                self.state.camera_regs[0] = new_val;
                // Trigger capture on 0→1 transition of bit 0
                if new_val & 1 != 0 && old & 1 == 0 {
                    // Randomize noise seed each capture
                    self.state.noise_seed = self
                        .state
                        .noise_seed
                        .wrapping_mul(1103515245)
                        .wrapping_add(12345);
                    self.capture_to_sram();
                    // Immediately mark capture complete (clear busy bit)
                    self.state.camera_regs[0] &= !1;
                }
            } else if reg < 0x36 {
                self.state.camera_regs[reg] = val;
            }
            return;
        }

        // Camera busy: forbid RAM writes
        if self.state.camera_regs[0] & 1 != 0 {
            return;
        }

        // Normal RAM write (camera bypasses ram_enable)
        let ram_bank = self.state.ram_bank & 0x0F;
        let idx = ram_bank * 0x2000 + (addr as usize - 0xA000);
        if idx < self.state.ram.len() {
            self.state.ram[idx] = val;
        }
    }

    fn has_battery(&self) -> bool {
        true
    }
    fn ram_data(&self) -> &[u8] {
        &self.state.ram
    }
    fn load_ram(&mut self, data: &[u8]) {
        let len = self.state.ram.len().min(data.len());
        self.state.ram[..len].copy_from_slice(&data[..len]);
    }
    fn has_camera(&self) -> bool {
        true
    }
    fn set_camera_image(&mut self, grayscale: &[u8; 128 * 112]) {
        let img = self
            .camera_image
            .get_or_insert_with(|| Box::new([0u8; 128 * 112]));
        img.copy_from_slice(grayscale);
    }
    fn snapshot_state(&self) -> CartState {
        CartState::PocketCamera(self.state.clone())
    }
    fn validate_state(&self, state: &CartState) -> Result<(), &'static str> {
        let CartState::PocketCamera(s) = state else {
            return Err(WRONG_MAPPER);
        };
        ensure_ram_len(&s.ram, &self.state.ram)?;
        ensure(s.rom_bank <= 0xFF && s.ram_bank <= 0xFF, BAD_REGISTERS)
    }
    fn restore_state(&mut self, state: &CartState) {
        let CartState::PocketCamera(s) = state else {
            panic!("{WRONG_MAPPER}");
        };
        self.state.clone_from(s);
    }
}
