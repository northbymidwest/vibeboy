use std::sync::Arc;
use super::Cartridge;

pub struct PocketCamera {
    rom: Arc<[u8]>,
    ram: Vec<u8>,
    rom_bank: usize,
    ram_bank: usize,
    camera_regs_mapped: bool,
    camera_regs: [u8; 0x36],
    image_ready: bool,       // set after capture completes, image can be read
    noise_seed: u32,
    camera_image: Option<Box<[u8; 128 * 112]>>,
}

impl PocketCamera {
    pub(super) fn new(rom: Arc<[u8]>) -> Self {
        PocketCamera {
            rom,
            ram: vec![0u8; 0x20000], // 128KB SRAM
            rom_bank: 1,
            ram_bank: 0,
            camera_regs_mapped: false,
            camera_regs: [0; 0x36],
            image_ready: false,
            noise_seed: 0x1234,
            camera_image: None,
        }
    }

    /// Generate a noise value for a pixel coordinate (deterministic hash).
    fn noise(&self, x: u8, y: u8) -> u8 {
        let value = (x as u32).wrapping_mul(151)
            .wrapping_add((y as u32).wrapping_mul(149))
            ^ self.noise_seed;
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
        let gain_idx = (self.camera_regs[4] & 0x1F) as usize;
        #[expect(clippy::excessive_precision)]
        const GAIN: [f64; 32] = [
            0.881, 0.915, 0.946, 0.974, 1.000, 1.024, 1.047, 1.068,
            1.088, 1.124, 1.157, 1.187, 1.214, 1.240, 1.274, 1.316,
            1.353, 1.386, 1.416, 1.443, 1.469, 1.493, 1.515, 1.536,
            1.555, 1.574, 1.591, 1.608, 1.624, 1.639, 1.653, 1.667,
        ];
        let color = (raw as f64 * GAIN[gain_idx]) as i32;

        // Apply exposure
        let exposure = ((self.camera_regs[2] as i32) << 8) | (self.camera_regs[3] as i32);
        color * exposure / 0x1000
    }

    /// Generate one byte of the 128×112 captured image on-the-fly.
    /// `offset` is relative to $A100, range 0..0xE00 (3584 bytes = 224 tiles).
    fn read_image_byte(&self, offset: u16) -> u8 {
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
                if color < self.camera_regs[pat_base] as i32 {
                    3
                } else if color < self.camera_regs[pat_base + 1] as i32 {
                    2
                } else if color < self.camera_regs[pat_base + 2] as i32 {
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
}

impl Cartridge for PocketCamera {
    fn read_rom(&self, addr: u16) -> u8 {
        let idx = match addr {
            0x0000..=0x3FFF => addr as usize,
            0x4000..=0x7FFF => self.rom_bank * 0x4000 + (addr as usize - 0x4000),
            _ => return 0xFF,
        };
        self.rom.get(idx % self.rom.len().max(1)).copied().unwrap_or(0xFF)
    }

    fn write_rom(&mut self, addr: u16, val: u8) {
        match addr {
            0x0000..=0x1FFF => {} // ram_enable accepted but camera ignores it
            0x2000..=0x3FFF => {
                // Full 8-bit ROM bank select
                self.rom_bank = val as usize;
                if self.rom_bank == 0 { self.rom_bank = 1; }
            }
            0x4000..=0x5FFF => {
                self.ram_bank = val as usize;
                self.camera_regs_mapped = val & 0x10 != 0;
            }
            _ => {}
        }
    }

    fn read_ram(&self, addr: u16) -> u8 {
        // Camera register reads: only register 0 returns data, rest return 0
        if self.camera_regs_mapped {
            if (addr & 0x7F) == 0 {
                return self.camera_regs[0];
            }
            return 0;
        }

        // Camera busy: all RAM reads return 0
        if self.camera_regs[0] & 1 != 0 {
            return 0;
        }

        // Bank 0, $A100-$AEFF: generate image on the fly
        let ram_bank = self.ram_bank & 0x0F;
        if self.image_ready && ram_bank == 0 && addr >= 0xA100 && addr < 0xAF00 {
            return self.read_image_byte(addr - 0xA100);
        }

        // Normal RAM read (camera bypasses ram_enable)
        let idx = ram_bank * 0x2000 + (addr as usize - 0xA000);
        self.ram.get(idx % self.ram.len()).copied().unwrap_or(0xFF)
    }

    fn write_ram(&mut self, addr: u16, val: u8) {
        if self.camera_regs_mapped {
            let reg = (addr as usize) & 0x7F;
            if reg == 0 {
                let old = self.camera_regs[0];
                let new_val = val & 0x07;
                self.camera_regs[0] = new_val;
                // Trigger capture on 0→1 transition of bit 0
                if new_val & 1 != 0 && old & 1 == 0 {
                    // Randomize noise seed each capture
                    self.noise_seed = self.noise_seed.wrapping_mul(1103515245).wrapping_add(12345);
                    self.image_ready = true;
                    // Immediately mark capture complete (clear busy bit)
                    self.camera_regs[0] &= !1;
                }
            } else if reg < 0x36 {
                self.camera_regs[reg] = val;
            }
            return;
        }

        // Camera busy: forbid RAM writes
        if self.camera_regs[0] & 1 != 0 { return; }

        // Normal RAM write (camera bypasses ram_enable)
        let ram_bank = self.ram_bank & 0x0F;
        let idx = ram_bank * 0x2000 + (addr as usize - 0xA000);
        if idx < self.ram.len() {
            self.ram[idx] = val;
        }
    }

    fn has_battery(&self) -> bool { true }
    fn ram_data(&self) -> &[u8] { &self.ram }
    fn load_ram(&mut self, data: &[u8]) {
        let len = self.ram.len().min(data.len());
        self.ram[..len].copy_from_slice(&data[..len]);
    }
    fn has_camera(&self) -> bool { true }
    fn set_camera_image(&mut self, grayscale: &[u8; 128 * 112]) {
        let img = self.camera_image.get_or_insert_with(|| Box::new([0u8; 128 * 112]));
        img.copy_from_slice(grayscale);
    }
    fn snapshot_state(&self) -> Vec<u8> {
        let mut s = Vec::new();
        s.extend_from_slice(&(self.rom_bank as u32).to_le_bytes());
        s.extend_from_slice(&(self.ram_bank as u32).to_le_bytes());
        s.push(self.camera_regs_mapped as u8);
        s.extend_from_slice(&self.camera_regs);
        s.push(self.image_ready as u8);
        s.extend_from_slice(&self.noise_seed.to_le_bytes());
        s.extend_from_slice(&self.ram);
        s
    }
    fn restore_state(&mut self, d: &[u8]) {
        if d.len() < 9 + 0x36 + 1 + 4 { return; }
        self.rom_bank = u32::from_le_bytes([d[0],d[1],d[2],d[3]]) as usize;
        self.ram_bank = u32::from_le_bytes([d[4],d[5],d[6],d[7]]) as usize;
        self.camera_regs_mapped = d[8] != 0;
        self.camera_regs.copy_from_slice(&d[9..9+0x36]);
        let o = 9 + 0x36;
        self.image_ready = d[o] != 0;
        self.noise_seed = u32::from_le_bytes([d[o+1],d[o+2],d[o+3],d[o+4]]);
        let ram = &d[o+5..];
        let len = self.ram.len().min(ram.len());
        self.ram[..len].copy_from_slice(&ram[..len]);
    }
}
