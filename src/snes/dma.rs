/// SNES DMA controller — 8 channels of general-purpose DMA.
/// Used by the SGB BIOS for VRAM/CGRAM bulk transfers.

#[derive(Clone, Copy)]
pub struct DmaChannel {
    pub dmap: u8,    // $43x0: transfer mode / direction
    pub bbad: u8,    // $43x1: B-bus address ($21xx)
    pub a1t: u32,    // $43x2-x4: A-bus source address (24-bit)
    pub das: u16,    // $43x5-x6: byte count (0 = 65536)
    pub dasb: u8,    // $43x7: indirect bank (HDMA only)
    pub a2a: u16,    // $43x8-x9: HDMA table address
    pub ntrl: u8,    // $43xA: HDMA line counter
    pub unused: u8,  // $43xB: unused
}

impl DmaChannel {
    pub fn new() -> Self {
        DmaChannel {
            dmap: 0xFF,
            bbad: 0xFF,
            a1t: 0xFFFFFF,
            das: 0xFFFF,
            dasb: 0xFF,
            a2a: 0xFFFF,
            ntrl: 0xFF,
            unused: 0xFF,
        }
    }
}

#[derive(Clone)]
pub struct DmaController {
    pub channels: [DmaChannel; 8],
}

impl DmaController {
    pub fn new() -> Self {
        DmaController {
            channels: [DmaChannel::new(); 8],
        }
    }

    pub fn read(&self, addr: u16) -> u8 {
        let ch = ((addr >> 4) & 7) as usize;
        let reg = addr & 0xF;
        let c = &self.channels[ch];
        match reg {
            0x0 => c.dmap,
            0x1 => c.bbad,
            0x2 => c.a1t as u8,
            0x3 => (c.a1t >> 8) as u8,
            0x4 => (c.a1t >> 16) as u8,
            0x5 => c.das as u8,
            0x6 => (c.das >> 8) as u8,
            0x7 => c.dasb,
            0x8 => c.a2a as u8,
            0x9 => (c.a2a >> 8) as u8,
            0xA => c.ntrl,
            0xB | 0xF => c.unused,
            _ => 0,
        }
    }

    pub fn write(&mut self, addr: u16, val: u8) {
        let ch = ((addr >> 4) & 7) as usize;
        let reg = addr & 0xF;
        let c = &mut self.channels[ch];
        match reg {
            0x0 => c.dmap = val,
            0x1 => c.bbad = val,
            0x2 => c.a1t = (c.a1t & 0xFFFF00) | val as u32,
            0x3 => c.a1t = (c.a1t & 0xFF00FF) | ((val as u32) << 8),
            0x4 => c.a1t = (c.a1t & 0x00FFFF) | ((val as u32) << 16),
            0x5 => c.das = (c.das & 0xFF00) | val as u16,
            0x6 => c.das = (c.das & 0x00FF) | ((val as u16) << 8),
            0x7 => c.dasb = val,
            0x8 => c.a2a = (c.a2a & 0xFF00) | val as u16,
            0x9 => c.a2a = (c.a2a & 0x00FF) | ((val as u16) << 8),
            0xA => c.ntrl = val,
            0xB | 0xF => c.unused = val,
            _ => {}
        }
    }

    /// Execute general-purpose DMA for channels indicated by the enable mask.
    /// Returns the number of master cycles consumed.
    /// `bus_read` reads from the A-bus (full 24-bit address space).
    /// `b_read` reads from B-bus ($2100-$21FF).
    /// `bus_write` writes to the A-bus.
    /// `b_write` writes to B-bus.
    pub fn run_dma(
        &mut self,
        enable: u8,
        bus_read: &dyn Fn(u32) -> u8,
        b_read: &dyn Fn(u16) -> u8,
        bus_write: &mut dyn FnMut(u32, u8),
        b_write: &mut dyn FnMut(u16, u8),
    ) -> u64 {
        let mut total_cycles = 0u64;
        for ch in 0..8 {
            if enable & (1 << ch) == 0 { continue; }
            let c = &self.channels[ch];
            let direction = c.dmap & 0x80; // 0=A→B, 0x80=B→A
            let mode = c.dmap & 0x07;
            let fixed = c.dmap & 0x08 != 0;
            let decrement = c.dmap & 0x10 != 0;
            let mut a_addr = c.a1t;
            let mut remaining = if c.das == 0 { 0x10000u32 } else { c.das as u32 };
            let b_base = 0x2100u16 | c.bbad as u16;

            // Transfer pattern offsets based on mode
            let offsets: &[u16] = match mode {
                0 => &[0],
                1 => &[0, 1],
                2 | 6 => &[0, 0],
                3 | 7 => &[0, 0, 1, 1],
                4 => &[0, 1, 2, 3],
                5 => &[0, 1, 0, 1],
                _ => &[0],
            };

            let mut offset_idx = 0usize;
            while remaining > 0 {
                let b_addr = b_base.wrapping_add(offsets[offset_idx % offsets.len()]);
                if direction == 0 {
                    // A→B
                    let val = bus_read(a_addr);
                    b_write(b_addr, val);
                } else {
                    // B→A
                    let val = b_read(b_addr);
                    bus_write(a_addr, val);
                }

                if !fixed {
                    if decrement {
                        a_addr = (a_addr & 0xFF0000) | ((a_addr as u16).wrapping_sub(1) as u32 & 0xFFFF);
                    } else {
                        a_addr = (a_addr & 0xFF0000) | ((a_addr as u16).wrapping_add(1) as u32 & 0xFFFF);
                    }
                }

                offset_idx += 1;
                remaining -= 1;
                total_cycles += 8; // ~8 master cycles per byte
            }

            // Update channel state
            self.channels[ch].a1t = a_addr;
            self.channels[ch].das = 0;
        }
        total_cycles
    }
}
