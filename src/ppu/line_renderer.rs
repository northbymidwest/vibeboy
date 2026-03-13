/// Line-based PPU renderer — renders one full scanline at a time.
///
/// Trades cycle-accuracy for simplicity and speed. Renders BG, window, and
/// sprites in a single pass per scanline using the register values captured
/// at the start of mode 3.

use super::Ppu;

/// Render scanline `ly` into the frame buffer using current PPU state.
/// Called once per scanline when mode 3 begins (after OAM scan).
pub fn render_scanline(ppu: &mut Ppu) {
    let ly = ppu.ly as usize;
    if ly >= 144 {
        return;
    }

    let lcdc = ppu.lcdc;
    let scx = ppu.scx;
    let scy = ppu.scy;
    let wx = ppu.wx;


    let mut line = [0u32; 160];
    // Track BG color indices for sprite priority
    let mut bg_color_indices = [0u8; 160];

    // ── Background ──────────────────────────────────────────────────────
    let bg_enabled = if ppu.cgb_mode { true } else { lcdc & 0x01 != 0 };

    if bg_enabled {
        let tile_map_base: usize = if lcdc & 0x08 != 0 { 0x1C00 } else { 0x1800 };
        let tile_data_signed = lcdc & 0x10 == 0;
        let y = scy.wrapping_add(ly as u8);
        let tile_row = (y as usize / 8) & 31;
        let fine_y = y as usize % 8;

        for px in 0..160u8 {
            let x = scx.wrapping_add(px);
            let tile_col = (x as usize / 8) & 31;
            let fine_x = x as usize % 8;

            let map_addr = tile_map_base + tile_row * 32 + tile_col;
            let tile_id = ppu.vram[0][map_addr];

            // CGB: read tile attributes from VRAM bank 1
            let attrs = if ppu.cgb_mode { ppu.vram[1][map_addr] } else { 0 };
            let vram_bank = if attrs & 0x08 != 0 && !ppu.dmg_compat { 1 } else { 0 };
            let x_flip = attrs & 0x20 != 0 && !ppu.dmg_compat;
            let y_flip = attrs & 0x40 != 0 && !ppu.dmg_compat;
            let bg_priority = attrs & 0x80 != 0 && !ppu.dmg_compat;
            let cgb_palette = if ppu.dmg_compat { 0 } else { (attrs & 0x07) as usize };

            let actual_fine_y = if y_flip { 7 - fine_y } else { fine_y };
            let actual_fine_x = if x_flip { 7 - fine_x } else { fine_x };

            let tile_addr = tile_data_address(tile_id, actual_fine_y, tile_data_signed);
            let lo = ppu.vram[vram_bank][tile_addr];
            let hi = ppu.vram[vram_bank][tile_addr + 1];

            let bit = 7 - actual_fine_x;
            let color_idx = ((hi >> bit) & 1) << 1 | ((lo >> bit) & 1);
            bg_color_indices[px as usize] = color_idx | if bg_priority { 0x80 } else { 0 };

            line[px as usize] = pixel_color(ppu, color_idx, cgb_palette, false, 0);
        }
    } else {
        // BG disabled (DMG only) — white
        for px in 0..160 {
            line[px] = Ppu::DMG_SHADES[0];
        }
    }

    // ── Window ──────────────────────────────────────────────────────────
    let window_enabled = lcdc & 0x20 != 0 && ppu.wy_triggered;
    if window_enabled && wx <= 166 {
        let win_x_start = if wx < 7 { 0i16 } else { wx as i16 - 7 };
        let win_y = ppu.window_line_counter as usize;
        let tile_map_base: usize = if lcdc & 0x40 != 0 { 0x1C00 } else { 0x1800 };
        let tile_data_signed = lcdc & 0x10 == 0;
        let tile_row = win_y / 8;
        let fine_y = win_y % 8;

        for px in win_x_start..160 {
            let win_px = (px - win_x_start) as usize;
            let tile_col = win_px / 8;
            let fine_x = win_px % 8;

            if tile_col >= 32 {
                break;
            }

            let map_addr = tile_map_base + (tile_row & 31) * 32 + tile_col;
            let tile_id = ppu.vram[0][map_addr];

            let attrs = if ppu.cgb_mode { ppu.vram[1][map_addr] } else { 0 };
            let vram_bank = if attrs & 0x08 != 0 && !ppu.dmg_compat { 1 } else { 0 };
            let x_flip = attrs & 0x20 != 0 && !ppu.dmg_compat;
            let y_flip = attrs & 0x40 != 0 && !ppu.dmg_compat;
            let bg_priority = attrs & 0x80 != 0 && !ppu.dmg_compat;
            let cgb_palette = if ppu.dmg_compat { 0 } else { (attrs & 0x07) as usize };

            let actual_fine_y = if y_flip { 7 - fine_y } else { fine_y };
            let actual_fine_x = if x_flip { 7 - fine_x } else { fine_x };

            let tile_addr = tile_data_address(tile_id, actual_fine_y, tile_data_signed);
            let lo = ppu.vram[vram_bank][tile_addr];
            let hi = ppu.vram[vram_bank][tile_addr + 1];

            let bit = 7 - actual_fine_x;
            let color_idx = ((hi >> bit) & 1) << 1 | ((lo >> bit) & 1);
            bg_color_indices[px as usize] = color_idx | if bg_priority { 0x80 } else { 0 };

            line[px as usize] = pixel_color(ppu, color_idx, cgb_palette, false, 0);
        }

        // Window was visible on this line — signal to caller
        ppu.alt_window_line_used = true;
    }

    // ── Sprites ─────────────────────────────────────────────────────────
    if lcdc & 0x02 != 0 || ppu.cgb_mode {
        let tall = lcdc & 0x04 != 0;
        let sprite_h: u8 = if tall { 16 } else { 8 };

        // Collect sprites for this line (max 10, same as hardware)
        let mut sprites: Vec<(u8, u8, u8, u8, u8)> = Vec::with_capacity(10);
        for i in 0..40usize {
            let base = i * 4;
            let sy = ppu.oam[base];
            let sx = ppu.oam[base + 1];
            let tile = ppu.oam[base + 2];
            let attrs = ppu.oam[base + 3];

            let screen_y = sy.wrapping_sub(16);
            if (ly as u8) >= screen_y && (ly as u8) < screen_y.wrapping_add(sprite_h) {
                sprites.push((sy, sx, tile, attrs, i as u8));
                if sprites.len() >= 10 {
                    break;
                }
            }
        }

        // Sort by priority: DMG = X-coordinate, CGB with OPRI bit 0 = 0 → OAM index
        if !ppu.cgb_mode || ppu.opri & 0x01 != 0 {
            // DMG priority: lower X first, then lower OAM index
            sprites.sort_by(|a, b| a.1.cmp(&b.1).then(a.4.cmp(&b.4)));
        }
        // CGB OAM-index priority: already in OAM order

        // Render back-to-front so higher-priority sprites overwrite
        for &(sy, sx, tile, attrs, _oam_idx) in sprites.iter().rev() {
            let x_flip = attrs & 0x20 != 0;
            let y_flip = attrs & 0x40 != 0;
            let bg_over = attrs & 0x80 != 0;
            let vram_bank = if ppu.cgb_mode && !ppu.dmg_compat && attrs & 0x08 != 0 { 1 } else { 0 };
            let cgb_palette = if ppu.dmg_compat {
                if attrs & 0x10 != 0 { 1usize } else { 0 }
            } else {
                (attrs & 0x07) as usize
            };
            let dmg_palette = if attrs & 0x10 != 0 { 1u8 } else { 0u8 };

            let sprite_y = (ly as u8).wrapping_sub(sy.wrapping_sub(16));
            let actual_y = if y_flip { sprite_h - 1 - sprite_y } else { sprite_y };

            let tile_id = if tall { tile & 0xFE | if actual_y >= 8 { 1 } else { 0 } } else { tile };
            let row = (actual_y % 8) as usize;
            let tile_addr = (tile_id as usize) * 16 + row * 2;
            let lo = ppu.vram[vram_bank][tile_addr];
            let hi = ppu.vram[vram_bank][tile_addr + 1];

            for bit in 0..8u8 {
                let screen_x = sx.wrapping_sub(8).wrapping_add(bit) as usize;
                if screen_x >= 160 {
                    continue;
                }

                let actual_bit = if x_flip { bit } else { 7 - bit };
                let color_idx = ((hi >> actual_bit) & 1) << 1 | ((lo >> actual_bit) & 1);
                if color_idx == 0 {
                    continue; // transparent
                }

                // Sprite priority check
                if !ppu.cgb_mode && lcdc & 0x02 == 0 {
                    continue; // DMG: sprites disabled
                }

                let bg_ci = bg_color_indices[screen_x];
                let bg_idx = bg_ci & 0x7F;
                let bg_prio = bg_ci & 0x80 != 0;

                if lcdc & 0x01 != 0 {
                    // BG enabled
                    if bg_over && bg_idx != 0 {
                        continue; // sprite behind non-zero BG (OAM attr)
                    }
                    if ppu.cgb_mode && bg_prio && bg_idx != 0 {
                        continue; // CGB BG priority
                    }
                }

                line[screen_x] = pixel_color(ppu, color_idx, cgb_palette, true, dmg_palette);
            }
        }
    }

    // Write scanline to frame buffer
    let fb_offset = ly * 160;
    ppu.frame_buffer[fb_offset..fb_offset + 160].copy_from_slice(&line);

    // SGB shade buffer
    if ppu.sgb_mode {
        for px in 0..160 {
            let c = line[px];
            let r = (c >> 16) & 0xFF;
            let g = (c >> 8) & 0xFF;
            let b = c & 0xFF;
            let lum = (r * 3 + g * 6 + b) / 10;
            let shade = if lum > 192 { 0 } else if lum > 128 { 1 } else if lum > 64 { 2 } else { 3 };
            ppu.shade_buffer[fb_offset + px] = shade as u8;
        }
    }
}

/// Compute tile data address in VRAM for a given tile ID and row.
fn tile_data_address(tile_id: u8, fine_y: usize, signed_mode: bool) -> usize {
    let base = if signed_mode {
        // $8800 mode: tile_id is signed offset from $9000
        (0x1000_i32 + (tile_id as i8 as i32) * 16) as usize
    } else {
        // $8000 mode: tile_id is unsigned offset from $8000
        (tile_id as usize) * 16
    };
    base + fine_y * 2
}

/// Convert a color index + palette info to a 32-bit ARGB pixel.
fn pixel_color(ppu: &Ppu, color_idx: u8, cgb_palette: usize, is_sprite: bool, dmg_pal: u8) -> u32 {
    if ppu.cgb_mode {
        if is_sprite {
            gbc_palette_color(&ppu.ocpd, cgb_palette, color_idx as usize)
        } else {
            gbc_palette_color(&ppu.bcpd, cgb_palette, color_idx as usize)
        }
    } else {
        let pal_reg = if is_sprite {
            if dmg_pal == 1 { ppu.obp1 } else { ppu.obp0 }
        } else {
            ppu.bgp
        };
        let shade = (pal_reg >> (color_idx * 2)) & 0x03;
        Ppu::DMG_SHADES[shade as usize]
    }
}

/// Read a color from GBC palette data (bcpd or ocpd).
fn gbc_palette_color(palette_data: &[u8; 64], palette_idx: usize, color_idx: usize) -> u32 {
    let offset = palette_idx * 8 + color_idx * 2;
    let lo = palette_data[offset] as u16;
    let hi = palette_data[offset + 1] as u16;
    let c = lo | (hi << 8);

    let r5 = (c & 0x1F) as u32;
    let g5 = ((c >> 5) & 0x1F) as u32;
    let b5 = ((c >> 10) & 0x1F) as u32;
    let r8 = (r5 << 3) | (r5 >> 2);
    let g8 = (g5 << 3) | (g5 >> 2);
    let b8 = (b5 << 3) | (b5 >> 2);
    (r8 << 16) | (g8 << 8) | b8
}
