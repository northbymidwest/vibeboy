mod apu;
mod bus;
mod cartridge;
mod cpu;
mod emulator;
mod joypad;
mod ppu;
mod timer;

use emulator::Emulator;
use minifb::{Key, Window, WindowOptions};
use std::env;
use std::fs;
use std::time::{Duration, Instant};

/// Display scale factor (160×144 → 480×432).
const SCALE: usize = 3;
const WIDTH: usize = 160 * SCALE;
const HEIGHT: usize = 144 * SCALE;
/// Target frame time for ~60 fps.
const FRAME_DURATION: Duration = Duration::from_nanos(16_742_706);

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <rom.gb>", args[0]);
        std::process::exit(1);
    }

    let rom = fs::read(&args[1]).unwrap_or_else(|e| {
        eprintln!("Failed to read ROM '{}': {}", args[1], e);
        std::process::exit(1);
    });

    let mut emu = Emulator::new(rom);

    let mut window = Window::new(
        "GBC Emulator",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: false,
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| {
        eprintln!("Failed to create window: {}", e);
        std::process::exit(1);
    });

    // Scaled frame buffer for minifb
    let mut scaled_buf = vec![0u32; WIDTH * HEIGHT];

    let mut frame_start = Instant::now();
    let mut frame_count = 0u32;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // ── Input ─────────────────────────────────────────────────────────────
        handle_input(&mut emu, &window);

        // ── Run one frame ─────────────────────────────────────────────────────
        emu.step_frame();
        frame_count += 1;

        // ── Debug dump every 60 frames and on key D ───────────────────────────
        let sprites_on = emu.bus.ppu.lcdc & 0x02 != 0;
        if window.is_key_pressed(Key::D, minifb::KeyRepeat::No) {
            eprintln!("[frame {frame_count}]");
            dump_debug(&emu);
        }

        // ── Scale frame buffer ────────────────────────────────────────────────
        let src = emu.frame_buffer();
        scale_frame(src, &mut scaled_buf, SCALE);

        // ── Render ────────────────────────────────────────────────────────────
        window
            .update_with_buffer(&scaled_buf, WIDTH, HEIGHT)
            .unwrap();

        // ── Frame rate cap ────────────────────────────────────────────────────
        let elapsed = frame_start.elapsed();
        if elapsed < FRAME_DURATION {
            std::thread::sleep(FRAME_DURATION - elapsed);
        }
        frame_start = Instant::now();
    }
}

fn dump_debug(emu: &Emulator) {
    let ppu = &emu.bus.ppu;
    let lcdc = ppu.lcdc;
    eprintln!("=== DEBUG DUMP ===");
    eprintln!("LCDC={:#04X}  SCY={}  SCX={}  WY={}  WX={}", lcdc, ppu.scy, ppu.scx, ppu.wy, ppu.wx);
    eprintln!("Sprites enabled={}, size={}", lcdc & 0x02 != 0, if lcdc & 0x04 != 0 { "8x16" } else { "8x8" });

    // Dump ALL on-screen OAM entries and their referenced tiles
    let sprite_h: i16 = if lcdc & 0x04 != 0 { 16 } else { 8 };
    eprintln!("--- OAM on-screen entries (sprite size {}x{}) ---", 8, sprite_h);
    let mut seen_tiles: std::collections::HashSet<(usize, u8)> = std::collections::HashSet::new();
    let mut any_onscreen = false;
    for i in 0..40usize {
        let y = ppu.oam[i * 4 + 0];
        let x = ppu.oam[i * 4 + 1];
        let tile = ppu.oam[i * 4 + 2];
        let attr = ppu.oam[i * 4 + 3];
        let bank = if attr & 0x08 != 0 { 1usize } else { 0usize };
        let sprite_y = y as i16 - 16;
        let sprite_x = x as i16 - 8;
        let on_screen = sprite_y >= -sprite_h && sprite_y < 144 && sprite_x > -8 && sprite_x < 160;
        if on_screen {
            any_onscreen = true;
            eprintln!("  [{i:02}] screen=({sprite_x:4},{sprite_y:4}) tile={tile:#04X} attr={attr:#010b} vram_bank={bank}");
            let t = if lcdc & 0x04 != 0 { tile & 0xFE } else { tile } as u8;
            seen_tiles.insert((bank, t));
            if lcdc & 0x04 != 0 { seen_tiles.insert((bank, t + 1)); }
        }
    }
    if !any_onscreen {
        eprintln!("  (no sprites on screen)");
    }

    // Dump the actual tile pixel data for each referenced tile
    eprintln!("--- Referenced sprite tiles ---");
    let mut tile_list: Vec<(usize, u8)> = seen_tiles.into_iter().collect();
    tile_list.sort();
    for (bank, tile_n) in &tile_list {
        let base = *tile_n as usize * 16;
        let mut rows = Vec::new();
        for row in 0..8usize {
            let lo = ppu.vram[*bank][base + row * 2];
            let hi = ppu.vram[*bank][base + row * 2 + 1];
            let mut line = String::new();
            for bit in (0..8).rev() {
                let idx = (((hi >> bit) & 1) << 1) | ((lo >> bit) & 1);
                line.push(match idx { 0 => '.', 1 => '1', 2 => '2', _ => '3' });
            }
            rows.push(line);
        }
        eprintln!("  bank{} tile {tile_n:#04X}: {}", bank, rows.join("|"));
    }

    // Dump BG palettes too
    eprintln!("--- BG palettes ---");
    for p in 0..8usize {
        let colors: Vec<String> = (0..4).map(|c| {
            let off = p * 8 + c * 2;
            let lo = ppu.bcpd[off] as u16;
            let hi = ppu.bcpd[off + 1] as u16;
            let rgb15 = lo | (hi << 8);
            format!("#{rgb15:04X}")
        }).collect();
        eprintln!("  bgpal{p}: {}", colors.join(" "));
    }

    // Dump all 8 OBJ palettes
    eprintln!("--- OBJ palettes ---");
    for p in 0..8usize {
        let colors: Vec<String> = (0..4).map(|c| {
            let off = p * 8 + c * 2;
            let lo = ppu.ocpd[off] as u16;
            let hi = ppu.ocpd[off + 1] as u16;
            let rgb15 = lo | (hi << 8);
            let r5 = (rgb15 & 0x1F) << 3;
            let g5 = ((rgb15 >> 5) & 0x1F) << 3;
            let b5 = ((rgb15 >> 10) & 0x1F) << 3;
            format!("c{c}=#{rgb15:04X}({r5},{g5},{b5})")
        }).collect();
        eprintln!("  pal{p}: {}", colors.join("  "));
    }
    eprintln!("==================");
}

fn handle_input(emu: &mut Emulator, window: &Window) {
    let map: &[(Key, u8)] = &[
        (Key::Z,          Emulator::BTN_A),
        (Key::X,          Emulator::BTN_B),
        (Key::Enter,      Emulator::BTN_START),
        (Key::RightShift, Emulator::BTN_SELECT),
        (Key::Right,      Emulator::BTN_RIGHT),
        (Key::Left,       Emulator::BTN_LEFT),
        (Key::Up,         Emulator::BTN_UP),
        (Key::Down,       Emulator::BTN_DOWN),
    ];
    for (key, btn) in map {
        emu.set_button(*btn, window.is_key_down(*key));
    }
}

/// Nearest-neighbor scale `src` (160×144) into `dst` (160*scale × 144*scale).
fn scale_frame(src: &[u32], dst: &mut [u32], scale: usize) {
    for y in 0..144 {
        for x in 0..160 {
            let pixel = src[y * 160 + x];
            for sy in 0..scale {
                for sx in 0..scale {
                    dst[(y * scale + sy) * WIDTH + x * scale + sx] = pixel;
                }
            }
        }
    }
}
