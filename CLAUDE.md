# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Game Boy / Game Boy Color emulator ("vibeboy") written in Rust. Supports DMG, DMG0, MGB, SGB, SGB2, CGB, and AGB (GBA in GBC mode) hardware models. Includes SGB Super Game Boy emulation with optional SNES LLE via a WDC 65C816 CPU.

## Build & Run

```bash
cargo build --release
cargo run --release -- path/to/rom.gbc

# With boot ROM and model override
cargo run --release -- path/to/rom.gbc --model dmg --boot-rom bootroms/dmg_boot.bin
```

## Testing

Test ROMs live in `test-roms/` (blargg, mooneye-test-suite, SameSuite). The `test_runner` binary runs them:

```bash
# Mooneye tests (breakpoint detection, Fibonacci register check)
cargo run --release --bin test_runner -- test-roms/mooneye-test-suite/build/acceptance/

# Blargg tests (serial output detection)
cargo run --release --bin test_runner -- test-roms/blargg/ blargg

# Single test
cargo run --release --bin test_runner -- test-roms/blargg/cpu_instrs/individual/01-special.gb blargg

# Screenshot a ROM after N frames
cargo run --release --bin test_runner -- path/to/rom.gb screenshot --frames 300 --out screenshot.png

# Force a specific model
cargo run --release --bin test_runner -- test-roms/mooneye-test-suite/build/acceptance/ --model dmg

# Run with boot ROM
cargo run --release --bin test_runner -- test-roms/mooneye-test-suite/build/acceptance/ --boot
```

Test runner auto-detects hardware model from filename suffixes (`-dmgABCmgb`, `-sgb2`, `-GS`, `-A`, etc.) and from the CGB cart header flag.

**Current test status:** 75/75 mooneye acceptance, 57/58 blargg (oam_bug test 7 hangs), 33/70 SameSuite APU.

## Architecture

The emulator loop is: `Emulator::step_frame()` calls `Cpu::step()` which executes one instruction, returning T-cycles consumed. `Bus::tick_mcycle()` advances all subsystems (PPU, APU, Timer, Serial) by 4 T-cycles (one M-cycle).

### Key data flow
- **CPU** (`cpu/mod.rs`) executes SM83 opcodes, calls `bus.tick_mcycle()` between M-cycles, reads/writes memory via `bus.read_byte()`/`bus.write_byte()`
- **Bus** (`bus.rs`) owns all subsystems and implements the memory map. `tick_mcycle()` steps Timer, PPU, APU, Joypad, OAM DMA, and HDMA each M-cycle
- **PPU** (`ppu/mod.rs`) is a pixel FIFO renderer ticked 1 T-cycle at a time internally via `step(4)`. VRAM and OAM live in the Ppu struct; Bus delegates access
- **APU** (`apu.rs`) uses a DIV-coupled frame sequencer; Bus detects DIV falling edges and calls `apu.div_event()`

### PPU timing model
- DMG line-start has a 5-dot state machine (`line_start_pending`, dots 1-5) that delays `visible_ly`, `ly_for_comparison`, and `mode_for_interrupt` transitions to match SameBoy-accurate timing
- Mode transitions happen internally before STAT register bits update (1T delay)
- `oam_bug_row` captures `accessed_oam_row` at end of `step()` for CPU-side OAM corruption checks

### Memory ownership
- VRAM (2 banks), OAM (160 bytes) → owned by `Ppu`
- WRAM (8 banks), HRAM, IO registers → owned by `Bus`
- Cart ROM/RAM → owned by `Cartridge` trait objects in `Bus`

### SGB subsystem
- `sgb.rs`: HLE command processing (palettes, attributes, borders, masking)
- `snes/`: Optional LLE mode with full 65C816 CPU, SNES memory map, DMA, ICD2 bridge
- PPU writes 2-bit shades to `shade_buffer`; SGB remaps to palettes per 20x18 attribute grid

## Conventions

- Models are `GbModel` enum in `model.rs`. Use `model.is_cgb()` to check CGB/AGB, `model.is_sgb()` for SGB/SGB2
- Double-speed mode: `bus_cycles = cpu_cycles / 2` — Bus handles this in `tick_mcycle()`
- Snapshots (`snapshot.rs`) support rewind (VecDeque ring buffer) and save states (F5/F7, slots 1-9)
- OAM DMA is instant (0xA0 byte copy); HDMA mode 0 instant, mode 1 per-HBlank
- Boot ROMs are in `bootroms/` directory; test runner loads them with `--boot` flag
