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

# With vectorized scaling filter
cargo run --release -- path/to/rom.gbc --filter vectorize

# Vectorize variants
cargo run --release -- path/to/rom.gbc --filter vectorize-adaptive
cargo run --release -- path/to/rom.gbc --filter vectorize-diffusion
cargo run --release -- path/to/rom.gbc --filter vectorize-spline-diffusion
cargo run --release -- path/to/rom.gbc --filter vectorize-spline-diffusion-adaptive

# With YUV visible edge threshold (paper's approach, can cause artifacts)
cargo run --release -- path/to/rom.gbc --filter vectorize --yuv-edges
```

## Testing

Test ROMs live in `game-boy-test-roms/` (c-sp/game-boy-test-roms v7.0, includes blargg, mooneye, gambatte, bully, and more). The `test_runner` binary runs them:

```bash
# Mooneye tests (breakpoint detection, Fibonacci register check)
cargo run --release --bin test_runner -- test mooneye game-boy-test-roms/mooneye-test-suite/acceptance/

# Blargg tests (serial output detection)
cargo run --release --bin test_runner -- test blargg game-boy-test-roms/blargg/

# Gambatte tests (hex output comparison, 15-frame capture)
cargo run --release --bin test_runner -- test gambatte game-boy-test-roms/gambatte/

# Gambatte subcategory
cargo run --release --bin test_runner -- test gambatte game-boy-test-roms/gambatte/sprites/

# Single test
cargo run --release --bin test_runner -- test blargg game-boy-test-roms/blargg/cpu_instrs/individual/01-special.gb

# Screenshot a ROM after N frames
cargo run --release --bin test_runner -- screenshot path/to/rom.gb --frames 300 --out screenshot.png

# Vectorize a frame to SVG
cargo run --release --bin test_runner -- screenshot path/to/rom.gb --frames 300 --out screenshot.svg

# Vectorize and rasterize at 4x scale
cargo run --release --bin test_runner -- screenshot path/to/rom.gb --frames 300 --out screenshot.png --format raster --scale 4

# Spline-diffusion rasterizer (paper's rendering)
cargo run --release --bin test_runner -- screenshot path/to/rom.gb --frames 300 --out screenshot.png --format spline-diffusion --scale 4

# Vectorize a standalone PNG image
cargo run --release --bin test_runner -- vectorize input.png --out output.svg
cargo run --release --bin test_runner -- vectorize input.png --out output.png --format raster --scale 8
cargo run --release --bin test_runner -- vectorize input.png --out output.png --format spline-diffusion --scale 8
cargo run --release --bin test_runner -- vectorize input.png --out output.png --format spline-diffusion --scale 8 --gpu

# Force a specific model
cargo run --release --bin test_runner -- test mooneye game-boy-test-roms/mooneye-test-suite/acceptance/ --model dmg

# Run with boot ROM
cargo run --release --bin test_runner -- test mooneye game-boy-test-roms/mooneye-test-suite/acceptance/ --boot

# Verbose output (extra diagnostics per test)
cargo run --release --bin test_runner -- test mooneye game-boy-test-roms/mooneye-test-suite/acceptance/ --verbose

# Quiet mode (summary only)
cargo run --release --bin test_runner -- test blargg game-boy-test-roms/blargg/ --quiet
```

Test runner auto-detects hardware model from filename suffixes (`-dmgABCmgb`, `-sgb2`, `-GS`, `-A`, etc.) and from the CGB cart header flag. Gambatte tests encode expected hex output in filenames after `_out` (e.g. `_out3` expects "3"). DMG tests have `dmg08` in the name, CGB tests have `cgb04c`.

**Current test status:** 75/75 mooneye acceptance, 57/58 blargg (oam_bug test 7 hangs), 55/70 SameSuite APU.

## Architecture

The emulator loop is: `Emulator::step_frame()` calls `Cpu::step()` which executes one instruction, returning T-cycles consumed. `Bus::tick_mcycle()` advances all subsystems (PPU, APU, Timer, Serial) by 4 T-cycles (one M-cycle).

### Key data flow
- **CPU** (`cpu/mod.rs`) executes SM83 opcodes, calls `bus.tick_mcycle()` between M-cycles, reads/writes memory via `bus.read_byte()`/`bus.write_byte()`
- **Bus** (`bus.rs`) owns all subsystems and implements the memory map. `tick_mcycle()` steps Timer, PPU, APU, Joypad, OAM DMA, and HDMA each M-cycle
- **PPU** (`ppu/mod.rs`) is a pixel FIFO renderer ticked 1 T-cycle at a time internally via `step(4)`. VRAM and OAM live in the Ppu struct; Bus delegates access
- **APU** (`apu.rs`) uses a DIV-coupled frame sequencer; Bus detects DIV falling edges and calls `apu.div_event()`

### PPU timing model
- DMG line-start has a 5-dot state machine (`line_start_pending`, dots 1-5) that delays `visible_ly`, `ly_for_comparison`, and `mode_for_interrupt` transitions to match hardware-accurate timing
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

### Vectorization subsystem (`src/vectorize/`)
Kopf-Lischinski pixel-art vectorization pipeline ([paper](https://johanneskopf.de/publications/pixelart/)). Converts frame buffers into smooth vector paths, then rasterizes at any scale with anti-aliased edges. Implementation aligned with the [GPU reference implementation](https://github.com/falichs/Depixelizing-Pixel-Art-on-GPUs).

Pipeline: `pixels → graph::build → contour::extract_cells_smooth → rasterize`

- `mod.rs`: Public API (`vectorize_to_svg`, `vectorize_to_raster`), `VectorizeCache` for frame caching, upscale detection/collapse, background color detection. No color quantization (removed — the paper doesn't use it).
- `graph.rs`: Similarity graph — YUV per-channel thresholds (48/7/6 per 255), diagonal crossing resolution with curves/islands/sparse heuristics. Ties keep both diagonals (matches reference, not paper).
- `voronoi.rs`: Voronoi cell corner reshaping at diagonal crossings (±0.25 pixel offsets)
- `contour.rs`: Core pipeline stages:
  - 81-entry compile-time Voronoi cell template table (3^4 corner states)
  - Sort-merge boundary edge deduplication (cache-friendly, ~1.9× faster than HashMap)
  - Chain construction with inline cpair valence, T-junction merging (shading/contour classification via YUV Euclidean distance ≤ 100/255)
  - T-junction position correction (`0.125*p0 + 0.75*p1 + 0.125*p2`)
  - Planar face algorithm for boundary loop tracing (flat sorted adjacency)
  - Gradient descent optimizer with κ² smoothness energy, (2.5×distance)⁴ positional energy
  - ×4 grid corner detection (angle ≥ 60°), corners excluded from curvature energy
  - VOID_COLOR sentinel (0x01000000) for image border edges
  - `VectorizeState` for split-phase optimization (CPU or GPU)
- `svg.rs`: Serializes paths to SVG document string
- `rasterize.rs`: Three rasterizers:
  - **Scanline** (`rasterize`/`rasterize_scaled`): 2×2 supersampling, nonzero winding. Default for `--filter vectorize`.
  - **Voronoi diffusion** (`rasterize_diffusion`): Gaussian blending (σ≈0.63) with graph-based region connectivity. For `--filter vectorize-diffusion`.
  - **Spline diffusion** (`rasterize_spline_diffusion`): B-spline contour boundaries + Gaussian blending with flood-fill connected-component regions. For `--filter vectorize-spline-diffusion`.

### GPU vectorize shaders (`src/shaders/`)
- `vectorize_raster.comp`: Scanline rasterizer (existing, for `--filter vectorize` GPU path)
- `vectorize_to_buf.comp`: Scanline rasterizer variant writing to storage buffer (pass 1 of spline-diffusion)
- `spline_diffusion.comp`: Gaussian diffusion with 2×2 supersampling (pass 2 of spline-diffusion)
- `diffusion_raster.comp`: Voronoi diffusion with diagonal state ownership computation
- `optimize_energy.comp`: Double-buffered spline optimizer with ping-pong (on `gpu-optimizer` branch)

## Tools & Scripts

### Disassemblers (`tools/`)

#### `tools/dis_sm83.py` — SM83 (Game Boy CPU) Disassembler

Disassembles Game Boy ROM files. Supports all SM83 opcodes including CB-prefixed bit operations. Can disassemble at arbitrary ROM offsets or GB addresses, search for byte patterns, hex dump, and display cartridge header info.

```bash
# Show cartridge header + entry point (default with no flags)
python3 tools/dis_sm83.py path/to/rom.gb

# Disassemble 50 instructions starting at GB address $0150
python3 tools/dis_sm83.py path/to/rom.gb --pc 0150 -n 50

# Disassemble from a raw ROM file offset
python3 tools/dis_sm83.py path/to/rom.gb --offset 4037 -n 20

# Hex dump 512 bytes at address $C000
python3 tools/dis_sm83.py path/to/rom.gb --pc C000 --hex --hex-len 512

# Search for a byte pattern (e.g. CP $0C instruction = FE 0C)
python3 tools/dis_sm83.py path/to/rom.gb --search FE0C --context 8

# Show cartridge header info
python3 tools/dis_sm83.py path/to/rom.gb --header
```

#### `tools/dis65816.py` — WDC 65C816 (SNES CPU) Disassembler

Disassembles SNES ROM files, primarily for SGB BIOS analysis. Automatically tracks M/X processor flag state through REP/SEP instructions to correctly decode 8-bit vs 16-bit immediate operands. Uses LoROM address mapping.

```bash
# Show interrupt vectors + reset handler (default with no flags)
python3 tools/dis65816.py sgb1.program.rom

# Disassemble 60 instructions starting at PC address $BF4A
python3 tools/dis65816.py sgb1.program.rom --pc BF4A -n 60

# Start in 16-bit accumulator/index mode
python3 tools/dis65816.py sgb1.program.rom --pc 8000 --m16 --x16

# Show interrupt vectors
python3 tools/dis65816.py sgb1.program.rom --vectors

# Search for a byte pattern
python3 tools/dis65816.py sgb1.program.rom --search 8D0042 --context 5
```

### Scripts (`scripts/`)

#### `scripts/fetch-test-roms.sh` — Download Test ROM Suite

Downloads the c-sp/game-boy-test-roms v7.0 release from GitHub and unpacks it into the `game-boy-test-roms/` directory. Will not overwrite an existing directory.

```bash
./scripts/fetch-test-roms.sh
```

#### `scripts/bundle_app.sh` — Build macOS Application Bundle

Builds the `vibeboy_cocoa` binary in release mode and packages it into a `VibeBoy.app` macOS application bundle under `target/VibeBoy.app`. Copies the binary, `Info.plist`, and app icon (`resources/AppIcon.icns`) into the bundle structure.

```bash
./scripts/bundle_app.sh

# Then run or install:
open target/VibeBoy.app
cp -r target/VibeBoy.app /Applications/
```

#### `scripts/generate_icon.py` — Generate App Icon

Generates the VibeBoy macOS app icon (a stylized Game Boy Color) at all required sizes (16x16 through 1024x1024), saves them as an `.iconset`, and converts to `.icns` using `iconutil`. Requires the Python `Pillow` library. Output goes to `resources/AppIcon.icns`.

```bash
pip install Pillow  # if not already installed
python3 scripts/generate_icon.py
```

#### `scripts/vectorize_comparison.sh` — Vectorize Comparison Test Suite

Downloads all 54 input sprites and the paper's 8× results from the Kopf-Lischinski supplementary page, then runs our scanline and spline-diffusion rasterizers (CPU and GPU) on each for side-by-side comparison. Generates an HTML page.

```bash
./scripts/vectorize_comparison.sh          # skip existing outputs
./scripts/vectorize_comparison.sh --force  # re-render all
open vectorize-tests/comparison.html
```

### Binaries

The project produces three binaries (defined in `Cargo.toml`):

- **`vibeboy`** (`src/main.rs`) — Main emulator with SDL3 window, audio, and input handling
- **`vibeboy_cocoa`** (`src/cocoa_ui.rs`) — Native macOS Cocoa/Metal UI frontend (used by `bundle_app.sh`)
- **`test_runner`** (`src/test_runner.rs`) — Headless test ROM runner with multiple test harness modes (mooneye, blargg, gambatte, screenshot). See the [Testing](#testing) section for usage

## Conventions

- Models are `GbModel` enum in `model.rs`. Use `model.is_cgb()` to check CGB/AGB, `model.is_sgb()` for SGB/SGB2
- Double-speed mode: `bus_cycles = cpu_cycles / 2` — Bus handles this in `tick_mcycle()`
- Snapshots (`snapshot.rs`) support rewind (VecDeque ring buffer) and save states (F5/F7, slots 1-9)
- OAM DMA is instant (0xA0 byte copy); HDMA mode 0 instant, mode 1 per-HBlank
- Boot ROMs are in `bootroms/` directory; test runner loads them with `--boot` flag
