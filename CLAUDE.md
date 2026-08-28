# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Game Boy / Game Boy Color emulator ("vibeboy") written in Rust (2024 edition). Supports DMG, DMG0, MGB, SGB, SGB2, CGB, and AGB (GBA in GBC mode) hardware models. Includes SGB Super Game Boy emulation with optional SNES LLE via a WDC 65C816 CPU.

## Build & Run

### Prerequisites

Rust 2024 edition, SDL3 >= 3.4, and `slangc` on PATH. Per-platform setup is in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

```bash
cargo build --release
cargo run --release -- path/to/rom.gbc

# WebAssembly browser build (requires wasm-pack + nightly toolchain)
PATH="$HOME/.rustup/toolchains/nightly-aarch64-apple-darwin/bin:$PATH" \
  wasm-pack build --target web --features web --no-default-features
# Serve web/index.html with any static file server

# With boot ROM and model override
cargo run --release -- path/to/rom.gbc --model dmg --bootrom bootroms/dmg_boot.bin

# Kopf-Lischinski pixel-art vectorization (6-stage GPU pipeline with CPU fallback)
cargo run --release -- path/to/rom.gbc --filter vectorize
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

# Vectorize a standalone PNG image
cargo run --release --bin test_runner -- vectorize input.png --out output.svg
cargo run --release --bin test_runner -- vectorize input.png --out output.png --scale 8 --gpu
cargo run --release --bin test_runner -- vectorize input.png --out output.png --scale 8 --cpu-filter

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
- **Bus** (`bus/mod.rs`) owns all subsystems and implements the memory map. `tick_mcycle()` steps Timer, PPU, APU, Joypad, OAM DMA, and HDMA each M-cycle
- **PPU** (`ppu/mod.rs`) is a pixel FIFO renderer ticked 1 T-cycle at a time internally via `step(4)`. VRAM and OAM live in the Ppu struct; Bus delegates access. DMG models use classic green LCD palette (`DMG_SHADES`), MGB uses grayscale (`MGB_SHADES`).
- **APU** (`apu/mod.rs`) uses a DIV-coupled frame sequencer; Bus detects DIV falling edges and calls `apu.div_event()`

### PPU timing model
- DMG line-start has a 5-dot state machine (`line_start_pending`, dots 1-5) that delays `visible_ly`, `ly_for_comparison`, and `mode_for_interrupt` transitions to match hardware-accurate timing
- Mode transitions happen internally before STAT register bits update (1T delay)
- `oam_bug_row` captures `accessed_oam_row` at end of `step()` for CPU-side OAM corruption checks

### Memory ownership
- VRAM (2 banks), OAM (160 bytes) -> owned by `Ppu`
- WRAM (8 banks), HRAM, IO registers -> owned by `Bus`
- Cart ROM/RAM -> owned by `Cartridge` trait objects in `Bus`. MBC5+Rumble supported with haptic feedback across SDL (set_rumble), Cocoa (CoreHaptics), and web (vibrationActuator).

### SGB subsystem
- `sgb.rs`: HLE command processing (palettes, attributes, borders, masking)
- `snes/`: Optional LLE mode with full 65C816 CPU (`cpu.rs`), SNES memory map (`bus.rs`), DMA (`dma.rs`), PPU registers (`ppu_regs.rs`), ICD2 bridge (`icd2.rs`)
- PPU writes 2-bit shades to `shade_buffer`; SGB remaps to palettes per 20x18 attribute grid

### Vectorization (`src/scaling/vectorize.rs`)
Kopf-Lischinski pixel-art vectorization pipeline ([paper](https://johanneskopf.de/publications/pixelart/)). CPU implementation that is a line-for-line faithful translation of the 6-stage GPU compute shaders — output is pixel-identical. Implementation aligned with the [GPU reference implementation](https://github.com/falichs/Depixelizing-Pixel-Art-on-GPUs).

Pipeline stages: `build_similarity_graph() -> resolve_crossings() -> build_cell_graph() -> update_tjunctions() -> optimize_energy() -> rasterize()`

- SVG export: `test_runner/gpu_svg.rs` (only used by test runner screenshot/vectorize commands)

### Scaling filter infrastructure (`src/scaling/`)
- `mod.rs`: `ScaleFilter` enum with `from_name()`, `validate_name()`, `ALL_NAMES` for centralized CLI parsing. `cpu_scale()` dispatcher for all CPU-side filters. 35 filter entries across 20 filter modules: `nearest_aa`, `bicubic`, `bilinear`, `dcci`, `eagle`, `edi`, `epx`, `hqx`, `lcd_grid`, `mmpx`, `nedi`, `omniscale`, `omniscale_legacy`, `sai`, `scale3x`, `scalefx`, `super_xbr`, `vectorize`, `xbr`, `xbrz`. Available on all platforms.
- `sdl/pipelines.rs`: `GpuPipelines` struct encapsulating all SDL3 GPU resources (device, textures, transfer buffers, compute pipelines). Lazy pipeline initialization via `ensure_pipeline()`. Render dispatch via `render_mode()` -> `GpuRenderMode` enum (`Native`, `ScaleCompute`, `FullGpuVectorize`, `Cpu`).
- `sdl/compute.rs`: SDL3 GPU compute shader dispatch helpers.
- `wgpu_vectorize.rs`: `WgpuVectorizePipeline` -- full 6-stage GPU vectorize pipeline using wgpu (WebGPU-compatible). Loads WGSL shaders (cross-compiled from Slang via `slangc` at build time). Cached bind groups, single-encoder submit, `encode()` API for external command encoder integration. Uses `ShaderRuntimeChecks::unchecked()` to avoid per-access bounds checks in the rasterizer hot path.

### Clock abstraction (`src/clock.rs`)
`Clock` trait provides wall-clock time to RTC cartridges (MBC3, HuC3, TAMA5). The core emulator never reads the system clock directly — frontends inject a `SystemClock` (native) or `JsClock` (wasm) via `Arc<dyn Clock>`.

### Printer (`src/printer.rs`)
Game Boy Printer implementation. All prints are queued as RGBA pixel data in memory via `has_pending_print()`/`take_print()`. Frontends poll and save to disk (native) or offer download (web).

### Save states (`src/savestate.rs`, `src/snapshot.rs`)
- `snapshot.rs`: `Snapshot` structs (serde-serializable) for all emulator state, rewind with reverse-delta compression (~10-minute capacity at ~21MB). Rewind plays at 3x speed with reverse audio.
- `savestate.rs`: Serialization via serde + bincode with magic header (`VIBEBOY\0`), layout version hash for compatibility detection. Produces/consumes `Vec<u8>` that frontends write as `rom.N.ss` files (native) or localStorage (web).

### Frontends (`src/frontends/`)

**SDL3 frontend** (`src/frontends/sdl/`):
- `main.rs`: SDL3 window loop, audio callback, input handling, file dialog. Supports `--runahead N` for reduced input latency and `--completions zsh/bash/fish/powershell` for shell completions (via clap_complete).
- `render.rs`: GPU rendering via `GpuPipelines` (SDL3 GPU API)
- `input.rs`: Keyboard/gamepad input mapping. Backspace=Rewind, Tab=Fast-forward, gamepad L1=Rewind, R1=Fast-forward.
- `camera.rs`: SDL3 webcam capture for Game Boy Camera
- `accel.rs`: Accelerometer input for MBC7
- Rumble: MBC5+Rumble cartridge support with gamepad haptic feedback (SDL set_rumble)

**Cocoa frontend** (`src/frontends/cocoa/`):
- `main.rs`: Native macOS Cocoa event loop, Metal rendering. Uses logical points for Metal drawable size (not Retina backing pixels). CoreHaptics rumble support for MBC5+Rumble.
- `metal_renderer.rs`: Metal GPU compute pipeline for all filters including the 6-stage vectorize pipeline
- `vectorize_metal.rs`: `MetalVectorizePipeline` -- Metal-native full GPU vectorize (similarity graph through rasterization)
- `menu.rs`: Native macOS menu bar (File, Emulation, Filter, Help)
- `audio.rs`: CoreAudio output
- `camera.rs`: AVFoundation webcam capture
- `gamepad.rs`: Game Controller framework input
- `controls.rs`, `font.rs`, `persistence.rs`, `accel.rs`: Input, OSD font, settings, accelerometer

**Winit frontend** (`src/frontends/winit/`):
- `main.rs`: Cross-platform winit/wgpu window with menus, file dialog, filter selection
- `app.rs`: Application state and event handling
- `gpu.rs`: wgpu rendering pipeline
- `audio.rs`: Audio output
- `camera.rs`: Webcam capture
- `menu.rs`: Native menu integration

**GTK4 frontend** (`src/frontends/gtk/`):
- `main.rs`: GTK4 window with menus, file dialog, filter selection, gamepad, printer
- `gpu.rs`: GLArea/glow OpenGL rendering
- `compute.rs`: wgpu GLES backend for GPU compute filters (Linux only)
- `audio.rs`: Audio output

**WebAssembly frontend** (`src/frontends/web/mod.rs`, `web/`):
- `mod.rs`: `WasmEmulator` struct with wasm-bindgen exports -- constructor from ROM bytes, `step_frame()`, `render_gpu()` for WebGPU, `init_gpu()` async initialization, camera/printer/accelerometer/rumble support, `save_data()`/`load_save()` for localStorage persistence.
- `web/index.html`: Markup with loading overlay, ROM selector, touch controls
- `web/style.css`: Responsive styles, mobile breakpoints, touch control layout, toast animations
- `web/emu.js`: ES module with state management, lazy wasm loading, frame loop, keyboard/gamepad input, audio (AudioWorklet at 96kHz), save states, toast notifications, `requestIdleCallback` save flushing
- `web/touch.js`: Multi-touch gamepad controls with per-identifier tracking
- `web/audio-processor.js`: Standalone AudioWorklet processor with buffer cap

**libretro frontend** (`src/frontends/libretro/mod.rs`):
- Full libretro API implementation for RetroArch compatibility
- XRGB8888 video, 48kHz stereo audio (downsampled from 96kHz)
- Save RAM persistence with RTC state (MBC3/HuC3/TAMA5 timestamps)
- Core option for hardware model selection
- Boot ROM auto-detection from RetroArch system directory

### GPU shaders (`src/shaders/`)

All shaders are compute shaders authored in [Slang](https://github.com/shader-slang/slang). All scaling filters use compute pipelines.

**Compute shaders (scaling filters):**
- `nearest.slang`, `nearest_aa.slang`, `bilinear.slang`, `bicubic.slang`, `dcci.slang`, `eagle.slang`, `edi.slang`, `epx.slang`, `hqx.slang`, `lcd_grid.slang`, `mmpx.slang`, `nedi.slang`, `omniscale.slang`, `omniscale_legacy.slang`, `sai2x.slang`, `super_sai2x.slang`, `super_eagle.slang`, `scale3x.slang`, `scalefx.slang`, `super_xbr.slang`, `xbr.slang`, `xbrz.slang`: GPU compute versions of the pixel scaling filters

**Compute shaders (full GPU vectorize pipeline):**
- `similarity_graph.slang`: Builds (2W+1)x(2H+1) connectivity graph with binary color matching
- `resolve_crossings.slang`: Diagonal crossing resolution with curves/islands/sparse heuristics (ties keep both)
- `cell_graph.slang`: Creates B-spline control points at grid corners, T-junction merging and position correction, corner detection with `DONT_OPTIMIZE_*` flags
- `update_tjunction.slang`: T-junction position update pass
- `optimize_energy.slang`: Double-buffered gradient descent optimizer -- kappa^2 smoothness + (2.5d)^4 positional energy, max move 0.25px
- `cell_rasterizer.slang`: Renders optimized B-spline curves to final output

**Shader cross-compilation (`build.rs`):**

All shaders are authored in Slang and cross-compiled at build time via `slangc` to multiple backend formats:
1. Slang -> SPIR-V (`-target spirv`, Vulkan/SDL3 backend)
2. Slang -> MSL (`-target metal`, Metal backend, macOS)
3. Slang -> DXIL (`-target dxil`, Direct3D 12 backend, Windows) -- requires `dxc`. Slang source files use explicit `register(tN,space0)` / `register(uN,space1)` / `register(bN,space2)` annotations matching SDL3 D3D12's type-based space grouping
4. Slang -> WGSL (`-target wgsl`, WebGPU backend, browser/wgpu)

Shared shader modules live in `src/shaders/modules/` and are imported via `import modules.color;` etc. Runtime shader loading tries SPIR-V first, then DXIL, then MSL. DXIL files are empty stubs on non-Windows builds so `include_bytes!` always compiles. WGSL files are loaded via `include_str!` for wgpu/WebGPU backends.

**Shader recompilation gotcha:** Editing `.slang` source files may not trigger a rebuild due to cargo's incremental compilation caching the build script output. Run `cargo clean -p vibeboy --release` to force `build.rs` to re-run `slangc`. Verify with `grep` on `.metal` files in `target/release/build/vibeboy-*/out/`.

## Tools & Scripts

### Disassemblers (`tools/`)

#### `tools/dis_sm83.py` -- SM83 (Game Boy CPU) Disassembler

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

#### `tools/dis65816.py` -- WDC 65C816 (SNES CPU) Disassembler

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

#### `scripts/fetch-test-roms.sh` -- Download Test ROM Suite

Downloads the c-sp/game-boy-test-roms v7.0 release from GitHub and unpacks it into the `game-boy-test-roms/` directory. Will not overwrite an existing directory.

```bash
./scripts/fetch-test-roms.sh
```

#### `scripts/bundle_app.sh` -- Build macOS Application Bundle

Builds the `vibeboy_cocoa` binary in release mode and packages it into a `VibeBoy.app` macOS application bundle under `target/VibeBoy.app`. Copies the binary, `Info.plist`, and app icon (`resources/AppIcon.icns`) into the bundle structure.

```bash
./scripts/bundle_app.sh

# Then run or install:
open target/VibeBoy.app
cp -r target/VibeBoy.app /Applications/
```

#### `scripts/generate_icon.py` -- Generate App Icon

Generates the VibeBoy macOS app icon (a stylized Game Boy Color) at all required sizes (16x16 through 1024x1024), saves them as an `.iconset`, and converts to `.icns` using `iconutil`. Requires the Python `Pillow` library. Output goes to `resources/AppIcon.icns`.

```bash
pip install Pillow  # if not already installed
python3 scripts/generate_icon.py
```

#### `scripts/vectorize_comparison.sh` -- Vectorize Comparison Test Suite

Downloads all 54 input sprites and the paper's 8x results from the Kopf-Lischinski supplementary page, then runs our CPU and GPU vectorizers on each for side-by-side comparison. Generates an HTML page.

```bash
./scripts/vectorize_comparison.sh          # skip existing outputs
./scripts/vectorize_comparison.sh --force  # re-render all
open vectorize-tests/comparison.html
```

### Binaries

The project produces five native binaries, a WebAssembly library, and a libretro core:

- **`vibeboy`** (`src/frontends/sdl/main.rs`) -- Main emulator with SDL3 window, audio, and input handling
- **`vibeboy_cocoa`** (`src/frontends/cocoa/main.rs`) -- Native macOS Cocoa/Metal UI frontend (requires `macos-ui` feature)
- **`vibeboy_winit`** (`src/frontends/winit/main.rs`) -- Cross-platform winit/wgpu UI frontend (requires `winit-ui` feature)
- **`vibeboy_gtk`** (`src/frontends/gtk/main.rs`) -- GTK4 UI frontend (requires `gtk-ui` feature, GPU compute on Linux)
- **`test_runner`** (`src/test_runner/main.rs`) -- Headless test ROM runner and vectorize tool
- **WebAssembly** (`src/frontends/web/`) -- Browser frontend via wasm-bindgen (requires `web` feature). Deployed to GitHub Pages.
- **libretro** (`src/frontends/libretro/`) -- RetroArch-compatible core (requires `libretro` feature). Built as cdylib.

## Conventions

- Models are `GbModel` enum in `model.rs`. Use `model.is_cgb()` to check CGB/AGB, `model.is_sgb()` for SGB/SGB2
- Double-speed mode: `bus_cycles = cpu_cycles / 2` -- Bus handles this in `tick_mcycle()`
- Snapshots (`snapshot.rs`) support rewind (reverse-delta compression, ~10-minute capacity at ~21MB, 3x playback with reverse audio) and save states (F5/F7, slots 0-9). Save states serialized via serde + bincode (`savestate.rs`), saved as `rom.N.ss` files on disk or localStorage in web.
- Fast-forward audio: all frontends downsample 4x audio through a Blackman-windowed sinc FIR filter. Rewind has reverse audio with the same filter.
- OAM DMA is instant (0xA0 byte copy); HDMA mode 0 instant, mode 1 per-HBlank
- Boot ROMs are in `bootroms/` directory; test runner loads them with `--boot` flag
- DMG models use classic green Game Boy LCD palette (`DMG_SHADES`: `#9BBC0F`, `#8BAC0F`, `#306230`, `#0F380F`). MGB uses grayscale (`MGB_SHADES`: `#C4CFA1`, `#8B956D`, `#4D533C`, `#1F1F1F`).
- The core emulator has no I/O, filesystem, or platform dependencies. Time is injected via the `Clock` trait (`src/clock.rs`). Frontends handle rendering, audio, input, and persistence.
- Pure utility functions (audio processing, model detection, frame timing) in `src/util.rs`. Frontend-specific I/O helpers in `src/ui_util.rs`.
