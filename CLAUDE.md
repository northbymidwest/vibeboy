# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Game Boy / Game Boy Color emulator ("vibeboy") written in Rust (2024 edition). Supports DMG, DMG0, MGB, SGB, SGB2, CGB, and AGB (GBA in GBC mode) hardware models. Includes SGB Super Game Boy emulation with optional SNES LLE via a WDC 65C816 CPU.

## Build & Run

### Prerequisites

**macOS:**
- Rust toolchain, SDL3 (via Homebrew: `brew install sdl3`)
- For GPU shaders: `brew install shaderc spirv-cross`

**Windows:**
- Rust toolchain (MSVC), Visual Studio Build Tools
- [LunarG Vulkan SDK](https://vulkan.lunarg.com/) — provides `glslc`, `spirv-cross`, and `dxc` for shader compilation
- SDL3 development libraries: download the [SDL3-devel-VC](https://github.com/libsdl-org/SDL/releases) zip, place `SDL3.lib` and `SDL3.dll` in the project's `lib/` directory. The `.cargo/config.toml` already points the linker there. Copy `SDL3.dll` next to the built `.exe` (or add `lib/` to your PATH).

**Linux:**
- Rust toolchain, SDL3 dev package, `glslc` or `glslangValidator`

```bash
cargo build --release
cargo run --release -- path/to/rom.gbc

# WebAssembly browser build (requires wasm-pack + nightly toolchain)
PATH="$HOME/.rustup/toolchains/nightly-aarch64-apple-darwin/bin:$PATH" \
  wasm-pack build --target web --features web --no-default-features
# Serve web/index.html with any static file server

# With boot ROM and model override
cargo run --release -- path/to/rom.gbc --model dmg --bootrom bootroms/dmg_boot.bin

# With vectorized scaling filter (shared-chain gap-free rendering)
cargo run --release -- path/to/rom.gbc --filter vectorize

# Vectorize variants
cargo run --release -- path/to/rom.gbc --filter vectorize-adaptive
cargo run --release -- path/to/rom.gbc --filter vectorize-diffusion
cargo run --release -- path/to/rom.gbc --filter vectorize-spline-diffusion
cargo run --release -- path/to/rom.gbc --filter vectorize-spline-diffusion-adaptive

# Legacy vectorize (original scanline rasterizer)
cargo run --release -- path/to/rom.gbc --filter vectorize-legacy
cargo run --release -- path/to/rom.gbc --filter vectorize-legacy-adaptive

# With YUV visible edge threshold (paper's approach, can cause artifacts)
cargo run --release -- path/to/rom.gbc --filter vectorize-legacy --yuv-edges
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
cargo run --release --bin test_runner -- vectorize input.png --out output.png --filter raster --scale 8
cargo run --release --bin test_runner -- vectorize input.png --out output.png --filter spline-diffusion --scale 8
cargo run --release --bin test_runner -- vectorize input.png --out output.png --filter spline-diffusion --scale 8 --gpu
cargo run --release --bin test_runner -- vectorize input.png --out output.png --filter spline-diffusion --scale 8 --cpu-filter

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
- **PPU** (`ppu/mod.rs`) is a pixel FIFO renderer ticked 1 T-cycle at a time internally via `step(4)`. VRAM and OAM live in the Ppu struct; Bus delegates access. DMG models use classic green LCD palette (`DMG_SHADES`).
- **APU** (`apu.rs`) uses a DIV-coupled frame sequencer; Bus detects DIV falling edges and calls `apu.div_event()`

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

### Vectorization subsystem (`src/vectorize/`)
Kopf-Lischinski pixel-art vectorization pipeline ([paper](https://johanneskopf.de/publications/pixelart/)). Converts frame buffers into smooth vector paths, then rasterizes at any scale with anti-aliased edges. Implementation aligned with the [GPU reference implementation](https://github.com/falichs/Depixelizing-Pixel-Art-on-GPUs).

Pipeline: `pixels -> graph::build -> contour::extract_cells_smooth -> rasterize`

- `mod.rs`: Public API (`vectorize_to_svg`, `vectorize_to_raster`), `VectorizeCache` (shared-chain) and `VectorizeLegacyCache` (original scanline) for frame caching, upscale detection/collapse, background color detection. No color quantization (removed -- the paper doesn't use it).
- `graph.rs`: Similarity graph -- YUV per-channel thresholds (48/7/6 per 255), diagonal crossing resolution with curves/islands/sparse heuristics. Ties keep both diagonals (matches reference, not paper).
- `voronoi.rs`: Voronoi cell corner reshaping at diagonal crossings (+/-0.25 pixel offsets)
- `contour/`: Core pipeline stages (split into submodules):
  - `cells.rs`: 81-entry compile-time Voronoi cell template table (3^4 corner states), per-pixel cell vertex precomputation
  - `edges.rs`: Boundary edge deduplication (FxHashMap), chain construction with inline cpair valence, T-junction merging (shading/contour classification via YUV Euclidean distance <= 100/255), T-junction position correction (`0.125*p0 + 0.75*p1 + 0.125*p2`)
  - `loops.rs`: Planar face algorithm for boundary loop tracing (flat sorted adjacency with cross-product angle ordering)
  - `optimize.rs`: Gradient descent optimizer with kappa^2 smoothness energy, (2.5x distance)^4 positional energy, x4 grid corner detection (angle >= 60 degrees), corners excluded from curvature energy
  - `mod.rs`: Orchestration, `VectorizeState` for split-phase optimization (CPU or GPU), VOID_COLOR sentinel (0x01000000) for image border edges
- `svg.rs`: Serializes paths to SVG document string (grouped by color, BTreeMap ordering)
- `rasterize/`: Three rasterizers (split into submodules):
  - `scanline.rs`: 2x2 supersampling, nonzero winding, recursive Bezier flattening (tolerance 0.25). Default for `--filter vectorize`.
  - `diffusion.rs`: Gaussian blending (sigma ~= 0.63, gauss_k=2.5, radius=2.0) with graph-based region connectivity via 8-connected flood fill. For `--filter vectorize-diffusion`.
  - `spline_diffusion.rs`: B-spline contour boundaries + Gaussian blending with flood-fill connected-component regions. For `--filter vectorize-spline-diffusion`.
  - `gpu.rs`: GPU rasterization wrappers (edge data upload, buffer management)
- `gpu_rasterize.rs`: GPU rasterization dispatch wrappers
- `rasterize.wgsl`: WebGPU compute shader for wgpu-based rasterization

### Scaling filter infrastructure (`src/scaling/`)
- `mod.rs`: `ScaleFilter` enum (41+ filter names) with `from_name()`, `validate_name()`, `ALL_NAMES` for centralized CLI parsing. `cpu_scale()` dispatcher for all CPU-side filters. 19 filter modules: `nearest_aa`, `bicubic`, `bilinear`, `dcci`, `eagle`, `edi`, `epx`, `hqx`, `lcd_grid`, `mmpx`, `nedi`, `omniscale`, `omniscale_legacy`, `sai`, `scale3x`, `super_xbr`, `vectorize_gpu`, `xbr`, `xbrz`. Available on all platforms (no longer gated behind `not(wasm32)`).
- `sdl/pipelines.rs`: `GpuPipelines` struct encapsulating all SDL3 GPU resources (device, textures, transfer buffers, compute pipelines). Lazy pipeline initialization via `ensure_pipeline()`. Render dispatch via `render_mode()` -> `GpuRenderMode` enum (`Native`, `ScaleCompute`, `Vectorize`, `Diffusion`, `SplineDiffusion`, `VectorizeSharedChain`, `FullGpuVectorize`, `Cpu`).
- `sdl/compute.rs`: SDL3 GPU compute shader dispatch helpers.
- `wgpu_vectorize.rs`: `WgpuVectorizePipeline` -- full 6-stage GPU vectorize pipeline using wgpu (WebGPU-compatible). Loads WGSL shaders (cross-compiled from GLSL via naga at build time). Cached bind groups, single-encoder submit, `encode()` API for external command encoder integration. Uses `ShaderRuntimeChecks::unchecked()` to avoid per-access bounds checks in the rasterizer hot path.

### Printer (`src/printer.rs`)
Unified Game Boy Printer implementation with `PrintOutput` enum:
- `PrintOutput::File { output_dir }` -- saves completed prints as PNG files to disk (used by native frontends)
- `PrintOutput::Memory` -- queues completed prints as RGBA pixel data in memory (used by WebAssembly frontend for browser download)

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
- `metal_renderer.rs`: Metal GPU compute pipeline for all filters including the 6-stage vectorize pipeline, GPU scanline rasterizer, diffusion rasterizer, and spline-diffusion 2-pass pipeline
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

**WebAssembly frontend** (`src/frontends/web/mod.rs`, `web/index.html`):
- `lib.rs`: Library crate re-exporting core emulator modules. Exposes `wgpu_vectorize` for the `web` feature. `scaling` module available on all platforms (no longer gated behind `not(wasm32)`).
- `mod.rs`: `WasmEmulator` struct with wasm-bindgen exports -- constructor from ROM bytes, `step_frame()`, zero-copy `frame_buffer_update()`/`frame_buffer_ptr()`, `render_gpu()` for WebGPU vectorize, `init_gpu()` async WebGPU initialization, `set_camera_image()`, printer support via downcasting, `save_data()`/`load_save()` for localStorage persistence. Rumble support via vibrationActuator.
- `web/index.html`: Browser UI with Canvas2D fallback, WebGPU rendering with all GPU filters (vectorize, OmniScale, HQx, xBR, xBRZ, Super xBR, EPX, Eagle, Scale3x, bicubic, AA nearest) via filter dropdown, model select dropdown, built-in ROM selector with public domain games, gamepad support (Gamepad API, L1=Rewind, R1=Fast-forward), accelerometer (DeviceMotion API for MBC7), frame-rate independent emulation (~59.73fps via time accumulator), AudioWorklet at native 96kHz with reverse/downsample audio processing, rewind (Backspace) and fast-forward (Tab), webcam for Game Boy Camera, drag-and-drop ROM loading, localStorage save persistence, favicon.

### GPU shaders (`src/shaders/`)

All shaders are compute shaders authored in GLSL 4.50. Fragment shaders have been removed; all scaling filters now use compute pipelines.

**Compute shaders (scaling filters):**
- `nearest_aa.comp`, `bicubic.comp`, `dcci.comp`, `eagle.comp`, `edi.comp`, `epx.comp`, `hqx.comp`, `lcd_grid.comp`, `mmpx.comp`, `nedi.comp`, `omniscale.comp`, `omniscale_legacy.comp`, `scale3x.comp`, `super_xbr.comp`, `xbr.comp`, `xbrz.comp`: GPU compute versions of the pixel scaling filters

**Compute shaders (rasterization):**
- `vectorize_raster.comp`: Scanline rasterizer with 2x2 supersampling, nonzero winding (for `--filter vectorize` GPU path)
- `vectorize_to_buf.comp`: Scanline rasterizer variant writing to storage buffer with no AA (pass 1 of spline-diffusion -- produces hard region boundaries)
- `spline_diffusion.comp`: Gaussian diffusion (gauss_k=2.5, radius=2.0) with 2x2 supersampling (pass 2 of spline-diffusion)
- `diffusion_raster.comp`: Voronoi diffusion with packed diagonal state ownership (2 bits per corner)

**Compute shaders (full GPU vectorize pipeline):**
- `similarity_graph.comp`: Builds (2W+1)x(2H+1) connectivity graph with binary color matching
- `resolve_crossings.comp`: Diagonal crossing resolution with curves/islands/sparse heuristics (ties keep both)
- `cell_graph.comp`: Creates B-spline control points at grid corners, T-junction merging and position correction, corner detection with `DONT_OPTIMIZE_*` flags
- `update_tjunction.comp`: T-junction position update pass
- `optimize_energy.comp`: Double-buffered gradient descent optimizer -- kappa^2 smoothness + (2.5d)^4 positional energy, max move 0.25px
- `cell_rasterizer.comp`: Renders optimized B-spline curves to final output

**Shader cross-compilation (`build.rs`):**

All shaders are authored in GLSL and cross-compiled at build time to multiple backend formats:
1. GLSL -> SPIR-V via `glslc` (Vulkan backend, all platforms)
2. SPIR-V -> MSL via `spirv-cross --msl` (Metal backend, macOS) -- requires per-shader `[[buffer(N)]]` index remapping (`msl_buffer_remap` in `ShaderInfo`) because MSL uses a single buffer namespace for all resource types
3. SPIR-V -> HLSL -> DXIL via `spirv-cross --hlsl` + `dxc` (Direct3D 12 backend, Windows) -- requires automated register/space normalization (`remap_hlsl_registers`) because spirv-cross assigns HLSL spaces based on SPIR-V descriptor sets, but SDL3 D3D12 expects type-based grouping: `t[n] space0` (SRVs), `u[n] space1` (UAVs), `b[n] space2` (CBVs)
4. SPIR-V -> WGSL via `naga` (WebGPU backend, browser/wgpu) -- naga is a Rust build dependency; converts at build time. Note: `cell_rasterizer.comp` has barriers restructured for WGSL uniformity rules (naga copies `workgroup_id` to a private variable, losing uniformity proof).

Runtime shader loading tries SPIR-V first, then DXIL, then MSL. DXIL files are empty stubs on non-Windows builds so `include_bytes!` always compiles. WGSL files are loaded via `include_str!` for wgpu/WebGPU backends.

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

Downloads all 54 input sprites and the paper's 8x results from the Kopf-Lischinski supplementary page, then runs our scanline and spline-diffusion rasterizers (CPU and GPU) on each for side-by-side comparison. Generates an HTML page.

```bash
./scripts/vectorize_comparison.sh          # skip existing outputs
./scripts/vectorize_comparison.sh --force  # re-render all
open vectorize-tests/comparison.html
```

### Binaries

The project produces four native binaries plus a WebAssembly library:

- **`vibeboy`** (`src/frontends/sdl/main.rs`) -- Main emulator with SDL3 window, audio, and input handling
- **`vibeboy_cocoa`** (`src/frontends/cocoa/main.rs`) -- Native macOS Cocoa/Metal UI frontend (requires `macos-ui` feature, used by `bundle_app.sh`)
- **`vibeboy_winit`** (`src/frontends/winit/main.rs`) -- Cross-platform winit/wgpu UI frontend (requires `winit-ui` feature, with menus, file dialog, filter selection)
- **`test_runner`** (`src/test_runner/main.rs`) -- Headless test ROM runner with multiple test harness modes (mooneye, blargg, gambatte, gbmicrotest, tearoom, screenshot). See the [Testing](#testing) section for usage
- **WebAssembly** (`src/lib.rs` + `src/frontends/web/mod.rs`) -- Browser frontend via wasm-bindgen (requires `web` feature). Builds to `pkg/vibeboy_bg.wasm` + JS glue. Served from `web/index.html`. Deployed to GitHub Pages via the `gh-pages` branch.

## Conventions

- Models are `GbModel` enum in `model.rs`. Use `model.is_cgb()` to check CGB/AGB, `model.is_sgb()` for SGB/SGB2
- Double-speed mode: `bus_cycles = cpu_cycles / 2` -- Bus handles this in `tick_mcycle()`
- Snapshots (`snapshot.rs`) support rewind (reverse-delta compression, ~10-minute capacity at ~21MB, 3x playback with reverse audio) and save states (F5/F7, slots 1-9). Save states serialized via serde + bincode (`savestate.rs`), saved as `rom.N.ss` files on disk or localStorage in web.
- Fast-forward audio: all frontends downsample 4x audio through a Blackman-windowed sinc FIR filter. Rewind has reverse audio with the same filter.
- OAM DMA is instant (0xA0 byte copy); HDMA mode 0 instant, mode 1 per-HBlank
- Boot ROMs are in `bootroms/` directory; test runner loads them with `--boot` flag
- DMG models use classic green Game Boy LCD palette (shades: `#9BBC0F`, `#8BAC0F`, `#306230`, `#0F380F`)
