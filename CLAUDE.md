# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Game Boy / Game Boy Color emulator ("vibeboy") written in Rust (2024 edition). Supports DMG, DMG0, MGB, SGB, SGB2, CGB, and AGB (GBA in GBC mode) hardware models. Includes SGB Super Game Boy emulation with optional SNES LLE via a WDC 65C816 CPU.

## Build & Run

### Prerequisites

Rust 1.98+ (2024 edition), SDL3 >= 3.4, and `slangc` on PATH. Per-platform setup is in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). With nix, `flake.nix` provides a dev shell (loaded by
direnv via `.envrc`, or `nix develop`) with SDL3, slang, GTK4, bindgen, wasm-pack, curl, unzip,
xxd and python3 (with Pillow); run cargo inside it, since some native libraries (e.g. libiconv on
macOS) only link from within the shell. The shell points `DEVELOPER_DIR`/`SDKROOT` at a nix Apple SDK, which breaks Apple's
`/usr/bin` tool shims (that is why it ships its own `python3`); run any other shim, such as
`xcrun`, with `env -u DEVELOPER_DIR -u SDKROOT`.

```bash
cargo build --release
cargo run --release -- path/to/rom.gbc

# WebAssembly browser build into web/pkg (requires wasm-pack); --roms also
# fetches the public-domain ROMs. .github/workflows/pages.yml runs the same
# script and deploys web/ to GitHub Pages; it is manual only
# (`gh workflow run pages.yml`).
./scripts/build-web.sh --roms
python3 -m http.server -d web 8080

# With boot ROM and model override
cargo run --release -- path/to/rom.gbc --model dmg --bootrom bootroms/dmg_boot.bin

# Kopf-Lischinski pixel-art vectorization (8-pass GPU pipeline with CPU fallback)
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

`test` exits 1 when any test fails, times out or errors (2 when the path holds no ROMs); `--allow-failures` keeps exit 0 for runs over suites with known failures. `scripts/accuracy.sh` runs every suite against the per-test baseline in `tests/accuracy-baseline.txt` (see below).

Test runner auto-detects hardware model from filename suffixes (`-dmgABCmgb`, `-sgb2`, `-GS`, `-A`, etc.) and from the CGB cart header flag. Gambatte tests encode expected hex output in filenames after `_out` (e.g. `_out3` expects "3"). DMG tests have `dmg08` in the name, CGB tests have `cgb04c`.

**Current test status** (`tests/accuracy-baseline.txt` has the per-test list): mooneye acceptance 75/75, emulator-only 26/28, misc 6/8; wilbertpol acceptance 94/105, misc 6/9; blargg 57/58 (oam_bug 7 times out); gambatte 2164/3077; same-suite 61/78 (APU 57/70); gbmicrotest 474/513; mealybug tearoom DMG 6/24, CGB 2/27.

Accuracy work is judged per test, not by totals: run `scripts/accuracy.sh` before and after a change and look at exactly which tests were gained and lost. CI runs `cargo test` in debug mode; large by-value structs can overflow the 2 MB test-thread stack there even when release tests pass, so keep big buffers boxed.

## Architecture

The emulator loop is: `Emulator::step_frame()` calls `run_one_frame()`, which loops `Emulator::step()` until the PPU enters VBlank (or, with the LCD off, until one frame's worth of dots has elapsed). `step()` drives the CPU one M-cycle at a time: `Cpu::mcycle()` returns an `McycleOp` saying what that M-cycle needs from the bus, and the emulator services it with `Bus::tick_read()` / `tick_write()` / `tick_internal()` (and the halt/speed-switch variants), each of which performs the access and advances every subsystem by one M-cycle.

### Key data flow
- **CPU** (`cpu/mod.rs`) is a pure M-cycle state machine that never touches the bus. `McycleOp` carries memory accesses, OAM-bug triggers (`ReadWithOamBug`, `InternalWithOamBug`, `DispatchOamBug`), interrupt dispatch (`DispatchWrite`, answered with `cpu.provide_vector()`), and instruction outcomes that need bus state (`HaltExecuted`, `StopExecuted`, `Locked`, `SpeedSwitchIdle`), answered through methods like `enter_halt()` and `resolve_stop()`. Only `regs` is public.
- **Bus** (`bus/mod.rs`) owns all subsystems and implements the memory map. Each M-cycle tick steps Timer, Serial, APU, Joypad, OAM DMA and HDMA; PPU dots are deferred and flushed before any access that could observe them (`flush_ppu_deferred()`), with per-register conflict handlers in `bus/io.rs` for writes that land mid-M-cycle
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
Kopf-Lischinski pixel-art vectorization pipeline ([paper](https://johanneskopf.de/publications/pixelart/)), aligned with the [GPU reference implementation](https://github.com/falichs/Depixelizing-Pixel-Art-on-GPUs). The CPU implementation mirrors the GPU compute passes stage for stage, but the output is not pixel-identical: `tests/filter_parity.rs` measures about 4% of pixels differing (max 58 levels) on its test frame. The optimizer parameters (`OPT_OUTER_PASSES`, `OPT_GRAD_ETA`, `OPT_GRAD_MAX_STEP` in `vectorize.rs`) are shared by the CPU and every GPU backend.

Pipeline stages (CPU): `build_similarity_graph() -> resolve_crossings() -> build_cell_graph() -> optimize_energy()` (Picard step + gradient correction, 3 outer passes) `-> update_tjunctions()` (T-junction snap + crossing parameters) `-> rasterize()`. The GPU runs 8 passes: similarity_graph, resolve_crossings, cell_graph, picard_step, gradient_correction, update_tjunction, crossing_pack, cell_rasterizer. `VBY_*` environment variables tune the CPU optimizer for experiments; they are read once per process.

- SVG export: `test_runner/gpu_svg.rs` (only used by test runner screenshot/vectorize commands)

### Scaling filter infrastructure (`src/scaling/`)
- `mod.rs`: `ScaleFilter` enum and the `REGISTRY` of `FilterInfo` entries (34 filters, plus `none` as a CLI alias for `nearest`), with `from_name()`, `validate_name()` and `all_names()` for CLI parsing and `cpu_scale()` dispatching every CPU filter. 20 CPU filter modules: `nearest_aa`, `bicubic`, `bilinear`, `dcci`, `eagle`, `edi`, `epx`, `hqx`, `lcd_grid`, `mmpx`, `nedi`, `omniscale`, `omniscale_legacy`, `sai`, `scale3x`, `scalefx`, `super_xbr`, `vectorize`, `xbr`, `xbrz`.
- The registry is also the single source of truth for GPU dispatch: `FilterInfo::gpu` is a `GpuShader { shader, pass, extra }` (shader module, `GpuPass::{Single, SuperXbr, ScaleFx}`, and the uniform `UniformExtra`), and `scale_shader_list!` is the one list of scale shader modules, which the SDL, wgpu and Metal backends each expand into their own bytecode tables. Adding a filter is one registry entry (plus a `scale_shader_list!` line and a `build.rs` `SHADERS` line for a new shader).
- `sdl/pipelines.rs`: `GpuPipelines` holds the SDL3 GPU device, textures and lazily created compute pipelines; `ensure_pipeline()` returns a `GpuRenderMode` (`ScaleCompute`, `FullGpuVectorize`, `Cpu`). `sdl/scale.rs` has the shared `init_scale_pipeline` / `encode_scale` used by the window and headless screenshot paths; scale buffers are cached and only re-created on size changes. `sdl/compute.rs` holds the SDL3 vectorize pipeline.
- `wgpu_scale.rs`: wgpu scale filters (web, winit, GTK) with cached per-filter pass resources. `wgpu_vectorize.rs`: `WgpuVectorizePipeline`, the full 8-pass GPU vectorize pipeline on wgpu (WebGPU-compatible), with cached bind groups, single-encoder submit and an `encode()` API for external command encoders. Uses `ShaderRuntimeChecks::unchecked()` to avoid per-access bounds checks in the rasterizer hot path.
- `tests/filter_parity.rs` compares every filter's CPU and wgpu GPU output with per-filter tolerances (`cargo test --release --features gpu --test filter_parity -- --ignored`).

### Clock abstraction (`src/clock.rs`)
`Clock` trait provides wall-clock time to RTC cartridges (MBC3, HuC3, TAMA5). The core emulator never reads the system clock directly; frontends inject a `SystemClock` (native) or `JsClock` (wasm) via `Arc<dyn Clock>`.

### Printer (`src/printer.rs`)
Game Boy Printer implementation. All prints are queued as RGBA pixel data in memory via `has_pending_print()`/`take_print()`. Frontends poll and save to disk (native) or offer download (web).

### Save states (`src/savestate.rs`, `src/snapshot.rs`)
- `snapshot.rs`: `Snapshot` structs (serde-serializable) for all emulator state. Cartridge state is the typed `CartState` enum (one variant per mapper, `cartridge/mod.rs`), part of the same encoding.
- `rewind.rs`: rewind buffer of reverse deltas between consecutive fixed-width-integer bincode encodings (fixed width keeps field offsets stable, so deltas stay small: roughly 0.2 to 1.5 KB per frame). 36,000 frames (~10 minutes) cost roughly 6 to 50 MB depending on the game. Rewind plays at 3x speed with reverse audio.
- `savestate.rs`: serde + bincode with header `VIBEBOY\0` + `FORMAT_VERSION` + payload length. `FORMAT_VERSION` is bumped by hand whenever the encoding changes; the `encoding_is_pinned_to_format_version` test hashes deterministic states for every mapper and fails on any encoding change, telling you to bump it (the hash covers the header, so re-read it after bumping). States from other versions are rejected; there is no backward compatibility. The format does not depend on pointer width, so native and web states are interchangeable.
- States from outside the process (files, libretro) go through `Emulator::restore_untrusted_snapshot()`, which validates the model, mapper type and every index-like field before applying; the in-memory rewind buffer skips validation. Frontends write states as `rom.N.ss` files (native, atomically) or localStorage (web).

### Frontends (`src/frontends/`)

The native frontends (SDL, Cocoa, winit, GTK) share `ui_util::Session`, which owns the emulator and implements everything frontend-independent: ROM load, reset and model change (each flushes the outgoing battery save first and keeps the printer and SGB LLE setup), the per-tick emulation (`tick(&HoldInputs, audio_queued)`: rewind 3x with reversed audio, fast-forward 4x, slow motion, pause and frame advance, and audio-queue-driven 0/1/2-frame pacing), save states, printer output and periodic battery-save flushing (`SavFlusher`, driven by `Emulator::save_generation()` so RTC ticking alone never rewrites the `.sav`). Frontends only map input, render and output audio. Audio goes through a lock-free stereo-frame ring (`ui_util::audio_ring`, on `rtrb`); `cpal_audio.rs` is the cpal output shared by winit and GTK.

**SDL3 frontend** (`src/frontends/sdl/`):
- `main.rs`: SDL3 window loop, audio callback, input handling, file dialog. Supports `--runahead N` for reduced input latency and `--completions zsh/bash/fish/powershell` for shell completions (via clap_complete).
- `render.rs`: GPU rendering via `GpuPipelines` (SDL3 GPU API)
- `input.rs`: Keyboard/gamepad input mapping. Backspace=Rewind, Tab=Fast-forward, gamepad L1=Rewind, R1=Fast-forward.
- `camera.rs`: SDL3 webcam capture for Game Boy Camera
- `accel.rs`: Accelerometer input for MBC7
- Rumble: MBC5+Rumble cartridge support with gamepad haptic feedback (SDL set_rumble)

**Cocoa frontend** (`src/frontends/cocoa/`):
- `main.rs`: Native macOS Cocoa event loop, Metal rendering. Uses logical points for Metal drawable size (not Retina backing pixels). CoreHaptics rumble support for MBC5+Rumble.
- `metal_renderer.rs`: Metal GPU compute pipeline for all filters (tables derived from the scaling registry)
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
- Audio: shared `src/cpal_audio.rs`
- `camera.rs`: Webcam capture
- `menu.rs`: Native menu integration

**GTK4 frontend** (`src/frontends/gtk/`):
- `main.rs`: GTK4 window with menus, file dialog, filter selection, gamepad, printer
- `gpu.rs`: GLArea/glow OpenGL rendering
- `compute.rs`: wgpu GLES backend for GPU compute filters (Linux only)
- Audio: shared `src/cpal_audio.rs`

**WebAssembly frontend** (`src/frontends/web/mod.rs`, `web/`):
- `mod.rs`: `WasmEmulator` struct with wasm-bindgen exports -- constructor from ROM bytes, `step_frame()`, `render_gpu()` for WebGPU, `init_gpu()` async initialization, camera/printer/accelerometer/rumble support, `save_data()`/`load_save()` for localStorage persistence.
- `web/index.html`: Markup with loading overlay, ROM selector, touch controls
- `web/style.css`: Responsive styles, mobile breakpoints, touch control layout, toast animations
- `web/emu.js`: ES module with state management, lazy wasm loading, frame loop, keyboard/gamepad input, audio (AudioWorklet at 96kHz), save states, toast notifications, `requestIdleCallback` save flushing
- `web/touch.js`: Multi-touch gamepad controls with per-identifier tracking
- `web/audio-processor.js`: Standalone AudioWorklet processor with buffer cap

**libretro frontend** (`src/frontends/libretro/mod.rs`):
- Full libretro API implementation for RetroArch compatibility
- XRGB8888 video, 48kHz stereo audio (the APU runs at 48kHz directly)
- Save RAM exposed through a stable `RETRO_MEMORY_SAVE_RAM` buffer synced in place, including RTC state (MBC3/HuC3/TAMA5 timestamps); reset keeps it
- `retro_serialize_size` reports a fixed upper bound (the frontend caches it for rewind and runahead)
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
- `picard_step.slang`: Per-control-point Newton (Picard) step on the local energy (curvature smoothness plus positional term)
- `gradient_correction.slang`: Global gradient step (`OPT_GRAD_ETA`, capped at `OPT_GRAD_MAX_STEP`) that removes the Picard fixed-point bias; the two alternate for `OPT_OUTER_PASSES` passes
- `update_tjunction.slang`: T-junction stem snap
- `crossing_pack.slang`: Crossing parameters for the rasterizer
- `cell_rasterizer.slang`: Renders optimized B-spline curves to final output

**Shader cross-compilation (`build.rs`):**

All shaders are authored in Slang and cross-compiled at build time via `slangc` to multiple backend formats:
1. Slang -> SPIR-V (`-target spirv`, Vulkan/SDL3 backend)
2. Slang -> MSL (`-target metal`, Metal backend, macOS)
3. Slang -> DXIL (`-target dxil`, Direct3D 12 backend, Windows) -- requires `dxc`. Slang source files use explicit `register(tN,space0)` / `register(uN,space1)` / `register(bN,space2)` annotations matching SDL3 D3D12's type-based space grouping
4. Slang -> WGSL (`-target wgsl`, WebGPU backend, browser/wgpu)

Shared shader modules live in `src/shaders/modules/` and are imported via `import modules.color;` etc. Runtime shader loading tries SPIR-V first, then DXIL, then MSL. DXIL files are empty stubs on non-Windows builds so `include_bytes!` always compiles. WGSL files are loaded via `include_str!` for wgpu/WebGPU backends.

**Debug env vars:** `VIBEBOY_SHADER_DEBUG=1` (build time) passes `-g` to `slangc` for shader debug info (RenderDoc, validation layers). `VIBEBOY_FORCE_VULKAN=1` (runtime, SDL frontend) requests only SPIR-V so SDL3 GPU uses the Vulkan backend on Windows.

**Shader rebuilds:** `build.rs` emits `rerun-if-changed` for every shader in `SHADERS` and every file in `modules/`, so editing a `.slang` file does recompile. Things that do not trigger a rebuild: a new `.slang` file not yet added to `SHADERS`, and a different `slangc` version on PATH (run `cargo clean -p vibeboy --release` after upgrading slang). Each feature combination has its own build-script `OUT_DIR` (`target/release/build/vibeboy-*/out/`), so when inspecting generated `.metal`/`.wgsl` files make sure you are looking at the one for the features you built.

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

#### `scripts/accuracy.sh` -- Accuracy Regression Check

Runs every test ROM suite (mooneye, wilbertpol, blargg, gambatte, same-suite, gbmicrotest, tearoom DMG and CGB) and compares each test's status against `tests/accuracy-baseline.txt`. Prints the tests gained and lost and exits 1 if any test that passed in the baseline no longer does. `.github/workflows/accuracy.yml` runs it on every push and PR. A change that trades tests must update the baseline in the same commit.

```bash
./scripts/accuracy.sh             # compare against the baseline
./scripts/accuracy.sh --update    # rewrite the baseline from this run
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

Generates the VibeBoy macOS app icon (a stylized Game Boy Color) at all required sizes (16x16 through 1024x1024), saves them as an `.iconset`, and converts to `.icns` using `iconutil`. Requires the Python `Pillow` library, which the nix dev shell's `python3` includes. Output goes to `resources/AppIcon.icns`.

```bash
pip install Pillow  # outside the nix dev shell, if not already installed
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
- Double-speed mode: `bus_cycles = cpu_cycles / 2` -- the Bus M-cycle ticks (`tick_read`/`tick_write`/`tick_internal`) handle this
- Snapshots (`snapshot.rs`) support rewind (see `rewind.rs`, 3x playback with reverse audio) and save states (F5/F7, slots 0-9). Any change to a snapshotted struct needs a `FORMAT_VERSION` bump (the encoding test enforces it). Battery saves (`.sav`) use the standard layouts shared with other emulators (e.g. MBC3's 48-byte RTC footer) and are written atomically.
- Fast-forward audio: all frontends downsample 4x audio through a Blackman-windowed sinc FIR filter. Rewind has reverse audio with the same filter.
- OAM DMA is a pipelined M-cycle model (one byte read per M-cycle and written to OAM the next, with CPU bus conflicts and OAM blocking). GDMA and HBlank HDMA run as real M-cycles at the PPU rate (8 M-cycles per 16-byte block in normal speed, 16 in double speed); HBlank HDMA transfers one block per HBlank
- Boot ROMs are in `bootroms/` directory; test runner loads them with `--boot` flag
- DMG models use classic green Game Boy LCD palette (`DMG_SHADES`: `#9BBC0F`, `#8BAC0F`, `#306230`, `#0F380F`). MGB uses grayscale (`MGB_SHADES`: `#C4CFA1`, `#8B956D`, `#4D533C`, `#1F1F1F`).
- The emulator core (everything reachable from `Emulator`) has no I/O, filesystem, or platform dependencies. Time is injected via the `Clock` trait (`src/clock.rs`). Frontends handle rendering, audio, input, and persistence. The library also contains frontend-support modules that do I/O (`ui_util.rs`, `cpal_audio.rs`, `macos_accel.rs`); the core never calls them.
- Pure utility functions (audio processing, model detection, frame timing) in `src/util.rs`. Frontend-specific I/O helpers in `src/ui_util.rs`.
