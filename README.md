# vibeboy

[**Play in your browser**](https://northbymidwest.github.io/vibeboy/)

A Game Boy / Game Boy Color emulator written in Rust.

Supports **DMG**, **DMG0**, **MGB**, **SGB**, **SGB2**, **CGB**, and **AGB** (GBA in GBC mode) hardware models.

## Building

Requires Rust 2024 edition and SDL3. Optional features: `macos-ui` (native Cocoa frontend), `winit-ui` (cross-platform winit/wgpu frontend), `gpu` (GPU compute), `sdl3-gpu-shaders` (enabled by default), `web` (WebAssembly browser frontend).

```bash
cargo build --release

# WebAssembly (browser) build
wasm-pack build --target web --features web --no-default-features
```

## Usage

```bash
# Launch with a ROM (auto-detects model from cart header)
cargo run --release -- path/to/rom.gb

# If no ROM is specified, a native file dialog opens to pick one

# Force a specific hardware model
cargo run --release -- path/to/rom.gbc --model cgb

# Use a boot ROM (auto-detected from bootroms/ if present)
cargo run --release -- path/to/rom.gb --bootrom bootroms/dmg_boot.bin

# Skip boot ROM
cargo run --release -- path/to/rom.gb --no-boot
```

### Controls

| Key | Action |
|-----|--------|
| Z | B |
| X | A |
| Enter | Start |
| Right Shift | Select |
| Arrow keys | D-pad |
| Backspace | Rewind |
| Tab (hold) | Fast forward |
| Minus | Slow motion |
| F5 | Save state |
| F7 | Load state |
| 1-9 | Select save slot |
| Escape | Quit |

### Scaling Filters

```bash
# Run with a scaling filter
cargo run --release -- path/to/rom.gb --filter hq4x

# Available filters: nearest, bilinear, bicubic, epx, scale2x, scale3x, scale4x,
#   eagle, 2xsai, super-2xsai, super-eagle, hq2x, hq3x, hq4x,
#   xbr2x, xbr3x, xbr4x, xbrz2x-6x, super-xbr,
#   nedi, dcci, edi, omniscale, omniscale-legacy, nearest-aa, mmpx, lcd-grid

# Kopf-Lischinski pixel-art vectorization (scales to window size)
cargo run --release -- path/to/rom.gb --filter vectorize
cargo run --release -- path/to/rom.gb --filter vectorize-adaptive

# Full GPU vectorize pipeline (all stages on GPU)
cargo run --release -- path/to/rom.gb --filter vectorize-gpu

# Legacy vectorize (original scanline rasterizer path)
cargo run --release -- path/to/rom.gb --filter vectorize-legacy
cargo run --release -- path/to/rom.gb --filter vectorize-legacy-adaptive

# Gaussian diffusion renderers (paper's rendering approach)
cargo run --release -- path/to/rom.gb --filter vectorize-diffusion
cargo run --release -- path/to/rom.gb --filter vectorize-spline-diffusion
cargo run --release -- path/to/rom.gb --filter vectorize-spline-diffusion-adaptive

# Force CPU-only rendering for any filter
cargo run --release -- path/to/rom.gb --filter vectorize --cpu-filter
```

The vectorize filters convert each frame to smooth vector paths using the [Kopf-Lischinski algorithm](https://johanneskopf.de/publications/pixelart/), then rasterize at the target scale. The scanline variant (`vectorize`) uses 2x2 supersampled fill with GPU compute shader acceleration. The spline-diffusion variants add Gaussian color blending within contour-bounded regions, matching the paper's rendering approach. The `vectorize-gpu` filter runs the entire pipeline on the GPU (similarity graph, crossing resolution, cell graph, optimization, and rasterization). All run in real-time.

## Features

- **CPU**: Full SM83 instruction set with accurate M-cycle timing
- **PPU**: Pixel FIFO renderer with per-T-cycle accuracy; DMG models use classic green LCD palette
- **APU**: DIV-coupled frame sequencer, all 4 channels
- **Cartridges**: ROM-only, MBC1, MBC2, MBC3 (with RTC), MBC5 (with rumble), MBC6, MBC7 (accelerometer + EEPROM)
- **Rumble**: MBC5+Rumble cartridge support with gamepad haptic feedback (SDL, CoreHaptics, Web vibrationActuator)
- **Runahead**: `--runahead N` for reduced input latency (SDL frontend)
- **CGB**: Double-speed mode, VRAM banking, color palettes, HDMA, WRAM banking
- **SGB**: HLE command processing (palettes, attributes, borders)
- **OAM DMA**: Bus conflict emulation for both DMG and CGB
- **Save states**: Serialized via serde + bincode (`rom.N.ss` files), 9 slots with rewind support (~10 minutes buffer, reverse-delta compressed)
- **Camera**: Game Boy Camera support via webcam (macOS native, SDL3, browser getUserMedia)
- **Printer**: Game Boy Printer emulation (saves PNG to `prints/`, or browser download)
- **Scaling**: 41+ filters including EPX, HQx, xBR, xBRZ, OmniScale, Super-xBR, NEDI, DCCI, EDI, MMPX, LCD Grid, and more — all GPU filters use compute shaders
- **Vectorization**: Kopf-Lischinski pixel-art vectorizer with 3 rendering modes (scanline, diffusion, spline-diffusion), GPU compute shaders, SVG export
- **Multiple frontends**: SDL3 (default), native macOS Cocoa/Metal, cross-platform winit/wgpu, GTK4, WebAssembly/WebGPU browser
- **Browser**: Runs in any WebGPU-capable browser — drag-and-drop ROM loading, built-in ROM selector with public domain games, all GPU filters via dropdown, model selection, gamepad support, accelerometer (DeviceMotion for MBC7), AudioWorklet at 96kHz, webcam for Game Boy Camera, localStorage save persistence

## Test Runner

A built-in test runner with explicit subcommands for each test harness. See [`src/test_runner/README.md`](src/test_runner/README.md) for full documentation.

```bash
# Mooneye tests (breakpoint + Fibonacci register check)
cargo run --release --bin test_runner -- test mooneye game-boy-test-roms/mooneye-test-suite/acceptance/

# Blargg tests (serial output detection)
cargo run --release --bin test_runner -- test blargg game-boy-test-roms/blargg/

# Gambatte tests (hex output comparison, 15-frame capture)
cargo run --release --bin test_runner -- test gambatte game-boy-test-roms/gambatte/

# Screenshot any ROM after N frames
cargo run --release --bin test_runner -- screenshot path/to/rom.gb --frames 300 --out shot.png
```

## Architecture

```
src/
├── frontends/       Frontend binaries (moved from src/ root)
│   ├── sdl/         SDL3 window, audio, input, file dialog
│   ├── cocoa/       Native macOS Cocoa/Metal UI (feature: macos-ui)
│   ├── winit/       Cross-platform winit/wgpu UI (feature: winit-ui)
│   └── web/         WebAssembly/wasm-bindgen frontend (feature: web)
├── emulator.rs      Frame loop, SGB compositing
├── cpu/             SM83 CPU (opcodes, interrupts, HALT)
├── bus.rs           Memory map, OAM DMA, HDMA, WRAM banking
├── ppu/             Pixel FIFO PPU (mode state machine, fetcher, sprites)
├── apu.rs           Audio: channels 1-4, frame sequencer, mixing
├── timer.rs         DIV/TIMA timer with reload delay
├── cartridge/       MBC implementations (1, 2, 3, 5, 6, 7)
├── sgb.rs           Super Game Boy HLE commands
├── serial.rs        Link cable / serial port
├── printer.rs       Unified Game Boy Printer (PrintOutput::File/Memory)
├── joypad.rs        Input handling
├── snapshot.rs      Rewind ring buffer + save states
├── savestate.rs     serde + bincode serialization (rom.N.ss files)
├── model.rs         GbModel enum and per-model configuration
├── scaling/         41+ pixel scaling filters (available on all platforms)
│   ├── sdl/         SDL3 GPU pipeline management (pipelines.rs, compute.rs)
│   ├── wgpu_vectorize.rs wgpu compute pipeline (6-stage, WebGPU-compatible)
│   └── *.rs         EPX, HQx, xBR, xBRZ, OmniScale, NEDI, DCCI, EDI, MMPX, LCD Grid, ...
├── shaders/         GPU compute shaders (21 .comp files, GLSL 4.50)
├── vectorize/       Kopf-Lischinski pixel-art vectorizer
│   ├── graph.rs     Similarity graph (YUV thresholds, crossing heuristics)
│   ├── contour/     Cell templates, edge chains, B-spline optimizer
│   ├── rasterize/   Scanline, diffusion, spline-diffusion rasterizers
│   └── svg.rs       SVG export
├── lib.rs           Library crate for WebAssembly builds
├── test_runner/     Automated test ROM runner (modular harnesses)
web/
├── index.html       Browser UI (Canvas2D/WebGPU, Web Audio, drag-and-drop)
├── favicon.ico      App icon
└── roms/            Built-in public domain ROMs
```

The main loop: `Emulator::step_frame()` calls `Cpu::step()` per instruction. Each M-cycle, `Bus::tick_mcycle()` advances PPU (4 T-cycles), APU, Timer, Serial, OAM DMA, and HDMA.

## License

[BSD Zero Clause License](LICENSE)

### Why 0BSD?

The majority of this codebase was generated by AI coding agents (primarily Claude). AI-generated code is not copyrightable and is effectively public domain, making 0BSD — which imposes no restrictions on use — the most appropriate license.

### Disclaimer

While AI-generated code itself is public domain, AI agents may have reproduced or closely derived code from copyrighted sources (training data, reference implementations, open-source projects, etc.). No audit has been conducted to identify such instances, as this is a personal side project. Any such code fragments remain subject to the licenses of their original creators. Use at your own discretion.
