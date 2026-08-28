# vibeboy

[**Play in your browser**](https://northbymidwest.github.io/vibeboy/)

A Game Boy / Game Boy Color emulator written in Rust.

Supports **DMG**, **DMG0**, **MGB**, **SGB**, **SGB2**, **CGB**, and **AGB** (GBA in GBC mode) hardware models. MGB uses an authentic grayscale palette.

## Building

Requires Rust 2024 edition and SDL3. For GPU shaders: [Slang](https://github.com/shader-slang/slang/releases) (`slangc` on PATH). Per-platform setup is in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

```bash
cargo build --release

# Other frontends
cargo build --release --bin vibeboy_cocoa --features macos-ui    # Native macOS Cocoa/Metal
cargo build --release --bin vibeboy_winit --features winit-ui    # Cross-platform winit/wgpu
cargo build --release --bin vibeboy_gtk   --features gtk-ui      # GTK4

# WebAssembly (browser)
wasm-pack build --target web --features web --no-default-features

# libretro core (for RetroArch)
cargo build --release --features libretro --no-default-features --lib
```

## Usage

```bash
# Launch with a ROM (auto-detects model from cart header)
cargo run --release -- path/to/rom.gb

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
| Arrow keys | D-pad |
| Z / X | B / A |
| Enter | Start |
| Right Shift | Select |
| Backspace | Rewind |
| Tab (hold) | Fast forward (4x) |
| Minus (hold) | Slow motion (0.5x) |
| Space | Pause / Resume |
| Period | Frame advance (while paused) |
| F5 / F7 | Save / Load state |
| 0-9 | Select save slot |
| Escape | Quit |

Gamepad: D-pad/left stick, South=B, East=A, Start, Back=Select, L1=Rewind, R1=Fast-forward.

### Scaling Filters

35 scaling filter options across 20 algorithm modules, each available as both CPU and GPU compute shader:

```bash
cargo run --release -- path/to/rom.gb --filter hq4x

# Pixel-art filters
#   nearest, nearest-aa, bilinear, bicubic
#   epx/scale2x, scale3x, scale4x, eagle
#   2xsai, super-2xsai, super-eagle
#   hq2x, hq3x, hq4x
#   xbr2x-4x, super-xbr, xbrz2x-6x
#   nedi, dcci, edi, mmpx
#   omniscale, omniscale-legacy, lcd-grid
#   scalefx, scalefx-9x

# Kopf-Lischinski pixel-art vectorization (full GPU pipeline)
cargo run --release -- path/to/rom.gb --filter vectorize

# Force CPU-only rendering for any filter
cargo run --release -- path/to/rom.gb --filter omniscale --cpu-filter
```

The vectorize filter converts each frame to smooth vector paths using the [Kopf-Lischinski algorithm](https://johanneskopf.de/publications/pixelart/) via a 6-stage GPU pipeline (similarity graph, crossing resolution, cell graph, T-junction update, energy optimization, B-spline rasterization). Falls back to a pixel-identical CPU implementation when no GPU is available. Runs in real-time.

## Features

- **CPU**: Full SM83 instruction set with accurate M-cycle timing
- **PPU**: Pixel FIFO renderer with per-T-cycle accuracy; DMG green palette, MGB grayscale palette
- **APU**: DIV-coupled frame sequencer, all 4 channels, band-limited synthesis at 96 kHz
- **Cartridges**: ROM-only, MBC1 (multicart), MBC2, MBC3 (with RTC), MBC5 (with rumble), MBC6, MBC7 (accelerometer + EEPROM), HuC1, HuC3, TAMA5, MMM01, Pocket Camera
- **CGB**: Double-speed mode, VRAM banking, color palettes, HDMA, WRAM banking
- **SGB**: HLE command processing (palettes, attributes, borders, masking); optional LLE with full 65C816 SNES CPU
- **OAM DMA**: Bus conflict emulation for both DMG and CGB
- **Rumble**: MBC5+Rumble with gamepad haptic feedback (SDL, CoreHaptics, Web vibrationActuator)
- **Camera**: Game Boy Camera via webcam (SDL3, AVFoundation, nokhwa, getUserMedia)
- **Printer**: Game Boy Printer emulation (PNG output on native, browser download on web)
- **Save states**: 10 slots (0-9), serde + bincode serialization, rewind (~10 minutes, reverse-delta compressed)
- **Runahead**: `--runahead N` for reduced input latency (SDL frontend)
- **Scaling**: 35 scaling filter options across 20 algorithm modules, all with GPU compute shader acceleration
- **Vectorization**: Kopf-Lischinski pixel-art vectorizer with 6-stage GPU pipeline (CPU fallback); SVG export
- **6 frontends**: SDL3, native macOS Cocoa/Metal, winit/wgpu, GTK4, WebAssembly/WebGPU, libretro
- **Browser**: WebGPU rendering, on-screen touch controls, gamepad, AudioWorklet at 96 kHz, webcam, accelerometer (DeviceMotion for MBC7), localStorage persistence, mobile-responsive
- **libretro**: RetroArch-compatible core with save RAM persistence (including RTC)

## Test Runner

```bash
# Mooneye tests (breakpoint + Fibonacci register check)
cargo run --release --bin test_runner -- test mooneye game-boy-test-roms/mooneye-test-suite/acceptance/

# Blargg tests (serial output detection)
cargo run --release --bin test_runner -- test blargg game-boy-test-roms/blargg/

# Gambatte tests (hex output comparison)
cargo run --release --bin test_runner -- test gambatte game-boy-test-roms/gambatte/

# Screenshot any ROM after N frames
cargo run --release --bin test_runner -- screenshot path/to/rom.gb --frames 300 --out shot.png

# Vectorize a standalone image to SVG
cargo run --release --bin test_runner -- vectorize input.png --out output.svg
```

## Architecture

```
src/
├── frontends/
│   ├── sdl/         SDL3 window, audio, input, GPU rendering
│   ├── cocoa/       Native macOS Cocoa/Metal UI
│   ├── winit/       Cross-platform winit/wgpu UI
│   ├── gtk/         GTK4 UI with GPU compute (Linux)
│   ├── web/         WebAssembly/wasm-bindgen frontend
│   └── libretro/    RetroArch libretro core
├── emulator.rs      Emulator facade API (step_frame, save states, rewind)
├── cpu/             SM83 CPU (opcodes, interrupts, HALT)
├── bus/             Memory map, OAM DMA, HDMA, IO registers
├── ppu/             Pixel FIFO PPU (timing, rendering, registers)
├── apu/             Audio: channels 1-4, frame sequencer, BLIP synthesis
├── timer.rs         DIV/TIMA timer with reload delay
├── cartridge/       13 mapper implementations (MBC1-7, HuC1/3, TAMA5, etc.)
├── clock.rs         Clock trait for RTC abstraction (no platform deps in core)
├── sgb.rs           Super Game Boy HLE + optional SNES LLE
├── snes/            65C816 CPU, LoROM, DMA, ICD2 bridge
├── serial.rs        Link cable / serial port
├── printer.rs       Game Boy Printer (memory-queued output)
├── scaling/         20 CPU scaling filters + GPU compute pipelines + vectorize
├── shaders/         Slang compute shaders (cross-compiled to SPIR-V/MSL/DXIL/WGSL)
├── util.rs          Pure utility functions (no I/O)
├── ui_util.rs       Frontend utilities (filesystem, gamepad, FPS counter)
└── test_runner/     Automated test ROM harnesses + SVG export
web/
├── index.html       Browser UI markup
├── style.css        Responsive styles + touch controls
├── emu.js           Emulator logic (lazy wasm loading, state management)
├── touch.js         Multi-touch gamepad controls
└── audio-processor.js  AudioWorklet processor
```

The core emulator is a pure computation engine with no I/O, filesystem, or platform dependencies. Time is injected via the `Clock` trait. Frontends handle rendering, audio output, input, and persistence.

## License

[BSD Zero Clause License](LICENSE)

### Why 0BSD?

The majority of this codebase was generated by AI coding agents (primarily Claude). AI-generated code is not copyrightable and is effectively public domain, making 0BSD — which imposes no restrictions on use — the most appropriate license.

### Disclaimer

While AI-generated code itself is public domain, AI agents may have reproduced or closely derived code from copyrighted sources (training data, reference implementations, open-source projects, etc.). No audit has been conducted to identify such instances, as this is a personal side project. Any such code fragments remain subject to the licenses of their original creators. Use at your own discretion.
