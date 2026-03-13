# vibeboy

A Game Boy / Game Boy Color emulator written in Rust.

Supports **DMG**, **DMG0**, **MGB**, **SGB**, **SGB2**, **CGB**, and **AGB** (GBA in GBC mode) hardware models.

## Building

Requires Rust 2024 edition and SDL3.

```bash
cargo build --release
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
| F5 | Save state |
| F7 | Load state |
| 1-9 | Select save slot |
| Escape | Quit |

### Scaling Filters

```bash
# Run with a scaling filter (nearest, bilinear, epx, hq2x-4x, xbrz2x-6x, omniscale2x-6x, ...)
cargo run --release -- path/to/rom.gb --filter hq4x

# Kopf-Lischinski pixel-art vectorization (2x-6x)
cargo run --release -- path/to/rom.gb --filter vectorize4x
```

The vectorize filter converts each frame to smooth vector paths using the Kopf-Lischinski algorithm, then rasterizes at the target scale with anti-aliased edges. Runs in real-time (~10ms/frame).

## Features

- **CPU**: Full SM83 instruction set with accurate M-cycle timing
- **PPU**: Pixel FIFO renderer with per-T-cycle accuracy
- **APU**: DIV-coupled frame sequencer, all 4 channels
- **Cartridges**: ROM-only, MBC1, MBC2, MBC3 (with RTC), MBC5, MBC6, MBC7 (accelerometer + EEPROM)
- **CGB**: Double-speed mode, VRAM banking, color palettes, HDMA, WRAM banking
- **SGB**: HLE command processing (palettes, attributes, borders)
- **OAM DMA**: Bus conflict emulation for both DMG and CGB
- **Save states**: 9 slots with rewind support (~10 seconds buffer)
- **Camera**: Game Boy Camera support via webcam (macOS)
- **Vectorization**: Kopf-Lischinski pixel-art vectorizer with real-time rasterization

## Test Status

| Suite | Result |
|-------|--------|
| Mooneye acceptance | 75/75 |
| Blargg | 57/58 |
| dmg-acid2 | Pass |
| cgb-acid2 | Pass |
| cgb-acid-hell | Pass |
| bully | Pass (DMG + CGB) |

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
├── main.rs          SDL3 window, audio, input, file dialog
├── emulator.rs      Frame loop, SGB compositing
├── cpu/             SM83 CPU (opcodes, interrupts, HALT)
├── bus.rs           Memory map, OAM DMA, HDMA, WRAM banking
├── ppu/             Pixel FIFO PPU (mode state machine, fetcher, sprites)
├── apu.rs           Audio: channels 1-4, frame sequencer, mixing
├── timer.rs         DIV/TIMA timer with reload delay
├── cartridge/       MBC implementations (1, 2, 3, 5, 6, 7)
├── sgb.rs           Super Game Boy HLE commands
├── serial.rs        Link cable / serial port
├── joypad.rs        Input handling
├── snapshot.rs      Rewind ring buffer + save states
├── model.rs         GbModel enum and per-model configuration
├── scaling/         Pixel scaling filters (EPX, HQx, xBRZ, OmniScale, ...)
├── vectorize/       Kopf-Lischinski pixel-art vectorizer + rasterizer
└── test_runner/     Automated test ROM runner (modular harnesses)
```

The main loop: `Emulator::step_frame()` calls `Cpu::step()` per instruction. Each M-cycle, `Bus::tick_mcycle()` advances PPU (4 T-cycles), APU, Timer, Serial, OAM DMA, and HDMA.

## License

[BSD Zero Clause License](LICENSE)
