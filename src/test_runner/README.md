# Test Runner

Headless test ROM runner for vibeboy. Supports multiple test harness formats with automatic model detection and boot ROM loading.

## Usage

All commands use explicit subcommands:

```bash
cargo run --release --bin test_runner -- <command> [options]
```

### Running Tests

```bash
# Mooneye tests (breakpoint + Fibonacci register check)
cargo run --release --bin test_runner -- test mooneye game-boy-test-roms/mooneye-test-suite/acceptance/

# Blargg tests (serial output detection)
cargo run --release --bin test_runner -- test blargg game-boy-test-roms/blargg/

# Gambatte tests (hex output comparison after 15 frames)
cargo run --release --bin test_runner -- test gambatte game-boy-test-roms/gambatte/

# GBMicrotest (HRAM result check after 2 frames)
cargo run --release --bin test_runner -- test gbmicrotest game-boy-test-roms/gbmicrotest/

# Mealybug Tearoom tests (screenshot comparison after LD B,B breakpoint)
cargo run --release --bin test_runner -- test tearoom game-boy-test-roms/mealybug-tearoom-tests/

# Run a single test file
cargo run --release --bin test_runner -- test blargg game-boy-test-roms/blargg/cpu_instrs/individual/01-special.gb

# Subdirectory of a test suite
cargo run --release --bin test_runner -- test gambatte game-boy-test-roms/gambatte/sprites/
```

#### Test Flags

| Flag | Description |
|------|-------------|
| `--model <model>` | Force hardware model (dmg, dmg0, mgb, sgb, sgb2, cgb, cgb0, agb) |
| `--boot` | Load boot ROM (auto-detected from `bootroms/` by model) |
| `--bootrom <path>` | Use a specific boot ROM file (implies --boot) |
| `--verbose` | Print extra diagnostics per test |
| `--quiet` | Only print the summary line |

### Screenshots

```bash
# Capture a PNG screenshot after 300 frames
cargo run --release --bin test_runner -- screenshot path/to/rom.gb --frames 300 --out shot.png

# Vectorize a frame to SVG
cargo run --release --bin test_runner -- screenshot path/to/rom.gb --frames 300 --out frame.svg --format svg

# Vectorize and rasterize at 4x scale
cargo run --release --bin test_runner -- screenshot path/to/rom.gb --frames 300 --out frame.png --format raster --scale 4

# Simulate button presses (frame:button pairs)
cargo run --release --bin test_runner -- screenshot path/to/rom.gb --frames 600 --keys "100:start,200:a"
```

### Other Commands

```bash
# Vectorize an existing PNG image to SVG
cargo run --release --bin test_runner -- vectorize input.png --out output.svg

# Analyze frame buffer (debug)
cargo run --release --bin test_runner -- analyze path/to/rom.gb --frames 300

# Trace timer state around boot ROM handoff (debug)
cargo run --release --bin test_runner -- trace-timer path/to/rom.gb --boot

# Dump PPU/timer state at PC=$0100 for all models with boot ROMs
cargo run --release --bin test_runner -- calibrate path/to/rom.gb
```

## Model Auto-Detection

When `--model` is not specified, the test runner detects the hardware model from:

1. **Filename suffix**: `-dmgABCmgb`, `-dmg0`, `-mgb`, `-sgb`, `-sgb2`, `-cgb`, `-cgbABCDE`, `-C`, `-A`, `-GS`, `-G`, `-S`
2. **Cart header CGB flag** (address `$0143`): `$80` or `$C0` → CGB, otherwise DMG
3. **Special cases**: `oam_bug` paths always use DMG

## Test Harnesses

| Harness | Detection Method | Pass Condition |
|---------|-----------------|----------------|
| **Mooneye** | LD B,B breakpoint | Fibonacci registers (B=3, C=5, D=8, E=13, H=21, L=34) |
| **Blargg** | Serial output | "Passed" in output, detected via JR -2 done-loop |
| **Gambatte** | Screenshot at frame 15 | Hex digit recognition matches expected output from filename |
| **GBMicrotest** | HRAM check at frame 2 | `$FF80` == 1 |
| **Tearoom** | LD B,B breakpoint screenshot | Pixel-exact match against reference PNG |

## Module Structure

```
src/test_runner/
├── main.rs              CLI entry point and command dispatch
├── harness.rs           TestResult enum, TestHarness trait, run_tests() orchestrator
├── harnesses/
│   ├── mod.rs
│   ├── mooneye.rs       Breakpoint detection + register check
│   ├── blargg.rs        Serial output parsing
│   ├── gambatte.rs      Screenshot hex digit recognition
│   ├── gbmicrotest.rs   HRAM result check
│   └── tearoom.rs       Screenshot comparison against reference PNGs
├── commands.rs          screenshot, vectorize, analyze, trace-timer, calibrate
├── model.rs             Model detection + boot ROM resolution
└── util.rs              Shared helpers (make_emu, collect_roms, parse_keys)
```
