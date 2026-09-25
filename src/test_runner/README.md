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

# GBMicrotest (HRAM/VRAM result check after 4 frames)
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
| `--allow-failures` | Exit 0 even when tests fail, time out or error |

Each test prints one line, `PASS`, `FAIL`, `TIMEOUT` or `ERR` (the ROM or a
harness input could not be read) followed by its path, then a summary line:

```
--- 57 passed, 0 failed, 1 timeout (58 total) ---
```

Errors and skipped ROMs (gambatte ROMs with no expected output in the name,
tearoom ROMs with no reference image for the model) are added to the summary
when there are any. Skipped ROMs are not part of the total.

#### Exit Status

| Status | Meaning |
|--------|---------|
| 0 | Every test passed, or `--allow-failures` was given |
| 1 | At least one test failed, timed out or errored |
| 2 | No `.gb`/`.gbc` ROMs under the path (whatever the flags) |

`--allow-failures` is for runs over whole suites with known failures, where the
per-test lines are the result. `scripts/accuracy.sh` uses it to run every suite
and compare each test's status against `tests/accuracy-baseline.txt`:

```bash
# Compare against the baseline; exits 1 if a baseline PASS no longer passes
./scripts/accuracy.sh

# Record this run as the new baseline (commit it with the change that moved it)
./scripts/accuracy.sh --update
```

### Screenshots

```bash
# Capture a PNG screenshot after 300 frames
cargo run --release --bin test_runner -- screenshot path/to/rom.gb --frames 300 --out shot.png

# Vectorize a frame to SVG
cargo run --release --bin test_runner -- screenshot path/to/rom.gb --frames 300 --out frame.svg --format svg

# Vectorize and rasterize at 4x scale
cargo run --release --bin test_runner -- screenshot path/to/rom.gb --frames 300 --out frame.png --format raster --scale 4

# Apply any scaling filter (same names as SDL --filter)
cargo run --release --bin test_runner -- screenshot path/to/rom.gb --frames 300 --out frame.png --filter hq4x --scale 4
cargo run --release --bin test_runner -- screenshot path/to/rom.gb --frames 300 --out frame.png --filter vectorize --scale 4

# Simulate button presses (frame:button pairs)
cargo run --release --bin test_runner -- screenshot path/to/rom.gb --frames 600 --keys "100:start,200:a"
```

### Vectorize Command

Vectorize a standalone PNG image. Uses `--gpu` to run the full GPU pipeline (falls back to CPU if unavailable) and `--cpu-filter` to force CPU-only rendering.

```bash
# Vectorize to SVG
cargo run --release --bin test_runner -- vectorize input.png --out output.svg

# Vectorize at 8x scale (GPU pipeline with CPU fallback)
cargo run --release --bin test_runner -- vectorize input.png --out output.png --scale 8 --gpu

# Force CPU-only (no GPU shaders)
cargo run --release --bin test_runner -- vectorize input.png --out output.png --scale 8 --cpu-filter
```

### Audio Dump

Dump APU audio to a 32-bit float stereo WAV file. The APU generates samples at
`--sample-rate` (8000 to 384000 Hz, default 96000), so the data matches the
header at any rate.

```bash
# Dump 300 frames of audio at 96kHz
cargo run --release --bin test_runner -- audio-dump path/to/rom.gb --frames 300 --out audio.wav

# Custom sample rate
cargo run --release --bin test_runner -- audio-dump path/to/rom.gb --frames 600 --out audio.wav --sample-rate 48000

# Force hardware model
cargo run --release --bin test_runner -- audio-dump path/to/rom.gb --frames 300 --model cgb
```

### Boot ROM Generation

Generate built-in boot ROMs to files.

```bash
# Generate CGB boot ROM (default)
cargo run --release --bin test_runner -- gen-bootrom --out bootroms/vibeboy_cgb_boot.bin

# Generate DMG/MGB/AGB boot ROM
cargo run --release --bin test_runner -- gen-bootrom --out bootroms/vibeboy_dmg_boot.bin --model dmg
```

### Debug Commands

```bash
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
| **Gambatte** | Screenshot at frame 15 | Hex digit recognition matches expected output from filename (dual `_dmg08_outX_cgb04c_outY` names run both models; both must pass) |
| **GBMicrotest** | HRAM check at frame 4 (30 for `is_if_set_during_ime0`) | `$FF82` == `$01` (`$FF` is a fail); ROMs that report in VRAM instead: `$8000` equals the value the ROM compares against |
| **Tearoom** | LD B,B breakpoint screenshot | Pixel-exact match against reference PNG |

Mooneye and Tearoom time out after 300 frames without reaching the breakpoint;
Blargg after 6000 frames without a result on the serial port.

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
├── commands.rs          screenshot, vectorize, audio-dump
├── debug_commands.rs    analyze, trace-timer, calibrate
├── model.rs             Model detection + boot ROM resolution
├── gpu_svg.rs           SVG export for vectorized frames (GPU pipeline)
└── util.rs              Shared helpers (make_emu, collect_roms, parse_keys)
```
