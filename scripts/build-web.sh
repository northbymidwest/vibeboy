#!/usr/bin/env bash
# Build the VibeBoy WebAssembly frontend into web/, ready to serve as is.
# The Pages workflow runs this same script and publishes web/.
#
# Usage:
#   ./scripts/build-web.sh              # build wasm only
#   ./scripts/build-web.sh --roms       # build wasm + download public domain ROMs
#   ./scripts/build-web.sh --roms-only  # download ROMs without rebuilding wasm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_WASM=true
FETCH_ROMS=false

for arg in "$@"; do
  case "$arg" in
    --roms)      FETCH_ROMS=true ;;
    --roms-only) FETCH_ROMS=true; BUILD_WASM=false ;;
    *)           echo "Unknown flag: $arg"; echo "Usage: $0 [--roms] [--roms-only]"; exit 1 ;;
  esac
done

if [ "$BUILD_WASM" = true ]; then
  echo "==> Building WebAssembly..."
  # Straight into web/pkg, where web/emu.js imports it from.
  (cd "$PROJECT_DIR" && wasm-pack build --target web --out-dir web/pkg \
    --features web --no-default-features)
  echo "==> WASM build complete: web/pkg/"
fi

if [ "$FETCH_ROMS" = true ]; then
  echo "==> Fetching public domain ROMs..."
  (cd "$PROJECT_DIR" && bash "$SCRIPT_DIR/fetch-pdroms.sh")
fi

echo ""
echo "==> Web build ready. Serve web/ with any static file server:"
echo "    python3 -m http.server -d web 8080"
