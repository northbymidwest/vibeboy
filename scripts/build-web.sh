#!/usr/bin/env bash
# Build the VibeBoy WebAssembly frontend.
#
# Usage:
#   ./scripts/build-web.sh              # build wasm only
#   ./scripts/build-web.sh --roms       # build wasm + download public domain ROMs
#   ./scripts/build-web.sh --roms-only  # download ROMs without rebuilding wasm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WEB_DIR="$PROJECT_DIR/web"

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
  # Use nightly toolchain for wasm-pack (required for some features)
  NIGHTLY_BIN="$HOME/.rustup/toolchains/nightly-$(rustc -vV | grep host | awk '{print $2}')/bin"
  if [ -d "$NIGHTLY_BIN" ]; then
    export PATH="$NIGHTLY_BIN:$PATH"
  fi
  (cd "$PROJECT_DIR" && wasm-pack build --target web --features web --no-default-features)
  # Copy wasm output to web/pkg/
  mkdir -p "$WEB_DIR/pkg"
  cp "$PROJECT_DIR/pkg/vibeboy.js" "$WEB_DIR/pkg/"
  cp "$PROJECT_DIR/pkg/vibeboy_bg.wasm" "$WEB_DIR/pkg/"
  echo "==> WASM build complete: web/pkg/"
fi

if [ "$FETCH_ROMS" = true ]; then
  echo "==> Fetching public domain ROMs..."
  bash "$SCRIPT_DIR/fetch-pdroms.sh"
fi

echo ""
echo "==> Web build ready. Serve web/ with any static file server:"
echo "    python3 -m http.server -d web 8080"
