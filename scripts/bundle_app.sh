#!/bin/bash
set -euo pipefail

# Build the VibeBoy.app macOS application bundle

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_NAME="VibeBoy"
APP_DIR="$PROJECT_DIR/target/${APP_NAME}.app"

echo "Building vibeboy_cocoa (release)..."
cargo build --release --bin vibeboy_cocoa --manifest-path "$PROJECT_DIR/Cargo.toml"

echo "Creating ${APP_NAME}.app bundle..."
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

# Copy binary
cp "$PROJECT_DIR/target/release/vibeboy_cocoa" "$APP_DIR/Contents/MacOS/vibeboy_cocoa"

# Copy Info.plist
cp "$PROJECT_DIR/Info.plist" "$APP_DIR/Contents/Info.plist"

echo "Bundle created at: $APP_DIR"
echo ""
echo "To run:  open $APP_DIR"
echo "To install: cp -r $APP_DIR /Applications/"
