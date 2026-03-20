#!/usr/bin/env bash
# Download public domain Game Boy ROMs from Zophar's Domain.
# These are all homebrew/public domain titles.
# Usage: ./scripts/fetch-pdroms.sh [--force]

set -euo pipefail

OUTDIR="web/roms"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

FORCE=false
if [ "${1:-}" = "--force" ]; then FORCE=true; fi

if [ -d "$OUTDIR" ] && [ "$(ls "$OUTDIR"/*.gb "$OUTDIR"/*.gbc 2>/dev/null | wc -l)" -gt 0 ] && [ "$FORCE" = false ]; then
  echo "ROMs already exist in $OUTDIR/. Use --force to re-download."
  exit 0
fi

mkdir -p "$OUTDIR"

download_rom() {
  local name="$1"
  local id="$2"

  echo "Downloading: $name (ID $id)..."
  local dl="$TMPDIR/$name.zip"
  curl -sL "https://www.zophar.net/download_file/$id" -o "$dl"

  if file "$dl" | grep -q "Zip"; then
    local extract_dir="$TMPDIR/extract_$id"
    mkdir -p "$extract_dir"
    unzip -o -j "$dl" -d "$extract_dir" 2>/dev/null || true
    local found=0
    for rom in "$extract_dir"/*; do
      [ -f "$rom" ] || continue
      local ext="${rom##*.}"
      ext=$(echo "$ext" | tr '[:upper:]' '[:lower:]')
      if [ "$ext" = "gb" ] || [ "$ext" = "gbc" ]; then
        cp "$rom" "$OUTDIR/${name}.${ext}"
        echo "  -> ${name}.${ext}"
        found=1
        break
      fi
    done
    if [ "$found" = 0 ]; then
      echo "  WARNING: No .gb/.gbc found in zip for $name, skipping"
    fi
  else
    local ext="gb"
    if [ "$(wc -c < "$dl")" -gt 323 ]; then
      local flag
      flag=$(xxd -s 0x143 -l 1 -p "$dl" 2>/dev/null || echo "00")
      if [ "$flag" = "80" ] || [ "$flag" = "c0" ]; then
        ext="gbc"
      fi
    fi
    cp "$dl" "$OUTDIR/${name}.${ext}"
    echo "  -> ${name}.${ext}"
  fi
}

download_rom "Drymouth"                   11693
download_rom "GB Lander"                  11697
download_rom "Horrible Demon 3"           11798
download_rom "Hungry are the Dead"        11694
download_rom "Iron Hike"                  11698
download_rom "Jet Pak DX"                 11699
download_rom "Mainblow"                   11799
download_rom "Marcbla 3D"                 11803
download_rom "Megamania"                  11700
download_rom "Moon Fighter"               11797
download_rom "Mr. Dig"                    11801
download_rom "Poke Mission 97"            11701
download_rom "Q-Bert"                     11823
download_rom "Shutdown"                   11705
download_rom "Skoardy"                    11703
download_rom "Sokoban"                    11704
download_rom "Sqrxz"                      11706
download_rom "The Horrible Demon"         11702
download_rom "They Came From Outer Space" 11709
download_rom "Tic-Tac-Toe"               11708
download_rom "Ultima III"                 11775
download_rom "Yars Revenge"               11711

echo ""
echo "Done! $(ls "$OUTDIR"/*.gb "$OUTDIR"/*.gbc 2>/dev/null | wc -l | tr -d ' ') ROMs in $OUTDIR/"
