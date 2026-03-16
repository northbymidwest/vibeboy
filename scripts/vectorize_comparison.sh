#!/bin/bash
# Download input images and paper's "ours" results at 8x from the
# Kopf-Lischinski supplementary page, then run our vectorizer on
# each input for side-by-side comparison.
#
# Usage: ./scripts/vectorize_comparison.sh
#
# Output: vectorize-tests/
#   {name}_input.png       — original pixel art
#   {name}_paper_8x.png    — paper's result at 8x
#   {name}_ours_8x.png     — our scanline vectorizer at 8x
#   {name}_sdiff_8x.png    — our spline-diffusion at 8x

set -e

BASE_URL="https://johanneskopf.de/publications/pixelart/supplementary"
OUT_DIR="vectorize-tests"
SCALE=8

NAMES=(
    smw2_yoshi_01 smw2_yoshi_02 smw_bowser smw_boo smw_dolphin
    smw_help smw_mario smw_mario_yoshi smw_mushroom smw_yoshi
    sma_chest sma_peach_01 sma_peach_02 smb_jump smw_cape_mario_yoshi
    sma_toad smw2_koopa
    invaders_01 invaders_02 invaders_03 invaders_04 invaders_05 invaders_06
    mana_granpa mana_joch mana_rabite mana_randi_01 mana_randi_02
    mana_salamando mana_sword
    sbm1_01 sbm1_02 sbm1_03 sbm1_04
    sbm4_01 sbm4_02 sbm4_03 sbm4_04
    gaxe2_axbattler_01 gaxe2_axbattler_02 gaxe_skeleton
    icon_atari_bomb icon_disk vista_cursor win31_cursor
    win31_386 win31_control_panel win31_fonts win31_keyboard
    win31_ports win31_setup
    vikings_baelog vikings_eric vikings_olaf
)

mkdir -p "$OUT_DIR"

echo "=== Downloading input images and paper results ==="
for name in "${NAMES[@]}"; do
    input="$OUT_DIR/${name}_input.png"
    paper="$OUT_DIR/${name}_paper_8x.png"

    if [ ! -f "$input" ]; then
        echo "  Downloading $name input..."
        curl -sL "$BASE_URL/input_images/${name}_input.png" -o "$input"
    fi
    if [ ! -f "$paper" ]; then
        echo "  Downloading $name paper 8x..."
        curl -sL "$BASE_URL/results_ours/${name}_ours_8x.png" -o "$paper"
    fi
done

echo ""
echo "=== Building test_runner ==="
cargo build --release --bin test_runner 2>&1 | tail -1

echo ""
echo "=== Running our vectorizer ==="
for name in "${NAMES[@]}"; do
    input="$OUT_DIR/${name}_input.png"
    ours="$OUT_DIR/${name}_ours_8x.png"
    sdiff="$OUT_DIR/${name}_sdiff_8x.png"

    if [ ! -f "$input" ]; then
        echo "  SKIP $name (no input)"
        continue
    fi

    if [ ! -f "$ours" ]; then
        echo "  Vectorize $name (scanline)..."
        cargo run --release --bin test_runner -- vectorize "$input" \
            --out "$ours" --format raster --scale "$SCALE" 2>/dev/null
    fi

    if [ ! -f "$sdiff" ]; then
        echo "  Vectorize $name (spline-diffusion)..."
        cargo run --release --bin test_runner -- vectorize "$input" \
            --out "$sdiff" --format spline-diffusion --scale "$SCALE" 2>/dev/null
    fi
done

echo ""
echo "=== Done ==="
echo "Results in $OUT_DIR/"
echo ""
echo "For each sprite:"
echo "  {name}_input.png       — original pixel art"
echo "  {name}_paper_8x.png    — paper's result"
echo "  {name}_ours_8x.png     — our scanline vectorizer"
echo "  {name}_sdiff_8x.png    — our spline-diffusion"
