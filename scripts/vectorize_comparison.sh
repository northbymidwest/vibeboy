#!/bin/bash
# Download input images and paper's "ours" results at 8x from the
# Kopf-Lischinski supplementary page, then run our vectorizer on
# each input for side-by-side comparison.
#
# Usage: ./scripts/vectorize_comparison.sh
#        ./scripts/vectorize_comparison.sh --force   # re-render all
#
# Output: vectorize-tests/
#   {name}_input.png       — original pixel art
#   {name}_paper_8x.png    — paper's result at 8x
#   {name}_ours_8x.png     — our scanline vectorizer at 8x
#   {name}_sdiff_8x.png    — our spline-diffusion at 8x
#   comparison.html        — side-by-side comparison page

set -e

FORCE=false
[ "$1" = "--force" ] && FORCE=true

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

    if [ ! -f "$ours" ] || [ "$FORCE" = true ]; then
        echo "  Vectorize $name (scanline)..."
        cargo run --release --bin test_runner -- vectorize "$input" \
            --out "$ours" --format raster --scale "$SCALE" 2>/dev/null
    fi

    if [ ! -f "$sdiff" ] || [ "$FORCE" = true ]; then
        echo "  Vectorize $name (spline-diffusion)..."
        cargo run --release --bin test_runner -- vectorize "$input" \
            --out "$sdiff" --format spline-diffusion --scale "$SCALE" 2>/dev/null
    fi
done

echo ""
echo "=== Generating comparison HTML ==="

HTML="$OUT_DIR/comparison.html"
cat > "$HTML" << 'HEADER'
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Vectorize Comparison — vibeboy vs Kopf-Lischinski</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 2rem; }
  h1 { text-align: center; margin-bottom: 0.5rem; font-size: 1.8rem; color: #fff; }
  .subtitle { text-align: center; color: #888; margin-bottom: 2rem; font-size: 0.9rem; }
  .legend { display: flex; justify-content: center; gap: 2rem; margin-bottom: 2rem; font-size: 0.85rem; }
  .legend span { display: flex; align-items: center; gap: 0.4rem; }
  .legend .dot { width: 12px; height: 12px; border-radius: 3px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(700px, 1fr)); gap: 2rem; }
  .card { background: #16213e; border-radius: 12px; overflow: hidden; border: 1px solid #2a2a4a; }
  .card-header { padding: 0.8rem 1.2rem; background: #0f3460; font-weight: 600; font-size: 1rem; }
  .card-body { display: grid; grid-template-columns: auto 1fr 1fr 1fr; gap: 1px; background: #2a2a4a; }
  .card-body > div { background: #16213e; display: flex; align-items: center; justify-content: center; padding: 0.5rem; }
  .card-body .label { font-size: 0.75rem; color: #888; writing-mode: vertical-rl; text-orientation: mixed; padding: 0.3rem; }
  .card-body img { display: block; max-width: 100%; height: auto; image-rendering: auto; }
  .card-body img.input { image-rendering: pixelated; }
  .col-header { font-size: 0.7rem; color: #aaa; text-transform: uppercase; letter-spacing: 0.05em; padding: 0.4rem !important; text-align: center; font-weight: 600; }
  @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>Vectorize Comparison</h1>
<p class="subtitle">vibeboy Kopf-Lischinski implementation vs paper's reference output — all at 8×</p>
<div class="legend">
  <span><span class="dot" style="background:#e94560"></span> Input (pixelated)</span>
  <span><span class="dot" style="background:#0f3460"></span> Paper (Kopf &amp; Lischinski)</span>
  <span><span class="dot" style="background:#533483"></span> Ours — Scanline</span>
  <span><span class="dot" style="background:#1a5276"></span> Ours — Spline Diffusion</span>
</div>
<div class="grid">
HEADER

for name in "${NAMES[@]}"; do
    input="${name}_input.png"
    paper="${name}_paper_8x.png"
    ours="${name}_ours_8x.png"
    sdiff="${name}_sdiff_8x.png"

    # Skip if any file missing
    [ ! -f "$OUT_DIR/$input" ] && continue

    display_name=$(echo "$name" | tr '_' ' ')

    cat >> "$HTML" << CARD
<div class="card">
  <div class="card-header">$display_name</div>
  <div class="card-body">
    <div class="col-header"></div>
    <div class="col-header">Paper</div>
    <div class="col-header">Ours (scanline)</div>
    <div class="col-header">Ours (spline diffusion)</div>
    <div class="label">Input</div>
    <div colspan="3" style="grid-column: 2 / -1;"><img class="input" src="$input"></div>
  </div>
  <div class="card-body">
    <div class="col-header"></div>
    <div class="col-header">Paper 8×</div>
    <div class="col-header">Scanline 8×</div>
    <div class="col-header">Spline Diffusion 8×</div>
    <div class="label">8×</div>
    <div><img src="$paper"></div>
    <div><img src="$ours"></div>
    <div><img src="$sdiff"></div>
  </div>
</div>
CARD
done

cat >> "$HTML" << 'FOOTER'
</div>
</body>
</html>
FOOTER

echo ""
echo "=== Done ==="
echo "Results in $OUT_DIR/"
echo "Open $HTML in a browser to compare."
