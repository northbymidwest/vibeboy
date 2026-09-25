#!/usr/bin/env bash
# Run every test ROM suite through test_runner and compare the per-test
# results against the checked-in baseline, tests/accuracy-baseline.txt.
#
# The baseline records the status of each test, one per line:
#   <suite> TAB <path under the suite directory> TAB <PASS|FAIL|TIMEOUT|ERR>
#
# A test that passes in the baseline and no longer passes (or no longer
# runs) is lost; a test that passes now and did not before is gained. The
# script exits 1 when anything is lost, so accuracy work that trades tests
# has to say so by updating the baseline in the same commit. Gains alone
# exit 0, but leave the baseline stale until --update records them.
#
# Usage:
#   ./scripts/accuracy.sh                  # run, compare, report
#   ./scripts/accuracy.sh --update         # run and rewrite the baseline
#
# Options:
#   --update             Rewrite the baseline from this run instead of comparing.
#   --markdown FILE      Append a markdown report to FILE (CI job summary).
#   --results DIR        Keep raw per-suite output in DIR (default: a temp dir).
#   --roms DIR           Test ROM directory (default: game-boy-test-roms).
#   --no-build           Use target/release/test_runner as is.
#
# Exit status: 0 when nothing is lost, 1 when a test is lost, 2 when a suite
# could not run at all (missing ROMs, a crash) or on bad usage.

set -euo pipefail

cd "$(dirname "$0")/.."

BASELINE=tests/accuracy-baseline.txt
TEST_RUNNER=target/release/test_runner
ROMS=game-boy-test-roms
UPDATE=false
BUILD=true
MARKDOWN=
RESULTS=

usage() { sed -n '2,/^$/s/^# \{0,1\}//p' "$0" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --update)   UPDATE=true ;;
    --no-build) BUILD=false ;;
    --markdown) [ $# -ge 2 ] || usage; MARKDOWN=$2; shift ;;
    --results)  [ $# -ge 2 ] || usage; RESULTS=$2; shift ;;
    --roms)     [ $# -ge 2 ] || usage; ROMS=$2; shift ;;
    -h|--help)  usage ;;
    *)          echo "Unknown argument: $1" >&2; usage ;;
  esac
  shift
done

if [ ! -d "$ROMS" ]; then
  echo "No test ROMs at '$ROMS'. Run ./scripts/fetch-test-roms.sh first." >&2
  exit 2
fi

if [ "$BUILD" = true ]; then
  cargo build --release --locked --bin test_runner
fi

if [ -z "$RESULTS" ]; then
  RESULTS=$(mktemp -d)
  trap 'rm -rf "$RESULTS"' EXIT
fi
mkdir -p "$RESULTS"

# name, harness, directory under $ROMS, extra test_runner arguments.
SUITES=(
  "mooneye-acceptance    mooneye     mooneye-test-suite/acceptance"
  "mooneye-emulator-only mooneye     mooneye-test-suite/emulator-only"
  "mooneye-misc          mooneye     mooneye-test-suite/misc"
  "wilbertpol-acceptance mooneye     mooneye-test-suite-wilbertpol/acceptance"
  "wilbertpol-misc       mooneye     mooneye-test-suite-wilbertpol/misc"
  "blargg                blargg      blargg"
  "gambatte              gambatte    gambatte"
  "same-suite            mooneye     same-suite"
  "gbmicrotest           gbmicrotest gbmicrotest"
  "tearoom-dmg           tearoom     mealybug-tearoom-tests"
  "tearoom-cgb           tearoom     mealybug-tearoom-tests --model cgb"
)

# The suites are independent and each is single threaded, so they run side
# by side. Each writes its own files; the report reads them in list order.
pids=()
for suite in "${SUITES[@]}"; do
  # shellcheck disable=SC2086 # word splitting of the suite line is the point
  set -- $suite
  name=$1 harness=$2 dir=$3; shift 3
  "$TEST_RUNNER" test "$harness" "$ROMS/$dir/" --allow-failures "$@" \
    >"$RESULTS/$name.txt" 2>"$RESULTS/$name.err" &
  pids+=("$!")
done

broken=()
i=0
for suite in "${SUITES[@]}"; do
  name=${suite%% *}
  if ! wait "${pids[$i]}"; then
    broken+=("$name")
  fi
  i=$((i + 1))
done

# One line per test, the baseline's format, sorted bytewise so the file
# diffs cleanly whatever the locale.
current="$RESULTS/accuracy.txt"
for suite in "${SUITES[@]}"; do
  name=${suite%% *}
  sed -nE "s/^(PASS|FAIL|TIMEOUT|ERR) +(.+)$/$name	\2	\1/p" "$RESULTS/$name.txt"
done | LC_ALL=C sort >"$current"

if [ ${#broken[@]} -gt 0 ]; then
  for name in "${broken[@]}"; do
    echo "Suite $name did not complete. Its stderr:" >&2
    tail -n 20 "$RESULTS/$name.err" >&2
  done
  if [ -n "$MARKDOWN" ]; then
    {
      echo '## Test ROM accuracy'
      echo
      echo "Suites that did not complete: ${broken[*]}"
    } >>"$MARKDOWN"
  fi
  exit 2
fi

if [ "$UPDATE" = true ]; then
  cp "$current" "$BASELINE"
  echo "Wrote $BASELINE ($(grep -c '	PASS$' "$BASELINE") of $(wc -l <"$BASELINE" | tr -d ' ') passing)"
  exit 0
fi

if [ ! -f "$BASELINE" ]; then
  echo "No baseline at $BASELINE. Run with --update to create it." >&2
  exit 2
fi

# gained: passes now, did not pass (or did not exist) in the baseline.
# lost:   passed in the baseline, does not pass (or did not run) now.
# Each line is "<suite>/<path>  <before> -> <after>".
compare() {
  awk -F'\t' -v want="$1" '
    FNR == NR { base[$1 "\t" $2] = $3; next }
    { now[$1 "\t" $2] = $3 }
    END {
      for (k in now) {
        b = (k in base) ? base[k] : "absent"
        if (want == "gained" && now[k] == "PASS" && b != "PASS") print k "\t" b "\t" now[k]
      }
      for (k in base) {
        n = (k in now) ? now[k] : "absent"
        if (want == "lost" && base[k] == "PASS" && n != "PASS") print k "\t" base[k] "\t" n
      }
    }' "$BASELINE" "$current" |
    LC_ALL=C sort |
    awk -F'\t' '{ print $1 "/" $2 "  " $3 " -> " $4 }'
}
gained=$(compare gained)
lost=$(compare lost)
count() { if [ -z "$1" ]; then echo 0; else printf '%s\n' "$1" | wc -l | tr -d ' '; fi; }
n_gained=$(count "$gained")
n_lost=$(count "$lost")

# Per-suite pass counts, now and in the baseline.
table=$(
  for suite in "${SUITES[@]}"; do
    name=${suite%% *}
    awk -F'\t' -v s="$name" '
      FNR == NR { if ($1 == s) { bt++; if ($3 == "PASS") bp++ } next }
      $1 == s { t++; if ($3 == "PASS") p++ }
      END { printf "%s\t%d/%d\t%d/%d\n", s, p, t, bp, bt }' "$BASELINE" "$current"
  done
)
total=$(awk -F'\t' '{ t++; if ($3 == "PASS") p++ } END { printf "%d/%d", p, t }' "$current")
base_total=$(awk -F'\t' '{ t++; if ($3 == "PASS") p++ } END { printf "%d/%d", p, t }' "$BASELINE")

printf '%-24s %-12s %s\n' Suite Passing Baseline
printf '%s\n' "$table" | awk -F'\t' '{ printf "%-24s %-12s %s\n", $1, $2, $3 }'
printf '%-24s %-12s %s\n' total "$total" "$base_total"
echo
echo "$n_gained gained, $n_lost lost against $BASELINE"
if [ "$n_gained" -gt 0 ]; then
  echo
  echo "Gained:"
  printf '  %s\n' "$gained"
fi
if [ "$n_lost" -gt 0 ]; then
  echo
  echo "Lost:"
  printf '  %s\n' "$lost"
fi
if [ "$n_gained" -gt 0 ] && [ "$n_lost" -eq 0 ]; then
  echo
  echo "Run ./scripts/accuracy.sh --update to record the gains."
fi

if [ -n "$MARKDOWN" ]; then
  {
    echo '## Test ROM accuracy'
    echo
    echo "**$n_gained gained, $n_lost lost** against \`$BASELINE\`."
    echo
    echo '| Suite | Passing | Baseline |'
    echo '| --- | --- | --- |'
    printf '%s\n' "$table" | awk -F'\t' '{ print "| " $1 " | " $2 " | " $3 " |" }'
    echo "| **total** | **$total** | **$base_total** |"
    for kind in Lost Gained; do
      if [ "$kind" = Lost ]; then list=$lost; else list=$gained; fi
      [ -n "$list" ] || continue
      echo
      echo "### $kind"
      echo
      echo '```'
      printf '%s\n' "$list"
      echo '```'
    done
  } >>"$MARKDOWN"
fi

[ "$n_lost" -eq 0 ]
