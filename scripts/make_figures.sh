#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Args:
#   $1 = results dir (optional)  [default: src/results]
#   $2 = out dir (optional)      [default: figures]
#   $3 = training window (opt)   [default: 200]
RESULTS_DIR="${1:-src/results}"
OUT_DIR="${2:-figures}"
TRAIN_WINDOW="${3:-200}"

PLOT_MAIN="src/extra/MakePlots/main.py"
SRC_DIR="src"

mkdir -p "$OUT_DIR"

if [ "$RESULTS_DIR" = "results" ] && [ -d "src/results" ] && [ ! -d "results" ]; then
  RESULTS_DIR="src/results"
fi

shopt -s nullglob
files=("$RESULTS_DIR"/td3_run_baseset_*_seed_*_test_metrics.csv)
if [ ${#files[@]} -eq 0 ]; then
  echo "No test_metrics CSVs found in: $RESULTS_DIR" >&2
  echo "Tip: use: bash scripts/make_figures.sh src/results figures" >&2
  exit 1
fi

BASESETS=$(printf "%s\n" "${files[@]}" \
  | sed -E 's@.*/td3_run_baseset_(.+)_seed_[0-9]+_test_metrics\.csv@\1@' \
  | sort -u)

echo "Repo root  : $REPO_ROOT"
echo "Src dir    : $SRC_DIR"
echo "Results dir: $RESULTS_DIR"
echo "Output dir : $OUT_DIR"
echo "Train plot : enabled (mode=delta, xaxis=episode, window=$TRAIN_WINDOW)"
echo "Found basesets:"
echo "$BASESETS" | sed 's/^/  - /'
echo

for b in $BASESETS; do
  echo "=== $b ==="
  python "$PLOT_MAIN" \
    --baseset "$b" \
    --src "$SRC_DIR" \
    --outdir "$OUT_DIR" \
    --quiet-scan \
    --make-training-plot \
    --training-mode delta \
    --training-xaxis episode \
    --training-window "$TRAIN_WINDOW" \
    --make-test-delta-plots \
    --make-runtime-ecdf \
    --runtime-ecdf-clip 0.01
  echo
done

echo "Wrote figures to: $OUT_DIR"
