#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASESET="${1:-TRIANGULATION_BASE_SET}"
SEED="${2:-0}"
PER="${3:-true}"
LSTM="${4:-false}"

julia --project=. src/main.jl \
  --baseset "$BASESET" \
  --seed "$SEED" \
  --PER="$PER" \
  --LSTM="$LSTM"
