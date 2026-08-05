#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASESETS=(
  "TRIANGULATION_BASE_SET"
  "RELATIVE_POSE_BASE_SET"
  "N_SITE_PHOSPHORYLATION_BASE_SET"
)

SEEDS=(0 1 2 3 4)
METHODS=("sa" "rs")

for baseset in "${BASESETS[@]}"; do
  for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "Running $method on $baseset with seed $seed..."
      julia --project=. src/baselines.jl --method="$method" --baseset="$baseset" --seed="$seed"
    done
  done
done

echo "All baseline experiments completed."
