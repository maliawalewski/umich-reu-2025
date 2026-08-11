#!/bin/bash
# run_wnt_baselines.sh

# Stop on first error
set -e

# Number of seeds to run
SEEDS="0 1 2 3 4"
BASESET="WNT_BASE_SET"

echo "Starting baseline experiments for $BASESET on Google Cloud"

for seed in $SEEDS; do
    echo "=================================================="
    echo "Running Random Search (RS) | Seed: $seed"
    echo "=================================================="
    julia src/baselines.jl --baseset $BASESET --method rs --seed $seed

    echo "=================================================="
    echo "Running Simulated Annealing (SA) | Seed: $seed"
    echo "=================================================="
    julia src/baselines.jl --baseset $BASESET --method sa --seed $seed
done

echo "All WNT baseline experiments finished!"
