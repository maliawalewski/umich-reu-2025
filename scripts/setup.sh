#!/usr/bin/env bash
set -euo pipefail

echo "Julia: instantiate + precompile..."
julia --project=. -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'

echo "Python: install requirements..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Done."
