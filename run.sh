#!/usr/bin/env bash
set -euo pipefail
KOCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KOCH_PYTHON="${KOCH_PYTHON:-/home/niel/miniconda3/envs/isaaclab/bin/python}"
# Avoid OpenBLAS thread/fork crashes during Kit startup on this installation.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec "$KOCH_PYTHON" "$KOCH_ROOT/scripts/run_scene.py" "$@"
