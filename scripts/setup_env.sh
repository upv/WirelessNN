#!/usr/bin/env bash
# Set up a fresh machine: virtual environment, dependencies, installation check.
#   scripts/setup_env.sh                 everything (simulator + analysis + tests)
#   EXTRAS=dev scripts/setup_env.sh      simulator and tests only
#   PYTHON=python3.12 scripts/setup_env.sh
# Safe to run again: an existing .venv is reused.
#
# PyTorch comes from PyPI with its default CUDA build. If the check at the end reports
# "GPU none visible" on a GPU machine, install the PyTorch wheel matching the CUDA
# driver into .venv (selector: https://pytorch.org/get-started/locally/) and re-run.
set -euo pipefail
cd "$(dirname "$0")/.."   # repository root

PYTHON=${PYTHON:-python3}
EXTRAS=${EXTRAS:-analysis,dev}

if [[ ! -x .venv/bin/python ]]; then
  echo "creating .venv with $($PYTHON --version)"
  "$PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[$EXTRAS]"

echo
.venv/bin/python scripts/check_install.py
