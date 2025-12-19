#!/usr/bin/env bash

set -e

PYTHON_BIN=${PYTHON_BIN:-python}

echo ">>> Using python: $PYTHON_BIN"
"$PYTHON_BIN" -m pip install --upgrade pip

"$PYTHON_BIN" -m pip install numpy matplotlib open3d scipy PyQt6

echo ">>> Done. If you have multiple Python versions, you can run:"
echo "    PYTHON_BIN=python3.10 ./install.sh"

