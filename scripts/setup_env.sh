#!/usr/bin/env bash
set -euo pipefail

# DreamZero environment setup
# Usage:
#   source scripts/setup_env.sh
# Optional env vars:
#   CUDA_MODULE=nvidia-cuda-12.9.0
#   CUDA_HOME_OVERRIDE=/path/to/cuda
#   MAX_JOBS=8
#   SKIP_FLASH_ATTN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CUDA_MODULE="${CUDA_MODULE:-nvidia-cuda-12.9.0}"
CUDA_HOME_DEFAULT="/opt/modules/${CUDA_MODULE}"
CUDA_HOME="${CUDA_HOME_OVERRIDE:-$CUDA_HOME_DEFAULT}"
MAX_JOBS="${MAX_JOBS:-8}"
SKIP_FLASH_ATTN="${SKIP_FLASH_ATTN:-0}"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "WARN: script is being executed, not sourced."
  echo "      Use: source scripts/setup_env.sh"
fi

# Load CUDA module if module command exists.
if command -v module >/dev/null 2>&1; then
  module load "$CUDA_MODULE" || true
fi

# Export CUDA paths if CUDA_HOME exists.
if [[ -d "$CUDA_HOME" ]]; then
  export CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
else
  echo "WARN: CUDA_HOME directory not found at $CUDA_HOME"
  echo "      Set CUDA_HOME_OVERRIDE to your CUDA installation if needed."
fi

# Ensure uv is available.
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing uv into user site..."
  python3 -m pip install --user uv
  export PATH="$HOME/.local/bin:$PATH"
fi

# Create/refresh virtual environment from pyproject.toml dependencies.
uv sync

# Activate local virtual environment.
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

# Install flash-attn (build from source) unless skipped.
if [[ "$SKIP_FLASH_ATTN" != "1" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    MAX_JOBS="$MAX_JOBS" uv pip install --no-build-isolation flash-attn --no-cache-dir
  else
    echo "WARN: nvcc not found; skipping flash-attn install."
    echo "      Set SKIP_FLASH_ATTN=1 to suppress this warning."
  fi
else
  echo "Skipping flash-attn install because SKIP_FLASH_ATTN=1"
fi

echo "Environment ready."
echo "Python: $(python --version 2>&1)"
echo "uv: $(uv --version 2>&1)"
if command -v nvcc >/dev/null 2>&1; then
  echo "CUDA: $(nvcc --version | tail -n 1)"
fi
