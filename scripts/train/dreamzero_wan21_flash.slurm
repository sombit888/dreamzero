#!/bin/bash
#SBATCH --job-name=dreamzero-flash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=72:00:00
#SBATCH --output=dreamzero-flash-%j.out
#SBATCH --error=dreamzero-flash-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Environment activation:
# - Prefer a conda env when conda is available.
# - Fall back to a local .venv in the repo root if present.
if [ -f "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
    conda activate "${DREAMZERO_CONDA_ENV:-dreamzero}"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "${DREAMZERO_CONDA_ENV:-dreamzero}"
elif [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
else
    echo "ERROR: No environment found. Set CONDA_SH/DREAMZERO_CONDA_ENV or create $REPO_ROOT/.venv"
    exit 1
fi

export DREAMZERO_ROOT=/home/sombit_dey/projects/dreamzero
export DATA_DIR=${DATA_DIR:-/data/dreamzero}
export PRETRAINED_CKPT_DIR=${PRETRAINED_CKPT_DIR:-/work/sombit_dey/pretrain_ckpt/dream0_droid}
export WAN_CKPT_DIR=${WAN_CKPT_DIR:-/home/sombit_dey/checkpoints/Wan2.1-I2V-14B-480P}
export OUTPUT_DIR=${OUTPUT_DIR:-/scratch/${USER}/dreamzero_droid_wan21_lora_flash}
export NUM_GPUS=4



srun bash scripts/train/droid_training_wan21_flash.sh