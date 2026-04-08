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

# Setup and activate environment inside the job.
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/setup_env.sh"

export DREAMZERO_ROOT=/home/sombit_dey/projects/dreamzero
export DATA_DIR=${DATA_DIR:-/data/dreamzero}
export PRETRAINED_CKPT_DIR=${PRETRAINED_CKPT_DIR:-/work/sombit_dey/pretrain_ckpt/dream0_droid}
export WAN_CKPT_DIR=${WAN_CKPT_DIR:-/home/sombit_dey/checkpoints/Wan2.1-I2V-14B-480P}
RUN_TS=$(date +%Y%m%d_%H%M%S)
export OUTPUT_DIR=${OUTPUT_DIR:-/scratch/${USER}/dreamzero_droid_wan21_lora_flash_${RUN_TS}}
export NUM_GPUS=4



srun bash scripts/train/droid_training_wan21_flash.sh