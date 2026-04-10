#!/bin/bash
#SBATCH --job-name=dreamzero-flash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-gpu=12

#SBATCH --mem=550G
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/%u/dreamzero_wan21_flash_%j.out
#SBATCH --error=/scratch/%u/dreamzero_wan21_flash_%j.err

set -euo pipefail

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# cd "$REPO_ROOT"
export REPO_ROOT=/home/sombit_dey/projects/dreamzero
# Setup and activate environment inside the job.
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/setup_env.sh"
cd $REPO_ROOT  
source .venv/bin/activate
export DREAMZERO_ROOT=/home/sombit_dey/projects/dreamzero
export DATA_DIR=${DATA_DIR:-/data/dreamzero}
export PRETRAINED_CKPT_DIR=${PRETRAINED_CKPT_DIR:-/work/sombit_dey/pretrain_ckpt/dream0_droid}
export WAN_CKPT_DIR=${WAN_CKPT_DIR:-/home/sombit_dey/checkpoints/Wan2.1-I2V-14B-480P}
RUN_TS=$(date +%Y%m%d_%H%M%S)
export OUTPUT_DIR=${OUTPUT_DIR:-/scratch/${USER}/dreamzero_droid_wan21_lora_flash_${RUN_TS}}
export NUM_GPUS=4

bash scripts/train/droid_training_wan21_flash.sh
