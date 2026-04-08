#!/bin/bash
# DreamZero DROID LoRA Flash Training Script (Wan2.1-I2V-14B-480P backbone)
#
# This script enables DreamZero-Flash style training:
# - Decoupled video/action noise schedule
# - High-noise video sampling via Beta(alpha, beta)
# - 1-step inference timestep setting for deployment alignment
#
# Required global environment variables:
#   DATA_DIR                  (example: /data/dreamzero)
#   PRETRAINED_CKPT_DIR       (example: /path/to/DreamZero-DROID)
#   WAN_CKPT_DIR              (example: /path/to/Wan2.1-I2V-14B-480P)
#
# Usage:
#   export DATA_DIR=/data/dreamzero
#   export PRETRAINED_CKPT_DIR=/path/to/DreamZero-DROID
#   export WAN_CKPT_DIR=/path/to/Wan2.1-I2V-14B-480P
#   bash scripts/train/droid_training_wan22_flash.sh

export HYDRA_FULL_ERROR=1

# Repo root: must be a directory that contains groot/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -n "$DREAMZERO_ROOT" ] && [ -d "$DREAMZERO_ROOT/groot" ]; then
    :
elif [ -d "/root/yejink/dreamzero/groot" ]; then
    DREAMZERO_ROOT=/root/yejink/dreamzero
elif [ -d "/root/dreamzero/groot" ]; then
    DREAMZERO_ROOT=/root/dreamzero
elif [ -d "$SCRIPT_REPO_ROOT/groot" ]; then
    DREAMZERO_ROOT="$SCRIPT_REPO_ROOT"
else
    DREAMZERO_ROOT="${DREAMZERO_ROOT:-/root/yejink/dreamzero}"
fi
if [ ! -d "$DREAMZERO_ROOT/groot" ]; then
    echo "ERROR: No groot/ under $DREAMZERO_ROOT. Set DREAMZERO_ROOT to the dreamzero repo root that contains groot/."
    exit 1
fi

# ============ USER CONFIGURATION ============
NUM_GPUS=${NUM_GPUS:-8}
OUTPUT_DIR=${OUTPUT_DIR:-"/scratch/${USER}/dreamzero_droid_wan21_lora_flash"}
DATA_DIR=${DATA_DIR:-/data/dreamzero}
PRETRAINED_CKPT_DIR=${PRETRAINED_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/DreamZero-DROID"}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-I2V-14B-480P"}

# Image encoder is included in Wan2.1-I2V-14B-480P.
IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"$WAN_CKPT_DIR"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$DREAMZERO_ROOT/checkpoints/umt5-xxl"}
# =============================================

# ============ AUTO-DOWNLOAD WEIGHTS ============
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-I2V-14B-480P not found at $WAN_CKPT_DIR. Downloading from HuggingFace..."
    hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    hf download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi

if [ ! -f "$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]; then
    echo "Image encoder not found. Downloading Wan2.1-I2V-14B-480P (for CLIP only)..."
    hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$IMAGE_ENCODER_DIR"
fi
# ================================================

# --- Path and File Validation ---
echo "INFO: Running validation checks..."
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: DATA_DIR is not a directory: $DATA_DIR"
    exit 1
fi
if [ ! -d "$PRETRAINED_CKPT_DIR" ]; then
    echo "ERROR: PRETRAINED_CKPT_DIR is not a directory: $PRETRAINED_CKPT_DIR"
    exit 1
fi
if [ ! -f "$WAN_CKPT_DIR/Wan2.1_VAE.pth" ]; then
    echo "ERROR: VAE file not found in $WAN_CKPT_DIR"
    exit 1
fi
mkdir -p "$OUTPUT_DIR"
echo "INFO: Validation checks passed."
# --- End Validation ---

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: DROID dataset not found at $DATA_DIR"
    echo "Set DATA_DIR to your DROID root (for example: /data/dreamzero)"
    exit 1
fi

if ! mkdir -p "$OUTPUT_DIR" 2>/dev/null; then
    echo "ERROR: Cannot create OUTPUT_DIR at $OUTPUT_DIR"
    echo "Set OUTPUT_DIR to a writable path (recommended: /scratch/${USER}/dreamzero_droid_wan21_lora_flash)"
    exit 1
fi

if [ ! -f "$WAN_CKPT_DIR/diffusion_pytorch_model.safetensors" ] && [ ! -f "$WAN_CKPT_DIR/diffusion_pytorch_model.safetensors.index.json" ]; then
    echo "ERROR: WAN_CKPT_DIR does not contain diffusion_pytorch_model.safetensors(.index.json)"
    echo "Current WAN_CKPT_DIR=$WAN_CKPT_DIR"
    echo "Set WAN_CKPT_DIR to the Wan2.1 checkpoint directory (e.g. .../checkpoints/Wan2.1-I2V-14B-480P)."
    exit 1
fi

if [ ! -f "$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" ]; then
    echo "ERROR: WAN_CKPT_DIR missing models_t5_umt5-xxl-enc-bf16.pth"
    echo "Current WAN_CKPT_DIR=$WAN_CKPT_DIR"
    exit 1
fi

if [ ! -f "$WAN_CKPT_DIR/Wan2.1_VAE.pth" ]; then
    echo "ERROR: WAN_CKPT_DIR missing Wan2.1_VAE.pth"
    echo "Current WAN_CKPT_DIR=$WAN_CKPT_DIR"
    exit 1
fi

EXPERIMENT_PY="$DREAMZERO_ROOT/groot/vla/experiment/experiment.py"
if [ ! -f "$EXPERIMENT_PY" ]; then
    echo "ERROR: Not found: $EXPERIMENT_PY"
    exit 1
fi

PYTHON_311="/usr/bin/python3.11"
if [ -x "$PYTHON_311" ]; then
    if [ -n "${FIX_NUMPY_IN_SCRIPT:-}" ]; then
        "$PYTHON_311" -m pip install "numpy==1.26.4" --force-reinstall -q 2>/dev/null || true
    fi
    RUN_CMD=( "$PYTHON_311" -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone "$EXPERIMENT_PY" )
    echo "Using image Python 3.11: $PYTHON_311"
else
    RUN_CMD=( python3 -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone "$EXPERIMENT_PY" )
    echo "Using: $(command -v python3)"
fi

if [ ! -d "$PRETRAINED_CKPT_DIR" ]; then
    echo "ERROR: PRETRAINED_CKPT_DIR not found at $PRETRAINED_CKPT_DIR"
    exit 1
fi

echo "Using DATA_DIR=$DATA_DIR"
echo "Using PRETRAINED_CKPT_DIR=$PRETRAINED_CKPT_DIR"
echo "Using WAN_CKPT_DIR=$WAN_CKPT_DIR"
echo "Using OUTPUT_DIR=$OUTPUT_DIR"

# DeepSpeed in some versions expects CUDA_VISIBLE_DEVICES to be numeric indices,
# but cluster launchers may set GPU UUIDs (for example: GPU-xxxx).
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [[ "$CUDA_VISIBLE_DEVICES" == *GPU-* ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        declare -A UUID_TO_INDEX
        while IFS=, read -r gpu_index gpu_uuid; do
            gpu_index="${gpu_index//[[:space:]]/}"
            gpu_uuid="${gpu_uuid//[[:space:]]/}"
            UUID_TO_INDEX["$gpu_uuid"]="$gpu_index"
        done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)

        mapped_devices=()
        map_failed=0
        IFS=',' read -ra requested_devices <<< "$CUDA_VISIBLE_DEVICES"
        for device in "${requested_devices[@]}"; do
            device="${device//[[:space:]]/}"
            if [ -n "${UUID_TO_INDEX[$device]+x}" ]; then
                mapped_devices+=("${UUID_TO_INDEX[$device]}")
            else
                map_failed=1
                break
            fi
        done

        if [ "$map_failed" -eq 0 ] && [ "${#mapped_devices[@]}" -gt 0 ]; then
            export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${mapped_devices[*]}")"
            echo "INFO: Converted CUDA_VISIBLE_DEVICES UUIDs to indices for DeepSpeed: $CUDA_VISIBLE_DEVICES"
        else
            echo "WARN: Could not fully map CUDA_VISIBLE_DEVICES UUIDs to indices; keeping original value."
        fi
    else
        echo "WARN: nvidia-smi not found; cannot convert CUDA_VISIBLE_DEVICES UUIDs for DeepSpeed."
    fi
fi

cd "$DREAMZERO_ROOT"
torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/droid_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=1000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=4 \
    max_steps=10000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=no \
    droid_data_root=$DATA_DIR \
    pretrained_model_path=$PRETRAINED_CKPT_DIR \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    action_head_cfg.config.decouple_video_action_noise=true \
    action_head_cfg.config.video_noise_beta_alpha=7.0 \
    action_head_cfg.config.video_noise_beta_beta=1.0 \
    action_head_cfg.config.num_inference_timesteps=1