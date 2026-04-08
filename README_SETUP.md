# DreamZero Installation

This document provides instructions for installing the DreamZero model and its dependencies.

## Installation Steps

1.  **Load CUDA Module**:
    Load the required CUDA module.

    ```bash
    module load nvidia-cuda-12.9.0
    ```

2.  **Set Environment Variables**:
    Set the necessary environment variables for CUDA.

    ```bash
    export CUDA_HOME=/opt/modules/nvidia-cuda-12.9.0
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    ```
    *Note: Replace `<path_to_your_cuda_12.1_installation>` with the actual path to your CUDA installation.*

3.  **Install Dependencies**:
    Use `uv` to install the required Python packages from `pyproject.toml`. Much faster to install f

    ```bash
    uv sync
    source .venv/bin/activate
    MAX_JOBS=8 uv  pip install --no-build-isolation flash-attn --no-cache-dir

    ```

5. **Download the Droid CKPT**
    ```
    hf download GEAR-Dreams/DreamZero-DROID --repo-type model --local-dir <path/to/checkpoint>
    ```

## Inference Steps
    ```
    uv run --project examples/dream0 python -m robot_inference.client.run_model --config examples/dream0/dream0_robot.yaml

    ```