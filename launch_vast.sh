#!/usr/bin/env bash
# ==============================================================================
# YarnBall SFT Training Launcher for Vast.ai (Datacenter GPU: A100 / H100)
# ==============================================================================
set -euo pipefail

echo "======================================================================"
echo "STARTING YARNBALL QWEN2.5-7B SFT LAUNCHER"
echo "======================================================================"

# 1. Check GPU environment
nvidia-smi

# 2. Check Python environment & dependencies
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Try installing flash-attn if compilation tools are available
python3 -m pip install flash-attn --no-build-isolation || echo "FlashAttention-2 compilation skipped, fallback to SDPA."

# 3. Check Hugging Face authentication
if [ -n "${HUGGINGFACE_FULL_ACCESS_TOKEN_01:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
    export HF_TOKEN="${HUGGINGFACE_FULL_ACCESS_TOKEN_01}"
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Warning: No HF token detected in environment. Loading from .env if present..."
fi

# 4. Optional Dry-Run verification (pass --dry-run as script argument)
if [[ "${1:-}" == "--dry-run" ]]; then
    echo "Executing dry-run verification..."
    python3 train.py --dry-run
    exit 0
fi

# 5. Launch Training
echo "Launching Full SFT Training Run..."
python3 train.py

echo "======================================================================"
echo "TRAINING COMPLETE: ADAPTER PUSHED TO HUGGING FACE"
echo "======================================================================"
