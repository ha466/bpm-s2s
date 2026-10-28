#!/usr/bin/env bash
# ==============================================================================
# BPM-S2S: Google Colab T4 GPU Setup & Training Script (Fast uv installation)
# ==============================================================================
set -e

echo "=============================================================================="
echo "          BPM-S2S: GOOGLE COLAB T4 GPU SETUP & TRAINING (UV)"
echo "=============================================================================="

# 1. GPU Check
echo -e "\n[1/5] Checking GPU acceleration..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "Warning: nvidia-smi not found. Running on CPU mode."
fi

# 2. Install system audio libraries and uv
echo -e "\n[2/5] Installing system audio libraries & uv..."
apt-get update -qq && apt-get install -y -qq libportaudio2 libsndfile1 ffmpeg > /dev/null 2>&1 || true

# Remove conflicting Colab packages (torchvision & torchcodec cause C++ ABI mismatches on Colab)
pip uninstall -y torchvision torchcodec > /dev/null 2>&1 || true

if ! command -v uv &> /dev/null; then
    pip install -q uv
fi

# 3. Fast package installation using uv
echo -e "\n[3/5] Installing Python dependencies with uv..."
uv pip install --system -r requirements.txt
pip uninstall -y torchvision torchcodec > /dev/null 2>&1 || true

# 4. Unit Test Verification
echo -e "\n[4/5] Running test suite to verify environment..."
python -m pytest tests -q

# 5. Train Model on T4 GPU with Real Voice Data
echo -e "\n[5/5] Starting BPM-S2S training with SmolLM2-135M backbone on T4 GPU..."
DATASET_NAME="${1:-hf-internal-testing/librispeech_asr_dummy}"
CONFIG_PATH="${2:-configs/colab_t4_smollm2.yaml}"
MAX_HOURS="${3:-3.5}"

echo "  Dataset:   $DATASET_NAME"
echo "  Config:    $CONFIG_PATH"
echo "  Max Hours: $MAX_HOURS hours (streaming mode: stops under 5 hours)"

python scripts/train.py \
    --config "$CONFIG_PATH" \
    --data real \
    --dataset "$DATASET_NAME" \
    --split validation \
    --max_hours "$MAX_HOURS" \
    --out runs/colab_t4_smollm2

# Run Streaming Inference Demo
echo -e "\nRunning streaming inference demo..."
python scripts/test_smollm2_io.py

echo -e "\n=============================================================================="
echo "  COLAB TRAINING & INFERENCE COMPLETE!"
echo "  Trained checkpoint: runs/colab_t4_smollm2/ckpt.pt"
echo "  Generated audio:    runs/real_test/output_voice_response.wav"
echo "=============================================================================="

