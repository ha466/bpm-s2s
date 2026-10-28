#!/usr/bin/env bash
# ==============================================================================
# BPM-S2S: Google Colab T4 GPU Setup & Training Script
# ==============================================================================
set -e

echo "=============================================================================="
echo "          BPM-S2S: GOOGLE COLAB T4 GPU SETUP & TRAINING"
echo "=============================================================================="

# 1. GPU Check
echo -e "\n[1/5] Checking GPU acceleration..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "Warning: nvidia-smi not found. Running on CPU mode."
fi

# 2. Dependency Installation
echo -e "\n[2/5] Installing dependencies..."
pip install -q --upgrade pip
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 || pip install -q torch
pip install -q moshi huggingface_hub datasets soundfile scipy transformers pyyaml pytest

# 3. Unit Test Verification
echo -e "\n[3/5] Running test suite to verify environment..."
python -m pytest tests -q

# 4. Train Model on T4 GPU with Real Voice Data
echo -e "\n[4/5] Starting BPM-S2S training with SmolLM2-135M backbone on T4 GPU..."
# Recommended best dataset for 2-5h training:
# - 'parler-tts/libritts_r_filtered' (24 kHz studio speech, CC-BY-4.0)
# - 'hf-internal-testing/librispeech_asr_dummy' (lightweight test set)
DATASET_NAME="${1:-hf-internal-testing/librispeech_asr_dummy}"
CONFIG_PATH="${2:-configs/colab_t4_smollm2.yaml}"

echo "  Dataset: $DATASET_NAME"
echo "  Config:  $CONFIG_PATH"

python scripts/train.py \
    --config "$CONFIG_PATH" \
    --data real \
    --dataset "$DATASET_NAME" \
    --split validation \
    --out runs/colab_t4_smollm2

# 5. Run Inference Test & Generate Audio Response
echo -e "\n[5/5] Running streaming inference test and generating output audio..."
python scripts/test_smollm2_io.py

echo -e "\n=============================================================================="
echo "  COLAB TRAINING & INFERENCE COMPLETE!"
echo "  Trained checkpoint: runs/colab_t4_smollm2/ckpt.pt"
echo "  Generated audio:    runs/real_test/output_voice_response.wav"
echo "=============================================================================="
