# BPM-S2S: Block-Parallel Multi-rate speech-to-speech prototype

A research prototype for one question: **can a small streaming model generate its reply in parallel blocks
(instead of strictly frame-by-frame) without losing conversational quality?** It operates on discrete
codec tokens (Mimi layout: 12.5 Hz, 8 codebooks x 2048). It is *not* a speech model yet.

## Component status (read this first)

| Component | File | Status |
|---|---|---|
| Sliding-window causal attention + KV cache, GRU backbone, Pretrained SmolLM2-135M | `bpm/modules.py` | **Working, unit-tested** (chunked streaming == full forward) |
| Block head, masked parallel acoustic refiner, AR depth baseline | `bpm/modules.py`, `bpm/model.py` | **Working, unit-tested**; trained on synthetic tokens only |
| Streaming engine (block plan, revision, activity re-plan, cancel) | `bpm/streaming.py` | **Working, unit-tested** on token streams |
| Param / memory / FLOP budget | `bpm/budget.py` | **Working**; params exact, memory/FLOPs are estimates |
| Synthetic task + closed-loop metrics | `bpm/data.py` | **Working**; validates *mechanics only* |
| Training / evaluation / benchmark scripts | `scripts/` | **Working** on synthetic tokens |
| Mimi audio<->token adapter | `bpm/mimi_adapter.py` | **Working, tested with Hugging Face & real audio** |
| Real-audio dataset loader, real-data training loop | `bpm/data.py` (`RealVoiceTask`), `scripts/train.py` (`--data real`) | **Working, unit-tested** (trains on real Mimi voice tokens) |
| Text/inner-monologue auxiliary loss | - | **Not implemented** (specified in the report only) |
| Human evaluation, public-benchmark evaluation | - | **Not implemented** |

External dependencies: PyTorch, NumPy, PyYAML, pytest. Real audio: `moshi`, `huggingface_hub`, `datasets`, `soundfile`, `scipy`.

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest tests -q                                   # 24 tests, ~7 s on CPU
python scripts/param_budget.py configs/ultra_small.yaml     # exact params + estimated memory/FLOPs
./run_smoke.sh                                              # trains bpm, ar, bpm_k8, bpm_gru (~40 min, 1 vCPU)
python scripts/train.py --config configs/real_voice.yaml --data real --out runs/real_voice  # trains on real HF voice data (CPU)
python scripts/evaluate.py --ckpt runs/bpm/ckpt.pt          # inference-policy ablations
python scripts/infer_stream.py --ckpt runs/bpm/ckpt.pt --cancel_at 18
python scripts/infer_real_sample.py                         # test model with real voice sample from HF
python scripts/benchmark.py --config configs/ultra_small.yaml            # wake latency, random weights
python scripts/benchmark.py --config configs/ultra_small.yaml --kind ar  # matched AR baseline
```

## What the smoke test does and does not show

The synthetic task ("say the user's words in reverse, in the user's voice") exercises: memory, response
length/stop, response timing, acoustic dependence on same-frame semantic tokens and on speaker, and
barge-in cancellation. Passing it shows the code and the training objective behave as designed.
It says **nothing** about speech quality, open-domain knowledge, or real-audio latency.

## Run in Google Colab (Free T4 GPU)

Run BPM-S2S with SmolLM2-135M on a free Google Colab T4 GPU instance:

```python
# 1. Clone your GitHub repository & navigate into it
!git clone https://github.com/ha466/bpm-s2s.git
%cd bpm-s2s

# 2. Run the all-in-one Colab setup (uses uv for lightning-fast install), training, and I/O test
!bash colab.sh
```

Or run step-by-step directly in Colab notebook cells:

```python
!pip install -q uv
!uv pip install --system -r requirements.txt
!python scripts/train.py --config configs/colab_t4_smollm2.yaml --data real
!python scripts/test_smollm2_io.py
```

To train on 2 to 5 hours of clean 24 kHz speech using **LibriTTS-R** (recommended best dataset for Mimi):

```python
!bash colab.sh parler-tts/libritts_r_filtered configs/colab_t4_smollm2.yaml
```

### Recommended Datasets for 2 to 5 Hours

1. **`parler-tts/libritts_r_filtered` (Recommended Best)**:
   - Native **24 kHz** (matches Mimi codec rate natively with zero resampling distortion).
   - High-fidelity studio speech, diverse speakers, CC-BY-4.0 license.
2. **`librispeech_asr` (`clean` split)**:
   - Clean, diverse English speech with aligned transcripts.
3. **`hf-internal-testing/librispeech_asr_dummy`**:
   - Lightweight test dataset for instant smoke testing and verification.

## Getting real data and checkpoints (not done here)

* Mimi codec: distributed with `kyutai/moshiko-pytorch-bf16` (CC-BY-4.0 per model card).
* Conversational audio: Fisher/Switchboard need an LDC licence (8 kHz telephone speech); Seamless
  Interaction is CC BY-NC 4.0 (non-commercial). Verify terms before use; see the report, section 8.
