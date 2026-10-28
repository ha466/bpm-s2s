"""Run SmolLM2-135M S2S model with real voice sample and show complete Input and Output details.

Usage:
    .venv/Scripts/python scripts/test_smollm2_io.py
"""
import io
import os
import sys
import time

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import scipy.signal
import soundfile as sf
import torch
from datasets import load_dataset

from bpm.config import ModelConfig
from bpm.mimi_adapter import MimiTokenizer
from bpm.model import S2SModel
from bpm.streaming import Streamer


def main():
    print("=" * 85)
    print("       BPM-S2S WITH PRETRAINED SMOLLM2-135M: INPUT / OUTPUT TEST DEMO")
    print("=" * 85)

    ckpt_path = "runs/colab_t4_smollm2/ckpt.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "runs/smollm2_voice/ckpt.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    # 1. Load HF Sample
    print("\n[1] FETCHING REAL VOICE INPUT FROM HUGGING FACE...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", ds.features["audio"].__class__(decode=False))
    item = ds[0]
    raw_wav, orig_sr = sf.read(io.BytesIO(item["audio"]["bytes"]))
    if raw_wav.ndim > 1:
        raw_wav = raw_wav.mean(axis=1)
    transcript = item.get("text", "N/A")
    orig_dur = len(raw_wav) / orig_sr

    # Preprocess to 24 kHz, 80ms frames (1920 samples)
    target_sr = 24000
    frame_len = 1920
    target_len = (int(round(len(raw_wav) * target_sr / orig_sr)) // frame_len) * frame_len
    resampled_wav = scipy.signal.resample(raw_wav, target_len).astype(np.float32)
    peak = np.abs(resampled_wav).max()
    if peak > 0:
        resampled_wav = resampled_wav * (0.95 / peak)

    input_rms = float(np.sqrt(np.mean(resampled_wav ** 2)))
    num_frames = len(resampled_wav) // frame_len
    dur_24k = len(resampled_wav) / target_sr

    # Save input audio
    os.makedirs("runs/real_test", exist_ok=True)
    in_wav_path = "runs/real_test/input_voice_sample.wav"
    sf.write(in_wav_path, resampled_wav, target_sr)

    print("  ---------------------------------------------------------------------------")
    print(f"  Input Dataset:       hf-internal-testing/librispeech_asr_dummy (clean, validation)")
    print(f"  Spoken Transcript:   \"{transcript}\"")
    print(f"  Original Sample:     {len(raw_wav):,} samples @ {orig_sr} Hz ({orig_dur:.2f} s)")
    print(f"  Processed Input:     {len(resampled_wav):,} samples @ {target_sr} Hz ({dur_24k:.2f} s)")
    print(f"  Audio Metrics:       Peak={peak:.3f}, RMS Energy={input_rms:.4f}")
    print(f"  Total Frames:        {num_frames} frames (80 ms per frame at 12.5 Hz)")
    print(f"  Saved Audio File:    {in_wav_path}")
    print("  ---------------------------------------------------------------------------")

    # 2. Tokenize with Mimi Codec
    print("\n[2] ENCODING INPUT AUDIO TO DISCRETE MIMI TOKENS (12.5 Hz)...")
    tok = MimiTokenizer(device="cpu", num_codebooks=8)
    wav_tensor = torch.tensor(resampled_wav).unsqueeze(0).unsqueeze(0)
    in_codes = tok.encode(wav_tensor)  # (1, T, 8)
    print(f"  Input Tokens Tensor: Shape={tuple(in_codes.shape)} (B={in_codes.shape[0]}, T={in_codes.shape[1]} frames, Q={in_codes.shape[2]} codebooks)")
    print(f"  Vocabulary Range:    Tokens in [{in_codes.min().item()}, {in_codes.max().item()}] (Vocab V=2048)")
    print("\n  Sample Input Tokens (First 8 frames across all 8 codebooks):")
    print("  Frame | CB0 (Semantic) | CB1   CB2   CB3   CB4   CB5   CB6   CB7 (Acoustic)")
    print("  " + "-" * 72)
    for f in range(min(8, num_frames)):
        cb0 = in_codes[0, f, 0].item()
        ac = " ".join(f"{x:5d}" for x in in_codes[0, f, 1:].tolist())
        print(f"  {f:5d} | {cb0:14d} | {ac}")

    # 3. Load Trained SmolLM2-135M S2S Model
    print(f"\n[3] LOADING PRETRAINED SMOLLM2-135M S2S MODEL ({ckpt_path})...")
    ck = torch.load(ckpt_path, map_location="cpu")
    mcfg = ModelConfig(**ck["cfg"]["model"])
    model = S2SModel(mcfg)
    model.load_state_dict(ck["model"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model Architecture:  Backbone={mcfg.backbone} (SmolLM2-135M), K={mcfg.K}, d={mcfg.d}, Q={mcfg.Q}, V={mcfg.V}")
    print(f"  Total Parameters:    {total_params:,} (~{total_params/1e6:.1f}M)")

    # 4. Stream Through Model
    print("\n[4] RUNNING STREAMING INFERENCE (FRAME-BY-FRAME WITH CAUSAL KV CACHING)...")
    st = Streamer(model, [0] * mcfg.Q, revision_tau=None)
    out_tokens = []
    out_acts = []
    step_records = []

    print(f"\n  {'Frame':>5} | {'Time (s)':>8} | {'Input CB0':>9} | {'Input VAD':>9} | {'Output CB0':>10} | {'Agent Speaking':>14} | {'Plan Left':>9} | {'Wake Call':>9}")
    print("  " + "-" * 88)

    for t in range(num_frames):
        u_t = in_codes[0, t].tolist()
        chunk = resampled_wav[t * frame_len : (t + 1) * frame_len]
        energy = float(np.sqrt(np.mean(chunk ** 2)))
        user_speaking = energy >= 0.03

        wakes_before = st.stats["wakes"]
        own_t, own_act = st.output_frame()
        woke = st.stats["wakes"] > wakes_before

        out_tokens.append(own_t)
        out_acts.append(own_act)
        time_s = t * 0.080

        record = {
            "t": t, "time": time_s, "in_cb0": u_t[0], "user_spk": user_speaking,
            "out_cb0": own_t[0], "own_spk": own_act, "plan": len(st.plan), "woke": woke
        }
        step_records.append(record)

        if t < 12 or t % 6 == 0 or t >= num_frames - 2 or woke:
            w_str = "WAKE" if woke else "-"
            spk_str = "SPEAKING" if own_act else "silent"
            u_str = "YES" if user_speaking else "no"
            print(f"  {t:5d} | {time_s:7.2f}s | {u_t[0]:9d} | {u_str:>9} | {own_t[0]:10d} | {spk_str:>14} | {len(st.plan):9d} | {w_str:>9}")

        st.prev_active = user_speaking
        fr = st.cur
        st.pu.append(list(u_t))
        st.pa.append(fr["a"])
        st.ps.append(fr["s"])

    summary = st.summary()

    # 5. Decode Output Tokens to Audio Waveform
    print("\n[5] DECODING MODEL OUTPUT TOKENS BACK TO 24 kHz AUDIO...")
    out_codes_tensor = torch.tensor([out_tokens], dtype=torch.long)
    out_audio = tok.decode(out_codes_tensor)
    out_audio_np = out_audio[0, 0].cpu().numpy()
    out_wav_path = "runs/real_test/output_voice_response.wav"
    sf.write(out_wav_path, out_audio_np, target_sr)

    out_peak = float(np.abs(out_audio_np).max())
    out_rms = float(np.sqrt(np.mean(out_audio_np ** 2)))
    out_dur = len(out_audio_np) / target_sr

    print("  ---------------------------------------------------------------------------")
    print(f"  Output Tokens:       Shape={tuple(out_codes_tensor.shape)} ({len(out_tokens)} frames, 8 codebooks)")
    print(f"  Decoded Output:      {len(out_audio_np):,} samples @ {target_sr} Hz ({out_dur:.2f} s)")
    print(f"  Output Audio Stats:  Peak={out_peak:.3f}, RMS Energy={out_rms:.4f}")
    print(f"  Saved Output File:   {out_wav_path}")
    print("  ---------------------------------------------------------------------------")

    print("\n  Sample Output Tokens (First 8 frames across all 8 codebooks):")
    print("  Frame | CB0 (Semantic) | CB1   CB2   CB3   CB4   CB5   CB6   CB7 (Acoustic)")
    print("  " + "-" * 72)
    for f in range(min(8, len(out_tokens))):
        cb0 = out_tokens[f][0]
        ac = " ".join(f"{x:5d}" for x in out_tokens[f][1:])
        print(f"  {f:5d} | {cb0:14d} | {ac}")

    # 6. Overall Performance Summary
    print("\n" + "=" * 85)
    print("                         EXECUTION & LATENCY SUMMARY")
    print("=" * 85)
    print(f"  Total Processed Frames:    {summary['frames']} frames ({summary['frames'] * 0.08:.2f} seconds)")
    print(f"  Total Backbone Wakes:      {summary['wakes']} wakes (every 4 frames = K=4)")
    print(f"  Backbone Calls / Frame:    {summary['calls_per_frame']:.4f} (theoretical 1/K = 0.2500)")
    print(f"  Total Refiner Passes:      {summary['refiner_passes']} passes")
    print(f"  CPU Wake Latency (Mean):   {summary.get('wake_ms_mean', 0.0):.2f} ms")
    print(f"  CPU Wake Latency (p50):    {summary.get('wake_ms_p50', 0.0):.2f} ms")
    print(f"  CPU Wake Latency (p95):    {summary.get('wake_ms_p95', 0.0):.2f} ms")
    print("=" * 85)


if __name__ == "__main__":
    main()

