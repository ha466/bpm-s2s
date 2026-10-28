"""Test the S2S model and Mimi codec with real voice samples downloaded from Hugging Face.

Usage:
    uv run --with moshi,huggingface_hub,datasets,soundfile,scipy python scripts/test_real_sample.py
"""
from __future__ import annotations
import argparse
import io
import math
import os
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import sys
import time
from typing import Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import scipy.signal
import soundfile as sf
import torch
from datasets import load_dataset

from bpm.config import ModelConfig, load_cfg
from bpm.mimi_adapter import MimiTokenizer
from bpm.model import S2SModel
from bpm.streaming import Streamer


def download_hf_sample(dataset_name: str = "hf-internal-testing/librispeech_asr_dummy",
                       split: str = "validation",
                       sample_idx: int = 0) -> dict:
    """Download audio sample from a Hugging Face dataset."""
    print(f"[*] Downloading dataset '{dataset_name}' (split='{split}')...")
    ds = load_dataset(dataset_name, "clean", split=split)
    ds = ds.cast_column("audio", ds.features["audio"].__class__(decode=False))
    item = ds[sample_idx]
    audio_bytes = item["audio"]["bytes"]
    raw_wav, orig_sr = sf.read(io.BytesIO(audio_bytes))
    if raw_wav.ndim > 1:
        raw_wav = raw_wav.mean(axis=1)  # convert to mono
    return {
        "raw_wav": raw_wav,
        "sample_rate": orig_sr,
        "text": item.get("text", "N/A"),
        "id": item.get("id", f"sample_{sample_idx}"),
    }


def resample_to_24k(wav: np.ndarray, orig_sr: int) -> tuple[np.ndarray, torch.Tensor]:
    """Resample to 24000 Hz and pad/truncate to integer number of 80ms (1920 sample) frames."""
    target_sr = 24000
    frame_len = 1920  # 80ms at 24kHz (12.5 Hz frame rate)
    target_len = int(round(len(wav) * target_sr / orig_sr))
    target_len = max(frame_len, (target_len // frame_len) * frame_len)
    resampled = scipy.signal.resample(wav, target_len).astype(np.float32)
    # Normalize peak to 0.95 to prevent clipping
    peak = np.abs(resampled).max()
    if peak > 0:
        resampled = resampled * (0.95 / peak)
    tensor = torch.tensor(resampled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return resampled, tensor


def compute_frame_energies(wav: np.ndarray, frame_len: int = 1920) -> list[float]:
    """Compute RMS energy for each 80ms frame."""
    energies = []
    num_frames = len(wav) // frame_len
    for i in range(num_frames):
        chunk = wav[i * frame_len : (i + 1) * frame_len]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        energies.append(rms)
    return energies


def run_streaming_test(model: S2SModel,
                       user_codes: torch.Tensor,
                       frame_energies: list[float],
                       energy_threshold: float = 0.05,
                       name: str = "Model") -> tuple[list[list[int]], dict]:
    """Feed frames through Streamer and collect emitted tokens and timing."""
    T = user_codes.shape[1]
    Q = model.cfg.Q
    init_own = [0] * Q

    # User active detection based on audio frame energy
    def is_user_active(u_frame: Sequence[int], frame_idx: int) -> bool:
        if frame_idx < len(frame_energies):
            return frame_energies[frame_idx] >= energy_threshold
        return u_frame[0] != 0

    st = Streamer(model, init_own, revision_tau=None)
    own_history = []
    timeline = []

    print(f"\n--- Streaming Timeline ({name}) ---")
    print(f"{'Frame':>5} | {'Time (s)':>8} | {'User CB0':>8} | {'User Spk':>8} | {'Own CB0':>8} | {'Own Spk':>7} | {'Plan':>4} | {'Wake':>5}")
    print("-" * 65)

    for t in range(T):
        u_t = user_codes[0, t].tolist()
        wakes_before = st.stats["wakes"]
        own_t, own_act = st.output_frame()
        woke = st.stats["wakes"] > wakes_before
        user_spk = is_user_active(u_t, t)

        own_history.append(own_t)
        plan_left = len(st.plan)
        time_s = t * 0.080  # 80ms per frame

        # Print first 20 frames, then every 5th frame, and final frames
        if t < 15 or t % 5 == 0 or t >= T - 3 or woke:
            wake_indicator = "WAKE" if woke else "-"
            print(f"{t:5d} | {time_s:7.2f}s | {u_t[0]:8d} | {'YES' if user_spk else 'no':>8} | {own_t[0]:8d} | {'YES' if own_act else 'no':>7} | {plan_left:4d} | {wake_indicator:>5}")

        # Update streamer state with observed frame
        st.prev_active = user_spk
        fr = st.cur
        st.pu.append(list(u_t))
        st.pa.append(fr["a"])
        st.ps.append(fr["s"])

    summary = st.summary()
    return own_history, summary


def main():
    ap = argparse.ArgumentParser(description="Test S2S model with real voice sample from Hugging Face")
    ap.add_argument("--dataset", default="hf-internal-testing/librispeech_asr_dummy", help="Hugging Face dataset name")
    ap.add_argument("--sample_idx", type=int, default=0, help="Sample index in dataset")
    ap.add_argument("--ckpt", default="runs/bpm/ckpt.pt", help="Path to trained checkpoint")
    ap.add_argument("--config", default="configs/ultra_small.yaml", help="Path to ultra_small model config")
    ap.add_argument("--out_dir", default="runs/real_test", help="Directory to save output wav files")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("  BPM-S2S: REAL AUDIO VOICE SAMPLE TEST VIA HUGGING FACE & KYUTAI MIMI")
    print("=" * 80)

    # 1. Download voice sample from Hugging Face
    sample = download_hf_sample(args.dataset, split="validation", sample_idx=args.sample_idx)
    raw_wav = sample["raw_wav"]
    orig_sr = sample["sample_rate"]
    text = sample["text"]
    orig_dur = len(raw_wav) / orig_sr

    print(f"\n[+] HF Sample Downloaded:")
    print(f"    Dataset:    {args.dataset}")
    print(f"    Sample ID:  {sample['id']}")
    print(f"    Transcript: \"{text}\"")
    print(f"    Audio:      {len(raw_wav)} samples @ {orig_sr} Hz ({orig_dur:.2f} seconds)")

    input_wav_path = os.path.join(args.out_dir, "01_input_sample.wav")
    sf.write(input_wav_path, raw_wav, orig_sr)
    print(f"    Saved:      {input_wav_path}")

    # 2. Resample to 24 kHz (Mimi native sample rate)
    resampled_wav, wav_tensor = resample_to_24k(raw_wav, orig_sr)
    num_frames = resampled_wav.shape[0] // 1920
    dur_24k = resampled_wav.shape[0] / 24000
    print(f"\n[+] Audio Preprocessing for Mimi Codec (24000 Hz, 12.5 Hz frames):")
    print(f"    Duration:   {dur_24k:.2f} seconds")
    print(f"    Samples:    {resampled_wav.shape[0]} samples")
    print(f"    Frames:     {num_frames} frames (80 ms each)")

    resampled_wav_path = os.path.join(args.out_dir, "02_input_resampled_24k.wav")
    sf.write(resampled_wav_path, resampled_wav, 24000)
    print(f"    Saved:      {resampled_wav_path}")

    frame_energies = compute_frame_energies(resampled_wav, 1920)
    avg_energy = float(np.mean(frame_energies))
    max_energy = float(np.max(frame_energies))
    active_thresh = 0.03
    active_frames = sum(1 for e in frame_energies if e >= active_thresh)
    print(f"    RMS Energy: mean={avg_energy:.4f}, max={max_energy:.4f}")
    print(f"    Active VAD: {active_frames}/{num_frames} frames active (threshold={active_thresh})")

    # 3. Initialize Mimi Tokenizer and encode real audio to discrete tokens
    print(f"\n[+] Loading Mimi Codec Tokenizer (kyutai/moshiko-pytorch-bf16)...")
    tok = MimiTokenizer(device="cpu", num_codebooks=8)
    user_codes_8 = tok.encode(wav_tensor)  # (1, T, 8)
    print(f"    Encoded shape: (B={user_codes_8.shape[0]}, T={user_codes_8.shape[1]}, Q={user_codes_8.shape[2]})")
    print(f"    Codebook 0 values: min={user_codes_8[0,:,0].min().item()}, max={user_codes_8[0,:,0].max().item()}")
    print(f"    First 10 cb0 tokens: {user_codes_8[0, :10, 0].tolist()}")

    # Mimi roundtrip test (decode original codes)
    print(f"    Testing Mimi neural autoencoder roundtrip decode...")
    mimi_rec = tok.decode(user_codes_8)
    rec_wav_np = mimi_rec[0, 0].cpu().numpy()
    rec_wav_path = os.path.join(args.out_dir, "03_mimi_reconstructed.wav")
    sf.write(rec_wav_path, rec_wav_np, 24000)
    print(f"    Saved:      {rec_wav_path} (shape: {rec_wav_np.shape})")

    # 4. Test Model A: Ultra-Small BPM Model (Native Mimi layout: Q=8, V=2048, K=4)
    print(f"\n" + "=" * 80)
    print("  TEST 1: NATIVE MIMI BPM MODEL (configs/ultra_small.yaml)")
    print("  Architecture: 8 codebooks, 2048 vocab, K=4 block parallel, d=384, 8 layers")
    print("=" * 80)

    mcfg_ultra, _ = load_cfg(args.config)
    model_ultra = S2SModel(mcfg_ultra)
    n_params_ultra = sum(p.numel() for p in model_ultra.parameters())
    print(f"    Parameters: {n_params_ultra:,} ({n_params_ultra/1e6:.1f}M)")

    own_codes_ultra, stats_ultra = run_streaming_test(
        model_ultra,
        user_codes_8,
        frame_energies,
        energy_threshold=active_thresh,
        name="Ultra-Small BPM (K=4, Q=8)"
    )

    print(f"\n[+] Ultra-Small BPM Streaming Summary:")
    print(f"    Total frames:         {stats_ultra['frames']}")
    print(f"    Backbone wakes:       {stats_ultra['wakes']}")
    print(f"    Calls per frame:      {stats_ultra['calls_per_frame']:.4f} (theoretical 1/K = {1/mcfg_ultra.K:.4f})")
    print(f"    Refiner passes/frame: {stats_ultra['refiner_passes']/stats_ultra['frames']:.2f}")
    print(f"    Wake latency (mean):  {stats_ultra.get('wake_ms_mean', 0.0):.2f} ms")
    print(f"    Wake latency (p50):   {stats_ultra.get('wake_ms_p50', 0.0):.2f} ms")
    print(f"    Wake latency (p95):   {stats_ultra.get('wake_ms_p95', 0.0):.2f} ms")

    # Decode ultra model output tokens to audio
    own_codes_ultra_tensor = torch.tensor([own_codes_ultra], dtype=torch.long)
    ultra_audio = tok.decode(own_codes_ultra_tensor)
    ultra_audio_np = ultra_audio[0, 0].cpu().numpy()
    ultra_wav_path = os.path.join(args.out_dir, "04_ultra_bpm_response.wav")
    sf.write(ultra_wav_path, ultra_audio_np, 24000)
    print(f"    Saved audio response: {ultra_wav_path} ({len(ultra_audio_np)/24000:.2f}s)")

    # 5. Test Model B: Trained Checkpoint (runs/bpm/ckpt.pt: Q=4, V=256, K=4)
    if os.path.exists(args.ckpt):
        print(f"\n" + "=" * 80)
        print("  TEST 2: TRAINED BPM CHECKPOINT (runs/bpm/ckpt.pt)")
        print("  Trained on: Echo-Reversed synthetic task (Q=4, V=256, K=4, d=128)")
        print("=" * 80)

        ck = torch.load(args.ckpt, map_location="cpu")
        mcfg_ck = ModelConfig(**ck["cfg"]["model"])
        model_ck = S2SModel(mcfg_ck)
        model_ck.load_state_dict(ck["model"])
        n_params_ck = sum(p.numel() for p in model_ck.parameters())
        print(f"    Parameters: {n_params_ck:,} ({n_params_ck/1e6:.2f}M)")

        # Map real Mimi tokens (Q=8, V=2048) to checkpoint dimensions (Q=4, V=256)
        user_codes_mapped = (user_codes_8[:, :, :4] % 256).clone()

        own_codes_ck, stats_ck = run_streaming_test(
            model_ck,
            user_codes_mapped,
            frame_energies,
            energy_threshold=active_thresh,
            name="Trained BPM Checkpoint (Q=4, V=256)"
        )

        print(f"\n[+] Trained Checkpoint Streaming Summary:")
        print(f"    Total frames:         {stats_ck['frames']}")
        print(f"    Backbone wakes:       {stats_ck['wakes']}")
        print(f"    Calls per frame:      {stats_ck['calls_per_frame']:.4f}")
        print(f"    Wake latency (mean):  {stats_ck.get('wake_ms_mean', 0.0):.2f} ms")
        print(f"    Wake latency (p50):   {stats_ck.get('wake_ms_p50', 0.0):.2f} ms")

        # To decode Q=4 codes with Mimi (which expects 8 codebooks), pad remaining 4 codebooks with zeros
        codes_for_decode = torch.zeros(1, len(own_codes_ck), 8, dtype=torch.long)
        codes_for_decode[0, :, :4] = torch.tensor(own_codes_ck, dtype=torch.long)
        ck_audio = tok.decode(codes_for_decode)
        ck_audio_np = ck_audio[0, 0].cpu().numpy()
        ck_wav_path = os.path.join(args.out_dir, "05_trained_bpm_response.wav")
        sf.write(ck_wav_path, ck_audio_np, 24000)
        print(f"    Saved audio response: {ck_wav_path} ({len(ck_audio_np)/24000:.2f}s)")

    # 6. Test Model C: Real-Voice Trained Checkpoint (runs/real_voice/ckpt.pt: Q=8, V=2048)
    real_ckpt_path = "runs/real_voice/ckpt.pt"
    if os.path.exists(real_ckpt_path):
        print(f"\n" + "=" * 80)
        print("  TEST 3: REAL-VOICE TRAINED CHECKPOINT (runs/real_voice/ckpt.pt)")
        print("  Trained on: Real LibriSpeech Mimi voice tokens (Q=8, V=2048, K=4, CPU)")
        print("=" * 80)

        ck_rv = torch.load(real_ckpt_path, map_location="cpu")
        mcfg_rv = ModelConfig(**ck_rv["cfg"]["model"])
        model_rv = S2SModel(mcfg_rv)
        model_rv.load_state_dict(ck_rv["model"])
        n_params_rv = sum(p.numel() for p in model_rv.parameters())
        print(f"    Parameters: {n_params_rv:,} ({n_params_rv/1e6:.2f}M)")

        own_codes_rv, stats_rv = run_streaming_test(
            model_rv,
            user_codes_8,
            frame_energies,
            energy_threshold=active_thresh,
            name="Real-Voice Trained Model (Q=8, V=2048)"
        )

        print(f"\n[+] Real-Voice Trained Model Streaming Summary:")
        print(f"    Total frames:         {stats_rv['frames']}")
        print(f"    Backbone wakes:       {stats_rv['wakes']}")
        print(f"    Calls per frame:      {stats_rv['calls_per_frame']:.4f}")
        print(f"    Wake latency (mean):  {stats_rv.get('wake_ms_mean', 0.0):.2f} ms")
        print(f"    Wake latency (p50):   {stats_rv.get('wake_ms_p50', 0.0):.2f} ms")

        own_codes_rv_tensor = torch.tensor([own_codes_rv], dtype=torch.long)
        rv_audio = tok.decode(own_codes_rv_tensor)
        rv_audio_np = rv_audio[0, 0].cpu().numpy()
        rv_wav_path = os.path.join(args.out_dir, "06_real_voice_trained_response.wav")
        sf.write(rv_wav_path, rv_audio_np, 24000)
        print(f"    Saved audio response: {rv_wav_path} ({len(rv_audio_np)/24000:.2f}s)")

    # 7. Test Model D: SmolLM2-135M Pretrained Backbone (runs/smollm2_voice/ckpt.pt)
    smol_ckpt_path = "runs/smollm2_voice/ckpt.pt"
    if os.path.exists(smol_ckpt_path):
        print(f"\n" + "=" * 80)
        print("  TEST 4: PRETRAINED SMOLLM2-135M BACKBONE (runs/smollm2_voice/ckpt.pt)")
        print("  Backbone: HuggingFaceTB/SmolLM2-135M (576-dim, 30 layers, 135M params)")
        print("=" * 80)

        ck_smol = torch.load(smol_ckpt_path, map_location="cpu")
        mcfg_smol = ModelConfig(**ck_smol["cfg"]["model"])
        model_smol = S2SModel(mcfg_smol)
        model_smol.load_state_dict(ck_smol["model"])
        n_params_smol = sum(p.numel() for p in model_smol.parameters())
        print(f"    Parameters: {n_params_smol:,} ({n_params_smol/1e6:.2f}M)")

        own_codes_smol, stats_smol = run_streaming_test(
            model_smol,
            user_codes_8,
            frame_energies,
            energy_threshold=active_thresh,
            name="SmolLM2-135M BPM Model (Q=8, V=2048)"
        )

        print(f"\n[+] SmolLM2-135M Streaming Summary:")
        print(f"    Total frames:         {stats_smol['frames']}")
        print(f"    Backbone wakes:       {stats_smol['wakes']}")
        print(f"    Calls per frame:      {stats_smol['calls_per_frame']:.4f}")
        print(f"    Wake latency (mean):  {stats_smol.get('wake_ms_mean', 0.0):.2f} ms")
        print(f"    Wake latency (p50):   {stats_smol.get('wake_ms_p50', 0.0):.2f} ms")

        own_codes_smol_tensor = torch.tensor([own_codes_smol], dtype=torch.long)
        smol_audio = tok.decode(own_codes_smol_tensor)
        smol_audio_np = smol_audio[0, 0].cpu().numpy()
        smol_wav_path = os.path.join(args.out_dir, "07_smollm2_trained_response.wav")
        sf.write(smol_wav_path, smol_audio_np, 24000)
        print(f"    Saved audio response: {smol_wav_path} ({len(smol_audio_np)/24000:.2f}s)")

    print(f"\n" + "=" * 80)
    print("  ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Generated artifacts in '{args.out_dir}':")
    for f in sorted(os.listdir(args.out_dir)):
        p = os.path.join(args.out_dir, f)
        print(f"  - {f:<30} ({os.path.getsize(p):,} bytes)")


if __name__ == "__main__":
    main()
