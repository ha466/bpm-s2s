"""Synthetic token-level 'spoken dialogue' used ONLY to validate mechanics (not speech, not evidence
about real conversational ability).

Task ("echo-reversed"): the user says n words (2 frames each), pauses D frames, and the agent must
say the same words in reverse order, in the user's 'voice'.
  * cb0 carries the word identity and the within-word frame index (semantic / content stream).
  * cb1..cb(Q-1) are a deterministic function of (cb0, speaker) plus noise (acoustic stream), so the
    acoustic decoder must read the speaker from context (speaker consistency) and cb0 of the same
    frame (the dependency DiffuPlex identifies as the source of NAR quality loss).
  * response length is implied by n -> tests length/stop handling; D tests response timing.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
import numpy as np
import torch


@dataclass
class Spec:
    Q: int = 4
    V: int = 256
    Wn: int = 16          # word vocabulary
    nmin: int = 2
    nmax: int = 4
    G: int = 2            # leading silence frames (minimum)
    Gmax: int = 2         # if > G, leading silence is drawn uniformly from [G, Gmax] (onset-phase jitter)
    D: int = 3            # pause between user's last frame and the agent's first frame
    S: int = 8            # speakers
    T: int = 32
    noise: float = 0.1    # prob. an acoustic token is replaced by a random one


class SynthTask:
    def __init__(self, spec: Spec, table_seed: int = 1234):
        self.sp = spec
        self.A = np.random.RandomState(table_seed).randint(0, spec.V, size=(spec.Q, spec.V))

    def codes(self, cb0: int, spk: int) -> np.ndarray:
        q = np.arange(self.sp.Q)
        c = (self.A[q, cb0] + spk * (q + 1)) % self.sp.V
        c[0] = cb0
        return c

    def sil(self, spk: int):
        return list(map(int, self.codes(0, spk)))

    def sample(self, rng: np.random.RandomState):
        sp = self.sp
        n = rng.randint(sp.nmin, sp.nmax + 1)
        words = rng.randint(0, sp.Wn, size=n)
        spk = rng.randint(0, sp.S)
        g = rng.randint(sp.G, sp.Gmax + 1) if sp.Gmax > sp.G else sp.G
        u_cb0 = np.zeros(sp.T, dtype=np.int64)
        a_cb0 = np.zeros(sp.T, dtype=np.int64)
        s = np.zeros(sp.T, dtype=np.int64)
        for i, w in enumerate(words):
            f = g + 2 * i
            u_cb0[f], u_cb0[f + 1] = 1 + w, 1 + sp.Wn + w
        r0 = g + 2 * n + sp.D
        for j, w in enumerate(words[::-1]):
            f = r0 + 2 * j
            a_cb0[f], a_cb0[f + 1] = 1 + w, 1 + sp.Wn + w
            s[f] = s[f + 1] = 1
        u = np.stack([self.codes(int(c), spk) for c in u_cb0])
        a = np.stack([self.codes(int(c), spk) for c in a_cb0])
        for arr in (u, a):                                     # acoustic noise
            m = rng.rand(*arr[:, 1:].shape) < sp.noise
            arr[:, 1:] = np.where(m, rng.randint(0, sp.V, size=m.shape), arr[:, 1:])
        meta = {"n": int(n), "words": words.tolist(), "spk": int(spk), "r0": int(r0),
                "user_end": int(g + 2 * n)}
        return u, a, s, meta

    def batch(self, B: int, rng: np.random.RandomState, device="cpu"):
        xs = [self.sample(rng) for _ in range(B)]
        u = torch.tensor(np.stack([x[0] for x in xs]), device=device)
        a = torch.tensor(np.stack([x[1] for x in xs]), device=device)
        s = torch.tensor(np.stack([x[2] for x in xs]), device=device)
        return u, a, s, [x[3] for x in xs]


def run_closed_loop(task: SynthTask, streamer_factory, n_examples: int, seed: int = 999):
    """Feed user streams to a Streamer and score the agent's *generated* stream (no teacher forcing)."""
    sp = task.sp
    rng = np.random.RandomState(seed)
    ok = talk_over = 0
    onset_delays, ac_hit, ac_tot = [], 0, 0
    agg = {"wakes": 0, "backbone_calls": 0, "refiner_passes": 0, "frames": 0, "revisions": 0,
           "activity_replans": 0}
    wake_ms = []
    for _ in range(n_examples):
        u, a_true, s_true, meta = task.sample(rng)
        st = streamer_factory(meta["spk"])
        out_codes, out_s = [], []
        for t in range(sp.T):
            codes, act = st.output_frame()
            out_codes.append(codes); out_s.append(act)
            st.observe(list(map(int, u[t])))
        oc = np.array(out_codes); osv = np.array(out_s)
        first = [int(c) - 1 for c in oc[:, 0] if 1 <= c <= sp.Wn]
        ok += int(first == meta["words"][::-1])
        speak = np.where(osv == 1)[0]
        if len(speak):
            onset_delays.append(int(speak[0]) - meta["r0"])
        talk_over += int(osv[: meta["user_end"]].any())
        for t in range(sp.T):
            if osv[t] == 1:
                ref = task.codes(int(oc[t, 0]), meta["spk"])
                ac_hit += int((oc[t, 1:] == ref[1:]).sum()); ac_tot += sp.Q - 1
        s_ = st.summary()
        for k in agg:
            agg[k] += s_[k]
        wake_ms.append(s_.get("wake_ms_mean", 0.0))
    return {
        "exact_response_acc": ok / n_examples,
        "acoustic_mode_acc": ac_hit / max(ac_tot, 1),
        "talk_over_rate": talk_over / n_examples,
        "onset_delay_frames_mean": float(np.mean(onset_delays)) if onset_delays else None,
        "onset_delay_frames_max": int(np.max(onset_delays)) if onset_delays else None,
        "calls_per_frame": agg["backbone_calls"] / agg["frames"],
        "refiner_passes_per_frame": agg["refiner_passes"] / agg["frames"],
        "revisions_per_example": agg["revisions"] / n_examples,
        "activity_replans_per_example": agg["activity_replans"] / n_examples,
        "wake_ms_mean": float(np.mean(wake_ms)),
    }


class RealVoiceTask:
    """Real voice dataset loader converting audio utterances to discrete Mimi tokens.
    Supports streaming mode so only the requested hours (under 5 hours) are downloaded.
    """
    def __init__(self, spec: Spec, dataset_name: str = "hf-internal-testing/librispeech_asr_dummy",
                 split: str = "validation", max_samples: int | None = None,
                 max_hours: float = 4.0,
                 cache_dir: str = "runs", split_ratio: float = 0.85):
        self.sp = spec
        self.Q = spec.Q
        self.V = spec.V
        self.T = spec.T
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples
        self.max_hours = max_hours
        safe_name = dataset_name.replace("/", "_")
        tag = f"_{max_hours:.1f}h"
        self.cache_path = os.path.join(cache_dir, f"{safe_name}_{split}{tag}_q{self.Q}.pt")
        # Legacy fallback for backward compatibility
        legacy_path = os.path.join(cache_dir, f"librispeech_mimi_q{self.Q}.pt")
        if not os.path.exists(self.cache_path) and os.path.exists(legacy_path) and "librispeech" in dataset_name:
            self.cache_path = legacy_path
        self.tokens = self._load_or_encode_tokens()
        n_train = max(1, int(len(self.tokens) * split_ratio))
        self.train_tokens = self.tokens[:n_train]
        self.val_tokens = self.tokens[n_train:]
        self.sil_frame = self._compute_silence()

    def sil(self, spk: int = 0) -> list[int]:
        return list(self.sil_frame)

    def _compute_silence(self) -> list[int]:
        from .mimi_adapter import MimiTokenizer
        tok = MimiTokenizer(device="cpu", num_codebooks=self.Q)
        sil_wav = torch.zeros(1, 1, 24000)
        c = tok.encode(sil_wav)[0, 0].tolist()
        return [int(x) % self.V for x in c]

    def _load_or_encode_tokens(self) -> list[torch.Tensor]:
        if os.path.exists(self.cache_path):
            cached = torch.load(self.cache_path, map_location="cpu")
            return [t % self.V for t in cached]

        import io, soundfile as sf, scipy.signal
        from datasets import load_dataset
        from .mimi_adapter import MimiTokenizer

        max_sec = self.max_hours * 3600.0
        total_sec = 0.0
        offline = os.environ.get("HF_DATASETS_OFFLINE") == "1"

        print(f"[*] Loading dataset '{self.dataset_name}' (target: under {self.max_hours:.1f} hours)...")
        # Try streaming mode first to avoid downloading full 60GB dataset upfront
        ds = None
        if not offline:
            for s_name in [None, "clean"]:
                try:
                    kwargs = {"split": self.split, "streaming": True}
                    if s_name:
                        kwargs["name"] = s_name
                    ds = load_dataset(self.dataset_name, **kwargs)
                    print(f"  Streaming mode enabled (downloading sample-by-sample on the fly).")
                    break
                except Exception:
                    ds = None

        if ds is None:
            # Fall back to standard cached load
            try:
                ds = load_dataset(self.dataset_name, "clean", split=self.split)
            except Exception:
                ds = load_dataset(self.dataset_name, split=self.split)
            ds = ds.cast_column("audio", ds.features["audio"].__class__(decode=False))

        tok = MimiTokenizer(device="cpu", num_codebooks=self.Q)
        encoded_list = []

        for i, item in enumerate(ds):
            if total_sec >= max_sec:
                print(f"  Reached target duration limit: {total_sec/3600:.2f} hours. Stopping download.")
                break
            if self.max_samples is not None and i >= self.max_samples:
                break

            audio_data = item.get("audio", {})
            if isinstance(audio_data, dict) and "array" in audio_data:
                raw_wav = np.array(audio_data["array"], dtype=np.float32)
                sr = int(audio_data.get("sampling_rate", 24000))
            elif isinstance(audio_data, dict) and audio_data.get("bytes"):
                raw_wav, sr = sf.read(io.BytesIO(audio_data["bytes"]))
            elif hasattr(audio_data, "__getitem__") and "array" in audio_data:
                raw_wav = np.array(audio_data["array"], dtype=np.float32)
                sr = int(audio_data.get("sampling_rate", 24000))
            else:
                continue

            if raw_wav.ndim > 1:
                raw_wav = raw_wav.mean(axis=1)

            dur = len(raw_wav) / sr
            total_sec += dur

            target_len = (int(len(raw_wav) * 24000 / sr) // 1920) * 1920
            if target_len < 1920:
                continue
            w = scipy.signal.resample(raw_wav, target_len).astype("float32")
            peak = abs(w).max()
            if peak > 0:
                w = w * (0.95 / peak)
            wav_t = torch.tensor(w).unsqueeze(0).unsqueeze(0)
            c = tok.encode(wav_t)[0]
            encoded_list.append(c)

            if len(encoded_list) % 100 == 0 or len(encoded_list) == 1:
                print(f"  Processed {len(encoded_list)} utterances ({total_sec/3600:.2f} / {self.max_hours:.1f} hours)...")

        print(f"[+] Tokenized {len(encoded_list)} utterances ({total_sec/3600:.2f} hours total). Caching to {self.cache_path}...")
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        torch.save(encoded_list, self.cache_path)
        return [t % self.V for t in encoded_list]

    def sample(self, rng: np.random.RandomState, split: str = "train"):
        pool = self.train_tokens if split == "train" else self.val_tokens
        T = self.T
        u = np.tile(self.sil_frame, (T, 1))
        a = np.tile(self.sil_frame, (T, 1))
        s = np.zeros(T, dtype=np.int64)

        idx_u = rng.randint(0, len(pool))
        idx_a = rng.randint(0, len(pool))
        toks_u = pool[idx_u].numpy()
        toks_a = pool[idx_a].numpy()

        g = rng.randint(1, 4)
        n_u = rng.randint(8, min(len(toks_u), T // 2))
        u_start_offset = rng.randint(0, max(1, len(toks_u) - n_u))
        u[g : g + n_u] = toks_u[u_start_offset : u_start_offset + n_u]

        d_pause = rng.randint(2, 5)
        r0 = g + n_u + d_pause
        n_a = min(T - r0, min(len(toks_a), rng.randint(8, max(9, T - r0))))
        if n_a > 0 and r0 < T:
            a_start_offset = rng.randint(0, max(1, len(toks_a) - n_a))
            a[r0 : r0 + n_a] = toks_a[a_start_offset : a_start_offset + n_a]
            s[r0 : r0 + n_a] = 1

        meta = {"user_idx": idx_u, "agent_idx": idx_a, "r0": r0, "user_end": g + n_u, "spk": 0}
        return u, a, s, meta

    def batch(self, B: int, rng: np.random.RandomState, split: str = "train"):
        us, as_, ss, metas = [], [], [], []
        for _ in range(B):
            u, a, s, meta = self.sample(rng, split=split)
            us.append(u); as_.append(a); ss.append(s); metas.append(meta)
        return (torch.tensor(np.stack(us), dtype=torch.long),
                torch.tensor(np.stack(as_), dtype=torch.long),
                torch.tensor(np.stack(ss), dtype=torch.long),
                metas)

    def eval_loss(self, model: torch.nn.Module, n_batches: int = 5, batch_size: int = 8, seed: int = 999):
        rng = np.random.RandomState(seed)
        was_training = model.training
        model.eval()
        dev = next(model.parameters()).device
        losses = []
        with torch.no_grad():
            for _ in range(n_batches):
                u, a, s, _ = self.batch(batch_size, rng, split="val")
                u, a, s = u.to(dev), a.to(dev), s.to(dev)
                loss, _ = model.loss(u, a, s)
                losses.append(loss.item())
        if was_training:
            model.train()
        return float(np.mean(losses))
