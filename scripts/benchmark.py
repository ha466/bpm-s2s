"""Wake-latency benchmark at REAL model sizes with random weights (speed does not depend on weights).

    python scripts/benchmark.py --config configs/ultra_small.yaml --frames 200

Measures only the token model (backbone + heads + refiner/depth) on THIS machine, single process,
fp32, no codec, no audio I/O. Reports wall-clock per wake (p50/p95/max) against the 80 ms frame budget.
STATUS: working code; numbers are hardware-specific and are not a latency claim for any other machine.
"""
import argparse, json, os, platform, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
from bpm.config import load_cfg
from bpm.model import S2SModel
from bpm.streaming import Streamer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--kind", default=None, help="override: bpm|ar")
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="refine steps")
    ap.add_argument("--warmup", type=int, default=20)
    a = ap.parse_args()
    m, _ = load_cfg(a.config)
    if a.kind: m.kind = a.kind
    if a.K: m.K = a.K
    if a.steps: m.refine_steps = a.steps
    torch.manual_seed(0)
    model = S2SModel(m).eval()
    st = Streamer(model, [0] * m.Q, revision_tau=None, wake_on_activity=False)
    rng = np.random.RandomState(0)
    for t in range(a.frames + a.warmup):
        if t == a.warmup:
            st.stats["wake_ms"].clear(); st.stats.update(wakes=0, backbone_calls=0, refiner_passes=0, frames=0)
        st.output_frame()
        st.observe(rng.randint(0, m.V, size=m.Q).tolist())
    s = st.summary()
    frame_ms = 1000 / m.frame_hz
    wake_total_ms = s["wake_ms_mean"] * s["wakes"]
    out = {"host": platform.processor() or platform.machine(), "threads": torch.get_num_threads(),
           "torch": torch.__version__, "kind": m.kind, "K": model.K, "backbone": m.backbone,
           "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
           "refine_steps": m.refine_steps if model.is_bpm else None,
           "wakes": s["wakes"], "calls_per_frame": round(s["calls_per_frame"], 3),
           "wake_ms_p50": round(s["wake_ms_p50"], 1), "wake_ms_p95": round(s["wake_ms_p95"], 1),
           "wake_ms_max": round(max(st.stats["wake_ms"]), 1), "frame_budget_ms": frame_ms,
           "model_real_time_factor": round(wake_total_ms / (s["frames"] * frame_ms), 3),
           "wake_over_budget_frac_vs_K_frames": round(float(np.mean(np.array(st.stats["wake_ms"]) > model.K * frame_ms)), 3)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
