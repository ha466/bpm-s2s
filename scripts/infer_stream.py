"""Frame-by-frame streaming demo on the synthetic task, including a barge-in cancel.

    python scripts/infer_stream.py --ckpt runs/bpm_k4/ckpt.pt --cancel_at 18
STATUS: working code (token level). Real audio would wrap bpm/mimi_adapter.py (UNTESTED).
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
from bpm.config import ModelConfig, TrainConfig
from bpm.model import S2SModel
from bpm.streaming import Streamer
from train import make_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cancel_at", type=int, default=-1, help="frame index at which to force a barge-in cancel")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    mcfg, tcfg = ModelConfig(**ck["cfg"]["model"]), TrainConfig(**ck["cfg"]["train"])
    model = S2SModel(mcfg); model.load_state_dict(ck["model"])
    task = make_task(mcfg, tcfg)
    u, a_true, s_true, meta = task.sample(np.random.RandomState(a.seed))
    st = Streamer(model, task.sil(meta["spk"]))
    print("user words:", meta["words"], "-> expected reply:", meta["words"][::-1], "| reply starts at frame", meta["r0"])
    for t in range(tcfg.T):
        if t == a.cancel_at:
            st.cancel(); print(f"  [t={t}] CANCEL (unplayed plan dropped)")
        codes, act = st.output_frame()
        wakes_before = st.stats["wakes"]
        print(f"t={t:2d} user_cb0={int(u[t,0]):3d} own_cb0={codes[0]:3d} speaking={act} plan_left={len(st.plan)}")
        st.observe(list(map(int, u[t])))
    print(st.summary())


if __name__ == "__main__":
    main()
