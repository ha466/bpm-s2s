"""Closed-loop inference-policy ablations on a trained synthetic-task checkpoint.

    python scripts/evaluate.py --ckpt runs/bpm_k4/ckpt.pt --n 300
Varies refine steps S, revision, and the user-activity wake. STATUS: working code.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from bpm.config import ModelConfig, TrainConfig
from bpm.model import S2SModel
from train import make_task, closed_loop_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    mcfg, tcfg = ModelConfig(**ck["cfg"]["model"]), TrainConfig(**ck["cfg"]["train"])
    model = S2SModel(mcfg); model.load_state_dict(ck["model"]); model.eval()
    task = make_task(mcfg, tcfg)
    rows = {}
    variants = [("default", {}, mcfg.refine_steps)]
    if model.is_bpm:
        variants += [("no_activity_wake", {"wake_on_activity": False}, mcfg.refine_steps),
                     ("no_revision_no_activity_wake", {"wake_on_activity": False, "revision_tau": None}, mcfg.refine_steps)]
        variants += [(f"refine_steps={s}", {}, s) for s in (1, 2, 4, 8)]
    for name, kw, S in variants:
        model.cfg.refine_steps = S
        rows[name] = closed_loop_eval(model, task, a.n, **kw)
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[name].items()}))
    out = a.out or os.path.join(os.path.dirname(a.ckpt), "ablation.json")
    json.dump(rows, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
