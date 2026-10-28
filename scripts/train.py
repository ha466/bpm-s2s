"""Train a model on the synthetic task, then score it closed-loop through the Streamer.

    python scripts/train.py --config configs/smoke_bpm.yaml --out runs/bpm_k4

STATUS: working code. Trains on synthetic tokens only. A real-audio training loop would replace
SynthTask with a dataset of Mimi-token dialogues (two channels per conversation) -- not included.
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from bpm.config import load_cfg, dump_cfg
from bpm.model import S2SModel, count_params
from bpm.streaming import Streamer
from bpm.data import SynthTask, RealVoiceTask, Spec, run_closed_loop


def make_task(mcfg, tcfg, data_mode="synth", dataset_name="hf-internal-testing/librispeech_asr_dummy",
              split="validation", max_samples=None, max_hours=4.0):
    spec = Spec(Q=mcfg.Q, V=mcfg.V, T=tcfg.T, Gmax=tcfg.Gmax)
    if data_mode == "real":
        return RealVoiceTask(spec, dataset_name=dataset_name, split=split, max_samples=max_samples, max_hours=max_hours)
    return SynthTask(spec)


def closed_loop_eval(model, task, n, revision_tau=4.0, wake_on_activity=True, seed=999):
    f = lambda spk: Streamer(model, task.sil(spk), revision_tau=revision_tau, wake_on_activity=wake_on_activity)
    return run_closed_loop(task, f, n, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--data", choices=["synth", "real"], default="synth", help="Data source: 'synth' or 'real'")
    ap.add_argument("--dataset", default="dev.clean", help="Dataset name or alias (e.g. 'dev.clean', 'openslr/librispeech_asr')")
    ap.add_argument("--split", default="validation", help="Dataset split (default: 'validation' for dev.clean)")
    ap.add_argument("--max_samples", type=int, default=None, help="Max audio samples to load")
    ap.add_argument("--max_hours", type=float, default=5.0, help="Maximum audio duration in hours (default: 5.0h)")
    args = ap.parse_args()
    mcfg, tcfg = load_cfg(args.config)
    if args.out: tcfg.out = args.out
    if args.steps: tcfg.steps = args.steps
    if args.seed is not None: tcfg.seed = args.seed
    os.makedirs(tcfg.out, exist_ok=True)
    torch.manual_seed(tcfg.seed); np.random.seed(tcfg.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    task = make_task(mcfg, tcfg, data_mode=args.data, dataset_name=args.dataset, split=args.split,
                     max_samples=args.max_samples, max_hours=args.max_hours)
    model = S2SModel(mcfg).to(device)
    pc = count_params(model)
    print(f"[{mcfg.kind} K={model.K} backbone={mcfg.backbone} data={args.data}] params: {pc}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.wd, betas=(0.9, 0.98))
    sched = lambda i: min(1.0, (i + 1) / tcfg.warmup) * 0.5 * (1 + math.cos(math.pi * i / tcfg.steps))
    t0, hist, start = time.time(), [], 0
    part = os.path.join(tcfg.out, "partial.pt")
    if os.path.exists(part):                                   # resume after an interruption
        ck = torch.load(part, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start, hist = ck["step"], ck["hist"]
        print(f"resumed from step {start}", flush=True)
    rng = np.random.RandomState(tcfg.seed * 100003 + start)   # data stream is re-seeded per resume point
    for step in range(start, tcfg.steps):
        for g in opt.param_groups:
            g["lr"] = tcfg.lr * sched(step)
        u, a, s, _ = task.batch(tcfg.batch, rng)
        u, a, s = u.to(device), a.to(device), s.to(device)
        loss, parts = model.loss(u, a, s)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % tcfg.log_every == 0 or step == tcfg.steps - 1:
            val_loss = task.eval_loss(model) if hasattr(task, "eval_loss") else None
            record = {"step": step, "loss": loss.item(), **parts}
            if val_loss is not None:
                record["val_loss"] = val_loss
            hist.append(record)
            if step > start:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "hist": hist}, part)
            val_str = f" [val_loss {val_loss:.4f}]" if val_loss is not None else ""
            print(f"step {step:5d} loss {loss.item():.4f}{val_str} {parts} ({time.time()-t0:.0f}s)", flush=True)
    train_s = time.time() - t0 + (hist and 0)
    torch.save({"model": model.state_dict(), "cfg": dump_cfg(mcfg, tcfg)}, os.path.join(tcfg.out, "ckpt.pt"))
    if hasattr(task, "eval_loss"):
        final_val = task.eval_loss(model, n_batches=10)
        metrics = {"val_loss": final_val}
        print(f"FINAL REAL-VOICE EVAL: val_loss={final_val:.4f}", flush=True)
    else:
        metrics = closed_loop_eval(model, task, tcfg.eval_n)
        print("CLOSED-LOOP:", json.dumps(metrics, indent=2), flush=True)
    res = {"config": dump_cfg(mcfg, tcfg), "params": pc, "train_seconds": train_s, "history": hist,
           "metrics": metrics}
    json.dump(res, open(os.path.join(tcfg.out, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
