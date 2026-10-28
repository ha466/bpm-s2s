"""Streaming inference loop. STATUS: working code (tested on token streams).

Per 80 ms frame t the caller does:
    own_codes, own_active = streamer.output_frame()   # frame t, planned from context <= t-1
    streamer.observe(user_codes_t)                    # user frame t arrives
A "wake" (one backbone call over all not-yet-ingested frames + plan of K future frames) happens only
when the committed plan is exhausted, when the user's activity flips (turn start/end), or when the
user's observed cb0 is surprising under the stored prediction (revision, cf. DiffuPlex).
Frames already emitted are never modified.
"""
from __future__ import annotations
import time
from typing import Callable, Sequence
import torch


class Streamer:
    def __init__(self, model, init_own: Sequence[int], revision_tau: float | None = 4.0,
                 wake_on_activity: bool = True,
                 user_active: Callable[[Sequence[int]], bool] = lambda u: u[0] != 0):
        self.m = model.eval()
        self.dev = next(model.parameters()).device
        self.revision_tau, self.wake_on_activity, self.user_active = revision_tau, wake_on_activity, user_active
        self.state = None
        self.pu, self.pa, self.ps = [], [], []
        self.plan = [{"a": list(init_own), "s": 0, "ulogp": None}]       # frame 0: silence by convention
        self.cur = None
        self.prev_active = False
        self.stats = {"wakes": 0, "backbone_calls": 0, "refiner_passes": 0, "revisions": 0,
                      "activity_replans": 0, "cancels": 0, "frames": 0, "wake_ms": []}

    @torch.no_grad()
    def _wake(self):
        t0 = time.perf_counter()
        u = torch.tensor([self.pu], device=self.dev)
        a = torch.tensor([self.pa], device=self.dev)
        s = torch.tensor([self.ps], device=self.dev)
        h, self.state = self.m.encode(u, a, s, self.state)               # ONE backbone call for the chunk
        self.pu, self.pa, self.ps = [], [], []
        out = self.m.plan(h[:, -1])
        K = out["cb0"].shape[0]
        self.plan = []
        for k in range(K):
            codes = [int(out["cb0"][k])] + [int(x) for x in out["ac"][k]]
            self.plan.append({"a": codes, "s": int(out["s"][k]), "ulogp": out["u_logp"][k]})
        st = self.stats
        st["wakes"] += 1
        st["backbone_calls"] += 1
        st["refiner_passes"] += out["passes"]
        st["wake_ms"].append((time.perf_counter() - t0) * 1e3)

    def output_frame(self):
        if not self.plan:
            self._wake()
        self.cur = self.plan.pop(0)
        self.stats["frames"] += 1
        return self.cur["a"], self.cur["s"]

    def observe(self, u: Sequence[int]):
        fr = self.cur
        if self.plan:                                                    # something unplayed to protect
            if self.revision_tau is not None and fr["ulogp"] is not None:
                if float(-fr["ulogp"][u[0]]) >= self.revision_tau:
                    self.plan.clear(); self.stats["revisions"] += 1
            act = self.user_active(u)
            if self.wake_on_activity and act != self.prev_active and self.plan:
                self.plan.clear(); self.stats["activity_replans"] += 1
        self.prev_active = self.user_active(u)
        self.pu.append(list(u)); self.pa.append(fr["a"]); self.ps.append(fr["s"])

    def cancel(self):
        """External barge-in (e.g. VAD): drop all unplayed frames. Next output_frame() re-plans."""
        if self.plan:
            self.plan.clear(); self.stats["cancels"] += 1

    def summary(self):
        st = dict(self.stats)
        w = st.pop("wake_ms")
        st["calls_per_frame"] = st["backbone_calls"] / max(st["frames"], 1)
        if w:
            ws = sorted(w)
            st["wake_ms_mean"], st["wake_ms_p50"] = sum(w) / len(w), ws[len(ws) // 2]
            st["wake_ms_p95"] = ws[min(len(ws) - 1, int(0.95 * len(ws)))]
        return st
