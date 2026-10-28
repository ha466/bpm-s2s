"""Block-Parallel Multi-rate (BPM) speech-token model and the matched AR baseline.

STATUS: working code operating on discrete codec tokens. Audio <-> token conversion is NOT done here
(see bpm/mimi_adapter.py, which is UNTESTED in the authoring environment).

Stream layout (12.5 Hz frames, Q codebooks): at backbone position p the input is
    x_p = E_user(u_p) + E_own(a_p) + E_act(s_p)
i.e. user frame p and the agent's own already-emitted frame p. The hidden state h_p then predicts the
agent's frames p+1 .. p+K ("anchor" p). K=1 + AR depth module == Moshi-style frame-by-frame baseline.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig
from .modules import make_backbone, BlockRefiner, DepthAR


class S2SModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        Q, V, d = cfg.Q, cfg.V, cfg.d
        self.is_bpm = cfg.kind == "bpm"
        self.K = cfg.K if self.is_bpm else 1
        self.emb_u = nn.Embedding(Q * V, d)
        self.emb_a = nn.Embedding(Q * V, d)
        self.emb_s = nn.Embedding(2, d)
        self.backbone = make_backbone(cfg)
        self.fut = nn.Linear(d, self.K * d)
        self.fut_norm = nn.RMSNorm(d)
        self.head_cb0 = nn.Linear(d, V, bias=False)
        self.head_s = nn.Linear(d, 2)
        self.head_u0 = nn.Linear(d, V, bias=False)           # aux: predict the user's next K cb0
        self.refiner = BlockRefiner(cfg) if self.is_bpm else DepthAR(cfg)
        self.register_buffer("off", torch.arange(Q) * V, persistent=False)

    # ---------------- encoding (training and streaming share this path) ----------------
    def embed(self, u, a, s):
        return self.emb_u(u + self.off).sum(2) + self.emb_a(a + self.off).sum(2) + self.emb_s(s)

    def encode(self, u, a, s, state=None):
        """u,a: (B,T,Q) long; s: (B,T) long -> h (B,T,d), new state."""
        return self.backbone(self.embed(u, a, s), state)

    def future(self, h):
        z = F.gelu(self.fut(h)).view(*h.shape[:-1], self.K, self.cfg.d)
        return self.fut_norm(z)

    # ---------------- training ----------------
    def loss(self, u, a, s):
        B, T, Q = u.shape
        K, V = self.K, self.cfg.V
        h, _ = self.encode(u, a, s)
        P = T - K
        hp = h[:, :P]
        idx = torch.arange(P, device=u.device)[:, None] + 1 + torch.arange(K, device=u.device)[None]
        z = self.future(hp)                                           # (B,P,K,d)
        t_cb0, t_s, t_u0 = a[:, idx, 0], s[:, idx], u[:, idx, 0]      # (B,P,K)
        ce = F.cross_entropy
        l_cb0 = ce(self.head_cb0(z).reshape(-1, V), t_cb0.reshape(-1))
        l_s = ce(self.head_s(z).reshape(-1, 2), t_s.reshape(-1))
        l_u = ce(self.head_u0(z).reshape(-1, V), t_u0.reshape(-1))
        if self.is_bpm:
            r0 = int(torch.randint(0, K, (1,)))
            anc = torch.arange(r0, P, K, device=u.device)
            hh = hp[:, anc].reshape(-1, hp.shape[-1])
            cb0 = t_cb0[:, anc].reshape(-1, K)
            ac_t = a[:, idx[anc], 1:].reshape(-1, K, Q - 1)
            N = ac_t.shape[0]
            ratio = torch.cos(math.pi / 2 * torch.rand(N, device=u.device))
            m = torch.rand(N, K, Q - 1, device=u.device) < ratio[:, None, None]
            m[:, 0, 0] |= m.flatten(1).sum(1) == 0                    # at least one masked token
            logits = self.refiner(hh, cb0, torch.where(m, torch.full_like(ac_t, V), ac_t))
            l_ac = ce(logits[m], ac_t[m])
        else:
            hh = hp.reshape(-1, hp.shape[-1])
            codes = a[:, 1:P + 1].reshape(-1, Q)
            logits = self.refiner(hh, codes)
            l_ac = ce(logits.reshape(-1, V), codes[:, 1:].reshape(-1))
        total = l_cb0 + l_s + self.cfg.user_pred_weight * l_u + l_ac
        return total, {"cb0": l_cb0.item(), "act": l_s.item(), "user0": l_u.item(), "acoustic": l_ac.item()}

    # ---------------- inference ----------------
    @torch.no_grad()
    def plan(self, h_last):
        """h_last: (1,d) hidden state at the anchor -> committed draft of the next K own frames."""
        z = self.future(h_last)                                       # (1,K,d)
        cb0 = self.head_cb0(z).argmax(-1)                             # (1,K)
        s = self.head_s(z).argmax(-1)
        u_logp = self.head_u0(z).log_softmax(-1)                      # (1,K,V)
        if self.is_bpm:
            ac, passes = self.refiner.decode(h_last, cb0, self.cfg.refine_steps)
        else:
            ac, passes = self.refiner.decode(h_last, cb0[:, 0])
            ac = ac[:, None]
        return {"cb0": cb0[0], "s": s[0], "ac": ac[0], "u_logp": u_logp[0], "passes": passes}


def count_params(model: nn.Module) -> dict:
    out = {}
    for name, child in model.named_children():
        out[name] = sum(p.numel() for p in child.parameters())
    for name, p in model.named_parameters(recurse=False):
        out[name] = p.numel()
    out["total"] = sum(p.numel() for p in model.parameters())
    return out
