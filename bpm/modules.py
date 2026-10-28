"""Building blocks. STATUS: working code, unit-tested for chunked-streaming equivalence."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rope(x: torch.Tensor, pos: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """x: (B,H,T,dh); pos: (T,) absolute positions."""
    dh = x.shape[-1]
    half = dh // 2
    freqs = theta ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = pos.to(torch.float32)[:, None] * freqs[None]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalAttention(nn.Module):
    """Sliding-window causal attention with an explicit KV cache (chunked streaming)."""

    def __init__(self, d: int, n_heads: int, window: int):
        super().__init__()
        assert d % n_heads == 0 and (d // n_heads) % 2 == 0
        self.h, self.dh, self.window = n_heads, d // n_heads, window
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, cache=None, pos0: int = 0):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        pos = torch.arange(pos0, pos0 + T, device=x.device)
        q, k = rope(q, pos), rope(k, pos)
        if cache is not None:
            kpos0 = pos0 - cache[0].shape[2]
            k_all, v_all = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
        else:
            kpos0, k_all, v_all = pos0, k, v
        kpos = torch.arange(kpos0, kpos0 + k_all.shape[2], device=x.device)
        diff = pos[:, None] - kpos[None, :]
        mask = (diff >= 0) & (diff < self.window)
        y = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask)
        y = y.transpose(1, 2).reshape(B, T, D)
        keep = max(self.window - 1, 1)
        new_cache = (k_all[:, :, -keep:].detach(), v_all[:, :, -keep:].detach())
        return self.out(y), new_cache


class TBlock(nn.Module):
    def __init__(self, d, n_heads, window, ff_mult):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.attn = CausalAttention(d, n_heads, window)
        self.mlp = nn.Sequential(nn.Linear(d, ff_mult * d, bias=False), nn.GELU(),
                                 nn.Linear(ff_mult * d, d, bias=False))

    def forward(self, x, cache=None, pos0=0):
        a, c = self.attn(self.n1(x), cache, pos0)
        x = x + a
        return x + self.mlp(self.n2(x)), c


class TransformerBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TBlock(cfg.d, cfg.n_heads, cfg.window, cfg.ff_mult) for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d)

    def forward(self, x, state=None):
        pos0 = 0 if state is None else state["pos"]
        caches = [None] * len(self.blocks) if state is None else state["caches"]
        new = []
        T = x.shape[1]
        for blk, c in zip(self.blocks, caches):
            x, c2 = blk(x, c, pos0)
            new.append(c2)
        return self.norm(x), {"pos": pos0 + T, "caches": new}


class GRUBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gru = nn.GRU(cfg.d, cfg.d, num_layers=cfg.n_layers, batch_first=True)
        self.norm = nn.RMSNorm(cfg.d)

    def forward(self, x, state=None):
        y, h = self.gru(x, None if state is None else state["h"])
        return self.norm(y), {"h": h.detach()}


class SmolLM2Backbone(nn.Module):
    """Pretrained SmolLM2-135M causal transformer backbone with streaming KV caching."""

    def __init__(self, cfg):
        super().__init__()
        import os
        from transformers import AutoModel
        repo = getattr(cfg, "pretrained_name", "HuggingFaceTB/SmolLM2-135M")
        # Prefer local cache if already downloaded
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M")
        local_only = os.path.exists(cache_dir)
        self.llama = AutoModel.from_pretrained(repo, local_files_only=local_only, dtype=torch.float32)
        if getattr(cfg, "freeze_backbone", False):
            for p in self.llama.parameters():
                p.requires_grad = False
        hidden_dim = self.llama.config.hidden_size  # 576
        self.proj_in = nn.Linear(cfg.d, hidden_dim, bias=False) if cfg.d != hidden_dim else nn.Identity()
        self.proj_out = nn.Linear(hidden_dim, cfg.d, bias=False) if cfg.d != hidden_dim else nn.Identity()

    def forward(self, x, state=None):
        """x: (B, T, d) -> h: (B, T, d), next_state."""
        x_proj = self.proj_in(x)
        out = self.llama(inputs_embeds=x_proj, past_key_values=state, use_cache=True)
        h = self.proj_out(out.last_hidden_state)
        return h, out.past_key_values


def make_backbone(cfg):
    if cfg.backbone == "smollm2":
        return SmolLM2Backbone(cfg)
    return TransformerBackbone(cfg) if cfg.backbone == "transformer" else GRUBackbone(cfg)


def _enc(d, heads, layers):
    layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class BlockRefiner(nn.Module):
    """Predicts codebooks 1..Q-1 for K future frames in parallel (masked, iterative).

    Sequence per anchor: K frames x Q slots = [cb0 (given), cb1..cb(Q-1) (masked or filled)],
    bidirectional inside the block, conditioned on the anchor hidden state.
    """

    def __init__(self, cfg):
        super().__init__()
        self.Q, self.V, self.K = cfg.Q, cfg.V, cfg.K
        r = cfg.ref_d
        self.cb0_emb = nn.Embedding(cfg.V, r)
        self.ac_emb = nn.Embedding((cfg.Q - 1) * (cfg.V + 1), r)   # +1 = MASK id per codebook
        self.pos = nn.Parameter(torch.zeros(cfg.K * cfg.Q, r))
        nn.init.normal_(self.pos, std=0.02)
        self.h_proj = nn.Linear(cfg.d, r, bias=False)
        self.enc = _enc(r, cfg.ref_heads, cfg.ref_layers)
        self.norm = nn.LayerNorm(r)
        self.head = nn.Parameter(torch.randn(cfg.Q - 1, r, cfg.V) * r ** -0.5)

    def forward(self, h, cb0, ac):
        """h: (N,d); cb0: (N,K); ac: (N,K,Q-1) (value V == MASK) -> logits (N,K,Q-1,V)."""
        N, K = cb0.shape
        Q, V = self.Q, self.V
        off = (torch.arange(Q - 1, device=ac.device) * (V + 1))[None, None]
        ac_e = self.ac_emb(ac + off)                              # (N,K,Q-1,r)
        c_e = self.cb0_emb(cb0)[:, :, None]                       # (N,K,1,r)
        x = torch.cat([c_e, ac_e], 2).reshape(N, K * Q, -1)
        x = x + self.pos[None] + self.h_proj(h)[:, None]
        z = self.norm(self.enc(x)).view(N, K, Q, -1)[:, :, 1:]    # acoustic slots only
        return torch.einsum("nkqr,qrv->nkqv", z, self.head)

    @torch.no_grad()
    def decode(self, h, cb0, steps: int):
        """MaskGIT-style iterative parallel decoding (greedy). Returns (N,K,Q-1), #passes."""
        N, K = cb0.shape
        Q1, V = self.Q - 1, self.V
        ac = torch.full((N, K, Q1), V, dtype=torch.long, device=h.device)
        total = K * Q1
        for i in range(steps):
            logits = self(h, cb0, ac)
            conf, tok = logits.softmax(-1).max(-1)
            is_mask = ac == V
            conf = torch.where(is_mask, conf, torch.full_like(conf, float("inf")))
            ac = torch.where(is_mask, tok, ac)
            remain = int(total * math.cos(math.pi / 2 * (i + 1) / steps)) if i < steps - 1 else 0
            if remain > 0:
                idx = conf.view(N, -1).topk(remain, dim=1, largest=False).indices
                ac.view(N, -1).scatter_(1, idx, V)
        return ac, steps


class DepthAR(nn.Module):
    """Moshi-style AR depth module (baseline): cb1..cb(Q-1) sequentially given h and cb0."""

    def __init__(self, cfg):
        super().__init__()
        self.Q, self.V = cfg.Q, cfg.V
        r = cfg.ref_d
        self.h_proj = nn.Linear(cfg.d, r, bias=False)
        self.emb = nn.Embedding(cfg.Q * cfg.V, r)
        self.pos = nn.Parameter(torch.zeros(cfg.Q, r))
        nn.init.normal_(self.pos, std=0.02)
        self.enc = _enc(r, cfg.ref_heads, cfg.dep_layers)
        self.norm = nn.LayerNorm(r)
        self.head = nn.Parameter(torch.randn(cfg.Q - 1, r, cfg.V) * r ** -0.5)

    def forward(self, h, codes):
        """h:(N,d) codes:(N,Q) teacher-forced. Input=[h, cb0..cb(Q-2)] -> logits for cb1..cb(Q-1)."""
        Q, V = self.Q, self.V
        off = (torch.arange(Q - 1, device=h.device) * V)[None]
        e = self.emb(codes[:, : Q - 1] + off)
        x = torch.cat([self.h_proj(h)[:, None], e], 1) + self.pos[None, :Q]
        causal = torch.triu(torch.full((Q, Q), float("-inf"), device=h.device), 1)
        z = self.norm(self.enc(x, mask=causal))[:, 1:]            # z[:,j] predicts cb(j+1)
        return torch.einsum("nqr,qrv->nqv", z, self.head)

    @torch.no_grad()
    def decode(self, h, cb0):
        N = h.shape[0]
        codes = torch.zeros(N, self.Q, dtype=torch.long, device=h.device)
        codes[:, 0] = cb0
        for j in range(1, self.Q):                                # Q-1 sequential passes
            codes[:, j] = self(h, codes)[:, j - 1].argmax(-1)
        return codes[:, 1:], self.Q - 1
