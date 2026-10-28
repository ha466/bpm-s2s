"""Parameter / memory / FLOP budget from a ModelConfig. STATUS: working code.

Parameter counts are EXACT for this implementation (model instantiated on the 'meta' device).
Memory and FLOP numbers are ESTIMATES from the stated equations; they exclude the codec (Mimi), the
Python/runtime overhead and audio I/O, which must be measured on the target machine.
"""
from __future__ import annotations
import torch
from .config import ModelConfig
from .model import S2SModel, count_params

BYTES = {"fp32": 4, "fp16/bf16": 2, "int8": 1}


def budget(cfg: ModelConfig) -> dict:
    with torch.device("meta"):
        model = S2SModel(cfg)
    pc = count_params(model)
    d, L, W, K, Q, V = cfg.d, cfg.n_layers, cfg.window, model.K, cfg.Q, cfg.V
    P = pc["total"]
    emb = pc["emb_u"] + pc["emb_a"] + pc["emb_s"]
    bb = pc["backbone"]
    heads = pc["fut"] + pc["fut_norm"] + pc["head_cb0"] + pc["head_s"] + pc["head_u0"]
    ref = pc["refiner"]
    out = {"params": {"user_embedding": pc["emb_u"], "own_embedding": pc["emb_a"] + pc["emb_s"],
                      "backbone": bb, "block_heads": heads, "refiner_or_depth": ref, "total": P}}
    out["weights_MB"] = {k: round(P * b / 1e6, 1) for k, b in BYTES.items()}
    # ---- state memory (estimate) ----
    if cfg.backbone == "transformer":
        state = L * 2 * (W - 1) * d                    # K and V, no GQA
    else:
        state = L * d
    out["state_MB_fp16"] = round(state * 2 / 1e6, 2)
    # ---- compute (estimate; 1 MAC = 2 FLOPs) ----
    fps = cfg.frame_hz
    bb_matmul = 12 * d * d * L if cfg.backbone == "transformer" else 6 * d * d * L   # GRU: 3 gates x 2 mats
    attn = 4 * L * d * min(W, 250) if cfg.backbone == "transformer" else 0          # worst case full window
    ingest = fps * (2 * bb_matmul + attn)                                           # per second of INPUT audio
    r, rl = cfg.ref_d, cfg.ref_layers if model.is_bpm else cfg.dep_layers
    head_per_anchor = 2 * d * K * d + K * 2 * (2 * d * V) + 2 * 2 * d
    if model.is_bpm:
        toks = K * Q
        per_pass = 2 * 12 * r * r * rl * toks + 2 * (Q - 1) * K * r * V
        gen_blocks = fps / K
        gen = gen_blocks * (head_per_anchor + cfg.refine_steps * per_pass)
        passes = gen_blocks * cfg.refine_steps
    else:
        # FAIR estimate: assumes the depth module uses a KV cache (total tokens processed = Q per frame).
        # The reference implementation in bpm/modules.py does NOT cache (it recomputes), so its measured
        # cost is higher than this estimate.
        gen = fps * (head_per_anchor + 2 * 12 * r * r * rl * Q + 2 * (Q - 1) * r * V)
        passes = fps * (Q - 1)
    out["GFLOPs_per_audio_second"] = {"ingest_input": round(ingest / 1e9, 3), "generate_output": round(gen / 1e9, 3),
                                      "total_full_duplex": round((ingest + gen) / 1e9, 3)}
    out["sequential_passes_per_second"] = {"backbone_calls_per_s": fps / K, "refiner_or_depth_passes_per_s": passes,
                                           "total_sequential": fps / K + passes}
    # weight bytes read per second by backbone+heads (memory-bandwidth bound estimate, fp16)
    out["backbone_weight_traffic_MBps_fp16"] = round((fps / K) * (bb + heads) * 2 / 1e6, 1)
    return out


if __name__ == "__main__":
    import json, sys
    from .config import load_cfg
    m, _ = load_cfg(sys.argv[1])
    print(json.dumps(budget(m), indent=2))
