"""Model/training configuration. STATUS: working code (exercised by tests/)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import yaml


@dataclass
class ModelConfig:
    # --- token layout (Mimi: Q=8 codebooks, V=2048, 12.5 Hz; smoke test uses Q=4, V=256) ---
    Q: int = 8
    V: int = 2048
    frame_hz: float = 12.5
    # --- backbone (12.5 Hz, causal) ---
    backbone: str = "transformer"      # "transformer" | "gru" | "smollm2"
    d: int = 384
    n_layers: int = 8
    n_heads: int = 6
    window: int = 250                  # sliding attention window in frames (250 f = 20 s)
    ff_mult: int = 4
    pretrained_name: str = "HuggingFaceTB/SmolLM2-135M"
    freeze_backbone: bool = False
    # --- block-parallel head ---
    K: int = 4                         # future frames predicted per backbone call
    user_pred_weight: float = 0.1      # aux user-cb0 prediction (predictive coding + revision)
    # --- acoustic refiner (masked parallel decoding of codebooks 1..Q-1) ---
    ref_d: int = 256
    ref_layers: int = 4
    ref_heads: int = 4
    refine_steps: int = 4              # S: MaskGIT-style iterations at inference
    # --- AR baseline (Moshi-style depth module) ---
    dep_layers: int = 2
    kind: str = "bpm"                  # "bpm" | "ar"


@dataclass
class TrainConfig:
    steps: int = 3000
    batch: int = 64
    lr: float = 2e-3
    wd: float = 0.01
    warmup: int = 100
    seed: int = 0
    log_every: int = 100
    eval_n: int = 200
    out: str = "runs/smoke"
    T: int = 32
    Gmax: int = 2                      # synthetic-task onset jitter (see bpm/data.py)


def load_cfg(path: str):
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ModelConfig(**raw.get("model", {})), TrainConfig(**raw.get("train", {}))


def dump_cfg(m: ModelConfig, t: TrainConfig) -> dict:
    return {"model": asdict(m), "train": asdict(t)}
