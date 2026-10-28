import numpy as np
import torch
from bpm.config import ModelConfig
from bpm.data import RealVoiceTask, Spec
from bpm.model import S2SModel


def test_real_voice_task_shapes():
    spec = Spec(Q=4, V=256, T=32)
    task = RealVoiceTask(spec)
    rng = np.random.RandomState(42)

    # 1. Test single sample
    u, a, s, meta = task.sample(rng)
    assert u.shape == (32, 4)
    assert a.shape == (32, 4)
    assert s.shape == (32,)
    assert (u >= 0).all() and (u < 256).all()
    assert (a >= 0).all() and (a < 256).all()
    assert set(np.unique(s)).issubset({0, 1})

    # 2. Test batch
    batch_u, batch_a, batch_s, metas = task.batch(4, rng)
    assert batch_u.shape == (4, 32, 4)
    assert batch_a.shape == (4, 32, 4)
    assert batch_s.shape == (4, 32)
    assert len(metas) == 4

    # 3. Test loss computation with S2SModel
    mcfg = ModelConfig(Q=4, V=256, backbone="transformer", d=64, n_layers=2, n_heads=2, window=16, K=4,
                       ref_d=48, ref_layers=1, ref_heads=2, refine_steps=2)
    model = S2SModel(mcfg)
    loss, parts = model.loss(batch_u, batch_a, batch_s)
    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert "cb0" in parts and "acoustic" in parts and "act" in parts

