import torch
from bpm.config import ModelConfig
from bpm.model import S2SModel
from bpm.modules import SmolLM2Backbone
from bpm.streaming import Streamer


def test_smollm2_backbone_streaming():
    cfg = ModelConfig(backbone="smollm2", d=128, freeze_backbone=True)
    bb = SmolLM2Backbone(cfg)
    B, T, D = 1, 4, 128

    # First chunk (cold cache)
    x1 = torch.randn(B, T, D)
    h1, state1 = bb(x1, state=None)
    assert h1.shape == (B, T, D)
    assert state1 is not None

    # Second chunk (warm cache)
    x2 = torch.randn(B, 4, D)
    h2, state2 = bb(x2, state=state1)
    assert h2.shape == (B, 4, D)
    assert state2 is not None


def test_smollm2_s2s_model_loss_and_stream():
    cfg = ModelConfig(Q=4, V=256, backbone="smollm2", d=128, K=4, freeze_backbone=True)
    model = S2SModel(cfg)

    # Loss computation
    B, T, Q = 2, 16, 4
    u = torch.randint(0, 256, (B, T, Q))
    a = torch.randint(0, 256, (B, T, Q))
    s = torch.randint(0, 2, (B, T))
    loss, parts = model.loss(u, a, s)
    assert torch.isfinite(loss)
    assert loss.item() > 0
    loss.backward()

    # Streaming test
    st = Streamer(model, [0] * 4)
    codes, act = st.output_frame()
    assert len(codes) == 4
    assert act in (0, 1)
    st.observe([50] * 4)
    assert len(st.pu) == 1

