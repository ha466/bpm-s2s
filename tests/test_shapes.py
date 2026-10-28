import pytest, torch
from bpm.config import ModelConfig
from bpm.model import S2SModel, count_params
from bpm.budget import budget

def tiny(**kw):
    base = dict(Q=4, V=64, d=32, n_layers=2, n_heads=4, window=16, K=4, ref_d=32, ref_layers=1,
                ref_heads=4, refine_steps=3, dep_layers=1)
    base.update(kw)
    return ModelConfig(**base)

@pytest.mark.parametrize("kind", ["bpm", "ar"])
@pytest.mark.parametrize("backbone", ["transformer", "gru"])
def test_loss_runs_and_backprops(kind, backbone):
    cfg = tiny(kind=kind, backbone=backbone)
    m = S2SModel(cfg)
    B, T = 3, 20
    u = torch.randint(0, cfg.V, (B, T, cfg.Q)); a = torch.randint(0, cfg.V, (B, T, cfg.Q))
    s = torch.randint(0, 2, (B, T))
    loss, parts = m.loss(u, a, s)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in m.refiner.parameters() if p.requires_grad)
    assert set(parts) == {"cb0", "act", "user0", "acoustic"}

def test_encode_shape_and_plan_shapes():
    cfg = tiny()
    m = S2SModel(cfg).eval()
    u = torch.randint(0, cfg.V, (1, 7, cfg.Q)); a = torch.randint(0, cfg.V, (1, 7, cfg.Q)); s = torch.zeros(1, 7, dtype=torch.long)
    h, st = m.encode(u, a, s)
    assert h.shape == (1, 7, cfg.d)
    out = m.plan(h[:, -1])
    assert out["cb0"].shape == (cfg.K,) and out["ac"].shape == (cfg.K, cfg.Q - 1) and out["u_logp"].shape == (cfg.K, cfg.V)
    assert out["passes"] == cfg.refine_steps

def test_refiner_decode_leaves_no_mask_tokens():
    cfg = tiny()
    m = S2SModel(cfg).eval()
    h = torch.randn(5, cfg.d); cb0 = torch.randint(0, cfg.V, (5, cfg.K))
    for steps in (1, 2, 5):
        ac, _ = m.refiner.decode(h, cb0, steps)
        assert ac.shape == (5, cfg.K, cfg.Q - 1) and int(ac.max()) < cfg.V and int(ac.min()) >= 0

def test_ar_depth_decode():
    cfg = tiny(kind="ar")
    m = S2SModel(cfg).eval()
    ac, passes = m.refiner.decode(torch.randn(2, cfg.d), torch.randint(0, cfg.V, (2,)))
    assert ac.shape == (2, cfg.Q - 1) and passes == cfg.Q - 1

def test_budget_matches_real_param_count():
    for cfg in (tiny(), tiny(kind="ar"), tiny(backbone="gru")):
        real = count_params(S2SModel(cfg))["total"]
        assert budget(cfg)["params"]["total"] == real
