import pytest, torch
from bpm.config import ModelConfig
from bpm.model import S2SModel

def cfg(backbone, window):
    return ModelConfig(Q=3, V=32, d=32, n_layers=2, n_heads=4, window=window, K=4, ref_d=32,
                       ref_layers=1, ref_heads=4, backbone=backbone)

@pytest.mark.parametrize("backbone,window", [("transformer", 64), ("transformer", 8), ("gru", 8)])
@pytest.mark.parametrize("chunks", [[1] * 24, [4] * 6, [1, 3, 5, 2, 7, 6]])
def test_chunked_streaming_equals_full_forward(backbone, window, chunks):
    """The KV-cache / recurrent state path must reproduce the parallel (training) path exactly."""
    torch.manual_seed(0)
    m = S2SModel(cfg(backbone, window)).eval()
    T = sum(chunks)
    u = torch.randint(0, 32, (1, T, 3)); a = torch.randint(0, 32, (1, T, 3)); s = torch.randint(0, 2, (1, T))
    with torch.no_grad():
        full, _ = m.encode(u, a, s)
        state, outs, i = None, [], 0
        for c in chunks:
            h, state = m.encode(u[:, i:i + c], a[:, i:i + c], s[:, i:i + c], state)
            outs.append(h); i += c
    assert torch.allclose(full, torch.cat(outs, 1), atol=1e-4, rtol=1e-4)

def test_cache_is_bounded_by_window():
    m = S2SModel(cfg("transformer", 8)).eval()
    state = None
    with torch.no_grad():
        for _ in range(30):
            _, state = m.encode(torch.randint(0, 32, (1, 3, 3)), torch.randint(0, 32, (1, 3, 3)),
                                torch.zeros(1, 3, dtype=torch.long), state)
    assert all(c[0].shape[2] <= 7 for c in state["caches"])   # window-1 keys kept
