import torch
from bpm.config import ModelConfig
from bpm.model import S2SModel
from bpm.streaming import Streamer

def mk(kind="bpm", K=4):
    torch.manual_seed(0)
    cfg = ModelConfig(Q=3, V=32, d=32, n_layers=2, n_heads=4, window=16, K=K, ref_d=32, ref_layers=1,
                      ref_heads=4, refine_steps=2, dep_layers=1, kind=kind)
    return S2SModel(cfg)

def run(st, T, user=lambda t: [1 + (t % 5), 3, 4]):
    emitted = []
    for t in range(T):
        emitted.append(st.output_frame()); st.observe(user(t))
    return emitted

def test_bpm_needs_fewer_backbone_calls_than_ar():
    T = 40
    bpm = Streamer(mk("bpm", 4), [0, 0, 0], revision_tau=None, wake_on_activity=False)
    ar = Streamer(mk("ar"), [0, 0, 0], revision_tau=None, wake_on_activity=False)
    run(bpm, T); run(ar, T)
    assert ar.stats["backbone_calls"] == T - 1          # frame 0 is the silent init frame
    assert bpm.stats["backbone_calls"] == (T - 1 + 3) // 4
    assert bpm.stats["backbone_calls"] < ar.stats["backbone_calls"] / 3

def test_first_frame_is_init_silence_and_frames_have_right_shape():
    st = Streamer(mk(), [7, 8, 9], revision_tau=None, wake_on_activity=False)
    codes, act = st.output_frame()
    assert codes == [7, 8, 9] and act == 0
    st.observe([0, 0, 0])
    codes, act = st.output_frame()
    assert len(codes) == 3 and all(0 <= c < 32 for c in codes)

def test_cancel_drops_unplayed_frames_and_replans():
    st = Streamer(mk(), [0, 0, 0], revision_tau=None, wake_on_activity=False)
    for t in range(3):
        st.output_frame(); st.observe([0, 0, 0])
    st.output_frame(); st.observe([0, 0, 0])             # triggers a wake at t=1, plan has 3 left
    assert len(st.plan) >= 1
    before = st.stats["wakes"]
    st.cancel()
    assert len(st.plan) == 0 and st.stats["cancels"] == 1
    st.output_frame()
    assert st.stats["wakes"] == before + 1               # re-planned from fresh context immediately

def test_activity_flip_replans_and_emitted_frames_are_never_modified():
    st = Streamer(mk(), [0, 0, 0], revision_tau=None, wake_on_activity=True)
    emitted = []
    user = lambda t: [0, 0, 0] if t < 6 else [5, 1, 1]   # user starts speaking at t=6
    for t in range(12):
        emitted.append(st.output_frame()); st.observe(user(t))
    assert st.stats["activity_replans"] >= 1
    # re-run identically: deterministic greedy decoding must reproduce the same emitted frames
    st2 = Streamer(mk(), [0, 0, 0], revision_tau=None, wake_on_activity=True)
    emitted2 = []
    for t in range(12):
        emitted2.append(st2.output_frame()); st2.observe(user(t))
    assert emitted == emitted2

def test_revision_triggers_on_surprising_user_token():
    st = Streamer(mk(), [0, 0, 0], revision_tau=-1e9, wake_on_activity=False)   # always "surprised"
    run(st, 12)
    assert st.stats["revisions"] >= 1
