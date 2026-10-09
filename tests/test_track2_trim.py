import numpy as np

from sonic.track2 import model as m


def test_trim_end_cuts_after_last_loud_frame_plus_pad():
    a = np.zeros(16000 * 2, np.float32)
    a[8000:16000] = 0.5                      # speech from 0.5 s to 1.0 s
    key, end = m.trim_end(a)
    assert key == 16000 and end == 16000 + int(m.PAD * 16000)


def test_trim_end_silence_keeps_everything():
    a = np.zeros(4000, np.float32)
    assert m.trim_end(a) == (0, 4000)


def test_incremental_loudness_matches_trim_end():
    rng = np.random.default_rng(0)
    a = np.zeros(16000 * 3, np.float32)
    a[3000:20000] = rng.normal(0, 0.2, 17000)
    x = m.Model.__new__(m.Model)               # no experts: only the bookkeeping
    m.Model.reset(x)
    for i in range(0, len(a), 1600):
        x.chunks.append(a[i:i + 1600]); x.n += len(a[i:i + 1600]); x._track_loudness()
    assert x.last_loud_end == m.trim_end(a)[0]
