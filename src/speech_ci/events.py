"""Label-free event features from a frozen VAD; all units in seconds."""
import hashlib
import numpy as np


def summary(values):
    x = np.log1p(np.asarray(values, float))
    if not len(x):
        return np.zeros(6)
    return np.r_[x.mean(), x.std(), np.quantile(x, [.25, .5, .75, .9])]


def progression(midpoint, duration, total):
    if len(duration) < 3:
        return np.zeros(2)
    t = np.asarray(midpoint) / total - .5
    x = np.column_stack([np.ones(len(t)), t, t*t])
    if np.linalg.matrix_rank(x) < 3:
        return np.zeros(2)
    return np.linalg.lstsq(x, np.log1p(duration), rcond=None)[0][1:]


def sequence_stats(duration):
    x = np.log1p(duration)
    difference = float(np.abs(np.diff(x)).mean()) if len(x) > 1 else 0.
    correlation = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 2 and np.std(x[:-1]) > 1e-10 and np.std(x[1:]) > 1e-10 else 0.
    return correlation, difference


def extract(segments, total):
    s = np.asarray(segments, float).reshape(-1, 2)
    assert total > 0 and np.isfinite(s).all()
    if len(s):
        assert (s[:, 0] >= 0).all() and (s[:, 1] <= total+1e-9).all()
        assert (s[:, 1] > s[:, 0]).all() and (s[1:, 0] >= s[:-1, 1]).all()
    duration = np.diff(s, axis=1).ravel()
    gaps = s[1:, 0] - s[:-1, 1] if len(s) > 1 else np.empty(0)
    leading = s[0, 0] if len(s) else total
    trailing = total-s[-1, 1] if len(s) else 0.
    static = np.r_[np.log1p(total), duration.sum()/total, 60*len(s)/total,
                   leading/total, trailing/total, float(not len(s)), summary(duration), summary(gaps),
                   [np.mean(gaps > t) if len(gaps) else 0. for t in [.5, 1., 2.]]]
    occupancy, onsets = [], []
    for lo, hi in zip(np.linspace(0, total, 5)[:-1], np.linspace(0, total, 5)[1:]):
        occupied = np.maximum(0., np.minimum(s[:, 1], hi)-np.maximum(s[:, 0], lo)).sum()
        occupancy.append(occupied/(hi-lo))
        onsets.append(60*np.sum((s[:, 0] >= lo) & (s[:, 0] < hi))/(hi-lo))
    a, b = s[:, 0]/total, s[:, 1]/total
    # Integrals over normalized time, with zeros during silence.
    linear = np.sum((b*b-b)-(a*a-a))
    quadratic = np.sum((2*b**3-3*b*b+b)-(2*a**3-3*a*a+a))
    speech_progress = progression(s.mean(1), duration, total)
    gap_mid = .5*(s[:-1, 1]+s[1:, 0]) if len(s)>1 else np.empty(0)
    gap_progress = progression(gap_mid, gaps, total)
    dynamic = np.r_[occupancy, onsets, linear, quadratic, speech_progress, gap_progress,
                    sequence_stats(duration), sequence_stats(gaps)]
    assert static.shape == (21,) and dynamic.shape == (18,)
    assert np.isfinite(static).all() and np.isfinite(dynamic).all()
    return static, dynamic


def shuffle(segments, total, pid, task, seed):
    s = np.asarray(segments, float).reshape(-1, 2)
    if len(s)<2:
        return s.copy()
    key = int.from_bytes(hashlib.sha256(f'{seed}|{pid}|{task}'.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(key)
    duration = rng.permutation(s[:, 1]-s[:, 0])
    gaps = rng.permutation(s[1:, 0]-s[:-1, 1])
    out = np.zeros_like(s)
    at = s[0, 0]
    for i, length in enumerate(duration):
        out[i] = [at, at+length]
        at += length+(gaps[i] if i<len(gaps) else 0.)
    # Remove tiny accumulated floating-point endpoint differences only.
    out[-1, 1] = s[-1, 1]
    np.testing.assert_allclose(extract(out, total)[0], extract(s, total)[0], atol=1e-9, rtol=1e-9)
    return out
