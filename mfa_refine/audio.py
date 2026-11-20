"""Audio loading, energy VAD, and silence-based chunking."""
import numpy as np
import soundfile as sf

SR = 16000  # Whisper, MMS and the chunk corpus all run at 16 kHz
HOP = 0.01  # VAD frame hop (s)


def load(path):
    """Mono float32 at 16 kHz."""
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, SR)
        y = resample_poly(y, SR // g, sr // g).astype(np.float32)
    return y


def speech_mask(y, rel_threshold=0.35):
    """Frame-level speech mask: dB above floor + rel_threshold * (peak - floor)."""
    win, hop = int(0.025 * SR), int(HOP * SR)
    if len(y) < win:
        return np.zeros(0, bool)
    fr = np.lib.stride_tricks.sliding_window_view(y, win)[::hop]
    db = 10 * np.log10((fr.astype(np.float64) ** 2).mean(1) + 1e-12)
    lo, hi = np.percentile(db, 10), np.percentile(db, 99)
    return remove_short(db > lo + rel_threshold * (hi - lo), True, 5)  # drop blips < 50 ms


def remove_short(mask, value, min_frames):
    """Flip runs of `value` shorter than min_frames."""
    m = mask.copy()
    d = np.diff(np.r_[0, (m == value).astype(int), 0])
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if b - a < min_frames:
            m[a:b] = not value
    return m


def regions(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return [(a * HOP, b * HOP) for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0])]


def frac_speech(sp, a, b):
    i0 = int(round(a / HOP))
    seg = sp[max(i0, 0):max(int(round(b / HOP)), i0 + 1)]
    return float(seg.mean()) if len(seg) else 0.0


def make_chunks(sp, dur, min_sil=0.30, pad=0.10):
    """Speech regions separated by >= min_sil of silence, padded, as [lo, hi] pairs."""
    regs = regions(remove_short(sp, False, int(min_sil / HOP)))
    chunks = []
    for i, (a, b) in enumerate(regs):
        lo = max(0.0, a - pad) if i == 0 else max(a - pad, (regs[i - 1][1] + a) / 2)
        hi = min(dur, b + pad) if i == len(regs) - 1 else min(b + pad, (b + regs[i + 1][0]) / 2)
        chunks.append([lo, hi])
    return chunks


def word_metrics(words, sp):
    """QA numbers for a word tier."""
    cov = np.zeros(len(sp) + 1, bool)
    on_sil = long_ = 0
    for a, b, t in words:
        if not t:
            continue
        cov[int(a / HOP):int(b / HOP) + 1] = True
        on_sil += frac_speech(sp, a, b) < 0.3
        long_ += (b - a) > 1.0
    return {"uncovered_speech_s": round(float((sp & ~cov[:len(sp)]).sum() * HOP), 2),
            "words_on_silence": int(on_sil), "words_over_1s": int(long_)}
