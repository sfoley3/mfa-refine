"""TIMIT -> benchmark corpora with exact ground truth.

Builds, from one TIMIT split:
  utt/         one file per utterance (best case for MFA)
  long/        each speaker's utterances joined by 0.3-4 s pauses (long recordings like the target data)
  long_disfl/  as long/, plus spliced repetitions and deleted words that the transcript doesn't show
Each has audio/ (<stem>.wav + <stem>.lab, the MFA corpus) and ref/ (<stem>.TextGrid ground truth).
Stems start with the 5-character speaker id, so `mfa align --speaker_characters 5` adapts per speaker.
"""
import csv
import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from .audio import SR
from .textgrid import fill_tier, write_textgrid

# The 24 core-test speakers (Lee & Hon / Kaldi TIMIT recipe)
CORE_TEST = {"MDAB0", "MWBT0", "FELC0", "MTAS1", "MWEW0", "FPAS0", "MJMP0", "MLNT0", "FPKT0",
             "MLLL0", "MTLS0", "FJLM0", "MBPM0", "MKLT0", "FNLP0", "MCMJ0", "MJDH0", "FMGD0",
             "MGRT0", "MNJM0", "FDHC0", "MJLN0", "MPAM0", "FMLD0"}

SILENCE = {"h#", "pau", "epi"}
# closure -> the releases it merges with (MFA labels closure + release as one stop)
CLOSURES = {"bcl": ("b",), "dcl": ("d", "jh"), "gcl": ("g",), "pcl": ("p",), "tcl": ("t", "ch"), "kcl": ("k",)}
# TIMIT phones -> MFA english_us_arpa phones (no stress). Glottal stop q has no ARPAbet phone and
# is merged into the following phone.
TO_ARPA = {"ax": "AH", "ax-h": "AH", "ix": "IH", "axr": "ER", "ux": "UW", "el": "L", "em": "M",
           "en": "N", "eng": "NG", "nx": "N", "hv": "HH", "dx": "T"}
TO_ARPA.update({c: r[0].upper() for c, r in CLOSURES.items()})


def fold_phones(phns):
    """[(a, b, timit_phone)] -> [(a, b, ARPA)] without silence: closures merged into their release,
    q merged into the next phone."""
    out, i, pending = [], 0, None  # pending: start time carried over from a q
    while i < len(phns):
        a, b, p = phns[i]
        if p in SILENCE:
            pending = None
            i += 1
            continue
        if p == "q":
            pending = a if pending is None else pending
            i += 1
            continue
        if p in CLOSURES and i + 1 < len(phns) and phns[i + 1][2] in CLOSURES[p]:
            b, p = phns[i + 1][1], phns[i + 1][2]
            i += 1
        out.append((a if pending is None else pending, b, TO_ARPA.get(p, p.upper())))
        pending = None
        i += 1
    return out


@dataclass
class Utterance:
    speaker: str
    name: str      # e.g. SI1386
    y: np.ndarray
    words: list    # [(a, b, word)] in seconds
    phones: list   # [(a, b, ARPA)] in seconds, no silence
    noise_rms: float


def _read_marks(path):
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        p = line.split()
        if len(p) >= 3:
            out.append((int(p[0]) / SR, int(p[1]) / SR, p[2]))
    return out


def _find(base, exts):
    for e in exts:
        for cand in (base + e, base + e.lower(), base + e.upper()):
            if os.path.exists(cand):
                return cand
    return None


def read_utterance(base, speaker):
    """base: path without extension, e.g. ROOT/TEST/DR1/FELC0/SI1386."""
    wav = _find(base, [".WAV.wav", ".wav", ".WAV", ".sph"])
    y, sr = sf.read(wav, dtype="float32")
    if sr != SR:
        raise SystemExit(f"{wav}: expected {SR} Hz, got {sr}")
    phns = _read_marks(_find(base, [".PHN"]))
    words = [(a, b, w.lower()) for a, b, w in _read_marks(_find(base, [".WRD"]))]
    sil = np.concatenate([y[int(a * SR):int(b * SR)] for a, b, p in phns if p == "h#"] or [y[:1]])
    return Utterance(speaker, os.path.basename(base).upper(), y, words, fold_phones(phns),
                     float(np.sqrt(np.mean(sil.astype(np.float64) ** 2))))


def list_utterances(root, split="core"):
    """[(speaker, base path)] for split 'core' (core test), 'test' or 'train'; SA sentences excluded."""
    top = "TRAIN" if split == "train" else "TEST"
    d = next((os.path.join(root, x) for x in os.listdir(root) if x.upper() == top), None)
    if d is None:
        raise SystemExit(f"no {top} folder in {root}")
    out = []
    for dirpath, _, files in os.walk(d):
        spk = os.path.basename(dirpath).upper()
        if split == "core" and spk not in CORE_TEST:
            continue
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.upper() == ".PHN" and not stem.upper().startswith("SA"):
                out.append((spk, os.path.join(dirpath, stem)))
    if not out:
        raise SystemExit(f"no TIMIT utterances found under {d}")
    return sorted(out)


# ------------------------------------------------------------------ assembly
def assemble(pieces, rng, noise_rms):
    """Join pieces into one recording with exact labels.
    pieces: (utt, lo, hi) to copy utt audio [lo, hi) with every label inside it, or a float = pause
    of that many seconds (Gaussian noise at noise_rms). Returns (y, words, phones)."""
    ys, words, phones, n = [], [], [], 0
    for p in pieces:
        if isinstance(p, float):
            k = int(round(p * SR))
            ys.append((rng.standard_normal(k) * noise_rms).astype(np.float32))
            n += k
            continue
        u, lo, hi = p
        i0, i1 = int(round(lo * SR)), int(round(hi * SR))
        shift = (n - i0) / SR
        inside = lambda a, b: a >= i0 / SR - 1e-6 and b <= i1 / SR + 1e-6
        words += [(a + shift, b + shift, w) for a, b, w in u.words if inside(a, b)]
        phones += [(a + shift, b + shift, t) for a, b, t in u.phones if inside(a, b)]
        ys.append(u.y[i0:i1])
        n += i1 - i0
    return np.concatenate(ys), words, phones


def disfluency(u, rng):
    """Pieces for u with one spliced disfluency, and a note; (None, None) if no word can be cut cleanly.
    'repeat': words k..k+n-1 are said, a pause, then the sentence resumes from word k (a restart).
    'delete': word k is cut out of the audio but stays in the transcript."""
    w = u.words
    ok = [k for k in range(len(w)) if (k == 0 or w[k - 1][1] <= w[k][0])
          and (k == len(w) - 1 or w[k][1] <= w[k + 1][0])]
    if not ok:
        return None, None
    k = int(rng.choice(ok))
    dur = len(u.y) / SR
    if rng.random() < 0.6:
        n = 1
        if k + 1 < len(w) and (k + 1) in ok and rng.random() < 0.5:
            n = 2
        a, b = w[k][0], w[k + n - 1][1]
        note = ("repeat", a, " ".join(x[2] for x in w[k:k + n]))
        return [(u, 0.0, a), (u, a, b), float(rng.uniform(0.2, 0.8)), (u, a, dur)], note
    a, b, _ = w[k]
    return [(u, 0.0, a), float(rng.uniform(0.1, 0.5)), (u, b, dur)], ("delete", a, w[k][2])


def _write(out, stem, y, words, phones, transcript):
    dur = len(y) / SR
    sf.write(os.path.join(out, "audio", stem + ".wav"), y, SR, subtype="PCM_16")
    open(os.path.join(out, "audio", stem + ".lab"), "w", encoding="utf-8").write(" ".join(transcript) + "\n")
    write_textgrid(os.path.join(out, "ref", stem + ".TextGrid"), dur,
                   [("words", fill_tier(sorted(words), dur)), ("phones", fill_tier(sorted(phones), dur))])


def build(root, out, split="core", seed=0, p_disfl=0.5):
    """Write the utt/, long/ and long_disfl/ corpora (see module docstring) under out."""
    rng = np.random.default_rng(seed)
    for sub in ("utt", "long", "long_disfl"):
        for d in ("audio", "ref"):
            os.makedirs(os.path.join(out, sub, d), exist_ok=True)
    by_spk = {}
    for spk, base in list_utterances(root, split):
        by_spk.setdefault(spk, []).append(read_utterance(base, spk))
    notes = []
    for spk, utts in sorted(by_spk.items()):
        noise = float(np.median([u.noise_rms for u in utts]))
        script = [w for u in utts for _, _, w in u.words]
        for u in utts:
            _write(os.path.join(out, "utt"), f"{spk}_{u.name}", u.y, u.words, u.phones, [w for _, _, w in u.words])
        clean, disfl = [], []
        for i, u in enumerate(utts):
            if i:
                gap = float(rng.uniform(0.3, 4.0))
                clean.append(gap)
                disfl.append(gap)
            whole = (u, 0.0, len(u.y) / SR)
            clean.append(whole)
            pieces, note = disfluency(u, rng) if rng.random() < p_disfl else (None, None)
            if pieces:
                start = sum(p if isinstance(p, float) else p[2] - p[1] for p in disfl)
                notes.append({"file": f"{spk}_long", "time": round(start + note[1], 3), "op": note[0],
                              "words": note[2]})
            disfl += pieces or [whole]
        _write(os.path.join(out, "long"), f"{spk}_long", *assemble(clean, rng, noise), script)
        _write(os.path.join(out, "long_disfl"), f"{spk}_long", *assemble(disfl, rng, noise), script)
    with open(os.path.join(out, "long_disfl", "disfluencies.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "time", "op", "words"])
        w.writeheader()
        w.writerows(notes)
    n_utt = sum(map(len, by_spk.values()))
    print(f"{n_utt} utterances from {len(by_spk)} speakers -> {out}/{{utt,long,long_disfl}}; "
          f"{len(notes)} disfluencies spliced")
