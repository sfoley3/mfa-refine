"""The refinement pipeline: transcribe -> prepare -> align -> ctc -> stitch.

Every step caches its output under the work directory, so a rerun only redoes what changed.
"""
import csv
import fnmatch
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from . import audio
from .audio import HOP, SR, frac_speech
from .reconcile import reconcile
from .text import ELONGATED, HALLUC_PHRASE, Pronouncer, is_hallucination, norm_token
from .textgrid import fill_tier, read_textgrid, write_textgrid

STOP_PHONES = {"P", "B", "T", "D", "K", "G", "CH", "JH",  # ARPAbet
               "p", "b", "t", "d", "k", "ɡ", "g", "tʃ", "dʒ", "pʰ", "tʰ", "kʰ", "c", "ɟ", "ʔ"}  # IPA (MFA 2+)
STEPS = ["transcribe", "prepare", "align", "ctc", "stitch"]


@dataclass
class Config:
    audio_dirs: list
    out_dir: str
    textgrid_dirs: list = field(default_factory=list)
    work_dir: str = None
    words_tier: str = "words"
    speaker_pattern: str = None  # regex; group 1 = speaker id (MFA adapts per speaker)
    include: list = field(default_factory=list)  # glob patterns on file stem
    exclude: list = field(default_factory=list)
    language: str = "en"
    whisper_model: str = "small.en"
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    dictionary: str = "english_us_arpa"
    acoustic_model: str = "english_us_arpa"
    g2p_model: str = "english_us_arpa"
    num_jobs: int = 4
    vad_threshold: float = 0.35
    min_silence: float = 0.30
    pad: float = 0.10
    snap_max: float = 0.05
    check_threshold: float = 0.15

    def __post_init__(self):
        self.work_dir = self.work_dir or os.path.join(self.out_dir, "_work")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ------------------------------------------------------------------ discovery
def discover(cfg):
    """[(speaker, name, wav_path, textgrid_path or None)] for the selected audio files."""
    tgs = {}
    for d in cfg.textgrid_dirs:
        for f in os.listdir(d):
            if f.endswith(".TextGrid"):
                tgs.setdefault(f[:-9], os.path.join(d, f))
    out = []
    for d, f in sorted((d, f) for d in cfg.audio_dirs for f in os.listdir(d)):
        name, ext = os.path.splitext(f)
        if ext.lower() not in (".wav", ".flac") or name.startswith("."):
            continue
        if cfg.include and not any(fnmatch.fnmatch(name, p) for p in cfg.include):
            continue
        if any(fnmatch.fnmatch(name, p) for p in cfg.exclude):
            continue
        tg = tgs.get(name)
        spk = "speaker"
        if cfg.speaker_pattern:
            m = re.search(cfg.speaker_pattern, name)
            spk = m.group(1) if m else "unmatched"
        out.append((spk, name, os.path.join(d, f), tg))
    if not out:
        raise SystemExit(f"no audio files selected in {cfg.audio_dirs}")
    dup = {n for _, n, _, _ in out if sum(o[1] == n for o in out) > 1}
    if dup:
        raise SystemExit(f"duplicate file names across audio folders: {sorted(dup)}")
    return out


def script_words(cfg, tg):
    if not tg:
        return []
    tiers = read_textgrid(tg)
    if cfg.words_tier not in tiers:
        raise SystemExit(f"{tg}: no tier named '{cfg.words_tier}' (has {list(tiers)})")
    return [t for a, b, t in tiers[cfg.words_tier] if t.strip() and t not in ("<eps>", "sil", "sp")]


def _chunks(cfg, y):
    sp = audio.speech_mask(y, cfg.vad_threshold)
    return sp, audio.make_chunks(sp, len(y) / SR, cfg.min_silence, cfg.pad)


def _whisper_dir(cfg):
    tag = f"{cfg.whisper_model.replace('/', '_')}_v{cfg.vad_threshold}_s{cfg.min_silence}_p{cfg.pad}"
    return os.path.join(cfg.work_dir, "whisper", tag)


# ------------------------------------------------------------------ 1. transcribe
def _load_whisper(cfg):
    """Whisper on the requested device; 'auto' picks CUDA if present, else CPU.
    Apple MPS is opt-in (--device mps): ctranslate2 marks it experimental."""
    import ctranslate2
    from faster_whisper import WhisperModel
    device = cfg.device
    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
    ct = cfg.compute_type if cfg.compute_type != "auto" else ("int8" if device == "cpu" else "float16")
    log(f"loading whisper {cfg.whisper_model} on {device} ({ct})")
    try:
        return WhisperModel(cfg.whisper_model, device=device, compute_type=ct)
    except Exception as e:
        if cfg.device != "auto" or device == "cpu":
            raise
        log(f"{device} failed ({e}); falling back to cpu (int8)")
        return WhisperModel(cfg.whisper_model, device="cpu", compute_type="int8")


def transcribe(cfg, files):
    """Whisper on each silence-separated chunk. (Whole-file Whisper skips isolated list words.)"""
    d = _whisper_dir(cfg)
    os.makedirs(d, exist_ok=True)
    todo = [f for f in files if not os.path.exists(os.path.join(d, f[1] + ".json"))]
    if not todo:
        return
    model = _load_whisper(cfg)
    pad = np.zeros(int(0.3 * SR), np.float32)
    for spk, name, wav, tg in todo:
        t0 = time.time()
        y = audio.load(wav)
        _, chunks = _chunks(cfg, y)
        prompt = " ".join(script_words(cfg, tg))[:600] or None
        res = []
        for lo, hi in chunks:
            segs, _ = model.transcribe(np.r_[pad, y[int(lo * SR):int(hi * SR)], pad],
                                       language=cfg.language, word_timestamps=True, vad_filter=False,
                                       condition_on_previous_text=False, beam_size=cfg.beam_size,
                                       initial_prompt=prompt)
            res.append({"lo": lo, "hi": hi, "words": [
                {"w": w.word, "s": lo - 0.3 + w.start, "e": lo - 0.3 + w.end, "p": w.probability}
                for seg in segs for w in (seg.words or [])]})
        json.dump(res, open(os.path.join(d, name + ".json"), "w"), indent=1)
        log(f"transcribed {name}: {len(chunks)} chunks in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ 2. prepare
def prepare(cfg, files):
    """Filter Whisper output, reconcile with the script, write the chunked MFA corpus."""
    corpus = os.path.join(cfg.work_dir, "corpus")
    shutil.rmtree(corpus, ignore_errors=True)
    pron = Pronouncer(cfg.dictionary)
    meta = {}
    for spk, name, wav, tg in files:
        y = audio.load(wav)
        sp = audio.speech_mask(y, cfg.vad_threshold)
        notes, voiced, hyps, untranscribed = [], [], [], []
        for c in json.load(open(os.path.join(_whisper_dir(cfg), name + ".json"))):
            text = " ".join(w["w"].strip() for w in c["words"])
            if is_hallucination(text, c["hi"] - c["lo"]):
                if text:
                    notes.append(f"rejected_chunk:{c['lo']:.2f}-{c['hi']:.2f}:'{text}'")
                untranscribed.append((c["lo"], c["hi"]))
                continue
            toks = []
            for w in c["words"]:
                if w["p"] < 0.3 and frac_speech(sp, w["s"] - 0.2, w["e"] + 0.2) == 0:
                    notes.append(f"drop_low_conf:{w['w'].strip()}@{w['s']:.2f}")
                    continue
                for t in norm_token(w["w"]):
                    if ELONGATED.search(t):
                        notes.append(f"drop_elongated:{t}@{w['s']:.2f}")
                    else:
                        toks.append(t)
            joined = " ".join(toks)
            if HALLUC_PHRASE.search(joined):
                notes.append(f"strip_phrase:'{joined}'")
                toks = HALLUC_PHRASE.sub(" ", joined).split()
            if toks:
                voiced.append((c["lo"], c["hi"]))
                hyps.append(toks)
            else:
                untranscribed.append((c["lo"], c["hi"]))
        labels = reconcile(hyps, [b - a for a, b in voiced], script_words(cfg, tg), notes, pron=pron)
        os.makedirs(os.path.join(corpus, spk), exist_ok=True)
        chunks = []
        for k, ((lo, hi), ws) in enumerate(zip(voiced, labels)):
            if not ws:
                untranscribed.append((lo, hi))
                continue
            cid = f"{name}__c{k:03d}"
            base = os.path.join(corpus, spk, cid)
            sf.write(base + ".wav", y[int(lo * SR):int(hi * SR)], SR)
            open(base + ".lab", "w", encoding="utf-8").write(" ".join(ws) + "\n")
            chunks.append({"id": cid, "lo": lo, "hi": hi, "words": ws,
                           "heard": hyps[k]})
        meta[name] = {"speaker": spk, "dur": len(y) / SR, "chunks": chunks, "notes": notes,
                      "untranscribed": sorted((round(a, 2), round(b, 2)) for a, b in untranscribed)}
        log(f"prepared {name}: {sum(map(len, labels))} words in {len(chunks)} chunks, {len(notes)} notes")
    json.dump(meta, open(os.path.join(cfg.work_dir, "meta.json"), "w"), indent=1)


# ------------------------------------------------------------------ 3. align
def align(cfg, files):
    out = os.path.join(cfg.work_dir, "aligned")
    cmd = ["mfa", "align", os.path.join(cfg.work_dir, "corpus"), cfg.dictionary, cfg.acoustic_model, out,
           "--clean", "--beam", "100", "--retry_beam", "400", "--num_jobs", str(cfg.num_jobs)]
    if cfg.g2p_model:
        cmd += ["--g2p_model_path", cfg.g2p_model]
    log(" ".join(cmd))
    subprocess.run(cmd, check=True)


def _aligned_path(cfg, spk, cid):
    for p in (os.path.join(cfg.work_dir, "aligned", spk, cid + ".TextGrid"),
              os.path.join(cfg.work_dir, "aligned", cid + ".TextGrid")):
        if os.path.exists(p):
            return p
    return None


# ------------------------------------------------------------------ 4. ctc cross-check
def ctc_viterbi(lp, word_tokens, blank=0):
    """CTC forced alignment of token sequences to log-probs lp [frames, vocab].
    Returns (first_frame, end_frame) per word. Plain numpy, so it does not depend on
    torchaudio having been built with its alignment extension (conda-forge builds are not)."""
    tokens = [t for w in word_tokens for t in w]
    owner = [i for i, w in enumerate(word_tokens) for _ in w]
    T, S = lp.shape[0], 2 * len(tokens) + 1
    ext = np.full(S, blank)
    ext[1::2] = tokens
    skip = np.zeros(S, bool)  # may jump over a blank: token differs from the token two states back
    skip[3::2] = ext[3::2] != ext[1:-2:2]
    dp = np.full((T, S), -np.inf)
    bp = np.zeros((T, S), np.int8)
    dp[0, :2] = lp[0, ext[:2]]
    for t in range(1, T):
        p = dp[t - 1]
        cand = np.stack([p, np.r_[-np.inf, p[:-1]], np.where(skip, np.r_[-np.inf, -np.inf, p[:-2]], -np.inf)])
        bp[t] = cand.argmax(0)
        dp[t] = cand.max(0) + lp[t, ext]
    s = S - 1 if S == 1 or dp[-1, S - 1] >= dp[-1, S - 2] else S - 2
    if not np.isfinite(dp[-1, s]):
        raise ValueError("audio too short for the transcript")
    spans = {}
    for t in range(T - 1, -1, -1):
        if s % 2:
            w = owner[s // 2]
            a, b = spans.get(w, (t, t + 1))
            spans[w] = (t, b)
        s -= int(bp[t, s])
    return [spans[i] for i in range(len(word_tokens))]


def ctc(cfg, files):
    """Word onsets from a second, independent aligner (torchaudio MMS_FA, CTC) for cross-checking."""
    import torch
    from torchaudio.pipelines import MMS_FA as BUNDLE
    d = os.path.join(cfg.work_dir, "ctc")
    os.makedirs(d, exist_ok=True)
    meta = json.load(open(os.path.join(cfg.work_dir, "meta.json")))
    model, tok = BUNDLE.get_model(with_star=False), BUNDLE.get_tokenizer()
    vocab = set(BUNDLE.get_dict())
    for name, m in meta.items():
        for c in m["chunks"]:
            words = [re.sub(r"[^a-z']", "", w.lower()) for w in c["words"]]
            keep = [i for i, w in enumerate(words) if w and set(w) <= vocab]
            res = []
            if keep:
                y, _ = sf.read(os.path.join(cfg.work_dir, "corpus", m["speaker"], c["id"] + ".wav"),
                               dtype="float32")
                wav = torch.from_numpy(y)[None]
                with torch.inference_mode():
                    em, _ = model(wav)
                try:
                    spans = ctc_viterbi(torch.log_softmax(em[0], -1).numpy(), tok([words[i] for i in keep]))
                    r = wav.shape[1] / em.shape[1] / SR
                    res = [[c["words"][i], a * r, b * r] for i, (a, b) in zip(keep, spans)]
                except ValueError as e:  # more tokens than frames
                    log(f"ctc failed on {c['id']}: {e}")
            json.dump(res, open(os.path.join(d, c["id"] + ".json"), "w"))
        log(f"ctc {name}")


# ------------------------------------------------------------------ 5. stitch
def snap(words, phones, sp, snap_max):
    """Trim silent word edges (<= snap_max) toward speech, unless the edge phone is a stop."""
    new_w, new_p = [], list(phones)
    for a, b, t in words:
        ps = [i for i, (pa, pb, _) in enumerate(new_p) if pa >= a - 1e-4 and pb <= b + 1e-4]
        na, nb = a, b
        if ps and new_p[ps[0]][2].rstrip("012") not in STOP_PHONES:
            lim = min(a + snap_max, new_p[ps[0]][1] - 0.03)
            while na < lim and frac_speech(sp, na, na + HOP) == 0:
                na += HOP
        if ps and new_p[ps[-1]][2].rstrip("012") not in STOP_PHONES:
            lim = max(b - snap_max, new_p[ps[-1]][0] + 0.03)
            while nb > lim and frac_speech(sp, nb - HOP, nb) == 0:
                nb -= HOP
        if ps:
            new_p[ps[0]] = (na,) + new_p[ps[0]][1:]
            new_p[ps[-1]] = new_p[ps[-1]][:1] + (nb, new_p[ps[-1]][2])
        new_w.append((na, nb, t))
    return new_w, new_p


def stitch(cfg, files):
    """Rebuild full-length TextGrids from the chunk alignments; write the QA report."""
    meta = json.load(open(os.path.join(cfg.work_dir, "meta.json")))
    os.makedirs(cfg.out_dir, exist_ok=True)
    rows = []
    for spk, name, wav, tg in files:
        m = meta[name]
        sp = audio.speech_mask(audio.load(wav), cfg.vad_threshold)
        words, phones, failed, disagree, check, unchecked = [], [], [], [], [], []
        for c in m["chunks"]:
            p = _aligned_path(cfg, spk, c["id"])
            if not p:  # MFA could not align: spread words evenly, no phones, flag it
                failed.append(f"{c['lo']:.2f}-{c['hi']:.2f}")
                d = (c["hi"] - c["lo"]) / len(c["words"])
                words += [(c["lo"] + k * d, c["lo"] + (k + 1) * d, w) for k, w in enumerate(c["words"])]
                continue
            T = read_textgrid(p)
            cw = [(a + c["lo"], b + c["lo"], t) for a, b, t in T["words"] if t]
            words += cw
            phones += [(a + c["lo"], b + c["lo"], t) for a, b, t in T["phones"] if t and t not in ("sil", "sp")]
            cp = os.path.join(cfg.work_dir, "ctc", c["id"] + ".json")
            if os.path.exists(cp):
                ref = json.load(open(cp))
                if ref and len(ref) == len(cw):
                    diffs = [abs(a - (c["lo"] + cs)) for (a, _, _), (_, cs, _) in zip(cw, ref)]
                    if float(np.median(diffs)) > cfg.check_threshold:
                        check.append(f"{c['lo']:.2f}-{c['hi']:.2f}")
                    disagree += [f"{t}@{a:.2f}(ctc {c['lo'] + cs:.2f})"
                                 for (a, _, t), (_, cs, _), dd in zip(cw, ref, diffs) if dd > 0.1]
                    continue
            unchecked.append(f"{c['lo']:.2f}-{c['hi']:.2f}")  # no usable CTC result to compare
        words.sort()
        phones.sort()
        words, phones = snap(words, phones, sp, cfg.snap_max)
        wt, pt = fill_tier(words, m["dur"]), fill_tier(phones, m["dur"])
        write_textgrid(os.path.join(cfg.out_dir, name + ".TextGrid"), m["dur"], [("words", wt), ("phones", pt)])
        before = {}
        if tg:
            orig = read_textgrid(tg).get(cfg.words_tier, [])
            before = audio.word_metrics(orig, sp)
        after = audio.word_metrics(wt, sp)
        flagged = [f"{t}@{a:.2f}" for a, b, t in wt if t and (frac_speech(sp, a, b) < 0.3 or b - a > 1.0)]
        rows.append({
            "file": name, "speaker": spk, "had_textgrid": bool(tg),
            "script_words": len(script_words(cfg, tg)), "output_words": sum(1 for w in wt if w[2]),
            **{f"before_{k}": v for k, v in before.items()}, **{f"after_{k}": v for k, v in after.items()},
            "check_by_hand": " ".join(check), "flagged_words": " ".join(flagged),
            "mfa_failed_chunks": " ".join(failed), "ctc_unchecked_chunks": " ".join(unchecked),
            "untranscribed_speech": " ".join(f"{a:.2f}-{b:.2f}" for a, b in m["untranscribed"]),
            "n_ctc_disagree": len(disagree), "ctc_disagree": " ".join(disagree),
            "transcript_changes": " | ".join(m["notes"])})
        log(f"wrote {name}: {after}, {len(check)} chunks to check, "
            f"{len(unchecked)}/{len(m['chunks'])} chunks not cross-checked")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    rpt = os.path.join(cfg.out_dir, "refine_report.csv")
    with open(rpt, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    n_check = sum(len(r["check_by_hand"].split()) for r in rows)
    n_flag = sum(len(r["flagged_words"].split()) for r in rows)
    log(f"done: {len(rows)} TextGrids in {cfg.out_dir}; {n_check} chunks to check by hand, "
        f"{n_flag} flagged words. Report: {rpt}")


def run(cfg, steps=STEPS):
    files = discover(cfg)
    log(f"{len(files)} files, speakers: {sorted({f[0] for f in files})}")
    for s in STEPS:
        if s in steps:
            globals()[s](cfg, files)


# ------------------------------------------------------------------ inspect
def inspect(cfg, name, t):
    """Print everything known about the chunk of `name` that contains time t (s)."""
    meta = json.load(open(os.path.join(cfg.work_dir, "meta.json")))
    if name not in meta:
        raise SystemExit(f"{name} not in {cfg.work_dir}/meta.json")
    m = meta[name]
    c = next((c for c in m["chunks"] if c["lo"] <= t <= c["hi"]), None)
    if c is None:
        near = [u for u in m["untranscribed"] if u[0] <= t <= u[1]]
        raise SystemExit(f"{t}s is not in a labelled chunk" + (f" (untranscribed speech {near[0]})" if near else ""))
    lo = c["lo"]
    wav = next(f[2] for f in discover(cfg) if f[1] == name)
    y = audio.load(wav)
    sp = audio.speech_mask(y, cfg.vad_threshold)
    print(f"chunk {c['id']}  {lo:.2f}-{c['hi']:.2f}s")
    print(f"  whisper heard : {' '.join(c['heard'])}")
    print(f"  labels        : {' '.join(c['words'])}")
    p = _aligned_path(cfg, m["speaker"], c["id"])
    if p:
        T = read_textgrid(p)
        print("  MFA words     : " + " ".join(f"{w}[{a + lo:.2f}-{b + lo:.2f}]" for a, b, w in T["words"] if w))
        print("  MFA phones    : " + " ".join(f"{w}[{a + lo:.2f}]" for a, b, w in T["phones"] if w))
    cp = os.path.join(cfg.work_dir, "ctc", c["id"] + ".json")
    if os.path.exists(cp):
        print("  CTC words     : " + " ".join(f"{w}[{a + lo:.2f}-{b + lo:.2f}]" for w, a, b in json.load(open(cp))))
    print("  speech regions: " + " ".join(f"{a:.2f}-{b:.2f}" for a, b in audio.regions(sp)
                                         if b > lo and a < c["hi"]))
    print("  energy profile (50 ms frames: time, dB, zero-crossing rate, spectral centroid Hz):")
    for t0 in np.arange(lo, c["hi"], 0.05):
        x = y[int(t0 * SR):int((t0 + 0.05) * SR)]
        if len(x) < 8:
            break
        db = 10 * np.log10((x.astype(np.float64) ** 2).mean() + 1e-12)
        zcr = (np.abs(np.diff(np.sign(x))) > 0).mean()
        S = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        cen = (S * np.fft.rfftfreq(len(x), 1 / SR)).sum() / (S.sum() + 1e-9)
        print(f"    {t0:7.2f} {db:6.1f} {zcr:5.2f} {cen:6.0f} " + "#" * max(0, int((db + 80) / 2)))
