"""Score hypothesis TextGrids (MFA, mfa-refine, ...) against reference TextGrids with manual boundaries.

Phones are compared as MFA english_us_arpa labels without stress. The two phone sequences are
aligned by edit distance; every reference phone paired with a hypothesis phone (same label or a
substitution) contributes its onset, and its offset when silence or the file end follows it.
Words are paired only when the text matches.
"""
import csv
import os
import re

import numpy as np
from rapidfuzz.distance import Levenshtein

from .textgrid import read_textgrid

CLASS = {**{p: "stop" for p in "P B T D K G".split()}, "CH": "affricate", "JH": "affricate",
         **{p: "fricative" for p in "F V TH DH S Z SH ZH HH".split()},
         **{p: "nasal" for p in "M N NG".split()}, **{p: "approximant" for p in "L R W Y".split()}}
GROSS = 0.1  # s


def phone_class(p):
    return CLASS.get(p, "vowel" if p[:1] in "AEIOU" else "other")


def _tier(tiers, name):
    if name in tiers:
        return tiers[name]
    return next((v for k, v in tiers.items() if k.lower().endswith(name.lower())), [])


def _phones(ivs):
    out = []
    for a, b, t in ivs:
        t = re.sub(r"\d", "", t.strip()).upper()
        if t and t not in ("SIL", "SP", "<EPS>"):
            out.append((a, b, t))
    return sorted(out)


def _words(ivs):
    return sorted((a, b, t.strip().lower()) for a, b, t in ivs if t.strip() and t not in ("<eps>", "sil", "sp"))


def _pairs(ref, hyp, equal_only=False):
    """Index pairs (i, j) of aligned labels; with equal_only, substitutions are left out."""
    out = []
    for op, i1, i2, j1, j2 in Levenshtein.opcodes([x[2] for x in ref], [x[2] for x in hyp]):
        if op == "equal" or (op == "replace" and not equal_only):
            out += zip(range(i1, i2), range(j1, j2))
    return out


def compare(ref_tiers, hyp_tiers, words_tier="words", phones_tier="phones"):
    """Errors (hyp - ref, s) for one file: phone boundaries with their class, and matched words."""
    rp, hp = _phones(_tier(ref_tiers, phones_tier)), _phones(_tier(hyp_tiers, phones_tier))
    bounds, same = [], 0
    for i, j in _pairs(rp, hp):
        a, b, p = rp[i]
        same += p == hp[j][2]
        prev = "sil" if i == 0 or rp[i - 1][1] < a - 1e-3 else phone_class(rp[i - 1][2])
        bounds.append((hp[j][0] - a, f"{prev}>{phone_class(p)}"))
        if i == len(rp) - 1 or rp[i + 1][0] > b + 1e-3:
            bounds.append((hp[j][1] - b, f"{phone_class(p)}>sil"))
    rw, hw = _words(_tier(ref_tiers, words_tier)), _words(_tier(hyp_tiers, words_tier))
    words = [(rw[i][2], hw[j][0], hw[j][0] - rw[i][0], hw[j][1] - rw[i][1]) for i, j in _pairs(rw, hw, True)]
    return {"bounds": bounds, "words": words, "n_ref_phones": len(rp), "n_hyp_phones": len(hp),
            "n_paired_phones": len({i for i, _ in _pairs(rp, hp)}), "n_same_phones": same,
            "n_ref_words": len(rw), "n_hyp_words": len(hw)}


def summarize(errs):
    """Boundary error statistics (ms / %) for a list of signed errors in seconds."""
    e = np.abs(np.asarray(errs, float)) * 1000
    if not len(e):
        return {"n": 0}
    return {"n": len(e), "median_ms": round(float(np.median(e)), 1), "mean_ms": round(float(e.mean()), 1),
            "bias_ms": round(float(np.mean(errs) * 1000), 1),
            **{f"within_{t}ms": round(float((e <= t).mean() * 100), 1) for t in (10, 20, 30, 50)},
            "gross_pct": round(float((e > GROSS * 1000).mean() * 100), 1)}


def _textgrids(d):
    out = {}
    for dirpath, _, files in os.walk(d):
        for f in files:
            if f.endswith(".TextGrid"):
                out.setdefault(f[:-9], os.path.join(dirpath, f))
    return out


def _spans(s):
    return [tuple(map(float, x.split("-"))) for x in s.split() if re.fullmatch(r"[\d.]+-[\d.]+", x)]


def detection(per_file, report_csv):
    """How well refine_report.csv flags (check_by_hand, mfa_failed_chunks, flagged_words) catch words
    whose onset or offset is off by more than GROSS."""
    flags = {}
    for r in csv.DictReader(open(report_csv, encoding="utf-8")):
        spans = _spans(r.get("check_by_hand", "")) + _spans(r.get("mfa_failed_chunks", ""))
        fw = [(w.rsplit("@", 1)[0].lower(), float(w.rsplit("@", 1)[1]))
              for w in r.get("flagged_words", "").split() if "@" in w]
        flags[r["file"]] = (spans, fw)
    tp = fp = fn = 0
    for name, res in per_file.items():
        spans, fw = flags.get(name, ([], []))
        for w, on, e_on, e_off in res["words"]:
            flagged = any(a <= on <= b for a, b in spans) or any(w == x and abs(on - t) < 0.015 for x, t in fw)
            bad = max(abs(e_on), abs(e_off)) > GROSS
            tp, fp, fn = tp + (bad and flagged), fp + (flagged and not bad), fn + (bad and not flagged)
    return {"gross_word_errors": tp + fn, "flagged": tp + fp, "caught": tp,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None}


def evaluate(ref_dir, hyp_dir, words_tier="words", phones_tier="phones", refine_report=None):
    refs, hyps = _textgrids(ref_dir), _textgrids(hyp_dir)
    per_file, rows = {}, []
    for name in sorted(refs):
        if name not in hyps:
            continue
        r = per_file[name] = compare(read_textgrid(refs[name]), read_textgrid(hyps[name]), words_tier, phones_tier)
        s = summarize([e for e, _ in r["bounds"]])
        rows.append({"file": name, **{k: v for k, v in r.items() if k.startswith("n_")},
                     **{f"phone_{k}": v for k, v in s.items() if k != "n"},
                     "word_onset_median_ms": summarize([w[2] for w in r["words"]]).get("median_ms")})
    bounds = [b for r in per_file.values() for b in r["bounds"]]
    words = [w for r in per_file.values() for w in r["words"]]
    tot = lambda k: sum(r[k] for r in per_file.values())
    res = {"files": rows, "missing": sorted(set(refs) - set(hyps)),
           "phones": {**summarize([e for e, _ in bounds]),
                      "paired_pct": round(100 * tot("n_paired_phones") / max(tot("n_ref_phones"), 1), 1),
                      "same_label_pct": round(100 * tot("n_same_phones") / max(tot("n_paired_phones"), 1), 1)},
           "word_onset": summarize([w[2] for w in words]), "word_offset": summarize([w[3] for w in words]),
           "words_matched_pct": round(100 * len(words) / max(tot("n_ref_words"), 1), 1),
           "by_class": {c: summarize([e for e, k in bounds if k == c]) for c in sorted({k for _, k in bounds})}}
    if refine_report:
        res["detection"] = detection(per_file, refine_report)
    return res


def format_result(res):
    cols = ["n", "median_ms", "mean_ms", "bias_ms", "within_10ms", "within_20ms", "within_30ms",
            "within_50ms", "gross_pct"]
    line = lambda label, s: f"{label:<24}" + "".join(f"{s.get(c, ''):>12}" for c in cols)
    p = res["phones"]
    out = [f"{len(res['files'])} files scored, {len(res['missing'])} missing from hyp",
           f"phones paired {p['paired_pct']}% of reference, same label {p['same_label_pct']}% of paired; "
           f"words matched {res['words_matched_pct']}%", "",
           f"{'':<24}" + "".join(f"{c:>12}" for c in cols),
           line("phone boundaries", p), line("word onsets", res["word_onset"]),
           line("word offsets", res["word_offset"]), "", "by boundary class (previous>current):"]
    for c, s in sorted(res["by_class"].items(), key=lambda kv: -kv[1]["n"]):
        if s["n"] >= 20:
            out.append(line("  " + c, s))
    if "detection" in res:
        out += ["", f"report flags vs gross word errors (>{GROSS * 1000:.0f} ms): {res['detection']}"]
    return "\n".join(out)


def write_rows(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
