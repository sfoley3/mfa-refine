"""Minimal Praat TextGrid IO (interval tiers). Reads long and short text formats, UTF-8 or UTF-16."""
import re

Interval = tuple  # (xmin, xmax, text)


def _decode(raw: bytes) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def read_textgrid(path):
    """Return {tier_name: [(xmin, xmax, text), ...]} for interval tiers (point tiers are skipped)."""
    s = _decode(open(path, "rb").read())
    s = re.sub(r"\[\d+\]", "", s)  # drop 'item [1]:' / 'intervals [3]:' indices
    toks = []
    for q, num in re.findall(r'"((?:[^"]|"")*)"|(?<![\w<])(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![\w>])', s):
        toks.append(q.replace('""', '"') if num == "" else float(num))
    # toks: "ooTextFile", "TextGrid", xmin, xmax, ntiers, then per tier
    i = 5
    tiers = {}
    ntiers = int(toks[4])
    for _ in range(ntiers):
        kind, name, n = toks[i], toks[i + 1], int(toks[i + 4])
        i += 5
        if kind == "IntervalTier":
            tiers[name] = [(toks[i + 3 * k], toks[i + 3 * k + 1], toks[i + 3 * k + 2]) for k in range(n)]
            i += 3 * n
        else:  # TextTier: (time, mark) pairs
            i += 2 * n
    return tiers


def fill_tier(labelled, dur):
    """Sorted, non-overlapping labelled intervals -> contiguous tier covering 0..dur."""
    out, t = [], 0.0
    for a, b, txt in labelled:
        a, b = max(a, t), min(b, dur)
        if b - a < 1e-4:
            continue
        if a - t > 1e-4:
            out.append((t, a, ""))
        out.append((a, b, txt))
        t = b
    if dur - t > 1e-4:
        out.append((t, dur, ""))
    return out


def write_textgrid(path, dur, tiers):
    """tiers: [(name, contiguous intervals)] -> long-format TextGrid (what MFA writes)."""
    q = lambda t: t.replace('"', '""')
    L = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
         "xmin = 0 ", f"xmax = {dur} ", "tiers? <exists> ", f"size = {len(tiers)} ", "item []: "]
    for i, (name, ivs) in enumerate(tiers, 1):
        L += [f"    item [{i}]:", '        class = "IntervalTier" ', f'        name = "{q(name)}" ',
              "        xmin = 0 ", f"        xmax = {dur} ", f"        intervals: size = {len(ivs)} "]
        for j, (a, b, t) in enumerate(ivs, 1):
            L += [f"        intervals [{j}]:", f"            xmin = {round(a, 4)} ",
                  f"            xmax = {round(b, 4)} ", f'            text = "{q(t)}" ']
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
