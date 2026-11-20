"""Match what Whisper heard (per speech chunk) against the original transcript."""
import difflib

from rapidfuzz.distance import Levenshtein


def reconcile(chunk_hyps, durs, script, log, pron=lambda w: list(w), max_words=60, c_skip=0.8,
              repeat_ratio=0.5, min_rate=0.5):
    """Assign consecutive script words to speech chunks, in order, by character edit distance.

    chunk_hyps: per chunk, the normalised words Whisper heard. durs: chunk durations (s).
    Each chunk either takes script[j:k] (script spelling wins), or is extra speech (keeps Whisper's
    words); script words may also go unspoken. Returns one label list per chunk ([] = leave unlabelled).
    """
    if not script:
        return [list(h) if len(h) / d >= min_rate else [] for h, d in zip(chunk_hyps, durs)]
    n, m = len(chunk_hyps), len(script)
    hs = ["".join(h) for h in chunk_hyps]
    INF = float("inf")
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            c = cost[i][j]
            if c == INF:
                continue
            if j < m and c + c_skip * len(script[j]) < cost[i][j + 1]:  # script word unspoken
                cost[i][j + 1], back[i][j + 1] = c + c_skip * len(script[j]), ("skip", i, j)
            if i == n:
                continue
            if c + c_skip * len(hs[i]) < cost[i + 1][j]:  # chunk is extra speech
                cost[i + 1][j], back[i + 1][j] = c + c_skip * len(hs[i]), ("extra", i, j)
            for k in range(j + 1, min(m, j + max_words) + 1):
                d = c + Levenshtein.distance(hs[i], "".join(script[j:k]).lower())
                if d < cost[i + 1][k]:
                    cost[i + 1][k], back[i + 1][k] = d, ("take", i, j)

    labels, kinds = [None] * n, [None] * n
    i, j = n, m
    while (i, j) != (0, 0):
        op, pi, pj = back[i][j]
        if op == "skip":
            log.append(f"unspoken:{script[pj]}")
        elif op == "extra":
            labels[pi], kinds[pi] = list(chunk_hyps[pi]), "extra"
        else:
            labels[pi], kinds[pi] = add_missing(list(script[pj:j]), chunk_hyps[pi], log), "take"
            if [w.lower() for w in script[pj:j]] != chunk_hyps[pi]:
                log.append(f"heard:'{' '.join(chunk_hyps[pi])}'->'{' '.join(script[pj:j])}'")
        i, j = pi, pj

    for i in range(n):
        if kinds[i] != "extra":
            continue
        # an extra chunk that closely matches a neighbouring item is a repeat: use the script words.
        # Ties are broken on the pronunciation of the final word ("two" -> "tube" rather than "team").
        best = None
        for nb in (i - 1, i + 1):
            if 0 <= nb < n and kinds[nb] == "take" and labels[nb]:
                ref = "".join(labels[nb]).lower()
                d = Levenshtein.distance(hs[i], ref)
                key = (d, Levenshtein.distance(pron(chunk_hyps[i][-1]), pron(labels[nb][-1])))
                if d <= repeat_ratio * max(len(hs[i]), len(ref)) and (best is None or key < best[0]):
                    best = (key, labels[nb])
        if best:
            log.append(f"repeat:'{' '.join(chunk_hyps[i])}'->'{' '.join(best[1])}'")
            labels[i] = list(best[1])
        elif len(chunk_hyps[i]) / durs[i] < min_rate:  # Whisper can't explain this chunk
            log.append(f"unlabelled_sparse:'{' '.join(chunk_hyps[i])}'")
            labels[i] = []
        else:
            log.append(f"extra_speech:'{' '.join(chunk_hyps[i])}'")
    return labels


def add_missing(script, heard, log, min_run=3):
    """Insert words Whisper heard that the script lacks: runs of >= min_run words, or at chunk edges.
    Shorter insertions inside a chunk are usually mishearings and are ignored."""
    sm = difflib.SequenceMatcher(a=[w.lower() for w in script], b=heard, autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "insert" and (j2 - j1 >= min_run or i1 == 0 or i1 == len(script)):
            out += heard[j1:j2]
            log.append(f"added:'{' '.join(heard[j1:j2])}'")
        else:
            out += script[i1:i2]
    return out
