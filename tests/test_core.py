import numpy as np

from mfa_refine import audio
from mfa_refine.reconcile import add_missing, reconcile
from mfa_refine.text import is_hallucination, norm_token
from mfa_refine.textgrid import fill_tier, read_textgrid, write_textgrid

PRON = {"two": ["T", "UW"], "tube": ["T", "UW", "B"], "team": ["T", "IY", "M"]}
pron = lambda w: PRON.get(w.lower(), list(w))


# ---------------------------------------------------------------- textgrid
def test_textgrid_roundtrip(tmp_path):
    words = fill_tier([(0.5, 0.9, 'say "hi"'), (1.2, 1.5, "bat")], 2.0)
    p = tmp_path / "x.TextGrid"
    write_textgrid(p, 2.0, [("words", words), ("phones", fill_tier([(0.5, 0.6, "S")], 2.0))])
    t = read_textgrid(p)
    assert [w for w in t["words"] if w[2]] == [(0.5, 0.9, 'say "hi"'), (1.2, 1.5, "bat")]
    assert t["words"][0] == (0.0, 0.5, "") and t["words"][-1][1] == 2.0
    assert list(t) == ["words", "phones"]


def test_textgrid_short_format_utf16(tmp_path):
    short = '\n'.join(['File type = "ooTextFile"', 'Object class = "TextGrid"', '', '0', '1.5', '<exists>', '2',
                       '"IntervalTier"', '"words"', '0', '1.5', '2', '0', '0.7', '"hello"', '0.7', '1.5', '""',
                       '"TextTier"', '"points"', '0', '1.5', '1', '0.3', '"x"'])
    p = tmp_path / "s.TextGrid"
    p.write_bytes(short.encode("utf-16"))
    t = read_textgrid(p)
    assert t == {"words": [(0.0, 0.7, "hello"), (0.7, 1.5, "")]}


def test_fill_tier_clips_overlaps():
    assert fill_tier([(0.0, 1.0, "a"), (0.8, 1.5, "b")], 2.0) == [(0.0, 1.0, "a"), (1.0, 1.5, "b"), (1.5, 2.0, "")]


# ---------------------------------------------------------------- text
def test_norm_token():
    assert norm_token(" Hello,") == ["hello"]
    assert norm_token("8") == ["eight"]
    assert norm_token("42") == ["forty", "two"]
    assert norm_token("can't!") == ["can't"]
    assert norm_token("--") == []


def test_hallucination_guard():
    assert is_hallucination("Thank you.", 1.0)
    assert is_hallucination("i got a bug again i got a bug again i got a bug again", 0.4)  # 12 words in 0.4 s
    assert not is_hallucination("buy tommy a dummy buy tommy a puppy", 1.4)  # fast but real
    assert not is_hallucination("bird", 0.9)
    # real words followed by an invented outro: keep the chunk (the phrase is stripped later)
    assert not is_hallucination("i got a seat Thank you so much for watching. We'll see you next week.", 1.18)


# ---------------------------------------------------------------- reconcile
def test_script_wins_and_repeats_and_asides():
    script = "i got a team i got a tube i got a top".split()
    heard = [["i", "can't", "wait"], ["i", "got", "a", "t", "bar"], ["i", "got", "two"],
             ["i", "got", "a", "tube"], ["i", "got", "a", "top"]]
    log = []
    labels = reconcile(heard, [1.5] * 5, script, log, pron=pron)
    assert labels == [["i", "can't", "wait"], "i got a team".split(), "i got a tube".split(),
                      "i got a tube".split(), "i got a top".split()]
    assert any(l.startswith("repeat:'i got two'->'i got a tube'") for l in log)  # pronunciation tie-break
    assert "extra_speech:'i can't wait'" in log


def test_unspoken_script_word_is_dropped():
    log = []
    labels = reconcile([["pig"], ["door"]], [1, 1], ["pig", "bird", "door"], log)
    assert labels == [["pig"], ["door"]] and "unspoken:bird" in log


def test_sparse_extra_chunk_left_unlabelled():
    log = []
    labels = reconcile([["oh"], ["pig"]], [4.6, 1.0], ["pig"], log)
    assert labels == [[], ["pig"]]


def test_no_script_uses_whisper():
    assert reconcile([["pa", "ka"], ["oh"]], [1.0, 5.0], [], []) == [["pa", "ka"], []]


def test_add_missing_runs_and_edges():
    log = []
    script = "her pinky toe".split()
    heard = "monkey is holding a toothbrush on her pinky toe".split()
    assert add_missing(script, heard, log) == heard  # 6-word run at the start
    assert add_missing("i got a".split(), "i got a paw".split(), []) == "i got a paw".split()  # edge
    assert add_missing("i got a top".split(), "i got uh a top".split(), []) == "i got a top".split()  # short, inside


# ---------------------------------------------------------------- audio
def test_chunking_splits_at_pauses():
    sr = audio.SR
    t = np.arange(int(3.0 * sr)) / sr
    y = np.zeros_like(t, dtype=np.float32) + 1e-5 * np.random.default_rng(0).standard_normal(len(t)).astype(np.float32)
    for a, b in [(0.5, 0.9), (1.0, 1.2), (2.0, 2.5)]:  # 0.1 s gap merges, 0.8 s gap splits
        y[int(a * sr):int(b * sr)] += 0.3 * np.sin(2 * np.pi * 200 * t[int(a * sr):int(b * sr)])
    sp = audio.speech_mask(y)
    chunks = audio.make_chunks(sp, 3.0)
    assert len(chunks) == 2
    assert chunks[0][0] < 0.5 < 1.2 < chunks[0][1] < chunks[1][0] < 2.0 < 2.5 < chunks[1][1]


def test_whisper_device_auto(monkeypatch):
    import ctranslate2
    import faster_whisper
    from mfa_refine.pipeline import Config, _load_whisper

    calls = []

    def fake_model(name, device, compute_type):
        calls.append((device, compute_type))
        if device == "cuda" and fail:
            raise RuntimeError("cuda broken")
        return "model"

    monkeypatch.setattr(faster_whisper, "WhisperModel", fake_model)
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 0)
    monkeypatch.setattr(ctranslate2, "get_mps_device_count", lambda: 1, raising=False)
    cfg = Config(audio_dirs=[], out_dir="x")
    fail = False
    _load_whisper(cfg)
    assert calls == [("cpu", "int8")]  # no CUDA: CPU, even with an Apple GPU present

    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 1)
    calls.clear()
    _load_whisper(cfg)
    assert calls == [("cuda", "float16")]

    fail = True
    calls.clear()
    assert _load_whisper(cfg) == "model"
    assert calls == [("cuda", "float16"), ("cpu", "int8")]  # broken GPU falls back to CPU


def test_ctc_viterbi():
    import pytest
    from mfa_refine.pipeline import ctc_viterbi

    path = [0, 1, 0, 1, 0, 2, 2, 3, 0]  # blank=0; word "aa" needs a blank between its repeated tokens
    lp = np.log(np.full((len(path), 4), 0.1 / 3))
    lp[np.arange(len(path)), path] = np.log(0.9)
    assert ctc_viterbi(lp, [[1, 1], [2, 3]]) == [(1, 4), (5, 8)]
    with pytest.raises(ValueError):
        ctc_viterbi(lp[:2], [[1, 1]])  # two frames cannot hold "a, blank, a"

    # long chunk: more than 127 states once blanks are interleaved
    words = [[1, 2, 3]] * 50
    path = [0] + [t for w in words for t in w] + [0]
    lp = np.log(np.full((len(path), 4), 0.1 / 3))
    lp[np.arange(len(path)), path] = np.log(0.9)
    assert ctc_viterbi(lp, words) == [(1 + 3 * i, 4 + 3 * i) for i in range(50)]


def test_hallucination_phrase_fragments():
    from mfa_refine.text import ELONGATED, HALLUC_PHRASE

    def strip(s):
        return " ".join(HALLUC_PHRASE.sub(" ", s).split())

    # phrases benchmarked from small.en on real recordings
    assert strip("i got a pot again thank you very much") == "i got a pot again"
    assert strip("i got a plug again thank you much") == "i got a plug again"
    assert strip("i got a top thank you watching") == "i got a top"
    assert strip("buy tommy a knotty thank you so much for joining us today") == "buy tommy a knotty"
    assert strip("i got a seat thank you so much for watching we'll see you next week") == "i got a seat"
    assert strip("buy bobby a puppy thank you for being with us today") == "buy bobby a puppy"
    assert strip("buy tommy a dummy") == "buy tommy a dummy"
    assert ELONGATED.search("meowwwww") and ELONGATED.search("helllll")
    assert not ELONGATED.search("balloon") and not ELONGATED.search("bookkeeper")
