import numpy as np
import soundfile as sf

from mfa_refine import timit
from mfa_refine.evaluate import compare, evaluate, summarize
from mfa_refine.textgrid import fill_tier, read_textgrid, write_textgrid

SR = 16000


# ---------------------------------------------------------------- TIMIT phone folding
def test_fold_merges_closures_and_drops_silence():
    phns = [(0.0, 0.1, "h#"), (0.1, 0.15, "bcl"), (0.15, 0.17, "b"), (0.17, 0.3, "ae"),
            (0.3, 0.35, "tcl"), (0.35, 0.4, "ch"), (0.4, 0.45, "q"), (0.45, 0.5, "ix"),
            (0.5, 0.55, "kcl"), (0.55, 0.6, "pau"), (0.6, 0.7, "dx"), (0.7, 0.8, "h#")]
    assert timit.fold_phones(phns) == [(0.1, 0.17, "B"), (0.17, 0.3, "AE"), (0.3, 0.4, "CH"),
                                       (0.4, 0.5, "IH"),   # q absorbed into the next phone
                                       (0.5, 0.55, "K"),   # unreleased closure
                                       (0.6, 0.7, "T")]


def _fake_timit(root):
    """Two utterances of one core-test speaker, plus an SA sentence that must be skipped."""
    d = root / "TEST" / "DR1" / "FELC0"
    d.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for name in ("SI1", "SX2", "SA1"):
        sf.write(d / f"{name}.WAV.wav", (rng.standard_normal(SR) * 0.1).astype(np.float32), SR)
        (d / f"{name}.PHN").write_text("0 1600 h#\n1600 3200 k\n3200 6400 ae\n6400 8000 t\n"
                                       "8000 9600 s\n9600 12800 ae\n12800 14400 t\n14400 16000 h#\n")
        (d / f"{name}.WRD").write_text("1600 8000 cat\n8000 14400 sat\n")


def test_build_writes_exact_reference(tmp_path):
    _fake_timit(tmp_path / "timit")
    out = tmp_path / "bench"
    timit.build(str(tmp_path / "timit"), str(out), "core", seed=0, p_disfl=1.0)
    utt = read_textgrid(out / "utt" / "ref" / "FELC0_SI1.TextGrid")
    assert [w for w in utt["words"] if w[2]] == [(0.1, 0.5, "cat"), (0.5, 0.9, "sat")]
    assert not (out / "utt" / "ref" / "FELC0_SA1.TextGrid").exists()
    assert (out / "long" / "audio" / "FELC0_long.lab").read_text().split() == ["cat", "sat"] * 2
    # the second utterance's labels are shifted by exactly where its audio was placed
    y, _ = sf.read(out / "long" / "audio" / "FELC0_long.wav", dtype="float32")
    lw = [w for w in read_textgrid(out / "long" / "ref" / "FELC0_long.TextGrid")["words"] if w[2]]
    src, _ = sf.read(tmp_path / "timit" / "TEST" / "DR1" / "FELC0" / "SX2.WAV.wav", dtype="float32")
    i = int(round(lw[2][0] * SR))
    assert np.allclose(y[i:i + 100], src[1600:1700], atol=1e-4)
    # disfluent version: every utterance got a repeat or a deletion, transcript unchanged
    assert (out / "long_disfl" / "audio" / "FELC0_long.lab").read_text().split() == ["cat", "sat"] * 2
    assert len((out / "long_disfl" / "disfluencies.csv").read_text().splitlines()) == 3


def test_assemble_repeat_keeps_labels():
    u = timit.Utterance("S", "X", np.arange(SR, dtype=np.float32), [(0.1, 0.5, "cat"), (0.5, 0.9, "sat")],
                        [(0.1, 0.5, "K"), (0.5, 0.9, "S")], 0.0)
    y, words, phones = timit.assemble([(u, 0.0, 0.5), (u, 0.1, 0.5), 0.25, (u, 0.1, 1.0)],
                                      np.random.default_rng(0), 0.0)
    assert [w for *_, w in words] == ["cat", "cat", "cat", "sat"]
    assert np.allclose([a for a, *_ in words], [0.1, 0.5, 1.15, 1.55])
    assert y[int(round(1.15 * SR))] == u.y[1600]


# ---------------------------------------------------------------- evaluation
def _tiers(words, phones, dur=2.0):
    return {"words": fill_tier(words, dur), "phones": fill_tier(phones, dur)}


REF = _tiers([(0.1, 0.5, "cat")], [(0.1, 0.2, "K"), (0.2, 0.4, "AE"), (0.4, 0.5, "T")])


def test_compare_identity_and_shift():
    r = compare(REF, REF)
    assert summarize([e for e, _ in r["bounds"]])["median_ms"] == 0
    assert len(r["bounds"]) == 4  # three onsets + offset before silence
    shifted = _tiers([(0.115, 0.515, "cat")], [(0.115, 0.215, "K1"), (0.215, 0.415, "AE1"), (0.415, 0.515, "T")])
    s = summarize([e for e, _ in compare(REF, shifted)["bounds"]])
    assert s["within_10ms"] == 0 and s["within_20ms"] == 100 and s["bias_ms"] == 15


def test_compare_pairs_around_insertion_and_substitution():
    hyp = _tiers([(0.1, 0.5, "cat")],
                 [(0.1, 0.2, "K"), (0.2, 0.3, "AH"), (0.3, 0.42, "AE"), (0.42, 0.5, "D")])
    r = compare(REF, hyp)
    assert r["n_paired_phones"] == 3 and r["n_same_phones"] == 2
    assert dict((k, round(e, 3)) for e, k in r["bounds"]) == {
        "sil>stop": 0.0, "stop>vowel": 0.1, "vowel>stop": 0.02, "stop>sil": 0.0}


def test_evaluate_dirs_and_detection(tmp_path):
    (tmp_path / "ref").mkdir()
    (tmp_path / "hyp" / "spk").mkdir(parents=True)
    write_textgrid(tmp_path / "ref" / "f.TextGrid", 2.0, list(REF.items()))
    write_textgrid(tmp_path / "ref" / "g.TextGrid", 2.0, list(REF.items()))
    late = _tiers([(0.4, 0.8, "cat")], [(0.4, 0.5, "K"), (0.5, 0.7, "AE"), (0.7, 0.8, "T")])
    write_textgrid(tmp_path / "hyp" / "spk" / "f.TextGrid", 2.0, list(late.items()))
    (tmp_path / "rpt.csv").write_text("file,check_by_hand,mfa_failed_chunks,flagged_words\nf,0.30-0.90,,\n")
    res = evaluate(tmp_path / "ref", tmp_path / "hyp", refine_report=tmp_path / "rpt.csv")
    assert res["missing"] == ["g"] and res["phones"]["gross_pct"] == 100
    assert res["detection"]["caught"] == 1 and res["detection"]["recall"] == 1.0
