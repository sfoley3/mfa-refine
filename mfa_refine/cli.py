"""mfa-refine command line."""
import argparse

from .pipeline import STEPS, Config, inspect, run


def _common(p):
    p.add_argument("--audio", required=True, action="append", help="folder of .wav/.flac files (repeatable)")
    p.add_argument("--out", required=True, help="output folder for refined TextGrids + report")
    p.add_argument("--textgrids", action="append", default=[],
                   help="folder of existing (MFA) TextGrids, matched to audio by file stem; their word "
                        "tier is used as the script (repeatable). Files without one use Whisper's words")
    p.add_argument("--work", help="cache folder (default: OUT/_work)")
    p.add_argument("--words-tier", default="words", help="word tier name in the input TextGrids")
    p.add_argument("--speaker-pattern", help=r"regex on file stem; group 1 = speaker (e.g. '^(spk\d+)_')")
    p.add_argument("--include", action="append", default=[], metavar="GLOB", help="only these file stems")
    p.add_argument("--exclude", action="append", default=[], metavar="GLOB", help="skip these file stems")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mfa-refine", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="refine alignments")
    _common(r)
    r.add_argument("--steps", default=",".join(STEPS),
                   help=f"comma-separated subset of {','.join(STEPS)} (each step is cached)")
    r.add_argument("--language", default="en")
    r.add_argument("--whisper-model", default="small.en", help="faster-whisper model name or path")
    r.add_argument("--device", default="auto",
                   help="Whisper device: auto (default: cuda if available, else cpu), cuda, cpu, or mps "
                        "(Apple Silicon, experimental)")
    r.add_argument("--compute-type", default="auto", help="auto (default: float16 on GPU, int8 on CPU) "
                   "or any ctranslate2 compute type")
    r.add_argument("--dictionary", default="english_us_arpa", help="MFA dictionary name or path")
    r.add_argument("--acoustic-model", default="english_us_arpa", help="MFA acoustic model name or path")
    r.add_argument("--g2p-model", default="english_us_arpa", help="MFA G2P model for OOV words ('' to disable)")
    r.add_argument("--num-jobs", type=int, default=4)
    r.add_argument("--vad-threshold", type=float, default=0.35,
                   help="speech = dB above floor + this fraction of the floor-to-peak range")
    r.add_argument("--min-silence", type=float, default=0.30, help="pause (s) that splits chunks")
    r.add_argument("--snap-max", type=float, default=0.05, help="max silence (s) trimmed from word edges")
    r.add_argument("--check-threshold", type=float, default=0.15,
                   help="flag chunks whose median MFA-vs-CTC onset difference exceeds this (s)")

    i = sub.add_parser("inspect", help="show MFA, CTC, Whisper and energy for the chunk at a time")
    _common(i)
    i.add_argument("name", help="file stem")
    i.add_argument("time", type=float, help="time (s) inside the chunk")

    t = sub.add_parser("timit-prep", help="build benchmark corpora with exact ground truth from TIMIT")
    t.add_argument("--timit", required=True, help="TIMIT root (the folder holding TRAIN/ and TEST/)")
    t.add_argument("--out", required=True, help="output folder (gets utt/, long/, long_disfl/)")
    t.add_argument("--split", default="core", choices=["core", "test", "train"],
                   help="core test set (24 speakers, default), full test set, or train set")
    t.add_argument("--seed", type=int, default=0, help="seed for pauses and spliced disfluencies")

    e = sub.add_parser("eval", help="score TextGrids against reference TextGrids (phone and word boundaries)")
    e.add_argument("--ref", required=True, help="folder of reference TextGrids (searched recursively)")
    e.add_argument("--hyp", required=True, help="folder of TextGrids to score, matched by file stem")
    e.add_argument("--words-tier", default="words")
    e.add_argument("--phones-tier", default="phones")
    e.add_argument("--report", help="write per-file scores to this CSV")
    e.add_argument("--refine-report", help="refine_report.csv for HYP: also score its flags against true errors")

    a = ap.parse_args(argv)
    if a.cmd == "timit-prep":
        from .timit import build
        build(a.timit, a.out, a.split, a.seed)
        return
    if a.cmd == "eval":
        from .evaluate import evaluate, format_result, write_rows
        res = evaluate(a.ref, a.hyp, a.words_tier, a.phones_tier, a.refine_report)
        print(format_result(res))
        if a.report:
            write_rows(a.report, res["files"])
        return
    cfg = Config(audio_dirs=a.audio, out_dir=a.out, textgrid_dirs=a.textgrids, work_dir=a.work,
                 words_tier=a.words_tier, speaker_pattern=a.speaker_pattern,
                 include=a.include, exclude=a.exclude)
    if a.cmd == "inspect":
        inspect(cfg, a.name, a.time)
        return
    for k in ("language", "whisper_model", "device", "compute_type", "dictionary", "acoustic_model",
              "g2p_model", "num_jobs", "vad_threshold", "min_silence", "snap_max", "check_threshold"):
        setattr(cfg, k, getattr(a, k))
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    bad = set(steps) - set(STEPS)
    if bad:
        ap.error(f"unknown steps: {bad}")
    run(cfg, steps)


if __name__ == "__main__":
    main()
