# mfa-refine

Fix Montreal Forced Aligner (MFA) TextGrids where words land in silence, drift off the speech, or
were never aligned. Runs on CPU; Whisper uses an NVIDIA GPU when one is available.

MFA aligns a whole recording against a transcript in one go. On long, pause-heavy or disordered
speech (word lists, reading passages with restarts, clinical tasks) one error can pull many words
off the speech. `mfa-refine` re-does the alignment in short, silence-bounded pieces and tells you
where it is still unsure:

1. **transcribe** – split each file at pauses (energy VAD) and run Whisper on each chunk, so it
   knows which words were actually said in which chunk.
2. **prepare** – drop Whisper hallucinations, then match what Whisper heard against your
   transcript (the word tier of the existing TextGrid). Your transcript supplies the labels and
   Whisper decides where they go, so a small Whisper model is enough.
3. **align** – run `mfa align` on the short chunks (words + phones), with speaker adaptation and
   G2P for out-of-vocabulary words.
4. **ctc** – align the same chunks with an independent CTC aligner (torchaudio MMS_FA) as a
   cross-check.
5. **stitch** – rebuild full-length TextGrids, trim silent word edges (never across stop
   closures), and write `refine_report.csv`.

Every step caches its output in `OUT/_work`, so re-running only redoes what changed.

## Install

```bash
git clone https://github.com/sfoley3/mfa-refine.git
cd mfa-refine
conda env create -f environment.yml
conda activate mfa-refine
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p english_us_arpa
```

PyTorch must come from conda-forge, as `environment.yml` does. pip torch alongside conda MFA
crashes on macOS with `OMP: Error #15`.

## Run

```bash
mfa-refine run \
  --audio data/spk1/wav --textgrids data/spk1/textgrids \
  --audio data/spk2/wav --textgrids data/spk2/textgrids \
  --out refined/ \
  --speaker-pattern '^(spk\d+)_'
```

- Audio files and TextGrids are matched by file stem. A file without a TextGrid is labelled with
  Whisper's words.
- `--speaker-pattern` is a regex on the file stem whose group 1 is the speaker. MFA adapts to each
  speaker.
- `--include` / `--exclude` take globs on the file stem (repeatable), e.g. `--exclude '*_bvt*'`.
- `--steps align,ctc,stitch` re-runs part of the pipeline.
- `--whisper-model` defaults to `small.en`. Your transcript supplies the labels, so Whisper only
  has to place them, and on read word lists `small.en` matched `large-v3`. On disordered or highly
  repetitive speech `medium.en` recovered more of the transcript, at about 1.7× the CPU time and
  ~4.5 GB RAM (vs ~3 GB). Try it when the report shows many `unspoken` words.
- `--device auto` (default) runs Whisper on an NVIDIA GPU (CUDA) if there is one, else on CPU, and
  falls back to CPU if the GPU fails to load. `--device mps` tries the Apple Silicon GPU
  (ctranslate2's MPS backend is experimental; check the output).
  MFA itself always runs on CPU.
  CUDA needs the NVIDIA cuBLAS/cuDNN libraries that faster-whisper requires (see its README).
- For other languages, set `--language`, `--dictionary`, `--acoustic-model` and `--g2p-model` to
  MFA models for that language.

Run `mfa-refine run -h` to see all options.

## Read the report

`OUT/refine_report.csv` has one row per file:

| column | meaning |
|---|---|
| `before_*` / `after_*` | `words_on_silence`, `words_over_1s` and `uncovered_speech_s` for the input and output word tiers |
| `check_by_hand` | chunks (start-end s) where MFA and the CTC aligner disagree by more than `--check-threshold` (median word onset) |
| `flagged_words` | output words that still sit mostly on silence or last over 1 s |
| `ctc_unchecked_chunks` | chunks the CTC cross-check couldn't cover, e.g. words with digits or symbols. These aren't in `check_by_hand`, so look at them yourself if it matters |
| `mfa_failed_chunks` | chunks MFA could not align. Words there are spread evenly and have no phones |
| `untranscribed_speech` | speech Whisper produced no usable words for, e.g. coughs, laughs, or words missing from your transcript |
| `ctc_disagree` | individual words whose onsets differ by more than 100 ms between the two aligners |
| `transcript_changes` | every decision made about the labels, as tagged notes (below) |

`transcript_changes` tags:

- `rejected_chunk`, `drop_low_conf`, `strip_phrase`: Whisper output thrown away as hallucination
- `heard:'x'->'y'`: Whisper heard `x`, labelled with the transcript's `y`
- `repeat`: a repeated stretch, labelled with the transcript words it repeats
- `extra_speech`, `added`: words Whisper heard that aren't in the transcript, kept as heard
- `unlabelled_sparse`: a chunk too sparse to match, left unlabelled
- `unspoken`: a transcript word not found in the audio, left out

## Fix what's flagged

```bash
mfa-refine inspect --audio data/spk1/wav --textgrids data/spk1/textgrids --out refined/ spk1_task1 42.3
```

This prints the chunk around 42.3 s: what Whisper heard, the labels used, the MFA and CTC word
times, the VAD speech regions and a 50 ms energy profile. Then open the refined TextGrid with the
audio in Praat and fix the boundaries by hand.

## Tuning

| option | default | change when |
|---|---|---|
| `--vad-threshold` | 0.35 | quiet speech counts as silence (lower it) or noise counts as speech (raise it) |
| `--min-silence` | 0.30 | chunks are too long (lower it) or words are split mid-utterance (raise it) |
| `--snap-max` | 0.05 | maximum silence (s) trimmed from each word edge |
| `--check-threshold` | 0.15 | too many or too few chunks are listed in `check_by_hand` |

## Using it with Claude Code

`.claude/skills/mfa-refine/SKILL.md` is a Claude Code skill. Open Claude Code in this repo, or copy
the skill folder to `~/.claude/skills/`, and ask it to "refine the MFA alignments in …". Claude
will run the pipeline, read the report and go through the flagged chunks with you.

## Tests

```bash
python -m pytest
```

## Data

`.gitignore` excludes `*.wav`, `*.flac` and `*.TextGrid`. Don't commit participant recordings or
annotations.
