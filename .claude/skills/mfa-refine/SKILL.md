---
name: mfa-refine
description: Refine Montreal Forced Aligner (MFA) TextGrids that are misaligned (words on silence, drifting boundaries, missing words) with the mfa-refine CLI, then review the QA report and flagged chunks with the user. Use when the user asks to fix, refine, realign or QA forced alignments or TextGrids.
---

# Refining MFA alignments

`mfa-refine` re-transcribes audio in pause-bounded chunks with Whisper, matches that against the
user's transcript, re-aligns each chunk with MFA, cross-checks it with a CTC aligner and writes
refined TextGrids plus `refine_report.csv`. The README in this repo documents every option and
report column. Read it before the first run.

## 1. Set up

- The conda env `mfa-refine` comes from `environment.yml`. Check it with
  `conda run -n mfa-refine mfa-refine -h`. If it is missing, create it and download the MFA models
  as the README shows. Don't pip-install torch into it: pip torch with conda MFA crashes with
  `OMP: Error #15`.
- Ask the user, or work out from the files:
  - the audio folders and TextGrid folders (matched by file stem)
  - the word tier name (`--words-tier`, default `words`)
  - how speakers are named in file stems (`--speaker-pattern`)
  - which tasks or files to skip (`--exclude`)
  - the language (English defaults to `english_us_arpa`; otherwise find the matching MFA models)
- Never write into the user's input folders. Use a fresh `--out` folder.

## 2. Run

- First run it on 2–3 representative files (`--include`) and show the user the before/after
  numbers. Then run everything.
- It's long-running on CPU (Whisper is the slow step), so run it in the background and report
  progress from the log lines.
- Cached steps are skipped. After changing VAD or chunking options, re-run from `transcribe`.
  After changing only labels or MFA options, use `--steps prepare,align,ctc,stitch`.

## 3. Review the report

Summarise `refine_report.csv` for the user, file by file:

- Compare `before_words_on_silence` and `before_uncovered_speech_s` with their `after_` values.
  That shows whether it helped.
- `check_by_hand` and `mfa_failed_chunks` are what the user needs to look at in Praat. Give exact
  file names and time ranges.
- Scan `transcript_changes`. Each change is a decision the pipeline made about the labels: Whisper
  heard an extra word, a hallucination was rejected, a transcript word was inserted where Whisper
  missed it. Point out any that look wrong for the task, e.g. a word-list task where Whisper
  "corrected" a misproduction to a real word.
- In `untranscribed_speech`, long spans can be missed words. Short ones are usually coughs or
  breaths.

For each chunk the user wants to look at, run `mfa-refine inspect ... <file> <time>` and explain
what it shows. The energy profile shows where speech really starts and ends.

## 4. Tune if needed

If many transcript words come back `unspoken` (common with disordered or highly repetitive
speech), re-run with `--whisper-model medium.en` (about 1.7× slower on CPU). Whisper output is
cached per model, so switching back to `small.en` doesn't re-transcribe.

If one kind of error dominates, adjust that option (see the README's Tuning table), re-run the
affected steps and compare the report with the previous run. Don't present a threshold change as an
improvement until the report numbers show it is one.
