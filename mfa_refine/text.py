"""Word normalisation, Whisper hallucination filters, pronunciation lookup."""
import os
import re

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num_words(n):
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        return [_TENS[n // 10]] + ([_ONES[n % 10]] if n % 10 else [])
    if n < 1000:
        return [_ONES[n // 100], "hundred"] + (num_words(n % 100) if n % 100 else [])
    return [str(n)]


def norm_token(w):
    """Lowercase, strip punctuation (keep apostrophes and letters of any script), spell out numbers."""
    w = re.sub(r"[^\w']", "", w.lower().strip()).replace("_", "").strip("'")
    if w.isdigit():
        return num_words(int(w))
    return [w] if w else []


HALLUC_CHUNK = re.compile(r"^(thank you|thanks for watching|you|bye|bye bye)$")
# YouTube-outro phrases Whisper invents in pauses, including fragments left after low-confidence words
# are dropped ("thank you much", "thank you watching").
HALLUC_PHRASE = re.compile(
    r"\b(?:(?:thank you|thanks)(?: (?:so |very )?much)?(?: (?:for )?(?:watching|listening|(?:joining|being with) us(?: today)?))?"
    r"|(?:we'll |i'll )?see you (?:next (?:week|time)|soon)|please subscribe)\b")
ELONGATED = re.compile(r"(.)\1{3,}")  # "meowwwww", "helllll": not words


def is_hallucination(text, chunk_dur, max_rate=8.0):
    """Whisper guard: impossible word rate, a stock phrase on its own, or nothing at all.
    Outro phrases are removed before the rate check, so real words before them survive."""
    norm = " ".join(t for w in text.split() for t in norm_token(w))
    words = HALLUC_PHRASE.sub(" ", norm).split()
    return not words or len(words) / max(chunk_dur, 1e-3) > max_rate or bool(HALLUC_CHUNK.match(norm))


class Pronouncer:
    """First pronunciation per word from an MFA dictionary (stress digits stripped); spelling if absent."""

    def __init__(self, dictionary):
        self.table = {}
        path = resolve_dictionary(dictionary)
        if path:
            for line in open(path, encoding="utf-8"):
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0] not in self.table:
                    self.table[parts[0]] = [re.sub(r"\d", "", p) for p in parts[-1].split()]

    def __call__(self, word):
        return self.table.get(word.lower(), list(word.lower()))


def resolve_dictionary(dictionary):
    """A path, or the name of a downloaded MFA dictionary model."""
    if os.path.isfile(dictionary):
        return dictionary
    root = os.environ.get("MFA_ROOT_DIR", os.path.expanduser("~/Documents/MFA"))
    p = os.path.join(root, "pretrained_models", "dictionary", f"{dictionary}.dict")
    return p if os.path.isfile(p) else None
