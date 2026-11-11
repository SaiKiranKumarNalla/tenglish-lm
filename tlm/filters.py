"""Scene-level quality filters and text heuristics used by the data card (Stage 0b) and later stages."""
import re

TELUGU = re.compile(r"[ఀ-౿]")
LATIN = re.compile(r"[A-Za-z]")
TE_WORD = re.compile(r"[ఀ-౿]+")
NOISE_TAG = re.compile(r"[\[\(♪♫]|సంగీతం|పాట")
VIRAMA = "్"


def drop_reason(lines: list[str]) -> str | None:
    """Why a scene should be excluded from training, or None to keep it. Checked in this order."""
    text = " ".join(lines)
    te, la = len(TELUGU.findall(text)), len(LATIN.findall(text))
    if any(NOISE_TAG.search(l) for l in lines):
        return "sound/music tag left"
    if la > te:
        return "mostly non-Telugu"
    if te < 20 or len(lines) < 2:
        return "too short"
    words = text.split()
    if len(words) >= 12 and len(set(words)) / len(words) < 0.45:
        return "repetitive (likely lyrics)"
    return None


def is_loanword(te_word: str) -> bool:
    """Heuristic: native Telugu words almost always end in a vowel. Words ending in a consonant
    with virama (లేట్ late, సార్ sir, ఫోన్ phone, ఆఫీస్ office) are mostly English or Urdu loans.
    Imperfect: colloquial clipped forms (రాసావ్ for రాశావు) also end in virama, so this OVER-counts."""
    return len(te_word) >= 2 and te_word.endswith(VIRAMA)


def has_punctuation(line: str) -> bool:
    return bool(re.search(r"[?!.,]", line))


def multi_sentence(line: str) -> bool:
    """A subtitle 'line' that holds more than one sentence (often two speakers merged)."""
    return len([s for s in re.split(r"[?!.]+", line) if len(TELUGU.findall(s)) >= 3]) >= 2
