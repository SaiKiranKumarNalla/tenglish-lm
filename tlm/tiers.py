"""Alternative synthetic romanization strategies, compared in Stage 1b.

TIERED (in the style of a design proposed by Gemini): every SENTENCE is written in one fixed style tier.
  P1 standard  canonical spelling, long vowels and aspirates kept   ->  "intha late enduku, nuvvu cheppu"
  P2 minimal   long vowels shortened, aspirates dropped (th->t)    ->  "inta late enduku, nuvvu cheppu"
  P4 lazy      P2 + doubled consonants dropped + e->a at word start ->  "inta late anduku, nuvu chepu"
               (the original proposal's 'acrophon' tier also produced forms like "Antra" for ఏంట్రా)

PER-WORD (tlm/romanizer.py, this project's default): every WORD gets its own randomly sampled spelling.

HYBRID: pick a tier per sentence (a person's habit), then give each word a 30% chance of being
        re-sampled per-word (people aren't perfectly consistent within a sentence).
"""
import random
import re

from tlm.romanizer import TELUGU_WORD, romanize, romanize_word

_P2 = [("chh", "ch"), ("aa", "a"), ("ee", "i"), ("oo", "u"), ("th", "t"), ("dh", "d"),
       ("kh", "k"), ("gh", "g"), ("bh", "b"), ("ph", "f")]


def _p2(w):
    for a, b in _P2:
        w = w.replace(a, b)
    return w


def _p4(w):
    w = re.sub(r"(.)\1", r"\1", _p2(w))
    return "a" + w[1:] if w.startswith("e") and len(w) > 2 else w


TIERS = {"P1": lambda w: w, "P2": _p2, "P4": _p4}


def tier_word(te_word: str, tier: str) -> str:
    return TIERS[tier](romanize_word(te_word, canonical=True))


def romanize_tiered(text: str, rng: random.Random, tier: str | None = None) -> str:
    t = tier or rng.choice(list(TIERS))
    return TELUGU_WORD.sub(lambda m: tier_word(m.group(), t), text)


def romanize_hybrid(text: str, rng: random.Random, p_word: float = 0.3) -> str:
    t = rng.choice(list(TIERS))
    style = rng.choice([0.0, 0.0, 0.5, 1.5])
    return TELUGU_WORD.sub(lambda m: romanize_word(m.group(), rng, False, style) if rng.random() < p_word
                           else tier_word(m.group(), t), text)


GENERATORS = {
    "tiered (per sentence)": romanize_tiered,
    "per-word (ours)": lambda text, rng: romanize(text, rng),
    "hybrid": romanize_hybrid,
}
