"""Two pre-tokenization transforms for romanized Telugu, tested in Stage 2.

phonetic_key  collapses spelling choices that don't change meaning, so different spellings of
              a word become identical BEFORE tokenization:  yenti / enti -> enti,  intha / inta -> inta,
              nuvvu / nuwwu / nuvu -> nuvu.  Cost: some genuinely different words can collide.

morph_split   peels one common Telugu suffix (case marker / postposition) off a word, so the vocabulary
              is spent on stems instead of every inflected form:  intlo -> int ##lo,  vaadiki -> vaadi ##ki.
              Telugu is agglutinative, so the same stem shows up with many endings.
"""
import re

_DIGRAPHS = [("chh", "ch"), ("th", "t"), ("dh", "d"), ("kh", "k"), ("gh", "g"), ("bh", "b"),
             ("jh", "j"), ("ph", "f"), ("ee", "i"), ("oo", "u"), ("aa", "a"), ("ow", "au"), ("w", "v")]
_WORD = re.compile(r"[a-z]+")


def _key(w: str) -> str:
    if w.startswith("ye"):
        w = w[1:]  # yenti -> enti, yela -> ela
    for a, b in _DIGRAPHS:
        w = w.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", w)  # doubled letters -> single: nuvvu -> nuvu


def phonetic_key(text: str) -> str:
    return _WORD.sub(lambda m: _key(m.group()), text.lower())


# Longest first, so "kunna" wins over "nu". Only split when a stem of >= 3 letters remains.
SUFFIXES = sorted(["lo", "ki", "ku", "ni", "tho", "to", "ga", "kunna", "kuni", "toti", "tho", "loni", "laga",
                   "gaaru", "garu", "vallu", "valla", "kosam", "nunchi", "nundi", "meeda", "midha", "daka"],
                  key=len, reverse=True)


def _split(w: str) -> str:
    for s in SUFFIXES:
        if w.endswith(s) and len(w) - len(s) >= 3:
            return f"{w[:-len(s)]} ##{s}"
    return w


def morph_split(text: str) -> str:
    return _WORD.sub(lambda m: _split(m.group()), text.lower())


if __name__ == "__main__":
    for s in ["yenti ra intha late", "enti ra inta lete", "nuwwu intlo unnava", "vaadiki cheppu, valla intiki vellu"]:
        print(f"{s:38} key: {phonetic_key(s):30} morph: {morph_split(s)}")
