"""Telugu script -> Latin script, the way people actually type it.

Romanized Telugu has no spelling standard: ఏంటి is typed enti / yenti / anti, అంతే is anthe / ante / amthe.
`romanize(text, rng)` samples ONE realistic spelling per word, so calling it many times on the same
line gives many different-but-plausible versions (data augmentation for spelling robustness).
`romanize(text, canonical=True)` gives one fixed, readable spelling (used for model *output*).

How it works: split each word into aksharas (consonant cluster + vowel sign), then pick a spelling
for every sound from a small table of observed alternatives, weighted by how common they are.
"""
import random
import re

# --- Unicode building blocks ---------------------------------------------------------------
VIRAMA, ANUSVARA, VISARGA, CHANDRA = "్", "ం", "ః", "ఁ"

# Each sound -> [(spelling, weight), ...]; the FIRST option is the canonical spelling.
VOWELS = {  # independent vowels (word-initial)
    "అ": [("a", 1)], "ఆ": [("aa", .55), ("a", .45)], "ఇ": [("i", 1)], "ఈ": [("ee", .6), ("i", .4)],
    "ఉ": [("u", 1)], "ఊ": [("oo", .6), ("u", .4)], "ఋ": [("ru", 1)], "ఎ": [("e", .6), ("ye", .3), ("a", .1)],
    "ఏ": [("e", .5), ("ye", .35), ("a", .15)], "ఐ": [("ai", .7), ("ay", .3)], "ఒ": [("o", 1)],
    "ఓ": [("o", .8), ("oo", .2)], "ఔ": [("au", .6), ("ow", .4)],
}
MATRAS = {  # vowel signs after a consonant
    "ా": [("aa", .5), ("a", .5)], "ి": [("i", 1)], "ీ": [("ee", .55), ("i", .45)],
    "ు": [("u", 1)], "ూ": [("oo", .55), ("u", .45)], "ృ": [("ru", 1)],
    "ె": [("e", 1)], "ే": [("e", .7), ("ee", .3)], "ై": [("ai", .7), ("ay", .3)],
    "ొ": [("o", 1)], "ో": [("o", .8), ("oo", .2)], "ౌ": [("au", .6), ("ow", .4)],
}
CONSONANTS = {
    "క": [("k", 1)], "ఖ": [("kh", .5), ("k", .5)], "గ": [("g", 1)], "ఘ": [("gh", .5), ("g", .5)],
    "ఙ": [("n", 1)], "చ": [("ch", 1)], "ఛ": [("chh", .4), ("ch", .6)], "జ": [("j", 1)],
    "ఝ": [("jh", .5), ("j", .5)], "ఞ": [("n", 1)], "ట": [("t", 1)], "ఠ": [("th", .5), ("t", .5)],
    "డ": [("d", 1)], "ఢ": [("dh", .5), ("d", .5)], "ణ": [("n", 1)], "త": [("th", .6), ("t", .4)],
    "థ": [("th", 1)], "ద": [("d", .8), ("dh", .2)], "ధ": [("dh", .6), ("d", .4)], "న": [("n", 1)],
    "ప": [("p", 1)], "ఫ": [("ph", .5), ("f", .5)], "బ": [("b", 1)], "భ": [("bh", .6), ("b", .4)],
    "మ": [("m", 1)], "య": [("y", 1)], "ర": [("r", 1)], "ల": [("l", 1)], "వ": [("v", .65), ("w", .35)],
    "శ": [("sh", .8), ("s", .2)], "ష": [("sh", .8), ("s", .2)], "స": [("s", 1)], "హ": [("h", 1)],
    "ళ": [("l", 1)], "ఱ": [("r", 1)],
}
LABIALS = set("పఫబభమ")

# English words written in Telugu script: people type the English spelling.
LOANWORDS = {
    "సార్": "sir", "సర్": "sir", "మేడం": "madam", "ఓకే": "ok", "హలో": "hello", "హాయ్": "hi", "ప్లీజ్": "please",
    "థాంక్స్": "thanks", "సారీ": "sorry", "లవ్": "love", "ఫోన్": "phone", "కాల్": "call", "ఆఫీస్": "office",
    "జాబ్": "job", "బాస్": "boss", "మనీ": "money", "బ్యాంక్": "bank", "ఎగ్జామ్": "exam", "ఎగ్జామ్స్": "exams",
    "లేట్": "late", "టైం": "time", "టైమ్": "time", "కార్": "car", "బైక్": "bike", "పోలీస్": "police",
    "డాక్టర్": "doctor", "హాస్పిటల్": "hospital", "కాలేజ్": "college", "క్లాస్": "class", "ఫ్రెండ్": "friend",
    "ఫ్రెండ్స్": "friends", "మూవీ": "movie", "సినిమా": "cinema", "పార్టీ": "party", "షాప్": "shop",
    "రూమ్": "room", "డాడీ": "daddy", "మమ్మీ": "mummy", "బ్రో": "bro", "సూపర్": "super", "ఫ్యామిలీ": "family",
}
# Optional English spelling choices for loanwords (people also type them sloppily)
LOAN_VARIANTS = {"please": ["please", "plz", "pls"], "sorry": ["sorry", "sry"], "ok": ["ok", "okay", "k"],
                 "thanks": ["thanks", "thx", "thanku"], "bro": ["bro", "bruh"]}

TELUGU_WORD = re.compile(r"[ఀ-౿]+")


def _pick(options, rng, canonical, style):
    if canonical or len(options) == 1:
        return options[0][0]
    # style > 0 shifts probability toward the shorter, more casual spellings
    weights = [w * (1 + style * (len(options[0][0]) - len(s))) for s, w in options]
    return rng.choices([s for s, _ in options], weights=[max(w, 1e-3) for w in weights])[0]


def romanize_word(word: str, rng: random.Random | None = None, canonical: bool = False, style: float = 0.0) -> str:
    if word in LOANWORDS:
        en = LOANWORDS[word]
        return en if canonical or rng is None else rng.choice(LOAN_VARIANTS.get(en, [en]))
    rng = rng or random.Random(0)
    out, i, chars, forced = [], 0, list(word), None
    while i < len(chars):
        c = chars[i]
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if c in VOWELS:
            out.append(_pick(VOWELS[c], rng, canonical or i > 0, style))  # initial-vowel variation only at word start
        elif c in CONSONANTS:
            cons = forced or _pick(CONSONANTS[c], rng, canonical, style)
            forced = None
            if nxt == VIRAMA:  # conjunct / doubled consonant / word-final consonant
                after = chars[i + 2] if i + 2 < len(chars) else ""
                if after == c and not canonical and rng.random() < 0.15:
                    i += 2  # gemination dropped ("amma" -> "ama"): skip this copy
                    continue
                if after == c:
                    forced = cons  # doubled consonant: same spelling twice (nuvvu / nuwwu, never nuwvu)
                out.append(cons)
                i += 2
                continue
            if nxt in MATRAS:
                out.append(cons + _pick(MATRAS[nxt], rng, canonical, style))
                i += 2
                continue
            out.append(cons + "a")  # inherent vowel
        elif c in (ANUSVARA, CHANDRA):
            after = chars[i + 1] if i + 1 < len(chars) else ""
            base = "m" if (after in LABIALS or not after) else "n"
            out.append(base if canonical else _pick([(base, .9), ("m" if base == "n" else "n", .1)], rng, False, 0))
        elif c == VISARGA:
            out.append("h")
        elif c in MATRAS:  # stray vowel sign
            out.append(MATRAS[c][0][0])
        else:
            out.append(c)
        i += 1
    w = "".join(out)
    # Word-final long "aa" is almost always typed short: entra, ela, kada
    if w.endswith("aa") and (canonical or rng.random() < 0.85):
        w = w[:-1]
    if canonical:
        return w
    # Chat shorthand, calibrated against the curated list in tlm/reference_spellings.py (Stage 1b):
    lazy = 0.5 + style  # lazier typists use more shorthand
    w = w.replace("chch", "cch") if rng.random() < 0.3 else w      # vachchava -> vacchava
    r = rng.random()
    if (w.endswith("aanu") and len(w) > 5) or (w.endswith("anu") and len(w) > 5):
        if r < 0.08 * lazy:
            return w[:w.rfind("anu") - (1 if w.endswith("aanu") else 0)] + "a"   # chesanu -> chesa
    if len(w) > 3 and w.endswith("u") and w[-2] not in "aeiou" and r < 0.05 * lazy:
        return w[:-1]                                                   # nuvvu -> nuvv, nenu -> nen
    if len(w) > 3 and w.endswith("i") and w[-2] not in "aeiou" and r < 0.06 * lazy:
        return w[:-1] + "y"                                             # enti -> enty
    if len(w) > 3 and w.endswith("e") and w[-2] not in "aeiou" and r < 0.06 * lazy:
        return w + "y"                                                  # anthe -> anthey
    return w


def romanize(text: str, rng: random.Random | None = None, canonical: bool = False) -> str:
    """Romanize a whole line. Each call with a different rng gives a different realistic spelling."""
    if canonical:
        return TELUGU_WORD.sub(lambda m: romanize_word(m.group(), canonical=True), text)
    rng = rng or random.Random()
    style = rng.choice([0.0, 0.0, 0.5, 1.5])  # per-sentence typing habit: careful .. lazy
    return TELUGU_WORD.sub(lambda m: romanize_word(m.group(), rng, False, style), text)


if __name__ == "__main__":
    for line in ["ఏంట్రా, ఇంత లేట్ అయ్యింది?", "అంతే కదా, నువ్వు ఎగ్జామ్ ఎలా రాసావ్?", "నేను నిన్ను వదలను"]:
        print("canonical:", romanize(line, canonical=True))
        print("   noisy :", " | ".join(romanize(line, random.Random(s)) for s in range(5)))
