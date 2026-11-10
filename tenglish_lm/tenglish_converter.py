"""
Multi-Paradigm Tenglish Generation Engine (P1, P2, P3).

Converts Telugu script dialogues into 3 standardized Tenglish formats:
- P1 (Loanword-Preserving Natural Tenglish):
    Preserves true English spelling for loans (Sir, phone, office, doctor) while romanizing
    Telugu grammar cleanly.
- P2 (Standard Phonetic Tenglish):
    Lossless phonetic Romanization (preserving long vowels e.g. aa, ee, oo, th, dh).
- P3 (Minimalist Chat / WhatsApp Tenglish):
    Informal colloquial texting style (e.g. nenu, vallu, doc ki call chey).
"""

import re
from typing import Dict, Tuple

# Vowels
TE_VOWELS = {
    "అ": "a", "ఆ": "aa", "ఇ": "i", "ఈ": "ee", "ఉ": "u", "ఊ": "oo",
    "ఋ": "ru", "ౠ": "roo", "ఎ": "e", "ఏ": "e", "ఐ": "ai",
    "ఒ": "o", "ఓ": "o", "ఔ": "au", "అం": "am", "అః": "aha"
}

# Matras (Vowel signs)
TE_MATRAS = {
    "\u0C3E": "aa", "\u0C3F": "i", "\u0C40": "ee", "\u0C41": "u", "\u0C42": "oo",
    "\u0C43": "ru", "\u0C44": "roo", "\u0C46": "e", "\u0C47": "e", "\u0C48": "ai",
    "\u0C4A": "o", "\u0C4B": "o", "\u0C4C": "au", "\u0C02": "m", "\u0C03": "h",
    "\u0C4D": ""  # virama
}

# Consonants
TE_CONSONANTS_P1 = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "ch", "ఛ": "chh", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "t", "ఠ": "th", "డ": "d", "ఢ": "dh", "ణ": "n",
    "త": "th", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "f", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ల": "l", "వ": "v", "శ": "sh",
    "ష": "sh", "స": "s", "హ": "h", "ళ": "l", "క్ష": "ksh", "ఱ": "r"
}

# Common English Loanword Catalog in Telugu Script
LOANWORD_MAP = {
    "సార్": "Sir", "సర్": "Sir", "మేడం": "Madam", "ఓకే": "okay", "హలో": "hello", "హాయ్": "hi",
    "హే": "hey", "ప్లీజ్": "please", "థాంక్స్": "thanks", "థాంక్యూ": "thank you", "సారీ": "sorry",
    "నో": "no", "ఎస్": "yes", "లవ్": "love", "మిస్": "miss", "కేర్": "care", "ఫోన్": "phone",
    "కాల్": "call", "మెసేజ్": "message", "నంబర్": "number", "ఆఫీస్": "office", "జాబ్": "job",
    "వర్క్": "work", "బాస్": "boss", "మేనేజర్": "manager", "స్టాఫ్": "staff", "మీటింగ్": "meeting",
    "సాలరీ": "salary", "మనీ": "money", "బ్యాంక్": "bank", "అకౌంట్": "account", "బిజినెస్": "business",
    "డీల్": "deal", "ప్రాజెక్ట్": "project", "కారు": "car", "కార్": "car", "బైక్": "bike", "లారీ": "lorry",
    "బస్సు": "bus", "ట్రైన్": "train", "ఫ్లైట్": "flight", "రోడ్డు": "road", "ట్రాఫిక్": "traffic",
    "డ్రైవర్": "driver", "లైసెన్స్": "license", "సిగ్నల్": "signal", "పోలీస్": "police", "స్టేషన్": "station",
    "జైలు": "jail", "కోర్టు": "court", "జడ్జి": "judge", "లాయర్": "lawyer", "కేసు": "case",
    "డాక్టర్": "doctor", "హాస్పిటల్": "hospital", "నర్స్": "nurse", "మెడిసిన్": "medicine",
    "కాలేజ్": "college", "స్కూల్": "school", "టీచర్": "teacher", "స్టూడెంట్": "student",
    "క్లాస్": "class", "ఎగ్జామ్": "exam", "సినిమా": "cinema", "మూవీ": "movie", "హీరో": "hero",
    "హీరోయిన్": "heroine", "డైరెక్టర్": "director", "సీన్": "scene", "రూమ్": "room", "హౌస్": "house",
    "లైట్": "light", "ఫ్యాన్": "fan", "పార్టీ": "party", "డ్రింక్": "drink", "టీ": "tea", "కాఫీ": "coffee",
    "షర్ట్": "shirt", "ప్యాంట్": "pant", "షూస్": "shoes", "వాచ్": "watch", "బ్యాగ్": "bag",
    "టెన్షన్": "tension", "ప్రాబ్లం": "problem", "డౌట్": "doubt", "చాన్స్": "chance", "లైఫ్": "life",
    "టైమ్": "time", "సూపర్": "super", "కూల్": "cool", "సింపుల్": "simple", "క్లియర్": "clear",
    "రెడీ": "ready", "ఫాస్ట్": "fast", "స్లో": "slow", "ఈజీ": "easy", "టఫ్": "tough", "స్ట్రాంగ్": "strong",
    "హ్యాపీ": "happy", "హాపీ": "happy", "క్రేజీ": "crazy", "సైలెంట్": "silent", "ఫ్రీ": "free"
}

# Common Telugu case suffixes attached to loanwords
CASE_SUFFIXES = [
    ("లు", "s"), ("ల", "s"), ("లోని", " loni"), ("లో", "-lo"), ("తో", "-to"),
    ("కి", "-ki"), ("కు", "-ku"), ("ని", "-ni"), ("ను", "-nu"), ("నుండి", " nundi"),
    ("నుంచి", " nunchi"), ("కోసం", " kosam"), ("లా", "-la"), ("గా", "-ga"),
    ("గారు", " garu"), ("చేస్తున్నా", " chestunna"), ("చేసాడు", " chesadu"),
    ("చేసింది", " chesindi"), ("చేయి", " cheyi"), ("చేయ్", " chey"), ("చేయండి", " cheyandi")
]


def transliterate_phonetic(word: str, mode: str = "P1") -> str:
    """Core phonetic converter for native Telugu words."""
    res = []
    i = 0
    chars = list(word)
    n = len(chars)

    while i < n:
        ch = chars[i]
        if ch in TE_VOWELS:
            res.append(TE_VOWELS[ch])
        elif ch in TE_CONSONANTS_P1:
            base = TE_CONSONANTS_P1[ch]
            if i + 1 < n and chars[i + 1] in TE_MATRAS:
                matra = chars[i + 1]
                matra_val = TE_MATRAS[matra]
                if matra == "\u0C02":  # Anusvara (సున్న)
                    # Use 'n' before dental/alveolar/palatal consonants or word-end
                    next_c = chars[i + 2] if i + 2 < n else ""
                    if next_c in ["త", "థ", "ద", "ధ", "ట", "ఠ", "డ", "ఢ", "చ", "ఛ", "జ", "శ", "ష", "స"]:
                        matra_val = "n"
                    elif not next_c:
                        matra_val = "m"
                    else:
                        matra_val = "n"
                if mode == "P3":
                    matra_val = matra_val.replace("aa", "a").replace("ee", "i").replace("oo", "u")
                res.append(base + matra_val)
                i += 1
            elif i + 1 < n and chars[i + 1] == "\u0C4D":  # virama
                res.append(base)
                i += 1
            else:
                res.append(base + "a")
        elif ch in TE_MATRAS:
            val = TE_MATRAS[ch]
            if ch == "\u0C02":
                val = "n"
            res.append(val)
        else:
            res.append(ch)
        i += 1

    out = "".join(res)
    # Natural Telugu phonetic cleanups
    out = out.replace("anm", "am").replace("enm", "em").replace("inm", "im")
    out = out.replace("eymdi", "eyandi").replace("eymdu", "eyandu")
    if mode == "P3":
        out = out.replace("aa", "a").replace("ee", "i").replace("oo", "u")
        out = out.replace("th", "t").replace("dh", "d")
    return out


def convert_token_to_tenglish(token: str) -> Tuple[str, str, str]:
    """Converts a single token into (P1, P2, P3)."""
    # Keep punctuation/digits
    if not re.search(r"[\u0C00-\u0C7F]", token):
        return token, token, token

    clean_w = token.strip(".,!?:;\"'()[]{}-_–—/\\")
    punct_prefix = token[:token.find(clean_w)] if clean_w else ""
    punct_suffix = token[token.find(clean_w) + len(clean_w):] if clean_w else ""

    # Check loanword map (Exact)
    if clean_w in LOANWORD_MAP:
        en_root = LOANWORD_MAP[clean_w]
        p1 = f"{punct_prefix}{en_root}{punct_suffix}"
        p2 = f"{punct_prefix}{transliterate_phonetic(clean_w, 'P1')}{punct_suffix}"
        p3 = f"{punct_prefix}{en_root.lower()}{punct_suffix}"
        return p1, p2, p3

    # Check loanword map (Inflected: Root + Suffix)
    for stem, en_root in LOANWORD_MAP.items():
        if clean_w.startswith(stem) and len(clean_w) > len(stem):
            rem = clean_w[len(stem):]
            for suff_te, suff_en in CASE_SUFFIXES:
                if rem == suff_te:
                    p1 = f"{punct_prefix}{en_root}{suff_en}{punct_suffix}"
                    p2 = f"{punct_prefix}{transliterate_phonetic(clean_w, 'P1')}{punct_suffix}"
                    p3 = f"{punct_prefix}{en_root.lower()}{suff_en}{punct_suffix}"
                    return p1, p2, p3

    # Native Telugu word
    p1 = f"{punct_prefix}{transliterate_phonetic(clean_w, 'P1')}{punct_suffix}"
    p2 = f"{punct_prefix}{transliterate_phonetic(clean_w, 'P2')}{punct_suffix}"
    p3 = f"{punct_prefix}{transliterate_phonetic(clean_w, 'P3')}{punct_suffix}"
    return p1, p2, p3


def convert_sentence_to_tenglish(sentence: str) -> Dict[str, str]:
    """Converts a full Telugu dialogue utterance into 3 Tenglish variants."""
    tokens = sentence.split()
    p1_tokens = []
    p2_tokens = []
    p3_tokens = []

    for t in tokens:
        p1, p2, p3 = convert_token_to_tenglish(t)
        p1_tokens.append(p1)
        p2_tokens.append(p2)
        p3_tokens.append(p3)

    return {
        "p1_natural": " ".join(p1_tokens),
        "p2_phonetic": " ".join(p2_tokens),
        "p3_chat": " ".join(p3_tokens),
    }


if __name__ == "__main__":
    sample = "సార్, వెంటనే ఆఫీస్ కి వెళ్లి డాక్టర్ కి ఫోన్ చేయండి. ప్రాబ్లం చాలా సీరియస్ గా ఉంది!"
    res = convert_sentence_to_tenglish(sample)
    print("TELUGU ORIGINAL:\n ", sample)
    print("\nP1 (Natural Loanword-Preserved Tenglish):\n ", res["p1_natural"])
    print("\nP2 (Standard Phonetic Tenglish):\n ", res["p2_phonetic"])
    print("\nP3 (Minimalist WhatsApp Chat Tenglish):\n ", res["p3_chat"])
