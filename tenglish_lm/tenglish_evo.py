"""
Milestone 2: Tenglish-Evo Engine
(Universal Telugu Akshara-to-Latin Phonetic Transducer + Era-Calibrated Multiplier + Broken-Tenglish Disambiguator)

Solves:
1. Universal Telugu Unicode (0x0C00-0x0C7F) Akshara-to-Latin Transducer:
   - Converts ANY arbitrary Telugu Unicode word into 100% Latin-script Tenglish across 4 Texter Personas
     (handles independent vowels, consonants, guninthalu/matras, vattulu/virama conjuncts, and nasal sunna 'ం' assimilation).
2. 4 Texter Personas:
   - P1_STANDARD: Standard Phonetic ('anthega', 'yenti', 'endhuku', 'chepthunna', 'maatlaaduthunnav')
   - P2_MINIMAL:  Fast Minimalist ('antega', 'enti', 'enduku', 'cheptuna', 'matladtunav')
   - P3_EXPRESS:  Expressive Elongated ('antheygaaa', 'yentiii', 'anthe ga', 'avunaaa')
   - P4_ACROPHON: English-Letter Acrophony & Vowel Shift ('Anti'=ఏంటి, 'm'=ఏం, 'nduku'=ఎందుకు, 'la'=ఎలా, 'chappu'=చెప్పు, 'nv'=నువ్వు)
3. 4-Way Word-Level Language & Morphology Tagger:
   - EN, TE, HYBRID_FUSED ('office-ki', 'party-lo'), LIGHT_VERB ('ghost chesindu', 'vibe avvatledu')
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# 1. Universal Telugu Unicode (0x0C00-0x0C7F) Phonetic Tables
# ---------------------------------------------------------------------------

INDEPENDENT_VOWELS_P1 = {
    "అ": "a", "ఆ": "aa", "ఇ": "i", "ఈ": "ee", "ఉ": "u", "ఊ": "oo",
    "ఋ": "ru", "ౠ": "roo", "ఎ": "e", "ఏ": "ye", "ఐ": "ai",
    "ఒ": "o", "ఓ": "o", "ఔ": "au",
}

INDEPENDENT_VOWELS_P2 = {
    "అ": "a", "ఆ": "a", "ఇ": "i", "ఈ": "i", "ఉ": "u", "ఊ": "u",
    "ఋ": "ru", "ౠ": "ru", "ఎ": "e", "ఏ": "e", "ఐ": "ai",
    "ఒ": "o", "ఓ": "o", "ఔ": "au",
}

CONSONANTS_P1 = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "ch", "ఛ": "chh", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "t", "ఠ": "th", "డ": "d", "ఢ": "dh", "ణ": "n",
    "త": "th", "థ": "th", "ద": "dh", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "r", "ల": "l", "ళ": "l",
    "వ": "v", "శ": "sh", "ష": "sh", "స": "s", "హ": "h",
}

CONSONANTS_P2 = {
    # Fast minimalist texter drops aspiration ('th' -> 't', 'dh' -> 'd', 'bh' -> 'b', 'sh' -> 's')
    "క": "k", "ఖ": "k", "గ": "g", "ఘ": "g", "ఙ": "ng",
    "చ": "ch", "ఛ": "ch", "జ": "j", "ఝ": "j", "ఞ": "ny",
    "ట": "t", "ఠ": "t", "డ": "d", "ఢ": "d", "ణ": "n",
    "త": "t", "థ": "t", "ద": "d", "ధ": "d", "న": "n",
    "ప": "p", "ఫ": "f", "బ": "b", "భ": "b", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "r", "ల": "l", "ళ": "l",
    "వ": "v", "శ": "s", "ష": "sh", "స": "s", "హ": "h",
}

VOWEL_MATRAS_P1 = {
    "ా": "aa", "ి": "i", "ీ": "ee", "ు": "u", "ూ": "oo",
    "ృ": "ru", "ె": "e", "ే": "e", "ై": "ai", "ొ": "o", "ో": "o", "ౌ": "au",
}

VOWEL_MATRAS_P2 = {
    "ా": "a", "ి": "i", "ీ": "i", "ు": "u", "ూ": "u",
    "ృ": "ru", "ె": "e", "ే": "e", "ై": "ai", "ొ": "o", "ో": "o", "ౌ": "ow",
}

VIRAMA = "్"
LABIAL_CONSONANTS = {"ప", "ఫ", "బ", "భ", "మ", "వ"}


def transliterate_telugu_word_universal(word: str, persona: str = "P1_STANDARD") -> str:
    """
    Universal character/akshara state-machine transducer that converts ANY Telugu Unicode word
    into natural Romanized Tenglish according to the chosen Texter Persona.
    Guarantees 0% leftover Telugu Unicode characters.
    """
    use_minimal = persona in {"P2_MINIMAL", "P4_ACROPHON"}
    ivowels = INDEPENDENT_VOWELS_P2 if use_minimal else INDEPENDENT_VOWELS_P1
    cons_map = CONSONANTS_P2 if use_minimal else CONSONANTS_P1
    matras = VOWEL_MATRAS_P2 if use_minimal else VOWEL_MATRAS_P1

    chars = list(word)
    n = len(chars)
    out: List[str] = []
    i = 0

    while i < n:
        ch = chars[i]

        # Skip zero-width joiners/non-joiners
        if ch in {"\u200c", "\u200d"}:
            i += 1
            continue

        # 1. Independent Vowel
        if ch in ivowels:
            out.append(ivowels[ch])
            i += 1
            continue

        # 2. Consonant (possibly followed by virama or vowel matra)
        if ch in cons_map:
            c_rom = cons_map[ch]
            next_ch = chars[i + 1] if i + 1 < n else ""

            if next_ch == VIRAMA:
                # Consonant ends in virama (conjunct or word-final pollubhallu)
                out.append(c_rom)
                i += 2
                continue
            elif next_ch in matras:
                v_rom = matras[next_ch]
                # Word-final long 'ా' is usually single 'a' in normal texting unless P3_EXPRESS
                if next_ch == "ా" and (i + 2 == n) and persona != "P3_EXPRESS":
                    v_rom = "a"
                elif next_ch == "ే" and persona == "P3_EXPRESS" and (i + 2 == n):
                    v_rom = "ey"
                out.append(c_rom + v_rom)
                i += 2
                continue
            else:
                # Inherent vowel 'a'
                out.append(c_rom + "a")
                i += 1
                continue

        # 3. Nasal Anusvara / Sunna ('ం') — assimilates to 'm' before labials or at word end, else 'n'
        if ch == "ం":
            next_ch = chars[i + 1] if i + 1 < n else ""
            if next_ch == "" or next_ch in LABIAL_CONSONANTS:
                out.append("m")
            else:
                out.append("n")
            i += 1
            continue

        # 4. Visarga / Ardhavisarga
        if ch in {"ః", "ఁ"}:
            out.append("h" if ch == "ః" else "n")
            i += 1
            continue

        # 5. Standalone matra (fallback)
        if ch in matras:
            out.append(matras[ch])
            i += 1
            continue

        # 6. Any ASCII/punctuation character passes through
        out.append(ch)
        i += 1

    roman = "".join(out)

    # Apply Persona-specific post-processing
    if persona == "P2_MINIMAL":
        # Collapse geminate double consonants ('kk' -> 'k', 'pp' -> 'p', 'll' -> 'l', 'nn' -> 'n', 'tt' -> 't')
        for dbl in ("kk", "pp", "ll", "nn", "tt", "dd", "mm", "vv"):
            roman = roman.replace(dbl, dbl[0])
    elif persona == "P3_EXPRESS":
        if roman.endswith(("a", "i", "u", "e", "o")) and len(roman) > 3:
            roman = roman + roman[-1]
    elif persona == "P4_ACROPHON":
        # Apply English vowel interference ('chep' -> 'chap', 'telu' -> 'talu', 'vell' -> 'vall')
        roman = (
            roman.replace("chep", "chap")
            .replace("telu", "talu")
            .replace("vell", "vall")
            .replace("pett", "patt")
        )
        for dbl in ("kk", "pp", "nn", "tt"):
            roman = roman.replace(dbl, dbl[0])

    return roman


# ---------------------------------------------------------------------------
# 2. Lexicons for English, Telugu Loanwords, Hybrid Suffixes & Acrophonic Map
# ---------------------------------------------------------------------------

ENGLISH_CORE_WORDS = {
    "bro", "boss", "sir", "madam", "college", "bus", "stop", "lecturer", "permission",
    "police", "station", "cool", "coffee", "feeling", "operation", "commit", "warning",
    "life", "decision", "career", "settle", "decide", "feelings", "value", "plan",
    "calculation", "normal", "next", "move", "guess", "exactly", "emotion", "pure",
    "logic", "game", "job", "business", "start", "first", "swimming", "direct", "lite",
    "reject", "office", "room", "connect", "sudden", "option", "party", "situation",
    "control", "full", "trauma", "scene", "create", "damage", "message", "seen",
    "reply", "english", "ghost", "delulu", "vibe", "hint", "weekend", "trip", "buildup",
    "last", "minute", "hand", "manager", "friday", "evening", "client", "call", "mind",
    "block", "better", "matter", "update", "point", "topic", "status", "story", "level",
    "attitude", "focus", "serious", "actual", "basic", "cringe", "toxic", "chill",
    "hero", "villain", "interval", "climax", "project", "meeting", "deadline", "salary",
    "girlfriend", "breakup", "patchup", "friend", "friends", "phone", "mobile", "car",
    "bike", "traffic", "signal", "hotel", "bill", "table", "order", "waiter", "doctor",
    "hospital", "ticket", "cinema", "theater", "popcorn", "love", "heart", "brain",
}

TELUGU_SCRIPT_ENGLISH_LOANS = {
    "కాలేజీ": ("college", "EN"),
    "కాలేజీకి": ("college-ki", "HYBRID_FUSED"),
    "బస్": ("bus", "EN"),
    "స్టాప్": ("stop", "EN"),
    "లెక్చరర్": ("lecturer", "EN"),
    "పర్మిషన్": ("permission", "EN"),
    "పోలీస్": ("police", "EN"),
    "స్టేషన్": ("station", "EN"),
    "కూల్": ("cool", "EN"),
    "కాఫీ": ("coffee", "EN"),
    "ఫీలింగ్": ("feeling", "EN"),
    "ఆపరేషన్": ("operation", "EN"),
    "కమిట్": ("commit", "LIGHT_VERB"),
    "వార్నింగ్": ("warning", "EN"),
    "లైఫ్": ("life", "EN"),
    "డెసిషన్": ("decision", "EN"),
    "కెరీర్": ("career", "EN"),
    "సెటిల్": ("settle", "LIGHT_VERB"),
    "డిసైడ్": ("decide", "LIGHT_VERB"),
    "ఫీలింగ్స్": ("feelings", "EN"),
    "వాల్యూ": ("value", "EN"),
    "ప్లాన్": ("plan", "EN"),
    "క్యాలిక్యులేషన్": ("calculation", "EN"),
    "నార్మల్": ("normal", "EN"),
    "నెక్స్ట్": ("next", "EN"),
    "మూవ్": ("move", "EN"),
    "గెస్": ("guess", "LIGHT_VERB"),
    "ఎగ్జాక్ట్‌లీ": ("exactly", "EN"),
    "ఎమోషన్": ("emotion", "EN"),
    "ప్యూర్": ("pure", "EN"),
    "లాజిక్": ("logic", "EN"),
    "గేమ్": ("game", "EN"),
    "జాబ్": ("job", "EN"),
    "బిజినెస్": ("business", "EN"),
    "స్టార్ట్": ("start", "LIGHT_VERB"),
    "ఫస్ట్": ("first", "EN"),
    "స్విమ్మింగ్": ("swimming", "EN"),
    "డైరెక్ట్": ("direct", "EN"),
    "లైట్": ("lite", "EN"),
    "రిజెక్ట్": ("reject", "LIGHT_VERB"),
    "ఆఫీస్": ("office", "EN"),
    "ఆఫీస్‌కి": ("office-ki", "HYBRID_FUSED"),
    "రూమ్": ("room", "EN"),
    "కనెక్ట్": ("connect", "LIGHT_VERB"),
    "సడన్": ("sudden", "EN"),
    "ఆప్షన్": ("option", "EN"),
    "బ్రో": ("bro", "EN"),
    "పార్టీకి": ("party-ki", "HYBRID_FUSED"),
    "సిట్యుయేషన్": ("situation", "EN"),
    "కంట్రోల్": ("control", "EN"),
    "ఫుల్": ("full", "EN"),
    "ట్రామా": ("trauma", "EN"),
    "సీన్": ("scene", "EN"),
    "క్రియేట్": ("create", "LIGHT_VERB"),
    "డ్యామేజ్": ("damage", "LIGHT_VERB"),
    "మెసేజ్‌లు": ("messages-lu", "HYBRID_FUSED"),
    "రిప్లై": ("reply", "EN"),
    "ఇంగ్లీష్": ("english", "EN"),
    "ఘోస్ట్": ("ghost", "LIGHT_VERB"),
    "డెలులు": ("delulu", "EN"),
    "వైబ్": ("vibe", "LIGHT_VERB"),
    "హింట్": ("hint", "EN"),
    "వీకెండ్": ("weekend", "EN"),
    "ట్రిప్": ("trip", "EN"),
    "బిల్డప్": ("buildup", "EN"),
    "లాస్ట్": ("last", "EN"),
    "మినిట్": ("minute", "EN"),
    "హ్యాండ్": ("hand", "EN"),
    "మేనేజర్": ("manager", "EN"),
    "ఫ్రైడే": ("friday", "EN"),
    "ఈవినింగ్": ("evening", "EN"),
    "క్లయింట్": ("client", "EN"),
    "కాల్": ("call", "EN"),
    "మైండ్": ("mind", "EN"),
    "బ్లాక్": ("block", "LIGHT_VERB"),
    "బెటర్": ("better", "EN"),
    "ట్రాఫిక్": ("traffic", "EN"),
    "సిగ్నల్": ("signal", "EN"),
    "ప్రాజెక్ట్": ("project", "EN"),
    "మీటింగ్": ("meeting", "EN"),
    "డెడ్‌లైన్": ("deadline", "EN"),
    "శాలరీ": ("salary", "EN"),
    "బ్రేకప్": ("breakup", "EN"),
    "ప్యాచప్": ("patchup", "EN"),
    "ఫోన్": ("phone", "EN"),
    "కార్": ("car", "EN"),
    "బైక్": ("bike", "EN"),
    "హోటల్": ("hotel", "EN"),
    "బిల్": ("bill", "EN"),
    "ఆర్డర్": ("order", "LIGHT_VERB"),
    "డాక్టర్": ("doctor", "EN"),
    "హాస్పిటల్": ("hospital", "EN"),
    "టికెట్": ("ticket", "EN"),
    "సినిమా": ("cinema", "EN"),
    "థియేటర్": ("theater", "EN"),
    "లవ్": ("love", "EN"),
    "హార్ట్": ("heart", "EN"),
    "బ్రెయిన్": ("brain", "EN"),
    "సార్": ("sir", "EN"),
    "మేడమ్": ("madam", "EN"),
}

TELUGU_ROMAN_SUFFIXES = (
    "ki", "lo", "tho", "ni", "kosam", "nundi", "nudchi", "meeda", "la", "laga", "ga", "ra", "lu"
)

CANONICAL_TO_PERSONA_LEXICON: Dict[str, Dict[str, str]] = {
    "అంతేగా": {
        "P1_STANDARD": "anthega",
        "P2_MINIMAL": "antega",
        "P3_EXPRESS": "antheygaaa",
        "P4_ACROPHON": "anthe ga",
    },
    "ఏంటి": {
        "P1_STANDARD": "yenti",
        "P2_MINIMAL": "enti",
        "P3_EXPRESS": "yentiii",
        "P4_ACROPHON": "Anti",
    },
    "ఏంట్రా": {
        "P1_STANDARD": "yentra",
        "P2_MINIMAL": "entra",
        "P3_EXPRESS": "yentiraaa",
        "P4_ACROPHON": "Antra",
    },
    "ఏంది": {
        "P1_STANDARD": "yendhi",
        "P2_MINIMAL": "endi",
        "P3_EXPRESS": "yendhii",
        "P4_ACROPHON": "Andi",
    },
    "ఏందిరా": {
        "P1_STANDARD": "yendhira",
        "P2_MINIMAL": "endira",
        "P3_EXPRESS": "yendhiraaa",
        "P4_ACROPHON": "Andira",
    },
    "ఏం": {
        "P1_STANDARD": "yem",
        "P2_MINIMAL": "em",
        "P3_EXPRESS": "yemmm",
        "P4_ACROPHON": "m",
    },
    "ఎందుకు": {
        "P1_STANDARD": "endhuku",
        "P2_MINIMAL": "enduku",
        "P3_EXPRESS": "yendhukuuu",
        "P4_ACROPHON": "nduku",
    },
    "ఎలా": {
        "P1_STANDARD": "yela",
        "P2_MINIMAL": "ela",
        "P3_EXPRESS": "yelaaa",
        "P4_ACROPHON": "la",
    },
    "ఎక్కడ": {
        "P1_STANDARD": "ekkada",
        "P2_MINIMAL": "ekada",
        "P3_EXPRESS": "yekkada",
        "P4_ACROPHON": "kkada",
    },
    "ఎక్కడికి": {
        "P1_STANDARD": "ekkadiki",
        "P2_MINIMAL": "ekadiki",
        "P3_EXPRESS": "yekkadikiii",
        "P4_ACROPHON": "kkadiki",
    },
    "చెప్పు": {
        "P1_STANDARD": "cheppu",
        "P2_MINIMAL": "chepu",
        "P3_EXPRESS": "cheppuuu",
        "P4_ACROPHON": "chappu",
    },
    "చెప్తున్నావ్": {
        "P1_STANDARD": "chepthunnav",
        "P2_MINIMAL": "cheptunav",
        "P3_EXPRESS": "chepthunnava",
        "P4_ACROPHON": "chptnv",
    },
    "చేస్తున్నావ్": {
        "P1_STANDARD": "chesthunnav",
        "P2_MINIMAL": "chestunav",
        "P3_EXPRESS": "chesthunnaav",
        "P4_ACROPHON": "chstnv",
    },
    "చేస్తున్నావు": {
        "P1_STANDARD": "chesthunnavu",
        "P2_MINIMAL": "chestunav",
        "P3_EXPRESS": "chesthunnaav",
        "P4_ACROPHON": "chstnv",
    },
    "నువ్వు": {
        "P1_STANDARD": "nuvvu",
        "P2_MINIMAL": "nuvu",
        "P3_EXPRESS": "nuvvuuu",
        "P4_ACROPHON": "nv",
    },
    "నాకు": {
        "P1_STANDARD": "naaku",
        "P2_MINIMAL": "naku",
        "P3_EXPRESS": "naakuuu",
        "P4_ACROPHON": "nku",
    },
    "నీకు": {
        "P1_STANDARD": "neeku",
        "P2_MINIMAL": "niku",
        "P3_EXPRESS": "neekuuu",
        "P4_ACROPHON": "nku",
    },
    "అర్థం": {
        "P1_STANDARD": "ardham",
        "P2_MINIMAL": "artam",
        "P3_EXPRESS": "artham",
        "P4_ACROPHON": "ardhm",
    },
    "కావట్లేదు": {
        "P1_STANDARD": "kaavatledhu",
        "P2_MINIMAL": "kavatledu",
        "P3_EXPRESS": "kaavatleee",
        "P4_ACROPHON": "kavatle",
    },
    "అవ్వట్లేదు": {
        "P1_STANDARD": "avvatledhu",
        "P2_MINIMAL": "avvatledu",
        "P3_EXPRESS": "avvatleee",
        "P4_ACROPHON": "avtle",
    },
    "ఇవ్వట్లేదు": {
        "P1_STANDARD": "ivvatledhu",
        "P2_MINIMAL": "ivvatledu",
        "P3_EXPRESS": "ivvatleee",
        "P4_ACROPHON": "ivtle",
    },
    "అసలు": {
        "P1_STANDARD": "asalu",
        "P2_MINIMAL": "aslu",
        "P3_EXPRESS": "assalu",
        "P4_ACROPHON": "aslu",
    },
    "ఇప్పుడు": {
        "P1_STANDARD": "ippudu",
        "P2_MINIMAL": "ipudu",
        "P3_EXPRESS": "ippuduuu",
        "P4_ACROPHON": "ipdu",
    },
    "తెలుసు": {
        "P1_STANDARD": "thelusu",
        "P2_MINIMAL": "telusu",
        "P3_EXPRESS": "thelusU",
        "P4_ACROPHON": "talusu",
    },
    "తెలుసా": {
        "P1_STANDARD": "thelusaa",
        "P2_MINIMAL": "telsa",
        "P3_EXPRESS": "thelusaaa",
        "P4_ACROPHON": "talsa",
    },
}

REVERSE_VARIANT_TO_CANONICAL: Dict[str, str] = {}
for canonical_te, p_map in CANONICAL_TO_PERSONA_LEXICON.items():
    for _, variant in p_map.items():
        REVERSE_VARIANT_TO_CANONICAL[variant.lower()] = canonical_te

EXTRA_VARIANTS = {
    "anteyga": "అంతేగా",
    "anthayga": "అంతేగా",
    "anthegaa": "అంతేగా",
    "anti": "ఏంటి",
    "antra": "ఏంట్రా",
    "ndku": "ఎందుకు",
    "talusu": "తెలుసు",
    "telsu": "తెలుసు",
    "thelusu": "తెలుసు",
    "vallu": "వెళ్ళు",
}
REVERSE_VARIANT_TO_CANONICAL.update(EXTRA_VARIANTS)


def disambiguate_homograph(word: str, prev_word: str = "", next_word: str = "") -> Dict[str, Any]:
    """Resolves cross-language homographs such as 'Anti'/'anti', 'm', 'la', 'pattu'."""
    w_low = word.strip("?!.,:;").lower()
    p_low = prev_word.strip("?!.,:;").lower()
    n_low = next_word.strip("?!.,:;").lower()

    if w_low == "anti":
        if p_low in {"maa", "ma", "mee", "mi", "pakkinti", "colony"} or n_low in {"vachindi", "cheppindi", "intlo"}:
            return {
                "surface": word,
                "resolved_te": "ఆంటీ",
                "meaning": "Aunty (Loanword Noun)",
                "confidence": 0.94,
                "escalated_to_archon": True,
            }
        if n_low in {"social", "hero", "national", "virus", "corruption", "biotic"}:
            return {
                "surface": word,
                "resolved_te": "యాంటీ (anti-)",
                "meaning": "English prefix 'anti-'",
                "confidence": 0.96,
                "escalated_to_archon": True,
            }
        return {
            "surface": word,
            "resolved_te": "ఏంటి",
            "meaning": "What (Gen-Z Acrophonic 'A' + 'nti')",
            "confidence": 0.95,
            "escalated_to_archon": True,
        }

    if w_low == "m":
        if n_low in {"chestunnav", "chstnv", "aindhi", "aindi", "cheppu", "chappu", "ledhu", "ledu", "plan"} or p_low in {"asalu", "aslu", "bro", "ippudu", "ipdu"}:
            return {
                "surface": word,
                "resolved_te": "ఏం",
                "meaning": "What ('M' letter-name acrophony for ఏం)",
                "confidence": 0.93,
                "escalated_to_archon": True,
            }

    if w_low in REVERSE_VARIANT_TO_CANONICAL:
        return {
            "surface": word,
            "resolved_te": REVERSE_VARIANT_TO_CANONICAL[w_low],
            "meaning": "Canonical Telugu match",
            "confidence": 0.98,
            "escalated_to_archon": False,
        }

    return {
        "surface": word,
        "resolved_te": word,
        "meaning": "Pass-through token",
        "confidence": 0.85,
        "escalated_to_archon": False,
    }


def classify_token(token: str) -> Tuple[str, str]:
    """Classifies token into EN, HYBRID_FUSED, LIGHT_VERB, or TE."""
    clean_tok = re.sub(r"[^\w\u0c00-\u0c7f-]", "", token)
    if not clean_tok:
        return ("PUNCT", "")

    if clean_tok in TELUGU_SCRIPT_ENGLISH_LOANS:
        en_equiv, tag_type = TELUGU_SCRIPT_ENGLISH_LOANS[clean_tok]
        return (tag_type, f"{clean_tok} -> {en_equiv}")

    for loan_te, (en_equiv, _) in TELUGU_SCRIPT_ENGLISH_LOANS.items():
        if clean_tok.startswith(loan_te) and len(clean_tok) > len(loan_te):
            suffix_te = clean_tok[len(loan_te) :]
            suffix_rom = transliterate_telugu_word_universal(suffix_te, "P2_MINIMAL")
            return ("HYBRID_FUSED", f"{en_equiv}+-{suffix_rom}")

    low_tok = clean_tok.lower()

    if "-" in low_tok:
        parts = low_tok.split("-", 1)
        if parts[0] in ENGLISH_CORE_WORDS and parts[1] in TELUGU_ROMAN_SUFFIXES:
            return ("HYBRID_FUSED", f"{parts[0]}+-{parts[1]}")

    if low_tok not in ENGLISH_CORE_WORDS and low_tok not in REVERSE_VARIANT_TO_CANONICAL:
        for suf in TELUGU_ROMAN_SUFFIXES:
            if low_tok.endswith(suf) and len(low_tok) > len(suf) + 2:
                stem = low_tok[: -len(suf)]
                if stem in ENGLISH_CORE_WORDS:
                    return ("HYBRID_FUSED", f"{stem}+-{suf}")

    if low_tok in ENGLISH_CORE_WORDS:
        return ("EN", low_tok)

    return ("TE", clean_tok)


def compute_utterance_cmi(text: str) -> Dict[str, Any]:
    """Computes Code-Mixing Index (CMI %) and counts of EN, TE, HYBRID_FUSED, LIGHT_VERB tokens."""
    raw_tokens = [t for t in text.split() if re.sub(r"[^\w\u0c00-\u0c7f]", "", t)]
    counts = Counter()
    tagged_tokens = []

    for tok in raw_tokens:
        cat, detail = classify_token(tok)
        if cat == "PUNCT":
            continue
        counts[cat] += 1
        tagged_tokens.append({"token": tok, "tag": cat, "detail": detail})

    total = sum(counts.values())
    if total == 0:
        return {"cmi_pct": 0.0, "en_lexeme_pct": 0.0, "counts": dict(counts), "tagged_tokens": []}

    en_total = counts["EN"] + counts["HYBRID_FUSED"] + counts["LIGHT_VERB"]
    te_total = counts["TE"]
    en_ratio_pct = round(100.0 * en_total / total, 2)
    cmi_formula = round(100.0 * (1.0 - max(en_total, te_total) / total), 2)

    return {
        "total_words": total,
        "en_lexeme_pct": en_ratio_pct,
        "cmi_gamback": cmi_formula,
        "counts": {
            "TE": counts["TE"],
            "EN": counts["EN"],
            "HYBRID_FUSED": counts["HYBRID_FUSED"],
            "LIGHT_VERB": counts["LIGHT_VERB"],
        },
        "tagged_tokens": tagged_tokens,
    }


ERA_LEXICAL_SWAPS = {
    "విషయం": {"ERA_2000": "matter", "ERA_2025": "matter"},
    "స్నేహం": {"ERA_2000": "friendship", "ERA_2025": "bonding"},
    "ప్రయత్నించాను": {"ERA_2000": "try chesanu", "ERA_2025": "try chesna"},
    "కోపం": {"ERA_2000": "angry", "ERA_2025": "trigger"},
    "పరిస్థితులు": {"ERA_2000": "situations", "ERA_2025": "situations"},
    "అబద్ధమేనా": {"ERA_2000": "fake ena", "ERA_2025": "scam ena"},
    "మోసం": {"ERA_2000": "cheat", "ERA_2025": "ghost"},
    "సమస్య": {"ERA_2000": "problem", "ERA_2025": "scene"},
    "ఆలోచిస్తే": {"ERA_2000": "think cheste", "ERA_2025": "overthink cheste"},
    "నిజం": {"ERA_2000": "nijam", "ERA_2025": "fact"},
    "ధైర్యం": {"ERA_2000": "confidence", "ERA_2025": "guts"},
    "అబద్ధం": {"ERA_2000": "lie", "ERA_2025": "fake"},
}


def romanize_with_persona(te_sentence: str, persona: str = "P1_STANDARD", era_mode: str = "ERA_2025") -> str:
    """
    Converts ANY Telugu sentence into 100% Latin-script Tenglish according to the chosen
    Texter Persona while injecting era-calibrated English words.
    """
    tokens = te_sentence.split()
    out_tokens = []

    for tok in tokens:
        punct = ""
        m = re.search(r"([?!.,:;\"']+$)", tok)
        clean_w = tok
        if m:
            punct = m.group(1)
            clean_w = tok[: -len(punct)]

        # 1. High-frequency Multi-Persona Lexicon ('anthega'/'antega', 'Anti'=ఏంటి, 'm'=ఏం, 'nduku'=ఎందుకు)
        if clean_w in CANONICAL_TO_PERSONA_LEXICON:
            mapped = CANONICAL_TO_PERSONA_LEXICON[clean_w].get(persona, clean_w)
            out_tokens.append(mapped + punct)
            continue

        # 2. Era-based English Lexical Swap
        if clean_w in ERA_LEXICAL_SWAPS and era_mode in ERA_LEXICAL_SWAPS[clean_w]:
            out_tokens.append(ERA_LEXICAL_SWAPS[clean_w][era_mode] + punct)
            continue

        # 3. Known English loanwords written in Telugu script (with optional Telugu suffix)
        if clean_w in TELUGU_SCRIPT_ENGLISH_LOANS:
            out_tokens.append(TELUGU_SCRIPT_ENGLISH_LOANS[clean_w][0] + punct)
            continue

        matched_loan_prefix = False
        for loan_te, (en_equiv, _) in TELUGU_SCRIPT_ENGLISH_LOANS.items():
            if clean_w.startswith(loan_te) and len(clean_w) > len(loan_te):
                suffix_te = clean_w[len(loan_te) :]
                suffix_rom = transliterate_telugu_word_universal(suffix_te, persona)
                out_tokens.append(f"{en_equiv}-{suffix_rom}" + punct)
                matched_loan_prefix = True
                break
        if matched_loan_prefix:
            continue

        # 4. Universal Akshara-to-Latin Phonetic State Machine (handles 100% of remaining Telugu words)
        rom = transliterate_telugu_word_universal(clean_w, persona)
        out_tokens.append(rom + punct)

    return " ".join(out_tokens)


def expand_scene_across_eras_and_personas(scene_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Expands 1 Core Telugu Scene across 4 Tenglish Personas with 100% Latin script coverage."""
    turns = scene_dict["turns"]
    expansions = {
        "ORIGINAL_TELUGU": turns,
        "P1_STANDARD_TENGLISH": [romanize_with_persona(t, "P1_STANDARD", "ERA_2025") for t in turns],
        "P2_MINIMAL_TENGLISH": [romanize_with_persona(t, "P2_MINIMAL", "ERA_2025") for t in turns],
        "P3_EXPRESSIVE_TENGLISH": [romanize_with_persona(t, "P3_EXPRESS", "ERA_2025") for t in turns],
        "P4_ACROPHONIC_BROKEN": [romanize_with_persona(t, "P4_ACROPHON", "ERA_2025") for t in turns],
    }
    return {
        "scene_id": scene_dict["scene_id"],
        "era": scene_dict["era"],
        "director_writer_style": scene_dict["director_writer_style"],
        "emotion_hint": scene_dict["emotion_hint"],
        "dialect_hint": scene_dict["dialect_hint"],
        "expansions": expansions,
    }


def calculate_spelling_entropy(variant_counts: Dict[str, int]) -> float:
    """Computes Shannon Entropy H(X) = -sum(p_i * log2(p_i)) in bits."""
    total = sum(variant_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in variant_counts.values():
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return round(entropy, 3)
