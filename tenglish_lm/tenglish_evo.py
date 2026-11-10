"""
Milestone 2: Tenglish-Evo Engine
(Era-Calibrated English % Multiplier + Broken/Acrophonic Tenglish Permutation Engine)

Solves:
1. Phonetic Persona Permutations:
   - Standard Phonetic ('anthega', 'chepthunna', 'neeku')
   - Fast Minimalist ('antega', 'ceptuna', 'niku')
   - Expressive Elongated ('antheygaaa', 'avunaaa', 'anthe ga')
   - Acrophonic & English-Sound Interference ('Anti'=ఏంటి, 'm'=ఏం, 'nduku'=ఎందుకు, 'la'=ఎలా, 'chappu'=చెప్పు, 'nv'=నువ్వు)
2. 4-Way Word-Level Language & Morphology Tagger:
   - EN (Pure English: 'bro', 'scene', 'vibe', 'matter')
   - TE (Telugu Unicode or Roman Telugu: 'అంతేగా', 'anthega', 'avvatledu')
   - HYBRID_FUSED (English root + Telugu suffix: 'office-ki', 'delulu-lo', 'party-ki', 'emotion-tho')
   - LIGHT_VERB (English verb + cheyu/avvu: 'ghost chesindu', 'connect ayindi', 'vibe avvatledu')
3. Code-Mixing Index (CMI %) & Era-Calibrated Synthetic Expansion:
   - Expands a single Core Telugu movie/literature scene into:
     * Era 1 (CMI ~0-5%): Classic / Literary Telugu
     * Era 2 (CMI ~18-25%): 2000s Urban Cinema Mix (Nouns + Light Verbs)
     * Era 3 (CMI ~40-55%): 2025 Gen-Z & Meme Tenglish across all 4 Texter Personas
"""

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# 1. Lexicons for English, Telugu Loanwords, Hybrid Suffixes & Acrophonic Map
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
}

# Telugu Unicode Loanwords (English words written in Telugu script in movie subtitles)
TELUGU_SCRIPT_ENGLISH_LOANS = {
    "కాలేజీ": ("college", "HYBRID_FUSED"),
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
}

TELUGU_ROMAN_SUFFIXES = (
    "ki", "lo", "tho", "ni", "kosam", "nundi", "nudchi", "meeda", "la", "laga", "ga", "ra", "lu"
)

# ---------------------------------------------------------------------------
# 2. Multi-Persona & Acrophonic Tenglish Dictionary (Core Telugu -> 4 Personas)
# ---------------------------------------------------------------------------
# Personas:
#   - P1_STANDARD: Standard Phonetic ('anthega', 'yenti', 'endhuku', 'chepthunna')
#   - P2_MINIMAL:  Fast Minimalist ('antega', 'enti', 'enduku', 'cheptuna')
#   - P3_EXPRESS:  Expressive / Spaced ('antheygaaa', 'yentii', 'anthe ga', 'avunaaa')
#   - P4_ACROPHON: English-Letter Acrophony & Vowel Shift ('Anti'=ఏంటి, 'm'=ఏం, 'nduku'=ఎందుకు, 'chappu'=చెప్పు, 'nv'=నువ్వు)

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
}

# Build reverse lookup for disambiguation & normalization
REVERSE_VARIANT_TO_CANONICAL: Dict[str, str] = {}
for canonical_te, p_map in CANONICAL_TO_PERSONA_LEXICON.items():
    for _, variant in p_map.items():
        REVERSE_VARIANT_TO_CANONICAL[variant.lower()] = canonical_te

# Additional common variants for reverse lookup
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


# ---------------------------------------------------------------------------
# 3. Context-Aware Homograph Disambiguator ('Anti' = ఏంటి vs ఆంటీ vs anti-)
# ---------------------------------------------------------------------------

def disambiguate_homograph(word: str, prev_word: str = "", next_word: str = "") -> Dict[str, Any]:
    """
    Resolves cross-language homographs such as 'Anti'/'anti', 'm', 'la', 'pattu'.
    """
    w_low = word.strip("?!.,:;").lower()
    p_low = prev_word.strip("?!.,:;").lower()
    n_low = next_word.strip("?!.,:;").lower()

    if w_low == "anti":
        # Rule 1: Possessive prefix ('maa anti', 'pakkinti anti') -> ఆంటీ (Aunty)
        if p_low in {"maa", "ma", "mee", "mi", "pakkinti", "colony"} or n_low in {"vachindi", "cheppindi", "intlo"}:
            return {
                "surface": word,
                "resolved_te": "ఆంటీ",
                "meaning": "Aunty (Loanword Noun)",
                "confidence": 0.94,
                "escalated_to_archon": True,
            }
        # Rule 2: Followed by English noun ('anti social', 'anti hero', 'anti national') -> English prefix
        if n_low in {"social", "hero", "national", "virus", "corruption", "biotic"}:
            return {
                "surface": word,
                "resolved_te": "యాంటీ (anti-)",
                "meaning": "English prefix 'anti-'",
                "confidence": 0.96,
                "escalated_to_archon": True,
            }
        # Rule 3: Default Gen-Z Tenglish question acrophony ('Anti bro', 'idi anti?') -> ఏంటి (What)
        return {
            "surface": word,
            "resolved_te": "ఏంటి",
            "meaning": "What (Gen-Z Acrophonic 'A' + 'nti')",
            "confidence": 0.95,
            "escalated_to_archon": True,
        }

    if w_low == "m":
        if n_low in {"chestunnav", "chstnv", "aindhi", "aindi", "cheppu", "chappu", "ledhu", "ledu"} or p_low in {"asalu", "aslu", "bro"}:
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


# ---------------------------------------------------------------------------
# 4. 4-Way Token Classifier & Code-Mixing Index (CMI %) Calculator
# ---------------------------------------------------------------------------

def classify_token(token: str) -> Tuple[str, str]:
    """
    Returns (category, details) where category is one of:
    - 'EN': Pure English word
    - 'HYBRID_FUSED': English root + Telugu suffix ('office-ki', 'party-ki')
    - 'LIGHT_VERB': English verb stem used with Telugu auxiliary ('connect', 'ghost', 'vibe')
    - 'TE': Pure Telugu word (in Telugu Unicode or Latin Tenglish)
    """
    clean_tok = re.sub(r"[^\w\u0c00-\u0c7f-]", "", token)
    if not clean_tok:
        return ("PUNCT", "")

    # Check Telugu script words against known English loanwords written in Telugu script
    if clean_tok in TELUGU_SCRIPT_ENGLISH_LOANS:
        en_equiv, tag_type = TELUGU_SCRIPT_ENGLISH_LOANS[clean_tok]
        return (tag_type, f"{clean_tok} -> {en_equiv}")

    # Check if Telugu Unicode word has a Telugu suffix attached to an English loan stem
    for loan_te, (en_equiv, tag_type) in TELUGU_SCRIPT_ENGLISH_LOANS.items():
        if clean_tok.startswith(loan_te) and len(clean_tok) > len(loan_te):
            suffix = clean_tok[len(loan_te) :]
            return ("HYBRID_FUSED", f"{en_equiv}+{suffix}")

    low_tok = clean_tok.lower()

    # Check hyphenated hybrid ('office-ki', 'delulu-lo')
    if "-" in low_tok:
        parts = low_tok.split("-", 1)
        if parts[0] in ENGLISH_CORE_WORDS and parts[1] in TELUGU_ROMAN_SUFFIXES:
            return ("HYBRID_FUSED", f"{parts[0]}+-{parts[1]}")

    # Check fused Roman hybrid ('officeki', 'partylo', 'friendstho')
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
    """
    Computes Code-Mixing Index (CMI %) and counts of EN, TE, HYBRID_FUSED, LIGHT_VERB tokens.
    """
    raw_tokens = [t for t in text.split() if re.sub(r"[^\w\u0c00-\u0c7f]", "", t)]
    counts = Counter()
    tagged_tokens = []

    for idx, tok in enumerate(raw_tokens):
        cat, detail = classify_token(tok)
        if cat == "PUNCT":
            continue
        counts[cat] += 1
        tagged_tokens.append({"token": tok, "tag": cat, "detail": detail})

    total = sum(counts.values())
    if total == 0:
        return {"cmi_pct": 0.0, "en_ratio_pct": 0.0, "counts": dict(counts), "tagged_tokens": []}

    en_total = counts["EN"] + counts["HYBRID_FUSED"] + counts["LIGHT_VERB"]
    te_total = counts["TE"]
    en_ratio_pct = round(100.0 * en_total / total, 2)
    # Gambäck & Das CMI formula + direct English lexeme percentage
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


# ---------------------------------------------------------------------------
# 5. Synthetic Era-Calibrated Tenglish Multiplier (Core Telugu -> 3 Eras x 4 Personas)
# ---------------------------------------------------------------------------

ERA_LEXICAL_SWAPS = {
    "виషయం": {"ERA_2000": "matter", "ERA_2025": "matter"},
    "విషయం": {"ERA_2000": "matter", "ERA_2025": "topic"},
    "స్నేహం": {"ERA_2000": "friendship", "ERA_2025": "bonding"},
    "ప్రయత్నించాను": {"ERA_2000": "try chesanu", "ERA_2025": "try chesna"},
    "కోపం": {"ERA_2000": "angry", "ERA_2025": "trigger"},
    "పరిస్థితులు": {"ERA_2000": "situations", "ERA_2025": "situations"},
    "అబద్ధమేనా": {"ERA_2000": "fake ena", "ERA_2025": "scam ena"},
    "మోసం": {"ERA_2000": "cheat", "ERA_2025": "ghost"},
    "సమస్య": {"ERA_2000": "problem", "ERA_2025": "scene"},
    "ఆలోచిస్తే": {"ERA_2000": "think cheste", "ERA_2025": "overthink cheste"},
}

SIMPLE_TELUGU_TO_ROMAN_MAP = {
    "మనిషికి": "manishiki",
    "మాటే": "maate",
    "ప్రాణం": "pranam",
    "అబ్బాయ్": "abbay",
    "ఆ": "aa",
    "మాట": "maata",
    "తప్పితే": "thappithe",
    "ఇక": "ika",
    "విలువ": "viluva",
    "ఏముంటుంది": "emuntundi",
    "నిజం": "nijam",
    "చెప్పాలంటే": "cheppalante",
    "తన": "thana",
    "మీద": "meeda",
    "లేదు": "ledhu",
    "రా": "ra",
    "కానీ": "kani",
    "తనని": "thanani",
    "మర్చిపోవడం": "marchipovadam",
    "మాత్రం": "matram",
    "నా": "na",
    "వల్ల": "valla",
    "బ్రో": "bro",
    "నిన్న": "ninna",
    "రాత్రి": "rathri",
    "పార్టీకి": "party-ki",
    "పిలిచి": "pilichi",
    "సంబంధం": "sambandham",
    "అంటావ్": "antav",
    "అరే": "arey",
    "భయ్యా": "bhayya",
    "అక్కడ": "akkada",
    "సిట్యుయేషన్": "situation",
    "మొత్తం": "motham",
    "కంట్రోల్": "control",
    "తప్పింది": "thappindi",
    "నాకే": "nake",
    "ఫుల్": "full",
    "ట్రామా": "trauma",
    "అయిపోయింది": "aipoyindi",
    "నువ్వు": "nuvvu",
    "ఇట్లా": "itla",
    "ప్రతిసారి": "prathisari",
    "సీన్": "scene",
    "క్రియేట్": "create",
    "చేసి": "chesi",
    "తోసేస్తే": "thoseste",
    "లైఫ్": "life",
    "డ్యామేజ్": "damage",
    "అయితది": "aitadi",
    "వీకెండ్": "weekend",
    "ట్రిప్": "trip",
    "కి": "ki",
    "వస్తా": "vastha",
    "అని": "ani",
    "బిల్డప్": "buildup",
    "ఇచ్చి": "ichi",
    "లాస్ట్": "last",
    "మినిట్": "minute",
    "లో": "lo",
    "హ్యాండ్": "hand",
    "ఇచ్చావ్": "ichav",
    "మా": "ma",
    "మేనేజర్": "manager",
    "గాడు": "gadu",
    "ఫ్రైడే": "friday",
    "ఈవినింగ్": "evening",
    "క్లయింట్": "client",
    "కాల్": "call",
    "పెట్టి": "petti",
    "మైండ్": "mind",
    "బ్లాక్": "block",
    "చేసిండు": "chesindu",
    "నేనేం": "nenem",
    "చేయను": "cheyanu",
    "ఇదే": "ide",
    "సాకు": "saaku",
    "ఇంక": "inka",
    "నీతో": "neetho",
    "ప్లాన్": "plan",
    "చేయడం": "cheyadam",
    "కంటే": "kante",
    "ఇంట్లో": "intlo",
    "పడుకోవడం": "padukovadam",
    "బెటర్": "better",
}


def romanize_with_persona(te_sentence: str, persona: str = "P1_STANDARD", era_mode: str = "ERA_2025") -> str:
    """
    Converts a Telugu sentence into a specific Latin Tenglish Persona while injecting
    era-calibrated English words.
    """
    tokens = te_sentence.split()
    out_tokens = []

    for tok in tokens:
        punct = ""
        m = re.search(r"([?!.,:;]+)$", tok)
        clean_w = tok
        if m:
            punct = m.group(1)
            clean_w = tok[: -len(punct)]

        # 1. Check if word is in our multi-persona lexicon (antega/anthega, Anti, m, nduku, chappu)
        if clean_w in CANONICAL_TO_PERSONA_LEXICON:
            mapped = CANONICAL_TO_PERSONA_LEXICON[clean_w].get(persona, clean_w)
            out_tokens.append(mapped + punct)
            continue

        # 2. Check if word is eligible for Era English Lexical Swap
        if clean_w in ERA_LEXICAL_SWAPS and era_mode in ERA_LEXICAL_SWAPS[clean_w]:
            out_tokens.append(ERA_LEXICAL_SWAPS[clean_w][era_mode] + punct)
            continue

        # 3. Check if Telugu script word is an English loanword
        if clean_w in TELUGU_SCRIPT_ENGLISH_LOANS:
            out_tokens.append(TELUGU_SCRIPT_ENGLISH_LOANS[clean_w][0] + punct)
            continue

        # 4. Standard word romanization + persona phonetic rules
        base_rom = SIMPLE_TELUGU_TO_ROMAN_MAP.get(clean_w, clean_w)
        if persona == "P2_MINIMAL":
            base_rom = base_rom.replace("th", "t").replace("dh", "d").replace("ee", "i").replace("oo", "u")
        elif persona == "P3_EXPRESS" and base_rom.endswith(("a", "i", "u")):
            base_rom = base_rom + base_rom[-1]
        elif persona == "P4_ACROPHON":
            base_rom = (
                base_rom.replace("th", "t")
                .replace("chep", "chap")
                .replace("telu", "talu")
                .replace("nuvvu", "nv")
            )

        out_tokens.append(base_rom + punct)

    return " ".join(out_tokens)


def expand_scene_across_eras_and_personas(scene_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes 1 Core Telugu Scene and multiplies it into:
    - Original Telugu Script turns
    - Persona 1 (Standard Phonetic Latin Tenglish)
    - Persona 2 (Minimalist Speed-Texter Latin Tenglish)
    - Persona 3 (Expressive Elongated Latin Tenglish)
    - Persona 4 (Acrophonic Broken-Tenglish: 'Anti', 'm', 'nduku', 'chappu', 'nv')
    """
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
    """Computes Shannon Entropy H(X) = -sum(p_i * log2(p_i)) in bits for a word's Latin spellings."""
    total = sum(variant_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in variant_counts.values():
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return round(entropy, 3)
