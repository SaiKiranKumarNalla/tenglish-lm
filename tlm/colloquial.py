"""Simple code (not AI): make translated, formal Telugu sound more like spoken Telugu, add regional flavour, and mix in
English words the way people really do.

Machine translation produces textbook Telugu ("నేను వెళ్తున్నాను", "క్షమించండి"). People say "నేను వెళ్తున్నా", "సారీ".
These rules cover common, safe patterns only; they don't make translated text fully natural.
"""
import random
import re

# Formal -> spoken (applied to every sentence)
SPOKEN = [
    (r"ఏమిటి", "ఏంటి"), (r"ఎందుకని", "ఎందుకు"), (r"అలాగే", "అలాగే"), (r"చాలా బాగుంది", "చాలా బాగుంది"),
    (r"(\S+)తున్నాను(?=[\s.,?!]|$)", r"\1తున్నా"),      # వెళ్తున్నాను -> వెళ్తున్నా
    (r"(\S+)తాను(?=[\s.,?!]|$)", r"\1తా"),               # చేస్తాను -> చేస్తా
    (r"(\S+)ాను(?=[\s.,?!]|$)", r"\1ా"),                 # చేశాను -> చేశా
    (r"ఉన్నాను(?=[\s.,?!]|$)", "ఉన్నా"), (r"లేదు(?=[\s.,?!]|$)", "లేదు"),
    (r"మీరు ఎలా ఉన్నారు", "ఎలా ఉన్నావ్"),
    (r"దయచేసి ", ""),                       # people rarely say "please" in Telugu words in chat
]

# English words people actually use in Telugu speech: formal Telugu -> English (written in Telugu script or Latin)
CODE_MIX = {
    "క్షమించండి": ["సారీ", "sorry"], "క్షమించు": ["సారీ", "sorry"], "ధన్యవాదాలు": ["థాంక్స్", "thanks"],
    "కార్యాలయం": ["ఆఫీస్", "office"], "కార్యాలయానికి": ["ఆఫీస్ కి", "office ki"], "పరీక్ష": ["ఎగ్జామ్", "exam"],
    "చలనచిత్రం": ["సినిమా", "movie"], "దూరవాణి": ["ఫోన్", "phone"], "ఫోన్": ["ఫోన్", "phone"], "సరే": ["సరే", "ok"],
    "ఉద్యోగం": ["జాబ్", "job"], "సమావేశం": ["మీటింగ్", "meeting"], "వైద్యుడు": ["డాక్టర్", "doctor"],
    "ఆసుపత్రి": ["హాస్పిటల్", "hospital"], "కళాశాల": ["కాలేజ్", "college"], "విందు": ["పార్టీ", "party"],
    "స్నేహితుడు": ["ఫ్రెండ్", "friend"], "స్నేహితులు": ["ఫ్రెండ్స్", "friends"], "సమయం": ["టైం", "time"],
}

# Regional flavour (a few well-known markers; applied to a share of conversations)
DIALECT = {
    "Telangana": [(r"ఏంటి", "ఏంది"), (r"(\S+)ాడు(?=[\s.,?!]|$)", r"\1ిండు"), (r"అలా(?=[\s.,?!]|$)", "గట్ల"), (r"ఇలా(?=[\s.,?!]|$)", "గిట్ల"),
                  (r"మళ్ళీ", "మల్ల")],
    "Rayalaseema": [(r"ఏంటి", "ఏందిరా"), (r"అమ్మా", "అమ్మీ")],
    "Coastal Andhra": [(r"అండి(?=[\s.,?!]|$)", "అండీ"), (r"ఏంటి", "ఏంటండీ")],
}


def spoken(text: str) -> str:
    for a, b in SPOKEN:
        text = re.sub(a, b, text)
    return text


def code_mix(text: str, rng: random.Random, rate: float = 0.7, latin_share: float = 0.4) -> str:
    def swap(m):
        w = m.group()
        if w in CODE_MIX and rng.random() < rate:
            te, en = CODE_MIX[w]
            return en if rng.random() < latin_share else te
        return w
    return re.sub(r"[ఀ-౿]+", swap, text)


def dialect(text: str, name: str) -> str:
    for a, b in DIALECT.get(name, []):
        text = re.sub(a, b, text)
    return text
