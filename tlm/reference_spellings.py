"""A SMALL, HAND-CURATED list of common real-world romanizations of frequent Telugu words.

This is not a dataset. It was assembled from commonly seen chat and comment spellings to sanity-check the
synthetic generators, and it is incomplete and biased toward frequent forms. A proper evaluation
would use human romanizations, e.g. Google's Dakshina dataset (Telugu romanized by native speakers);
see the roadmap.

Includes shorthand people really use (`nen`, `nuvv`, `enty`, `anthey`) that rule-based generators may never produce.
"""

REFERENCE = {
    "ఏంటి": ["enti", "yenti", "enty", "yenty"],
    "ఏమిటి": ["emiti", "yemiti"],
    "ఎలా": ["ela", "elaa", "yela"],
    "ఇంత": ["intha", "inta"],
    "అంతే": ["anthe", "ante", "anthey", "antey"],
    "లేదు": ["ledu", "ledhu"],
    "కాదు": ["kadu", "kaadu", "kaadhu", "kadhu"],
    "నువ్వు": ["nuvvu", "nuvu", "nuvv"],
    "నేను": ["nenu", "neenu", "nen"],
    "చెప్పు": ["cheppu", "chepu", "cheppu"],
    "చేసాను": ["chesanu", "chesaanu", "chesa"],
    "వెళ్ళు": ["vellu", "wellu"],
    "ఉంది": ["undi", "undhi"],
    "ఎందుకు": ["enduku", "yenduku", "endhuku"],
    "ఎక్కడ": ["ekkada", "yekkada", "ekada"],
    "ఇప్పుడు": ["ippudu", "ipudu"],
    "బాగుంది": ["bagundi", "baagundi", "bagundhi"],
    "తెలుసు": ["telusu", "thelusu"],
    "నాన్న": ["nanna", "naanna"],
    "చాలా": ["chala", "chaala"],
    "కదా": ["kada", "kadaa", "kadha"],
    "రా": ["ra", "raa"],
    "ఏం": ["em", "yem"],
    "వచ్చావా": ["vachava", "vachaava", "vacchava"],
    "చూడు": ["chudu", "choodu", "chuudu"],
}
