"""Is this Latin-script token English, or romanized Telugu?  (`exam` -> English, `rasav` -> Telugu)

A character n-gram Naive Bayes classifier: it learns which letter sequences are typical of each language
(English: "tion", "ck", "ght"; romanized Telugu: "aa", "ndi", "ppu", words ending in -u/-i/-a), then
scores a word by how well each language's n-gram statistics explain it.

Ambiguous by nature: `ante` is English and Telugu (అంటే). A word the Telugu corpus uses often is
treated as Telugu unless it is far more English-looking (see `telugu_prior`).
"""
import json
import math
from collections import Counter


def ngrams(w, n_max=4):
    w = f"<{w}>"
    return [w[i:i + n] for n in range(1, n_max + 1) for i in range(len(w) - n + 1)]


class LangID:
    def __init__(self, n_max=4, alpha=0.1):
        self.n_max, self.alpha = n_max, alpha
        self.counts, self.totals, self.vocab_size = {}, {}, 1
        self.telugu_freq = Counter()

    def fit(self, english_words, telugu_words, telugu_freq=None):
        for lang, words in (("en", english_words), ("te", telugu_words)):
            c = Counter(g for w in words for g in ngrams(w, self.n_max))
            self.counts[lang], self.totals[lang] = c, sum(c.values())
        self.vocab_size = len(set(self.counts["en"]) | set(self.counts["te"]))
        self.telugu_freq = Counter(telugu_freq or {})
        return self

    def score(self, w):
        """log P(word | English) - log P(word | Telugu), per n-gram. > 0 means English-looking."""
        gs = ngrams(w.lower(), self.n_max)
        s = 0.0
        for g in gs:
            pe = (self.counts["en"].get(g, 0) + self.alpha) / (self.totals["en"] + self.alpha * self.vocab_size)
            pt = (self.counts["te"].get(g, 0) + self.alpha) / (self.totals["te"] + self.alpha * self.vocab_size)
            s += math.log(pe) - math.log(pt)
        return s / len(gs)

    def telugu_prior(self, w):
        # frequent romanized-Telugu words need much stronger English evidence to flip
        return 0.5 * math.log1p(self.telugu_freq.get(w.lower(), 0))

    def is_english(self, w, threshold=0.0):
        return w.isalpha() and self.score(w) - self.telugu_prior(w) > threshold

    def save(self, path):
        json.dump({"n_max": self.n_max, "alpha": self.alpha, "counts": self.counts, "totals": self.totals,
                   "vocab_size": self.vocab_size, "telugu_freq": dict(self.telugu_freq.most_common(20000))},
                  open(path, "w"))

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        m = cls(d["n_max"], d["alpha"])
        m.counts = {k: Counter(v) for k, v in d["counts"].items()}
        m.totals, m.vocab_size, m.telugu_freq = d["totals"], d["vocab_size"], Counter(d["telugu_freq"])
        return m
