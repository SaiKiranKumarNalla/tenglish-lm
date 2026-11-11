"""Stage 4b (part 1) — token language ID: English vs romanized Telugu.

    python stages/s4b_langid.py [--english /usr/share/dict/words]

English examples: a standard English wordlist (90% train / 10% held out).
Telugu examples: romanized words from the TRAIN movies (canonical + noisy), excluding English loanwords.
Test: held-out English words, romanized words from the TEST movies, and English loanwords
that occur in the movies (`late`, `exam`, `phone`...). Writes data/normalizer/langid.json.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402
from tlm.langid import LangID  # noqa: E402
from tlm.plotstyle import INK2, SERIES, plt, save  # noqa: E402
from tlm.romanizer import LOANWORDS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--english", default="/usr/share/dict/words")
a = ap.parse_args()
rng = random.Random(0)

eng = sorted({w.strip().lower() for w in open(a.english) if re.fullmatch(r"[A-Za-z]{2,}", w.strip())})
rng.shuffle(eng)
eng_train, eng_test = eng[: int(0.9 * len(eng))], eng[int(0.9 * len(eng)):]
loans = set(LOANWORDS.values())
WORD = re.compile(r"[a-z]+")


def roman_words(split):
    c = Counter()
    for l in open(ROOT / "data" / "romanized" / f"{split}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        for t in [r["canon"], *r["noisy"]]:
            c.update(w for w in WORD.findall(t.lower()) if w not in loans)
    return c


te_train, te_test = roman_words("train"), roman_words("test")
model = LangID().fit(eng_train, list(te_train), telugu_freq=te_train)
model.save(ROOT / "data" / "normalizer" / "langid.json")

# Evaluate on types (each distinct word once), plus the loanwords people really mix in
te_eval = [w for w in te_test if len(w) >= 2]
loan_eval = sorted(loans)
res = {"English held-out (dictionary words)": sum(map(model.is_english, eng_test)) / len(eng_test),
       "Romanized Telugu, unseen movies": 1 - sum(map(model.is_english, te_eval)) / len(te_eval),
       "English loanwords used in the movies": sum(map(model.is_english, loan_eval)) / len(loan_eval)}
# frequency-weighted: how often a Telugu TOKEN in running text is wrongly called English
tok_err = sum(c for w, c in te_test.items() if model.is_english(w)) / sum(te_test.values())
false_en = Counter({w: c for w, c in te_test.items() if model.is_english(w)}).most_common(15)
missed = [w for w in loan_eval if not model.is_english(w)]

fig, ax = plt.subplots(figsize=(9, 3.2))
names = list(res)
ax.barh(range(len(names)), [res[n] for n in names], color=SERIES[0], height=0.55)
for i, n in enumerate(names):
    ax.text(res[n], i, f"  {res[n]:.1%}", va="center", color=INK2, fontsize=9)
ax.set_yticks(range(len(names)), names)
ax.invert_yaxis()
ax.set_xlim(0, 1.15)
ax.set_xlabel("correctly identified")
ax.set_title("Token language ID: English vs romanized Telugu (character n-gram Naive Bayes)")
save(fig, "s4b_langid")
out = {"accuracy": {k: round(v, 4) for k, v in res.items()}, "telugu_tokens_wrongly_english": round(tok_err, 4),
       "top_false_english": false_en, "missed_loanwords": missed}
json.dump(out, open(ROOT / "docs" / "results" / "s4b_langid.json", "w"), indent=1)
print(json.dumps(out, indent=1))
