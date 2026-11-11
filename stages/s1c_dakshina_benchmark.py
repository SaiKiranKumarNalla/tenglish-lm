"""Stage 1c — benchmark synthetic romanization against HUMAN romanizations (Google's Dakshina dataset).

    DAKSHINA_TE=/path/to/dakshina_dataset_v1.0/te python stages/s1c_dakshina_benchmark.py

Dakshina's Telugu lexicon lists ~30k native words, each romanized by several native speakers, with how
many annotators typed each spelling (అంకిత -> ankita x2, ankitha x2, amkita x1).

1. calibrate  estimate how often people choose each spelling option (aa vs a, th vs t, ...) from the TRAIN words
2. benchmark  on held-out TEST words, compare generators (tiered, per-word hand-set, per-word calibrated):
     recall     share of human spellings the generator produces (in 50 draws), weighted by annotator count
     attested   share of generated spellings that at least one human typed (a lower bound on precision)
     top-1      the generator's most frequent spelling equals the humans' most frequent
3. overlap    how much of the movie vocabulary Dakshina (Wikipedia words) covers at all
Writes docs/plots/s1c_*.png, docs/results/s1c_dakshina.json and tlm/romanizer_calibrated.json.
"""
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ["TLM_HANDSET"] = "1"  # import the hand-set weights; calibrated ones are applied explicitly below
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm import romanizer  # noqa: E402
from tlm.corpus import ROOT, load_scenes  # noqa: E402
from tlm.filters import TE_WORD  # noqa: E402
from tlm.plotstyle import INK2, SERIES, plt, save  # noqa: E402
from tlm.tiers import romanize_tiered  # noqa: E402

D = Path(os.environ.get("DAKSHINA_TE", "")).expanduser() / "lexicons"


def load_lex(name):
    by = defaultdict(Counter)
    for line in open(D / f"te.translit.sampled.{name}.tsv", encoding="utf-8"):
        n, r, c = line.rstrip("\n").split("\t")
        by[n][r.lower()] += int(c)
    return by


train, test = load_lex("train"), load_lex("test")

# ---------------- 1. Calibration: how often do humans pick each option? ---------------------
# For each feature, look only at words where the feature is unambiguous, and check the human spelling.
FEATURES = {  # name: (native condition, roman test for the canonical choice)
    "ā as aa": (lambda n: "ా" in n or "ఆ" in n, lambda r: "aa" in r),
    "ī as ee": (lambda n: "ీ" in n or "ఈ" in n, lambda r: "ee" in r),
    "ū as oo": (lambda n: "ూ" in n or "ఊ" in n, lambda r: "oo" in r),
    "ū as uu": (lambda n: "ూ" in n or "ఊ" in n, lambda r: "uu" in r),
    "ē as ee": (lambda n: "ే" in n, lambda r: "ee" in r),
    "త as th": (lambda n: "త" in n and not re.search("[థటఠ]", n), lambda r: "th" in r),
    "ద as dh": (lambda n: "ద" in n and not re.search("[ధడఢ]", n), lambda r: "dh" in r),
    "ధ as dh": (lambda n: "ధ" in n and not re.search("[దడఢ]", n), lambda r: "dh" in r),
    "భ as bh": (lambda n: "భ" in n, lambda r: "bh" in r),
    "వ as w": (lambda n: "వ" in n, lambda r: "w" in r),
    "శ/ష as sh": (lambda n: re.search("[శష]", n) and "స" not in n, lambda r: "sh" in r),
    "initial ఏ/ఎ as ye": (lambda n: n[:1] in ("ఏ", "ఎ"), lambda r: r.startswith("ye")),
    "doubled consonant kept": (lambda n: re.search(r"([క-హ])్\1", n) is not None,
                               lambda r: re.search(r"([bcdfghjklmnpqrstvwyz])\1", r) is not None),
    "ం as m (not before p/b/m)": (lambda n: re.search("ం[^పబభమఫ]", n) is not None,
                                  lambda r: re.search(r"m[^pbmaeiou]", r) is not None),
}
rates = {}
for name, (cond, test_fn) in FEATURES.items():
    hit = tot = 0
    for n, rs in train.items():
        if cond(n):
            for r, c in rs.items():
                tot += c
                hit += c * bool(test_fn(r))
    rates[name] = round(hit / tot, 3) if tot else None

# Turn the measured rates into romanizer weights (canonical option first, as in tlm/romanizer.py).
a, i_, u_, uu = rates["ā as aa"], rates["ī as ee"], rates["ū as oo"], rates["ū as uu"]
calibrated = {
    "MATRAS": {"ా": [["aa", a], ["a", 1 - a]], "ీ": [["ee", i_], ["i", 1 - i_]],
               "ూ": [["oo", u_], ["u", max(1 - u_ - uu, 0.01)], ["uu", max(uu, 0.01)]],
               "ే": [["e", 1 - rates["ē as ee"]], ["ee", rates["ē as ee"]]]},
    "VOWELS": {"ఆ": [["aa", a], ["a", 1 - a]], "ఈ": [["ee", i_], ["i", 1 - i_]], "ఊ": [["oo", u_], ["u", 1 - u_]],
               "ఏ": [["e", 1 - rates["initial ఏ/ఎ as ye"] - 0.05], ["ye", rates["initial ఏ/ఎ as ye"]], ["a", 0.05]],
               "ఎ": [["e", 1 - rates["initial ఏ/ఎ as ye"] - 0.05], ["ye", rates["initial ఏ/ఎ as ye"]], ["a", 0.05]]},
    "CONSONANTS": {"త": [["th", rates["త as th"]], ["t", 1 - rates["త as th"]]],
                   "ద": [["d", 1 - rates["ద as dh"]], ["dh", rates["ద as dh"]]],
                   "ధ": [["dh", rates["ధ as dh"]], ["d", 1 - rates["ధ as dh"]]],
                   "భ": [["bh", rates["భ as bh"]], ["b", 1 - rates["భ as bh"]]],
                   "వ": [["v", 1 - rates["వ as w"]], ["w", rates["వ as w"]]],
                   "శ": [["sh", rates["శ/ష as sh"]], ["s", 1 - rates["శ/ష as sh"]]],
                   "ష": [["sh", rates["శ/ష as sh"]], ["s", 1 - rates["శ/ష as sh"]]]},
}
(ROOT / "tlm" / "romanizer_calibrated.json").write_text(json.dumps(calibrated, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")

# ---------------- 2. Benchmark on held-out test words ---------------------------------------
HAND = {k: dict(getattr(romanizer, k)) for k in ("MATRAS", "VOWELS", "CONSONANTS")}


def set_weights(table):
    for k, d in table.items():
        getattr(romanizer, k).update({s: [tuple(o) for o in opts] for s, opts in d.items()})


def per_word(n, rng):
    return romanizer.romanize_word(n, rng, False, rng.choice([0.0, 0.0, 0.5, 1.5]))


def evaluate(gen, draws=50):
    rng = random.Random(0)
    rec = att = top = n_words = 0.0
    rec_w = tot_w = 0
    for n, human in test.items():
        made = Counter(gen(n, rng) for _ in range(draws))
        rec_w += sum(c for r, c in human.items() if r in made)
        tot_w += sum(human.values())
        att += sum(1 for s in made if s in human) / len(made)
        top += made.most_common(1)[0][0] == human.most_common(1)[0][0]
        n_words += 1
    return {"recall_weighted": round(rec_w / tot_w, 3), "attested": round(att / n_words, 3),
            "top1": round(top / n_words, 3)}


results = {}
set_weights(HAND)
results["tiered (per sentence)"] = evaluate(lambda n, rng: romanize_tiered(n, rng))
results["per-word, hand-set"] = evaluate(per_word)
set_weights(calibrated)
results["per-word, calibrated on Dakshina"] = evaluate(per_word)
set_weights(HAND)  # leave the module as it was

# ---------------- 3. Vocabulary overlap with the movie corpus -------------------------------
movie_words = Counter(w for s in load_scenes() for l in s.lines for w in TE_WORD.findall(l))
dak_words = set(train) | set(test) | set(load_lex("dev"))
top5k = [w for w, _ in movie_words.most_common(5000)]
overlap = {"movie_top5k_in_dakshina": round(sum(w in dak_words for w in top5k) / 5000, 3),
           "movie_tokens_in_dakshina": round(sum(c for w, c in movie_words.items() if w in dak_words)
                                             / sum(movie_words.values()), 3)}

# ---------------- plots ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3.8))
names, metrics = list(results), ["recall_weighted", "attested", "top1"]
labels = {"recall_weighted": "human spellings produced", "attested": "generated spellings a human typed",
          "top1": "most-common spelling matches"}
w = 0.26
for j, n in enumerate(names):
    xs = [k + (j - 1) * (w + 0.02) for k in range(len(metrics))]
    ax.bar(xs, [results[n][m] for m in metrics], width=w, color=SERIES[j], label=n)
    for x, m in zip(xs, metrics):
        ax.text(x, results[n][m] + 0.01, f"{results[n][m]:.0%}", ha="center", fontsize=8.5, color=INK2)
ax.set_xticks(range(len(metrics)), [labels[m] for m in metrics])
ax.set_ylim(0, 1.08)
ax.set_title(f"Synthetic vs human romanizations ({len(test):,} held-out Dakshina words)")
ax.legend(loc="upper right", fontsize=8.5)
save(fig, "s1c_dakshina_benchmark")

hand_rates = {"ā as aa": .5, "ī as ee": .55, "ū as oo": .5, "ū as uu": .08, "ē as ee": .3, "త as th": .6,
              "ద as dh": .2, "ధ as dh": .6, "భ as bh": .6, "వ as w": .35, "శ/ష as sh": .8,
              "initial ఏ/ఎ as ye": .35, "doubled consonant kept": .85, "ం as m (not before p/b/m)": .1}
feat = [f for f in rates if rates[f] is not None]
fig, ax = plt.subplots(figsize=(11.5, 4.4))
ys = range(len(feat))
ax.barh([y + 0.2 for y in ys], [hand_rates[f] for f in feat], height=0.38, color=SERIES[0], label="hand-set guess")
ax.barh([y - 0.2 for y in ys], [rates[f] for f in feat], height=0.38, color=SERIES[1], label="measured from humans")
ax.set_yticks(list(ys), [f.replace("ం", "anusvara").replace("ā", "long a").replace("ī", "long i")
                         .replace("ū", "long u").replace("ē", "long e").replace("త", "t-sound").replace("ద", "d-sound")
                         .replace("ధ", "dh-sound").replace("భ", "bh-sound").replace("వ", "v-sound")
                         .replace("శ/ష", "sh-sounds").replace("ఏ/ఎ", "e") for f in feat], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("probability of choosing the spelling")
ax.set_title("Spelling habits: our guesses vs what Dakshina annotators actually typed")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
save(fig, "s1c_calibration")

out = {"feature_rates_from_humans": rates, "benchmark_test_words": results, "vocab_overlap": overlap,
       "n_train_words": len(train), "n_test_words": len(test)}
json.dump(out, open(ROOT / "docs" / "results" / "s1c_dakshina.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
