"""Stage 1b — which way of generating synthetic romanized text is best?

    TLM_CORPUS=... python stages/s1b_generator_comparison.py

Compares three generators (tlm/tiers.py): tiered per sentence, per-word (ours), hybrid.
  1. diversity        distinct spellings per frequent word
  2. mixing           share of sentences that mix careful and lazy spellings (real people do this)
  3. coverage         share of a small curated list of real spellings the generator can produce
  4. cross-robustness a tokenizer trained on generator A's text, tested on generator B's text:
                      does training on A prepare a model for spellings it has never seen?
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT, load_scenes, split_of  # noqa: E402
from tlm.filters import TE_WORD, drop_reason  # noqa: E402
from tlm.plotstyle import INK2, SERIES, plt, save  # noqa: E402
from tlm.reference_spellings import REFERENCE  # noqa: E402
from tlm.romanizer import romanize, romanize_word  # noqa: E402
from tlm.tiers import GENERATORS  # noqa: E402

scenes = [s for s in load_scenes() if drop_reason(s.lines) is None]
split = split_of()
train_lines = [l for s in scenes if split.get(s.movie) == "train" for l in s.lines]
test_lines = [l for s in scenes if split.get(s.movie) == "test" for l in s.lines]
words = Counter(w for l in train_lines for w in TE_WORD.findall(l))
top = [w for w, _ in words.most_common(2000)]
names = list(GENERATORS)
res = {n: {} for n in names}

for n, g in GENERATORS.items():
    rng = random.Random(0)
    # 1. diversity: the generator is sentence-level, so feed single words as one-word sentences
    res[n]["spellings_per_word"] = round(sum(len({g(w, rng) for _ in range(100)}) for w in top) / len(top), 2)
    # 2. mixing: among words with a long vowel, are some typed long and some short in the same sentence?
    mixed = total = 0
    for l in random.Random(1).sample(train_lines, 3000):
        out = g(l, rng).split()
        canon = romanize(l, canonical=True).split()
        flags = [o == c for o, c in zip(out, canon) if any(v in c for v in ("aa", "ee", "oo", "th", "dh"))]
        if len(flags) >= 2:
            total += 1
            mixed += 0 < sum(flags) < len(flags)
    res[n]["mixing_share"] = round(mixed / total, 3)
    # 3. coverage of curated real spellings
    hit = tot = 0
    for te, real in REFERENCE.items():
        made = {g(te, rng) for _ in range(300)}
        hit += sum(r in made for r in set(real))
        tot += len(set(real))
    res[n]["curated_coverage"] = round(hit / tot, 3)

# 4. cross-generator robustness matrix (plus a canonical-only baseline row)
TOK = ROOT / "data" / "tok"
TOK.mkdir(parents=True, exist_ok=True)
rows = ["canonical only"] + names


def text_for(name, lines, seed, copies):
    rng = random.Random(seed)
    if name == "canonical only":
        return [romanize(l, canonical=True) for l in lines]
    return [GENERATORS[name](l, rng) for l in lines for _ in range(copies)]


test_text = {n: text_for(n, test_lines, 99, 1) for n in names}
matrix = {}
for r in rows:
    f = TOK / f"gen_{r.split()[0]}.txt"
    f.write_text("\n".join(text_for(r, train_lines, 7, 2)), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(f), model_prefix=str(TOK / f"gen_{r.split()[0]}"), vocab_size=4000,
                                   model_type="unigram", character_coverage=1.0, minloglevel=2, num_threads=4)
    sp = spm.SentencePieceProcessor(model_file=str(TOK / f"gen_{r.split()[0]}.model"))
    matrix[r] = {c: round(sum(len(sp.encode(t)) for t in test_text[c]) / sum(len(t.split()) for t in test_text[c]), 3)
                 for c in names}

# --- plots ------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, key, title, fmt in [(axes[0], "spellings_per_word", "Distinct spellings per word", "{:.1f}"),
                            (axes[1], "mixing_share", "Sentences mixing careful + lazy spellings", "{:.0%}"),
                            (axes[2], "curated_coverage", "Curated real spellings it can produce", "{:.0%}")]:
    vals = [res[n][key] for n in names]
    ax.bar(range(len(names)), vals, color=[SERIES[i] for i in range(len(names))], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(range(len(names)), [n.replace(" (", "\n(") for n in names], fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(vals) * 1.25)
save(fig, "s1b_generator_properties")

fig, ax = plt.subplots(figsize=(7.5, 3.8))
import numpy as np  # noqa: E402

M = np.array([[matrix[r][c] for c in names] for r in rows])
im = ax.imshow(M, cmap="Blues_r", aspect="auto")
for i in range(len(rows)):
    for j in range(len(names)):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10,
                color="#fcfcfb" if M[i, j] < M.mean() else "#0b0b0b")
ax.set_xticks(range(len(names)), [f"test:\n{n}" for n in names], fontsize=9)
ax.set_yticks(range(len(rows)), [f"train: {r}" for r in rows], fontsize=9)
ax.grid(False)
ax.set_title("Tokens per word on another generator's spellings (lower = better)", fontsize=10.5)
fig.colorbar(im, ax=ax, shrink=0.8)
save(fig, "s1b_cross_generator")

# Which curated spellings does nobody produce? (honest list of blind spots)
rng = random.Random(3)
made_any = {te: set().union(*({g(te, rng) for _ in range(300)} for g in GENERATORS.values())) for te in REFERENCE}
missing = {te: [r for r in real if r not in made_any[te]] for te, real in REFERENCE.items()}
missing = {k: v for k, v in missing.items() if v}
json.dump({"generators": res, "cross_generator_tokens_per_word": matrix, "never_generated": missing},
          open(ROOT / "docs" / "results" / "s1b_generators.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({"generators": res, "matrix": matrix, "never_generated": missing}, indent=1, ensure_ascii=False))
