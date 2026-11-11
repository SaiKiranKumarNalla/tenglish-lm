"""Stage 2 — which tokenizer suits romanized movie Telugu? BPE vs Unigram vs character (+ Sarvam's).

    python stages/s2_tokenizer_study.py [--sarvam path/to/sarvam tokenizer.json]

Needs data/romanized/*.jsonl from Stage 1. Trains tokenizers on the TRAIN movies only and
measures on TEST movies:
  1. fertility     tokens per word (lower = shorter sequences = faster, more context)
  2. consistency   do different spellings of the same word get similar tokens? (higher = more robust)
  3. vocab sweep   how fertility on noisy text changes with vocabulary size
Writes docs/plots/s2_*.png, docs/results/s2_*.{json,md}; tokenizer models go to data/tok/.
"""
import argparse
import itertools
import json
import re
import random
import sys
from pathlib import Path

import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402
from tlm.plotstyle import SERIES, plt, save  # noqa: E402
from tlm.romanizer import romanize_word  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sarvam", default=None, help="optional sarvam-2b tokenizer.json for a pretrained baseline")
ap.add_argument("--vocab", type=int, default=8000)
a = ap.parse_args()

R = ROOT / "data" / "romanized"
TOK = ROOT / "data" / "tok"
TOK.mkdir(parents=True, exist_ok=True)
load = lambda s: [json.loads(l) for l in open(R / f"{s}.jsonl", encoding="utf-8")]  # noqa: E731
train, test = load("train"), load("test")

# Training text: canonical + all noisy versions (the tokenizer must see messy spellings too)
latin_txt = TOK / "train_latin.txt"
latin_txt.write_text("\n".join(t for r in train for t in [r["canon"], *r["noisy"]]), encoding="utf-8")
te_txt = TOK / "train_te.txt"
te_txt.write_text("\n".join(r["te"] for r in train), encoding="utf-8")


def train_spm(text_file, kind, vocab, name):
    prefix = TOK / name
    spm.SentencePieceTrainer.train(input=str(text_file), model_prefix=str(prefix), vocab_size=vocab,
                                   model_type=kind, character_coverage=1.0, num_threads=4,
                                   hard_vocab_limit=False, minloglevel=2)
    return spm.SentencePieceProcessor(model_file=str(prefix) + ".model")


class Char:  # one token per character (spaces kept as a token)
    def encode(self, s, out_type=str):
        return list(s)


toks = {
    "BPE": (train_spm(latin_txt, "bpe", a.vocab, "bpe_latin"), train_spm(te_txt, "bpe", a.vocab, "bpe_te")),
    "Unigram": (train_spm(latin_txt, "unigram", a.vocab, "uni_latin"), train_spm(te_txt, "unigram", a.vocab, "uni_te")),
    "Char": (Char(), Char()),
}
if a.sarvam:
    from tokenizers import Tokenizer

    class Sarvam:
        t = Tokenizer.from_file(a.sarvam)

        def encode(self, s, out_type=str):
            return self.t.encode(s, add_special_tokens=False).tokens

    toks["Sarvam-2B (pretrained)"] = (Sarvam(), Sarvam())


def fertility(tok, texts):
    return sum(len(tok.encode(t, out_type=str)) for t in texts) / sum(len(t.split()) for t in texts)


# --- 1. Fertility -------------------------------------------------------------------------------
kinds = {"Telugu script": [r["te"] for r in test], "canonical Latin": [r["canon"] for r in test],
         "noisy Latin": [r["noisy"][0] for r in test]}
fert = {name: {k: round(fertility(te if k == "Telugu script" else lat, v), 2) for k, v in kinds.items()}
        for name, (lat, te) in toks.items()}
fig, ax = plt.subplots(figsize=(9, 3.8))
names, w = list(fert), 0.26
for j, k in enumerate(kinds):
    xs = [i + (j - 1) * (w + 0.02) for i in range(len(names))]
    ax.bar(xs, [fert[n][k] for n in names], width=w, color=SERIES[j], label=k)
    for x, n in zip(xs, names):
        ax.text(x, fert[n][k] + 0.08, f"{fert[n][k]:.1f}", ha="center", fontsize=8, color="#5f5e58")
ax.set_xticks(range(len(names)), names)
ax.set_ylabel("tokens per word (lower = better)")
ax.set_title(f"Fertility on unseen test movies (our tokenizers: vocab {a.vocab})")
ax.legend(loc="upper left")
save(fig, "s2_fertility")

# --- 2. Spelling consistency ----------------------------------------------------------------
# For frequent test words: make 8 spellings, tokenize each, average pairwise Jaccard overlap of the token sets.
from collections import Counter  # noqa: E402

te_words = Counter(w for r in test for w in r["te"].split() if any("ఀ" <= c <= "౿" for c in w))
probe = [w.strip(".,!?\"'") for w, _ in te_words.most_common(600)]
rng = random.Random(0)
variants = {w: list({romanize_word(w, rng, False, rng.choice([0, 0, .5, 1.5])) for _ in range(30)})[:8]
            for w in probe}
variants = {w: v for w, v in variants.items() if len(v) >= 2}


def consistency(tok):
    scores = []
    for v in variants.values():
        sets = [set(tok.encode(x, out_type=str)) for x in v]
        scores += [len(p & q) / len(p | q) for p, q in itertools.combinations(sets, 2)]
    return round(sum(scores) / len(scores), 3)


cons = {n: consistency(lat) for n, (lat, _) in toks.items()}
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.bar(list(cons), list(cons.values()), color=SERIES[0], width=0.55)
for i, (n, c) in enumerate(cons.items()):
    ax.text(i, c + 0.01, f"{c:.2f}", ha="center", fontsize=9, color="#5f5e58")
ax.set_ylim(0, 1)
ax.set_ylabel("token overlap between spellings")
ax.set_title(f"Spelling consistency across {len(variants)} words (higher = more robust)")
save(fig, "s2_spelling_consistency")

# --- 3. Vocab sweep: BPE vs Unigram on noisy Latin -----------------------------------------
# The trade-off: bigger vocab -> fewer tokens per word, but spellings stop sharing tokens.
sweep = {"BPE": {}, "Unigram": {}}
sweep_cons = {"BPE": {}, "Unigram": {}}
VOCABS = [500, 1000, 2000, 4000, 8000, 16000]
for v in VOCABS:
    for kind, label in [("bpe", "BPE"), ("unigram", "Unigram")]:
        t = train_spm(latin_txt, kind, v, f"sweep_{kind}_{v}")
        sweep[label][v] = round(fertility(t, kinds["noisy Latin"]), 3)
        sweep_cons[label][v] = consistency(t)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, data, ylab, title in [
        (axes[0], sweep, "tokens per word, noisy Latin", "Efficiency: bigger vocab = fewer tokens"),
        (axes[1], sweep_cons, "token overlap between spellings", "Robustness: bigger vocab = spellings drift apart")]:
    for j, (label, d) in enumerate(data.items()):
        ax.plot(list(d), list(d.values()), marker="o", markersize=6, color=SERIES[j], label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(VOCABS, [f"{v}" if v < 1000 else f"{v // 1000}k" for v in VOCABS])
    ax.set_xlabel("vocabulary size")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend()
axes[1].axhline(cons["Char"], color="#5f5e58", linewidth=1, linestyle="--")
axes[1].text(VOCABS[-1], cons["Char"] - 0.06, "char-level", ha="right", color="#5f5e58", fontsize=9)
save(fig, "s2_vocab_tradeoff")

# --- Examples: how each tokenizer cuts different spellings of the same word ------------------
md = ["| spelling | " + " | ".join(n for n in toks if n != "Char") + " |", "|---|" + "---|" * (len(toks) - 1)]
for s in ["enti", "yenti", "anti", "entra", "yentra", "chesanu", "chesaanu", "chesa", "intha", "inta"]:
    md.append(f"| `{s}` | " + " | ".join(" ".join(f"`{p}`" for p in lat.encode(s, out_type=str))
                                        for n, (lat, _) in toks.items() if n != "Char") + " |")
uni = toks["Unigram"][0]
md += ["", "Unigram can **sample** different segmentations of the same word (subword regularization):", ""]
md += [f"- `{' '.join(uni.encode('vellipothunnanu', out_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1))}`"
       for _ in range(4)]
(ROOT / "docs" / "results" / "s2_segmentations.md").write_text("\n".join(md) + "\n", encoding="utf-8")
json.dump({"fertility": fert, "consistency": cons, "vocab_sweep_fertility": sweep, "vocab_sweep_consistency": sweep_cons},
          open(ROOT / "docs" / "results" / "s2_tokenizers.json", "w"), indent=1)
print(json.dumps({"sweep_fertility": sweep, "sweep_consistency": sweep_cons}))

# ============================================================================================
# Part B — pre-tokenization transforms: phonetic keys and morphology-aware splitting
# Same Unigram tokenizer (vocab 4k), but the text is transformed first. Both metrics are
# measured per ORIGINAL word, so splitting suffixes is not allowed to hide extra tokens.
# ============================================================================================
from tlm.textnorm import _key, morph_split, phonetic_key  # noqa: E402

TRANSFORMS = {"none": lambda s: s, "phonetic": phonetic_key, "morph": morph_split,
              "phonetic+morph": lambda s: morph_split(phonetic_key(s))}
VB = 4000


def fert_t(tok, texts, f):
    return sum(len(tok.encode(f(t), out_type=str)) for t in texts) / sum(len(t.split()) for t in texts)


def cons_t(tok, f):
    scores = []
    for v in variants.values():
        sets = [set(tok.encode(f(x), out_type=str)) for x in v]
        scores += [len(p & q) / len(p | q) for p, q in itertools.combinations(sets, 2)]
    return sum(scores) / len(scores)


frontier = {}
for name, f in TRANSFORMS.items():
    txt = TOK / f"train_latin_{name}.txt"
    txt.write_text("\n".join(f(t) for r in train for t in [r["canon"], *r["noisy"]]), encoding="utf-8")
    t = train_spm(txt, "unigram", VB, f"uni_{name}_{VB}")
    frontier[f"Unigram 4k + {name}" if name != "none" else "Unigram 4k"] = (
        round(fert_t(t, kinds["noisy Latin"], f), 3), round(cons_t(t, f), 3))
for name in ["Char", "Sarvam-2B (pretrained)"]:
    if name in toks:
        frontier[name] = (fert[name]["noisy Latin"], cons[name])

# Cost of phonetic keys: how many genuinely different canonical words share a key?
canon_words = Counter(w for r in test for w in re.findall(r"[a-z]+", r["canon"].lower()))
common = [w for w, n in canon_words.most_common(5000)]
by_key = {}
for w in common:
    by_key.setdefault(_key(w), []).append(w)
collided = [ws for ws in by_key.values() if len(ws) > 1]
collision_rate = sum(len(ws) for ws in collided) / len(common)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]
for n, (x, y) in frontier.items():
    is_new = "+" in n
    ax.scatter(x, y, s=60, color=SERIES[1] if is_new else SERIES[0], zorder=3,
               edgecolor="#fcfcfb", linewidth=2)
    ax.annotate(n, (x, y), textcoords="offset points", xytext=(8, 7 if n.endswith("phonetic+morph") else (-12 if n.endswith("+ phonetic") else -3)), fontsize=8.5, color="#5f5e58")
ax.set_xlabel("tokens per original word, noisy Latin (lower = more efficient)")
ax.set_ylabel("token overlap between spellings (higher = more robust)")
ax.set_title("Efficiency vs robustness: transforms (orange) vs plain tokenizers")
ax.set_xlim(right=max(x for x, _ in frontier.values()) + 1.6)
ax = axes[1]
ax.barh(["share of top-5k words that\ncollide with another word"], [collision_rate], color=SERIES[1], height=0.25)
ax.set_xlim(0, max(0.3, collision_rate * 1.3))
ax.text(collision_rate + 0.005, 0, f"{collision_rate:.1%}", va="center", color="#5f5e58")
ax.set_title("Cost of phonetic keys")
ax.set_xlabel("share of words")
save(fig, "s2_transforms_frontier")

ex = sorted(collided, key=lambda ws: -sum(canon_words[w] for w in ws))[:12]
md = ["| phonetic key | different words that collapse to it |", "|---|---|"]
md += [f"| `{_key(ws[0])}` | {', '.join(f'`{w}`' for w in ws)} |" for ws in ex]
(ROOT / "docs" / "results" / "s2_phonetic_collisions.md").write_text("\n".join(md) + "\n", encoding="utf-8")
res = json.load(open(ROOT / "docs" / "results" / "s2_tokenizers.json"))
res.update({"transforms_frontier": frontier, "phonetic_collision_rate_top5k": round(collision_rate, 4)})
json.dump(res, open(ROOT / "docs" / "results" / "s2_tokenizers.json", "w"), indent=1)
print(json.dumps({"frontier": frontier, "collision_rate": collision_rate}, indent=1))
print("\n".join(md))
