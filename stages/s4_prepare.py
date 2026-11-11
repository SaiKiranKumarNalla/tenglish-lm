"""Stage 4a — word pairs for the normalizer: (romanized spelling -> Telugu script).

    TLM_CORPUS=... DAKSHINA_TE=.../dakshina_dataset_v1.0/te python stages/s4_prepare.py

Writes data/normalizer/*.tsv (kept out of git):
  dak_train / dak_test   real human spellings from Google's Dakshina lexicon (formal, Wikipedia vocabulary)
  syn_train              synthetic noisy spellings of words from the TRAIN movies (spoken vocabulary)
  syn_test               synthetic noisy spellings of words from the TEST movies, split into words the
                         normalizer has seen in training vs never seen (true generalization)
"""
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT, load_scenes, split_of  # noqa: E402
from tlm.filters import TE_WORD, drop_reason  # noqa: E402
from tlm.romanizer import romanize, romanize_word  # noqa: E402

OUT = ROOT / "data" / "normalizer"
OUT.mkdir(parents=True, exist_ok=True)
D = Path(os.environ["DAKSHINA_TE"]).expanduser() / "lexicons"
VALID = re.compile(r"^[a-z]+$")


def write(name, pairs):
    with open(OUT / f"{name}.tsv", "w", encoding="utf-8") as f:
        for r, n, *extra in pairs:
            f.write("\t".join([r, n, *map(str, extra)]) + "\n")
    print(f"{name}: {len(pairs):,} pairs")


# Dakshina: one row per (spelling, word), annotator count kept for weighting
for split in ["train", "test"]:
    pairs = []
    for line in open(D / f"te.translit.sampled.{split}.tsv", encoding="utf-8"):
        n, r, c = line.rstrip("\n").split("\t")
        r = r.lower()
        if VALID.match(r):
            pairs.append((r, n, c))
    write(f"dak_{split}", pairs)

# Synthetic spoken vocabulary from the movies
scenes = [s for s in load_scenes() if drop_reason(s.lines) is None]
split = split_of()
vocab = {k: Counter() for k in ("train", "test")}
for s in scenes:
    sp = split.get(s.movie)
    if sp in vocab:
        for l in s.lines:
            vocab[sp].update(w for w in TE_WORD.findall(l) if len(w) >= 2)


def spellings(word, k, seed):
    rng = random.Random(seed)
    out = {romanize_word(word, canonical=True)}
    for _ in range(k * 4):
        out.add(romanize(word, rng))
        if len(out) >= k + 1:
            break
    return [o for o in out if VALID.match(o)]


rng = random.Random(0)
train_words = [w for w, c in vocab["train"].items()]
syn_train = [(r, w, 1) for i, w in enumerate(train_words) for r in spellings(w, 3, i)]
rng.shuffle(syn_train)
write("syn_train", syn_train)

test_words = [w for w, c in vocab["test"].most_common() if c >= 2]  # skip one-off ASR noise
seen = set(train_words)
sample = rng.sample(test_words, min(3000, len(test_words)))
syn_test = [(r, w, "seen" if w in seen else "unseen") for i, w in enumerate(sample)
            for r in spellings(w, 2, 10_000 + i)[1:]]  # noisy spellings only (skip canonical)
write("syn_test", syn_test)
json.dump({"train_words": len(train_words), "test_words_sampled": len(sample),
           "test_unseen_share": round(sum(w not in seen for w in sample) / len(sample), 3)},
          open(OUT / "stats.json", "w"), indent=1)
print(open(OUT / "stats.json").read())
