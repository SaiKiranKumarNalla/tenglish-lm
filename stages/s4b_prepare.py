"""Stage 4b (part 2) — sentence pairs for the sentence-level normalizer.

    TLM_CORPUS=... DAKSHINA_TE=.../te python stages/s4b_prepare.py

English words are replaced by a placeholder '@' on BOTH sides, so the model learns to leave them alone;
at inference the language-ID model marks English tokens, and they are put back unchanged afterwards:
    "nuvvu exam ela rasav?"  ->  "nuvvu @ ela rasav?"  ->  "నువ్వు @ ఎలా రాసావ్?"  ->  "నువ్వు exam ఎలా రాసావ్?"

Writes data/normalizer/sent_{train,movie_test,dak_test}.tsv:
  sent_train       movie TRAIN turns (3 noisy spellings each) + Dakshina dev sentences (real human romanization)
  sent_movie_test  movie TEST turns, one noisy spelling each (synthetic, spoken)
  sent_dak_test    Dakshina test sentences (real human romanization, formal)
"""
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT, load_scenes, split_of  # noqa: E402
from tlm.filters import drop_reason  # noqa: E402
from tlm.romanizer import LOANWORDS, romanize, romanize_word  # noqa: E402

OUT = ROOT / "data" / "normalizer"
MAXLEN = 160
TOKEN = re.compile(r"[ఀ-౿]+|[^ఀ-౿]+")


def pair(line, rng):
    """(romanized with @ for English, Telugu with @ for English) for one native-script line."""
    src, tgt = [], []
    for t in TOKEN.findall(line):
        if t[0] >= "ఀ" and t[0] <= "౿":
            if t in LOANWORDS:
                src.append("@")
                tgt.append("@")
            else:
                src.append(romanize_word(t, rng, False, rng.choice([0.0, 0.0, 0.5, 1.5])) if rng else
                           romanize_word(t, canonical=True))
                tgt.append(t)
        else:
            src.append(t)
            tgt.append(t)
    return "".join(src).lower(), "".join(tgt)


def ok(s, t):
    return 3 <= len(s) <= MAXLEN and 1 <= len(t) <= MAXLEN and re.search(r"[a-z]", s)


scenes = [s for s in load_scenes() if drop_reason(s.lines) is None]
split = split_of()
rows = {"train": [], "movie_test": []}
for s in scenes:
    sp = split.get(s.movie)
    if sp not in ("train", "test"):
        continue
    for i, line in enumerate(s.lines):
        rng = random.Random(f"{s.scene_id}:{i}")
        reps = 3 if sp == "train" else 1
        for _ in range(reps):
            src, tgt = pair(line, rng)
            if ok(src, tgt):
                rows["train" if sp == "train" else "movie_test"].append((src, tgt))

D = Path(os.environ["DAKSHINA_TE"]).expanduser() / "romanized"


def dak(split_name):
    nat = open(D / f"te.romanized.rejoined.{split_name}.native.txt", encoding="utf-8").read().splitlines()
    rom = open(D / f"te.romanized.rejoined.{split_name}.roman.txt", encoding="utf-8").read().splitlines()
    return [(r.lower(), n) for r, n in zip(rom, nat) if ok(r.lower(), n)]


dak_dev = dak("dev")
rows["train"] += dak_dev * 3  # real human sentence spellings, up-weighted
random.Random(0).shuffle(rows["train"])
rows["dak_test"] = dak("test")
movie_test = rows["movie_test"]
rows["movie_test"] = random.Random(1).sample(movie_test, min(5000, len(movie_test)))
for k, v in rows.items():
    with open(OUT / f"sent_{k}.tsv", "w", encoding="utf-8") as f:
        f.writelines(f"{s}\t{t}\n" for s, t in v)
    print(f"sent_{k}: {len(v):,}")
