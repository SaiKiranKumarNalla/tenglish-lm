"""Stage 4b (part 3) — sentence-level normalizer vs the Stage 4 word-level one.

    python stages/s4b_sentence_normalizer.py --word-model data/normalizer/normalizer.pt [--quick]

The sentence model reads a whole romanized sentence, so it can use context to resolve words that are spelled
the same but mean different things (`nee`/`ni`, `anti`). Both models are scored on:
  Dakshina test sentences  real human romanization (formal)
  Movie test sentences     synthetic noisy romanization of turns from the 8 unseen movies (spoken)
Metrics: word accuracy (words aligned by position), sentence character error rate, and accuracy on
AMBIGUOUS words (spellings that map to 2+ different Telugu words in the training pairs).
Finally a demo of the full pipeline with English detection on code-mixed sentences.
"""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402
from tlm.langid import LangID  # noqa: E402
from tlm.seq2seq import PAD, Seq2Seq, Vocab  # noqa: E402
from tlm.textnorm import _key  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--word-model", required=True)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--steps", type=int, default=12000)
ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "s4b_sentence.json"))
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
N = ROOT / "data" / "normalizer"
read = lambda f: [tuple(l.rstrip("\n").split("\t")) for l in open(N / f, encoding="utf-8")]  # noqa: E731
train, movie_test, dak_test = read("sent_train.tsv"), read("sent_movie_test.tsv"), read("sent_dak_test.tsv")
if a.quick:
    train, movie_test, dak_test, a.steps = train[:5000], movie_test[:200], dak_test[:200], 200
ML = 162

# ---- word model from Stage 4 -----------------------------------------------------------------
ck = torch.load(a.word_model, map_location=dev)
wsrc, wtgt = Vocab([]), Vocab([])
wsrc.chars, wtgt.chars = ck["src"], ck["tgt"]
wsrc.idx, wtgt.idx = {c: i + 3 for i, c in enumerate(wsrc.chars)}, {c: i + 3 for i, c in enumerate(wtgt.chars)}
word_model = Seq2Seq(len(wsrc), len(wtgt), d=256, layers=3, maxlen=40).to(dev)
word_model.load_state_dict(ck["model"])
word_model.eval()


def pad(xs):
    return torch.tensor([x + [PAD] * (max(map(len, xs)) - len(x)) for x in xs], device=dev)


def word_normalize(sents):
    """Stage 4 approach: normalize each romanized word on its own, keep everything else."""
    words = sorted({w for s in sents for w in re.findall(r"[a-z]+", s)})
    out = {}
    for i in range(0, len(words), 1024):
        chunk = words[i:i + 1024]
        ys = word_model.greedy(pad([wsrc.encode(w, 40) for w in chunk]))
        out.update({w: wtgt.decode(y.tolist()) for w, y in zip(chunk, ys)})
    return [re.sub(r"[a-z]+", lambda m: out[m.group()], s) for s in sents]


# ---- sentence model ------------------------------------------------------------------------
src_v = Vocab([s for s, _ in train + movie_test + dak_test])
tgt_v = Vocab([t for _, t in train + movie_test + dak_test])
model = Seq2Seq(len(src_v), len(tgt_v), d=128 if a.quick else 256, layers=2 if a.quick else 4, maxlen=ML).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-4, total_steps=a.steps, pct_start=0.05)
rng, t0 = random.Random(0), time.time()
for step in range(1, a.steps + 1):
    b = rng.sample(train, 64)
    src, tgt = pad([src_v.encode(s, ML) for s, _ in b]), pad([tgt_v.encode(t, ML) for _, t in b])
    logits = model(src, tgt[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=PAD,
                           label_smoothing=0.05)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 1000 == 0 or step == a.steps:
        print(f"sentence model step {step}/{a.steps} loss {loss.item():.3f}", flush=True)
model.eval()
train_min = round((time.time() - t0) / 60, 1)


def sent_normalize(sents):
    out = []
    for i in range(0, len(sents), 128):
        ys = model.greedy(pad([src_v.encode(s, ML) for s in sents[i:i + 128]]), maxlen=ML)
        out += [tgt_v.decode(y.tolist()) for y in ys]
    return out


# ---- metrics ---------------------------------------------------------------------------------
def words(s):
    return re.findall(r"[ఀ-౿]+|@", s)


def cer(p, g):
    d = list(range(len(g) + 1))
    for i, pc in enumerate(p, 1):
        prev, d[0] = d[0], i
        for j, gc in enumerate(g, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (pc != gc))
    return d[-1] / max(len(g), 1)


# ambiguous spellings: the same phonetic key maps to 2+ Telugu words, each reasonably common
amb_table = defaultdict(Counter)
for f in ("syn_train.tsv", "dak_train.tsv"):
    for r, n, *_ in read(f):
        amb_table[_key(r)][n] += 1
AMBIG = {k for k, c in amb_table.items() if len(c) >= 2 and c.most_common(2)[1][1] >= 0.2 * c.most_common(1)[0][1]}


def score(pairs, preds):
    hit = tot = amb_hit = amb_tot = 0
    for (src, gold), p in zip(pairs, preds):
        gw, pw, sw = words(gold), words(p), re.findall(r"[a-z]+|@", src)
        for i, g in enumerate(gw):
            ok = i < len(pw) and pw[i] == g
            hit += ok
            tot += 1
            if len(sw) == len(gw) and sw[i] != "@" and _key(sw[i]) in AMBIG:
                amb_hit += ok
                amb_tot += 1
    return {"word_acc": round(hit / tot, 4), "ambiguous_word_acc": round(amb_hit / max(amb_tot, 1), 4),
            "ambiguous_words": amb_tot,
            "sentence_cer": round(sum(cer(p, g) for (_, g), p in zip(pairs, preds)) / len(pairs), 4)}


results = {}
for name, pairs in [("Dakshina test (real human)", dak_test), ("Movie test (unseen movies)", movie_test)]:
    srcs = [s for s, _ in pairs]
    results[name] = {"word-level (Stage 4)": score(pairs, word_normalize(srcs)),
                     "sentence-level (Stage 4b)": score(pairs, sent_normalize(srcs))}
    print(name, json.dumps(results[name]), flush=True)

# ---- full pipeline demo: English detection + sentence normalizer ---------------------------
lid = LangID.load(N / "langid.json")
EN_CHAT = {"ok", "okay", "hi", "bro", "thanks", "sorry", "please", "plz", "yes", "no"}  # tiny override list


def pipeline(text):
    toks = re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)
    english = [t for t in toks if t.isalpha() and (t.lower() in EN_CHAT or lid.is_english(t))]
    masked = "".join("@" if (t.isalpha() and t in english) else t.lower() for t in toks)
    out = sent_normalize([masked])[0]
    it = iter(english)
    return masked, re.sub("@", lambda m: next(it, "@"), out)


demo_in = ["enti ra intha late ayyindi?", "anti ra inta lete?", "bro exam ela rasav, results eppudu?",
           "nee phone number ivvu", "ni peru enti?", "nenu office ki vellanu, meeting undi",
           "sorry ra, traffic lo stuck ayya", "amma ki call chey, worry avthundi"]
results["demo"] = [dict(zip(["input", "masked", "output"], (s, *pipeline(s)))) for s in demo_in]
results["train_minutes"] = train_min
results["params"] = model.n_params()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(a.out, "w"), indent=1, ensure_ascii=False)
print(json.dumps(results["demo"], indent=1, ensure_ascii=False))
print("DONE")
