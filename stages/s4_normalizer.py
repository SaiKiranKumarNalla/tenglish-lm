"""Stage 4 — a spelling normalizer trained from scratch: any romanized spelling -> Telugu script.

    python stages/s4_normalizer.py [--quick]     # GPU recommended (~5-10 min per configuration on a T4)

Needs data/normalizer/*.tsv from stages/s4_prepare.py. Trains the same character seq2seq model
(tlm/seq2seq.py) on three data mixes and tests every model on the same two test sets:
  Dakshina test   real human spellings, formal (Wikipedia) words never seen in training
  Movie test      synthetic spellings of words from the 8 unseen movies (spoken vocabulary),
                  split into words seen in the training movies vs never seen
Metric: word accuracy (exact Telugu output) and character error rate (CER).
Baseline: a lookup table from phonetic key -> most frequent Telugu word in the training pairs.
"""
import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402
from tlm.seq2seq import PAD, Seq2Seq, Vocab  # noqa: E402
from tlm.textnorm import _key  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--steps", type=int, default=8000)
ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "s4_normalizer.json"))
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
N = ROOT / "data" / "normalizer"
read = lambda f: [l.rstrip("\n").split("\t") for l in open(N / f, encoding="utf-8")]  # noqa: E731
dak_train, dak_test = read("dak_train.tsv"), read("dak_test.tsv")
syn_train, syn_test = read("syn_train.tsv"), read("syn_test.tsv")
if a.quick:
    dak_train, syn_train, dak_test, syn_test = dak_train[:3000], syn_train[:3000], dak_test[:300], syn_test[:300]
    a.steps = 200
MIXES = {"Dakshina only": dak_train, "Synthetic only": syn_train, "Dakshina + synthetic": dak_train + syn_train}
TESTS = {"Dakshina test (real human spellings)": [(r, n) for r, n, _ in dak_test],
         "Movie test: seen words": [(r, n) for r, n, t in syn_test if t == "seen"],
         "Movie test: unseen words": [(r, n) for r, n, t in syn_test if t == "unseen"]}

src_v = Vocab([r for r, *_ in dak_train + syn_train + dak_test + syn_test])
tgt_v = Vocab([n for _, n, *_ in dak_train + syn_train + dak_test + syn_test])
ML = 40


def batchify(pairs):
    s = [src_v.encode(r, ML) for r, n in pairs]
    t = [tgt_v.encode(n, ML) for r, n in pairs]
    pad = lambda xs: torch.tensor([x + [PAD] * (max(map(len, xs)) - len(x)) for x in xs], device=dev)  # noqa: E731
    return pad(s), pad(t)


def cer(p, g):  # character error rate = edit distance / length of the correct word
    d = list(range(len(g) + 1))
    for i, pc in enumerate(p, 1):
        prev, d[0] = d[0], i
        for j, gc in enumerate(g, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (pc != gc))
    return d[-1] / max(len(g), 1)


def evaluate(predict):
    res, errors = {}, []
    for name, pairs in TESTS.items():
        preds = predict([r for r, _ in pairs])
        acc = sum(p == n for p, (_, n) in zip(preds, pairs)) / len(pairs)
        res[name] = {"word_acc": round(acc, 4), "cer": round(sum(cer(p, n) for p, (_, n) in zip(preds, pairs)) / len(pairs), 4),
                     "n": len(pairs)}
        errors += [(name, r, n, p) for p, (r, n) in zip(preds, pairs) if p != n][:15]
    return res, errors


results = {}
# --- Baseline: phonetic-key lookup ---------------------------------------------------------
for mix, pairs in MIXES.items():
    table = defaultdict(Counter)
    for r, n, *_ in pairs:
        table[_key(r)][n] += 1
    look = {k: c.most_common(1)[0][0] for k, c in table.items()}
    results[f"Lookup baseline ({mix})"] = {"scores": evaluate(lambda rs: [look.get(_key(r), "") for r in rs])[0]}

# --- Neural normalizer ---------------------------------------------------------------------
for mix, pairs in MIXES.items():
    pairs = [(r, n) for r, n, *_ in pairs]
    model = Seq2Seq(len(src_v), len(tgt_v), d=128 if a.quick else 256, layers=2 if a.quick else 3, maxlen=ML).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=a.steps, pct_start=0.05)
    rng, t0, curve = random.Random(0), time.time(), []
    for step in range(1, a.steps + 1):
        src, tgt = batchify(rng.sample(pairs, 256))
        logits = model(src, tgt[:, :-1])            # teacher forcing: feed the correct previous characters
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=PAD,
                               label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == a.steps:
            curve.append((step, round(loss.item(), 4)))
            print(f"{mix:22} step {step}/{a.steps} loss {loss.item():.3f}", flush=True)
    model.eval()

    def predict(rs, model=model):
        out = []
        for i in range(0, len(rs), 512):
            src, _ = batchify([(r, "") for r in rs[i:i + 512]])
            out += [tgt_v.decode(x.tolist()) for x in model.greedy(src)]
        return out

    scores, errors = evaluate(predict)
    results[f"Normalizer ({mix})"] = {"scores": scores, "params": model.n_params(), "loss_curve": curve,
                                      "minutes": round((time.time() - t0) / 60, 1), "errors": errors}
    print(mix, json.dumps(scores), flush=True)
    if mix == "Dakshina + synthetic":
        torch.save({"model": model.state_dict(), "src": src_v.chars, "tgt": tgt_v.chars}, N / "normalizer.pt")
        demo = ["yenti", "enti", "anti", "intha", "inta", "nuvvu", "nuwu", "chesanu", "chesa", "vacchava",
                "ekkada", "yekkada", "late", "exam", "bro", "intlo", "vellipothunna", "chuudu", "choodu", "bagundi"]
        results["demo"] = dict(zip(demo, predict(demo)))

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump({"device": dev, "quick": a.quick, "steps": a.steps, "results": results}, open(a.out, "w"), indent=1,
          ensure_ascii=False)
print("DONE")
