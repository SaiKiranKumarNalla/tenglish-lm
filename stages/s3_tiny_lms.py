"""Stage 3 — does the tokenizer choice change how well a small model learns romanized movie Telugu?

    python stages/s3_tiny_lms.py            # full run (a GPU is recommended; ~20-40 min on a T4)
    python stages/s3_tiny_lms.py --quick    # tiny smoke test

Same model (tlm/tinygpt.py), same data (noisy romanized train movies), same budget (2 passes over the text)
for every tokenizer. Scored on the unseen TEST movies:
  bits per character (BPC)  -log2 p(text) / number of characters in the ORIGINAL text. Unlike per-token loss,
                            this is comparable across tokenizers, because every model is charged for the same characters.
  spelling spread           the same test sentence in 4 spellings (canonical + 3 noisy): average gap between
                            the best- and worst-scored spelling, in BPC. Lower = the model is less thrown by spelling.
The phonetic-key configuration changes the text (lossy), so its BPC is not directly comparable; it's reported separately.
Needs data/romanized/*.jsonl (Stage 1) and the tokenizer models in data/tok/ (Stages 1b/2).
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402
from tlm.textnorm import phonetic_key  # noqa: E402
from tlm.tinygpt import Config, TinyGPT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "s3_tiny_lms.json"))
ap.add_argument("--only", default=None, help="run just this configuration and merge into an existing --out file")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

load = lambda s: [json.loads(l) for l in open(ROOT / "data" / "romanized" / f"{s}.jsonl", encoding="utf-8")]  # noqa
train, val, test = load("train"), load("val"), load("test")
if a.quick:
    train, val, test = train[:2000], val[:300], test[:300]
train_text = [t for r in train for t in r["noisy"]]
val_text = [r["noisy"][0] for r in val]
test_text = [r["noisy"][0] for r in test]
_pool = [r for r in test if len(r["canon"]) >= 20]
spread_set = random.Random(0).sample(_pool, min(400, len(_pool)))


class CharTok:
    def __init__(self, texts):
        self.chars = sorted({c for t in texts for c in t})
        self.idx = {c: i + 1 for i, c in enumerate(self.chars)}  # 0 = unknown

    def vocab(self):
        return len(self.chars) + 2  # + unknown + end-of-line

    def encode(self, s, sample=False):
        return [self.idx.get(c, 0) for c in s]

    @property
    def eos(self):
        return len(self.chars) + 1


class SPTok:
    def __init__(self, path, sample=False, transform=None):
        self.sp = spm.SentencePieceProcessor(model_file=str(path))
        self.sample, self.f = sample, transform or (lambda s: s)

    def vocab(self):
        return self.sp.get_piece_size() + 1

    def encode(self, s, sample=False):
        if sample and self.sample:  # subword regularization: a different segmentation each time
            return self.sp.encode(self.f(s), enable_sampling=True, alpha=0.1, nbest_size=-1)
        return self.sp.encode(self.f(s))

    @property
    def eos(self):
        return self.sp.get_piece_size()


T = ROOT / "data" / "tok"
CONFIGS = {
    "Character": lambda: CharTok(train_text),
    "BPE 4k": lambda: SPTok(T / "sweep_bpe_4000.model"),
    "Unigram 4k": lambda: SPTok(T / "sweep_unigram_4000.model"),
    "Unigram 4k + sampling": lambda: SPTok(T / "sweep_unigram_4000.model", sample=True),
    "Unigram 4k + phonetic key (lossy)": lambda: SPTok(T / "uni_phonetic_4000.model", transform=phonetic_key),
}


def stream(tok, texts, sample=False):
    ids = []
    for t in texts:
        ids += tok.encode(t, sample) + [tok.eos]
    return torch.tensor(ids, dtype=torch.long)


@torch.no_grad()
def bpc(model, tok, texts, ctx):
    """Bits per character of the ORIGINAL text (each line + its newline)."""
    model.eval()
    ids = stream(tok, texts).to(dev)
    nll = 0.0
    for i in range(0, len(ids) - 1, ctx):
        x, y = ids[i:i + ctx][None], ids[i + 1:i + ctx + 1][None]
        x = x[:, :y.shape[1]]
        nll += model(x, y)[1].item()
    model.train()
    return nll / math.log(2) / sum(len(t) + 1 for t in texts)


results = {}
import os  # noqa: E402

if a.only and os.path.exists(a.out):
    results = json.load(open(a.out))["results"]
for name, make in CONFIGS.items():
    if a.only and name != a.only:
        continue
    tok = make()
    c = Config(vocab=tok.vocab(), ctx=256, d=128 if a.quick else 256, layers=2 if a.quick else 4)
    model = TinyGPT(c).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    # With subword sampling, words split into more pieces: size the budget on a SAMPLED stream so every
    # configuration reads the same amount of text (an earlier version under-trained this config by ~2x).
    data = stream(tok, train_text, sample=getattr(tok, "sample", False)).to(dev)
    B = 64
    steps = max(int(a.epochs * len(data) / (B * c.ctx)), 20)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
    curve, t0 = [], time.time()
    train_chars = sum(len(t) + 1 for t in train_text)
    for step in range(1, steps + 1):
        if getattr(tok, "sample", False) and step % 200 == 1:  # re-sample segmentations regularly
            data = stream(tok, random.Random(step).sample(train_text, len(train_text)), sample=True).to(dev)
        ix = torch.randint(len(data) - c.ctx - 1, (B,), device=dev)
        x = torch.stack([data[i:i + c.ctx] for i in ix])
        y = torch.stack([data[i + 1:i + c.ctx + 1] for i in ix])
        loss = model(x, y)[1] / y.numel()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(steps // 12, 1) == 0 or step == steps:
            # x-axis: how much training text the model has read, in characters (comparable across tokenizers)
            curve.append((round(step * B * c.ctx / len(data) * train_chars), round(bpc(model, tok, val_text, c.ctx), 4)))
            print(f"{name:34} step {step}/{steps}  val BPC {curve[-1][1]:.3f}", flush=True)
    spreads = []
    for r in spread_set:
        vs = [bpc(model, tok, [s], c.ctx) for s in [r["canon"], *r["noisy"]]]
        spreads.append(max(vs) - min(vs))
    results[name] = {"params": model.n_params(), "steps": steps, "train_tokens": len(data),
                     "minutes": round((time.time() - t0) / 60, 1), "val_curve": curve,
                     "test_bpc": round(bpc(model, tok, test_text, c.ctx), 4),
                     "spelling_spread_bpc": round(sum(spreads) / len(spreads), 4)}
    print(name, json.dumps({k: v for k, v in results[name].items() if k != "val_curve"}), flush=True)

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump({"device": dev, "quick": a.quick, "epochs": a.epochs, "results": results}, open(a.out, "w"), indent=1)
print("DONE")
