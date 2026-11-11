"""Stage 5b — better replies: more training, better decoding, a context-usage test, and the full pipeline.

    python stages/s5b_better_replies.py --adapter path/to/stage5/lora_adapter \
        --sentence-normalizer data/normalizer/sentence_normalizer.pt [--quick]

1. continue LoRA training from the Stage 5 adapter (1,200 -> 3,000 steps total) and re-measure next-line BPC
2. context usage: BPC of the real reply given its TRUE context vs a RANDOM context from another scene.
   The gap (in bits per character) says how much the model actually responds to what was said.
3. decoding: three settings on 200 unseen contexts. Loop rate = share of replies that repeat a 3-word phrase;
   distinct-2 = share of unique word pairs (diversity)
4. full pipeline on code-mixed chat: English detection -> sentence normalizer -> reply -> romanized reply
"""
import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402
from tlm.langid import LangID  # noqa: E402
from tlm.romanizer import romanize  # noqa: E402
from tlm.seq2seq import PAD, Seq2Seq, Vocab  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="sarvamai/sarvam-1")
ap.add_argument("--adapter", required=True)
ap.add_argument("--sentence-normalizer", required=True)
ap.add_argument("--more-steps", type=int, default=1800)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "s5b_better_replies.json"))
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(0)
load = lambda s: [json.loads(l) for l in open(ROOT / "data" / "dialogue" / f"{s}.jsonl", encoding="utf-8")]  # noqa
train, test = load("train"), load("test")
N_DEC = 200
if a.quick:
    train, test, a.more_steps, N_DEC = train[:300], test[:30], 8, 10
LN2, R = math.log(2), {}

from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.base)
tok.pad_token = tok.pad_token or tok.eos_token
base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map=dev)
base.gradient_checkpointing_enable()
base.enable_input_require_grads()
model = PeftModel.from_pretrained(base, a.adapter, is_trainable=True)
NL = tok("\n", add_special_tokens=False)["input_ids"][-1]


def prompt(ctx):
    return "".join(f"- {c}\n" for c in ctx) + "- "


def encode_pair(ctx, target, max_len=256):
    p = tok(prompt(ctx), add_special_tokens=True)["input_ids"]
    t = tok(target + "\n", add_special_tokens=False)["input_ids"]
    p = p[-(max_len - len(t)):] if len(p) + len(t) > max_len else p
    return p + t, [-100] * len(p) + t


@torch.no_grad()
def bpc(m, pairs, shuffle_ctx=False):
    m.eval()
    rng = random.Random(5)
    nll = chars = 0.0
    for r in pairs:
        ctx = rng.choice(train)["context"] if shuffle_ctx else r["context"]
        ids, lab = encode_pair(ctx, r["target"])
        x, y = torch.tensor([ids], device=dev), torch.tensor([lab], device=dev)
        n = (y[:, 1:] != -100).sum().item()
        nll += m(input_ids=x, labels=y).loss.item() * n
        chars += len(r["target"]) + 1
    return nll / LN2 / chars


def context_usage(m, label):
    true, rand = bpc(m, test), bpc(m, test, shuffle_ctx=True)
    R.setdefault("context_usage", {})[label] = {"bpc_true_context": round(true, 4), "bpc_random_context": round(rand, 4),
                                                "gain_bits_per_char": round(rand - true, 4)}
    print(label, R["context_usage"][label], flush=True)
    return true


# --- 1 + 2: before more training (= Stage 5 adapter), base without adapter, after more training ----
with model.disable_adapter():
    context_usage(model, "sarvam-1 (no fine-tuning)")
context_usage(model, "LoRA, 1,200 steps (Stage 5)")

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-5, total_steps=a.more_steps, pct_start=max(0.05, 3 / a.more_steps))
enc = [encode_pair(r["context"], r["target"]) for r in train]
rng, t0, curve = random.Random(1), time.time(), []
model.train()
for step in range(1, a.more_steps + 1):
    opt.zero_grad(set_to_none=True)
    for _ in range(4):  # 4 micro-batches of 4 = effective batch 16
        b = rng.sample(enc, 4)
        L = max(len(i) for i, _ in b)
        x = torch.tensor([i + [tok.pad_token_id] * (L - len(i)) for i, _ in b], device=dev)
        y = torch.tensor([l + [-100] * (L - len(l)) for _, l in b], device=dev)
        att = torch.tensor([[1] * len(i) + [0] * (L - len(i)) for i, _ in b], device=dev)
        (model(input_ids=x, attention_mask=att, labels=y).loss / 4).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 200 == 0 or step == a.more_steps:
        print(f"more training step {step}/{a.more_steps}", flush=True)
R["extra_training_minutes"] = round((time.time() - t0) / 60, 1)
context_usage(model, "LoRA, 3,000 steps (Stage 5b)")
model.save_pretrained(ROOT / "data" / "dialogue" / "lora_adapter_5b")
model.eval()

# --- 3: decoding settings ------------------------------------------------------------------
DECODING = {
    "sampling (Stage 5 setting)": dict(do_sample=True, temperature=0.8, top_p=0.9, repetition_penalty=1.1),
    "sampling + no repeated trigrams": dict(do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.2,
                                            no_repeat_ngram_size=3),
    "greedy": dict(do_sample=False, repetition_penalty=1.1),
}


@torch.no_grad()
def reply(ctx, cfg, seed=0):
    torch.manual_seed(seed)
    x = tok(prompt(ctx), return_tensors="pt").to(dev)
    y = model.generate(**x, max_new_tokens=50, eos_token_id=[tok.eos_token_id, NL], pad_token_id=tok.pad_token_id, **cfg)
    return tok.decode(y[0][x["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0].strip()


def loops(t):
    w = t.split()
    tri = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return len(tri) != len(set(tri)) or any(w[i] == w[i + 1] == w[i + 2] for i in range(len(w) - 2))


ctxs = [r["context"] for r in test[:N_DEC]]
R["decoding"] = {}
samples = {}
for name, cfg in DECODING.items():
    outs = [reply(c, cfg, i) for i, c in enumerate(ctxs)]
    bigrams = [tuple(o.split()[i:i + 2]) for o in outs for i in range(len(o.split()) - 1)]
    R["decoding"][name] = {"loop_rate": round(sum(map(loops, outs)) / len(outs), 4),
                           "distinct_2": round(len(set(bigrams)) / max(len(bigrams), 1), 4),
                           "avg_words": round(sum(len(o.split()) for o in outs) / len(outs), 2)}
    samples[name] = outs[:8]
    print(name, R["decoding"][name], flush=True)
R["decoding_samples"] = [{"context": ctxs[i], "real": test[i]["target"], **{n: samples[n][i] for n in samples}}
                         for i in range(min(8, len(ctxs)))]
best = "sampling + no repeated trigrams"

# --- 4: full pipeline ----------------------------------------------------------------------------
ck = torch.load(a.sentence_normalizer, map_location=dev)
sv, tv = Vocab([]), Vocab([])
sv.chars, tv.chars = ck["src"], ck["tgt"]
sv.idx, tv.idx = {c: i + 3 for i, c in enumerate(sv.chars)}, {c: i + 3 for i, c in enumerate(tv.chars)}
norm = Seq2Seq(len(sv), len(tv), d=ck["d"], layers=ck["layers"], maxlen=ck["maxlen"]).to(dev)
norm.load_state_dict(ck["model"])
norm.eval()
lid = LangID.load(ROOT / "data" / "normalizer" / "langid.json")
EN_CHAT = {"ok", "okay", "hi", "bro", "thanks", "sorry", "please", "plz", "yes", "no"}


def normalize(text):
    toks = re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)
    english = [t for t in toks if t.isalpha() and (t.lower() in EN_CHAT or lid.is_english(t))]
    masked = "".join("@" if (t.isalpha() and t in english) else t.lower() for t in toks)
    ids = sv.encode(masked, ck["maxlen"])
    out = tv.decode(norm.greedy(torch.tensor([ids], device=dev), maxlen=ck["maxlen"])[0].tolist())
    it = iter(english)
    return re.sub("@", lambda m: next(it, "@"), out)


chats = ["enti ra intha late ayyindi?", "bro exam ela rasav, results eppudu?", "amma, nenu city ki vellipothunna",
         "nuvvu naaku nachchav", "ekkadiki veltunnav ra?", "sorry ra, traffic lo stuck ayya",
         "ee oorilo naa maate chellutundi", "repu movie ki veldama?"]
R["pipeline"] = []
for s in chats:
    te = normalize(s)
    rep = reply([te], DECODING[best], 1)
    R["pipeline"].append({"typed": s, "understood": te, "reply": rep, "reply_romanized": romanize(rep, canonical=True)})
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
print(json.dumps(R["pipeline"], indent=1, ensure_ascii=False))
print("DONE")
