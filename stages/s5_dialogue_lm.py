"""Stage 5 — a spoken-style reply model: previous line(s) -> next line, in movie-style Telugu.

    python stages/s5_dialogue_lm.py --word-model path/to/normalizer.pt [--quick]    # needs a GPU (L4/A100)

Three models, all scored the same way on the 1,500 test pairs from the 8 unseen movies:
  1. sarvam-1 as-is        an open 2B Indic model (Sarvam AI), no fine-tuning
  2. sarvam-1 + LoRA       fine-tuned on 29k movie dialogue pairs; the loss is computed on the REPLY only
  3. tiny GPT from scratch ~4M parameters (tlm/tinygpt.py) with this project's Telugu Unigram tokenizer
Metric: bits per character of the REAL next line given its context (lower = the model expects what was actually said).
Then: side-by-side replies, and an end-to-end demo: romanized input -> Stage 4 normalizer -> reply ->
Telugu script + canonical romanization.
"""
import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402
from tlm.romanizer import romanize  # noqa: E402
from tlm.seq2seq import PAD, Seq2Seq, Vocab  # noqa: E402
from tlm.tinygpt import Config, TinyGPT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="sarvamai/sarvam-1")
ap.add_argument("--word-model", required=True)
ap.add_argument("--steps", type=int, default=1200)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "s5_dialogue.json"))
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(0)
load = lambda s: [json.loads(l) for l in open(ROOT / "data" / "dialogue" / f"{s}.jsonl", encoding="utf-8")]  # noqa
train, test = load("train"), load("test")
if a.quick:
    train, test, a.steps = train[:400], test[:40], 20


def prompt(ctx):
    return "".join(f"- {c}\n" for c in ctx) + "- "


results, LN2 = {}, math.log(2)

# =========================== 1 + 2: sarvam-1, before and after LoRA ===========================
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.base)
tok.pad_token = tok.pad_token or tok.eos_token
base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map=dev)
NL = tok("\n", add_special_tokens=False)["input_ids"][-1]


def encode_pair(r, max_len=256):
    p = tok(prompt(r["context"]), add_special_tokens=True)["input_ids"]
    t = tok(r["target"] + "\n", add_special_tokens=False)["input_ids"]
    p = p[-(max_len - len(t)):] if len(p) + len(t) > max_len else p
    return p + t, [-100] * len(p) + t  # labels: only the reply is scored


@torch.no_grad()
def hf_bpc(model, pairs):
    model.eval()
    nll = chars = 0.0
    for r in pairs:
        ids, lab = encode_pair(r)
        x, y = torch.tensor([ids], device=dev), torch.tensor([lab], device=dev)
        n = (y[:, 1:] != -100).sum().item()
        nll += model(input_ids=x, labels=y).loss.item() * n
        chars += len(r["target"]) + 1
    return nll / LN2 / chars


@torch.no_grad()
def hf_reply(model, ctx, rng_seed=0):
    torch.manual_seed(rng_seed)
    x = tok(prompt(ctx), return_tensors="pt").to(dev)
    y = model.generate(**x, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.9, repetition_penalty=1.1,
                       eos_token_id=[tok.eos_token_id, NL], pad_token_id=tok.pad_token_id)
    return tok.decode(y[0][x["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0].strip()


probe = test[:6] + [{"context": ["ఏంట్రా ఇంత లేట్ అయ్యింది?"], "target": ""},
                    {"context": ["అమ్మా, నేను సిటీకి వెళ్ళిపోతున్నాను."], "target": ""},
                    {"context": ["ఈ ఊరిలో నా మాటే చెల్లుతుంది."], "target": ""}]
results["sarvam-1 (no fine-tuning)"] = {"test_bpc": round(hf_bpc(base, test), 4)}
before = [hf_reply(base, r["context"]) for r in probe]
print("base", results["sarvam-1 (no fine-tuning)"], flush=True)

# Gradient checkpointing: recompute activations in the backward pass instead of storing them (fits a 22 GB L4)
base.gradient_checkpointing_enable()
base.enable_input_require_grads()
model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                        "up_proj", "down_proj"]))
model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-4, total_steps=a.steps, pct_start=max(0.05, 3 / a.steps))
model.train()
rng, t0, curve = random.Random(0), time.time(), []
enc_train = [encode_pair(r) for r in train]
for step in range(1, a.steps + 1):
    # Gradient accumulation: 4 micro-batches of 4 = an effective batch of 16, at a quarter of the peak memory
    opt.zero_grad(set_to_none=True)
    for _ in range(4):
        batch = rng.sample(enc_train, 4)
        L = max(len(i) for i, _ in batch)
        x = torch.tensor([i + [tok.pad_token_id] * (L - len(i)) for i, _ in batch], device=dev)
        y = torch.tensor([l + [-100] * (L - len(l)) for _, l in batch], device=dev)
        att = torch.tensor([[1] * len(i) + [0] * (L - len(i)) for i, _ in batch], device=dev)
        loss = model(input_ids=x, attention_mask=att, labels=y).loss
        (loss / 4).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 100 == 0 or step == a.steps:
        curve.append((step, round(loss.item(), 4)))
        print(f"lora step {step}/{a.steps} loss {loss.item():.3f}", flush=True)
results["sarvam-1 + LoRA (movie dialogue)"] = {"test_bpc": round(hf_bpc(model, test), 4), "loss_curve": curve,
                                               "minutes": round((time.time() - t0) / 60, 1)}
after = [hf_reply(model, r["context"]) for r in probe]
print("lora", results["sarvam-1 + LoRA (movie dialogue)"]["test_bpc"], flush=True)
model.save_pretrained(ROOT / "data" / "dialogue" / "lora_adapter")

# =========================== 3: tiny GPT from scratch ==========================================
sp = spm.SentencePieceProcessor(model_file=str(ROOT / "data" / "tok" / "uni_te.model"))
EOS = sp.get_piece_size()
stream = []
for r in train:
    for line in [*r["context"][-1:], r["target"]]:
        stream += sp.encode(line) + [EOS]
data = torch.tensor(stream, device=dev)
c = Config(vocab=EOS + 1, ctx=256, d=128 if a.quick else 256, layers=2 if a.quick else 4)
tiny = TinyGPT(c).to(dev)
topt = torch.optim.AdamW(tiny.parameters(), lr=1e-3, weight_decay=0.1)
tsteps = 50 if a.quick else 3000
tsched = torch.optim.lr_scheduler.OneCycleLR(topt, max_lr=1e-3, total_steps=tsteps, pct_start=0.05)
for step in range(1, tsteps + 1):
    ix = torch.randint(len(data) - c.ctx - 1, (64,), device=dev)
    x = torch.stack([data[i:i + c.ctx] for i in ix])
    y = torch.stack([data[i + 1:i + c.ctx + 1] for i in ix])
    loss = tiny(x, y)[1] / y.numel()
    topt.zero_grad(set_to_none=True)
    loss.backward()
    topt.step()
    tsched.step()
tiny.eval()


@torch.no_grad()
def tiny_bpc(pairs):
    nll = chars = 0.0
    for r in pairs:
        p = [t for line in r["context"] for t in sp.encode(line) + [EOS]]
        t = sp.encode(r["target"]) + [EOS]
        ids = (p + t)[-(c.ctx + 1):]
        k = min(len(t), len(ids) - 1)
        x, y = torch.tensor([ids[:-1]], device=dev), torch.tensor([ids[1:]], device=dev)
        logits = tiny(x)
        lp = torch.log_softmax(logits[0, -k:], -1).gather(1, y[0, -k:, None])
        nll -= lp.sum().item()
        chars += len(r["target"]) + 1
    return nll / LN2 / chars


results["tiny GPT from scratch (4M params)"] = {"test_bpc": round(tiny_bpc(test), 4), "params": tiny.n_params()}
print("tiny", results["tiny GPT from scratch (4M params)"], flush=True)

# =========================== end-to-end demo: romanized in, reply out ===========================
ck = torch.load(a.word_model, map_location=dev)
wsrc, wtgt = Vocab([]), Vocab([])
wsrc.chars, wtgt.chars = ck["src"], ck["tgt"]
wsrc.idx, wtgt.idx = {ch: i + 3 for i, ch in enumerate(wsrc.chars)}, {ch: i + 3 for i, ch in enumerate(wtgt.chars)}
norm = Seq2Seq(len(wsrc), len(wtgt), d=256, layers=3, maxlen=40).to(dev)
norm.load_state_dict(ck["model"])
norm.eval()


def normalize(text):
    ws = re.findall(r"[a-z]+", text.lower())
    enc = [wsrc.encode(w, 40) for w in ws]
    L = max(map(len, enc), default=1)  # pad to the longest word; must stay within the model's 40 positions
    src = torch.tensor([x + [PAD] * (L - len(x)) for x in enc], device=dev)
    m = dict(zip(ws, (wtgt.decode(y.tolist()) for y in norm.greedy(src)))) if ws else {}
    return re.sub(r"[a-z]+", lambda g: m[g.group()], text.lower())


demo = []
for s in ["enti ra intha late ayyindi?", "amma, nenu city ki vellipothunna", "ee oorilo naa maate chellutundi",
          "nuvvu naaku nachchav", "ekkadiki veltunnav ra?"]:
    te = normalize(s)
    rep = hf_reply(model, [te])
    demo.append({"typed": s, "normalized": te, "reply_telugu": rep, "reply_romanized": romanize(rep, canonical=True)})

results["side_by_side"] = [{"context": r["context"], "real": r["target"], "before": b, "after": f}
                           for r, b, f in zip(probe, before, after)]
results["demo"] = demo
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(a.out, "w"), indent=1, ensure_ascii=False)
print(json.dumps(demo, indent=1, ensure_ascii=False))
print("DONE")
