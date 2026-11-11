"""Stage 5c — automatic evaluation of the whole system (no human labels needed).

    python stages/s5c_evaluation.py --adapter-1200 .../lora_adapter --adapter-3000 .../lora_adapter_5b \
        --sentence-normalizer .../sentence_normalizer.pt [--quick]

Layer 3  Pick-the-right-reply: for each unseen-movie context, the real reply is hidden among 9 replies from
         other scenes. The model scores all 10 (average log-probability per character) and should rank
         the real one first. Chance = 10%. Reported as accuracy@1 and mean reciprocal rank (MRR).
Layer 4  Spelling robustness of the pipeline: the same message typed 4 different ways must be understood as
         the same Telugu sentence; reported as the share of messages where all 4 spellings agree, and as the
         average character difference between the understood versions.
Also     efficiency: parameters, adapter size, replies per second.
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
from tlm.seq2seq import Seq2Seq, Vocab  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="sarvamai/sarvam-1")
ap.add_argument("--adapter-1200", required=True)
ap.add_argument("--adapter-3000", required=True)
ap.add_argument("--sentence-normalizer", required=True)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "s5c_evaluation.json"))
a = ap.parse_args()
dev = "cuda"
load = lambda s: [json.loads(l) for l in open(ROOT / "data" / "dialogue" / f"{s}.jsonl", encoding="utf-8")]  # noqa
test = load("test")
N_RANK, N_SPELL = (20, 10) if a.quick else (500, 300)
R = {}

from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.base)
tok.pad_token = tok.pad_token or tok.eos_token
base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map=dev)
model = PeftModel.from_pretrained(base, a.adapter_1200, adapter_name="lora_1200")
model.load_adapter(a.adapter_3000, adapter_name="lora_3000")
model.eval()


def prompt(ctx):
    return "".join(f"- {c}\n" for c in ctx) + "- "


@torch.no_grad()
def scores(ctx, candidates):
    """Average log-probability per character of each candidate reply, given the context (batched)."""
    p = tok(prompt(ctx), add_special_tokens=True)["input_ids"][-200:]
    seqs, labs = [], []
    for c in candidates:
        t = tok(c + "\n", add_special_tokens=False)["input_ids"][:100]
        seqs.append(p + t)
        labs.append([-100] * len(p) + t)
    L = max(map(len, seqs))
    x = torch.tensor([s + [tok.pad_token_id] * (L - len(s)) for s in seqs], device=dev)
    y = torch.tensor([l + [-100] * (L - len(l)) for l in labs], device=dev)
    att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=dev)
    logits = model(input_ids=x, attention_mask=att).logits[:, :-1].float()
    tgt = y[:, 1:]
    lp = torch.log_softmax(logits, -1).gather(2, tgt.clamp(min=0)[..., None])[..., 0]
    tot = (lp * (tgt != -100)).sum(1)
    return [t.item() / (len(c) + 1) for t, c in zip(tot, candidates)]


# ---- Layer 3: pick the right reply -------------------------------------------------------------
rng = random.Random(0)
items = []
pool = [r["target"] for r in test]
for r in test[:N_RANK]:
    distractors = rng.sample([t for t in pool if t != r["target"]], 9)
    items.append((r["context"], [r["target"]] + distractors))


def rank_eval():
    top1 = mrr = 0.0
    for ctx, cands in items:
        s = scores(ctx, cands)
        rank = 1 + sum(v > s[0] for v in s[1:])
        top1 += rank == 1
        mrr += 1 / rank
    return {"accuracy_at_1": round(top1 / len(items), 4), "mrr": round(mrr / len(items), 4)}


R["pick_the_right_reply"] = {"chance": {"accuracy_at_1": 0.1, "mrr": round(sum(1 / k for k in range(1, 11)) / 10, 4)}}
with model.disable_adapter():
    R["pick_the_right_reply"]["sarvam-1 (no fine-tuning)"] = rank_eval()
for name in ["lora_1200", "lora_3000"]:
    model.set_adapter(name)
    R["pick_the_right_reply"][f"LoRA, {name.split('_')[1]} steps"] = rank_eval()
print(json.dumps(R["pick_the_right_reply"]), flush=True)

# ---- Layer 4: same message, 4 spellings ----------------------------------------------------------
ck = torch.load(a.sentence_normalizer, map_location=dev)
sv, tv = Vocab([]), Vocab([])
sv.chars, tv.chars = ck["src"], ck["tgt"]
sv.idx, tv.idx = {c: i + 3 for i, c in enumerate(sv.chars)}, {c: i + 3 for i, c in enumerate(tv.chars)}
norm = Seq2Seq(len(sv), len(tv), d=ck["d"], layers=ck["layers"], maxlen=ck["maxlen"]).to(dev)
norm.load_state_dict(ck["model"])
norm.eval()
lid = LangID.load(ROOT / "data" / "normalizer" / "langid.json")
EN_CHAT = {"ok", "okay", "hi", "bro", "thanks", "sorry", "please", "plz", "yes", "no"}


def normalize_batch(texts):
    masked_all, eng_all = [], []
    for text in texts:
        toks = re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)
        eng = [t for t in toks if t.isalpha() and (t.lower() in EN_CHAT or lid.is_english(t))]
        masked_all.append("".join("@" if (t.isalpha() and t in eng) else t.lower() for t in toks))
        eng_all.append(eng)
    enc = [sv.encode(m, ck["maxlen"]) for m in masked_all]
    L = max(map(len, enc))
    src = torch.tensor([e + [0] * (L - len(e)) for e in enc], device=dev)
    outs = [tv.decode(y.tolist()) for y in norm.greedy(src, maxlen=ck["maxlen"])]
    res = []
    for o, eng in zip(outs, eng_all):
        it = iter(eng)
        res.append(re.sub("@", lambda m: next(it, "@"), o))
    return res


def cer(p, g):
    d = list(range(len(g) + 1))
    for i, pc in enumerate(p, 1):
        prev, d[0] = d[0], i
        for j, gc in enumerate(g, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (pc != gc))
    return d[-1] / max(len(g), 1)


msgs = [r["context"][-1] for r in test if 10 <= len(r["context"][-1]) <= 120][:N_SPELL]
agree = exact = 0
spread, correct_cer = [], []
for i, m in enumerate(msgs):
    r = random.Random(i)
    variants = [romanize(m, r) for _ in range(4)]
    und = normalize_batch(variants)
    agree += len(set(und)) == 1
    exact += sum(u == m for u in und) / 4
    spread.append(sum(cer(u, und[0]) for u in und[1:]) / 3)
    correct_cer.append(sum(cer(u, m) for u in und) / 4)
R["spelling_robustness"] = {"messages": len(msgs), "all_4_spellings_agree": round(agree / len(msgs), 4),
                            "understood_exactly_right": round(exact / len(msgs), 4),
                            "avg_char_difference_between_spellings": round(sum(spread) / len(spread), 4),
                            "avg_char_error_vs_original": round(sum(correct_cer) / len(correct_cer), 4)}
print(json.dumps(R["spelling_robustness"]), flush=True)

# ---- Efficiency -----------------------------------------------------------------------------------
model.set_adapter("lora_3000")
trainable = sum(p.numel() for n, p in model.named_parameters() if "lora_3000" in n)
x = tok(prompt(["ఏంట్రా ఇంత లేట్ అయ్యింది?"]), return_tensors="pt").to(dev)
torch.cuda.synchronize()
t0 = time.time()
n = 3 if a.quick else 10
for _ in range(n):
    model.generate(**x, max_new_tokens=30, do_sample=False, pad_token_id=tok.pad_token_id)
torch.cuda.synchronize()
R["efficiency"] = {"base_params": sum(p.numel() for p in base.parameters()) - 2 * trainable,
                   "adapter_params": trainable, "adapter_mb_bf16": round(trainable * 2 / 1e6, 1),
                   "normalizer_params": sum(p.numel() for p in norm.parameters()),
                   "seconds_per_reply_30_tokens": round((time.time() - t0) / n, 3),
                   "gpu": torch.cuda.get_device_name(0)}
print(json.dumps(R["efficiency"]), flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
print("DONE")
