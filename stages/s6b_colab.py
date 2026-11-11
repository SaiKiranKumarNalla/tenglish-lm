"""Stage 6b (GPU part) — our normalizer on the same 300 Dakshina sentences, and Sarvam-M judging ours vs frontier.

    python stages/s6b_colab.py --sentence-normalizer ... --frontier claude

Needs data/realworld/frontier/{dakshina_300.json, <model>_replies.json} (from stages/s6b_frontier.py) and
data/realworld/generated.json (Stage 6). Writes docs/results/s6b_gpu.json.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402
from tlm.seq2seq import Seq2Seq, Vocab  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sentence-normalizer", required=True)
ap.add_argument("--frontier", nargs="+", default=["claude"])
ap.add_argument("--judge", default="sarvamai/sarvam-m")
a = ap.parse_args()
F = ROOT / "data" / "realworld" / "frontier"
dak = json.load(open(F / "dakshina_300.json", encoding="utf-8"))
G = json.load(open(ROOT / "data" / "realworld" / "generated.json", encoding="utf-8"))
R = {}

# --- our sentence normalizer on the same 300 sentences (Dakshina has almost no English, so no masking needed) ---
ck = torch.load(a.sentence_normalizer, map_location="cuda")
sv, tv = Vocab([]), Vocab([])
sv.chars, tv.chars = ck["src"], ck["tgt"]
sv.idx, tv.idx = {c: i + 3 for i, c in enumerate(sv.chars)}, {c: i + 3 for i, c in enumerate(tv.chars)}
norm = Seq2Seq(len(sv), len(tv), d=ck["d"], layers=ck["layers"], maxlen=ck["maxlen"]).cuda()
norm.load_state_dict(ck["model"])
norm.eval()
preds = []
for i in range(0, len(dak), 64):
    enc = [sv.encode(r[:ck["maxlen"] - 2], ck["maxlen"]) for r, _ in dak[i:i + 64]]
    L = max(map(len, enc))
    src = torch.tensor([e + [0] * (L - len(e)) for e in enc], device="cuda")
    preds += [tv.decode(y.tolist()) for y in norm.greedy(src, maxlen=ck["maxlen"])]
hit = tot = 0
for p, (_, g) in zip(preds, dak):
    pw, gw = re.findall(r"[ఀ-౿]+", p), re.findall(r"[ఀ-౿]+", g)
    hit += sum(x == y for x, y in zip(pw, gw))
    tot += len(gw)
R["our_normalizer_word_accuracy_dak300"] = round(hit / tot, 4)
json.dump(preds, open(F / "ours_normalization.json", "w"), ensure_ascii=False)  # per-sentence outputs, for the public report
del norm
torch.cuda.empty_cache()

# --- Sarvam-M judges ours vs each frontier model, and the frontier model vs the real reply ---
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

PROMPT = ("You are judging replies in a Telugu conversation. A person said:\n\"{m}\"\n\n"
          "Reply A: \"{a}\"\nReply B: \"{b}\"\n\n"
          "Which reply is a better next line in a natural, spoken Telugu conversation: relevant to what was said, "
          "fluent, and sounding like how people really talk? Answer with a single letter: A or B.")
tok = AutoTokenizer.from_pretrained(a.judge, padding_side="left")
tok.pad_token = tok.pad_token or tok.eos_token
jm = AutoModelForCausalLM.from_pretrained(a.judge, device_map="cuda", quantization_config=BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16))


def verdicts(prompts):
    out = []
    for i in range(0, len(prompts), 16):
        chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for p in prompts[i:i + 16]]
        x = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        y = jm.generate(**x, max_new_tokens=8, do_sample=False)
        for t in tok.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True):
            m = re.search(r"\b([AB])\b", re.sub(r"<think>.*?</think>", "", t, flags=re.S).upper())
            out.append(m.group(1) if m else "?")
    return out


def compare(pairs):
    v = verdicts([PROMPT.format(m=m, a=x, b=y) for m, x, y in pairs] + [PROMPT.format(m=m, a=y, b=x) for m, x, y in pairs])
    n = len(pairs)
    w = {"first": 0, "second": 0, "tie/inconsistent": 0}
    for p, q in zip(v[:n], v[n:]):
        w["first" if (p, q) == ("A", "B") else "second" if (p, q) == ("B", "A") else "tie/inconsistent"] += 1
    return {**w, "n": n, "first_win_rate": round(w["first"] / n, 4), "second_win_rate": round(w["second"] / n, 4)}


movie = G["movie"]
for fm in a.frontier:
    rep = json.load(open(F / f"{fm}_replies.json", encoding="utf-8"))
    ok = [(m, r) for m, r in zip(movie, rep) if r.strip()]
    R[f"Sarvam-M judge: fine-tuned vs {fm}"] = compare([(m["message"], m["tuned"], r) for m, r in ok])
    R[f"Sarvam-M judge: {fm} vs real reply"] = compare([(m["message"], r, m["real"]) for m, r in ok])
(ROOT / "docs" / "results").mkdir(parents=True, exist_ok=True)  # results folder
json.dump(R, open(ROOT / "docs" / "results" / "s6b_gpu.json", "w"), indent=1)
print(json.dumps(R, indent=1))
print("DONE")
