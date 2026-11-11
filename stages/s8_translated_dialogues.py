"""Stage 8 — a large everyday-conversation set: English dialogues (SODA) -> Telugu -> spoken style -> Tenglish.

    python stages/s8_translated_dialogues.py --n-dialogues 5000          # pilot  (GPU; ~400k English words)
    python stages/s8_translated_dialogues.py --n-dialogues 150000        # full   (~10M Telugu words)

Pipeline (which parts are AI and which are simple code):
  1. SODA, 1.5M English everyday dialogues (Allen AI, CC BY 4.0), a random sample          [data]
  2. IndicTrans2 en->te (AI4Bharat, MIT): the ONLY AI step, used for translation only     [AI model]
  3. tlm/colloquial.py: formal -> spoken endings, English loanwords, regional markers      [simple code]
  4. tlm/romanizer.py: a romanized (Tenglish) copy of each turn, in realistic spellings     [simple code]
Writes data/conversations/soda_te_<n>.jsonl (git-ignored) and docs/results/s8_translation.json.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.colloquial import code_mix, dialect, spoken  # noqa: E402
from tlm.corpus import ROOT  # noqa: E402
from tlm.romanizer import romanize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n-dialogues", type=int, default=5000)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--model", default="ai4bharat/indictrans2-en-indic-dist-200M")
a = ap.parse_args()
rng = random.Random(0)

from datasets import load_dataset  # noqa: E402

soda = load_dataset("allenai/soda", split="train")
idx = rng.sample(range(len(soda)), a.n_dialogues)
dialogues = [soda[i] for i in idx]
utts = [(d_i, t_i, u) for d_i, d in enumerate(dialogues) for t_i, u in enumerate(d["dialogue"])]
print(f"{len(dialogues)} dialogues, {len(utts)} utterances, {sum(len(u.split()) for *_, u in utts):,} English words", flush=True)

from IndicTransToolkit.processor import IndicProcessor  # noqa: E402
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(a.model, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()
ip = IndicProcessor(inference=True)
order = sorted(range(len(utts)), key=lambda k: len(utts[k][2]))  # similar lengths together = faster batches
te = [None] * len(utts)
t0 = time.time()
for b in range(0, len(order), a.batch):
    ids = order[b:b + a.batch]
    src = ip.preprocess_batch([utts[k][2] for k in ids], src_lang="eng_Latn", tgt_lang="tel_Telu")
    x = tok(src, truncation=True, padding="longest", return_tensors="pt", max_length=256).to("cuda")
    with torch.no_grad():
        y = model.generate(**x, num_beams=1, max_length=256, use_cache=True)
    out = ip.postprocess_batch(tok.batch_decode(y, skip_special_tokens=True, clean_up_tokenization_spaces=True), lang="tel_Telu")
    for k, o in zip(ids, out):
        te[k] = o
    if (b // a.batch) % 50 == 0:
        print(f"translated {b + len(ids):,}/{len(utts):,}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

DIALECTS = ["Standard"] * 5 + ["Telangana", "Rayalaseema", "Coastal Andhra"]
out_path = ROOT / "data" / "conversations" / f"soda_te_{a.n_dialogues}.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)
words = 0
with open(out_path, "w", encoding="utf-8") as f:
    pos = 0  # translations are stored in the same order as the utterances list
    for d_i, d in enumerate(dialogues):
        r = random.Random(d_i)
        dia = r.choice(DIALECTS)
        conv = []
        for en in d["dialogue"]:
            formal = te[pos]
            pos += 1
            sp = dialect(code_mix(spoken(formal), r), dia)
            conv.append({"en": en, "te_formal": formal, "te_spoken": sp, "tenglish": romanize(sp, r)})
            words += len(sp.split())
        f.write(json.dumps({"source": "soda-translated", "soda_index": idx[d_i], "dialect": dia, "turns": conv},
                           ensure_ascii=False) + "\n")
stats = {"dialogues": len(dialogues), "utterances": len(utts), "telugu_words": words,
         "minutes": round((time.time() - t0) / 60, 1), "model": a.model}
(ROOT / "docs" / "results").mkdir(parents=True, exist_ok=True)
json.dump(stats, open(ROOT / "docs" / "results" / f"s8_translation_{a.n_dialogues}.json", "w"), indent=1)
print(json.dumps(stats))
print("DONE")
