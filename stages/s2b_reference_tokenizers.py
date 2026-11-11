"""Stage 2b — why Telugu is expensive for English-centric tokenizers (for the README explainer).

    python stages/s2b_reference_tokenizers.py --gpt2 gpt2/tokenizer.json --sarvam sarvam/tokenizer.json

Compares tokens per word on the same test lines: GPT-2 (English-centric byte-level BPE),
Sarvam-2B (Indic-optimised), and this project's Unigram 4k + phonetic key.
"""
import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402
from tlm.plotstyle import SERIES, plt, save  # noqa: E402
from tlm.textnorm import phonetic_key  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gpt2", required=True)
ap.add_argument("--sarvam", required=True)
a = ap.parse_args()
test = [json.loads(l) for l in open(ROOT / "data" / "romanized" / "test.jsonl", encoding="utf-8")]
te, lat = [r["te"] for r in test], [r["noisy"][0] for r in test]
words = lambda xs: sum(len(x.split()) for x in xs)  # noqa: E731

hf = {"GPT-2 (English-centric)": Tokenizer.from_file(a.gpt2), "Sarvam-2B (Indic)": Tokenizer.from_file(a.sarvam)}
res = {n: {"Telugu script": sum(len(t.encode(x, add_special_tokens=False).ids) for x in te) / words(te),
           "romanized": sum(len(t.encode(x, add_special_tokens=False).ids) for x in lat) / words(lat)}
       for n, t in hf.items()}
ours = spm.SentencePieceProcessor(model_file=str(ROOT / "data" / "tok" / "uni_phonetic_4000.model"))
res["This project (Unigram 4k + phonetic)"] = {
    "Telugu script": None, "romanized": sum(len(ours.encode(phonetic_key(x))) for x in lat) / words(lat)}

fig, ax = plt.subplots(figsize=(9, 3.8))
names, w = list(res), 0.36
for j, k in enumerate(["Telugu script", "romanized"]):
    xs = [i + (j - 0.5) * (w + 0.02) for i in range(len(names))]
    vals = [res[n][k] for n in names]
    ax.bar([x for x, v in zip(xs, vals) if v], [v for v in vals if v], width=w, color=SERIES[j], label=k)
    for x, v in zip(xs, vals):
        if v:
            ax.text(x, v + 0.25, f"{v:.1f}", ha="center", fontsize=9, color="#5f5e58")
ax.set_xticks(range(len(names)), names)
ax.set_ylabel("tokens per word (lower = better)")
ax.set_title("The same Telugu sentences cost very different amounts depending on the tokenizer")
ax.legend(loc="upper right")
save(fig, "s2b_reference_tokenizers")
json.dump(res, open(ROOT / "docs" / "results" / "s2b_reference.json", "w"), indent=1)
print(json.dumps(res, indent=1))
