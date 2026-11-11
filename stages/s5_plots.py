"""Plots and example tables for Stage 5 from docs/results/s5_dialogue.json.

    python stages/s5_plots.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402
from tlm.plotstyle import INK2, SERIES, plt, save  # noqa: E402

R = json.load(open(ROOT / "docs" / "results" / "s5_dialogue.json"))
models = [k for k in R if isinstance(R[k], dict) and "test_bpc" in R[k]]
fig, ax = plt.subplots(figsize=(9, 3.2))
vals = [R[m]["test_bpc"] for m in models]
ax.barh(range(len(models)), vals, color=[SERIES[i] for i in range(len(models))], height=0.55)
for i, v in enumerate(vals):
    ax.text(v, i, f"  {v:.2f}", va="center", color=INK2, fontsize=9)
ax.set_yticks(range(len(models)), models)
ax.invert_yaxis()
ax.set_xlim(0, max(vals) * 1.2)
ax.set_xlabel("bits per character of the real next line (lower = better)")
ax.set_title("Predicting the next line of dialogue in 8 unseen movies")
save(fig, "s5_next_line_bpc")

md = ["## Replies before and after fine-tuning", "", "| context (last line) | real next line | sarvam-1 as-is | sarvam-1 + LoRA |",
      "|---|---|---|---|"]
md += [f"| {r['context'][-1]} | {r['real'] or '—'} | {r['before']} | {r['after']} |" for r in R["side_by_side"]]
md += ["", "## End to end: romanized input → normalizer → reply", "",
       "| typed | normalized (Stage 4) | reply | reply, romanized |", "|---|---|---|---|"]
md += [f"| `{d['typed']}` | {d['normalized']} | {d['reply_telugu']} | `{d['reply_romanized']}` |" for d in R["demo"]]
(ROOT / "docs" / "results" / "s5_examples.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print({m: R[m]["test_bpc"] for m in models})
print("\n".join(md))
