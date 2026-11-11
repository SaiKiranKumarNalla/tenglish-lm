"""Plots and tables for Stage 4b from docs/results/s4b_sentence.json.

    python stages/s4b_plots.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402
from tlm.plotstyle import INK2, SERIES, plt, save  # noqa: E402

R = json.load(open(ROOT / "docs" / "results" / "s4b_sentence.json"))
tests = [k for k in R if isinstance(R[k], dict) and "word-level (Stage 4)" in R[k]]
models = ["word-level (Stage 4)", "sentence-level (Stage 4b)"]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, key, title in [(axes[0], "word_acc", "All words"), (axes[1], "ambiguous_word_acc", "Ambiguous spellings only")]:
    for j, m in enumerate(models):
        xs = [i + (j - 0.5) * 0.38 for i in range(len(tests))]
        vals = [R[t][m][key] for t in tests]
        ax.bar(xs, vals, width=0.36, color=SERIES[j], label=m)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.0%}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(range(len(tests)), [t.replace(" (", "\n(") for t in tests])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("word accuracy")
    ax.set_title(title)
axes[0].legend(loc="upper left")
save(fig, "s4b_word_vs_sentence")

md = ["| typed | English detected → placeholder | output |", "|---|---|---|"]
md += [f"| `{d['input']}` | `{d['masked']}` | {d['output']} |" for d in R["demo"]]
(ROOT / "docs" / "results" / "s4b_examples.md").write_text("\n".join(md) + "\n", encoding="utf-8")
for t in tests:
    print(t, R[t])
print("\n".join(md))
