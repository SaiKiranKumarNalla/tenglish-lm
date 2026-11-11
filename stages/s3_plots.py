"""Plots for Stage 3 from docs/results/s3_tiny_lms.json (training runs on a GPU; plotting runs anywhere).

    python stages/s3_plots.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402
from tlm.plotstyle import INK2, SERIES, plt, save  # noqa: E402

R = json.load(open(ROOT / "docs" / "results" / "s3_tiny_lms.json"))["results"]
lossless = [n for n in R if "lossy" not in n]
colors = {n: SERIES[i % 4] for i, n in enumerate(lossless)}

# 1. Learning curves: validation BPC vs characters of training text read
fig, ax = plt.subplots(figsize=(9, 4))
for n in lossless:
    xs, ys = zip(*R[n]["val_curve"])
    ax.plot([x / 1e6 for x in xs], ys, color=colors[n], marker="o", markersize=4, label=n)
    ax.annotate(f"{ys[-1]:.2f}", (xs[-1] / 1e6, ys[-1]), textcoords="offset points", xytext=(6, 0),
                fontsize=8.5, color=INK2, va="center")
ax.set_xlabel("training text read (million characters)")
ax.set_ylabel("validation bits per character (lower = better)")
ax.set_title("Same tiny model, same text: how fast does each tokenizer let it learn?")
ax.legend()
save(fig, "s3_learning_curves")

# 2. Final scores: test BPC and spelling spread
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, key, title in [(axes[0], "test_bpc", "Test bits per character (lower = better)"),
                       (axes[1], "spelling_spread_bpc", "Spelling spread (4 spellings; lower = more robust)")]:
    names = list(R)
    vals = [R[n][key] for n in names]
    ax.barh(range(len(names)), vals, color=[colors.get(n, SERIES[1]) if "lossy" not in n else "#b9b8b0" for n in names],
            height=0.6)
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v:.3f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(names)), [n.replace(" (lossy)", "\n(lossy, not comparable)") for n in names], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_title(title, fontsize=10.5)
save(fig, "s3_final_scores")
print({n: (R[n]["test_bpc"], R[n]["spelling_spread_bpc"], R[n]["params"], R[n]["minutes"]) for n in R})
