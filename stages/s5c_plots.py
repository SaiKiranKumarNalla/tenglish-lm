"""Plots for Stage 5c from docs/results/s5c_evaluation.json.

    python stages/s5c_plots.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402
from tlm.plotstyle import INK2, SERIES, plt, save  # noqa: E402

R = json.load(open(ROOT / "docs" / "results" / "s5c_evaluation.json"))
rk = R["pick_the_right_reply"]
names = list(rk)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for ax, key, title in [(axes[0], "accuracy_at_1", "Real reply ranked first (of 10)"),
                       (axes[1], "mrr", "Mean reciprocal rank")]:
    vals = [rk[n][key] for n in names]
    ax.bar(range(len(names)), vals, color=["#b9b8b0"] + SERIES[:len(names) - 1], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.0%}" if key == "accuracy_at_1" else f"{v:.2f}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(range(len(names)), [n.replace(" (", "\n(").replace("LoRA, ", "LoRA\n") for n in names], fontsize=8.5)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title(title)
save(fig, "s5c_pick_the_right_reply")
print(json.dumps(R, indent=1))
