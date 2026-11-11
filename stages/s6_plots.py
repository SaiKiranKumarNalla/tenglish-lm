"""Plots for Stage 6 from docs/results/s6_real_world.json.

    python stages/s6_plots.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402
from tlm.plotstyle import INK2, SERIES, plt, save  # noqa: E402

R = json.load(open(ROOT / "docs" / "results" / "s6_real_world.json"))
pw = R["pairwise"]
names = list(pw)
fig, ax = plt.subplots(figsize=(11, 0.6 * len(names) + 1.6))
for i, n in enumerate(names):
    f, s = pw[n]["first_win_rate"], pw[n]["second_win_rate"]
    t = 1 - f - s
    ax.barh(i, f, color=SERIES[0], height=0.55, label="first wins" if i == 0 else None)
    ax.barh(i, t, left=f, color="#d9d8d2", height=0.55, label="tie / judge inconsistent" if i == 0 else None)
    ax.barh(i, s, left=f + t, color=SERIES[1], height=0.55, label="second wins" if i == 0 else None)
    ax.text(0.01, i, f"{f:.0%}", va="center", color="#fcfcfb", fontsize=9, fontweight="bold")
    ax.text(0.99, i, f"{s:.0%}", va="center", ha="right", color="#fcfcfb", fontsize=9, fontweight="bold")
ax.set_yticks(range(len(names)), [n.replace(" vs ", "  vs  ") for n in names], fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("share of pairs (each judged in both orders)")
ax.set_title(f"Pairwise judgments by {R['judge'].split('/')[-1]}")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)
save(fig, "s6_pairwise_judge")
print(json.dumps(R, indent=1))
