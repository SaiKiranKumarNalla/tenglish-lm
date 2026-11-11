"""Plots and tables for Stage 4 from docs/results/s4_normalizer.json.

    python stages/s4_plots.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402
from tlm.plotstyle import INK2, SERIES, plt, save  # noqa: E402

R = json.load(open(ROOT / "docs" / "results" / "s4_normalizer.json"))["results"]
methods = ["Lookup baseline (Dakshina + synthetic)", "Normalizer (Dakshina only)", "Normalizer (Synthetic only)",
           "Normalizer (Dakshina + synthetic)"]
tests = list(R[methods[-1]]["scores"])

fig, ax = plt.subplots(figsize=(11, 4.2))
w = 0.2
for j, m in enumerate(methods):
    xs = [i + (j - 1.5) * (w + 0.015) for i in range(len(tests))]
    vals = [R[m]["scores"][t]["word_acc"] for t in tests]
    ax.bar(xs, vals, width=w, color=SERIES[j], label=m.replace("Normalizer", "Normalizer trained on").replace(
        "Lookup baseline", "Lookup baseline,").replace("(", "").replace(")", ""))
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.01, f"{v:.0%}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(range(len(tests)), [t.replace(" (", "\n(") for t in tests])
ax.set_ylim(0, 1.12)
ax.set_ylabel("word accuracy (exact Telugu output)")
ax.set_title("Spelling normalizer: romanized word -> Telugu script, by training data")
ax.legend(loc="upper left", fontsize=8.5, ncol=2)
save(fig, "s4_normalizer_accuracy")

best = R["Normalizer (Dakshina + synthetic)"]
md = ["| typed | normalizer output |", "|---|---|"] + [f"| `{k}` | {v} |" for k, v in R.get("demo", {}).items()]
md += ["", "| test set | typed | correct | normalizer output |", "|---|---|---|---|"]
md += [f"| {t.split(':')[-1].split('(')[0].strip()} | `{r}` | {n} | {p or '∅'} |" for t, r, n, p in best["errors"][:30]]
(ROOT / "docs" / "results" / "s4_examples.md").write_text("\n".join(md) + "\n", encoding="utf-8")
for m in R:
    if m != "demo":
        print(m, {t: (s["word_acc"], s["cer"]) for t, s in R[m]["scores"].items()})
