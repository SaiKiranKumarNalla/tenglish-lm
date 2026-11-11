"""Plots and example tables for Stage 5b from docs/results/s5b_better_replies.json.

    python stages/s5b_plots.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402
from tlm.plotstyle import INK2, SERIES, plt, save  # noqa: E402

R = json.load(open(ROOT / "docs" / "results" / "s5b_better_replies.json"))
cu = R["context_usage"]
models = list(cu)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
w = 0.38
for j, (key, label) in enumerate([("bpc_true_context", "true context"), ("bpc_random_context", "random context")]):
    xs = [i + (j - 0.5) * (w + 0.02) for i in range(len(models))]
    vals = [cu[m][key] for m in models]
    axes[0].bar(xs, vals, width=w, color=SERIES[j], label=label)
    for x, v in zip(xs, vals):
        axes[0].text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5, color=INK2)
axes[0].set_xticks(range(len(models)), [m.replace(" (", "\n(") for m in models], fontsize=8.5)
axes[0].set_ylabel("bits per character of the real reply")
axes[0].set_title("Next-line BPC (lower = better)")
axes[0].legend(loc="upper right")
gains = [cu[m]["gain_bits_per_char"] for m in models]
axes[1].bar(range(len(models)), gains, color=SERIES[2], width=0.55)
for i, g in enumerate(gains):
    axes[1].text(i, g + 0.003, f"{g:.3f}", ha="center", fontsize=9, color=INK2)
axes[1].set_xticks(range(len(models)), [m.replace(" (", "\n(") for m in models], fontsize=8.5)
axes[1].set_ylabel("bits per character saved by the true context")
axes[1].set_title("Context usage (higher = more responsive)")
save(fig, "s5b_context_usage")

dec = R["decoding"]
names = list(dec)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for ax, key, title, fmt in [(axes[0], "loop_rate", "Replies that loop (lower = better)", "{:.1%}"),
                            (axes[1], "distinct_2", "Distinct word pairs (higher = varied)", "{:.0%}")]:
    vals = [dec[n][key] for n in names]
    ax.barh(range(len(names)), vals, color=[SERIES[i] for i in range(len(names))], height=0.55)
    for i, v in enumerate(vals):
        ax.text(v, i, "  " + fmt.format(v), va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(vals) * 1.3, 0.1))
    ax.set_title(title, fontsize=10.5)
save(fig, "s5b_decoding")

md = ["## Decoding settings, same contexts", "", "| last line | real reply | " + " | ".join(names) + " |",
      "|---|---|" + "---|" * len(names)]
md += [f"| {s['context'][-1]} | {s['real']} | " + " | ".join(s[n] for n in names) + " |" for s in R["decoding_samples"]]
md += ["", "## Full pipeline: code-mixed chat in, reply out", "", "| typed | understood as | reply | reply, romanized |",
       "|---|---|---|---|"]
md += [f"| `{p['typed']}` | {p['understood']} | {p['reply']} | `{p['reply_romanized']}` |" for p in R["pipeline"]]
(ROOT / "docs" / "results" / "s5b_examples.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(json.dumps({"context_usage": cu, "decoding": dec}, indent=1))
print("\n".join(md))
