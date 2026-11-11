"""Two simple charts for the front page, meant to be readable by anyone.

    python stages/simple_charts.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402
from tlm.plotstyle import INK2, SERIES, plt, save  # noqa: E402

RES = ROOT / "docs" / "results"
fair = json.load(open(RES / "s6d_frontier_fair.json"))
gpu = json.load(open(RES / "s6b_gpu.json"))
man = json.load(open(RES / "s6b_manual.json"))

# 1. Reading test: out of 10 words, how many does each model get exactly right?
scores = {"Our small model": gpu["our_normalizer_word_accuracy_dak300"],
          "Claude Haiku": fair["Claude Haiku 5.5"]["normalization_word_accuracy"],
          "Claude Sonnet": fair["Claude Sonnet 5.5"]["normalization_word_accuracy"],
          "Gemini Flash": man["gemini"]["normalization_word_accuracy"],
          "Claude Opus": fair["Claude Opus 5.5"]["normalization_word_accuracy"]}
goal = 0.8 * scores["Claude Opus"]
fig, ax = plt.subplots(figsize=(9, 3.6))
names = list(scores)
ax.barh(range(len(names)), [scores[n] * 10 for n in names],
        color=[SERIES[1] if n.startswith("Our") else "#9fb7d9" for n in names], height=0.6)
for i, n in enumerate(names):
    ax.text(scores[n] * 10 + 0.1, i, f"{scores[n] * 10:.1f} out of 10", va="center", color=INK2, fontsize=10)
ax.axvline(goal * 10, color="#0b0b0b", linestyle="--", linewidth=1.2)
ax.text(goal * 10 + 0.08, -0.62, f"our goal: {goal * 10:.1f} (80% of the best)", fontsize=9, va="bottom")
ax.set_yticks(range(len(names)), names, fontsize=10)
ax.set_xlim(0, 11.5)
ax.set_xlabel("words written correctly in Telugu, out of every 10")
ax.set_title("Reading test: English-letter Telugu → real Telugu script", pad=12)
ax.grid(axis="y", visible=False)
save(fig, "simple_reading_test")

# 2. Reply matchups: who did the referee pick?
cross = json.load(open(RES / "s6b_cross.json"))
frontier = json.load(open(RES / "s6b_frontier.json"))["claude"]
rows = [("Ours  vs  before training", frontier["judge: fine-tuned vs base (movie)"]["first_win_rate"],
         frontier["judge: fine-tuned vs base (movie)"]["second_win_rate"], "referee: Claude", "before"),
        ("Ours  vs  Claude", man["gemini"]["judge: fine-tuned vs claude"]["ours_win_rate"],
         man["gemini"]["judge: fine-tuned vs claude"]["claude_win_rate"], "referee: Gemini", "Claude"),
        ("Ours  vs  Gemini", cross["claude judges: fine-tuned vs gemini"]["ours_win_rate"],
         cross["claude judges: fine-tuned vs gemini"]["gemini_win_rate"], "referee: Claude", "Gemini")]
fig, ax = plt.subplots(figsize=(9, 3.2))
for i, (label, ours, other, ref, who) in enumerate(rows):
    tie = 1 - ours - other
    ax.barh(i, ours * 100, color=SERIES[1], height=0.55)
    ax.barh(i, tie * 100, left=ours * 100, color="#d9d8d2", height=0.55)
    ax.barh(i, other * 100, left=(ours + tie) * 100, color="#9fb7d9", height=0.55)
    ax.text(1, i, f"ours won {ours * 100:.0f}", va="center", fontsize=9.5, color="#0b0b0b")
    ax.text(99, i, f"{who} won {other * 100:.0f}", va="center", ha="right", fontsize=9.5, color="#0b0b0b")
    ax.text(101, i, ref, va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=10)
ax.invert_yaxis()
ax.set_xlim(0, 125)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("out of 100 matchups (grey = draw)")
ax.set_title("Reply matchups: a referee picks the better of two replies")
ax.grid(axis="y", visible=False)
save(fig, "simple_reply_matchups")
print(scores, goal)
