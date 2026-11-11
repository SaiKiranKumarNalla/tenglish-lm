"""Stage 0 — know your data before training anything.

    TLM_CORPUS=~/Downloads/Tenglish_LM/movie_corpus python stages/s0_corpus_stats.py

Writes docs/plots/s0_*.png and docs/results/s0_stats.json.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT, load_scenes, split_of, telugu_lines  # noqa: E402
from tlm.plotstyle import INK2, SERIES, plt, save  # noqa: E402

scenes = load_scenes()
split = split_of()
lines = telugu_lines(scenes)
words_per_line = [len(l.split()) for l in lines]

# --- 1. How much text per movie, and which split it belongs to -----------------------------
per_movie = Counter()
for s in scenes:
    per_movie[s.movie] += len(s.lines)
movies = sorted(per_movie, key=per_movie.get, reverse=True)
colors = {"train": SERIES[0], "val": SERIES[1], "test": SERIES[2]}
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(range(len(movies)), [per_movie[m] for m in movies], width=0.8,
       color=[colors.get(split.get(m), INK2) for m in movies])
ax.set_xticks([])
ax.set_xlabel(f"{len(movies)} movies, sorted by size")
ax.set_ylabel("dialogue lines")
ax.set_title("Dialogue lines per movie — split by movie, so test movies are never trained on")
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color=c, label=f"{k} ({sum(1 for m in movies if split.get(m) == k)} movies)")
                   for k, c in colors.items()], loc="upper right")
save(fig, "s0_lines_per_movie")

# --- 2. Line length: short lines = conversational ------------------------------------------
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.hist([min(w, 40) for w in words_per_line], bins=40, color=SERIES[0], rwidth=0.85)
ax.set_xlabel("words per line (40+ grouped)")
ax.set_ylabel("lines")
med = sorted(words_per_line)[len(words_per_line) // 2]
ax.axvline(med, color=INK2, linewidth=1, linestyle="--")
ax.text(med + 0.5, ax.get_ylim()[1] * 0.9, f"median {med} words", color=INK2)
ax.set_title("Most lines are short, spoken-style utterances")
save(fig, "s0_words_per_line")

# --- 3. Era: is this really contemporary Telugu? -------------------------------------------
years = Counter()
for m in {s.movie: s.year for s in scenes}.items():
    if m[1]:
        years[m[1]] += 1
fig, ax = plt.subplots(figsize=(7, 3.4))
ys = sorted(years)
ax.bar(ys, [years[y] for y in ys], width=0.8, color=SERIES[0])
ax.set_xlabel("release year")
ax.set_ylabel("movies")
ax.set_title("Corpus era: movies by release year")
save(fig, "s0_movies_by_year")

# --- 4. Why the v1 keyword tags can't be trusted -------------------------------------------
dial = Counter(s.kw_dialect for s in scenes)
violence = ["కత్తి", "రక్తం", "నరికేస్తా"]
rs = [s for s in scenes if s.kw_dialect == "RAYALASEEMA"]
rs_by_violence = sum(1 for s in rs if any(v in " ".join(s.lines) for v in violence))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
ks = [k for k, _ in dial.most_common()]
axes[0].barh(ks[::-1], [dial[k] for k in ks[::-1]], color=SERIES[0], height=0.6)
axes[0].set_xlabel("scenes")
axes[0].set_title("v1 dialect tags: 'standard' = no keyword matched")
axes[1].barh(["triggered by a violence word\n(knife / blood / 'I'll hack you')", "other keywords"],
             [rs_by_violence, len(rs) - rs_by_violence], color=[SERIES[1], SERIES[0]], height=0.6)
axes[1].set_xlabel("scenes tagged RAYALASEEMA")
axes[1].set_title(f"{rs_by_violence / len(rs):.0%} of 'Rayalaseema' tags = violence words")
save(fig, "s0_keyword_tags_unreliable")

stats = {"movies": len(per_movie), "scenes": len(scenes), "telugu_lines": len(lines),
         "words": sum(words_per_line), "median_words_per_line": med,
         "years": [min(ys), max(ys)] if ys else None,
         "kw_dialect": dict(dial), "rayalaseema_from_violence_words": round(rs_by_violence / len(rs), 3)}
(ROOT / "docs" / "results").mkdir(parents=True, exist_ok=True)
json.dump(stats, open(ROOT / "docs" / "results" / "s0_stats.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(stats, ensure_ascii=False))
