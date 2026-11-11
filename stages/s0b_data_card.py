"""Stage 0b — a data card: where the text comes from, what cleaning removed, and what's inside.

    TLM_CORPUS=... python stages/s0b_data_card.py

Writes docs/plots/s0b_*.png and docs/results/s0b_stats.json (summarised in docs/DATA_CARD.md).
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT, corpus_dir, load_scenes, split_of  # noqa: E402
from tlm.filters import TE_WORD, drop_reason, has_punctuation, is_loanword, multi_sentence  # noqa: E402
from tlm.plotstyle import INK2, SERIES, plt, save  # noqa: E402
from tlm.romanizer import LOANWORDS  # noqa: E402

scenes = load_scenes()
split = split_of()
cdir = corpus_dir()
stats = {}

# --- 1. Cleaning funnel: subtitle blocks -> scenes -> kept --------------------------------------
raw_blocks = sum(open(f, encoding="utf-8", errors="ignore").read().count("-->") for f in cdir.glob("*/*_raw*.srt"))
clean_blocks = sum(open(f, encoding="utf-8", errors="ignore").read().count("-->") for f in cdir.glob("*/*_clean.srt"))
scene_lines = sum(len(s.lines) for s in scenes)
reasons = Counter(drop_reason(s.lines) for s in scenes)
kept_lines = sum(len(s.lines) for s in scenes if drop_reason(s.lines) is None)
stages = [("raw subtitle blocks", raw_blocks), ("after extractor cleaning", clean_blocks),
          ("merged into dialogue turns", scene_lines), ("after quality filters", kept_lines)]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
axes[0].barh([n for n, _ in stages][::-1], [v for _, v in stages][::-1], color=SERIES[0], height=0.55)
for i, (n, v) in enumerate(stages[::-1]):
    axes[0].text(v, i, f"  {v:,}", va="center", color=INK2, fontsize=9)
axes[0].set_xlim(0, raw_blocks * 1.22)
axes[0].xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
axes[0].set_title("Cleaning funnel (subtitle blocks / dialogue lines)")
dr = [(r, n) for r, n in reasons.most_common() if r]
axes[1].barh([r for r, _ in dr][::-1], [n for _, n in dr][::-1], color=SERIES[1], height=0.55)
for i, (r, n) in enumerate(dr[::-1]):
    axes[1].text(n, i, f"  {n:,}", va="center", color=INK2, fontsize=9)
axes[1].set_xlim(0, max(n for _, n in dr) * 1.3)
axes[1].set_title(f"Scenes removed: {sum(n for _, n in dr):,} of {len(scenes):,}")
axes[1].set_xlabel("scenes")
save(fig, "s0b_cleaning_funnel")
stats["funnel"] = dict(stages)
stats["dropped_scenes"] = {r: n for r, n in dr}

# --- 2. Caption source proxy: punctuation rate per movie --------------------------------------
# Auto-generated (speech-recognition) captions carry almost no punctuation; manual ones do.
per_movie = {}
for s in scenes:
    per_movie.setdefault(s.movie, []).extend(s.lines)
punct = {m: sum(map(has_punctuation, ls)) / len(ls) for m, ls in per_movie.items()}
ms = sorted(punct, key=punct.get)
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.bar(range(len(ms)), [punct[m] for m in ms], color=[SERIES[1] if punct[m] < 0.1 else SERIES[0] for m in ms],
       width=0.8)
ax.axhline(0.1, color=INK2, linewidth=1, linestyle="--")
low = sum(1 for m in ms if punct[m] < 0.1)
ax.text(0, 0.12, f"{low} movies below 10%: likely auto-generated captions", color=INK2, fontsize=9)
ax.set_xticks([])
ax.set_xlabel(f"{len(ms)} movies, sorted")
ax.set_ylabel("share of lines with punctuation")
ax.set_title("Caption-source proxy: auto-generated captions have almost no punctuation")
save(fig, "s0b_caption_source_proxy")
stats["likely_auto_caption_movies"] = low

# --- 3. What a 'line' is: multi-sentence lines --------------------------------------------------
all_lines = [l for s in scenes for l in s.lines]
stats["share_lines_multi_sentence"] = round(sum(map(multi_sentence, all_lines)) / len(all_lines), 3)

# --- 4. English inside Telugu: lower and upper bound, by release year ---------------------------
by_year = {}
for s in scenes:
    if not s.year:
        continue
    ws = [w for l in s.lines for w in TE_WORD.findall(l)]
    d = by_year.setdefault(s.year, [0, 0, 0])
    d[0] += len(ws)
    d[1] += sum(w in LOANWORDS for w in ws)
    d[2] += sum(map(is_loanword, ws))
ys = sorted(by_year)
lo = [by_year[y][1] / by_year[y][0] for y in ys]
hi = [by_year[y][2] / by_year[y][0] for y in ys]
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.fill_between(ys, lo, hi, color=SERIES[0], alpha=0.18, linewidth=0)
ax.plot(ys, hi, color=SERIES[0], marker="o", markersize=4, label="upper bound: word ends in a consonant (virama)")
ax.plot(ys, lo, color=SERIES[1], marker="o", markersize=4, label="lower bound: known loanword list")
ax.set_xlabel("release year")
ax.set_ylabel("share of Telugu-script words")
ax.set_title("English words written in Telugu script (true rate lies in the shaded band)")
ax.legend(loc="upper left")
save(fig, "s0b_english_mixing")
tot = [sum(by_year[y][i] for y in ys) for i in range(3)]
stats["english_share_bounds"] = [round(tot[1] / tot[0], 4), round(tot[2] / tot[0], 4)]
cons_end = Counter(w for s in scenes for l in s.lines for w in TE_WORD.findall(l) if is_loanword(w))
stats["top_consonant_final_words"] = [w for w, _ in cons_end.most_common(25)]

# --- 5. Vocabulary: Zipf, growth, and unseen words in test movies ----------------------------
tr = Counter(w for s in scenes if split.get(s.movie) == "train" for l in s.lines for w in TE_WORD.findall(l))
te_words = [w for s in scenes if split.get(s.movie) == "test" for l in s.lines for w in TE_WORD.findall(l)]
oov_tok = sum(w not in tr for w in te_words) / len(te_words)
oov_type = len({w for w in te_words if w not in tr}) / len(set(te_words))
stream = [w for s in random.Random(0).sample(scenes, len(scenes)) for l in s.lines for w in TE_WORD.findall(l)]
seen, growth = set(), []
for i, w in enumerate(stream, 1):
    seen.add(w)
    if i % 5000 == 0:
        growth.append((i, len(seen)))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
freqs = sorted(tr.values(), reverse=True)
axes[0].loglog(range(1, len(freqs) + 1), freqs, color=SERIES[0])
axes[0].set_xlabel("word rank")
axes[0].set_ylabel("frequency")
axes[0].set_title(f"Zipf: {len(tr):,} distinct words; {sum(1 for f in freqs if f == 1) / len(freqs):.0%} appear once")
axes[1].plot([g[0] for g in growth], [g[1] for g in growth], color=SERIES[0])
axes[1].set_xlabel("words read")
axes[1].set_ylabel("distinct words seen")
axes[1].set_title(f"Vocabulary keeps growing ({oov_tok:.1%} of test words unseen)")
axes[1].xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
save(fig, "s0b_vocabulary")
stats.update({"vocab_train": len(tr), "hapax_share": round(sum(1 for f in freqs if f == 1) / len(freqs), 3),
              "test_oov_token_rate": round(oov_tok, 4), "test_oov_type_rate": round(oov_type, 4)})

(ROOT / "docs" / "results").mkdir(parents=True, exist_ok=True)
json.dump(stats, open(ROOT / "docs" / "results" / "s0b_stats.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(stats, indent=1, ensure_ascii=False))
