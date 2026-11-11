"""Stage 1 — noisy romanization: how many ways can one Telugu word be typed?

    TLM_CORPUS=... python stages/s1_romanizer.py

Writes docs/plots/s1_*.png, docs/results/s1_examples.md, and data/romanized/{train,val,test}.jsonl
(the romanized corpus used by later stages; kept out of git because it is derived from subtitles).
"""
import json
import random
import sys
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT, load_scenes, split_of, telugu_lines  # noqa: E402
from tlm.plotstyle import INK2, SERIES, plt, save  # noqa: E402
from tlm.romanizer import TELUGU_WORD, romanize, romanize_word  # noqa: E402
from tlm.filters import drop_reason  # noqa: E402

N_VARIANTS = 3  # noisy versions per line in the derived corpus
scenes = [s for s in load_scenes() if drop_reason(s.lines) is None]  # same quality filters as Stage 0b
lines = telugu_lines(scenes)
words = Counter(w for l in lines for w in TELUGU_WORD.findall(l))

# --- 1. Spelling diversity: distinct spellings per word, from 200 random draws ---------------
rng = random.Random(0)
top = [w for w, _ in words.most_common(3000)]
div = {}
for w in top:
    style_draws = [rng.choice([0.0, 0.0, 0.5, 1.5]) for _ in range(200)]
    div[w] = len({romanize_word(w, rng, False, s) for s in style_draws})
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
counts = Counter(div.values())
ks = sorted(counts)
axes[0].bar(ks, [counts[k] for k in ks], color=SERIES[0], width=0.7)
axes[0].set_xlabel("distinct spellings generated (200 draws)")
axes[0].set_ylabel("words (3,000 most frequent)")
axes[0].set_title("Most words get several plausible spellings")
by_len = {}
for w, d in div.items():
    by_len.setdefault(min(len(romanize_word(w, canonical=True)), 14), []).append(d)
ls = sorted(by_len)
axes[1].plot(ls, [sum(by_len[x]) / len(by_len[x]) for x in ls], color=SERIES[0], marker="o", markersize=5)
axes[1].set_xlabel("word length (canonical Latin letters, 14+ grouped)")
axes[1].set_ylabel("avg distinct spellings")
axes[1].set_title("Longer words have more ways to be typed")
save(fig, "s1_spelling_diversity")

# --- 2. Example gallery (markdown, renders fine on GitHub) -----------------------------------
freq_words = [w for w, _ in words.most_common(40) if len(w) >= 3][:12]
md = ["| Telugu | canonical | 5 random noisy spellings |", "|---|---|---|"]
for w in freq_words:
    noisy = sorted({romanize(w, random.Random(s)) for s in range(30)})[:5]
    md.append(f"| {w} | `{romanize_word(w, canonical=True)}` | {', '.join(f'`{n}`' for n in noisy)} |")
sample_lines = random.Random(1).sample(lines, 3)
md += ["", "| canonical line | one noisy version |", "|---|---|"]
for l in sample_lines:
    md.append(f"| `{romanize(l, canonical=True)}` | `{romanize(l, random.Random(7))}` |")
(ROOT / "docs" / "results").mkdir(parents=True, exist_ok=True)
(ROOT / "docs" / "results" / "s1_examples.md").write_text("\n".join(md) + "\n", encoding="utf-8")

# --- 3. Derived romanized corpus for later stages ---------------------------------------------
split = split_of()
out = ROOT / "data" / "romanized"
out.mkdir(parents=True, exist_ok=True)
files = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8") for s in ["train", "val", "test"]}
for sc in scenes:
    s = split.get(sc.movie)
    if not s:
        continue
    for i, l in enumerate(sc.lines):
        r = random.Random(zlib.crc32(f"{sc.scene_id}:{i}".encode()))  # stable across runs
        files[s].write(json.dumps({"scene_id": sc.scene_id, "te": l, "canon": romanize(l, canonical=True),
                                   "noisy": [romanize(l, r) for _ in range(N_VARIANTS)]},
                                  ensure_ascii=False) + "\n")
for f in files.values():
    f.close()
mean_div = sum(div.values()) / len(div)
json.dump({"mean_spellings_per_word_top3000": round(mean_div, 2),
           "share_words_with_1_spelling": round(counts[1] / len(div), 3)},
          open(ROOT / "docs" / "results" / "s1_stats.json", "w"), indent=1)
print(f"mean distinct spellings per frequent word: {mean_div:.2f}")
