"""Stage 6c — a public evaluation report: exact prompts, inputs and every model's outputs, side by side.

    python stages/s6c_report.py

Since the trained weights are private, this lets anyone inspect how the models actually behave. Writes docs/eval/:
  README.md                         how to read the report, plus all scores
  PROMPTS.md                        every prompt and generation setting used
  normalization_dakshina300.jsonl   300 real romanized sentences: input, true Telugu, each model's output
  normalization_examples.md         the first 40 of them as a readable table
  replies_movie_sample.jsonl / .md  50 unseen-movie lines: real reply, untuned sarvam-1, fine-tuned, frontier replies
Not included: YouTube comments (other people's words) or any model weights.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402

F = ROOT / "data" / "realworld" / "frontier"
E = ROOT / "docs" / "eval"
E.mkdir(parents=True, exist_ok=True)
RES = ROOT / "docs" / "results"
load = lambda p: json.load(open(p, encoding="utf-8")) if Path(p).exists() else None  # noqa: E731

dak = load(F / "dakshina_300.json")
movie = load(ROOT / "data" / "realworld" / "generated.json")["movie"]
norm_out = {"ours (7.6M normalizer)": load(F / "ours_normalization.json"), "Claude": load(F / "claude_normalization.json"),
            "ChatGPT": load(F / "chatgpt_normalization.json"), "Gemini": load(F / "gemini_normalization.json")}
norm_out = {k: v for k, v in norm_out.items() if v}
rep_out = {"Claude": load(F / "claude_replies.json"), "ChatGPT": load(F / "chatgpt_replies.json"),
           "Gemini": load(F / "gemini_replies.json")}
rep_out = {k: v for k, v in rep_out.items() if v}


def words(s):
    return re.findall(r"[ఀ-౿]+", s or "")


def word_acc(p, g):
    gw = words(g)
    return sum(x == y for x, y in zip(words(p), gw)) / max(len(gw), 1)


# ---- normalization: all 300 ----------------------------------------------------------------------
with open(E / "normalization_dakshina300.jsonl", "w", encoding="utf-8") as f:
    for i, (r, g) in enumerate(dak):
        f.write(json.dumps({"romanized_input": r, "true_telugu": g,
                            **{f"{m}": norm_out[m][i] for m in norm_out},
                            **{f"{m} word accuracy": round(word_acc(norm_out[m][i], g), 3) for m in norm_out}},
                           ensure_ascii=False) + "\n")
cols = list(norm_out)
md = ["# Romanized → Telugu script: 40 of the 300 test sentences", "",
      "Input typed by native speakers (Google's Dakshina dataset, test split). ✅ = every word right.", "",
      "| romanized input | true Telugu | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
for i, (r, g) in enumerate(dak[:40]):
    cells = [f"{norm_out[m][i]} {'✅' if words(norm_out[m][i]) == words(g) else ''}" for m in cols]
    md.append(f"| `{r}` | {g} | " + " | ".join(cells) + " |")
(E / "normalization_examples.md").write_text("\n".join(md) + "\n", encoding="utf-8")

# ---- replies: a 50-item sample ------------------------------------------------------------------
sample = list(range(0, min(50, len(movie))))
with open(E / "replies_movie_sample.jsonl", "w", encoding="utf-8") as f:
    for i in sample:
        m = movie[i]
        f.write(json.dumps({"last_line": m["message"], "real_next_line": m["real"], "sarvam-1 untuned": m["base"],
                            "fine-tuned (ours)": m["tuned"], **{k: v[i] for k, v in rep_out.items()}},
                           ensure_ascii=False) + "\n")
rc = ["sarvam-1 untuned", "fine-tuned (ours)"] + list(rep_out)
md = ["# Replies: 50 lines from unseen movies", "",
      "Each model sees the last line and writes the next one. *Real* is what the movie actually says next; it's one good answer "
      "among many, not the only right one.", "",
      "| last line | real next line | " + " | ".join(rc) + " |", "|---|---|" + "---|" * len(rc)]
for i in sample:
    m = movie[i]
    vals = [m["base"], m["tuned"]] + [rep_out[k][i] for k in rep_out]
    md.append(f"| {m['message']} | {m['real']} | " + " | ".join(v.replace('|', '/') for v in vals) + " |")
(E / "replies_movie_sample.md").write_text("\n".join(md) + "\n", encoding="utf-8")

# ---- prompts --------------------------------------------------------------------------------------
from stages_prompts import PROMPTS_MD  # noqa: E402  (kept in a separate file so it can be read on its own)

(E / "PROMPTS.md").write_text(PROMPTS_MD, encoding="utf-8")

# ---- scores ---------------------------------------------------------------------------------------
s6 = load(RES / "s6_real_world.json") or {}
s6b = load(RES / "s6b_frontier.json") or {}
s6g = load(RES / "s6b_gpu.json") or {}
s6m = load(RES / "s6b_manual.json") or {}
lines = ["# Public evaluation report", "",
         "The trained weights are private, so this folder shows **exactly what each model was asked and what it answered**, "
         "so anyone can judge the results.", "",
         "| File | Contents |", "|---|---|",
         "| [PROMPTS.md](PROMPTS.md) | every prompt and generation setting |",
         "| [normalization_examples.md](normalization_examples.md) | 40 sentences side by side (all 300 in `normalization_dakshina300.jsonl`) |",
         "| [replies_movie_sample.md](replies_movie_sample.md) | 50 replies side by side (`replies_movie_sample.jsonl`) |", "",
         "## Romanized → Telugu (300 real Dakshina sentences, word accuracy)", "", "| Model | words exactly right |", "|---|---|"]
for m in cols:
    acc = sum(word_acc(norm_out[m][i], g) * len(words(g)) for i, (_, g) in enumerate(dak)) / sum(len(words(g)) for _, g in dak)
    lines.append(f"| {m} | {acc:.1%} |")
lines += ["", "## Judged replies (each pair judged in both orders; a win needs both orders to agree)", "",
          "| Judge | Comparison | first wins | second wins | tie / inconsistent | pairs |", "|---|---|---|---|---|---|"]


def row(judge, name, r):
    lines.append(f"| {judge} | {name} | {r.get('first_win_rate', r.get('ours_win_rate', 0)):.0%} | "
                 f"{r.get('second_win_rate', r.get('claude_win_rate', 0)):.0%} | "
                 f"{r['tie/inconsistent'] / r['n']:.0%} | {r['n']} |")


for name, r in (s6.get("pairwise") or {}).items():
    row("Sarvam-M", name, r)
for name, r in s6g.items():
    if isinstance(r, dict) and "n" in r:
        row("Sarvam-M", name.replace("Sarvam-M judge: ", ""), r)
for model, res in s6b.items():
    for name, r in res.items():
        if name.startswith("judge:"):
            row(model.capitalize(), name.replace("judge: ", ""), r)
for app, res in s6m.items():
    for name, r in res.items():
        if name.startswith("judge:"):
            row(app.capitalize(), name.replace("judge: ", "") + " (first = ours)", r)
(E / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
