"""Stage 6b — cross-judging: a frontier CLI judges our replies vs ANOTHER frontier model's (never its own).

    python stages/s6b_cross_judge.py --judge claude --opponent gemini
Each pair is judged in both orders (25 pairs per call). Appends to docs/results/s6b_cross.json.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--judge", default="claude")
ap.add_argument("--opponent", default="gemini")
a = ap.parse_args()
assert a.judge != a.opponent, "a model never judges its own replies"
F = ROOT / "data" / "realworld" / "frontier"
movie = json.load(open(ROOT / "data" / "realworld" / "generated.json", encoding="utf-8"))["movie"]
opp = json.load(open(F / f"{a.opponent}_replies.json", encoding="utf-8"))
JUDGE = ("For each item, a person said something in Telugu, and there are two candidate replies A and B. "
         "Which reply is the better next line in a natural, spoken Telugu conversation (relevant, fluent, how people "
         "really talk)? Return ONLY a JSON list of {n} letters, \"A\" or \"B\", in the same order.\n\n{items}")


def ask(prompt):
    cmd = {"claude": ["claude", "-p", prompt, "--output-format", "text"],
           "gemini": ["gemini", "--skip-trust", "-p", prompt]}[a.judge]
    for _ in range(3):
        out = subprocess.run(cmd, cwd=F / "cli_workdir", capture_output=True, text=True, timeout=600).stdout
        m = re.search(r"\[.*\]", out, flags=re.S)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return []


def run(pairs):
    v = []
    for i in range(0, len(pairs), 25):
        c = pairs[i:i + 25]
        r = ask(JUDGE.format(n=len(c), items="\n\n".join(f"{k + 1}. Said: {m}\n   A: {x}\n   B: {y}" for k, (m, x, y) in enumerate(c))))
        v += [str(x).strip().upper()[:1] for x in r] if len(r) == len(c) else ["?"] * len(c)
    return v


pairs = [(m["message"], m["tuned"], o) for m, o in zip(movie, opp) if o.strip()]
v1, v2 = run(pairs), run([(m, y, x) for m, x, y in pairs])
w = {"ours": 0, a.opponent: 0, "tie/inconsistent": 0}
for p, q in zip(v1, v2):
    w["ours" if (p, q) == ("A", "B") else a.opponent if (p, q) == ("B", "A") else "tie/inconsistent"] += 1
n = len(pairs)
res = {**w, "n": n, "ours_win_rate": round(w["ours"] / n, 4), f"{a.opponent}_win_rate": round(w[a.opponent] / n, 4)}
p = ROOT / "docs" / "results" / "s6b_cross.json"
allr = json.load(open(p)) if p.exists() else {}
allr[f"{a.judge} judges: fine-tuned vs {a.opponent}"] = res
json.dump(allr, open(p, "w"), indent=1)
print(json.dumps(res))
