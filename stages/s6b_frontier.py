"""Stage 6b — compare with frontier models through their command-line apps (no API keys; your own CLI logins).

    python stages/s6b_frontier.py --models claude [gemini] --generated data/realworld/generated.json

For each available CLI (Claude Code `claude -p`, Gemini CLI `gemini -p`):
  1. normalization   300 real Dakshina romanized sentences -> Telugu script; scored against the TRUE Telugu
  2. replies         the 150 unseen-movie contexts used in Stage 6 -> next line in spoken Telugu
  3. judging         as a judge on comparisons that don't involve its own replies (real vs random reply, ours vs
                     untuned sarvam-1), each pair in both orders, to measure agreement with the Sarvam-M judge
Only public research data (Dakshina) and short movie-test lines are sent; YouTube comments are NOT sent.
Prompts are batched (25 items per call). Raw responses stay in data/realworld/frontier/ (git-ignored).
"""
import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["claude"])
ap.add_argument("--generated", default=str(ROOT / "data" / "realworld" / "generated.json"))
ap.add_argument("--batch", type=int, default=25)
a = ap.parse_args()
OUT = ROOT / "data" / "realworld" / "frontier"
OUT.mkdir(parents=True, exist_ok=True)
WORKDIR = OUT / "cli_workdir"  # neutral folder, so the CLIs don't pick up this project's context
WORKDIR.mkdir(exist_ok=True)


def ask(model, prompt):
    cmd = {"claude": ["claude", "-p", prompt, "--output-format", "text"],
           "gemini": ["gemini", "--skip-trust", "-p", prompt]}[model]
    r = subprocess.run(cmd, cwd=WORKDIR, capture_output=True, text=True, timeout=600)
    return r.stdout


def ask_list(model, prompt, n, tries=3):
    for _ in range(tries):
        out = ask(model, prompt)
        m = re.search(r"\[.*\]", out, flags=re.S)
        if m:
            try:
                lst = json.loads(m.group())
                if len(lst) == n:
                    return [str(x) for x in lst]
            except json.JSONDecodeError:
                pass
    return [""] * n


def batched(model, items, make_prompt):
    out = []
    for i in range(0, len(items), a.batch):
        chunk = items[i:i + a.batch]
        out += ask_list(model, make_prompt(chunk), len(chunk))
        print(f"{model}: {min(i + a.batch, len(items))}/{len(items)}", flush=True)
    return out


# --- shared test items ---------------------------------------------------------------------------
dak = [l.rstrip("\n").split("\t") for l in open(ROOT / "data" / "normalizer" / "sent_dak_test.tsv", encoding="utf-8")]
dak = random.Random(7).sample(dak, 300)
json.dump(dak, open(OUT / "dakshina_300.json", "w"), ensure_ascii=False)
G = json.load(open(a.generated, encoding="utf-8"))
movie = G["movie"]

NORM = ("Convert each romanized Telugu sentence below into Telugu script. Keep the meaning and words exactly; "
        "do not translate or add anything. Return ONLY a JSON list of {n} strings, in the same order.\n\n{items}")
REPLY = ("Each item is the last line someone said in a Telugu movie conversation. For each, write the next line: "
         "a short, natural reply in spoken, colloquial Telugu (Telugu script, under 20 words), as a character "
         "would really say it. Return ONLY a JSON list of {n} strings, in the same order.\n\n{items}")
JUDGE = ("For each item, a person said something in Telugu, and there are two candidate replies A and B. "
         "Which reply is the better next line in a natural, spoken Telugu conversation (relevant, fluent, how people "
         "really talk)? Return ONLY a JSON list of {n} letters, \"A\" or \"B\", in the same order.\n\n{items}")


def numbered(xs):
    return "\n".join(f"{i + 1}. {x}" for i, x in enumerate(xs))


results = {}
for model in a.models:
    res = {}
    # 1. normalization against the true Telugu
    pred = batched(model, [r for r, _ in dak], lambda c: NORM.format(n=len(c), items=numbered(c)))
    hit = tot = 0
    for p, (_, g) in zip(pred, dak):
        pw, gw = re.findall(r"[ఀ-౿]+", p), re.findall(r"[ఀ-౿]+", g)
        hit += sum(x == y for x, y in zip(pw, gw))
        tot += len(gw)
    res["normalization_word_accuracy"] = round(hit / tot, 4)
    json.dump(pred, open(OUT / f"{model}_normalization.json", "w"), ensure_ascii=False)
    # 2. replies to the unseen-movie contexts
    replies = batched(model, [m["message"] for m in movie], lambda c: REPLY.format(n=len(c), items=numbered(c)))
    json.dump(replies, open(OUT / f"{model}_replies.json", "w"), ensure_ascii=False)

    # 3. as a judge on comparisons without its own replies, both orders
    def judge(pairs):
        def fmt(c):
            return JUDGE.format(n=len(c), items="\n\n".join(
                f"{i + 1}. Said: {m}\n   A: {x}\n   B: {y}" for i, (m, x, y) in enumerate(c)))
        v1 = batched(model, pairs, fmt)
        v2 = batched(model, [(m, y, x) for m, x, y in pairs], fmt)
        w = {"first": 0, "second": 0, "tie/inconsistent": 0}
        for p, q in zip(v1, v2):
            p, q = p.strip().upper()[:1], q.strip().upper()[:1]
            w["first" if (p, q) == ("A", "B") else "second" if (p, q) == ("B", "A") else "tie/inconsistent"] += 1
        n = len(pairs)
        return {**w, "n": n, "first_win_rate": round(w["first"] / n, 4), "second_win_rate": round(w["second"] / n, 4)}

    res["judge: real vs random (reliability)"] = judge([(m["message"], m["real"], m["random"]) for m in movie])
    res["judge: fine-tuned vs base (movie)"] = judge([(m["message"], m["tuned"], m["base"]) for m in movie])
    results[model] = res
    print(model, json.dumps(res), flush=True)

json.dump(results, open(ROOT / "docs" / "results" / "s6b_frontier.json", "w"), indent=1)
print("DONE")
