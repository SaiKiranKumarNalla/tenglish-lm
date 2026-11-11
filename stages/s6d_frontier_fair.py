"""Stage 6d — a fairer frontier comparison: same context as our model, several Claude sizes, and a few-shot variant.

    python stages/s6d_frontier_fair.py

Fixes an unfairness in Stage 6b: there, frontier models saw only the LAST line, while our model saw the previous 1-2
lines. Here every model gets the same 1-2 lines of context.
  * Claude Haiku 5.5, Sonnet 5.5, Opus 5.5 (via `claude -p --model`) on normalization (300 Dakshina sentences, scored
    against the true Telugu) and on replies (150 unseen-movie contexts)
  * "more context": Opus also gets 6 example exchanges from the TRAINING movies first (few-shot), to see how much
    guidance changes its replies
Outputs stay in data/realworld/frontier/ (git-ignored); scores go to docs/results/s6d_frontier_fair.json.
"""
import json
import random
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402

F = ROOT / "data" / "realworld" / "frontier"
WD = F / "cli_workdir"
dak = json.load(open(F / "dakshina_300.json", encoding="utf-8"))
G = json.load(open(ROOT / "data" / "realworld" / "generated.json", encoding="utf-8"))
test = [json.loads(l) for l in open(ROOT / "data" / "dialogue" / "test.jsonl", encoding="utf-8")]
ctxs = [r["context"] for r in test[:len(G["movie"])]]  # the same contexts our model was given in Stage 6
assert all(c[-1] == m["message"] for c, m in zip(ctxs, G["movie"]))
train = [json.loads(l) for l in open(ROOT / "data" / "dialogue" / "train.jsonl", encoding="utf-8")]
shots = [r for r in random.Random(3).sample(train, 200) if 15 <= len(r["target"]) <= 80][:6]

MODELS = {"Claude Haiku 5.5": "claude-haiku-5-5", "Claude Sonnet 5.5": "claude-sonnet-5-5", "Claude Opus 5.5": "claude-opus-5-5"}
NORM = ("Convert each romanized Telugu sentence below into Telugu script. Keep the meaning and words exactly; do not translate "
        "or add anything. Return ONLY a JSON list of {n} strings, in the same order.\n\n{items}")
REPLY = ("Each item is a short excerpt of a Telugu movie conversation (one or two lines, the last line is the most recent). "
         "For each, write the next line: a short, natural reply in spoken, colloquial Telugu (Telugu script, under 20 words), "
         "as a character would really say it.{shots} Return ONLY a JSON list of {n} strings, in the same order.\n\n{items}")


def ask(model_id, prompt, n):
    for _ in range(3):
        out = subprocess.run(["claude", "-p", prompt, "--model", model_id, "--output-format", "text"], cwd=WD,
                             capture_output=True, text=True, timeout=900).stdout
        m = re.search(r"\[.*\]", out, flags=re.S)
        if m:
            try:
                lst = json.loads(m.group())
                if len(lst) == n:
                    return [str(x) for x in lst]
            except json.JSONDecodeError:
                pass
    return [""] * n


def batched(model_id, items, make, label):
    out = []
    for i in range(0, len(items), 25):
        c = items[i:i + 25]
        out += ask(model_id, make(c), len(c))
        print(f"{label}: {min(i + 25, len(items))}/{len(items)}", flush=True)
    return out


def numbered(xs):
    return "\n".join(f"{i + 1}. {x}" for i, x in enumerate(xs))


def fmt_ctx(c):
    return " / ".join(f'"{line}"' for line in c)


def word_acc(preds):
    hit = tot = 0
    for p, (_, g) in zip(preds, dak):
        pw, gw = re.findall(r"[ఀ-౿]+", p), re.findall(r"[ఀ-౿]+", g)
        hit += sum(x == y for x, y in zip(pw, gw))
        tot += len(gw)
    return round(hit / tot, 4)


SHOTS = (" Here are examples of real exchanges from Telugu movies, to show the style:\n" +
         "\n".join(f"   {fmt_ctx(s['context'])}  ->  \"{s['target']}\"" for s in shots) + "\n")
res = {}
for name, mid in MODELS.items():
    key = mid.replace("claude-", "").replace("-5-5", "")
    nf = F / f"claude_{key}_normalization.json"
    if mid == "claude-opus-5-5" and (F / "claude_normalization.json").exists():
        preds = json.load(open(F / "claude_normalization.json", encoding="utf-8"))  # Stage 6b already ran Opus
    else:
        preds = batched(mid, [r for r, _ in dak], lambda c: NORM.format(n=len(c), items=numbered(c)), f"{name} normalize")
    json.dump(preds, open(nf, "w"), ensure_ascii=False)
    replies = batched(mid, ctxs, lambda c: REPLY.format(shots="", n=len(c), items=numbered([fmt_ctx(x) for x in c])),
                      f"{name} replies")
    json.dump(replies, open(F / f"claude_{key}_replies_ctx.json", "w"), ensure_ascii=False)
    res[name] = {"normalization_word_accuracy": word_acc(preds), "replies": sum(1 for r in replies if r.strip())}
    print(name, res[name], flush=True)

few = batched("claude-opus-5-5", ctxs, lambda c: REPLY.format(shots=SHOTS, n=len(c), items=numbered([fmt_ctx(x) for x in c])),
              "Opus few-shot replies")
json.dump(few, open(F / "claude_opus_replies_fewshot.json", "w"), ensure_ascii=False)
res["Claude Opus 5.5 + 6 movie examples"] = {"replies": sum(1 for r in few if r.strip())}
json.dump({"few_shot_examples": shots}, open(F / "fewshot_examples.json", "w"), ensure_ascii=False, indent=1)
json.dump(res, open(ROOT / "docs" / "results" / "s6d_frontier_fair.json", "w"), indent=1)
print(json.dumps(res, indent=1))
print("DONE")
