"""Stage 6f — the plain question: does the reply actually RESPOND to what was said? (yes / no)

    python stages/s6f_relevance.py          # lenient: does it react to the message at all?
    python stages/s6f_relevance.py strict   # strict: is the WHOLE reply a sensible message a person would send?

Matchup scores compare models with each other; this asks an absolute question per reply. A judge (Claude Opus via its CLI)
answers yes/no for each reply to the 20 showcase messages and the 150 unseen-movie lines. Claude only judges models that are
not Claude (ours, untuned sarvam-1, Gemini); Claude's own replies are judged by Gemini through the task file this script writes.
Writes docs/results/s6f_relevance.json.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402

F = ROOT / "data" / "realworld" / "frontier"
show = json.load(open(ROOT / "docs" / "eval" / "showcase.json", encoding="utf-8"))
movie = json.load(open(ROOT / "data" / "realworld" / "generated.json", encoding="utf-8"))["movie"]
test_ctx = [json.loads(l)["context"] for l in open(ROOT / "data" / "dialogue" / "test.jsonl", encoding="utf-8")][:len(movie)]
gem_movie = json.load(open(F / "gemini_replies_ctx.json", encoding="utf-8")) if (F / "gemini_replies_ctx.json").exists() else None

STRICT = ("For each item, someone said something in Telugu (sometimes mixed with English), and someone else replied. "
          "Be strict, like a human reader: is the WHOLE reply a sensible, coherent message that a real person would send in reply "
          "to what was said? If any part rambles, drifts to unrelated topics, contradicts itself or doesn't make sense, answer no. "
          "Return ONLY a JSON list of {n} answers, \"yes\" or \"no\", in the same order.\n\n{items}")
MODE = sys.argv[1] if len(sys.argv) > 1 else "lenient"
PROMPT = ("For each item, someone said something in Telugu (sometimes mixed with English), and someone else replied. "
          "Does the reply actually RESPOND to what was said, as a sensible next line in that conversation (it answers, reacts to "
          "or follows up on it), rather than ignoring it or going off-topic? Judge relevance only, not style. "
          "Return ONLY a JSON list of {n} answers, \"yes\" or \"no\", in the same order.\n\n{items}")


def judge(items):
    out = []
    for i in range(0, len(items), 25):
        c = items[i:i + 25]
        txt = "\n\n".join(f"{k + 1}. Said: {s}\n   Reply: {r}" for k, (s, r) in enumerate(c))
        ans = []
        for _ in range(3):
            o = subprocess.run(["claude", "-p", (STRICT if MODE == "strict" else PROMPT).format(n=len(c), items=txt), "--output-format", "text"],
                               cwd=F / "cli_workdir", capture_output=True, text=True, timeout=900).stdout
            m = re.search(r"\[.*\]", o, flags=re.S)
            try:
                ans = json.loads(m.group()) if m else []
            except json.JSONDecodeError:
                ans = []
            if len(ans) == len(c):
                break
        out += [str(a).strip().lower().startswith("y") for a in ans] if len(ans) == len(c) else [None] * len(c)
    return out


def _unused():
    out = []
    return out


res = {}
msgs = [m for _, m in show["messages"]]
for model in ["ours", "sarvam-1 untuned", "Gemini 3.8 Flash"]:
    if model not in show["answers"]:
        continue
    said = [show["answers"][model][i].get("telugu") or msgs[i] for i in range(len(msgs))]
    v = judge(list(zip(said, [a.get("reply", "") for a in show["answers"][model]])))
    res.setdefault("showcase (20 everyday messages)", {})[model] = {"yes": sum(x is True for x in v), "n": len(v),
                                                                   "per_message": v}
    print("showcase", model, sum(x is True for x in v), "/", len(v), flush=True)
movie_sets = {"ours": [m["tuned"] for m in movie], "sarvam-1 untuned": [m["base"] for m in movie],
              "real movie line": [m["real"] for m in movie]}
if gem_movie:
    movie_sets["Gemini 3.8 Flash"] = gem_movie
said = [" / ".join(c) for c in test_ctx]
for model, reps in movie_sets.items():
    v = judge(list(zip(said, reps)))
    res.setdefault("movie lines (150 unseen scenes)", {})[model] = {"yes": sum(x is True for x in v), "n": len(v)}
    print("movie", model, sum(x is True for x in v), "/", len(v), flush=True)
json.dump(res, open(ROOT / "docs" / "results" / f"s6f_relevance{'_strict' if MODE == 'strict' else ''}.json", "w"), indent=1)

# Claude's own replies must be judged by a different model: a task file for Gemini
D = Path.home() / "Downloads" / "Tenglish_LM" / "frontier_tasks"
cl = show["answers"]["Claude Opus 5.5"]
json.dump({"instruction": PROMPT.split("\n\n")[0].format(n=len(cl)) + " Answer for the items in 'items'.",
           "items": [{"said": cl[i].get("telugu") or msgs[i], "reply": cl[i]["reply"]} for i in range(len(cl))]},
          open(D / "6_relevance_claude.json", "w"), ensure_ascii=False, indent=1)
print("DONE")
