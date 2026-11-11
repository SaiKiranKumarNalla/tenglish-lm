"""Stage 6b (manual route) — task files for chat apps (ChatGPT app, Antigravity/Gemini) and a scorer for their answers.

    python stages/s6b_manual_tasks.py make  --dir ~/Downloads/Tenglish_LM/frontier_tasks
    python stages/s6b_manual_tasks.py score --dir ~/Downloads/Tenglish_LM/frontier_tasks

"make" writes three self-contained JSON files (instructions + items). For each app, upload a file, paste the
instruction, and save the app's answer as  <app>_<task>.json  in the same folder (e.g. chatgpt_1_normalize.json):
a JSON list in the same order as the items. The A/B order in the judging file is shuffled; the key stays in
_key_3_judge.json (do not upload it). "score" reads whatever answer files exist and writes docs/results/s6b_manual.json.
Only Dakshina sentences (public research data) and short movie-test lines are included; no YouTube comments.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("action", choices=["make", "score"])
ap.add_argument("--dir", required=True)
a = ap.parse_args()
D = Path(a.dir).expanduser()
D.mkdir(parents=True, exist_ok=True)
F = ROOT / "data" / "realworld" / "frontier"

if a.action == "make":
    dak = json.load(open(F / "dakshina_300.json", encoding="utf-8"))
    movie = json.load(open(ROOT / "data" / "realworld" / "generated.json", encoding="utf-8"))["movie"]
    claude = json.load(open(F / "claude_replies.json", encoding="utf-8"))
    tasks = {
        "1_normalize": {
            "instruction": "Convert each romanized Telugu sentence in 'items' into Telugu script. Keep the words and meaning "
                           "exactly; do not translate or add anything. Answer with ONLY a JSON list of 300 strings, "
                           "in the same order as 'items'.",
            "items": [r for r, _ in dak]},
        "2_reply": {
            "instruction": "Each item is the last line someone said in a Telugu movie conversation. For each, write the "
                           "next line: a short, natural reply in spoken, colloquial Telugu (Telugu script, under 20 "
                           "words), as a character would really say it. Answer with ONLY a JSON list of 150 strings, "
                           "in the same order as 'items'.",
            "items": [m["message"] for m in movie]},
    }
    rng, items, key = random.Random(11), [], []
    for i, (m, c) in enumerate(zip(movie, claude)):
        if not c.strip():
            continue
        for order in (0, 1):  # every pair appears twice, once in each order
            ours_first = rng.random() < 0.5 if order == 0 else not key[-1]["ours_is_A"]
            a_, b_ = (m["tuned"], c) if ours_first else (c, m["tuned"])
            items.append({"said": m["message"], "A": a_, "B": b_})
            key.append({"pair": i, "ours_is_A": ours_first})
    perm = list(range(len(items)))
    rng.shuffle(perm)
    tasks["3_judge"] = {
        "instruction": f"For each item, a person said something in Telugu ('said') and there are two candidate replies A and B. "
                       f"Which reply is the better next line in a natural, spoken Telugu conversation (relevant, fluent, "
                       f"how people really talk)? Answer with ONLY a JSON list of {len(items)} letters, \"A\" or \"B\", "
                       f"in the same order as 'items'.",
        "items": [items[p] for p in perm]}
    json.dump([key[p] for p in perm], open(D / "_key_3_judge.json", "w"), indent=1)
    for name, t in tasks.items():
        json.dump(t, open(D / f"{name}.json", "w"), ensure_ascii=False, indent=1)
    (D / "README.txt").write_text(
        "For each app (ChatGPT, Gemini/Antigravity):\n"
        "  1. Upload 1_normalize.json, paste its 'instruction', save the answer as <app>_1_normalize.json here\n"
        "  2. Same for 2_reply.json   -> <app>_2_reply.json\n"
        "  3. Same for 3_judge.json   -> <app>_3_judge.json\n"
        "Use 'chatgpt' or 'gemini' as <app>. The answer file must be just the JSON list.\n"
        "Do NOT upload _key_3_judge.json.\n", encoding="utf-8")
    print("wrote", sorted(p.name for p in D.iterdir()))

else:
    dak = json.load(open(F / "dakshina_300.json", encoding="utf-8"))
    key = json.load(open(D / "_key_3_judge.json"))
    out = {}
    for app in ["chatgpt", "gemini"]:
        r = {}
        f1, f2, f3 = (D / f"{app}_{t}.json" for t in ["1_normalize", "2_reply", "3_judge"])
        if f1.exists():
            pred = json.load(open(f1, encoding="utf-8"))
            hit = tot = 0
            for p, (_, g) in zip(pred, dak):
                pw, gw = re.findall(r"[ఀ-౿]+", str(p)), re.findall(r"[ఀ-౿]+", g)
                hit += sum(x == y for x, y in zip(pw, gw))
                tot += len(gw)
            r["normalization_word_accuracy"] = round(hit / tot, 4)
            r["normalization_items_answered"] = len(pred)
        if f2.exists():
            replies = json.load(open(f2, encoding="utf-8"))
            json.dump(replies, open(F / f"{app}_replies.json", "w"), ensure_ascii=False)
            r["replies_answered"] = len(replies)
        if f3.exists():
            v = [str(x).strip().upper()[:1] for x in json.load(open(f3))]
            by_pair = {}
            for k, ans in zip(key, v):
                ours_wins = (ans == "A") == k["ours_is_A"] if ans in "AB" and ans else None
                by_pair.setdefault(k["pair"], []).append(ours_wins)
            w = {"ours": 0, "claude": 0, "tie/inconsistent": 0}
            for votes in by_pair.values():
                w["ours" if votes == [True, True] else "claude" if votes == [False, False] else "tie/inconsistent"] += 1
            n = len(by_pair)
            r["judge: fine-tuned vs claude"] = {**w, "n": n, "ours_win_rate": round(w["ours"] / n, 4),
                                                "claude_win_rate": round(w["claude"] / n, 4)}
        if r:
            out[app] = r
    json.dump(out, open(ROOT / "docs" / "results" / "s6b_manual.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
