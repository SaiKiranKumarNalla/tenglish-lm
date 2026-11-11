"""Stage 7 — cleaner conversation data: repaired movie scenes + style-conditioned new conversations.

    TLM_CORPUS=... python stages/s7_conversation_data.py make     --dir ~/Downloads/Tenglish_LM/claude_data_tasks
    TLM_CORPUS=... python stages/s7_conversation_data.py validate --dir ~/Downloads/Tenglish_LM/claude_data_tasks

Stage 6 showed that only ~45% of real "next lines" in subtitles are sensible replies (merged speakers, scene jumps, songs).
"make" writes pilot batches for an external writer model (instructions in CLAUDE_INSTRUCTIONS.md):
  A_repair/batch_XX.json     500 training-movie scenes to split into speakers, lightly correct, and flag real replies
  B_style/requests_XX.json   500 requests for new conversations: 300 movie-style (genre x director style x era x emotion x
                             dialect) and 200 everyday chats (topic x relationship x dialect)
"validate" checks the returned out_*.json files: format, Telugu script, reply flags, faithfulness of repairs to the original
words, and that new conversations don't copy subtitle lines. Clean conversation pairs go to data/conversations/ (git-ignored).
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT, load_scenes, split_of  # noqa: E402
from tlm.filters import TELUGU, drop_reason  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("action", choices=["make", "validate"])
ap.add_argument("--dir", required=True)
ap.add_argument("--n-repair", type=int, default=500)
ap.add_argument("--n-style", type=int, default=500)
a = ap.parse_args()
D = Path(a.dir).expanduser()
(D / "A_repair").mkdir(parents=True, exist_ok=True)
(D / "B_style").mkdir(parents=True, exist_ok=True)
rng = random.Random(7)

DIRECTORS = ["Trivikram", "Sekhar Kammula", "Sukumar", "Koratala Shiva", "Boyapati Srinu", "Ram Gopal Varma", "Rajamouli", "none"]
GENRES = ["family drama", "romance", "comedy", "mass action", "emotional drama", "college / youth", "village drama", "crime thriller"]
ERAS = ["2000s", "2010s", "2020s"]
EMOTIONS = ["happy", "funny / teasing", "angry / warning", "sad / heartbreak", "romantic", "tense", "inspiring", "sarcastic"]
DIALECTS = ["Standard", "Standard", "Telangana", "Rayalaseema", "Coastal Andhra"]
SETTINGS = ["at home over dinner", "on a college campus", "at a tea stall", "in a village panchayat", "on a phone call",
            "at a wedding", "in an office", "on a bike ride", "at a hospital", "at night on a terrace", "in a police station",
            "at a bus stop"]
TOPICS = ["making weekend plans", "running late", "exam results", "food and where to eat", "an apology after a fight",
          "office deadline and boss", "a cricket match", "a family function", "missing someone", "teasing a friend",
          "money problems", "a new job", "a movie release", "health and a doctor visit", "festival preparations",
          "travel plans", "a breakup", "parents and marriage talk", "a phone or gadget problem", "bargaining while shopping"]
RELATIONS = ["two close friends", "mother and son", "father and daughter", "siblings", "a couple", "colleagues",
             "roommates", "neighbours", "a young person and a grandparent", "cousins"]

if a.action == "make":
    split = split_of()
    scenes = [s for s in load_scenes() if split.get(s.movie) == "train" and drop_reason(s.lines) is None and len(s.lines) >= 3]
    pick = rng.sample(scenes, a.n_repair)
    meta = {}
    import csv
    from tlm.corpus import corpus_dir
    for f in corpus_dir().glob("*/*_scenes.csv"):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            meta[r["movie_name"]] = {"director": r.get("director", ""), "genre": r.get("movie_genre", ""), "year": r.get("year", "")}
            break
    items = [{"scene_id": s.scene_id, "movie": s.movie, **meta.get(s.movie, {}), "lines": s.lines} for s in pick]
    for b in range(0, len(items), 50):
        json.dump(items[b:b + 50], open(D / "A_repair" / f"batch_{b // 50 + 1:02d}.json", "w"), ensure_ascii=False, indent=1)
    reqs = []
    for i in range(a.n_style):
        if i < int(a.n_style * 0.6):
            r = {"kind": "movie", "genre": rng.choice(GENRES), "director_style": rng.choice(DIRECTORS), "era": rng.choice(ERAS),
                 "emotion": rng.choice(EMOTIONS), "dialect": rng.choice(DIALECTS), "setting": rng.choice(SETTINGS),
                 "characters": rng.choice(["hero and his friend", "hero and heroine", "father and son", "villain and hero",
                                           "two friends", "mother and daughter", "a boss and an employee"]),
                 "turns": rng.choice([4, 6, 8])}
        else:
            r = {"kind": "everyday", "topic": rng.choice(TOPICS), "characters": rng.choice(RELATIONS),
                 "dialect": rng.choice(DIALECTS), "emotion": rng.choice(EMOTIONS[:6]),
                 "medium": rng.choice(["WhatsApp chat", "face to face", "phone call"]), "turns": rng.choice([4, 6, 8])}
        reqs.append(r)
    rng.shuffle(reqs)
    for i, r in enumerate(reqs):
        r["request_id"] = f"B{i // 50 + 1:02d}_{i % 50 + 1:03d}"
    for b in range(0, len(reqs), 50):
        json.dump(reqs[b:b + 50], open(D / "B_style" / f"requests_{b // 50 + 1:02d}.json", "w"), ensure_ascii=False, indent=1)
    (D / "README.txt").write_text(
        "How to run this with Claude (on the other account):\n"
        "  1. Start a new conversation and upload CLAUDE_INSTRUCTIONS.md. Ask Claude to read it.\n"
        "  2. Upload A_repair/batch_01.json and say: 'Do Task A for this batch.' Save the JSON answer as A_repair/out_batch_01.json\n"
        "     Repeat for batch_02 ... batch_10 (a fresh conversation every few batches keeps answers complete).\n"
        "  3. Upload B_style/requests_01.json and say: 'Do Task B for this batch.' Save as B_style/out_requests_01.json\n"
        "     Repeat for requests_02 ... requests_10.\n"
        "  4. Tell the project script to validate: python stages/s7_conversation_data.py validate --dir <this folder>\n"
        "The answer files must contain only the JSON list.\n", encoding="utf-8")
    print(f"wrote {len(items)} scenes in {(len(items) + 49) // 50} repair batches and {len(reqs)} style requests "
          f"({sum(r['kind'] == 'movie' for r in reqs)} movie, {sum(r['kind'] == 'everyday' for r in reqs)} everyday)")

else:
    from tlm.corpus import load_scenes as _ls

    subtitle_ngrams = set()
    for s in _ls():
        for line in s.lines:
            w = line.split()
            subtitle_ngrams.update(tuple(w[i:i + 6]) for i in range(len(w) - 5))
    out = ROOT / "data" / "conversations"
    out.mkdir(parents=True, exist_ok=True)
    stats, pairs = Counter(), []

    def te_ok(t):
        return len(TELUGU.findall(t)) >= max(2, 0.4 * len(re.sub(r"\s", "", t)))

    for f in sorted((D / "A_repair").glob("out_batch_*.json")):
        src = {x["scene_id"]: x for x in json.load(open(D / "A_repair" / f.name.replace("out_", ""), encoding="utf-8"))}
        for sc in json.load(open(f, encoding="utf-8")):
            stats["A scenes"] += 1
            if not sc.get("usable") or sc.get("scene_id") not in src:
                stats["A unusable / unknown id"] += 1
                continue
            orig = Counter(w for l in src[sc["scene_id"]]["lines"] for w in l.split())
            turns = sc.get("turns", [])
            words = [w for t in turns for w in t.get("text", "").split()]
            kept = sum(1 for w in words if w in orig) / max(len(words), 1)
            stats["A faithfulness sum"] += kept
            if kept < 0.6:  # a repair should mostly reuse the original words
                stats["A rejected: too rewritten"] += 1
                continue
            for p, t in zip(turns, turns[1:]):
                stats["A turn pairs"] += 1
                if t.get("responds") is True and te_ok(p["text"]) and te_ok(t["text"]):
                    stats["A pairs kept (real replies)"] += 1
                    pairs.append({"source": "repaired", "scene_id": sc["scene_id"], "context": [p["text"]], "reply": t["text"]})
    for f in sorted((D / "B_style").glob("out_requests_*.json")):
        reqs = {x["request_id"]: x for x in json.load(open(D / "B_style" / f.name.replace("out_", ""), encoding="utf-8"))}
        for conv in json.load(open(f, encoding="utf-8")):
            stats["B conversations"] += 1
            r = reqs.get(conv.get("request_id"))
            turns = conv.get("turns", [])
            if not r or len(turns) < 2:
                stats["B unusable"] += 1
                continue
            copied = sum(1 for t in turns for i in range(len(t["text"].split()) - 5)
                         if tuple(t["text"].split()[i:i + 6]) in subtitle_ngrams)
            if copied:
                stats["B rejected: copies subtitle text"] += 1
                continue
            for k in range(1, len(turns)):
                if te_ok(turns[k]["text"]):
                    pairs.append({"source": f"style:{r['kind']}", "request": r, "context": [t["text"] for t in turns[max(0, k - 2):k]],
                                  "reply": turns[k]["text"]})
                    stats["B pairs kept"] += 1
    if stats["A scenes"]:
        stats["A avg faithfulness"] = round(stats.pop("A faithfulness sum") / max(stats["A scenes"] - stats["A unusable / unknown id"], 1), 3)
    with open(out / "pilot_pairs.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs)
    json.dump(dict(stats), open(ROOT / "docs" / "results" / "s7_pilot_data.json", "w"), indent=1)
    print(json.dumps(dict(stats), indent=1))
