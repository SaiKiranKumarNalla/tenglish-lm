"""Stage 5a — dialogue pairs: previous line(s) -> next line, in Telugu script.

    TLM_CORPUS=... python stages/s5_prepare.py

Within each (quality-filtered) scene, every turn after the first becomes a target, with up to two previous
turns as context. Movie-level split as always, so test pairs come from the 8 unseen movies.
Writes data/dialogue/{train,test}.jsonl (kept out of git).
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT, load_scenes, split_of  # noqa: E402
from tlm.filters import drop_reason  # noqa: E402

OUT = ROOT / "data" / "dialogue"
OUT.mkdir(parents=True, exist_ok=True)
split = split_of()
pairs = {"train": [], "test": []}
for s in load_scenes():
    sp = split.get(s.movie)
    if sp not in pairs or drop_reason(s.lines) is not None:
        continue
    for i in range(1, len(s.lines)):
        ctx, tgt = s.lines[max(0, i - 2):i], s.lines[i]
        if 2 <= len(tgt.split()) and len(tgt) <= 300 and sum(map(len, ctx)) <= 500:
            pairs[sp].append({"scene_id": s.scene_id, "context": ctx, "target": tgt})
random.Random(0).shuffle(pairs["train"])
pairs["test"] = random.Random(1).sample(pairs["test"], min(1500, len(pairs["test"])))
for k, v in pairs.items():
    with open(OUT / f"{k}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in v)
    print(k, len(v))
