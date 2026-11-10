"""Stage 1 data prep: movie_corpus/<movie>/<movie>_scenes.csv -> data/processed/{train,val,test}.jsonl

Usage:
    python mlm/prepare_data.py --corpus "<path to movie_corpus>" --out data/processed

Splits are by movie, from data/splits/split_manifest.json, so no scene from a held-out movie is ever trained on.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

TELUGU = re.compile(r"[ఀ-౿]")
LATIN = re.compile(r"[A-Za-z]")
# Lines that are almost certainly not dialogue: devotional/song lyric markers and leftover tags
NOISE = re.compile(r"[\[\(♪♫]|సంగీతం|పాట")


def clean_turn(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip(" -–—")
    return t


def keep_scene(turns: list[str]) -> tuple[bool, str]:
    text = " ".join(turns)
    te, la = len(TELUGU.findall(text)), len(LATIN.findall(text))
    if len(turns) < 2:
        return False, "too_few_turns"
    if te < 20:
        return False, "too_short"
    if la > te:  # mostly English / broken ASR
        return False, "not_telugu"
    if any(NOISE.search(t) for t in turns):
        return False, "noise_tag"
    # Song-lyric heuristic: many short lines with high word repetition
    words = text.split()
    if len(words) >= 12 and len(set(words)) / len(words) < 0.45:
        return False, "repetitive_lyric"
    return True, "ok"


def to_example(row: dict) -> dict | None:
    turns = [clean_turn(t) for t in row["dialogue_telugu"].split("//")]
    turns = [t for t in turns if t]
    ok, why = keep_scene(turns)
    if not ok:
        return {"_drop": why}
    header = f"భావోద్వేగం: {row.get('scene_emotion', '')} | మాండలికం: {row.get('scene_dialect', '')}"
    body = "\n".join(f"- {t}" for t in turns)
    return {
        "scene_id": row["scene_id"],
        "movie_name": row["movie_name"],
        "text": f"{header}\n{body}",  # EOS is appended at tokenization time
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--manifest", default="data/splits/split_manifest.json")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()

    manifest = json.load(open(a.manifest, encoding="utf-8"))
    movie_to_split = {m: s for s, p in manifest["partitions"].items() for m in p["movies"]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    writers = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8") for s in manifest["partitions"]}

    seen, stats = set(), Counter()
    for csv_path in sorted(Path(a.corpus).glob("*/*_scenes.csv")):
        for row in csv.DictReader(open(csv_path, encoding="utf-8")):
            split = movie_to_split.get(row["movie_name"])
            if split is None:
                stats["unknown_movie"] += 1
                continue
            ex = to_example(row)
            if "_drop" in ex:
                stats[f"drop_{ex['_drop']}"] += 1
                continue
            # Exact-duplicate removal across the whole corpus (recaps, repeated title cards)
            h = hashlib.md5(ex["text"].split("\n", 1)[1].encode()).hexdigest()
            if h in seen:
                stats["drop_duplicate"] += 1
                continue
            seen.add(h)
            writers[split].write(json.dumps(ex, ensure_ascii=False) + "\n")
            stats[f"kept_{split}"] += 1
    for w in writers.values():
        w.close()
    print(json.dumps(dict(sorted(stats.items())), indent=2))


if __name__ == "__main__":
    main()
