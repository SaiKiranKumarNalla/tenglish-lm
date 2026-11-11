"""Stage 6a — a real-world test set: romanized Telugu typed by real people (public YouTube comments).

    TLM_CORPUS=... python stages/s6_collect_comments.py [--per-video 300]

For each movie video in the corpus, fetch up to N top-level public comments (yt-dlp, no API key) and keep those
that look like romanized Telugu: Latin script, 4-40 words, and at least two words the language-ID model calls Telugu.
Privacy: author names, ids and timestamps are DROPPED at collection; only the comment text and the movie it was
posted under are kept. The text stays local (data/realworld/, git-ignored); the repository only gets statistics.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import yt_dlp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT, corpus_dir  # noqa: E402
from tlm.langid import LangID  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--per-video", type=int, default=300)
a = ap.parse_args()
OUT = ROOT / "data" / "realworld"
OUT.mkdir(parents=True, exist_ok=True)
lid = LangID.load(ROOT / "data" / "normalizer" / "langid.json")

videos = {}
for f in sorted(corpus_dir().glob("*/*_scenes.csv")):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        if r.get("youtube_url"):
            videos[r["movie_name"]] = r["youtube_url"]
            break


def keep(text):
    t = re.sub(r"https?://\S+|@\S+|#\S+", " ", text)          # drop links, @mentions, hashtags
    if re.search(r"[ఀ-౿]", t):                        # native script -> not romanized
        return None
    words = re.findall(r"[A-Za-z]+", t)
    if not 4 <= len(words) <= 40:
        return None
    te = sum(not lid.is_english(w) for w in words if len(w) >= 2)
    return re.sub(r"\s+", " ", t).strip() if te >= 2 and te / len(words) >= 0.3 else None


opts = {"skip_download": True, "getcomments": True, "quiet": True, "no_warnings": True,
        "extractor_args": {"youtube": {"max_comments": [str(a.per_video), str(a.per_video), "0", "0"]}}}
seen, rows, stats = set(), [], {"videos": 0, "comments_fetched": 0}
with yt_dlp.YoutubeDL(opts) as ydl:
    for movie, url in videos.items():
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as e:  # removed / private videos are skipped
            print(f"skip {movie}: {str(e)[:80]}")
            continue
        comments = info.get("comments") or []
        stats["videos"] += 1
        stats["comments_fetched"] += len(comments)
        for c in comments:
            t = keep(c.get("text", ""))
            if t and t.lower() not in seen:
                seen.add(t.lower())
                rows.append({"movie": movie, "text": t})  # no author / id / time kept
        print(f"{movie}: {len(comments)} comments, kept so far {len(rows)}", flush=True)

with open(OUT / "comments.jsonl", "w", encoding="utf-8") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
stats["romanized_telugu_kept"] = len(rows)
json.dump(stats, open(ROOT / "docs" / "results" / "s6_collection.json", "w"), indent=1)
print(stats)
