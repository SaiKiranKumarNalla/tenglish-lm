"""Load the movie scene corpus produced by extractor/ (movie_corpus/<movie>/<movie>_scenes.csv).

The corpus itself is NOT in git (copyrighted subtitles). Point TLM_CORPUS at your local copy:
    export TLM_CORPUS=~/Downloads/Tenglish_LM/movie_corpus
"""
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TELUGU = re.compile(r"[ఀ-౿]")


@dataclass
class Scene:
    scene_id: str
    movie: str
    year: int | None
    lines: list[str]
    kw_tone: str      # v1 keyword tag (known to be unreliable, kept only for analysis)
    kw_dialect: str   # v1 keyword tag (known to be unreliable, kept only for analysis)


def corpus_dir(path: str | None = None) -> Path:
    p = path or os.environ.get("TLM_CORPUS")
    if not p:
        raise SystemExit("Set TLM_CORPUS to your local movie_corpus folder (see tlm/corpus.py).")
    return Path(p).expanduser()


def load_scenes(path: str | None = None) -> list[Scene]:
    scenes = []
    for f in sorted(corpus_dir(path).glob("*/*_scenes.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            lines = [re.sub(r"\s+", " ", t).strip() for t in r["dialogue_telugu"].split("//")]
            year = int(r["year"]) if r.get("year", "").isdigit() else None
            scenes.append(Scene(r["scene_id"], r["movie_name"], year, [l for l in lines if l],
                                r.get("scene_emotion", ""), r.get("scene_dialect", "")))
    return scenes


def split_of() -> dict[str, str]:
    """movie -> train/val/test, fixed by data/split_manifest.json (split by movie, seed 42)."""
    m = json.load(open(ROOT / "data" / "split_manifest.json", encoding="utf-8"))
    return {mv: s for s, p in m["partitions"].items() for mv in p["movies"]}


def telugu_lines(scenes: list[Scene], min_telugu_chars: int = 4) -> list[str]:
    """Every dialogue line that is mostly Telugu script."""
    return [l for s in scenes for l in s.lines if len(TELUGU.findall(l)) >= min_telugu_chars]
