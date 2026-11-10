"""
Builds the Generic YouTube Movie Subtitle & Scene Corpus Extractor Colab Notebook:
- `notebooks/00_Colab_QuickTest_YouTube_Trailer_Extractor.ipynb`
- `notebooks/01_YouTube_Movie_Corpus_Extractor.ipynb`

Features:
- Allows attaching `movie_name`, `year`, `director`, `genre`, and `dialect` alongside every YouTube movie link.
- 100% Generic (NO hardcoded Bommarillu scenes, NO Tenglish romanization clutter).
- Extracts Manual or YouTube Auto-Generated Telugu captions (`te`, `te-IN`).
- Strips `[సంగీతం]`, `[బెల్]`, `[చప్పట్లు]`, statutory smoking disclaimers, and BGM humming (`తారా తరారారారా`).
- Saves `<movie>_raw.srt`, `<movie>_clean.srt`, `<movie>_scenes.csv`, `<movie>_scenes.txt`, and `corpus_manifest.csv` directly to Google Drive!
"""

import json
from pathlib import Path


CELL_1_MARKDOWN = """# 🎬 Telugu Movie Subtitle & Scene Corpus Extractor (YouTube -> `.srt`, `.csv`, `.txt`)
**Runtime:** Google Colab **Free CPU** (Fast & Clean — Extracts & stores full movies in seconds)

### What This Notebook Does:
1. **Step 1:** Installs `yt-dlp` and `youtube-transcript-api` and connects your Google Drive (`/MyDrive/Tenglish_LM/movie_corpus`).
2. **Step 2:** Loads the **Generic Telugu Movie Harvester & Scene Tagger**:
   - Downloads official Telugu captions (`te`) or YouTube's auto-generated Telugu captions (`AUTO_YOUTUBE_TE`).
   - Automatically strips `[సంగీతం]`, `[బెల్]`, `[చప్పట్లు]`, statutory smoking/alcohol disclaimers (`పొగ త్రాగుట...`), and BGM humming (`తారా తరారారారా...`).
   - Slices the movie into timestamped conversational scenes and attaches your **Movie Name, Year, Director, and Genre**, plus auto-tagged **Scene Emotion** (`COMEDY`, `EMOTIONAL_HEARTBREAK`, `ANGER_WARNING`, `SARCASTIC_ROAST`, `PHILOSOPHICAL`, `ELEVATION_MASS`) and **Dialect**.
3. **Step 3:** Attach your YouTube link(s) along with **`movie_name`, `year`, `director`, and `genre`** and run! For every movie, it saves:
   - `<movie_name>_raw.te.srt` (raw YouTube subtitle file)
   - `<movie_name>_clean.srt` (noise-cleaned Telugu `.srt` file)
   - `<movie_name>_scenes.csv` (with `movie_name`, `year`, `director`, `movie_genre`, `scene_emotion`, `scene_dialect`, `start_time`, `end_time`, `dialogue_telugu`)
   - `<movie_name>_scenes.txt` (human-readable Telugu scene script with movie & scene headers)
   - `corpus_manifest.csv` (master index of all extracted movies in your corpus)"""

CELL_2_SETUP_CODE = """# @title Step 1: Install yt-dlp + youtube-transcript-api & Mount Google Drive
!pip install -q -U yt-dlp youtube-transcript-api "bottleneck>=1.5.0"

import os
from pathlib import Path

try:
    from google.colab import drive
    drive.mount('/content/drive')
    CORPUS_DIR = Path('/content/drive/MyDrive/Tenglish_LM/movie_corpus')
    print("✅ Mounted Google Drive! Movie corpus will be saved to:", CORPUS_DIR)
except Exception:
    CORPUS_DIR = Path('/content/Tenglish_LM/movie_corpus')
    print("ℹ️ Running in local storage:", CORPUS_DIR)

CORPUS_DIR.mkdir(parents=True, exist_ok=True)"""


def build_generic_colab_notebooks() -> None:
    root = Path(__file__).parent
    harvester_code = (root / "tenglish_lm" / "movie_harvester.py").read_text(encoding="utf-8")
    extractor_code = (root / "extract_youtube_movie.py").read_text(encoding="utf-8")

    extractor_clean = []
    skip_import = False
    for line in extractor_code.splitlines():
        if line.startswith("from tenglish_lm.movie_harvester import"):
            skip_import = True
            continue
        if skip_import:
            if ")" in line:
                skip_import = False
            continue
        if line.startswith("if __name__"):
            break
        extractor_clean.append(line)

    bundled_engine_cell = (
        "# @title Step 2: Load Generic Telugu Movie SRT Cleaner & Scene Extractor\n"
        + harvester_code
        + "\n\n# --- Generic YouTube Movie Extractor & Corpus Exporter ---\n"
        + "\n".join(extractor_clean)
        + "\nprint('✅ Generic Telugu Movie Extractor loaded (Zero hardcoded data, Pure Telugu SRT/CSV/TXT output)!')"
    )

    cell_3_runner = """# @title Step 3: Attach YouTube Link + Movie Name, Year, Director & Genre
import json
import pandas as pd

# 👇 QUICK SINGLE-MOVIE INPUTS (Or add multiple movies to `EXTRA_MOVIES_BATCH` below!):
YOUTUBE_URL = "https://www.youtube.com/watch?v=sOOrFtP6DIE&t=1125s"  # @param {type:"string"}
MOVIE_NAME = "Subramanyam_For_Sale"  # @param {type:"string"}
YEAR = "2015"  # @param {type:"string"}
DIRECTOR = "Harish Shankar"  # @param {type:"string"}
GENRE = "RomCom / Family Entertainer"  # @param {type:"string"}
DIALECT = "STANDARD_TELUGU"  # @param ["STANDARD_TELUGU", "TELANGANA", "RAYALASEEMA", "COASTAL_ANDHRA", "URBAN_HYDERABAD"]
ALLOW_YOUTUBE_AUTO_CAPTIONS = True  # @param {type:"boolean"}

# 👇 OPTIONAL: ADD MULTIPLE MOVIES WITH THEIR METADATA TO EXTRACT A FULL BATCH AT ONCE:
MOVIE_CORPUS_LIST = [
    {
        "url": YOUTUBE_URL,
        "movie_name": MOVIE_NAME,
        "year": YEAR,
        "director": DIRECTOR,
        "genre": GENRE,
        "dialect": DIALECT,
    },
    # Example of adding more movies to your batch list:
    # {
    #     "url": "https://www.youtube.com/watch?v=...",
    #     "movie_name": "Athadu",
    #     "year": "2005",
    #     "director": "Trivikram Srinivas",
    #     "genre": "Action / Family Drama",
    #     "dialect": "STANDARD_TELUGU",
    # },
]

batch_summary = extract_movie_corpus_batch(
    movie_links=MOVIE_CORPUS_LIST,
    out_dir=CORPUS_DIR,
    allow_auto_captions=ALLOW_YOUTUBE_AUTO_CAPTIONS,
)

print("\\n==================================================================")
print(f"✅ CORPUS EXTRACTION COMPLETE: {batch_summary['movies_extracted']}/{batch_summary['total_links']} Movies Saved")
print(f"📂 Master Manifest: {batch_summary['manifest_csv']}")
print("==================================================================")

# Display the Master Corpus Manifest
manifest_df = pd.read_csv(batch_summary["manifest_csv"])
display(manifest_df)

# Preview the scenes of the most recently extracted movie
if batch_summary["results"]:
    latest = batch_summary["results"][-1]
    print(
        f"\\n🎬 Previewing Scenes from: {latest['movie_name']} ({latest['year']}) "
        f"| Director: {latest['director']} | Genre: {latest['movie_genre']} "
        f"| Total Scenes: {latest['scenes_extracted']}"
    )
    print("Scene Emotion Breakdown:", json.dumps(latest["emotion_distribution"], indent=2))
    scenes_df = pd.read_csv(latest["files"]["csv"])
    display(
        scenes_df[
            [
                "scene_id",
                "movie_name",
                "year",
                "director",
                "movie_genre",
                "scene_emotion",
                "scene_dialect",
                "start_time",
                "end_time",
                "turn_count",
                "dialogue_telugu",
            ]
        ].head(20)
    )"""

    cell_4_txt_preview = """# @title Step 4: Inspect the Clean Telugu `.txt` Scene Script & Cleaned `.srt`
if batch_summary["results"]:
    latest = batch_summary["results"][-1]
    txt_path = Path(latest["files"]["txt"])
    clean_srt_path = Path(latest["files"]["clean_srt"])
    print(f"📄 Scene Script ({txt_path}):\\n")
    print(txt_path.read_text(encoding="utf-8")[:4000])
    print("\\n------------------------------------------------------------------")
    print(f"📄 Cleaned .SRT Preview ({clean_srt_path}):\\n")
    print(clean_srt_path.read_text(encoding="utf-8")[:1500])"""

    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "include_colab_link": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [l + "\n" for l in CELL_1_MARKDOWN.splitlines()],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [l + "\n" for l in CELL_2_SETUP_CODE.splitlines()],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [l + "\n" for l in bundled_engine_cell.splitlines()],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [l + "\n" for l in cell_3_runner.splitlines()],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [l + "\n" for l in cell_4_txt_preview.splitlines()],
            },
        ],
    }

    nb_dir = root / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)
    for fname in [
        "00_Colab_QuickTest_YouTube_Trailer_Extractor.ipynb",
        "01_YouTube_Movie_Corpus_Extractor.ipynb",
    ]:
        out_nb = nb_dir / fname
        out_nb.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Built generic Colab notebook:", out_nb)


if __name__ == "__main__":
    build_generic_colab_notebooks()
