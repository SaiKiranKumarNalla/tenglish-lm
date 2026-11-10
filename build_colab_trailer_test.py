"""
Builds the Generic YouTube Movie Subtitle & Scene Corpus Extractor Colab Notebook:
- `notebooks/00_Colab_QuickTest_YouTube_Trailer_Extractor.ipynb`
- `notebooks/01_YouTube_Movie_Corpus_Extractor.ipynb`

Features:
- 100% Generic (NO hardcoded Bommarillu scenes, NO Tenglish romanization clutter).
- Takes a single YouTube Movie URL or a Batch List of YouTube Movie URLs.
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
   - Slices the movie into timestamped conversational scenes and tags `[EMOTION]`, `[GENRE]`, and `[DIALECT]`.
3. **Step 3:** Add **1 or 50 YouTube Movie URLs** to `MOVIE_LINKS` and run! For every movie, it saves:
   - `<movie_name>_raw.srt` (raw YouTube subtitle file)
   - `<movie_name>_clean.srt` (noise-cleaned Telugu `.srt` file)
   - `<movie_name>_scenes.csv` (clean Telugu scenes + timestamps + emotion tags)
   - `<movie_name>_scenes.txt` (human-readable Telugu scene script)
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

    cell_3_runner = """# @title Step 3: Build Your Movie Corpus from YouTube Links
import json
import pandas as pd

# 👇 ADD ONE OR MORE YOUTUBE TELUGU MOVIE LINKS HERE:
# You can pass a plain URL string (title is auto-detected from YouTube!),
# OR a ("Custom_Movie_Name", "https://www.youtube.com/watch?v=...") tuple.
MOVIE_LINKS = [
    "https://www.youtube.com/watch?v=sOOrFtP6DIE&t=1125s",
    # Add more YouTube Telugu movie links below:
    # "https://www.youtube.com/watch?v=...",
]

ALLOW_YOUTUBE_AUTO_CAPTIONS = True  # @param {type:"boolean"}

batch_summary = extract_movie_corpus_batch(
    movie_links=MOVIE_LINKS,
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
    print(f"\\n🎬 Previewing Scenes from: {latest['movie_name']} ({latest['scenes_extracted']} Scenes)")
    print("Emotion Breakdown:", json.dumps(latest["emotion_distribution"], indent=2))
    scenes_df = pd.read_csv(latest["files"]["csv"])
    display(scenes_df[["scene_id", "start_time", "end_time", "emotion", "genre", "dialect", "turn_count", "dialogue_telugu"]].head(20))"""

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
