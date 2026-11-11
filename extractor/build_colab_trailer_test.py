"""
Builds the GitHub Live-Synced YouTube Movie Subtitle & Scene Corpus Extractor Colab Notebooks:
- `extractor/01_YouTube_Movie_Corpus_Extractor.ipynb`
- `extractor/00_Colab_QuickTest_YouTube_Trailer_Extractor.ipynb`

Key Upgrades:
1. Embeds the official `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]` button at the top of the notebook so GitHub always renders a 1-click "Open in Colab" badge.
2. Live-syncs directly with `https://github.com/SaiKiranKumarNalla/tenglish-lm.git` via `git clone` / `git pull` + `importlib.reload()` in Step 1, so the notebook is ALWAYS 100% synced with the latest GitHub code!
"""

import json
from pathlib import Path


COLAB_URL = "https://colab.research.google.com/github/SaiKiranKumarNalla/tenglish-lm/blob/main/extractor/01_YouTube_Movie_Corpus_Extractor.ipynb"

CELL_1_MARKDOWN = f"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

# 🎬 Telugu Movie Subtitle & Scene Corpus Extractor (Live-Synced with GitHub)
**Repository:** [`SaiKiranKumarNalla/tenglish-lm`](https://github.com/SaiKiranKumarNalla/tenglish-lm) | **Runtime:** Google Colab **Free CPU**

### How This Notebook Stays 100% Synced with GitHub:
* **Step 1** automatically runs `git clone` / `git pull origin main` from `https://github.com/SaiKiranKumarNalla/tenglish-lm.git` and reloads `extractor.movie_harvester` & `extract_youtube_movie`. Every time you run Step 1, you get the exact latest code from your GitHub repo!
* **Step 2** lets you attach your YouTube movie link(s) along with **`movie_name`, `year`, `director`, `genre`, and `dialect`**, and saves the clean `.srt`, `.csv`, and `.txt` files directly to your Google Drive (`/MyDrive/Tenglish_LM/movie_corpus/`)."""

CELL_2_LIVE_SYNC_SETUP = """# @title Step 1: Live-Sync Latest Code from GitHub & Mount Google Drive
!pip install -q -U yt-dlp youtube-transcript-api "bottleneck>=1.5.0"

import importlib
import os
import subprocess
import sys
from pathlib import Path

# 1. Clone or Pull latest code directly from your GitHub repository
REPO_URL = "https://github.com/SaiKiranKumarNalla/tenglish-lm.git"
REPO_DIR = Path("/content/tenglish-lm")

if not REPO_DIR.exists():
    print("🔄 Cloning latest code from GitHub (SaiKiranKumarNalla/tenglish-lm)...")
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
else:
    print("🔄 Pulling latest updates from GitHub (SaiKiranKumarNalla/tenglish-lm)...")
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", "main"], check=True)

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# Reload modules so any GitHub changes take effect immediately in this Colab session
import extractor.movie_harvester
import extractor.ncm_concept_hierarchy
import extractor.extract_youtube_movie as extract_youtube_movie

importlib.reload(extractor.movie_harvester)
importlib.reload(extractor.ncm_concept_hierarchy)
importlib.reload(extract_youtube_movie)

from extractor.extract_youtube_movie import (
    extract_full_movie_pipeline,
    extract_movie_corpus_batch,
    load_movie_list_from_csv,
)

# Show the exact Git commit currently active in Colab
commit_info = subprocess.run(
    ["git", "-C", str(REPO_DIR), "log", "-1", "--oneline"],
    capture_output=True,
    text=True,
    check=False,
).stdout.strip()
print(f"✅ Synced with GitHub commit: {commit_info}")

# 2. Mount Google Drive for persistent corpus storage
try:
    from google.colab import drive
    drive.mount("/content/drive")
    CORPUS_DIR = Path("/content/drive/MyDrive/Tenglish_LM/movie_corpus")
    print("✅ Mounted Google Drive! Movie corpus will be saved to:", CORPUS_DIR)
except Exception:
    CORPUS_DIR = Path("/content/Tenglish_LM/movie_corpus")
    print("ℹ️ Running in local storage:", CORPUS_DIR)

CORPUS_DIR.mkdir(parents=True, exist_ok=True)"""

CELL_3_RUNNER = """# @title Step 2: Attach YouTube Link + Movie Name, Year, Director & Genre -> Extract Telugu Corpus + NCM Hierarchy
import json
import pandas as pd

# 👇 SINGLE-MOVIE FORM INPUTS (Set MOVIE_NAME to "AUTO" to auto-detect title from YouTube, or type your own!):
YOUTUBE_URL = "https://www.youtube.com/watch?v=fBSjiV745_k"  # @param {type:"string"}
MOVIE_NAME = "AUTO"  # @param {type:"string"}
YEAR = "2020s"  # @param {type:"string"}
DIRECTOR = "UNKNOWN"  # @param {type:"string"}
GENRE = "Drama / Entertainer"  # @param {type:"string"}
DIALECT = "STANDARD_TELUGU"  # @param ["STANDARD_TELUGU", "TELANGANA", "RAYALASEEMA", "COASTAL_ANDHRA", "URBAN_HYDERABAD"]
ALLOW_YOUTUBE_AUTO_CAPTIONS = True  # @param {type:"boolean"}

# 👇 ADD ONE OR MORE MOVIES WITH METADATA TO EXTRACT YOUR CORPUS (Strictly Telugu `te` Captions Only!):
MOVIE_CORPUS_LIST = [
    {
        "url": YOUTUBE_URL,
        "movie_name": MOVIE_NAME,
        "year": YEAR,
        "director": DIRECTOR,
        "genre": GENRE,
        "dialect": DIALECT,
    },
    # Example of adding more movies here:
    # {
    #     "url": "https://www.youtube.com/watch?v=sOOrFtP6DIE&t=1125s",
    #     "movie_name": "Subramanyam_For_Sale",
    #     "year": "2015",
    #     "director": "Harish Shankar",
    #     "genre": "RomCom / Family Entertainer",
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

manifest_df = pd.read_csv(batch_summary["manifest_csv"])
display(manifest_df)

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
                "ncm_rvq_code",
                "ncm_r1_coarse_trajectory",
                "ncm_r2_semantic_anchors",
                "start_time",
                "end_time",
                "dialogue_telugu",
            ]
        ].head(20)
    )"""

CELL_4_TXT_PREVIEW = """# @title Step 3: Inspect the Clean Telugu `.txt` Scene Script & Cleaned `.srt`
if batch_summary["results"]:
    latest = batch_summary["results"][-1]
    txt_path = Path(latest["files"]["txt"])
    clean_srt_path = Path(latest["files"]["clean_srt"])
    print(f"📄 Scene Script ({txt_path}):\\n")
    print(txt_path.read_text(encoding="utf-8")[:4000])
    print("\\n------------------------------------------------------------------")
    print(f"📄 Cleaned .SRT Preview ({clean_srt_path}):\\n")
    print(clean_srt_path.read_text(encoding="utf-8")[:1500])"""


def build_generic_colab_notebooks() -> None:
    root = Path(__file__).parent
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "include_colab_link": True,
                "name": "01_YouTube_Movie_Corpus_Extractor.ipynb",
            },
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"id": "view-in-github", "colab_type": "text"},
                "source": [l + "\n" for l in CELL_1_MARKDOWN.splitlines()],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "step1_github_sync"},
                "outputs": [],
                "source": [l + "\n" for l in CELL_2_LIVE_SYNC_SETUP.splitlines()],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "step2_extract_corpus"},
                "outputs": [],
                "source": [l + "\n" for l in CELL_3_RUNNER.splitlines()],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "step3_preview_txt_srt"},
                "outputs": [],
                "source": [l + "\n" for l in CELL_4_TXT_PREVIEW.splitlines()],
            },
        ],
    }

    nb_dir = root / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)
    for fname in [
        "01_YouTube_Movie_Corpus_Extractor.ipynb",
        "00_Colab_QuickTest_YouTube_Trailer_Extractor.ipynb",
    ]:
        out_nb = nb_dir / fname
        out_nb.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Built GitHub-synced Colab notebook:", out_nb)


if __name__ == "__main__":
    build_generic_colab_notebooks()
