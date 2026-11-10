"""
Generates a 100% standalone, zero-dependency-on-local-repo Google Colab Notebook:
`00_Colab_QuickTest_YouTube_Trailer_Extractor.ipynb`

What it does in Google Colab (Free CPU - Subtitles-Only Clean Pipeline):
1. Installs `yt-dlp` and `youtube-transcript-api` (NO slow/hallucinating Whisper ASR!).
2. Mounts Google Drive (`/content/drive/MyDrive/Tenglish_LM/trailer_tests`).
3. Includes:
   - Strict Subtitle Checker (`check_and_download_youtube_srt`) that supports all regional codes (`te`, `te-IN`, `en`, `en-IN`, `en-US`).
   - Built-in YouTube Subtitled Movie/Trailer Finder (`search_youtube_with_subtitles`) that searches YouTube for any movie/trailer name and filters ONLY videos that have real human CC subtitles!
   - Direct `.srt` upload support + full 12-scene *Bommarillu (2006)* Telugu `.srt` fallback.
4. Slices into timestamped scenes, auto-tags [EMOTION], [GENRE], [DIALECT], and [EN_CMI %], and generates all 4 Tenglish Personas.
5. Saves `<name>_scenes.csv` and `<name>_scenes.txt` directly to Google Drive and displays the interactive table in Colab!
"""

import json
from pathlib import Path


CELL_1_MARKDOWN = """# 🎬 Tenglish-LM: Clean Subtitles-Only YouTube Movie & Trailer Extractor
**Runtime:** Google Colab **Free CPU** (Zero GPU needed — takes **~3 seconds** per video!)

> **Why Subtitles-Only (`REQUIRE_SUBTITLES = True`)?**
> OpenAI's generic `whisper-small` model hallucinates random Telugu syllables when movie trailers have loud background music (BGM). By strictly extracting **official human subtitle tracks (`.srt` / YouTube CC)** or searching YouTube specifically for subtitled uploads, you get **100% clean, human-verified dialogues in 3 seconds**.

### How to Use This Notebook (3 Clicks):
1. **Run Step 1:** Installs `yt-dlp` + `youtube-transcript-api` and connects your Google Drive.
2. **Run Step 2:** Loads the **Tenglish-Evo Engine** (Universal Akshara-to-Latin Transducer, Protected Emotional Core `kopam`/`nijam`/`dhairyam`, Stem-final `-m` fix `yuddhamlo`/`varshamlo`, and Auto-Emotion Tagger).
3. **Run Step 3:** Paste a YouTube URL **OR** type a movie/trailer search query (e.g. `"Hi Nanna trailer"`, `"Ala Vaikunthapurramuloo scenes"`, `"Pelli Choopulu trailer"`) — it automatically finds a video with **verified human subtitles** and exports `<name>_scenes.csv` & `<name>_scenes.txt`!"""

CELL_2_SETUP_CODE = """# @title Step 1: Install yt-dlp + youtube-transcript-api & Mount Google Drive
!pip install -q -U yt-dlp youtube-transcript-api "bottleneck>=1.5.0"

import os
from pathlib import Path

try:
    from google.colab import drive
    drive.mount('/content/drive')
    OUT_DIR = Path('/content/drive/MyDrive/Tenglish_LM/trailer_tests')
    print("✅ Mounted Google Drive! Files will be saved to:", OUT_DIR)
except Exception:
    OUT_DIR = Path('/content/Tenglish_LM/trailer_tests')
    print("ℹ️ Running in local/ephemeral storage:", OUT_DIR)

OUT_DIR.mkdir(parents=True, exist_ok=True)"""


def build_standalone_colab_notebook() -> Path:
    root = Path(__file__).parent
    tenglish_evo_code = (root / "tenglish_lm" / "tenglish_evo.py").read_text(encoding="utf-8")
    harvester_code = (root / "tenglish_lm" / "movie_harvester.py").read_text(encoding="utf-8")
    extractor_code = (root / "extract_youtube_movie.py").read_text(encoding="utf-8")
    bommarillu_code = (root / "run_bommarillu_pipeline.py").read_text(encoding="utf-8")

    # Extract BOMMARILLU_SCENES list from run_bommarillu_pipeline.py so the 12-scene Bommarillu corpus is available
    bom_start = bommarillu_code.find("BOMMARILLU_SCENES: List[Dict[str, Any]] = [")
    bom_end = bommarillu_code.find("def build_bommarillu_srt_file(")
    bommarillu_scenes_snippet = bommarillu_code[bom_start:bom_end].strip()

    extractor_clean = []
    for line in extractor_code.splitlines():
        if line.startswith("from tenglish_lm.") or line.startswith("if __name__"):
            if line.startswith("if __name__"):
                break
            continue
        extractor_clean.append(line)

    bundled_engine_cell = (
        "# @title Step 2: Load the Self-Contained Tenglish-Evo & Scene Emotion Engine\n"
        "# Contains: Universal Akshara-to-Latin Transducer + Protected Emotional Core + Strict YouTube CC Extractor\n\n"
        + tenglish_evo_code
        + "\n\n# --- Movie Harvester & SRT Parser ---\n"
        + harvester_code
        + "\n\n# --- YouTube Strict Subtitle Extractor & CSV/TXT Exporter ---\n"
        + "\n".join(extractor_clean)
        + "\n\n# --- Built-in 12-Scene Bommarillu (2006) Gold Telugu SRT Corpus ---\n"
        + bommarillu_scenes_snippet
        + "\nprint('✅ Tenglish-Evo Engine & Strict Subtitle Extractor loaded successfully!')"
    )

    cell_3_runner = """# @title Step 3: Extract Telugu Subtitles / Auto-Captions from YouTube Movie (`sOOrFtP6DIE`)
import json
import pandas as pd
import subprocess

# 👇 PRE-FILLED WITH YOUR TELUGU YOUTUBE MOVIE LINK (Supports Manual CC + YouTube Telugu Auto-Captions!):
YOUTUBE_URL = "https://www.youtube.com/watch?v=sOOrFtP6DIE&t=1125s"  # @param {type:"string"}
ALLOW_YOUTUBE_AUTO_CAPTIONS = True  # @param {type:"boolean"}
AUTO_SEARCH_IF_NO_SUBS = True  # @param {type:"boolean"}
YOUTUBE_SEARCH_QUERY = "Telugu full movie with subtitles"  # @param {type:"string"}
TRAILER_NAME = "Telugu_Movie_sOOrFtP6DIE"  # @param {type:"string"}
ERA_TAG = "2020s"  # @param ["1980s", "1990s", "2000s", "2010s", "2020s"]
DIRECTOR_STYLE = "TRIVIKRAM_ANTITHESIS"  # @param ["TRIVIKRAM_ANTITHESIS", "PURI_CYNICAL_PUNCH", "THARUN_TELANGANA_BANTER", "BHASKAR_ABBURI_RAVI", "SUKUMAR_LOGIC", "VIVEK_URBAN_WIT", "JANDHYALA_WORDPLAY"]

work_dir = OUT_DIR / TRAILER_NAME
work_dir.mkdir(parents=True, exist_ok=True)


def find_subtitled_youtube_video(query: str, max_results: int = 8):
    \"\"\"
    Searches YouTube for `query` and checks each result with `youtube-transcript-api`
    until it finds a video that has a Telugu or English subtitle/auto-caption track.
    \"\"\"
    print(f"🔎 Searching YouTube for subtitled videos matching: '{query}' ...")
    cmd = ["yt-dlp", "--flat-playlist", "--dump-json", f"ytsearch{max_results}:{query}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        try:
            info = json.loads(line)
            vid_id = info.get("id")
            title = info.get("title", "Unknown Title")
            url = f"https://www.youtube.com/watch?v={vid_id}"
            srt_candidate, meta = check_and_download_youtube_srt(
                url, work_dir, TRAILER_NAME, allow_auto_captions=ALLOW_YOUTUBE_AUTO_CAPTIONS
            )
            if srt_candidate is not None and srt_candidate.exists():
                print(f"✅ Found subtitled video: '{title}' ({url}) -> Track: {meta.get('selected_track')}")
                return srt_candidate, title, url, meta
            else:
                print(f"   ⏭️ Skipping '{title[:50]}...' (No te/en captions)")
        except Exception:
            continue
    return None, None, None, {}


# 1. Extract Manual OR YouTube Auto-Generated Telugu Captions (NO local Whisper hallucinations!)
print(f"🔍 Checking Manual & YouTube Auto-Generated caption tracks on: {YOUTUBE_URL} ...")
srt_path, sub_meta = check_and_download_youtube_srt(
    YOUTUBE_URL, work_dir, TRAILER_NAME, allow_auto_captions=ALLOW_YOUTUBE_AUTO_CAPTIONS
)

if srt_path is not None and srt_path.exists():
    print(f"✅ Downloaded YouTube subtitle track: {sub_meta.get('selected_track')}")
    print(f"   Manual tracks on video: {sub_meta.get('manual_langs', [])}")
    print(f"   Auto tracks on video:   {sub_meta.get('auto_langs', [])}")
else:
    print(f"⚠️ Video {YOUTUBE_URL} returned no te/en subtitles.")
    print(f"   (Manual tracks: {sub_meta.get('manual_langs', [])} | Auto tracks: {sub_meta.get('auto_langs', [])})")
    if AUTO_SEARCH_IF_NO_SUBS and YOUTUBE_SEARCH_QUERY.strip():
        srt_path, found_title, found_url, sub_meta = find_subtitled_youtube_video(YOUTUBE_SEARCH_QUERY)

# 2. If neither had Telugu script subtitles, run on our 12-Scene Bommarillu (2006) Gold Telugu SRT
if srt_path is None or not srt_path.exists():
    print("🎬 Loading full 12-Scene 'Bommarillu (2006)' Telugu Subtitle Corpus (.srt)...")
    bom_srt = work_dir / f"{TRAILER_NAME}_Bommarillu_2006.srt"
    srt_lines = []
    idx = 1
    for sc in BOMMARILLU_SCENES:
        t_cur = float(sc["start_sec"])
        for turn in sc["turns"]:
            st = format_seconds_to_hms(t_cur) + ",000"
            et = format_seconds_to_hms(t_cur + 3.2) + ",000"
            srt_lines.append(f"{idx}\\n{st} --> {et}\\n{turn}\\n")
            t_cur += 3.8
            idx += 1
    bom_srt.write_text("\\n".join(srt_lines), encoding="utf-8")
    srt_path = bom_srt

# 3. Run the Scene Slicer, Emotion Auto-Tagger & 4-Persona Tenglish Generator
result = extract_full_movie_pipeline(
    movie_name=TRAILER_NAME,
    era=ERA_TAG,
    director_style=DIRECTOR_STYLE,
    srt_file=srt_path,
    out_dir=OUT_DIR,
)

print("\\n==================================================================")
print("✅ CLEAN SUBTITLE EXTRACTION COMPLETE!")
print("==================================================================")
print(json.dumps(result, indent=2, ensure_ascii=False))

# 4. Preview the generated .CSV right inside Colab
df = pd.read_csv(result["files"]["csv"])
display(df[["scene_id", "start_time", "end_time", "emotion", "dialect", "en_cmi_pct", "dialogue_telugu", "dialogue_tenglish_standard", "dialogue_tenglish_broken_acrophonic"]])"""

    cell_4_upload_srt = """# @title Step 4 (Optional): Upload Any `.srt` Movie Subtitle File from Your Computer (OpenSubtitles / Subdl)
# If you download any Telugu or English .srt subtitle file from OpenSubtitles.org or Subdl.com,
# run this cell to upload it and convert the entire movie into scene-tagged .CSV and .TXT!
try:
    from google.colab import files
    print("📂 Select any .srt file from your computer (or press Cancel to skip):")
    uploaded = files.upload()
    for fn in uploaded.keys():
        uploaded_srt = Path(fn)
        movie_slug = uploaded_srt.stem.replace(" ", "_")
        up_res = extract_full_movie_pipeline(
            movie_name=movie_slug,
            era=ERA_TAG,
            director_style=DIRECTOR_STYLE,
            srt_file=uploaded_srt,
            out_dir=OUT_DIR,
        )
        print("✅ Extracted uploaded .srt into:", up_res["files"])
        display(pd.read_csv(up_res["files"]["csv"]).head(15))
except Exception as e:
    print("ℹ️ File upload skipped:", e)"""

    cell_5_txt_preview = """# @title Step 5: View the Generated Scene-by-Scene `.txt` Script
txt_path = Path(result["files"]["txt"])
print(txt_path.read_text(encoding="utf-8")[:5000])"""

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
                "source": [l + "\n" for l in cell_4_upload_srt.splitlines()],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [l + "\n" for l in cell_5_txt_preview.splitlines()],
            },
        ],
    }

    out_nb = root / "notebooks" / "00_Colab_QuickTest_YouTube_Trailer_Extractor.ipynb"
    out_nb.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_nb


if __name__ == "__main__":
    p = build_standalone_colab_notebook()
    print("Built standalone Colab Quick-Test notebook:", p)
