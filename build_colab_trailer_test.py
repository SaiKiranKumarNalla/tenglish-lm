"""
Generates a 100% standalone, zero-dependency-on-local-repo Google Colab Notebook:
`00_Colab_QuickTest_YouTube_Trailer_Extractor.ipynb`

What it does in Google Colab (Free CPU):
1. Installs `yt-dlp` (and optional `faster-whisper` if the trailer has no manual captions).
2. Mounts Google Drive (`/content/drive/MyDrive/Tenglish_LM/trailer_tests`).
3. Accepts ANY YouTube Movie Trailer URL (with a curated list of Telugu trailers ready to test).
4. Extracts subtitles (or transcribes Telugu audio if subtitles are missing), slices into timestamped scenes,
   auto-tags [EMOTION], [GENRE], [DIALECT], and [EN_CMI %], and generates all 4 Tenglish Personas
   (preserving 'yuddhamlo'/'varshamlo' and protecting 'kopam'/'nijam'/'dhairyam').
5. Saves `<name>_scenes.csv` and `<name>_scenes.txt` directly to Google Drive and displays the interactive table in Colab!
"""

import json
from pathlib import Path


CELL_1_MARKDOWN = """# 🎬 Tenglish-LM: YouTube Movie Trailer & Scene Extractor (Colab Quick-Test)
**Runtime:** Google Colab **Free CPU** (Switch to T4 GPU *only* if testing Whisper audio transcription on uncaptioned trailers)

### How to Use This Notebook (3 Clicks):
1. **Run Cell 1:** Installs `yt-dlp` and connects your Google Drive (`/MyDrive/Tenglish_LM/trailer_tests`).
2. **Run Cell 2:** Loads the **Tenglish-Evo Engine** (Universal Akshara-to-Latin Transducer, Protected Emotional Core `kopam`/`nijam`/`dhairyam`, Stem-final `-m` fix `yuddhamlo`/`varshamlo`, and Auto-Emotion Tagger).
3. **Run Cell 3:** Paste **any YouTube Telugu Movie Trailer (or Scene) URL** into `YOUTUBE_URL` and run! It outputs:
   * `<trailer_name>_scenes.csv` (opens directly in Google Sheets / Excel)
   * `<trailer_name>_scenes.txt` (human-readable tagged script with all 4 Tenglish Personas)
   * An interactive Pandas table right inside Colab!"""

CELL_2_SETUP_CODE = """# @title Step 1: Install yt-dlp & Mount Google Drive
!pip install -q -U yt-dlp pandas

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

# Read tenglish_evo.py, movie_harvester.py, and extract_youtube_movie.py to bundle into Cell 2
def build_standalone_colab_notebook() -> Path:
    root = Path(__file__).parent
    tenglish_evo_code = (root / "tenglish_lm" / "tenglish_evo.py").read_text(encoding="utf-8")
    harvester_code = (root / "tenglish_lm" / "movie_harvester.py").read_text(encoding="utf-8")
    extractor_code = (root / "extract_youtube_movie.py").read_text(encoding="utf-8")

    # Strip internal package imports so it runs in a single self-contained Colab cell
    extractor_clean = []
    for line in extractor_code.splitlines():
        if line.startswith("from tenglish_lm.") or line.startswith("if __name__"):
            if line.startswith("if __name__"):
                break
            continue
        extractor_clean.append(line)

    bundled_engine_cell = (
        "# @title Step 2: Load the Self-Contained Tenglish-Evo & Scene Emotion Engine\n"
        "# Contains: Universal Akshara-to-Latin Transducer + Protected Emotional Core + SRT Pause-Gap Slicer\n\n"
        + tenglish_evo_code
        + "\n\n# --- Movie Harvester & SRT Parser ---\n"
        + harvester_code
        + "\n\n# --- YouTube Extractor & CSV/TXT Exporter ---\n"
        + "\n".join(extractor_clean)
        + "\nprint('✅ Tenglish-Evo Engine & Scene Tagger loaded successfully!')"
    )

    cell_3_runner = """# @title Step 3: Paste Any YouTube Telugu Trailer / Movie Scene URL & Run!
import pandas as pd
import subprocess

# 👇 PASTE ANY YOUTUBE TELUGU TRAILER OR SCENE URL HERE:
YOUTUBE_URL = "https://www.youtube.com/watch?v=e4N9al7vhVQ"  # @param {type:"string"}
TRAILER_NAME = "Telugu_Trailer_Test_01"  # @param {type:"string"}
ERA_TAG = "2020s"  # @param ["1980s", "1990s", "2000s", "2010s", "2020s"]
DIRECTOR_STYLE = "THARUN_TELANGANA_BANTER"  # @param ["TRIVIKRAM_ANTITHESIS", "PURI_CYNICAL_PUNCH", "THARUN_TELANGANA_BANTER", "BHASKAR_ABBURI_RAVI", "SUKUMAR_LOGIC", "VIVEK_URBAN_WIT", "JANDHYALA_WORDPLAY"]
USE_WHISPER_IF_NO_SUBS = True  # @param {type:"boolean"}

work_dir = OUT_DIR / TRAILER_NAME
work_dir.mkdir(parents=True, exist_ok=True)

# 1. Try downloading official/auto captions via yt-dlp
srt_path = download_srt_from_youtube(YOUTUBE_URL, work_dir, TRAILER_NAME)

# 2. If the trailer has NO captions on YouTube, optionally transcribe audio with Faster-Whisper!
if srt_path is None and USE_WHISPER_IF_NO_SUBS:
    print("⚠️ No caption track found on YouTube. Downloading audio & running Whisper ASR...")
    os.system("pip install -q faster-whisper")
    audio_path = work_dir / f"{TRAILER_NAME}.mp3"
    subprocess.run(["yt-dlp", "-x", "--audio-format", "mp3", "-o", str(audio_path), YOUTUBE_URL], check=False)
    if audio_path.exists():
        from faster_whisper import WhisperModel
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(audio_path), language="te")
        srt_lines = []
        for idx, seg in enumerate(segments, start=1):
            st = format_seconds_to_hms(seg.start) + ",000"
            et = format_seconds_to_hms(seg.end) + ",000"
            srt_lines.append(f"{idx}\\n{st} --> {et}\\n{seg.text.strip()}\\n")
        srt_path = work_dir / f"{TRAILER_NAME}.te.srt"
        srt_path.write_text("\\n".join(srt_lines), encoding="utf-8")
        print("✅ Whisper ASR generated Telugu .srt:", srt_path)

# 3. Fallback demo if URL is geo-blocked or offline: run on built-in Bommarillu / Trailer SRT
if srt_path is None or not srt_path.exists():
    print("ℹ️ Using built-in Telugu Trailer + Bommarillu test SRT...")
    demo_srt = work_dir / f"{TRAILER_NAME}_demo.srt"
    demo_scenes = build_curated_multi_era_seed_corpus()
    demo_lines = []
    t_sec = 5
    idx = 1
    for sc in demo_scenes[:6]:
        for turn in sc.turns:
            st = format_seconds_to_hms(t_sec) + ",000"
            et = format_seconds_to_hms(t_sec + 3) + ",000"
            demo_lines.append(f"{idx}\\n{st} --> {et}\\n{turn}\\n")
            t_sec += 4
            idx += 1
        t_sec += 6
    demo_srt.write_text("\\n".join(demo_lines), encoding="utf-8")
    srt_path = demo_srt

# 4. Run the Scene Slicer, Emotion Auto-Tagger & 4-Persona Tenglish Generator
result = extract_full_movie_pipeline(
    movie_name=TRAILER_NAME,
    era=ERA_TAG,
    director_style=DIRECTOR_STYLE,
    srt_file=srt_path,
    out_dir=OUT_DIR,
)

print("\\n==================================================================")
print("✅ EXTRACTION COMPLETE!")
print("==================================================================")
print(json.dumps(result, indent=2, ensure_ascii=False))

# 5. Preview the generated .CSV right inside Colab
df = pd.read_csv(result["files"]["csv"])
display(df[["scene_id", "start_time", "end_time", "emotion", "dialect", "en_cmi_pct", "dialogue_telugu", "dialogue_tenglish_standard", "dialogue_tenglish_broken_acrophonic"]])"""

    cell_4_txt_preview = """# @title Step 4: View the Generated Scene-by-Scene `.txt` Script
txt_path = Path(result["files"]["txt"])
print(txt_path.read_text(encoding="utf-8")[:4000])"""

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

    out_nb = root / "notebooks" / "00_Colab_QuickTest_YouTube_Trailer_Extractor.ipynb"
    out_nb.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_nb


if __name__ == "__main__":
    p = build_standalone_colab_notebook()
    print("Built standalone Colab Quick-Test notebook:", p)
