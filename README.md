# 🎬 Telugu Movie Subtitle & Scene Corpus Extractor

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaiKiranKumarNalla/tenglish-lm/blob/main/notebooks/01_YouTube_Movie_Corpus_Extractor.ipynb)

👉 **[Click Here to Open the Live-Synced Notebook in Google Colab](https://colab.research.google.com/github/SaiKiranKumarNalla/tenglish-lm/blob/main/notebooks/01_YouTube_Movie_Corpus_Extractor.ipynb)**

A clean, generic pipeline to extract **raw and noise-cleaned Telugu `.srt` subtitles**, **scene-tagged `.csv` files**, and **human-readable `.txt` scripts** from a list of YouTube Telugu movie URLs along with your attached movie metadata (`movie_name`, `year`, `director`, `genre`, `dialect`).

---

## 🔄 Always Live-Synced with GitHub

When you open the Colab notebook above, **Step 1 automatically runs `git pull origin main` from this repository (`SaiKiranKumarNalla/tenglish-lm`)** and reloads the Python modules (`tenglish_lm.movie_harvester` and `extract_youtube_movie`). Any update pushed to GitHub is immediately active in your Colab session without needing to copy-paste code!

---

## ✨ What It Does

Given one or more YouTube Telugu movie URLs + your metadata (`movie_name`, `year`, `director`, `genre`, `dialect`), the extractor:
1. **Downloads Captions Instantly (`youtube-transcript-api` + `yt-dlp`):**
   - Supports both **Manual Telugu Subtitles (`te`, `te-IN`)** and **YouTube Auto-Generated Telugu Captions (`AUTO_YOUTUBE_TE`)**.
2. **Cleans Non-Dialogue Noise (`tenglish_lm/movie_harvester.py`):**
   - Strips bracketed sound/music tags in Telugu and English (`[సంగీతం]`, `[బెల్]`, `[చప్పట్లు]`, `[నవ్వులు]`, `[Music]`, `[Applause]`).
   - Removes statutory smoking/drinking disclaimer lines (`పొగ త్రాగుట మధ్యం సేవించుట ఆరోగ్యానికి హానికరం...`).
   - Removes pure BGM humming lines (`తారా తరారారారా...`).
   - Deduplicates rolling YouTube auto-caption lines and clamps overlapping timestamps.
3. **Slices Into Timestamped Scenes & Tags Emotions (`extract_youtube_movie.py`):**
   - Groups subtitle blocks into 2-to-4 speaker turn scenes with `start_time` and `end_time`.
   - Attaches your movie-level metadata (`movie_name`, `year`, `director`, `movie_genre`) and auto-tags each scene with `[SCENE_EMOTION]` (`COMEDY`, `EMOTIONAL_HEARTBREAK`, `ANGER_WARNING`, `SARCASTIC_ROAST`, `PHILOSOPHICAL`, `ELEVATION_MASS`, `CONVERSATIONAL_DRAMA`) and `[SCENE_DIALECT]`.
4. **Stores Clean Movie Files in Google Drive (`/MyDrive/Tenglish_LM/movie_corpus/<movie_name>/`):**
   - `<movie_name>_raw.te.srt` — Original subtitle track downloaded from YouTube
   - `<movie_name>_clean.srt` — Cleaned Telugu `.srt` subtitle file
   - `<movie_name>_scenes.csv` — Spreadsheet-ready scene table (`scene_id`, `movie_name`, `year`, `director`, `movie_genre`, `scene_emotion`, `scene_dialect`, `start_time`, `end_time`, `duration_sec`, `turn_count`, `dialogue_telugu`, `youtube_url`)
   - `<movie_name>_scenes.txt` — Human-readable scene-by-scene Telugu script
   - `corpus_manifest.csv` — Master index of all extracted movies in your corpus
