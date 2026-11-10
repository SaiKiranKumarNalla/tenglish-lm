# 🎬 Telugu Movie Subtitle & Scene Corpus Extractor

A clean, generic pipeline to extract **raw and noise-cleaned Telugu `.srt` subtitles**, **scene-tagged `.csv` files**, and **human-readable `.txt` scripts** from a list of YouTube Telugu movie URLs.

---

## ✨ What It Does

Given one or more YouTube Telugu movie URLs, the extractor:
1. **Downloads Captions Instantly (`youtube-transcript-api` + `yt-dlp`):**
   - Supports both **Manual Telugu Subtitles (`te`, `te-IN`)** and **YouTube Auto-Generated Telugu Captions (`AUTO_YOUTUBE_TE`)**.
   - Automatically detects the movie title from YouTube if no custom name is provided.
2. **Cleans Non-Dialogue Noise (`tenglish_lm/movie_harvester.py`):**
   - Strips bracketed sound/music tags in Telugu and English (`[సంగీతం]`, `[బెల్]`, `[చప్పట్లు]`, `[నవ్వులు]`, `[Music]`, `[Applause]`).
   - Removes statutory smoking/drinking disclaimer lines (`పొగ త్రాగుట మధ్యం సేవించుట ఆరోగ్యానికి హానికరం...`).
   - Removes pure BGM humming lines (`తారా తరారారారా...`).
   - Deduplicates rolling YouTube auto-caption lines and clamps overlapping timestamps.
3. **Slices Into Timestamped Scenes & Tags Emotions (`extract_youtube_movie.py`):**
   - Groups subtitle blocks into 2-to-4 speaker turn scenes with `start_time` and `end_time`.
   - Tags each scene with `[EMOTION]` (`COMEDY`, `EMOTIONAL_HEARTBREAK`, `ANGER_WARNING`, `SARCASTIC_ROAST`, `PHILOSOPHICAL`, `ELEVATION_MASS`, `CONVERSATIONAL_DRAMA`), `[GENRE]`, and `[DIALECT]` (`TELANGANA`, `RAYALASEEMA`, `COASTAL_ANDHRA`, `STANDARD_TELUGU`).
4. **Stores Clean Movie Files in Google Drive / Local Storage:**
   - `<movie_name>_raw.srt` — Original subtitle track downloaded from YouTube
   - `<movie_name>_clean.srt` — Cleaned Telugu `.srt` subtitle file
   - `<movie_name>_scenes.csv` — Spreadsheet-ready scene table (`scene_id`, `start_time`, `end_time`, `emotion`, `genre`, `dialect`, `turn_count`, `dialogue_telugu`)
   - `<movie_name>_scenes.txt` — Human-readable scene-by-scene Telugu script
   - `corpus_manifest.csv` — Master index of all extracted movies in your corpus

---

## 🚀 Run in Google Colab (Free CPU)

Open [`notebooks/00_Colab_QuickTest_YouTube_Trailer_Extractor.ipynb`](notebooks/00_Colab_QuickTest_YouTube_Trailer_Extractor.ipynb) in Google Colab:
1. Run **Step 1** to install `yt-dlp` & `youtube-transcript-api` and mount Google Drive (`/MyDrive/Tenglish_LM/movie_corpus`).
2. Run **Step 2** to load the Generic Telugu Movie Extractor.
3. In **Step 3**, add your list of YouTube Telugu movie links to `MOVIE_LINKS` and run!

---

## 💻 CLI Usage

### Extract a Single YouTube Movie
```bash
python3 extract_youtube_movie.py \
  --url "https://www.youtube.com/watch?v=sOOrFtP6DIE" \
  --out-dir "./movie_corpus"
```

### Extract a Batch of Movies from a Text File of YouTube Links
```bash
python3 extract_youtube_movie.py \
  --urls-file ./youtube_movie_links.txt \
  --out-dir "./movie_corpus"
```
