"""
Generic YouTube Movie Subtitle & Scene Corpus Extractor (`extract_youtube_movie.py`).

Allows attaching full movie metadata (`movie_name`, `year`, `director`, `genre`, `dialect_hint`)
alongside each YouTube link (either as a Python list of dicts in Colab or via a `.csv` input file).

Outputs per movie in `<out_dir>/<movie_name>/`:
  1. `<movie_name>_raw.srt`     -> Raw subtitle track downloaded from YouTube
  2. `<movie_name>_clean.srt`   -> Cleaned Telugu `.srt` ([సంగీతం], [బెల్], disclaimers & BGM humming removed)
  3. `<movie_name>_scenes.csv`  -> Scene-based CSV with movie metadata (`movie_name`, `year`, `director`, `movie_genre`)
                                   + scene tags (`scene_emotion`, `scene_dialect`, `start_time`, `end_time`, `dialogue_telugu`)
  4. `<movie_name>_scenes.txt`  -> Clean human-readable Telugu scene script with movie & scene headers
  5. `<movie_name>_scenes.jsonl`-> Structured JSONL for downstream corpus storage
"""

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tenglish_lm.movie_harvester import (
    SubtitleBlock,
    blocks_to_clean_srt,
    parse_srt_content,
)


# ---------------------------------------------------------------------------
# 1. Automated Telugu Scene Emotion & Dialect Classifier
# ---------------------------------------------------------------------------

EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "EMOTIONAL_HEARTBREAK": [
        "బాధ", "ఏడుపు", "మోసం", "వదిలేసి", "మర్చిపోవడం", "మర్చిపోలేను", "చచ్చిపోతా",
        "ఒంటరి", "తట్టుకోలేను", "వద్దు", "నచ్చట్లేదు", "వెళ్ళిపోతున్నాను", "కన్నీళ్లు",
        "ప్రేమ", "గుండె", "విడిపో", "క్షమించు", "సారీ",
    ],
    "ANGER_WARNING": [
        "చంపేస్తా", "నరికేస్తా", "కత్తి", "రక్తం", "ఎదురు", "జాగ్రత్త", "వార్నింగ్",
        "కోపం", "తప్పు", "ఏంట్రా", "ఏందిరా", "మూసుకో", "కొడతా", "అరెస్ట్", "పోలీస్",
        "ప్రాణభయం", "చంపు", "దమ్ము",
    ],
    "COMEDY": [
        "నవ్వు", "సరదా", "ఐస్", "క్రీమ్", "కిక్", "సెలవు", "పుట్టింటికి", "డబ్బా",
        "ఉప్పు", "కారం", "మెస్", "ఫైన్", "సంతకం", "లోన్", "కాఫీ", "పెళ్లి", "అల్లుడు",
        "మామ", "బిల్డప్", "కామెడీ", "జోక్",
    ],
    "SARCASTIC_ROAST": [
        "అంతేగా", "సాకు", "మునిగిపోతవ్", "లైట్", "ఆప్షన్", "డిసైడ్", "షర్ట్", "షూ",
        "ఓవర్", "యాక్షన్", "సోది", "బిల్డప్",
    ],
    "PHILOSOPHICAL": [
        "జీవితం", "మనిషికి", "మాటే", "ప్రాణం", "విలువ", "యుద్ధం", "యుద్ధంలో",
        "గెలవడం", "శత్రువు", "ఓడిపోయినా", "ధైర్యం", "నిజాయితీ", "బాధ్యత",
        "సొంతంగా", "నడవడం", "వీలైతే", "నాలుగు మాటలు", "నిజం", "ధర్మం",
    ],
    "ELEVATION_MASS": [
        "కమిట్", "భయం", "ఆపరేషన్", "లెక్క", "చరిత్ర", "రికార్డ్", "పవర్", "రాజు",
        "సింహం", "వేట", "సామ్రాజ్యం",
    ],
}

DIALECT_KEYWORDS: Dict[str, List[str]] = {
    "TELANGANA": [
        "ఏంది", "భయ్యా", "అంటున్నవ్", "చెప్పిండు", "చేసిండు", "అయితది", "అయితంది",
        "దోస్తు", "బై", "గట్ల", "కూసున్నవ్", "వస్తుండు", "పోతుండు", "మల్ల", "గిట్ల",
    ],
    "RAYALASEEMA": [
        "ఏందిరా", "పొలిమేర", "కత్తి", "రక్తం", "నరికేస్తా", "సామి", "అప్పా", "బిడ్డ",
    ],
    "COASTAL_ANDHRA": [
        "ఏమండోయ్", "గారూ", "అబ్బాయ్", "ఆవిడ", "కదండీ", "చెప్పమంటారు", "అండి",
    ],
}


def detect_scene_emotion_and_dialect(
    turns: List[str],
    default_dialect: str = "STANDARD_TELUGU",
) -> Tuple[str, str]:
    """
    Classifies a scene's [SCENE_EMOTION] and [SCENE_DIALECT] from its Telugu dialogue turns.
    """
    joined = " ".join(turns).lower()

    scores: Dict[str, int] = {emo: 0 for emo in EMOTION_KEYWORDS}
    for emo, kw_list in EMOTION_KEYWORDS.items():
        for kw in kw_list:
            if kw.lower() in joined:
                scores[emo] += 2

    best_emotion = max(scores.items(), key=lambda x: x[1])[0]
    if scores[best_emotion] == 0:
        best_emotion = "CONVERSATIONAL_DRAMA"

    best_dialect = default_dialect or "STANDARD_TELUGU"
    for dial, kw_list in DIALECT_KEYWORDS.items():
        if any(kw in joined for kw in kw_list):
            best_dialect = dial
            break

    return best_emotion, best_dialect


def format_seconds_to_hms(seconds: float) -> str:
    """Formats seconds (e.g. 3725.4) into 'HH:MM:SS'."""
    s = int(max(0, seconds))
    hrs = s // 3600
    mins = (s % 3600) // 60
    secs = s % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def sanitize_movie_slug(title: str) -> str:
    """Converts a movie name or raw YouTube video title into a clean folder/file slug."""
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", title.strip()).strip("_")
    return slug[:60] if slug else "Telugu_Movie"


# ---------------------------------------------------------------------------
# 2. Timestamped Scene Slicer with User Movie Metadata
# ---------------------------------------------------------------------------

def slice_movie_blocks_with_timestamps(
    blocks: List[SubtitleBlock],
    movie_name: str,
    youtube_url: str = "",
    year: str = "UNKNOWN",
    director: str = "UNKNOWN",
    movie_genre: str = "DRAMA",
    dialect_hint: str = "STANDARD_TELUGU",
    same_turn_gap_sec: float = 1.2,
    new_scene_gap_sec: float = 4.5,
    max_turns_per_scene: int = 4,
    max_words_per_turn: int = 16,
) -> List[Dict[str, Any]]:
    """
    Slices an entire movie's cleaned subtitle blocks into timestamped 2-to-4 turn scenes,
    attaches user-supplied movie metadata (`movie_name`, `year`, `director`, `movie_genre`),
    and auto-tags each scene's `scene_emotion` and `scene_dialect`.
    """
    if not blocks:
        return []

    raw_grouped_scenes: List[Tuple[float, float, List[str]]] = []
    cur_turns: List[str] = []
    cur_buf: List[str] = [blocks[0].text]
    cur_buf_words: int = len(blocks[0].text.split())
    scene_start = blocks[0].start_sec
    prev_end = blocks[0].end_sec

    for b in blocks[1:]:
        gap = b.start_sec - prev_end
        b_words = len(b.text.split())

        if gap <= same_turn_gap_sec and (cur_buf_words + b_words <= max_words_per_turn):
            cur_buf.append(b.text)
            cur_buf_words += b_words
        elif gap <= new_scene_gap_sec and len(cur_turns) < max_turns_per_scene - 1:
            cur_turns.append(" ".join(cur_buf))
            cur_buf = [b.text]
            cur_buf_words = b_words
        else:
            cur_turns.append(" ".join(cur_buf))
            if len(cur_turns) >= 2:
                raw_grouped_scenes.append((scene_start, prev_end, cur_turns[:max_turns_per_scene]))
            cur_turns = []
            cur_buf = [b.text]
            cur_buf_words = b_words
            scene_start = b.start_sec
        prev_end = b.end_sec

    if cur_buf:
        cur_turns.append(" ".join(cur_buf))
    if len(cur_turns) >= 2:
        raw_grouped_scenes.append((scene_start, prev_end, cur_turns[:max_turns_per_scene]))
    elif len(cur_turns) == 1 and raw_grouped_scenes:
        last_st, _, last_turns = raw_grouped_scenes[-1]
        if len(last_turns) < max_turns_per_scene:
            raw_grouped_scenes[-1] = (last_st, prev_end, last_turns + cur_turns)

    tagged_scenes: List[Dict[str, Any]] = []
    for idx, (st_sec, end_sec, turns) in enumerate(raw_grouped_scenes, start=1):
        scene_emotion, scene_dialect = detect_scene_emotion_and_dialect(
            turns, default_dialect=dialect_hint
        )
        tagged_scenes.append(
            {
                "scene_id": f"{movie_name}_SC_{idx:04d}",
                "movie_name": movie_name,
                "year": str(year),
                "director": director,
                "movie_genre": movie_genre,
                "scene_emotion": scene_emotion,
                "scene_dialect": scene_dialect,
                "start_time": format_seconds_to_hms(st_sec),
                "end_time": format_seconds_to_hms(end_sec),
                "duration_sec": round(max(0.0, end_sec - st_sec), 1),
                "turn_count": len(turns),
                "turns_telugu": turns,
                "youtube_url": youtube_url,
            }
        )

    return tagged_scenes


# ---------------------------------------------------------------------------
# 3. Export Pure Telugu Movie Corpus (.SRT, .CSV, .TXT, .JSONL)
# ---------------------------------------------------------------------------

def export_movie_to_csv_txt_jsonl(
    scenes: List[Dict[str, Any]],
    blocks: List[SubtitleBlock],
    movie_name: str,
    year: str,
    director: str,
    movie_genre: str,
    youtube_url: str,
    out_dir: str | Path,
) -> Dict[str, Path]:
    """
    Writes the extracted & cleaned Telugu movie files to `<out_dir>`:
      1. `<movie_name>_clean.srt`
      2. `<movie_name>_scenes.csv`
      3. `<movie_name>_scenes.txt`
      4. `<movie_name>_scenes.jsonl`
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    clean_srt_file = out_path / f"{movie_name}_clean.srt"
    csv_file = out_path / f"{movie_name}_scenes.csv"
    txt_file = out_path / f"{movie_name}_scenes.txt"
    jsonl_file = out_path / f"{movie_name}_scenes.jsonl"

    # 1. Write cleaned Telugu .SRT
    clean_srt_file.write_text(blocks_to_clean_srt(blocks), encoding="utf-8")

    # 2. Write .CSV (Movie Metadata + Scene Tags + Pure Telugu Dialogue)
    csv_columns = [
        "scene_id",
        "movie_name",
        "year",
        "director",
        "movie_genre",
        "scene_emotion",
        "scene_dialect",
        "start_time",
        "end_time",
        "duration_sec",
        "turn_count",
        "dialogue_telugu",
        "youtube_url",
    ]
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for sc in scenes:
            writer.writerow(
                {
                    "scene_id": sc["scene_id"],
                    "movie_name": sc["movie_name"],
                    "year": sc["year"],
                    "director": sc["director"],
                    "movie_genre": sc["movie_genre"],
                    "scene_emotion": sc["scene_emotion"],
                    "scene_dialect": sc["scene_dialect"],
                    "start_time": sc["start_time"],
                    "end_time": sc["end_time"],
                    "duration_sec": sc["duration_sec"],
                    "turn_count": sc["turn_count"],
                    "dialogue_telugu": " // ".join(sc["turns_telugu"]),
                    "youtube_url": sc["youtube_url"],
                }
            )

    # 3. Write .TXT (Clean Scene-by-Scene Tagged Telugu Script)
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write(f"TELUGU MOVIE SCENE CORPUS: {movie_name} ({len(scenes)} Scenes)\n")
        f.write(f"YEAR: {year} | DIRECTOR: {director} | MOVIE GENRE: {movie_genre}\n")
        if youtube_url:
            f.write(f"YOUTUBE URL: {youtube_url}\n")
        f.write("================================================================================\n\n")
        for sc in scenes:
            header = (
                f"[{sc['scene_id']}] [{sc['start_time']} -> {sc['end_time']}] "
                f"[DIRECTOR: {sc['director']}] [MOVIE_GENRE: {sc['movie_genre']}] "
                f"[SCENE_EMOTION: {sc['scene_emotion']}] [DIALECT: {sc['scene_dialect']}]"
            )
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for i, line in enumerate(sc["turns_telugu"], start=1):
                f.write(f"  Speaker {i}: {line}\n")
            f.write("\n")

    # 4. Write .JSONL
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for sc in scenes:
            f.write(json.dumps(sc, ensure_ascii=False) + "\n")

    return {
        "clean_srt": clean_srt_file,
        "csv": csv_file,
        "txt": txt_file,
        "jsonl": jsonl_file,
    }


# ---------------------------------------------------------------------------
# 4. YouTube Subtitle Downloader (Single URL & Batch Movie Corpus Builder)
# ---------------------------------------------------------------------------

def fetch_youtube_video_title(youtube_url: str) -> Optional[str]:
    """Fetches the video title from YouTube via yt-dlp when movie_name is not manually specified."""
    try:
        proc = subprocess.run(
            ["yt-dlp", "--get-title", "--no-warnings", youtube_url],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        title = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
        return sanitize_movie_slug(title) if title else None
    except Exception:
        return None


def check_and_download_youtube_srt(
    youtube_url: str,
    work_dir: Path,
    movie_name: str,
    allow_auto_captions: bool = True,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """
    Extracts existing subtitles or YouTube auto-generated Telugu/English captions.
    Priority order:
      1. Manual Telugu track (`te`, `te-IN`)
      2. YouTube Auto-Generated Telugu track (`te`, `te-IN`)
      3. Manual English track (`en`, `en-IN`, `en-US`, `en-GB`)
      4. YouTube Auto-Generated English track (`en.*`)
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {
        "youtube_url": youtube_url,
        "manual_langs": [],
        "auto_langs": [],
        "selected_track": None,
        "status": "NO_SUBTITLES_FOUND",
    }

    vid_match = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", youtube_url)
    video_id = vid_match.group(1) if vid_match else None

    if video_id:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore

            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.list(video_id)
            manual_tracks = [t for t in transcript_list if not t.is_generated]
            auto_tracks = [t for t in transcript_list if t.is_generated]

            meta["manual_langs"] = [f"{t.language_code} ({t.language})" for t in manual_tracks]
            meta["auto_langs"] = [f"{t.language_code} ({t.language})" for t in auto_tracks]

            chosen = None
            for t in manual_tracks:
                if t.language_code.startswith("te"):
                    chosen = t
                    meta["selected_track"] = f"MANUAL:{t.language_code}"
                    break
            if chosen is None and allow_auto_captions:
                for t in auto_tracks:
                    if t.language_code.startswith("te"):
                        chosen = t
                        meta["selected_track"] = f"AUTO_YOUTUBE_TE:{t.language_code}"
                        break
            if chosen is None:
                for t in manual_tracks:
                    if t.language_code.startswith("en"):
                        chosen = t
                        meta["selected_track"] = f"MANUAL:{t.language_code}"
                        break
            if chosen is None and allow_auto_captions:
                for t in auto_tracks:
                    if t.language_code.startswith("en"):
                        chosen = t
                        meta["selected_track"] = f"AUTO_YOUTUBE_EN:{t.language_code}"
                        break

            if chosen is not None:
                fetched = list(chosen.fetch())
                srt_lines = []
                for idx, item in enumerate(fetched):
                    start_s = float(getattr(item, "start", item["start"] if isinstance(item, dict) else 0.0))
                    dur_s = float(getattr(item, "duration", item.get("duration", 2.5) if isinstance(item, dict) else 2.5))
                    text_val = str(getattr(item, "text", item["text"] if isinstance(item, dict) else "")).strip()
                    if not text_val:
                        continue
                    end_s = start_s + max(0.8, dur_s)
                    if idx + 1 < len(fetched):
                        next_item = fetched[idx + 1]
                        next_start = float(
                            getattr(next_item, "start", next_item["start"] if isinstance(next_item, dict) else end_s)
                        )
                        if end_s > next_start and next_start > start_s:
                            end_s = next_start
                    st_hms = format_seconds_to_hms(start_s) + ",000"
                    et_hms = format_seconds_to_hms(end_s) + ",000"
                    srt_lines.append(f"{idx + 1}\n{st_hms} --> {et_hms}\n{text_val}\n")

                if srt_lines:
                    srt_out = work_dir / f"{movie_name}_raw.{chosen.language_code}.srt"
                    srt_out.write_text("\n".join(srt_lines), encoding="utf-8")
                    meta["status"] = "SUCCESS_YOUTUBE_TRANSCRIPT_API"
                    return srt_out, meta
        except Exception as e:
            meta["transcript_api_note"] = str(e)

    # Fallback: yt-dlp subtitle download
    out_template = str(work_dir / f"{movie_name}_raw.%(ext)s")
    cmd = [
        "yt-dlp",
        "--write-sub",
        *(["--write-auto-sub"] if allow_auto_captions else []),
        "--sub-langs",
        "te.*,en.*",
        "--sub-format",
        "srt/vtt",
        "--convert-subs",
        "srt",
        "--skip-download",
        "-o",
        out_template,
        youtube_url,
    ]
    print(f"[YT-DLP] Downloading subtitles (`te.*`, `en.*`, auto={allow_auto_captions}) from: {youtube_url} ...")
    subprocess.run(cmd, check=False)
    srts = sorted(work_dir.glob(f"{movie_name}_raw*.srt")) + sorted(work_dir.glob(f"{movie_name}_raw*.vtt"))
    if srts:
        te_srts = [p for p in srts if ".te" in p.name]
        best_srt = te_srts[0] if te_srts else srts[0]
        meta["selected_track"] = best_srt.name
        meta["status"] = "SUCCESS_YTDLP_SUBS"
        return best_srt, meta

    return None, meta


def extract_full_movie_pipeline(
    youtube_url: Optional[str] = None,
    movie_name: Optional[str] = None,
    year: str = "UNKNOWN",
    director: str = "UNKNOWN",
    movie_genre: str = "DRAMA",
    dialect_hint: str = "STANDARD_TELUGU",
    srt_file: Optional[str | Path] = None,
    out_dir: str | Path = "./movie_corpus",
    allow_auto_captions: bool = True,
) -> Dict[str, Any]:
    """
    Extracts a single movie with its attached metadata (`movie_name`, `year`, `director`, `movie_genre`, `dialect_hint`),
    cleans BGM/disclaimers, slices into scenes, tags scene emotions, and stores `<movie_name>_raw.srt`,
    `<movie_name>_clean.srt`, `<movie_name>_scenes.csv`, `<movie_name>_scenes.txt`, and `<movie_name>_scenes.jsonl`.
    """
    if not movie_name or movie_name.strip() == "" or movie_name == "AUTO":
        if youtube_url:
            detected_title = fetch_youtube_video_title(youtube_url)
            vid_match = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", youtube_url)
            vid_id = vid_match.group(1) if vid_match else "Movie"
            movie_name = f"{detected_title}_{vid_id}" if detected_title else f"Movie_{vid_id}"
        elif srt_file:
            movie_name = sanitize_movie_slug(Path(srt_file).stem)
        else:
            movie_name = "Telugu_Movie"
    else:
        movie_name = sanitize_movie_slug(movie_name)

    out_path = Path(out_dir) / movie_name
    out_path.mkdir(parents=True, exist_ok=True)

    target_srt: Optional[Path] = Path(srt_file) if srt_file else None
    sub_meta: Dict[str, Any] = {}
    if youtube_url and (target_srt is None or not target_srt.exists()):
        target_srt, sub_meta = check_and_download_youtube_srt(
            youtube_url, out_path, movie_name, allow_auto_captions=allow_auto_captions
        )

    if target_srt is None or not target_srt.exists():
        raise FileNotFoundError(
            f"❌ No manual or auto-generated subtitles found on YouTube for '{movie_name}' ({youtube_url}).\n"
            f"   Manual tracks: {sub_meta.get('manual_langs', [])} | Auto tracks: {sub_meta.get('auto_langs', [])}"
        )

    srt_text = target_srt.read_text(encoding="utf-8", errors="replace")
    blocks = parse_srt_content(srt_text)
    scenes = slice_movie_blocks_with_timestamps(
        blocks=blocks,
        movie_name=movie_name,
        youtube_url=youtube_url or "",
        year=str(year),
        director=director,
        movie_genre=movie_genre,
        dialect_hint=dialect_hint,
    )
    exported = export_movie_to_csv_txt_jsonl(
        scenes=scenes,
        blocks=blocks,
        movie_name=movie_name,
        year=str(year),
        director=director,
        movie_genre=movie_genre,
        youtube_url=youtube_url or "",
        out_dir=out_path,
    )

    return {
        "movie_name": movie_name,
        "year": str(year),
        "director": director,
        "movie_genre": movie_genre,
        "dialect_hint": dialect_hint,
        "youtube_url": youtube_url or "",
        "selected_track": sub_meta.get("selected_track", target_srt.name),
        "raw_srt_file": str(target_srt),
        "clean_subtitle_blocks": len(blocks),
        "scenes_extracted": len(scenes),
        "files": {k: str(v) for k, v in exported.items()},
        "emotion_distribution": {
            emo: sum(1 for s in scenes if s["scene_emotion"] == emo)
            for emo in sorted({s["scene_emotion"] for s in scenes})
        },
    }


def extract_movie_corpus_batch(
    movie_links: List[Any],
    out_dir: str | Path = "./movie_corpus",
    allow_auto_captions: bool = True,
) -> Dict[str, Any]:
    """
    Processes a list of movie dicts (with `url`, `movie_name`, `year`, `director`, `genre`, `dialect_hint`),
    extracting and storing each movie in `<out_dir>/<movie_name>/` and writing `<out_dir>/corpus_manifest.csv`.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    for idx, entry in enumerate(movie_links, start=1):
        url = ""
        name = None
        year = "UNKNOWN"
        director = "UNKNOWN"
        movie_genre = "DRAMA"
        dialect_hint = "STANDARD_TELUGU"

        if isinstance(entry, str):
            url = entry.strip()
        elif isinstance(entry, dict):
            url = str(entry.get("url") or entry.get("youtube_url") or "").strip()
            name = entry.get("movie_name") or entry.get("name")
            year = str(entry.get("year") or entry.get("era") or "UNKNOWN").strip()
            director = str(entry.get("director") or entry.get("director_style") or "UNKNOWN").strip()
            movie_genre = str(entry.get("genre") or entry.get("movie_genre") or "DRAMA").strip()
            dialect_hint = str(entry.get("dialect") or entry.get("dialect_hint") or "STANDARD_TELUGU").strip()
        elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
            name, url = str(entry[0]).strip(), str(entry[1]).strip()

        if not url:
            continue

        label = name if name else url
        print(
            f"\n[{idx}/{len(movie_links)}] 🎬 Extracting '{label}' "
            f"(Year: {year} | Director: {director} | Genre: {movie_genre}) ..."
        )
        try:
            res = extract_full_movie_pipeline(
                youtube_url=url,
                movie_name=name,
                year=year,
                director=director,
                movie_genre=movie_genre,
                dialect_hint=dialect_hint,
                out_dir=out_root,
                allow_auto_captions=allow_auto_captions,
            )
            results.append(res)
            manifest_rows.append(
                {
                    "movie_name": res["movie_name"],
                    "year": res["year"],
                    "director": res["director"],
                    "movie_genre": res["movie_genre"],
                    "dialect_hint": res["dialect_hint"],
                    "youtube_url": url,
                    "status": "EXTRACTED",
                    "selected_track": res["selected_track"],
                    "clean_subtitle_blocks": res["clean_subtitle_blocks"],
                    "scenes_extracted": res["scenes_extracted"],
                    "clean_srt_path": res["files"]["clean_srt"],
                    "csv_path": res["files"]["csv"],
                    "txt_path": res["files"]["txt"],
                }
            )
            print(
                f"   ✅ Saved {res['scenes_extracted']} clean Telugu scenes -> {res['files']['csv']}"
            )
        except Exception as e:
            print(f"   ⚠️ Skipped ({url}): {e}")
            manifest_rows.append(
                {
                    "movie_name": name or "UNKNOWN",
                    "year": year,
                    "director": director,
                    "movie_genre": movie_genre,
                    "dialect_hint": dialect_hint,
                    "youtube_url": url,
                    "status": f"SKIPPED: {e}",
                    "selected_track": "",
                    "clean_subtitle_blocks": 0,
                    "scenes_extracted": 0,
                    "clean_srt_path": "",
                    "csv_path": "",
                    "txt_path": "",
                }
            )

    manifest_file = out_root / "corpus_manifest.csv"
    if manifest_rows:
        with open(manifest_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    return {
        "total_links": len(movie_links),
        "movies_extracted": len(results),
        "manifest_csv": str(manifest_file),
        "results": results,
    }


def load_movie_list_from_csv(csv_path: str | Path) -> List[Dict[str, str]]:
    """Loads a batch movie list from a `.csv` file with columns: url, movie_name, year, director, genre, dialect."""
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): (v.strip() if v else "") for k, v in row.items() if k})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract clean Telugu .srt, scene-tagged .csv, and .txt files from YouTube movie URLs with metadata."
    )
    parser.add_argument("--url", type=str, default=None, help="Single YouTube movie URL")
    parser.add_argument("--movie-name", type=str, default=None, help="Movie name (e.g. Subramanyam_For_Sale)")
    parser.add_argument("--year", type=str, default="UNKNOWN", help="Release year or era (e.g. 2015 or 2010s)")
    parser.add_argument("--director", type=str, default="UNKNOWN", help="Director / Writer name (e.g. Harish Shankar)")
    parser.add_argument("--genre", type=str, default="DRAMA", help="Movie genre (e.g. RomCom / Family)")
    parser.add_argument("--dialect", type=str, default="STANDARD_TELUGU", help="Default dialect hint")
    parser.add_argument(
        "--movies-csv",
        type=str,
        default=None,
        help="Path to a .csv file with columns: url,movie_name,year,director,genre,dialect",
    )
    parser.add_argument("--srt-file", type=str, default=None, help="Local .srt subtitle file path")
    parser.add_argument("--out-dir", type=str, default="./movie_corpus", help="Output corpus folder")
    args = parser.parse_args()

    if args.movies_csv:
        movie_entries = load_movie_list_from_csv(args.movies_csv)
        summary = extract_movie_corpus_batch(movie_entries, out_dir=args.out_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        res = extract_full_movie_pipeline(
            youtube_url=args.url,
            movie_name=args.movie_name,
            year=args.year,
            director=args.director,
            movie_genre=args.genre,
            dialect_hint=args.dialect,
            srt_file=args.srt_file,
            out_dir=args.out_dir,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
