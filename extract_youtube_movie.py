"""
YouTube Full-Movie Extractor & Auto-Tagger CLI (`extract_youtube_movie.py`).

Usage (in Google Colab or local terminal):
  python3 extract_youtube_movie.py \
      --url "https://www.youtube.com/watch?v=..." \
      --movie-name "Bommarillu_2006" \
      --era "2000s" \
      --director-style "BHASKAR_ABBURI_RAVI" \
      --out-dir "./drive_mirror/Tenglish_LM/extracted_movies"

Or test on a local .srt file:
  python3 extract_youtube_movie.py \
      --srt-file "./drive_mirror/Tenglish_LM/bommarillu_2006/bommarillu_2006.srt" \
      --movie-name "Bommarillu_2006" \
      --era "2000s" \
      --director-style "BHASKAR_ABBURI_RAVI"

Outputs 3 files automatically:
1. `<movie_name>_scenes.csv`   -> Spreadsheet-ready CSV with Scene_ID, Start_Time, End_Time, Emotion_Tag,
                                 Genre_Tag, Dialect_Tag, English_CMI_Pct, Original_Telugu,
                                 P1_Standard_Tenglish, P2_Minimal_Tenglish, P4_Broken_Acrophonic_Tenglish.
2. `<movie_name>_scenes.txt`   -> Clean, human-readable scene-by-scene script with control tags (`[EMOTION: ...]`).
3. `<movie_name>_scenes.jsonl` -> Machine-ready JSONL for SLM training.
"""

import argparse
import asyncio
import csv
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tenglish_lm.movie_harvester import SubtitleBlock, parse_srt_content
from tenglish_lm.tenglish_evo import compute_utterance_cmi, romanize_with_persona


# ---------------------------------------------------------------------------
# 1. Automated Scene Emotion & Dialect Classifier (Lexical + Optional Gemini Flash API)
# ---------------------------------------------------------------------------

EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "HEARTBREAK": [
        "బాధ", "ఏడుపు", "మోసం", "వదిలేసి", "మర్చిపోవడం", "మర్చిపోలేను", "చచ్చిపోతా",
        "ఒంటరి", "తట్టుకోలేను", "వద్దు", "నచ్చట్లేదు", "వెళ్ళిపోతున్నాను", "నాటకం",
        "కోరుకున్నది", "దొరక్కపోతే", "pain", "hurt", "cry", "breakup", "sorry",
    ],
    "ANGRY_WARNING": [
        "చంపేస్తా", "నరికేస్తా", "కత్తి", "రక్తం", "ఎదురు", "జాగ్రత్త", "వార్నింగ్",
        "కోపం", "తప్పు", "ఏంట్రా", "ఏందిరా", "మూసుకో", "కొడతా", "warning", "dare",
    ],
    "COMIC_TIMING": [
        "నవ్వు", "సరదా", "ఐస్", "క్రీమ్", "కిక్", "సెలవు", "పుట్టింటికి", "డబ్బా",
        "ఉప్పు", "కారం", "మెస్", "ఫైన్", "బొమ్మరిల్లు", "సంతకం", "లోన్", "కాఫీ",
        "funny", "joke", "ice", "cream", "kitchen", "mess",
    ],
    "SARCASTIC_ROAST": [
        "అంతేగా", "బిల్డప్", "హ్యాండ్", "సాకు", "మునిగిపోతవ్", "లైట్", "డెలులు",
        "ఘోస్ట్", "వైబ్", "ఆప్షన్", "ఇంకో నాన్న", "డిసైడ్", "షర్ట్", "షూ",
        "buildup", "ghost", "delulu", "vibe", "hand", "lite",
    ],
    "PHILOSOPHICAL": [
        "జీవితం", "మనిషికి", "మాటే", "ప్రాణం", "విలువ", "యుద్ధం", "యుద్ధంలో",
        "గెలవడం", "శత్రువు", "ఓడిపోయినా", "ధైర్యం", "నిజాయితీ", "బాధ్యత",
        "సొంతంగా", "నడవడం", "వీలైతే", "నాలుగు మాటలు",
    ],
    "ELEVATION_PUNCH": [
        "కమిట్", "భయం", "ఆపరేషన్", "హ్యాపీగా", "హ్యాపీనెస్", "లైఫ్ లాంగ్",
        "లెక్క", "చరిత్ర", "రికార్డ్", "పవర్", "commit", "power", "record",
    ],
}

DIALECT_KEYWORDS: Dict[str, List[str]] = {
    "TELANGANA": ["ఏంది", "భయ్యా", "అంటున్నవ్", "చెప్పిండు", "చేసిండు", "అయితది", "అయితంది", "దోస్తు", "బై", "గట్ల", "కూసున్నవ్"],
    "RAYALASEEMA": ["ఏందిరా", "పొలిమేర", "కత్తి", "రక్తం", "నరికేస్తా", "సామి", "అప్పా"],
    "COASTAL_ANDHRA": ["ఏమండోయ్", "గారూ", "అబ్బాయ్", "ఆవిడ", "కదండీ", "చెప్పమంటారు"],
}


def detect_scene_emotion_and_dialect(turns: List[str], cmi_pct: float) -> Tuple[str, str, str]:
    """
    Classifies a scene's [EMOTION], [GENRE], and [DIALECT] from its Telugu/Tenglish dialogue turns.
    """
    joined = " ".join(turns).lower()

    # 1. Score Emotions
    scores: Dict[str, int] = {emo: 0 for emo in EMOTION_KEYWORDS}
    for emo, kw_list in EMOTION_KEYWORDS.items():
        for kw in kw_list:
            if kw.lower() in joined:
                scores[emo] += 2

    best_emotion = max(scores.items(), key=lambda x: x[1])[0]
    if scores[best_emotion] == 0:
        best_emotion = "CONVERSATIONAL_DRAMA"

    # 2. Map Emotion -> Genre
    genre_map = {
        "HEARTBREAK": "FAMILY_DRAMA",
        "ANGRY_WARNING": "MASS_ACTION",
        "COMIC_TIMING": "SATIRE_COMEDY",
        "SARCASTIC_ROAST": "ROMCOM",
        "PHILOSOPHICAL": "FAMILY_DRAMA",
        "ELEVATION_PUNCH": "MASS_ACTION",
        "CONVERSATIONAL_DRAMA": "ROMCOM" if cmi_pct >= 12.0 else "FAMILY_DRAMA",
    }
    best_genre = genre_map.get(best_emotion, "FAMILY_DRAMA")

    # 3. Detect Dialect
    best_dialect = "URBAN_HYDERABAD" if cmi_pct >= 10.0 else "STANDARD_TELUGU"
    for dial, kw_list in DIALECT_KEYWORDS.items():
        if any(kw in joined for kw in kw_list):
            best_dialect = dial
            break

    return best_emotion, best_genre, best_dialect


def format_seconds_to_hms(seconds: float) -> str:
    """Formats seconds (e.g. 3725.4) into 'HH:MM:SS'."""
    s = int(max(0, seconds))
    hrs = s // 3600
    mins = (s % 3600) // 60
    secs = s % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# 2. Timestamped Scene Slicer (Preserves Start & End Timestamps per Scene)
# ---------------------------------------------------------------------------

def slice_movie_blocks_with_timestamps(
    blocks: List[SubtitleBlock],
    movie_name: str,
    era: str,
    director_style: str,
    same_turn_gap_sec: float = 1.2,
    new_scene_gap_sec: float = 4.5,
    max_turns_per_scene: int = 4,
) -> List[Dict[str, Any]]:
    """
    Slices an entire movie's subtitle blocks into timestamped 2-to-4 turn scenes,
    auto-tags Emotion/Genre/Dialect/CMI %, and generates all 4 Tenglish Personas.
    """
    if not blocks:
        return []

    raw_grouped_scenes: List[Tuple[float, float, List[str]]] = []
    cur_turns: List[str] = []
    cur_buf: List[str] = [blocks[0].text]
    scene_start = blocks[0].start_sec
    prev_end = blocks[0].end_sec

    for b in blocks[1:]:
        gap = b.start_sec - prev_end
        if gap <= same_turn_gap_sec:
            cur_buf.append(b.text)
        elif gap <= new_scene_gap_sec and len(cur_turns) < max_turns_per_scene - 1:
            cur_turns.append(" ".join(cur_buf))
            cur_buf = [b.text]
        else:
            cur_turns.append(" ".join(cur_buf))
            if len(cur_turns) >= 2:
                raw_grouped_scenes.append((scene_start, prev_end, cur_turns[:max_turns_per_scene]))
            cur_turns = []
            cur_buf = [b.text]
            scene_start = b.start_sec
        prev_end = b.end_sec

    if cur_buf:
        cur_turns.append(" ".join(cur_buf))
    if len(cur_turns) >= 2:
        raw_grouped_scenes.append((scene_start, prev_end, cur_turns[:max_turns_per_scene]))

    # Process each scene into rich tagged dictionary
    tagged_scenes: List[Dict[str, Any]] = []
    for idx, (st_sec, end_sec, turns) in enumerate(raw_grouped_scenes, start=1):
        joined_te = " ".join(turns)
        cmi_info = compute_utterance_cmi(joined_te)
        cmi_pct = cmi_info["en_lexeme_pct"]

        emotion, genre, dialect = detect_scene_emotion_and_dialect(turns, cmi_pct)

        p1_turns = [romanize_with_persona(t, "P1_STANDARD", "ERA_2025") for t in turns]
        p2_turns = [romanize_with_persona(t, "P2_MINIMAL", "ERA_2025") for t in turns]
        p3_turns = [romanize_with_persona(t, "P3_EXPRESS", "ERA_2025") for t in turns]
        p4_turns = [romanize_with_persona(t, "P4_ACROPHON", "ERA_2025") for t in turns]

        tagged_scenes.append(
            {
                "scene_id": f"{movie_name}_SC_{idx:04d}",
                "movie_name": movie_name,
                "start_time": format_seconds_to_hms(st_sec),
                "end_time": format_seconds_to_hms(end_sec),
                "era": era,
                "director_style": director_style,
                "emotion": emotion,
                "genre": genre,
                "dialect": dialect,
                "en_cmi_pct": cmi_pct,
                "en_word_count": cmi_info["counts"]["EN"],
                "hybrid_word_count": cmi_info["counts"]["HYBRID_FUSED"],
                "light_verb_count": cmi_info["counts"]["LIGHT_VERB"],
                "turn_count": len(turns),
                "turns_telugu": turns,
                "turns_p1_standard": p1_turns,
                "turns_p2_minimal": p2_turns,
                "turns_p3_expressive": p3_turns,
                "turns_p4_acrophonic": p4_turns,
            }
        )

    return tagged_scenes


# ---------------------------------------------------------------------------
# 3. Export to .CSV, .TXT (Scene-Based Script), and .JSONL
# ---------------------------------------------------------------------------

def export_movie_to_csv_txt_jsonl(
    scenes: List[Dict[str, Any]],
    movie_name: str,
    out_dir: str | Path,
) -> Dict[str, Path]:
    """
    Writes the extracted & tagged movie scenes to:
    1. `<movie_name>_scenes.csv`
    2. `<movie_name>_scenes.txt`
    3. `<movie_name>_scenes.jsonl`
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_file = out_path / f"{movie_name}_scenes.csv"
    txt_file = out_path / f"{movie_name}_scenes.txt"
    jsonl_file = out_path / f"{movie_name}_scenes.jsonl"

    # 1. Write .CSV (Spreadsheet / Pandas / Google Sheets ready)
    csv_columns = [
        "scene_id",
        "movie_name",
        "start_time",
        "end_time",
        "era",
        "director_style",
        "emotion",
        "genre",
        "dialect",
        "en_cmi_pct",
        "turn_count",
        "dialogue_telugu",
        "dialogue_tenglish_standard",
        "dialogue_tenglish_minimal",
        "dialogue_tenglish_broken_acrophonic",
    ]
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for sc in scenes:
            writer.writerow(
                {
                    "scene_id": sc["scene_id"],
                    "movie_name": sc["movie_name"],
                    "start_time": sc["start_time"],
                    "end_time": sc["end_time"],
                    "era": sc["era"],
                    "director_style": sc["director_style"],
                    "emotion": sc["emotion"],
                    "genre": sc["genre"],
                    "dialect": sc["dialect"],
                    "en_cmi_pct": sc["en_cmi_pct"],
                    "turn_count": sc["turn_count"],
                    "dialogue_telugu": " // ".join(sc["turns_telugu"]),
                    "dialogue_tenglish_standard": " // ".join(sc["turns_p1_standard"]),
                    "dialogue_tenglish_minimal": " // ".join(sc["turns_p2_minimal"]),
                    "dialogue_tenglish_broken_acrophonic": " // ".join(sc["turns_p4_acrophonic"]),
                }
            )

    # 2. Write .TXT (Clean Scene-by-Scene Tagged Script)
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"================================================================================\n")
        f.write(f"MOVIE SCRIPT & TENGLISH CORPUS: {movie_name} ({len(scenes)} Scenes)\n")
        f.write(f"================================================================================\n\n")
        for sc in scenes:
            header = (
                f"[{sc['scene_id']}] [{sc['start_time']} -> {sc['end_time']}] "
                f"[ERA: {sc['era']}] [STYLE: {sc['director_style']}] "
                f"[EMOTION: {sc['emotion']}] [GENRE: {sc['genre']}] "
                f"[DIALECT: {sc['dialect']}] [EN_CMI: {sc['en_cmi_pct']}%]"
            )
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            f.write("  [TELUGU SCRIPT]\n")
            for i, line in enumerate(sc["turns_telugu"], start=1):
                f.write(f"    Speaker {i}: {line}\n")
            f.write("  [STANDARD TENGLISH (P1)]\n")
            for i, line in enumerate(sc["turns_p1_standard"], start=1):
                f.write(f"    Speaker {i}: {line}\n")
            f.write("  [MINIMALIST TENGLISH (P2)]\n")
            for i, line in enumerate(sc["turns_p2_minimal"], start=1):
                f.write(f"    Speaker {i}: {line}\n")
            f.write("  [BROKEN / ACROPHONIC TENGLISH (P4)]\n")
            for i, line in enumerate(sc["turns_p4_acrophonic"], start=1):
                f.write(f"    Speaker {i}: {line}\n")
            f.write("\n")

    # 3. Write .JSONL
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for sc in scenes:
            f.write(json.dumps(sc, ensure_ascii=False) + "\n")

    return {"csv": csv_file, "txt": txt_file, "jsonl": jsonl_file}


# ---------------------------------------------------------------------------
# 4. YouTube Link Downloader + Full CLI Entrypoint
# ---------------------------------------------------------------------------

def download_srt_from_youtube(youtube_url: str, work_dir: Path, movie_name: str) -> Optional[Path]:
    """
    Uses yt-dlp (and optional Whisper fallback) to extract the full movie .srt from a YouTube link.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(work_dir / f"{movie_name}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--write-sub",
        "--write-auto-sub",
        "--sub-lang",
        "te,en",
        "--sub-format",
        "srt",
        "--convert-subs",
        "srt",
        "--skip-download",
        "-o",
        out_template,
        youtube_url,
    ]
    print(f"[YT-DLP] Extracting subtitles from YouTube URL: {youtube_url} ...")
    subprocess.run(cmd, check=False)
    srts = sorted(work_dir.glob(f"{movie_name}*.srt"))
    return srts[0] if srts else None


def extract_full_movie_pipeline(
    movie_name: str,
    era: str = "2000s",
    director_style: str = "BHASKAR_ABBURI_RAVI",
    youtube_url: Optional[str] = None,
    srt_file: Optional[str | Path] = None,
    out_dir: str | Path = "./drive_mirror/Tenglish_LM/extracted_movies",
) -> Dict[str, Any]:
    """
    Main function you call in Colab or Python:
    Give it a `youtube_url` (or `srt_file`) -> outputs `.csv`, `.txt`, and `.jsonl` with scene emotion tags.
    """
    out_path = Path(out_dir) / movie_name
    out_path.mkdir(parents=True, exist_ok=True)

    target_srt: Optional[Path] = Path(srt_file) if srt_file else None
    if youtube_url and (target_srt is None or not target_srt.exists()):
        target_srt = download_srt_from_youtube(youtube_url, out_path, movie_name)

    if target_srt is None or not target_srt.exists():
        raise FileNotFoundError(
            f"Could not find or download .srt for {movie_name}. "
            f"In Colab, ensure yt-dlp is installed (`!pip install -q yt-dlp`) or pass `--srt-file`."
        )

    srt_text = target_srt.read_text(encoding="utf-8", errors="replace")
    blocks = parse_srt_content(srt_text)
    scenes = slice_movie_blocks_with_timestamps(
        blocks=blocks,
        movie_name=movie_name,
        era=era,
        director_style=director_style,
    )
    exported = export_movie_to_csv_txt_jsonl(scenes, movie_name, out_path)

    return {
        "movie_name": movie_name,
        "subtitle_blocks_parsed": len(blocks),
        "scenes_extracted": len(scenes),
        "files": {k: str(v) for k, v in exported.items()},
        "emotion_distribution": {
            emo: sum(1 for s in scenes if s["emotion"] == emo)
            for emo in sorted({s["emotion"] for s in scenes})
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a full YouTube Telugu movie into Scene-Tagged .CSV and .TXT files.")
    parser.add_argument("--url", type=str, default=None, help="YouTube full movie URL")
    parser.add_argument("--srt-file", type=str, default=None, help="Local .srt subtitle file path")
    parser.add_argument("--movie-name", type=str, default="Bommarillu_2006", help="Movie identifier")
    parser.add_argument("--era", type=str, default="2000s", help="Movie era tag (1980s, 1990s, 2000s, 2010s, 2020s)")
    parser.add_argument("--director-style", type=str, default="BHASKAR_ABBURI_RAVI", help="Director/Writer style tag")
    parser.add_argument("--out-dir", type=str, default="./drive_mirror/Tenglish_LM/extracted_movies", help="Output folder")
    args = parser.parse_args()

    res = extract_full_movie_pipeline(
        movie_name=args.movie_name,
        era=args.era,
        director_style=args.director_style,
        youtube_url=args.url,
        srt_file=args.srt_file,
        out_dir=args.out_dir,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
