"""
Milestone 1 (Day 1): Core Telugu Movie & Literature Seed Harvester.

Provides:
1. SRT / YouTube caption parser with pause-gap turn diarization (delta_t > 1.5s = new speaker turn)
2. Automated ASR / BGM noise cleaner & copyright-safe 2-to-4 turn scene slicer
3. yt-dlp wrapper for Colab CPU harvesting from YouTube Telugu movie channels
4. Rich curated Multi-Era Seed Corpus (1980s -> 2020s + Literature) spanning Director/Writer styles,
   Genres, Emotions, and Regional Dialects for immediate baseline calibration.
"""

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SubtitleBlock:
    index: int
    start_sec: float
    end_sec: float
    text: str


@dataclass
class DialogueScene:
    scene_id: str
    era: str
    source_type: str  # "youtube_movie" | "literature" | "short_film"
    genre_hint: str
    director_writer_style: str
    dialect_hint: str
    emotion_hint: str
    turns: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


NOISE_PATTERNS = [
    re.compile(r"\[(?:music|applause|laughter|bgm|silence|noise|singing).*?\]", re.IGNORECASE),
    re.compile(r"\((?:music|bgm|applause|sighs|laughs).*?\)", re.IGNORECASE),
    re.compile(r"[♪♫♬]+"),
    re.compile(r"<[^>]+>"),
    re.compile(r"\b(?:la\s+){3,}\b", re.IGNORECASE),
]

USER_HANDLE_PATTERN = re.compile(r"(?:@[\w_]+|u/[\w_-]+)")


def clean_utterance(text: str) -> str:
    """Strips BGM tags, HTML tags, music symbols, and user handles."""
    cleaned = text.strip()
    for pat in NOISE_PATTERNS:
        cleaned = pat.sub("", cleaned)
    cleaned = USER_HANDLE_PATTERN.sub("[USER]", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_timestamp_to_seconds(ts: str) -> float:
    """Parses SRT/VTT timestamp 'HH:MM:SS,mmm' or 'MM:SS.mmm' to seconds."""
    ts = ts.strip().split()[0].replace(",", ".")
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
    except ValueError:
        return 0.0
    return 0.0


def parse_srt_content(srt_text: str) -> List[SubtitleBlock]:
    """
    Parses raw .srt or .vtt file content into structured SubtitleBlock items.
    Includes rolling YouTube auto-caption deduplication and overlap clamping.
    """
    blocks: List[SubtitleBlock] = []
    raw_chunks = re.split(r"\n\s*\n", srt_text.strip())
    prev_text = ""

    for idx, chunk in enumerate(raw_chunks, start=1):
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if not lines or lines[0].startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        ts_line_idx = -1
        for i, l in enumerate(lines):
            if "-->" in l:
                ts_line_idx = i
                break
        if ts_line_idx == -1:
            continue

        ts_parts = lines[ts_line_idx].split("-->")
        start_s = parse_timestamp_to_seconds(ts_parts[0])
        end_s = parse_timestamp_to_seconds(ts_parts[1])
        text_body = clean_utterance(" ".join(lines[ts_line_idx + 1 :]))
        if not text_body or len(text_body) < 2:
            continue

        # Deduplicate rolling 2-line YouTube auto-captions
        if text_body == prev_text:
            continue
        if prev_text and text_body.startswith(prev_text) and len(text_body) > len(prev_text):
            text_body = text_body[len(prev_text) :].strip()
        if not text_body or len(text_body) < 2:
            continue

        prev_text = text_body
        blocks.append(SubtitleBlock(index=len(blocks) + 1, start_sec=start_s, end_sec=end_s, text=text_body))

    # Clamp overlapping YouTube auto-caption end_sec so real inter-cue gaps are preserved
    for i in range(len(blocks) - 1):
        if blocks[i].end_sec > blocks[i + 1].start_sec:
            blocks[i].end_sec = blocks[i + 1].start_sec

    return blocks


def slice_blocks_into_scenes(
    blocks: List[SubtitleBlock],
    era: str,
    genre_hint: str,
    director_writer_style: str,
    dialect_hint: str,
    emotion_hint: str,
    scene_prefix: str = "MOV",
    same_turn_gap_sec: float = 1.2,
    new_scene_gap_sec: float = 4.5,
    max_turns_per_scene: int = 4,
) -> List[DialogueScene]:
    """
    Groups subtitle blocks into 2-to-4 turn conversational exchanges based on timing gaps.
    Enforces copyright-safe max_turns_per_scene <= 4.
    """
    if not blocks:
        return []

    scenes: List[DialogueScene] = []
    current_turns: List[str] = []
    current_turn_buffer: List[str] = [blocks[0].text]
    prev_end = blocks[0].end_sec
    scene_counter = 1

    for b in blocks[1:]:
        gap = b.start_sec - prev_end
        if gap <= same_turn_gap_sec:
            current_turn_buffer.append(b.text)
        elif gap <= new_scene_gap_sec and len(current_turns) < max_turns_per_scene - 1:
            current_turns.append(" ".join(current_turn_buffer))
            current_turn_buffer = [b.text]
        else:
            current_turns.append(" ".join(current_turn_buffer))
            if len(current_turns) >= 2:
                scenes.append(
                    DialogueScene(
                        scene_id=f"{scene_prefix}_{era}_{scene_counter:04d}",
                        era=era,
                        source_type="youtube_movie",
                        genre_hint=genre_hint,
                        director_writer_style=director_writer_style,
                        dialect_hint=dialect_hint,
                        emotion_hint=emotion_hint,
                        turns=current_turns[:max_turns_per_scene],
                    )
                )
                scene_counter += 1
            current_turns = []
            current_turn_buffer = [b.text]
        prev_end = b.end_sec

    if current_turn_buffer:
        current_turns.append(" ".join(current_turn_buffer))
    if len(current_turns) >= 2:
        scenes.append(
            DialogueScene(
                scene_id=f"{scene_prefix}_{era}_{scene_counter:04d}",
                era=era,
                source_type="youtube_movie",
                genre_hint=genre_hint,
                director_writer_style=director_writer_style,
                dialect_hint=dialect_hint,
                emotion_hint=emotion_hint,
                turns=current_turns[:max_turns_per_scene],
            )
        )
    return scenes


def harvest_youtube_subtitles(
    youtube_url: str,
    output_dir: str | Path,
    langs: str = "te,en",
) -> List[Path]:
    """
    Downloads official/manual subtitles from a YouTube video using yt-dlp (Colab CPU friendly).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "--write-sub",
        "--write-auto-sub",
        "--sub-lang",
        langs,
        "--sub-format",
        "srt",
        "--skip-download",
        "-o",
        str(out_dir / "%(id)s.%(ext)s"),
        youtube_url,
    ]
    subprocess.run(cmd, check=False, capture_output=True, text=True)
    return sorted(out_dir.glob("*.srt"))


def build_curated_multi_era_seed_corpus() -> List[DialogueScene]:
    """
    Returns an authentic, multi-era Telugu Cinema & Literature Seed Corpus (1980s -> 2020s)
    covering 6 iconic Writer/Director archetypes, 6 emotions, and 5 dialects.
    Used to calibrate historical English % (CMI) baselines and feed the Tenglish-Evo Multiplier.
    """
    raw_seed = [
        # --- ERA 0: CLASSIC LITERATURE & FOLK PROSE (0% English Baseline) ---
        {
            "scene_id": "SEED_LIT_0001",
            "era": "1980s_LITERARY",
            "source_type": "literature",
            "genre_hint": "FAMILY_DRAMA",
            "director_writer_style": "CLASSIC_LITERARY",
            "dialect_hint": "STANDARD_TELUGU",
            "emotion_hint": "PHILOSOPHICAL",
            "turns": [
                "మనిషికి మాటే ప్రాణం అబ్బాయ్, ఆ మాట తప్పితే ఇక విలువ ఏముంటుంది?",
                "కానీ పరిస్థితులు మారినప్పుడు మనుషులు కూడా మారుతున్నారు కదా తాతయ్య.",
                "పరిస్థితులకు తలొగ్గేవాడు సామాన్యుడు, కష్టాల్లో కూడా నిజాయితీగా నిలబడేవాడే గొప్పవాడు.",
            ],
        },
        {
            "scene_id": "SEED_LIT_0002",
            "era": "1980s_LITERARY",
            "source_type": "literature",
            "genre_hint": "SATIRE_COMEDY",
            "director_writer_style": "JANDHYALA_WORDPLAY",
            "dialect_hint": "COASTAL_ANDHRA",
            "emotion_hint": "COMIC_TIMING",
            "turns": [
                "ఏమండోయ్ శాస్త్రి గారూ, తమరు పొద్దున్నే ఇలా హడావిడిగా ఎక్కడికి బయలుదేరారు?",
                "ఏం చెప్పమంటారు సుబ్బారావు గారూ, మా ఆవిడ పుట్టింటికి వెళ్తానని భీష్మించుకొని కూర్చుంది!",
                "అయ్యో, అది సంతోషించాల్సిన విషయమే కదండీ, ఇంతకీ రైలు ఖర్చులకు డబ్బులు ఇచ్చారా లేదా?",
            ],
        },
        # --- ERA 1: 1980s-1990s CLASSIC CINEMA (Low English Loanwords ~2-5%) ---
        {
            "scene_id": "SEED_1990_0001",
            "era": "1990s",
            "source_type": "youtube_movie",
            "genre_hint": "SATIRE_COMEDY",
            "director_writer_style": "JANDHYALA_WORDPLAY",
            "dialect_hint": "COASTAL_ANDHRA",
            "emotion_hint": "COMIC_TIMING",
            "turns": [
                "ఏరా అబ్బాయ్, కాలేజీకి వెళ్ళకుండా ఇక్కడ బస్ స్టాప్ దగ్గర ఏం చేస్తున్నావు?",
                "అది కాదండీ నాన్నగారూ, మా లెక్చరర్ గారు ఈరోజు సెలవు పెట్టారు.",
                "ఆహా! ఆయన సెలవు పెడితే నువ్వు పుస్తకాలు మూసేసి ఇలా రోడ్ల మీద తిరుగుతావా?",
            ],
        },
        {
            "scene_id": "SEED_1990_0002",
            "era": "1990s",
            "source_type": "youtube_movie",
            "genre_hint": "MASS_ACTION",
            "director_writer_style": "RAYALASEEMA_MASS",
            "dialect_hint": "RAYALASEEMA",
            "emotion_hint": "ANGRY_WARNING",
            "turns": [
                "ఏందిరా, మా ఊరి పొలిమేరల్లోకి వచ్చి మాకే ఎదురు తిరుగుతావా?",
                "న్యాయం ఎక్కడుంటే అక్కడికి రావడానికి నాకు ఎవ్వడి పర్మిషన్ అక్కర్లేదు.",
                "ఒక్కసారి కత్తి పడితే వెనక్కి తగ్గే రక్తం కాదురా ఇది, జాగ్రత్త!",
            ],
        },
        {
            "scene_id": "SEED_1990_0003",
            "era": "1990s",
            "source_type": "youtube_movie",
            "genre_hint": "FAMILY_DRAMA",
            "director_writer_style": "CLASSIC_DRAMA",
            "dialect_hint": "STANDARD_TELUGU",
            "emotion_hint": "HEARTBREAK",
            "turns": [
                "ఇన్నాళ్లూ మన మధ్య ఉన్న స్నేహం అంతా అబద్ధమేనా? నాకు ఒక్క మాట కూడా చెప్పాలనిపించలేదా?",
                "చెప్పాలని చాలాసార్లు ప్రయత్నించాను రా, కానీ నీ ముఖం చూసి నిజం చెప్పే ధైర్యం రాలేదు.",
                "నిజం దాచిపెట్టి నన్ను మోసం చేయడం కంటే, ఆ రోజే నన్ను వదిలేసి వెళ్ళిపోవాల్సింది.",
            ],
        },
        # --- ERA 2: 2000s URBAN & TRIVIKRAM / PURI ERA (~12-22% English Mix) ---
        {
            "scene_id": "SEED_2000_0001",
            "era": "2000s",
            "source_type": "youtube_movie",
            "genre_hint": "ROMCOM",
            "director_writer_style": "TRIVIKRAM_ANTITHESIS",
            "dialect_hint": "STANDARD_TELUGU",
            "emotion_hint": "PHILOSOPHICAL",
            "turns": [
                "నిజం చెప్పాలంటే నాకు తన మీద కోపం లేదు రా, కానీ తనని మర్చిపోవడం మాత్రం నా వల్ల కావట్లేదు.",
                "ప్రేమించేటప్పుడు ఉన్న ధైర్యం, ఆ ప్రేమని నిలబెట్టుకునేటప్పుడు ఎందుకు ఉండదు రా మీకు?",
                "యుద్ధంలో గెలవడం అంటే శత్రువుని చంపడం కాదు... మనల్ని మనం గెలవడం. ప్రేమలో ఓడిపోయినా పర్లేదు రా.",
            ],
        },
        {
            "scene_id": "SEED_2000_0002",
            "era": "2000s",
            "source_type": "youtube_movie",
            "genre_hint": "MASS_ACTION",
            "director_writer_style": "PURI_CYNICAL_PUNCH",
            "dialect_hint": "URBAN_HYDERABAD",
            "emotion_hint": "ELEVATION_PUNCH",
            "turns": [
                "ఏంటి భయ్యా, పోలీస్ స్టేషన్ లో కూర్చొని కూడా ఇంత కూల్ గా కాఫీ తాగుతున్నావ్?",
                "నాకు భయం అనే ఫీలింగ్ చిన్నప్పుడే ఆపరేషన్ చేసి తీసేశారు రా.",
                "నేను ఒక్కసారి కమిట్ అయితే నా మాట నేనే వినను, ఇక నీ వార్నింగ్ ఎవడు వింటాడు?",
            ],
        },
        {
            "scene_id": "SEED_2000_0003",
            "era": "2000s",
            "source_type": "youtube_movie",
            "genre_hint": "FAMILY_DRAMA",
            "director_writer_style": "BOMMARILLU_EMOTIONAL",
            "dialect_hint": "URBAN_HYDERABAD",
            "emotion_hint": "HEARTBREAK",
            "turns": [
                "నాన్న, నా లైఫ్ లో ఏ డెసిషన్ అయినా నేనే తీసుకోవాలని ఎప్పుడైనా అడిగానా?",
                "నీ మంచి కోసమే కదరా ఇదంతా చేస్తున్నాను, నీ కెరీర్ సెటిల్ అవ్వాలని నా తాపత్రయం.",
                "మొత్తం మీరే డిసైడ్ చేసేస్తే ఇక నేను బతకడం ఎందుకు నాన్న? నా ఫీలింగ్స్ కి అసలు వాల్యూ లేదా?",
            ],
        },
        # --- ERA 3: 2010s MULTIPLEX & LOGIC/NEW-WAVE ERA (~22-32% English Mix) ---
        {
            "scene_id": "SEED_2010_0001",
            "era": "2010s",
            "source_type": "youtube_movie",
            "genre_hint": "THRILLER",
            "director_writer_style": "SUKUMAR_LOGIC",
            "dialect_hint": "URBAN_HYDERABAD",
            "emotion_hint": "ELEVATION_PUNCH",
            "turns": [
                "వాడు వేసిన ప్రతి ప్లాన్ వెనుక ఒక క్యాలిక్యులేషన్ ఉంది సార్, మనం నార్మల్ గా ఆలోచిస్తే దొరకడు.",
                "అంటే ఇప్పుడు వాడు మన నెక్స్ట్ మూవ్ ని కూడా ముందే గెస్ చేశాడంటావా?",
                "ఎగ్జాక్ట్‌లీ! వాడు ఎమోషన్ తో కాదు, ప్యూర్ లాజిక్ తో గేమ్ ఆడుతున్నాడు.",
            ],
        },
        {
            "scene_id": "SEED_2010_0002",
            "era": "2010s",
            "source_type": "youtube_movie",
            "genre_hint": "ROMCOM",
            "director_writer_style": "THARUN_TELANGANA_BANTER",
            "dialect_hint": "TELANGANA",
            "emotion_hint": "SARCASTIC_ROAST",
            "turns": [
                "ఏంది రా బై, జాబ్ వదిలేసి బిజినెస్ స్టార్ట్ చేస్తా అంటున్నవ్, అసలు ప్లాన్ ఏమన్నా ఉందా?",
                "ప్లాన్ గీన్ ఏం లేదు రా, ఫస్ట్ దిగినంకనే స్విమ్మింగ్ వస్తది అని మా దోస్తుగాడు చెప్పిండు.",
                "నీ దోస్తుగాని మాటలు విని దిగితే స్విమ్మింగ్ కాదు రా, డైరెక్ట్ మునిగిపోతవ్! లైట్ తీసుకో.",
            ],
        },
        {
            "scene_id": "SEED_2010_0003",
            "era": "2010s",
            "source_type": "youtube_movie",
            "genre_hint": "ROMCOM",
            "director_writer_style": "TRIVIKRAM_ANTITHESIS",
            "dialect_hint": "STANDARD_TELUGU",
            "emotion_hint": "SARCASTIC_ROAST",
            "turns": [
                "ఏంట్రా, ఆ అమ్మాయి నిన్ను రిజెక్ట్ చేసిందని పొద్దున్నుంచి ఆఫీస్ కి రాకుండా రూమ్ లోనే కూర్చున్నావా?",
                "తనతో కనెక్ట్ అయ్యాను రా, ఇప్పుడు సడన్ గా కుదరదు అంటే ఎలా తీసుకోవాలో అర్థం కావట్లేదు.",
                "అవసరం ఉన్నప్పుడు గుర్తొచ్చేది ఆప్షన్ రా, ఊపిరి ఉన్నంతవరకు గుర్తుండేది ఎమోషన్. తను నిన్ను ఆప్షన్ లా చూసింది, వదిలేయ్!",
            ],
        },
        # --- ERA 4: 2020s GEN-Z / DJ TILLU / VIVEK ATHREYA ERA (~35-50% English Mix) ---
        {
            "scene_id": "SEED_2020_0001",
            "era": "2020s",
            "source_type": "youtube_movie",
            "genre_hint": "SATIRE_COMEDY",
            "director_writer_style": "THARUN_TELANGANA_BANTER",
            "dialect_hint": "TELANGANA",
            "emotion_hint": "COMIC_TIMING",
            "turns": [
                "ఏంది బ్రో ఇది, నిన్న రాత్రి పార్టీకి పిలిచి ఇప్పుడు నాకు సంబంధం లేదు అంటావ్ ఏంది?",
                "అరే భయ్యా, అక్కడ సిట్యుయేషన్ మొత్తం కంట్రోల్ తప్పింది రా, నాకే ఫుల్ ట్రామా అయిపోయింది!",
                "నువ్వు ఇట్లా ప్రతిసారి సీన్ క్రియేట్ చేసి నా మీద తోసేస్తే నా లైఫ్ మొత్తం డ్యామేజ్ అయితది అంతేగా!",
            ],
        },
        {
            "scene_id": "SEED_2020_0002",
            "era": "2020s",
            "source_type": "youtube_movie",
            "genre_hint": "ROMCOM",
            "director_writer_style": "VIVEK_URBAN_WIT",
            "dialect_hint": "URBAN_HYDERABAD",
            "emotion_hint": "SARCASTIC_ROAST",
            "turns": [
                "బ్రో, నిన్నటి నుంచి తనకి పది మెసేజ్‌లు పెట్టాను, సీన్ చేసింది కానీ అసలు రిప్లై ఇవ్వట్లేదు ఏంటి?",
                "దాన్నే ఇంగ్లీష్ లో ఘోస్ట్ చేయడం అంటారు రా, ఇంకా ఆ డెలులు లో నుంచి బయటికి రా!",
                "అంటే ఇప్పుడు మనకి అక్కడ అసలు వైబ్ అవ్వట్లేదు అని డైరెక్ట్ గా హింట్ ఇచ్చేసిందా?",
            ],
        },
        {
            "scene_id": "SEED_2020_0003",
            "era": "2020s",
            "source_type": "youtube_movie",
            "genre_hint": "SATIRE_COMEDY",
            "director_writer_style": "GENZ_MEME_BANTER",
            "dialect_hint": "URBAN_HYDERABAD",
            "emotion_hint": "SARCASTIC_ROAST",
            "turns": [
                "ఏంటి బ్రో, వీకెండ్ ట్రిప్ కి వస్తా అని ఫుల్ బిల్డప్ ఇచ్చి లాస్ట్ మినిట్ లో హ్యాండ్ ఇచ్చావ్?",
                "మా మేనేజర్ గాడు ఫ్రైడే ఈవినింగ్ క్లయింట్ కాల్ పెట్టి మైండ్ బ్లాక్ చేసిండు రా, నేనేం చేయను చెప్పు?",
                "ప్రతిసారి ఇదే సాకు చెప్తున్నావ్ రా, ఇంక నీతో ట్రిప్ ప్లాన్ చేయడం కంటే ఇంట్లో పడుకోవడం బెటర్ అంతేగా!",
            ],
        },
    ]

    return [DialogueScene(**item) for item in raw_seed]


def save_scenes_jsonl(scenes: List[DialogueScene], out_path: str | Path) -> Path:
    """Writes DialogueScene list to JSONL format."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for s in scenes:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    return p
