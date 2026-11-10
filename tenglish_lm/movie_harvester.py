"""
Generic Telugu Movie Subtitle Parser, BGM/Noise Cleaner & Timestamped Scene Slicer (`tenglish_lm/movie_harvester.py`).

Responsibilities:
1. Parse raw `.srt` or `.vtt` subtitle tracks (manual captions or YouTube auto-generated Telugu captions).
2. Clean non-dialogue noise:
   - Bracketed audio cues in Telugu & English (`[సంగీతం]`, `[బెల్]`, `[చప్పట్లు]`, `[నవ్వులు]`, `[Music]`, `[Applause]`)
   - Statutory smoking/drinking disclaimers (`పొగ త్రాగుట మధ్యం సేవించుట ఆరోగ్యానికి హానికరం...`)
   - Pure BGM humming (`తారా తరారారారా`, `లలలలా`)
3. Deduplicate rolling YouTube auto-caption repetitions and clamp overlapping cue timestamps.
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class SubtitleBlock:
    index: int
    start_sec: float
    end_sec: float
    text: str


# Bracketed or parenthetical sound/music cues in Telugu and English
BRACKET_NOISE_PATTERNS = [
    re.compile(r"\[[^\]]*\]"),  # Strips [సంగీతం], [బెల్], [చప్పట్లు], [Music], [Song], etc.
    re.compile(
        r"\((?:music|bgm|applause|sighs|laughs|bell|singing|song|lyric|పాట|సంగీతం|బెల్|చప్పట్లు|నవ్వులు).*?\)",
        re.IGNORECASE,
    ),
    re.compile(r"[♪♫♬♩]+"),
    re.compile(r"<[^>]+>"),
]

# Statutory disclaimers, traffic safety PSAs, & opening prayer slokas
DISCLAIMER_SUBSTRINGS = [
    "పొగ త్రాగుట",
    "పొగ తాగుట",
    "మధ్యం సేవించుట",
    "మద్యం సేవించుట",
    "ఆరోగ్యానికి హానికరం",
    "స్మోకింగ్ అండ్ డ్రింకింగ్",
    "ఇంజరస్ టు హెల్త్",
    "injurious to health",
    "smoking and drinking",
    "నమస్తస్మై నమో నమః",
    "సప్తశైల నివాసినే",
    "డ్రైవింగ్ చేసేటప్పుడు",
    "సీట్‌బెల్ట్",
    "హెల్మెట్ ధరించండి",
    "రోడ్డు భద్రత",
    "పాదచారుల",
    "పైరసీ నేరం",
    "యాంటీ పైరసీ",
    "రక్తదానం",
    "నేత్రదానం",
]

# Production house cards and title entry markers
TITLE_CARD_PATTERNS = [
    re.compile(
        r"^(?:సమర్పణ|సమర్పించు|దర్శకత్వం|దర్శకుడు|నిర్మాత|నిర్మాణం|కథ\s*[\-\–]\s*మాటలు|మాటలు|రచన|సంగీతం|ఛాయాగ్రహణం|ఎడిటింగ్|బ్యానర్|ప్రొడక్షన్స్|పిక్చర్స్|చిత్రం|నటీనటులు|టైటిల్స్)\s*[:\-\–]",
        re.UNICODE,
    ),
    re.compile(
        r"^(?:directed by|produced by|music by|story & screenplay|cinematography|presents|banner)\s*[:\-\–]",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:సురేష్ ప్రొడక్షన్స్|శ్రీ వెంకటేశ్వర క్రియేషన్స్|మైత్రి మూవీ మేకర్స్|హారిక అండ్ హాసిని|గీతా ఆర్ట్స్|వైజయంతి మూవీస్)\s*(?:సమర్పించు|సమర్పణ)",
        re.UNICODE,
    ),
]

# Pure BGM humming syllables and classical dance rhythmic jatis (solkattu)
BGM_HUMMING_PATTERN = re.compile(
    r"^(?:[\s,.!?…]*?(?:తారా|తరారా\w*|తార\w*|లల\w*|లాలా\w*|హోహో|హేహే|ఆఆ\w*|ఓఓ\w*|నానా\w*|రారా\w*|హరే|తంగు|ధికు|ధింగిన|ధనకు|కిణకు|తకిట|జం|తై))+[\s,.!?…]*$",
    re.UNICODE,
)

# Common song vocable tokens
SONG_VOCABLES = ["లాల", "లాలా", "తానాన", "తారరార", "తరార", "హోహో", "హేహే", "జిలేలే", "డుండుం", "డిండిం", "రుక్కు", "తారానా"]


def is_song_lyric_line(text: str) -> bool:
    """Returns True if a subtitle cue is part of a song lyric rather than conversational dialogue."""
    t = text.strip()
    if not t:
        return True
    if any(ch in t for ch in "♪♫♬♩"):
        return True
    words = t.split()
    if len(words) <= 12:
        voc_hits = sum(1 for w in words if any(v in w for v in SONG_VOCABLES))
        if voc_hits >= 2 or (len(words) <= 4 and voc_hits >= 1):
            return True
    # Rhyming poetic meters common in Telugu movie songs (e.g. చేసేనులే, వచ్చేనులే, చేరేనులే)
    rhymes = re.findall(r"(?:^|\s)[\u0C00-\u0C7F]+(?:ేనులే|ెనుగా|ిందిలే|ావులే|ెనేలే)(?:\s|$)", t)
    if len(rhymes) >= 2 and "?" not in t and "!" not in t:
        return True
    return False


def is_non_dialogue_disclaimer_or_bgm(text: str, start_sec: float = 0.0) -> bool:
    """Returns True if the line is a statutory disclaimer, pre-movie PSA ad, title card, song, or pure BGM humming."""
    lowered = text.lower().strip()
    if not lowered:
        return True
    for sub in DISCLAIMER_SUBSTRINGS:
        if sub in lowered:
            return True
    if BGM_HUMMING_PATTERN.match(lowered):
        return True
    if start_sec <= 360.0:  # In first 6 minutes of video
        for pat in TITLE_CARD_PATTERNS:
            if pat.search(lowered):
                return True
    if is_song_lyric_line(text):
        return True
    return False


def clean_utterance(text: str, start_sec: float = 0.0) -> str:
    """
    Strips Telugu/English bracketed noise tags ([సంగీతం], [బెల్]), HTML tags,
    music symbols, BGM humming, song lyrics, title cards, and statutory disclaimers.
    """
    cleaned = text.strip()
    for pat in BRACKET_NOISE_PATTERNS:
        cleaned = pat.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if is_non_dialogue_disclaimer_or_bgm(cleaned, start_sec=start_sec):
        return ""
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


def format_seconds_to_srt_timestamp(seconds: float) -> str:
    """Formats seconds into standard SRT timestamp 'HH:MM:SS,mmm'."""
    s = max(0.0, float(seconds))
    total_sec = int(s)
    millis = int(round((s - total_sec) * 1000))
    if millis >= 1000:
        total_sec += 1
        millis -= 1000
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def parse_srt_content(srt_text: str) -> List[SubtitleBlock]:
    """
    Parses raw .srt or .vtt file content into structured, noise-cleaned SubtitleBlock items.
    Includes rolling YouTube auto-caption deduplication and overlap clamping.
    """
    blocks: List[SubtitleBlock] = []
    raw_chunks = re.split(r"\n\s*\n", srt_text.strip())
    prev_text = ""

    for chunk in raw_chunks:
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
        text_body = clean_utterance(" ".join(lines[ts_line_idx + 1 :]), start_sec=start_s)
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


def blocks_to_clean_srt(blocks: List[SubtitleBlock]) -> str:
    """Serializes cleaned SubtitleBlock items back into a standard, noise-free `.srt` string."""
    lines: List[str] = []
    for idx, b in enumerate(blocks, start=1):
        st = format_seconds_to_srt_timestamp(b.start_sec)
        et = format_seconds_to_srt_timestamp(max(b.start_sec + 0.5, b.end_sec))
        lines.append(f"{idx}\n{st} --> {et}\n{b.text}\n")
    return "\n".join(lines)
