"""
Next-Concept Modeling (NCM) Coarse-to-Fine Scene Hierarchy (`extractor/ncm_concept_hierarchy.py`).
Inspired by: "Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation" (arXiv:2610.14162v1).

Adapts NCM's multiscale coarse-to-fine representation (r_1 -> r_2 -> r_3) to Telugu Movie Scenes:
  - Level 1 (r_1, Coarse Concept Trajectory, l_1 << L):
      Turn-by-turn pragmatic speech-act & emotional beat trajectory
      (e.g., `NARRATIVE_SETUP -> URGENT_COMMAND -> IMPERATIVE_DEMAND`).
  - Level 2 (r_2, Intermediate Semantic & Morphological Anchors, l_2 < L):
      Extracts the content-bearing skeleton that locks in before surface generation:
        1. Protected Emotional Core words (`కోపం`, `ప్రాణభయం`, `ప్రేమ`, `బాధ`, `ధైర్యం`, `నిజం`, `జీవితం`)
        2. Transliterated English loanwords naturally spoken in Telugu cinema (`అరెస్ట్`, `అర్జెంట్`, `ఇమీడియట్`, `ఫ్యామిలీ`, `సార్`)
        3. Agglutinative case-marked Telugu stems (`-లో`, `-కి`, `-తో`, `-ని`, `-గా`)
  - Level 3 (r_3, Fine Surface Resolution, l_3 = L):
      Full Telugu dialogue utterance sequence.
  - Multiscale Residual VQ Code Assignment (`compute_multiscale_rvq_codes`):
      Assigns deterministic 3-level discrete residual codes (`L1:xx|L2:yy|L3:zz`) to every scene
      for coarse-to-fine clustering, retrieval, and conditional generation.
"""

import hashlib
import re
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# 1. Level 1 (r_1): Coarse Speech-Act & Beat Trajectory
# ---------------------------------------------------------------------------

INTERROGATIVE_MARKERS = [
    "ఎందుకు", "ఏంటి", "ఏమిటి", "ఎలా", "ఎక్కడ", "ఎవరు", "ఎవరిని", "ఎవరికి", "ఎప్పుడు", "కారణం", "తెలుసుకుందామా", "కదా", "ఏంట్రా", "ఏంది",
]

IMPERATIVE_COMMAND_MARKERS = [
    "రండి", "పరిగెత్తండి", "చేయండి", "చెప్పండి", "చెప్పు", "ఆగు", "ఆపండి", "వెళ్ళు", "వెళ్ళండి", "మూసుకో", "చూడు", "చూడండి", "తీసుకో", "అరెస్ట్", "కొట్టు", "పిలువు",
]

CONFLICT_WARNING_MARKERS = [
    "చంపేస్తా", "నరికేస్తా", "జాగ్రత్త", "వార్నింగ్", "వదల", "ప్రాణభయం", "కోపం", "రక్తం", "కత్తి", "తప్పు", "దమ్ము",
]

EMOTIONAL_VULNERABILITY_MARKERS = [
    "బాధ", "ఏడుపు", "కన్నీళ్లు", "ప్రేమ", "గుండె", "ఒంటరి", "మర్చిపోలేను", "క్షమించు", "సారీ", "నమ్మకం",
]

COMIC_BANTER_MARKERS = [
    "నవ్వు", "సరదా", "బిల్డప్", "కామెడీ", "జోక్", "అల్లుడు", "మామ", "పెళ్లి", "కాఫీ", "ఐస్", "సూపర్",
]


def classify_turn_coarse_concept(turn_text: str) -> str:
    """
    Maps a single speaker turn to its Level-1 Coarse Concept Beat (r_1^i).
    """
    t = turn_text.strip()
    has_qmark = "?" in t

    if any(m in t for m in CONFLICT_WARNING_MARKERS):
        return "CONFLICT_THREAT"
    if any(m in t for m in IMPERATIVE_COMMAND_MARKERS):
        return "URGENT_COMMAND"
    if has_qmark or any(m in t for m in INTERROGATIVE_MARKERS):
        return "QUESTION_PROBE"
    if any(m in t for m in EMOTIONAL_VULNERABILITY_MARKERS):
        return "EMOTIONAL_PLEA"
    if any(m in t for m in COMIC_BANTER_MARKERS):
        return "WITTY_BANTER"
    if len(t.split()) >= 12:
        return "NARRATIVE_MONOLOGUE"
    return "DIALOGUE_RESPONSE"


def extract_level1_coarse_trajectory(turns: List[str]) -> str:
    """
    Constructs the Level-1 Coarse Concept sequence r_1 = [r_1^1, ..., r_1^m] across speaker turns.
    """
    if not turns:
        return "EMPTY_SCENE"
    beats = [classify_turn_coarse_concept(t) for t in turns]
    return " -> ".join(beats)


# ---------------------------------------------------------------------------
# 2. Level 2 (r_2): Intermediate Semantic & Morphological Anchors
# ---------------------------------------------------------------------------

PROTECTED_EMOTIONAL_STEMS = [
    "కోపం", "ప్రాణభయం", "ప్రాణం", "ప్రేమ", "బాధ", "ధైర్యం", "నిజం", "జీవితం", "యుద్ధం",
    "భయం", "నమ్మకం", "గౌరవం", "విలువ", "బాధ్యత", "కన్నీళ్లు", "గుండె", "మనిషి", "మాట",
]

TELUGU_SCRIPT_ENGLISH_LOANS = [
    "బిజీ", "ఫ్యామిలీ", "అరెస్ట్", "సార్", "అర్జెంట్", "ఇమీడియట్", "కానిస్టేబుల్", "పోలీస్",
    "లైఫ్", "కెరీర్", "డిసైడ్", "డెసిషన్", "సెటిల్", "కాఫీ", "ఆఫీస్", "కాలేజ్", "ఫ్రెండ్",
    "లవ్", "మ్యారేజ్", "ప్రాబ్లం", "టైమ్", "фоన్", "ఫోన్", "కార్", "బైక్", "డాక్టర్",
    "హాస్పిటల్", "కేస్", "కోర్ట్", "స్టేషన్", "బోస్", "బాస్", "మేడమ్", "బిల్డప్", "పార్టీ",
]

AGGLUTINATIVE_SUFFIXES = ("లో", "కి", "తో", "ని", "గా", "లోని", "లోపల", "కోసం", "వల్ల", "నుంచి")


def extract_level2_semantic_anchors(turns: List[str], max_anchors_per_type: int = 5) -> Dict[str, List[str]]:
    """
    Extracts Level-2 (r_2) intermediate concept anchors from the Telugu dialogue turns:
      - `emotional_core`: visceral Telugu emotional words that anchor the scene's affect.
      - `english_loans`: natural cinema English words spoken inside Telugu sentences.
      - `case_inflected_stems`: agglutinative Telugu words carrying `-లో`, `-కి`, `-తో`, `-గా` suffixes.
    """
    joined = " ".join(turns)
    tokens = re.findall(r"[\u0C00-\u0C7F]+", joined)

    emotional_found: List[str] = []
    loans_found: List[str] = []
    case_stems_found: List[str] = []

    for tok in tokens:
        if len(tok) < 2:
            continue
        for emo in PROTECTED_EMOTIONAL_STEMS:
            if tok.startswith(emo) and tok not in emotional_found:
                emotional_found.append(tok)
                break
        for loan in TELUGU_SCRIPT_ENGLISH_LOANS:
            if tok.startswith(loan) and tok not in loans_found:
                loans_found.append(tok)
                break
        if len(tok) >= 4 and tok.endswith(AGGLUTINATIVE_SUFFIXES):
            if tok not in case_stems_found and tok not in emotional_found and tok not in loans_found:
                case_stems_found.append(tok)

    return {
        "emotional_core": emotional_found[:max_anchors_per_type],
        "english_loans": loans_found[:max_anchors_per_type],
        "case_inflected_stems": case_stems_found[:max_anchors_per_type],
    }


def format_level2_anchors_compact(anchors: Dict[str, List[str]]) -> str:
    """Formats the Level-2 semantic anchors into a compact single-line string for CSV/TXT."""
    parts: List[str] = []
    if anchors["emotional_core"]:
        parts.append(f"CORE_TE:[{', '.join(anchors['emotional_core'])}]")
    if anchors["english_loans"]:
        parts.append(f"EN_LOAN:[{', '.join(anchors['english_loans'])}]")
    if anchors["case_inflected_stems"]:
        parts.append(f"MORPH:[{', '.join(anchors['case_inflected_stems'])}]")
    return " | ".join(parts) if parts else "GENERAL_DIALOGUE"


# ---------------------------------------------------------------------------
# 3. Multiscale Residual VQ Codebook Quantizer (r_1 -> r_2 -> r_3)
# ---------------------------------------------------------------------------

def compute_multiscale_rvq_codes(
    scene_emotion: str,
    scene_dialect: str,
    r1_trajectory: str,
    r2_anchors_str: str,
    turns: List[str],
    codebook_size: int = 64,
) -> str:
    """
    Computes a deterministic 3-level Multiscale Residual Code (`L1:c1|L2:c2|L3:c3`)
    following NCM's coarse-to-fine hierarchy:
      - L1 code (0..63): Quantizes coarse emotion + turn beat trajectory (r_1)
      - L2 code (0..63): Quantizes residual semantic & morphological anchor skeleton (r_2 | r_1)
      - L3 code (0..63): Quantizes fine surface realization & rhythm (r_3 | r_1, r_2)
    """
    h1 = int(hashlib.md5(f"{scene_emotion}::{r1_trajectory}".encode("utf-8")).hexdigest()[:8], 16) % codebook_size
    h2 = int(
        hashlib.md5(f"{h1}::{scene_dialect}::{r2_anchors_str}".encode("utf-8")).hexdigest()[:8], 16
    ) % codebook_size
    h3 = int(
        hashlib.md5(f"{h1}::{h2}::{' '.join(turns)}".encode("utf-8")).hexdigest()[:8], 16
    ) % codebook_size
    return f"L1:{h1:02d}|L2:{h2:02d}|L3:{h3:02d}"


def build_ncm_scene_representation(
    turns: List[str],
    scene_emotion: str,
    scene_dialect: str,
) -> Dict[str, Any]:
    """
    Computes the complete Next-Concept Modeling (NCM) 3-level coarse-to-fine representation for a scene.
    """
    r1_trajectory = extract_level1_coarse_trajectory(turns)
    r2_dict = extract_level2_semantic_anchors(turns)
    r2_compact = format_level2_anchors_compact(r2_dict)
    rvq_code = compute_multiscale_rvq_codes(
        scene_emotion=scene_emotion,
        scene_dialect=scene_dialect,
        r1_trajectory=r1_trajectory,
        r2_anchors_str=r2_compact,
        turns=turns,
    )
    return {
        "ncm_rvq_code": rvq_code,
        "ncm_r1_coarse_trajectory": r1_trajectory,
        "ncm_r2_semantic_anchors": r2_compact,
        "ncm_r2_anchor_details": r2_dict,
    }
