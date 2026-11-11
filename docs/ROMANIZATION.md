# How the synthetic romanized text is generated

Real romanized Telugu (the way people type it in chats and comments) is scarce as clean, aligned data. This project
**generates** it from the native-script movie corpus. This document explains exactly how, and how the approach was chosen.

## Step by step (`tlm/romanizer.py`)

1. **Split each word into aksharas:** consonant (+ optional conjunct consonants) + vowel sign, independent vowels,
   anusvara (ం), visarga (ః).
2. **Look up a spelling for each sound** in a table of alternatives with weights. The first option is the *canonical* spelling.
3. **Sample one alternative per sound**, independently for every word. Each sentence also draws a **typing habit** (`style`)
   that shifts probability toward shorter, lazier spellings.
4. **Apply word-level shorthand** with small probabilities, more often for lazy typists.
5. **Keep English loanwords in English** (`లేట్` → `late`, `ప్లీజ్` → `please`/`plz`/`pls`).

Calling the romanizer several times on the same sentence gives several different, plausible spellings. Each turn in the
derived corpus gets 3 noisy versions plus 1 canonical version. Seeds are fixed, so the output is reproducible.

## The variation table (main entries)

| Sound | Spellings (canonical first, approximate weights) | Example |
|---|---|---|
| long ā (ా, ఆ) | `aa` 50–55% · `a` 45–50% | `raasaanu` / `rasanu` |
| long ī (ీ, ఈ) | `ee` 55–60% · `i` 40–45% | `neeku` / `niku` |
| long ū (ూ, ఊ) | `oo` 50% · `u` 42% · `uu` 8% | `choodu` / `chudu` / `chuudu` |
| long ē (ే, ఏ) | `e` 70% · `ee` 30% (word-initial ఏ: `e` 50% · `ye` 35% · `a` 15%) | `enti` / `yenti` / `anti` |
| త, ద, ధ | `th`/`t`, `d`/`dh`, `dh`/`d` | `intha` / `inta` |
| aspirates ఖ ఘ ఠ ఢ భ ఫ | aspirated or plain (`bh`/`b`, `ph`/`f`) | `bhayam` / `bayam` |
| వ | `v` 65% · `w` 35% | `vellu` / `wellu` |
| శ, ష | `sh` 80% · `s` 20% | |
| ం anusvara | `n` (or `m` before p/b/m) 90% · the other 10% | `anthe` / `amthe` |
| doubled consonant | kept (same spelling both times) 85% · single 15% | `nuvvu` / `nuvu`, never `nuwvu` |
| చ్చ | `chch` or `cch` | `vachchava` / `vacchava` |
| word-final ā | shortened 85% | `entra`, `ela`, `kada` |

## Word-level shorthand (added after Stage 1b calibration)

| Pattern | Example | Probability (scaled by laziness) |
|---|---|---|
| verb ending `-anu` → `-a` | `chesanu` → `chesa` | ~4–16% |
| final `-u` dropped | `nuvvu` → `nuvv`, `nenu` → `nen` | ~2–10% |
| final `-i` → `-y` | `enti` → `enty` | ~3–12% |
| final `-e` → `-ey` | `anthe` → `anthey` | ~3–12% |

## How the approach was chosen (Stage 1b)

Three strategies were compared (`tlm/tiers.py`, `stages/s1b_generator_comparison.py`):

| Strategy | How it varies | Idea |
|---|---|---|
| **Tiered** (a design proposed by Gemini) | one fixed style **per sentence**: P1 standard / P2 minimal / P4 lazy | people have a consistent typing habit |
| **Per-word** (this project) | every **word** sampled independently, plus a per-sentence habit | people aren't consistent within a sentence |
| **Hybrid** | tier per sentence, 30% of words re-sampled per word | both |

See the README's Stage 1b section for the plots and numbers, and for why per-word was kept as the default.

## Limitations

- The weights are **hand-set**, informed by common spellings, and **not estimated from real data**. The next step is
  calibrating against human romanizations such as Google's Dakshina dataset, which includes Telugu.
- Some real habits are missing: numbers for sounds, vowel-less abbreviations (`nvu`, `chpu`), English slang inside words,
  and inconsistent capitalization.
- Telugu–English code-mixing is limited to English words already present in the subtitles.
