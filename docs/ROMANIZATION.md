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

## The variation table (main entries, calibrated on Dakshina, Stage 1c)

Probabilities were measured from ~58k human spelling entries for 25k Telugu words in Google's Dakshina lexicon (train split).
The original hand-set guesses are kept in the code and can be restored with `TLM_HANDSET=1`.

| Sound | Spellings (canonical first, measured share) | Example |
|---|---|---|
| long ā (ా, ఆ) | `aa` 66% · `a` 34% | `raasaanu` / `rasanu` |
| long ī (ీ, ఈ) | `ee` 43% · `i` 57% | `neeku` / `niku` |
| long ū (ూ, ఊ) | `oo` 34% · `uu` 37% · `u` 29% | `choodu` / `chuudu` / `chudu` |
| ే (long ē sign) | `e` 83% · `ee` 17% | `ledu` / `leedu` |
| word-initial ఏ/ఎ | `e` ~78% · `ye` 17% · `a` 5% | `enti` / `yenti` / `anti` |
| త | `th` 45% · `t` 55% | `intha` / `inta` |
| ద | `d` 77% · `dh` 23% | `adi` / `adhi` |
| ధ, భ | aspirate kept ~95% | `dhairyam`, `bhayam` |
| వ | `v` 93% · `w` 7% | `vellu` / `wellu` |
| శ, ష | `sh` 67% · `s` 33% | `shivudu` / `sivudu` |
| ం anusvara | `n` (or `m` before p/b/m), with the other letter used in the rest of cases* | `anthe` / `amthe` |
| doubled consonant | kept 85%* · single 15% | `nuvvu` / `nuvu` |
| చ్చ | `chch` or `cch` | `vachchava` / `vacchava` |
| word-final ā | shortened 85%* | `entra`, `ela`, `kada` |

\* Not yet calibrated. Dakshina measures anusvara as `m` 25% of the time and doubled consonants kept 95%; these rules still use the hand-set values.

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

- Sound-level weights are calibrated on Dakshina, but Dakshina's words come from Wikipedia (formal vocabulary) and its annotators
  typed carefully. The shorthand rules, and several word-level rules, are still hand-set and unvalidated against real chat text.
- Some real habits are missing: numbers for sounds, vowel-less abbreviations (`nvu`, `chpu`), English slang inside words,
  and inconsistent capitalization.
- Telugu–English code-mixing is limited to English words already present in the subtitles.
