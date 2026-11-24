# Framework: turning bookish (Grandhika) Telugu in subtitles back into spoken Telugu

Status: **design only, not implemented.**

## The problem

Some subtitles in the corpus are not what the actors said. They are **machine translations** (often from a Hindi or English
subtitle track), so they come out in bookish, textbook Telugu (*grandhika* / *granthika*), not the spoken Telugu
(*vyavaharika*) people really use.

Example from the corpus:

| | Text |
|---|---|
| Subtitle (machine translated) | 40% మాత్రమే, **తండ్రి**. 20% మీ **కుమార్తె** సునందకు చెందినది. 25 ఏళ్లుగా ఆమె బోర్డు **సమావేశానికి హాజరు కాలేదు**. యాజమాన్యాన్ని క్లెయిమ్ చేయలేదు. అందుకే మా షేర్లను **కొనడం మొదలుపెట్టాడు**. అతనితో **మాట్లాడండి** |
| What the actor said | 40% మాత్రమే **నాన్న**. 20% **నీ కూతురు** సునంద ది… |

A person says నాన్న (dad), not తండ్రి (father). They say కూతురు (daughter), not కుమార్తె, and మీటింగ్, not సమావేశం.
A model trained on the bookish version learns to talk like a newspaper.

## What we change and what we never touch

| Change (bookish → spoken) | Never touch |
|---|---|
| Sanskrit-heavy or textbook words: భూగోళశాస్త్రం, కుమార్తె, తండ్రి, సమావేశం, హాజరు, క్షమించండి, ధన్యవాదాలు | **Slang and dialect**: ఏంది, గట్ల, చేసిండు, ఏందిరా, రా/రే, మామా, బావా |
| Formal verb endings: చేస్తున్నాను → చేస్తున్నా; వెళ్ళండి (as a plain order) → వెళ్ళు | Words people really use, even old ones (అమ్మ, ఇల్లు, పని) |
| Formal pronouns in a family scene: మీ కుమార్తె → నీ కూతురు | Names, numbers, places, and English already in the line |
| Over-written English ideas: యాజమాన్యం → ఓనర్‌షిప్ (only when people really say it) | Lines that are already spoken Telugu |

**Rule of thumb:** a word changes only if a normal speaker would almost never say it in conversation.

## The pipeline (5 steps)

```
subtitle scene
  → 1. Is this scene machine-translated?          (scene-level flag)
  → 2. Find bookish words in it                   (word-level flag)
  → 3. Replace from ONE fixed word table          (same rule for every movie and era)
  → 4. Spoken-ending rules (tlm/colloquial.py)
  → 5. Romanizer (tlm/romanizer.py)               → Tenglish copy
```

### 1. Detect machine-translated scenes
Bookish words cluster in machine-translated scenes. The scene-level signals are:
- many bookish words per line (step 2's count)
- formal pronouns and endings everywhere (మీరు/మీ + -ండి in casual family talk)
- full stops after every short clause, perfect grammar, and no fillers (అరే, అబ్బా, ఏంటి రా)
- the movie is a dub (many such scenes are already noted in the Stage 7 repair notes)

The output is a score from 0 to 1 per scene. Only scenes above a threshold get step 3. Clean, native scenes pass through
untouched, which protects the real dialogue.

### 2. Find bookish words (automatic, no manual labelling)
- **Frequency contrast:** a word is a candidate if it is common in machine-translated or formal text but rare in casual text.
  The formal sources are Dakshina and the flagged scenes. The casual sources are native movie scenes and the YouTube comments,
  used for counting only.
- **Letter clues:** Sanskrit clusters such as క్ష, జ్ఞ, ఋ, శాస్త్ర, -త్వం and -కరణ.
- **Protect list:** a fixed list of dialect and slang markers (taken from `tlm/colloquial.py` DIALECT plus the Stage 7
  repair notes) that are never flagged.

The output is a short candidate list, a few hundred words.

### 3. One fixed word table
`tlm/grandhika_table.json`, with one row per bookish word:

```json
{"కుమార్తె": {"spoken": "కూతురు", "english": null, "kind": "family"},
 "తండ్రి":   {"spoken": "నాన్న",  "english": "dad", "kind": "family", "only_when": "addressing"},
 "సమావేశం": {"spoken": null,     "english": "మీటింగ్", "kind": "loanword"},
 "భూగోళశాస్త్రం": {"spoken": null, "english": "జియోగ్రఫీ", "kind": "loanword"}}
```
- **The choice order:** use the English loanword if people really use it (మీటింగ్, జియోగ్రఫీ, ఆఫీస్). Otherwise use the
  everyday Telugu word (కూతురు, నాన్న). Otherwise keep the word.
- **Who fills it:** a model drafts the table, a *different* model checks every row, and rows where the two disagree are
  dropped. There is no manual labelling.
- **Why a fixed table:** every movie and every era gets exactly the same rule, so the data ends up in one consistent style.
- **Context rules** (`only_when`): for example, తండ్రి → నాన్న only when someone is speaking *to* the father. In
  "his father died", తండ్రి can stay or become నాన్న depending on register.

### 4. Spoken endings
These are the existing rules in `tlm/colloquial.py`, extended:
- -తున్నాను → -తున్నా
- -ండి → plain imperative inside family or friend scenes
- మీ → నీ when the scene is informal (the same flag as step 1)

### 5. Romanize
The existing calibrated romanizer then produces the Tenglish copies, for example `40% matrame nanna, 20% nee kuthuru…`.

## Hard cases to handle carefully
- **Formal on purpose:** a judge, priest, news reader or politician *should* sound formal. Step 1's scene flag plus the
  `kind` and `only_when` fields limit the damage, and scenes from those settings can be skipped.
- **Ambiguous words:** words with two meanings keep their original form unless the context rule matches.
- **Inflected words:** కుమార్తెకు, కుమార్తెను and so on. The replacement must work on the word stem and keep the suffix
  (కూతురికి, కూతుర్ని). This needs a small suffix list, not a full morphology model.
- **Over-correcting:** if a scene ends up with most of its words changed, it is too far from the original, so it is
  dropped instead (the same ≥ 60% faithfulness rule as Stage 7).

## How we measure it
1. **Bookish-word rate** per 100 words, before and after, on native and on flagged scenes. Native scenes should barely change.
2. **Strict "does the reply make sense?" check** (Opus and Gemini, as in Task 8), before and after. It must not go down.
3. **"Sounds spoken?" check:** a judge picks which of two versions sounds like real speech (both orders, a win needs both).
   It runs on a random sample.
4. **Spot list:** 30 before/after examples, written to `docs/eval/`, so a human can read them, including mistakes.

## Where it plugs in
- Stage 7 repaired pairs (`data/conversations/pilot_pairs.jsonl`)
- the raw movie corpus, before `s5_prepare.py`
- Stage 8 translated dialogues (IndicTrans2 output is bookish too, so this is the same problem)

## Files it would add (when implemented)
- `tlm/grandhika.py`: scene flag, word finder, table lookup with suffix handling
- `tlm/grandhika_table.json`: the fixed word table
- `stages/s7b_grandhika.py`: builds the candidate list and table, applies them, writes before/after statistics and examples
