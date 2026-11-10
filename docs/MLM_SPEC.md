# MLM — Movie Language Model (Telugu screenwriting assistant)

"MLM" here is the project codename. It is not a *masked* language model: it is a causal LM.

## Goal
Help writers draft Telugu movie dialogue that sounds like Telugu cinema: emotion, dialect, and the rhythm of mass, comedy and heartbreak scenes. New-age users can talk to it in Tenglish.

## Three layers

| Layer | What it learns | Data | Stage |
|---|---|---|---|
| 1. Knowledge | How Telugu movie dialogue sounds | 76 movies of subtitles, native Telugu script | Stage 1: domain adaptation |
| 2. Tasks | Screenwriting requests (write, continue, restyle, punch up) | situation → scene pairs (to be built) | Stage 2: instruction SFT |
| 3. Interface | Tenglish in and out | transliteration in a pipeline; later, real Tenglish | Stage 3 |

## Why native script for Stages 1–2
- Native script costs 2.07 Sarvam tokens per word; synthetic Tenglish costs 3.21. That makes native script cheaper and gives longer context.
- The Tenglish in the corpus is machine-transliterated, so the model would learn the converter's spelling, not how people type.

## Stage 1 record format (one scene per example)
```
భావోద్వేగం: EMOTIONAL_HEARTBREAK | మాండలికం: COASTAL_ANDHRA
- <turn 1>
- <turn 2>
...</s>
```
Turns are subtitle lines, not attributed speakers, so the format makes no speaker claims.

## Splits
Splits are by movie, from `data/splits/split_manifest.json`: 60 train / 8 val / 8 test movies, seed 42. No scene from a val or test movie is ever trained on.

## Stage 1 success criterion
Test-set perplexity of the adapted model is clearly lower than the base sarvam-2b's on the same 8 held-out movies. Qualitative check: 10 fixed prompts, generated with both models side by side.

## Example user requests (targets for Stage 2/3)
1. "Hero villain ni warn chese mass scene raayi, Rayalaseema slang lo" (Tenglish in)
2. "అమ్మ కొడుకుని వెళ్ళొద్దని బతిమాలే సీన్ రాయి" (Telugu in)
3. "Continue this scene: …" plus 3 turns
4. "Make this dialogue comedy" plus a dialogue
5. "Punch up this line: 'nenu ninnu vadalanu'"
