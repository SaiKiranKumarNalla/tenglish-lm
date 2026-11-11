# Prompts and settings used in evaluation

## Our fine-tuned reply model (sarvam-1 + LoRA)
The model was trained on, and is prompted with, plain dialogue lines, each prefixed with "- ". It continues after the final "- ":
```
- <previous line, optional>
- <last line>
-
```
Decoding: sampling, temperature 0.7, top-p 0.9, repetition penalty 1.2, no repeated 3-word phrases, at most 50 new tokens,
stop at the end of the line. The untuned sarvam-1 baseline gets the identical prompt and settings.

## Our spelling normalizer
No prompt: a character-level model. English words are first detected and replaced by `@`, then put back after normalization.
Greedy decoding.

## Frontier models (Claude via `claude -p`; ChatGPT and Gemini via their apps): 25 items per request
**Romanized → Telugu**
```
Convert each romanized Telugu sentence below into Telugu script. Keep the meaning and words exactly; do not translate or
add anything. Return ONLY a JSON list of {n} strings, in the same order.
1. <sentence>
2. ...
```
**Replies**
```
Each item is the last line someone said in a Telugu movie conversation. For each, write the next line: a short, natural reply
in spoken, colloquial Telugu (Telugu script, under 20 words), as a character would really say it. Return ONLY a JSON list of
{n} strings, in the same order.
```
**Judging (frontier models as judges)**
```
For each item, a person said something in Telugu, and there are two candidate replies A and B. Which reply is the better next
line in a natural, spoken Telugu conversation (relevant, fluent, how people really talk)? Return ONLY a JSON list of {n}
letters, "A" or "B", in the same order.
```

## Sarvam-M as judge (one pair per prompt, chat template, thinking disabled, greedy, 4-bit weights)
```
You are judging replies in a Telugu conversation. A person said:
"<message>"

Reply A: "<reply>"
Reply B: "<reply>"

Which reply is a better next line in a natural, spoken Telugu conversation: relevant to what was said, fluent, and sounding
like how people really talk? Answer with a single letter: A or B.
```

## Fairness rules
- Every pair is judged **twice, with A and B swapped**. A win only counts when both orders agree; otherwise it's a tie.
- **No model judges its own replies.**
- Each judge is first checked on **real reply vs a random reply from another scene**, where the right answer is known.
