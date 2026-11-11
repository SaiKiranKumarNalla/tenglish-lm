# Public evaluation report

The trained weights are private, so this folder shows **exactly what each model was asked and what it answered**, so anyone can judge the results.

| File | Contents |
|---|---|
| [PROMPTS.md](PROMPTS.md) | every prompt and generation setting |
| [normalization_examples.md](normalization_examples.md) | 40 sentences side by side (all 300 in `normalization_dakshina300.jsonl`) |
| [replies_movie_sample.md](replies_movie_sample.md) | 50 replies side by side (`replies_movie_sample.jsonl`) |

## Romanized → Telugu (300 real Dakshina sentences, word accuracy)

| Model | words exactly right |
|---|---|
| ours (7.6M normalizer) | 65.3% |
| Claude Haiku 5.5 | 79.2% |
| Claude Sonnet 5.5 | 82.2% |
| Claude Opus 5.5 | 86.7% |
| Gemini 3.8 Flash | 84.2% |

## Judged replies (each pair judged in both orders; a win needs both orders to agree)

| Judge | Comparison | first wins | second wins | tie / inconsistent | pairs |
|---|---|---|---|---|---|
| Sarvam-M | real vs random (reliability) | 51% | 15% | 34% | 150 |
| Sarvam-M | fine-tuned vs random | 58% | 10% | 32% | 150 |
| Sarvam-M | fine-tuned vs real | 29% | 25% | 46% | 150 |
| Sarvam-M | fine-tuned vs base (movie) | 48% | 15% | 37% | 150 |
| Sarvam-M | fine-tuned vs base (real comments) | 61% | 18% | 21% | 250 |
| Sarvam-M | fine-tuned vs claude | 5% | 63% | 32% | 150 |
| Sarvam-M | claude vs real reply | 66% | 5% | 29% | 150 |
| Claude Opus 5.5 | real vs random (reliability) | 91% | 6% | 3% | 150 |
| Claude Opus 5.5 | fine-tuned vs base (movie) | 94% | 4% | 2% | 150 |
| Claude | fine-tuned vs gemini (first = ours) | 0% | 99% | 1% | 150 |
| Gemini | fine-tuned vs claude (first = ours) | 0% | 97% | 3% | 150 |
