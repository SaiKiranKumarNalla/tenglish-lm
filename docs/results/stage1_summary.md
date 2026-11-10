# Stage 1 results (sarvam-2b-v0.5 + LoRA r=16, L4, 2 epochs)

| Metric | Base | Tuned |
|---|---|---|
| Test perplexity (8 unseen movies, 365 blocks x 512 tok) | 32.89 | 14.07 |

Training time: 35 min. Blocks: {'train': 2757, 'val': 346, 'test': 365}.

Val loss by step: 50: 2.698, 100: 2.636, 150: 2.607, 200: 2.590, 250: 2.582, 300: 2.578, 346: 2.578

The adapter (88 MB) is kept out of git: download it from Colab or Drive.
