# Models used in this project

All the models that appear anywhere in the results, what they are, and how big they are. Sizes marked *measured* were counted in
this project; proprietary models don't publish their architecture or size, and that's stated rather than guessed.

## Built or trained in this project

| Model | Role | Architecture | Size | Trained on |
|---|---|---|---|---|
| **English detector** (`tlm/langid.py`) | marks which words are English | character n-gram (1–4) Naive Bayes | a table of n-gram counts | English wordlist + romanized movie words |
| **Word normalizer** (Stage 4) | romanized word → Telugu script | character-level encoder–decoder transformer: 3 + 3 layers, 256-dim, 4 heads | **5.6M** (*measured*) | 305k word pairs (Dakshina + synthetic) |
| **Sentence normalizer** (Stage 4b) | romanized sentence → Telugu script | character-level encoder–decoder transformer: 4 + 4 layers, 256-dim, 4 heads | **7.6M** (*measured*) | 136k sentence pairs (movie + Dakshina) |
| **Tiny GPT** (Stages 3, 5) | small-model reference | decoder-only transformer: 4 layers, 256-dim, 4 heads, tied embeddings | 3–5M | romanized / Telugu movie text |
| **Reply model** (Stage 5/5b) | writes the next line | **sarvam-1** + **LoRA** (rank 16) on all attention and MLP projections | 2.5B base (*measured*: 2,525M) + **24M** trained (*measured*) | 29k movie dialogue pairs, 3,000 steps |

### How big is the full system?

| Part | Parameters | Memory |
|---|---|---|
| English detector | ~111k n-gram counts (not a neural network) | < 2 MB |
| Sentence normalizer | **7.6M** | ~30 MB |
| Reply model: sarvam-1 base | **2,525M** | ~5 GB (bf16), < 2 GB (4-bit) |
| Reply model: LoRA adapter (the part trained here) | **24M** | ~48 MB |
| **Total** | **≈ 2.56B** | ~5 GB, or under 2 GB in 4-bit |

The normalizer, which handles the hardest part (messy spelling), is genuinely tiny. The reply model is a *small* LLM, not a tiny one:
it runs on one consumer GPU, or on a laptop in 4-bit. A 5M-parameter model trained from scratch was far worse at replies (Stage 5),
because 1.2M words of dialogue aren't enough to learn Telugu from zero. **Distillation** (training a 100–300M student from the 2.5B model)
is the natural next step toward a truly lightweight end-to-end system.

## Pretrained models used as a base or for comparison

| Model | By | Architecture | Size | Used for |
|---|---|---|---|---|
| **sarvam-1** | Sarvam AI | decoder-only transformer with a tokenizer optimized for Indian languages; open weights | ~2.5B (*measured*) | base of the reply model; the "untuned" baseline |
| sarvam-2b-v0.5 | Sarvam AI | decoder-only transformer, open weights | ~2.5B | v1 experiments; tokenizer comparison |
| **Sarvam-M** | Sarvam AI | open 24B model; Sarvam describes it as built on Mistral Small | 24B | **judge**, run locally on Colab in 4-bit |
| Qwen2.5-0.5B | Alibaba | decoder-only transformer, open weights | 0.5B | tokenizer comparison only |
| GPT-2 | OpenAI | decoder-only transformer, byte-level BPE tokenizer | 124M | tokenizer comparison only |

## Frontier models (compared through their apps, no API)

| Model | By | Accessed through | Architecture / size |
|---|---|---|---|
| **Claude Opus 5.5** | Anthropic | Claude Code CLI (`claude -p`) | not publicly disclosed |
| **Claude Sonnet 5.5** | Anthropic | Claude Code CLI (`--model claude-sonnet-5-5`) | not publicly disclosed |
| **Claude Haiku 5.5** | Anthropic | Claude Code CLI (`--model claude-haiku-5-5`) | not publicly disclosed |
| **Gemini 3.8 Flash** | Google | Antigravity app (files uploaded by hand) | not publicly disclosed |

Frontier models are general-purpose models many times larger than anything trained here. They weren't trained for this task
specifically; they were given the same instructions as described in [eval/PROMPTS.md](eval/PROMPTS.md).
