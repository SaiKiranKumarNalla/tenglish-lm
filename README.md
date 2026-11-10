# Tenglish-LM & Tenglish-Evo

**A Morphologically-Aware Small Language Model & Era-Calibrated Code-Mixing Pipeline for Colloquial Telugu Cinema Dialogues, Memes, and Broken Gen-Z Tenglish.**

---

## Cloud Execution Stack (Zero Local Compute)
* **GitHub (`tenglish-lm`):** Version-controlled Python modules & 4 Google Colab Notebooks.
* **Google Drive (`/MyDrive/Tenglish_LM/`):** Persistent storage for harvested movie scenes, synthetic `Tenglish-Evo` personas, interactive audit reports, and `GATE_1..4_APPROVED.json` locks.
* **Google Colab:**
  * **Notebooks 1, 2, 3 (`CPU Runtime`):** Harvesting, `asyncio` Parallel Flash Swarm + Archon Stylometry/Homograph routing, CMI Evolution analytics, and Biscriptal Tokenizer compilation.
  * **Notebook 4 (`T4 GPU Runtime`):** Unsloth 4-bit QLoRA fine-tuning of `Sarvam-1-2B` / `Llama-3.2-1B` (locked behind `GATE_3_APPROVED.json`).

---

## 🚀 1-Click Google Colab Notebooks

| Milestone / Tool | Runtime | Open in Colab |
| :--- | :---: | :--- |
| **00. Quick-Test: YouTube Movie Trailer $\rightarrow$ `.CSV` & `.TXT` Scene Extractor** | **Free CPU** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaiKiranKumarNalla/tenglish-lm/blob/main/notebooks/00_Colab_QuickTest_YouTube_Trailer_Extractor.ipynb) |
| **01. Milestone 1: Core Telugu Movie & Literature Harvester** | **Free CPU** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaiKiranKumarNalla/tenglish-lm/blob/main/notebooks/01_Core_Movie_Harvester_CPU.ipynb) |
| **02. Milestone 2: `Tenglish-Evo` Era Multiplier & Broken-Spelling Engine** | **Free CPU** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaiKiranKumarNalla/tenglish-lm/blob/main/notebooks/02_Tenglish_Evo_Multiplier_CPU.ipynb) |
| **03. Milestone 3: Biscriptal Tokenizer & 4-Task Merger (Pre-GPU Gate 3)** | **Free CPU** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaiKiranKumarNalla/tenglish-lm/blob/main/notebooks/03_Merge_Tokenizer_And_Dataset_CPU.ipynb) |
| **04. Milestone 4: `Tenglish-LM` QLoRA Fine-Tuning** | **T4 GPU** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaiKiranKumarNalla/tenglish-lm/blob/main/notebooks/04_Train_Tenglish_LM_GPU.ipynb) |

---

## Repository Structure

```text
tenglish-lm/
├── tenglish_lm/
│   ├── __init__.py
│   ├── gate_system.py          # Hard Gate Lock & SHA-256 HITL Audit Sign-off system
│   ├── movie_harvester.py      # Milestone 1: SRT Pause-Gap Slicer & Multi-Era Seed Corpus (1980s-2020s)
│   ├── tenglish_evo.py         # Milestone 2: 4-Persona Permuter (antega/anthega, Anti=ఏంటి, m=ఏం, nduku=ఎందుకు) & CMI Engine
│   ├── swarm_workflow.py       # Non-Sequential Parallel Flash Swarm (96.5%) + Hierarchical Archon Judge (3.5%)
│   └── dataset_compiler.py     # Milestone 3: Biscriptal Fertility Benchmark & 4-Task Dataset Merger
├── notebooks/
│   ├── 00_Colab_QuickTest_YouTube_Trailer_Extractor.ipynb
│   ├── 01_Core_Movie_Harvester_CPU.ipynb
│   ├── 02_Tenglish_Evo_Multiplier_CPU.ipynb
│   ├── 03_Merge_Tokenizer_And_Dataset_CPU.ipynb
│   └── 04_Train_Tenglish_LM_GPU.ipynb
├── extract_youtube_movie.py    # CLI to extract any YouTube Movie/Trailer URL into Scene-Tagged .CSV and .TXT
└── run_pipeline_and_audit.py   # End-to-end CPU verification & Gate 1-3 Audit generator
```
