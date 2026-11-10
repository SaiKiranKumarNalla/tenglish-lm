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
│   ├── 01_Core_Movie_Harvester_CPU.ipynb
│   ├── 02_Tenglish_Evo_Multiplier_CPU.ipynb
│   ├── 03_Merge_Tokenizer_And_Dataset_CPU.ipynb
│   └── 04_Train_Tenglish_LM_GPU.ipynb
└── run_pipeline_and_audit.py   # End-to-end CPU verification & Gate 1-3 Audit generator
```
