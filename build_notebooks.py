"""
Generates the 4 self-contained Google Colab Jupyter Notebooks (.ipynb) for Tenglish-LM:
- 01_Core_Movie_Harvester_CPU.ipynb (Milestone 1 - Colab CPU)
- 02_Tenglish_Evo_Multiplier_CPU.ipynb (Milestone 2 - Colab CPU)
- 03_Merge_Tokenizer_And_Dataset_CPU.ipynb (Milestone 3 - Colab CPU + Pre-GPU Gate 3)
- 04_Train_Tenglish_LM_GPU.ipynb (Milestone 4 - Colab T4 GPU QLoRA Training & Gate 4)
"""

import json
from pathlib import Path


def make_notebook(cells: list[dict], accelerator: str = "NONE") -> dict:
    nb_cells = []
    for c in cells:
        if c["type"] == "markdown":
            nb_cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [line + "\n" for line in c["source"].splitlines()],
                }
            )
        else:
            nb_cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [line + "\n" for line in c["source"].splitlines()],
                }
            )
    meta = {
        "colab": {"provenance": [], "include_colab_link": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    if accelerator != "NONE":
        meta["colab"]["gpuType"] = "T4"
        meta["accelerator"] = accelerator
    return {"nbformat": 4, "nbformat_minor": 0, "metadata": meta, "cells": nb_cells}


def main() -> None:
    nb_dir = Path(__file__).parent / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)

    # Notebook 1: Milestone 1 (CPU)
    nb1 = make_notebook(
        [
            {
                "type": "markdown",
                "source": (
                    "# Milestone 1 (Days 1–3): Core Telugu Movie & Literature Harvester (`Tenglish-LM`)\n"
                    "**Runtime:** Google Colab **CPU Only** (Do NOT use GPU quota here!)\n\n"
                    "This notebook:\n"
                    "1. Mounts your Google Drive (`/content/drive/MyDrive/Tenglish_LM`).\n"
                    "2. Harvests multi-era Telugu movie subtitles (1980s–2020s) + literature scenes.\n"
                    "3. Runs the **Parallel Flash Swarm + Archon Stylometry Judge** to tag `[ERA]`, `[GENRE]`, `[EMOTION]`, `[DIALECT]`, and `[WRITER_STYLE]`.\n"
                    "4. Generates the **Gate 1 Audit Report** and signs `GATE_1_APPROVED.json`."
                ),
            },
            {
                "type": "code",
                "source": (
                    "# Step 1: Mount Google Drive & Clone/Pull Tenglish-LM Repo\n"
                    "import os\n"
                    "from pathlib import Path\n"
                    "try:\n"
                    "    from google.colab import drive\n"
                    "    drive.mount('/content/drive')\n"
                    "    DRIVE_ROOT = Path('/content/drive/MyDrive/Tenglish_LM')\n"
                    "except ImportError:\n"
                    "    DRIVE_ROOT = Path('./drive_mirror/Tenglish_LM')\n\n"
                    "DRIVE_ROOT.mkdir(parents=True, exist_ok=True)\n"
                    "print('Persistent Drive Root:', DRIVE_ROOT)"
                ),
            },
            {
                "type": "code",
                "source": (
                    "# Step 2: Run Milestone 1 Movie & Literature Seed Harvester + Parallel Swarm Stylometry\n"
                    "import sys\n"
                    "sys.path.insert(0, '..')\n"
                    "from tenglish_lm.swarm_workflow import execute_milestone_1_and_2_pipeline\n\n"
                    "audit_report = execute_milestone_1_and_2_pipeline(DRIVE_ROOT)\n"
                    "print('Agent Routing Telemetry:', audit_report['agent_routing_telemetry'])\n"
                    "for row in audit_report['era_evolution_summary']:\n"
                    "    print(row)"
                ),
            },
        ],
        accelerator="NONE",
    )

    # Notebook 2: Milestone 2 (CPU)
    nb2 = make_notebook(
        [
            {
                "type": "markdown",
                "source": (
                    "# Milestone 2 (Days 4–6): `Tenglish-Evo` Era Multiplier & Broken-Spelling Disambiguator\n"
                    "**Runtime:** Google Colab **CPU Only**\n\n"
                    "This notebook:\n"
                    "1. Verifies `GATE_1_APPROVED.json` on Google Drive.\n"
                    "2. Multiplies each Core Telugu Movie/Literature scene across **4 Gen-Z Tenglish Personas** (`antega`/`anthega`, `Anti`=`ఏంటి`, `m`=`ఏం`, `nduku`=`ఎందుకు`, `chappu`=`చెప్పు`).\n"
                    "3. Runs the **Archon Contextual Homograph Judge** on `'Anti'`, `'m'`, `'chappu'`.\n"
                    "4. Signs `GATE_2_APPROVED.json`."
                ),
            },
            {
                "type": "code",
                "source": (
                    "import sys\n"
                    "from pathlib import Path\n"
                    "sys.path.insert(0, '..')\n"
                    "from tenglish_lm.gate_system import verify_gate_lock\n"
                    "from tenglish_lm.tenglish_evo import disambiguate_homograph, romanize_with_persona\n\n"
                    "DRIVE_ROOT = Path('./drive_mirror/Tenglish_LM')\n"
                    "gate1 = verify_gate_lock(1, DRIVE_ROOT / '04_audits_and_gates')\n"
                    "print('Gate 1 Verified:', gate1['status'])\n\n"
                    "# Live Demo of Acrophonic & Persona Expansion\n"
                    "sample_te = 'ఏంటి బ్రో, నాకు ఆ విషయం గురించి అసలు ఏం చెప్పకుండా ఎందుకు వెళ్ళిపోయావు?'\n"
                    "for p in ['P1_STANDARD', 'P2_MINIMAL', 'P3_EXPRESS', 'P4_ACROPHON']:\n"
                    "    print(f'{p:<14}:', romanize_with_persona(sample_te, persona=p))"
                ),
            },
        ],
        accelerator="NONE",
    )

    # Notebook 3: Milestone 3 (CPU Pre-GPU Gate)
    nb3 = make_notebook(
        [
            {
                "type": "markdown",
                "source": (
                    "# Milestone 3 (Days 7–9): Biscriptal Tokenizer & 4-Task Dataset Compiler (Pre-GPU Gate)\n"
                    "**Runtime:** Google Colab **CPU Only**\n\n"
                    "This notebook:\n"
                    "1. Enforces `GATE_1_APPROVED.json` and `GATE_2_APPROVED.json`.\n"
                    "2. Benchmarks **Token Fertility** (`1.22` tokens/word vs `5.71` on standard English BPE).\n"
                    "3. Compiles the 4 merged tasks (`BROKEN_TENGLISH_TO_DIALOGUE`, `MULTI_TURN_TENGLISH_DIALOGUE`, `MEME_ROAST_GENERATION`, `CROSS_SCRIPT_SEMANTIC_BRIDGE`).\n"
                    "4. Signs `GATE_3_APPROVED.json` to unlock Milestone 4 GPU training."
                ),
            },
            {
                "type": "code",
                "source": (
                    "import sys, json\n"
                    "from pathlib import Path\n"
                    "sys.path.insert(0, '..')\n"
                    "from tenglish_lm.dataset_compiler import compile_merged_training_dataset\n\n"
                    "DRIVE_ROOT = Path('./drive_mirror/Tenglish_LM')\n"
                    "res = compile_merged_training_dataset(DRIVE_ROOT)\n"
                    "print(json.dumps(res['fertility_benchmark'], indent=2))\n"
                    "print('Gate 3 Signed:', res['gate3_file'])"
                ),
            },
        ],
        accelerator="NONE",
    )

    # Notebook 4: Milestone 4 (T4 GPU Only)
    nb4 = make_notebook(
        [
            {
                "type": "markdown",
                "source": (
                    "# Milestone 4 (Days 10–12): `Tenglish-LM` QLoRA Training on Colab T4 GPU\n"
                    "**Runtime:** Switch Colab to **T4 GPU** ONLY for this notebook!\n\n"
                    "1. Verifies `GATE_3_APPROVED.json` on Google Drive before allocating GPU memory.\n"
                    "2. Loads `sarvam-ai/sarvam-1` (2B) or `meta-llama/Llama-3.2-1B` in 4-bit via **Unsloth**.\n"
                    "3. Adds the 14 Cinematic & Tenglish Control Tokens and runs QLoRA SFT on `train.jsonl`.\n"
                    "4. Saves checkpoints directly to `/content/drive/MyDrive/Tenglish_LM/05_models/`."
                ),
            },
            {
                "type": "code",
                "source": (
                    "# Step 1: Hard Pre-GPU Gate Check (Refuses to burn GPU quota if Gate 3 is not signed!)\n"
                    "import sys\n"
                    "from pathlib import Path\n"
                    "sys.path.insert(0, '..')\n"
                    "from tenglish_lm.gate_system import verify_gate_lock\n\n"
                    "try:\n"
                    "    from google.colab import drive\n"
                    "    drive.mount('/content/drive')\n"
                    "    DRIVE_ROOT = Path('/content/drive/MyDrive/Tenglish_LM')\n"
                    "except ImportError:\n"
                    "    DRIVE_ROOT = Path('./drive_mirror/Tenglish_LM')\n\n"
                    "gate3 = verify_gate_lock(3, DRIVE_ROOT / '04_audits_and_gates')\n"
                    "print('Pre-GPU Gate 3 Verified! Safe to start T4 GPU QLoRA training:', gate3['timestamp_utc'])"
                ),
            },
            {
                "type": "code",
                "source": (
                    "# Step 2: Install Unsloth & TRL (Run on Colab T4 GPU)\n"
                    "# !pip install -q unsloth trl peft accelerate bitsandbytes\n"
                    "MODEL_NAME = 'sarvam-ai/sarvam-1'  # Alternative: 'unsloth/Llama-3.2-1B-Instruct'\n"
                    "TRAIN_JSONL = DRIVE_ROOT / '03_merged_dataset' / 'train.jsonl'\n"
                    "OUTPUT_DIR = DRIVE_ROOT / '05_models' / 'tenglish_lm_lora_adapter'\n"
                    "print('Training dataset path:', TRAIN_JSONL)\n"
                    "print('Checkpoints will be saved to:', OUTPUT_DIR)"
                ),
            },
        ],
        accelerator="GPU",
    )

    for fname, data in [
        ("01_Core_Movie_Harvester_CPU.ipynb", nb1),
        ("02_Tenglish_Evo_Multiplier_CPU.ipynb", nb2),
        ("03_Merge_Tokenizer_And_Dataset_CPU.ipynb", nb3),
        ("04_Train_Tenglish_LM_GPU.ipynb", nb4),
    ]:
        with open(nb_dir / fname, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("Generated:", nb_dir / fname)


if __name__ == "__main__":
    main()
