"""
End-to-End CPU Runner for Tenglish-LM & Tenglish-Evo (Milestones 1, 2, and 3).
Executes the Parallel Flash Swarm + Archon Judge workflow and compiles the Gate 1..3 Audit Reports.
"""

import json
from pathlib import Path

from tenglish_lm.dataset_compiler import compile_merged_training_dataset
from tenglish_lm.swarm_workflow import execute_milestone_1_and_2_pipeline


def main() -> None:
    workspace_out = Path(__file__).parent / "drive_mirror" / "Tenglish_LM"
    print("=== Running Milestone 1 (Movie Stylometry) & Milestone 2 (Tenglish-Evo Multiplier) ===")
    m12_report = execute_milestone_1_and_2_pipeline(workspace_out)

    print(
        f"Agent Routing Telemetry: Flash Swarm = {m12_report['agent_routing_telemetry']['flash_worker_tasks']} tasks "
        f"({m12_report['agent_routing_telemetry']['flash_share_pct']}%), "
        f"Archon Escalations = {m12_report['agent_routing_telemetry']['archon_escalations']} "
        f"({m12_report['agent_routing_telemetry']['archon_share_pct']}%)"
    )

    print("\n--- Decade-by-Decade English % (CMI) Evolution ---")
    for row in m12_report["era_evolution_summary"]:
        print(
            f"  {row['era']:<16} | Avg English %: {row['avg_english_pct']:>5.1f}% | "
            f"Hybrid (-ki/-lo): {row['hybrid_fused_tokens']} | Light-Verbs: {row['light_verb_compounds']}"
        )

    print("\n--- Adversarial Homograph & Acrophonic Disambiguation ('Anti', 'm', 'chappu', 'anthega') ---")
    for test in m12_report["adversarial_homograph_tests"]:
        status = "PASS" if test["passed"] else "FAIL"
        print(f"  [{status}] '{test['sentence']}' -> '{test['surface']}' resolved to: {test['resolved_te']} ({test['meaning']})")

    print("\n=== Running Milestone 3 (Biscriptal Tokenizer Fertility & 4-Task Dataset Merge) ===")
    m3_report = compile_merged_training_dataset(workspace_out)
    print(f"Compiled {m3_report['train_count']} train examples and {m3_report['val_count']} val examples.")
    print("Token Fertility Benchmark:", json.dumps(m3_report["fertility_benchmark"], indent=2))
    print(f"Gate 3 Lock written to: {m3_report['gate3_file']}")


if __name__ == "__main__":
    main()
