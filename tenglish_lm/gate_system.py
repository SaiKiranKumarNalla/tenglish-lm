"""
Human-in-the-Loop (HITL) Gate Lock & Semi-Automated Audit System.

Every Milestone (1..4) produces:
1. Automated schema & quality validation checks.
2. Stratified Audit Sample + Archon Anomaly Report (.json & .html).
3. Hard Gate Lock file (GATE_X_APPROVED.json) required before the next milestone runs.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class GateLockError(RuntimeError):
    """Raised when a downstream milestone runs before the prerequisite gate is approved."""


def compute_file_sha256(filepath: str | Path) -> str:
    """Computes SHA-256 checksum of a dataset file for gate integrity verification."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_gate_lock(gate_number: int, gates_dir: str | Path) -> Dict[str, Any]:
    """
    Verifies that GATE_<gate_number>_APPROVED.json exists and has status == 'APPROVED'.
    Raises GateLockError if missing or unapproved.
    """
    gate_file = Path(gates_dir) / f"GATE_{gate_number}_APPROVED.json"
    if not gate_file.exists():
        raise GateLockError(
            f"[GATE LOCK BLOCKED] Prerequisite Gate {gate_number} file not found at {gate_file}. "
            f"Please complete Milestone {gate_number} audit and sign off GATE_{gate_number}_APPROVED.json first."
        )
    with open(gate_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("status") != "APPROVED":
        raise GateLockError(
            f"[GATE LOCK BLOCKED] Gate {gate_number} status is '{data.get('status')}', not 'APPROVED'. "
            f"Review the audit report and approve all checklist items first."
        )
    return data


def sign_gate_approval(
    gate_number: int,
    milestone_name: str,
    gates_dir: str | Path,
    checklist: Dict[str, bool],
    metrics: Dict[str, Any],
    artifact_files: List[str | Path],
    reviewer_notes: str = "Audited and approved via AI-assisted HITL review.",
    human_overrides: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Writes GATE_<gate_number>_APPROVED.json after verifying all checklist items are True.
    """
    Path(gates_dir).mkdir(parents=True, exist_ok=True)
    all_passed = all(checklist.values())
    status = "APPROVED" if all_passed else "PENDING_HUMAN_REVIEW"

    checksums = {}
    for fp in artifact_files:
        p = Path(fp)
        if p.exists():
            checksums[p.name] = compute_file_sha256(p)

    payload = {
        "gate_number": gate_number,
        "milestone_name": milestone_name,
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checklist": checklist,
        "metrics": metrics,
        "artifact_sha256": checksums,
        "reviewer_notes": reviewer_notes,
        "human_overrides": human_overrides or {},
    }

    out_path = Path(gates_dir) / f"GATE_{gate_number}_{status}.json"
    if all_passed:
        out_path = Path(gates_dir) / f"GATE_{gate_number}_APPROVED.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_path
