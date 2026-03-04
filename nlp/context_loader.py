"""context_loader.py — Load optimization and demand-forecast output files.

Reads every available output file from:
  • optimization/output-json/   (JSON files)
  • demand-forecast/output/     (CSV files)

and returns a single formatted context string that can be injected into the
OpenAI system prompt so the chatbot answers only from real data.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths (relative to the project root)
# ---------------------------------------------------------------------------
_OPT_JSON_DIR = Path("optimization/output-json")
_DEMAND_OUTPUT_DIR = Path("demand-forecast/output")


def _load_json_file(path: Path) -> str:
    """Read a JSON file and return a compact string representation."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=None, separators=(",", ":"))
    except Exception:
        return ""


def _load_csv_file(path: Path, max_rows: int = 200) -> str:
    """Read a CSV file and return a plain-text table (header + up to max_rows rows)."""
    try:
        lines: list[str] = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i > max_rows:
                    lines.append("... (additional rows truncated)")
                    break
                lines.append(", ".join(row))
        return "\n".join(lines)
    except Exception:
        return ""


def load_context(base_path: str | Path = ".") -> str:
    """Return a formatted context string with all available output data.

    Parameters
    ----------
    base_path:
        Root directory of the project (default: current working directory).

    Returns
    -------
    str
        Multi-section text block describing every loaded output file.
        Returns an empty string if no files are found.
    """
    base = Path(base_path)
    sections: list[str] = []

    # ── Optimization JSON outputs ─────────────────────────────────────────────
    opt_dir = base / _OPT_JSON_DIR
    if opt_dir.is_dir():
        for json_file in sorted(opt_dir.glob("*.json")):
            content = _load_json_file(json_file)
            if content:
                sections.append(
                    f"=== {json_file.name} (optimization output) ===\n{content}"
                )

    # ── Demand-forecast CSV outputs ───────────────────────────────────────────
    demand_dir = base / _DEMAND_OUTPUT_DIR
    if demand_dir.is_dir():
        for csv_file in sorted(demand_dir.glob("*.csv")):
            content = _load_csv_file(csv_file)
            if content:
                sections.append(
                    f"=== {csv_file.name} (demand-forecast output) ===\n{content}"
                )

    return "\n\n".join(sections)
