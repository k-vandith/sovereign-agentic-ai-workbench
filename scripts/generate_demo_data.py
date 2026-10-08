#!/usr/bin/env python3
"""Generate synthetic industrial documents for demo mode."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_settings

SAMPLE_DOCS = {
    "safety_protocol.txt": """
Industrial Safety Protocol – Confidential

1. All personnel must wear PPE in Zone A and Zone B.
2. Maximum allowable temperature for Reactor R-12 is 450°C.
3. Emergency shutdown procedure: press red button panel, evacuate within 90 seconds.
4. Chemical spill response: contain with absorbent, notify Safety Officer immediately.
5. Last safety audit: 2025-11-15. Next due: 2026-05-15.
""",
    "equipment_manual.txt": """
Equipment Manual – Pump Unit P-204

Model: Centrifugal Industrial Pump Series X
Operating pressure: 8–12 bar
Flow rate: 120–180 m³/h
Maintenance interval: every 2,000 operating hours
Known issues: seal wear after 18 months under high-solids conditions.
Spare parts list: mechanical seal kit (SKU-MS-204), impeller (SKU-IMP-204).
""",
    "process_notes.md": """
# Process Notes – Batch Line 3

## Daily Targets
- Throughput: 42 tons/day
- Reject rate target: < 0.8%
- Energy budget: 1.2 MWh/shift

## Observed Anomalies (last week)
- Slight vibration increase on conveyor C-7 (investigating bearing)
- Temperature drift on dryer D-3 (+3°C average)
""",
}


def main() -> None:
    settings = get_settings()
    sample_dir = settings.data_dir / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for name, content in SAMPLE_DOCS.items():
        path = sample_dir / name
        path.write_text(content.strip() + "\n", encoding="utf-8")
        print(f"Wrote {path}")
    print(f"\nDemo documents ready in {sample_dir}")
    print("You can now upload them via the Streamlit UI or API.")


if __name__ == "__main__":
    main()
