"""Run the full pipeline: python run_all.py  (raw workbook optional, see data/README.md)."""
import subprocess, sys
from pathlib import Path

import config as C

SRC = Path(__file__).resolve().parent / "src"
steps = ["01_build_panel.py", "02_descriptives.py", "03_main_did.py",
         "04_resilience_metrics.py", "05_composition_robustness.py"]
for s in steps:
    if s.startswith("01") and not C.RAW_XLSX.exists():
        print("[skip] 01_build_panel.py: raw workbook not found, using processed panel")
        continue
    print(f"\n=== {s} ===")
    subprocess.run([sys.executable, str(SRC / s)], check=True)
