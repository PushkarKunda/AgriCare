#!/usr/bin/env python3
"""
================================================================================
FINAL YEAR PROJECT: DIGITAL SOIL NUTRIENT MAPPING USING REMOTE SENSING (MBU)
MODULE: ROOT INFERENCE WRAPPER
================================================================================
Quick CLI entry point for running soil nutrient predictions from the project root.
Usage:
    python predict.py <input_features.csv>
    python predict.py  (runs sample template)
================================================================================
"""

import sys
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent
PRODUCTION_DIR = PROJECT_ROOT / "05_Production_Models"
PREDICTOR = PRODUCTION_DIR / "predict_soil.py"
DEFAULT_INPUT = PRODUCTION_DIR / "sample_input_template.csv"

def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_INPUT)
    
    if not Path(csv_file).exists():
        print(f"[!] Error: File not found: {csv_file}")
        sys.exit(1)
        
    cmd = [sys.executable, str(PREDICTOR), str(csv_file)]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
