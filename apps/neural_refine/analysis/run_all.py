#!/usr/bin/env python3
"""
Run all analysis scripts in sequence.

Usage:
    python apps/neural_refine/analysis/run_all.py
"""

import subprocess
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "01_residual_correlation_analysis.py",
    "02_sequence_feature_analysis.py",
    "03_session_bias_analysis.py",
    "04_online_adaptation_analysis.py",
]


def main():
    print("=" * 70)
    print("RUNNING ALL ANALYSIS SCRIPTS")
    print("=" * 70)

    for script in SCRIPTS:
        script_path = ANALYSIS_DIR / script
        print(f"\n{'=' * 70}")
        print(f"Running: {script}")
        print("=" * 70 + "\n")

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
        )

        if result.returncode != 0:
            print(f"\nERROR: {script} failed with return code {result.returncode}")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)
    print("\nOutput figures saved to: outputs/analysis/")
    print("\nKey takeaways:")
    print("  1. SimRBF residuals are random noise (no spatial correlation)")
    print("  2. Session bias is the main reducible error source")
    print("  3. Online adaptation with 8-12 calibration points is the best approach")
    print("  4. Neural networks have limited value for this specific problem")


if __name__ == "__main__":
    main()
