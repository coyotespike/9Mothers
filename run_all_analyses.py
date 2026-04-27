#!/usr/bin/env python3
"""
Master Analysis Runner

Executes all production analyses and generates visualizations.
Run this to reproduce all results from the report.

Usage:
    python run_all_analyses.py [--skip-tests]

Output:
    - Console reports from each analysis
    - Figures saved to figures/ directory
    - All results referenced in Report.md
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report status."""
    print("\n" + "="*80)
    print(f"Running: {description}")
    print("="*80)

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    else:
        print(f"✓ Completed: {description}")
        return True


def main():
    skip_tests = "--skip-tests" in sys.argv

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TURRET ANALYSIS - MASTER RUNNER                           ║
║                                                                              ║
║  This script reproduces all analyses from the report:                       ║
║    1. Data loading and validation (tests)                                   ║
║    2. Tracking performance analysis                                         ║
║    3. Step response and latency measurement                                 ║
║    4. Cross-correlation validation                                          ║
║    5. Pitch vs yaw dynamics comparison                                      ║
║    6. Firing disturbance characterization                                   ║
║                                                                              ║
║  Expected runtime: 3-5 minutes                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    success = True

    # Step 1: Run tests (optional)
    if not skip_tests:
        success &= run_command(
            "pytest -v",
            "Test Suite - Data loading, alignment, segmentation validation"
        )
        if not success:
            print("\n⚠️  Tests failed. Continue anyway? [y/N] ", end="")
            if input().lower() != 'y':
                sys.exit(1)
    else:
        print("\nℹ️  Skipping tests (--skip-tests flag set)")

    # Create figures directory
    Path("figures").mkdir(exist_ok=True)

    # Step 2: Tracking Performance
    success &= run_command(
        "python scripts/analyze_tracking_performance.py",
        "Tracking Performance - Lag, RMS error, bandwidth analysis"
    )

    # Step 3: Step Response
    success &= run_command(
        "python scripts/analyze_step_response.py",
        "Step Response - Dead time, rise time, second-order fit"
    )

    # Step 4: Validation
    success &= run_command(
        "python scripts/validate_xcorr.py",
        "Cross-Correlation Validation - Sanity check on latency estimates"
    )

    # Step 5: Dynamics Comparison
    success &= run_command(
        "python scripts/analyze_dynamics_comparison.py",
        "Dynamics Comparison - Pitch vs yaw performance characterization"
    )

    # Step 6: Firing Disturbance
    success &= run_command(
        "python scripts/analyze_firing_disturbance.py",
        "Firing Disturbance - Recoil deflection and recovery time"
    )

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if success:
        print("""
✓ All analyses completed successfully!

Results:
  - Console output above shows quantitative findings
  - Figures saved to figures/ directory
  - See Report.md for complete writeup with visualizations

Next steps:
  1. Review figures/: ls -lh figures/
  2. Read Report.md for interpretation
  3. Check individual scripts for methodology details

To regenerate specific analyses:
  python scripts/analyze_step_response.py
  python scripts/analyze_firing_disturbance.py
  python scripts/analyze_dynamics_comparison.py
  etc.
        """)
        sys.exit(0)
    else:
        print("""
⚠️  Some analyses failed (see above)

Troubleshooting:
  - Check that motor.rrd exists in current directory
  - Verify dependencies: pip install -e ".[dev]"
  - Run tests individually: pytest -v
  - Check individual script: python scripts/analyze_step_response.py
        """)
        sys.exit(1)


if __name__ == "__main__":
    main()
