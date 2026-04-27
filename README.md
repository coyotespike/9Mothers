# Turret Motor Analysis


It took me a few days to dive in, but I had a lot of fun with this takehome challenge.

  I put around 6 hours into this takehome challenge, which was more than needed but let me explore different methodologies and have some fun with it.

A couple big takeaways:

Recoil recovery isn't fast enough for burst mode. Recovery takes 600ms vs 300ms required. Recoil affects both axes (2.4° pitch, 1.5° yaw), not just pitch. Fixes: disturbance feedforward (counter-torque against predicted recoil) or stiffen the mechanical mount.

Tracking error of 0.9-1.4° grows with commanded velocity (0.03-0.04 deg per deg/s). Root cause: low closed-loop bandwidth (5-6 Hz). Pitch is overdamped (ζ=1.04) and 2× slower than yaw (155ms vs 75ms step response). Fixes: increase K_p on both axes to raise bandwidth to ~10-15 Hz and also reduce K_d on pitch to avoid overdamping, or else add velocity feedforward. Since 90% of operation is continuous tracking, velocity feedforward could be very effective, but I'd start by tuning the gains first.

The command update rate (57 Hz) shows the vision system is already pretty fast and not the bottleneck. Slew saturation never got reached, so the motors are powerful enough, too.

## Results

See Report.md for complete analysis.

Key findings:
- Yaw is 2× faster than pitch (75ms vs 155ms total response)
- Both axes fail the 300ms rapid-fire budget by 200-300ms
- Recoil shows cross-axis coupling (yaw deflection is 60% of pitch)

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

Run all analyses:
```bash
python run_all_analyses.py
```

Run individual scripts:
```bash
python scripts/analyze_step_response.py
python scripts/phase8_comparison.py
python scripts/phase9_disturbance.py
```

Run tests:
```bash
pytest
```

Interactive analysis:
```bash
jupyter notebook notebooks/01_analysis.ipynb
```

## Structure

- Report.md - Main deliverable
- src/turret_analysis/ - Core package
- scripts/ - Analysis runners
- tests/ - Test suite
- figures/ - Visualizations
- notebooks/ - Interactive analysis

## Dependencies

- Python 3.10+
- Rerun SDK
- Polars
- SciPy
- NumPy
- Matplotlib
