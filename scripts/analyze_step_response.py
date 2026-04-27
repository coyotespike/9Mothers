"""
Script to analyze step response characteristics .

Runs step response analysis on motor telemetry data:
- Dead time measurement
- Rise time, overshoot, settling time
- Second-order system fitting (ωn, ζ, τ)
- Pitch vs yaw comparison

Usage:
    python scripts/analyze_step_response.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
import numpy as np
from turret_analysis import (
    load_recording,
    align_signals,
    compute_aligned_signals,
    classify_commanded_changes,
    extract_discrete_steps,
)
from turret_analysis.step_response import analyze_all_steps


def analyze_axis(axis: str, recording: dict):
    """Analyze step response for a single axis."""
    print(f"\n{'='*60}")
    print(f"Analyzing {axis.upper()} axis step response")
    print(f"{'='*60}")

    # Extract axis-specific data
    cmd_key = f"{axis}_cmd"
    actual_key = f"{axis}_actual"

    if cmd_key not in recording or actual_key not in recording:
        print(f"Error: {axis} data not found in recording!")
        return None

    cmd_df = recording[cmd_key]
    actual_df = recording[actual_key]

    # Align signals
    print("Aligning signals...")
    aligned_df, metadata = align_signals(cmd_df, actual_df)

    # Classify commanded changes
    print("Classifying commanded changes...")
    classified = classify_commanded_changes(
        cmd_df,
        step_gap_ms=200.0,
        tracking_gap_ms=100.0,
        min_detection_deg=0.01,
    )

    print(f"  Total transitions detected: {len(classified)}")
    print(f"  Discrete steps: {(classified['regime'] == 'step').sum()}")

    # Extract discrete steps
    print("\nExtracting discrete steps...")
    fire_events = recording.get("fire")
    steps = extract_discrete_steps(
        classified,
        aligned_df,
        axis,
        min_magnitude_deg=0.5,
        fire_events=fire_events,
        pre_window_ms=200.0,
        post_window_ms=1000.0,
    )

    print(f"  Total steps extracted: {len(steps)}")
    if len(steps) > 0:
        print(f"  Isolated steps: {steps['isolated'].sum()}")
        print(f"  During recoil: {steps['during_recoil'].sum()}")
        print(f"  Pre-fire: {steps['pre_fire'].sum()}")
        print(f"  Edge truncated: {steps['edge_truncated'].sum()}")
        print(f"  Has interp gap: {steps['has_interp_gap'].sum()}")
        print(f"  Below min magnitude: {steps['below_min_magnitude'].sum()}")

    # Analyze step responses
    if len(steps) == 0:
        print("\nNo discrete steps found!")
        return None

    print("\nAnalyzing step responses (clean steps only)...")
    steps_with_metrics = analyze_all_steps(
        steps,
        aligned_df,
        filter_flags=True,  # Only analyze clean, isolated steps
    )

    print(f"  Clean steps analyzed: {len(steps_with_metrics)}")

    if len(steps_with_metrics) == 0:
        print("No clean steps available for analysis!")
        return None

    # Filter valid results (successful fits)
    valid = steps_with_metrics.filter(
        ~pl.col('dead_time_ms').is_nan() &
        ~pl.col('rise_time_ms').is_nan()
    )

    print(f"  Valid step responses: {len(valid)}")

    if len(valid) == 0:
        print("No valid step responses!")
        return None

    # Compute statistics
    print("\n" + "-"*60)
    print("STEP RESPONSE SUMMARY")
    print("-"*60)

    print(f"\nDead Time:")
    print(f"  Median: {valid['dead_time_ms'].median():.1f} ms")
    print(f"  Std: {valid['dead_time_ms'].std():.1f} ms")
    print(f"  Range: [{valid['dead_time_ms'].min():.1f}, {valid['dead_time_ms'].max():.1f}] ms")

    print(f"\nRise Time (10% to 90%):")
    print(f"  Median: {valid['rise_time_ms'].median():.1f} ms")
    print(f"  Std: {valid['rise_time_ms'].std():.1f} ms")
    print(f"  Range: [{valid['rise_time_ms'].min():.1f}, {valid['rise_time_ms'].max():.1f}] ms")

    overshoot_valid = valid.filter(~pl.col('overshoot_deg').is_nan())
    if len(overshoot_valid) > 0:
        print(f"\nOvershoot:")
        print(f"  Median: {overshoot_valid['overshoot_deg'].median():.3f}°")
        print(f"  Std: {overshoot_valid['overshoot_deg'].std():.3f}°")
        print(f"  Max: {overshoot_valid['overshoot_deg'].max():.3f}°")

    settling_valid = valid.filter(~pl.col('settling_time_ms').is_nan())
    if len(settling_valid) > 0:
        print(f"\nSettling Time (±0.1°):")
        print(f"  Median: {settling_valid['settling_time_ms'].median():.1f} ms")
        print(f"  Std: {settling_valid['settling_time_ms'].std():.1f} ms")
        print(f"  Range: [{settling_valid['settling_time_ms'].min():.1f}, {settling_valid['settling_time_ms'].max():.1f}] ms")

    # Second-order model fit statistics
    fit_valid = valid.filter(
        ~pl.col('wn_rad_s').is_nan() &
        ~pl.col('zeta').is_nan()
    )

    if len(fit_valid) > 0:
        print(f"\nSecond-Order Model Fit:")
        print(f"  Successful fits: {len(fit_valid)}/{len(valid)}")
        print(f"  Natural frequency (ωn): {fit_valid['wn_rad_s'].median():.2f} ± {fit_valid['wn_rad_s'].std():.2f} rad/s")
        print(f"  Damping ratio (ζ): {fit_valid['zeta'].median():.3f} ± {fit_valid['zeta'].std():.3f}")
        print(f"  Model delay (τ): {fit_valid['tau_ms'].median():.1f} ± {fit_valid['tau_ms'].std():.1f} ms")
        print(f"  R² median: {fit_valid['fit_r_squared'].median():.3f}")

    # Analyze by direction (for pitch: up vs down, gravity effects)
    if axis == "pitch":
        print("\n" + "-"*60)
        print("DIRECTION ANALYSIS (Pitch: Gravity Effects)")
        print("-"*60)

        up_steps = valid.filter(pl.col('direction') == 'up')
        down_steps = valid.filter(pl.col('direction') == 'down')

        print(f"\nUP-SLEW (against gravity): n={len(up_steps)}")
        if len(up_steps) > 0:
            print(f"  Dead time: {up_steps['dead_time_ms'].median():.1f} ms")
            print(f"  Rise time: {up_steps['rise_time_ms'].median():.1f} ms")

        print(f"\nDOWN-SLEW (with gravity): n={len(down_steps)}")
        if len(down_steps) > 0:
            print(f"  Dead time: {down_steps['dead_time_ms'].median():.1f} ms")
            print(f"  Rise time: {down_steps['rise_time_ms'].median():.1f} ms")

        if len(up_steps) > 0 and len(down_steps) > 0:
            rise_time_diff = down_steps['rise_time_ms'].median() - up_steps['rise_time_ms'].median()
            print(f"\n  Rise time difference (down - up): {rise_time_diff:+.1f} ms")
            if rise_time_diff < 0:
                print(f"  → Down-slew is {abs(rise_time_diff):.1f} ms FASTER (gravity assist)")

    return {
        'steps_with_metrics': steps_with_metrics,
        'valid': valid,
    }


def main():
    """Run step response analysis on both axes."""
    data_path = Path("motor.rrd")

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    print("="*60)
    print("STEP RESPONSE ANALYSIS ")
    print("="*60)

    # Load recording once
    print("\nLoading motor telemetry data...")
    recording = load_recording(data_path, verbose=False)

    # Analyze pitch
    pitch_results = analyze_axis("pitch", recording)

    # Analyze yaw
    yaw_results = analyze_axis("yaw", recording)

    # Summary comparison
    print("\n" + "="*60)
    print("PITCH VS YAW COMPARISON")
    print("="*60)

    if pitch_results and yaw_results:
        pitch_valid = pitch_results['valid']
        yaw_valid = yaw_results['valid']

        print(f"\nDead Time:")
        print(f"  Pitch: {pitch_valid['dead_time_ms'].median():.1f} ms")
        print(f"  Yaw: {yaw_valid['dead_time_ms'].median():.1f} ms")

        print(f"\nRise Time:")
        print(f"  Pitch: {pitch_valid['rise_time_ms'].median():.1f} ms")
        print(f"  Yaw: {yaw_valid['rise_time_ms'].median():.1f} ms")

        pitch_fit = pitch_valid.filter(~pl.col('wn_rad_s').is_nan())
        yaw_fit = yaw_valid.filter(~pl.col('wn_rad_s').is_nan())

        if len(pitch_fit) > 0 and len(yaw_fit) > 0:
            print(f"\nNatural Frequency (ωn):")
            print(f"  Pitch: {pitch_fit['wn_rad_s'].median():.2f} rad/s")
            print(f"  Yaw: {yaw_fit['wn_rad_s'].median():.2f} rad/s")

            print(f"\nDamping Ratio (ζ):")
            print(f"  Pitch: {pitch_fit['zeta'].median():.3f}")
            print(f"  Yaw: {yaw_fit['zeta'].median():.3f}")

    print("\nStep response analysis complete!")


if __name__ == "__main__":
    main()
