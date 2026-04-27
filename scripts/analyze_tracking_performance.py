"""
Script to analyze tracking performance .

Runs tracking performance analysis on motor telemetry data:
- Tracking lag (cross-correlation)
- RMS tracking error
- Bandwidth estimation
- Clean vs recoil comparison

Usage:
    python scripts/analyze_tracking_performance.py
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
    extract_tracking_sequences,
)
from turret_analysis.tracking_analysis import (
    analyze_all_tracking_sequences,
    compare_tracking_performance,
    estimate_bandwidth,
    diagnose_error_source,
)


def analyze_axis(axis: str, recording: dict):
    """Analyze tracking performance for a single axis."""
    print(f"\n{'='*60}")
    print(f"Analyzing {axis.upper()} axis tracking performance")
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
    print(f"  Tracking updates: {(classified['regime'] == 'tracking').sum()}")
    print(f"  Ambiguous: {(classified['regime'] == 'ambiguous').sum()}")

    # Extract tracking sequences
    print("\nExtracting tracking sequences...")
    fire_events = recording.get("fire")
    sequences = extract_tracking_sequences(
        classified,
        min_length=3,
        fire_events=fire_events,
    )

    print(f"  Total sequences: {len(sequences)}")
    if len(sequences) > 0:
        print(f"  Clean sequences: {(~sequences['during_recoil']).sum()}")
        print(f"  Recoil-contaminated: {sequences['during_recoil'].sum()}")
        print(f"  Total duration: {sequences['duration_s'].sum():.1f}s")

    # Analyze tracking performance
    if len(sequences) == 0:
        print("\nNo tracking sequences found!")
        return None

    print("\nAnalyzing tracking sequences...")
    sequences_with_metrics = analyze_all_tracking_sequences(sequences, aligned_df)

    # Filter valid results
    valid = sequences_with_metrics.filter(
        ~pl.col('lag_ms').is_nan() &
        ~pl.col('rms_error').is_nan()
    )

    print(f"  Valid sequences analyzed: {len(valid)}")

    if len(valid) == 0:
        print("No valid tracking sequences!")
        return None

    # Compare clean vs recoil performance
    print("\n" + "-"*60)
    print("TRACKING PERFORMANCE SUMMARY")
    print("-"*60)

    comparison = compare_tracking_performance(sequences_with_metrics)

    for regime_name, stats in comparison.items():
        print(f"\n{regime_name.upper()} OPERATION:")
        print(f"  Sequences: {stats['n_sequences']}")
        print(f"  Total duration: {stats['total_duration_s']:.1f}s")

        if stats['n_sequences'] > 0:
            print(f"  Tracking lag: {stats['lag_ms']:.1f} ± {stats['lag_ms_std']:.1f} ms")
            print(f"  RMS error: {stats['rms_error']:.3f} ± {stats['rms_error_std']:.3f} deg")
            print(f"  Max error: {stats['max_error']:.3f} deg")
            print(f"  Update rate: {stats['update_rate_hz']:.1f} Hz")

    # Estimate bandwidth
    print("\n" + "-"*60)
    print("BANDWIDTH ESTIMATION")
    print("-"*60)

    bandwidth_hz, binned_data = estimate_bandwidth(
        sequences_with_metrics,
        error_threshold_deg=0.5
    )

    if not np.isnan(bandwidth_hz):
        print(f"  Bandwidth (0.5° threshold): {bandwidth_hz:.1f} Hz")
    else:
        print("  Bandwidth: Could not estimate (insufficient data or error never exceeds threshold)")

    if len(binned_data) > 0:
        print("\n  Error vs Update Rate:")
        for row in binned_data.iter_rows(named=True):
            if row['n_sequences'] > 0 and not np.isnan(row['median_error']):
                print(f"    {row['rate_bin_hz']:5.1f} Hz: {row['median_error']:.3f}° (n={row['n_sequences']})")

    # Diagnose error source
    print("\n" + "-"*60)
    print("ERROR SOURCE DIAGNOSTIC")
    print("-"*60)

    diagnosis = diagnose_error_source(sequences_with_metrics)
    print(f"\nAnalyzed {diagnosis.get('n_sequences', 0)} sequences")
    print(f"Mean error: {diagnosis.get('mean_error', np.nan):.3f}° ± {diagnosis.get('std_error', np.nan):.3f}°")
    print(f"Mean velocity: {diagnosis.get('mean_velocity', np.nan):.1f}° ± {diagnosis.get('std_velocity', np.nan):.1f}°/s")
    print(f"\nError-Velocity Correlation: r = {diagnosis.get('correlation', np.nan):.3f} (p = {diagnosis.get('p_value', np.nan):.4f})")
    print(f"\n{diagnosis.get('interpretation', 'No interpretation available')}")

    return {
        'sequences_with_metrics': sequences_with_metrics,
        'comparison': comparison,
        'bandwidth_hz': bandwidth_hz,
        'binned_data': binned_data,
        'diagnosis': diagnosis,
    }


def main():
    """Run tracking performance analysis on both axes."""
    data_path = Path("motor.rrd")

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    print("="*60)
    print("TRACKING PERFORMANCE ANALYSIS ")
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
        print("\nCLEAN OPERATION:")
        pitch_clean = pitch_results['comparison']['clean']
        yaw_clean = yaw_results['comparison']['clean']

        print(f"  Pitch lag: {pitch_clean['lag_ms']:.1f} ms | Yaw lag: {yaw_clean['lag_ms']:.1f} ms")
        print(f"  Pitch RMS error: {pitch_clean['rms_error']:.3f}° | Yaw RMS error: {yaw_clean['rms_error']:.3f}°")
        print(f"  Pitch bandwidth: {pitch_results['bandwidth_hz']:.1f} Hz | Yaw bandwidth: {yaw_results['bandwidth_hz']:.1f} Hz")

        print("\nRECOIL IMPACT:")
        pitch_recoil = pitch_results['comparison']['recoil']
        yaw_recoil = yaw_results['comparison']['recoil']

        if pitch_recoil['n_sequences'] > 0:
            pitch_degradation = ((pitch_recoil['rms_error'] - pitch_clean['rms_error']) /
                                pitch_clean['rms_error'] * 100)
            print(f"  Pitch error increase: {pitch_degradation:+.1f}%")

        if yaw_recoil['n_sequences'] > 0:
            yaw_degradation = ((yaw_recoil['rms_error'] - yaw_clean['rms_error']) /
                              yaw_clean['rms_error'] * 100)
            print(f"  Yaw error increase: {yaw_degradation:+.1f}%")

    print("\nTracking performance analysis complete!")


if __name__ == "__main__":
    main()
