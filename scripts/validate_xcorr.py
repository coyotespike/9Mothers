"""
Script to validate whole-trace cross-correlation against Phase 5a/5b results.

This is a sanity check that cross-references:
- Tracking lag (within sequences)
- Dead time (discrete steps)
- Whole-trace cross-correlation

Expected relationship:
    dead_time <= whole_trace_lag <= tracking_lag (with tolerance)

Usage:
    python scripts/validate_xcorr.py
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
    classify_commanded_changes,
    extract_discrete_steps,
    extract_tracking_sequences,
)
from turret_analysis.step_response import analyze_all_steps
from turret_analysis.tracking_analysis import analyze_all_tracking_sequences
from turret_analysis.xcorr import (
    analyze_whole_trace_lag,
    validate_regime_consistency,
)


def analyze_axis(axis: str, recording: dict):
    """Analyze cross-correlation for a single axis and validate."""
    print(f"\n{'='*60}")
    print(f"Validating {axis.upper()} axis cross-correlation")
    print(f"{'='*60}")

    # Extract axis-specific data
    cmd_key = f"{axis}_cmd"
    actual_key = f"{axis}_actual"

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

    #  Tracking lag
    print("\n Computing tracking lag...")
    fire_events = recording.get("fire")
    sequences = extract_tracking_sequences(classified, min_length=3, fire_events=fire_events)
    sequences_with_metrics = analyze_all_tracking_sequences(sequences, aligned_df)

    valid_tracking = sequences_with_metrics.filter(
        ~pl.col('lag_ms').is_nan() &
        ~pl.col('rms_error').is_nan()
    )
    tracking_lag_ms = valid_tracking['lag_ms'].median() if len(valid_tracking) > 0 else np.nan
    tracking_rms_error_deg = valid_tracking['rms_error'].median() if len(valid_tracking) > 0 else np.nan

    print(f"  Tracking lag (median): {tracking_lag_ms:.1f} ms")
    print(f"  Tracking RMS error (median): {tracking_rms_error_deg:.3f} deg")
    print(f"  Valid sequences: {len(valid_tracking)}")

    #  Dead time
    print("\n Computing dead time...")
    steps = extract_discrete_steps(
        classified,
        aligned_df,
        axis,
        min_magnitude_deg=0.5,
        fire_events=fire_events,
    )
    steps_with_metrics = analyze_all_steps(steps, aligned_df, filter_flags=True)

    valid_steps = steps_with_metrics.filter(
        ~pl.col('dead_time_ms').is_nan() &
        ~pl.col('rise_time_ms').is_nan()
    )
    dead_time_ms = valid_steps['dead_time_ms'].median() if len(valid_steps) > 0 else np.nan
    rise_time_ms = valid_steps['rise_time_ms'].median() if len(valid_steps) > 0 else np.nan

    print(f"  Dead time (median): {dead_time_ms:.1f} ms")
    print(f"  Rise time (median): {rise_time_ms:.1f} ms")
    print(f"  Valid steps: {len(valid_steps)}")

    #  Whole-trace cross-correlation
    print("\nComputing whole-trace cross-correlation...")
    xcorr_result = analyze_whole_trace_lag(aligned_df)

    whole_trace_lag_ms = xcorr_result['lag_ms']
    print(f"  Whole-trace lag: {whole_trace_lag_ms:.1f} ms")
    print(f"  Correlation: {xcorr_result['correlation']:.3f}")
    print(f"  Samples: {xcorr_result['n_samples']}")

    # Validation - regime consistency check
    print("\n" + "-"*60)
    print("REGIME CONSISTENCY CHECK")
    print("-"*60)

    if np.isnan(dead_time_ms) or np.isnan(rise_time_ms) or np.isnan(tracking_rms_error_deg):
        print("WARNING: Cannot validate - incomplete measurements from Phase 5a/5b")
        return None

    validation = validate_regime_consistency(
        whole_trace_lag_ms=whole_trace_lag_ms,
        dead_time_ms=dead_time_ms,
        rise_time_ms=rise_time_ms,
        tracking_lag_ms=tracking_lag_ms,
        tracking_rms_error_deg=tracking_rms_error_deg,
    )

    print("\nMeasurements:")
    print(f"  Step response lag: {validation['step_total_lag_ms']:.1f}ms = {dead_time_ms:.1f}ms dead + {rise_time_ms:.1f}ms rise")
    print(f"  Tracking temporal lag: {tracking_lag_ms:.1f}ms (spatial offset: {tracking_rms_error_deg:.3f}°)")
    print(f"  Whole-trace lag: {whole_trace_lag_ms:.1f}ms")

    print(f"\n{validation['interpretation']}")

    if validation['valid']:
        print("\n✓ Measurements are consistent with regime-dependent lag behavior")
    else:
        print("\n✗ Measurements show unexpected inconsistencies")

    return {
        'axis': axis,
        'dead_time_ms': dead_time_ms,
        'tracking_lag_ms': tracking_lag_ms,
        'whole_trace_lag_ms': whole_trace_lag_ms,
        'validation': validation,
    }


def main():
    """Run cross-correlation validation on both axes."""
    data_path = Path("motor.rrd")

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    print("="*60)
    print("CROSS-CORRELATION VALIDATION ")
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
    print("SUMMARY: PITCH VS YAW")
    print("="*60)

    if pitch_results and yaw_results:
        print(f"\nDead Time :")
        print(f"  Pitch: {pitch_results['dead_time_ms']:.1f} ms")
        print(f"  Yaw: {yaw_results['dead_time_ms']:.1f} ms")

        print(f"\nTracking Lag :")
        print(f"  Pitch: {pitch_results['tracking_lag_ms']:.1f} ms")
        print(f"  Yaw: {yaw_results['tracking_lag_ms']:.1f} ms")

        print(f"\nWhole-Trace Lag :")
        print(f"  Pitch: {pitch_results['whole_trace_lag_ms']:.1f} ms")
        print(f"  Yaw: {yaw_results['whole_trace_lag_ms']:.1f} ms")

        print(f"\nValidation Results:")
        print(f"  Pitch: {'PASS ✓' if pitch_results['validation']['valid'] else 'FAIL ✗'}")
        print(f"  Yaw: {'PASS ✓' if yaw_results['validation']['valid'] else 'FAIL ✗'}")

    print("\nCross-correlation validation complete!")


if __name__ == "__main__":
    main()
