"""
Pitch vs Yaw Comparison

Assembles all findings into comparison tables and checks asymmetries:
1. Step-response metrics comparison table
2. Tracking metrics comparison table
3. Gravity asymmetry check (pitch up-slew vs down-slew)
4. Tracking asymmetry check (up vs down moving targets)
5. MOI ratio interpretation (effective ratio from τ)

Usage:
    python scripts/phase8_comparison.py
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
from turret_analysis.tracking_analysis import (
    analyze_all_tracking_sequences,
    compare_tracking_performance,
    estimate_bandwidth,
)


def compute_step_metrics(axis: str, recording: dict):
    """Extract step-response metrics for an axis."""
    cmd_key = f"{axis}_cmd"
    actual_key = f"{axis}_actual"
    cmd_df = recording[cmd_key]
    actual_df = recording[actual_key]

    aligned_df, metadata = align_signals(cmd_df, actual_df)

    classified = classify_commanded_changes(
        cmd_df,
        step_gap_ms=200.0,
        tracking_gap_ms=100.0,
        min_detection_deg=0.01,
    )

    fire_events = recording.get("fire")
    steps = extract_discrete_steps(
        classified,
        aligned_df,
        axis,
        min_magnitude_deg=0.5,
        fire_events=fire_events,
    )

    isolated_steps = steps.filter(pl.col('isolated') == True)

    # Fit step responses
    fitted_steps = analyze_all_steps(
        isolated_steps,
        aligned_df,
        filter_flags=False,
    )

    # Filter to valid fits with all metrics present
    valid_fits = fitted_steps.filter(
        ~pl.col('dead_time_ms').is_nan() &
        ~pl.col('rise_time_ms').is_nan() &
        ~pl.col('wn_rad_s').is_nan() &
        ~pl.col('zeta').is_nan()
    )

    # Further filter to magnitude ≥ 1.0° to avoid encoder noise
    valid_fits = valid_fits.filter(pl.col('magnitude').abs() >= 1.0)

    # Extract timing parameters
    dead_times = valid_fits['dead_time_ms'].to_numpy()
    rise_times = valid_fits['rise_time_ms'].to_numpy()
    omegas = valid_fits['wn_rad_s'].to_numpy()
    zetas = valid_fits['zeta'].to_numpy()
    overshoots = valid_fits['overshoot_deg'].to_numpy()

    # Compute analytical settling time from ωn and ζ (2% criterion)
    # For second-order system: t_settle ≈ 4 / (ζ × ωn)
    settling_times_analytical = 4000.0 / (zetas * omegas)  # Convert to ms

    # Filter out outliers (some fits may have very small ζ×ωn)
    # Keep settling times < 1000 ms (reasonable for this system)
    valid_settling = settling_times_analytical[settling_times_analytical < 1000.0]

    if len(valid_settling) > 0:
        settling_time_median = np.median(valid_settling)
        settling_time_std = np.std(valid_settling)
    else:
        settling_time_median = np.nan
        settling_time_std = np.nan

    return {
        'dead_time_ms': (np.median(dead_times), np.std(dead_times)),
        'rise_time_ms': (np.median(rise_times), np.std(rise_times)),
        'omega_n': (np.median(omegas), np.std(omegas)),
        'zeta': (np.median(zetas), np.std(zetas)),
        'overshoot_deg': (np.median(overshoots), np.std(overshoots)),
        'settling_time_ms': (settling_time_median, settling_time_std),
        'tau_ms': (1000.0 / np.median(omegas), np.std(1000.0 / omegas)),
        'n_steps': len(valid_fits),
    }


def compute_tracking_metrics(axis: str, recording: dict):
    """Extract tracking metrics for an axis."""
    cmd_key = f"{axis}_cmd"
    actual_key = f"{axis}_actual"
    cmd_df = recording[cmd_key]
    actual_df = recording[actual_key]

    aligned_df, metadata = align_signals(cmd_df, actual_df)

    classified = classify_commanded_changes(
        cmd_df,
        step_gap_ms=200.0,
        tracking_gap_ms=100.0,
        min_detection_deg=0.01,
    )

    # Extract tracking sequences
    fire_events = recording.get("fire")
    sequences = extract_tracking_sequences(classified, min_length=3, fire_events=fire_events)

    if len(sequences) == 0:
        return {
            'rms_error_deg': np.nan,
            'rms_clean_deg': np.nan,
            'rms_recoil_deg': np.nan,
            'bandwidth_hz': np.nan,
            'n_samples': 0,
        }

    # Analyze sequences
    sequences_with_metrics = analyze_all_tracking_sequences(sequences, aligned_df)

    valid = sequences_with_metrics.filter(
        ~pl.col('lag_ms').is_nan() &
        ~pl.col('rms_error').is_nan()
    )

    if len(valid) == 0:
        return {
            'rms_error_deg': np.nan,
            'rms_clean_deg': np.nan,
            'rms_recoil_deg': np.nan,
            'bandwidth_hz': np.nan,
            'n_samples': 0,
        }

    # Use functions for comparison and bandwidth estimation
    comparison = compare_tracking_performance(sequences_with_metrics)
    bandwidth_hz, binned_data = estimate_bandwidth(
        sequences_with_metrics,
        error_threshold_deg=0.5
    )

    # Extract metrics from comparison results
    all_stats = comparison.get('all', {})
    clean_stats = comparison.get('clean', {})
    recoil_stats = comparison.get('recoil', {})

    return {
        'rms_error_deg': all_stats.get('rms_error', np.nan),
        'rms_clean_deg': clean_stats.get('rms_error', np.nan),
        'rms_recoil_deg': recoil_stats.get('rms_error', np.nan),
        'bandwidth_hz': bandwidth_hz,
        'n_samples': all_stats.get('n_sequences', 0),
    }


def check_gravity_asymmetry(recording: dict):
    """Check pitch up-slew vs down-slew timing."""
    cmd_df = recording['pitch_cmd']
    actual_df = recording['pitch_actual']

    aligned_df, metadata = align_signals(cmd_df, actual_df)

    classified = classify_commanded_changes(
        cmd_df,
        step_gap_ms=200.0,
        tracking_gap_ms=100.0,
        min_detection_deg=0.01,
    )

    fire_events = recording.get("fire")
    steps = extract_discrete_steps(
        classified,
        aligned_df,
        'pitch',
        min_magnitude_deg=0.5,
        fire_events=fire_events,
    )

    isolated_steps = steps.filter(pl.col('isolated') == True)

    # Fit step responses
    fitted_steps = analyze_all_steps(
        isolated_steps,
        aligned_df,
        filter_flags=False,
    )

    # Filter to valid fits with all metrics present
    valid_fits = fitted_steps.filter(
        ~pl.col('dead_time_ms').is_nan() &
        ~pl.col('rise_time_ms').is_nan() &
        ~pl.col('wn_rad_s').is_nan() &
        ~pl.col('zeta').is_nan()
    )

    # Further filter to magnitude ≥ 1.0° to avoid encoder noise
    valid_fits = valid_fits.filter(pl.col('magnitude').abs() >= 1.0)

    # Split by direction ("up" or "down" strings in 'direction' column)
    up_steps = valid_fits.filter(pl.col('direction') == "up")
    down_steps = valid_fits.filter(pl.col('direction') == "down")

    up_dead_time = np.median(up_steps['dead_time_ms'].to_numpy()) if len(up_steps) > 0 else np.nan
    down_dead_time = np.median(down_steps['dead_time_ms'].to_numpy()) if len(down_steps) > 0 else np.nan

    up_rise_time = np.median(up_steps['rise_time_ms'].to_numpy()) if len(up_steps) > 0 else np.nan
    down_rise_time = np.median(down_steps['rise_time_ms'].to_numpy()) if len(down_steps) > 0 else np.nan

    return {
        'up_dead_time_ms': up_dead_time,
        'down_dead_time_ms': down_dead_time,
        'up_rise_time_ms': up_rise_time,
        'down_rise_time_ms': down_rise_time,
        'n_up': len(up_steps),
        'n_down': len(down_steps),
    }


def check_tracking_asymmetry(axis: str, recording: dict):
    """Check tracking up vs down by computing net displacement direction."""
    cmd_key = f"{axis}_cmd"
    actual_key = f"{axis}_actual"
    cmd_df = recording[cmd_key]
    actual_df = recording[actual_key]

    aligned_df, metadata = align_signals(cmd_df, actual_df)

    classified = classify_commanded_changes(
        cmd_df,
        step_gap_ms=200.0,
        tracking_gap_ms=100.0,
        min_detection_deg=0.01,
    )

    # Extract tracking sequences
    fire_events = recording.get("fire")
    sequences = extract_tracking_sequences(classified, min_length=3, fire_events=fire_events)

    if len(sequences) == 0:
        return {
            'up_rms_deg': np.nan,
            'down_rms_deg': np.nan,
            'n_up': 0,
            'n_down': 0,
        }

    # Analyze sequences
    sequences_with_metrics = analyze_all_tracking_sequences(sequences, aligned_df)

    valid = sequences_with_metrics.filter(
        ~pl.col('rms_error').is_nan()
    )

    if len(valid) == 0:
        return {
            'up_rms_deg': np.nan,
            'down_rms_deg': np.nan,
            'n_up': 0,
            'n_down': 0,
        }

    # Compute net displacement for each sequence to determine direction
    net_displacements = []
    for row in valid.iter_rows(named=True):
        start_time = row['start_time']
        end_time = row['end_time']

        # Get commanded values at start and end of sequence
        mask = (aligned_df['time_s'] >= start_time) & (aligned_df['time_s'] <= end_time)
        window = aligned_df.filter(mask)

        if len(window) >= 2:
            commanded = window['commanded'].to_numpy()
            net_displacement = commanded[-1] - commanded[0]
        else:
            net_displacement = 0.0

        net_displacements.append(net_displacement)

    # Add net_displacement column
    valid = valid.with_columns(
        pl.Series(name='net_displacement', values=net_displacements)
    )

    # Split by net displacement direction (positive = up, negative = down)
    up_seqs = valid.filter(pl.col('net_displacement') > 0)
    down_seqs = valid.filter(pl.col('net_displacement') < 0)

    up_rms = up_seqs['rms_error'].median() if len(up_seqs) > 0 else np.nan
    down_rms = down_seqs['rms_error'].median() if len(down_seqs) > 0 else np.nan

    return {
        'up_rms_deg': up_rms,
        'down_rms_deg': down_rms,
        'n_up': len(up_seqs),
        'n_down': len(down_seqs),
    }


def main():
    """Pitch vs Yaw comparison."""
    data_path = Path("motor.rrd")

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    print("="*70)
    print("PHASE 8: PITCH vs YAW COMPARISON")
    print("="*70)

    recording = load_recording(data_path, verbose=False)

    print("\nComputing step-response metrics...")
    pitch_step = compute_step_metrics("pitch", recording)
    yaw_step = compute_step_metrics("yaw", recording)

    print("Computing tracking metrics...")
    pitch_track = compute_tracking_metrics("pitch", recording)
    yaw_track = compute_tracking_metrics("yaw", recording)

    print("Checking gravity asymmetry...")
    gravity = check_gravity_asymmetry(recording)

    print("Checking tracking asymmetry...")
    pitch_track_asym = check_tracking_asymmetry("pitch", recording)
    yaw_track_asym = check_tracking_asymmetry("yaw", recording)

    # ========== STEP-RESPONSE COMPARISON TABLE ==========
    print("\n" + "="*70)
    print("STEP-RESPONSE METRICS")
    print("="*70)
    print(f"\n{'Metric':<25} {'Pitch':<20} {'Yaw':<20} {'Ratio':<10}")
    print("-"*70)

    print(f"{'Dead time (ms)':<25} {pitch_step['dead_time_ms'][0]:6.1f} ± {pitch_step['dead_time_ms'][1]:4.1f}  {yaw_step['dead_time_ms'][0]:6.1f} ± {yaw_step['dead_time_ms'][1]:4.1f}  {pitch_step['dead_time_ms'][0] / yaw_step['dead_time_ms'][0]:6.2f}×")
    print(f"{'Rise time (ms)':<25} {pitch_step['rise_time_ms'][0]:6.1f} ± {pitch_step['rise_time_ms'][1]:4.1f}  {yaw_step['rise_time_ms'][0]:6.1f} ± {yaw_step['rise_time_ms'][1]:4.1f}  {pitch_step['rise_time_ms'][0] / yaw_step['rise_time_ms'][0]:6.2f}×")
    print(f"{'τ = 1/ωn (ms)':<25} {pitch_step['tau_ms'][0]:6.1f} ± {pitch_step['tau_ms'][1]:4.1f}  {yaw_step['tau_ms'][0]:6.1f} ± {yaw_step['tau_ms'][1]:4.1f}  {pitch_step['tau_ms'][0] / yaw_step['tau_ms'][0]:6.2f}×")
    print(f"{'ωn (rad/s)':<25} {pitch_step['omega_n'][0]:6.1f} ± {pitch_step['omega_n'][1]:4.1f}  {yaw_step['omega_n'][0]:6.1f} ± {yaw_step['omega_n'][1]:4.1f}  {yaw_step['omega_n'][0] / pitch_step['omega_n'][0]:6.2f}×")
    print(f"{'ζ (damping)':<25} {pitch_step['zeta'][0]:6.2f} ± {pitch_step['zeta'][1]:4.2f}  {yaw_step['zeta'][0]:6.2f} ± {yaw_step['zeta'][1]:4.2f}  {pitch_step['zeta'][0] / yaw_step['zeta'][0]:6.2f}×")
    print(f"{'Overshoot (deg)':<25} {pitch_step['overshoot_deg'][0]:6.3f} ± {pitch_step['overshoot_deg'][1]:5.3f}  {yaw_step['overshoot_deg'][0]:6.3f} ± {yaw_step['overshoot_deg'][1]:5.3f}  {'N/A':<10}")
    print(f"{'Settling time (ms)':<25} {pitch_step['settling_time_ms'][0]:6.1f} ± {pitch_step['settling_time_ms'][1]:4.1f}  {yaw_step['settling_time_ms'][0]:6.1f} ± {yaw_step['settling_time_ms'][1]:4.1f}  {pitch_step['settling_time_ms'][0] / yaw_step['settling_time_ms'][0]:6.2f}×")
    print(f"{'n (isolated steps)':<25} {pitch_step['n_steps']:<20} {yaw_step['n_steps']:<20} {'N/A':<10}")

    # ========== TRACKING COMPARISON TABLE ==========
    print("\n" + "="*70)
    print("TRACKING METRICS")
    print("="*70)
    print(f"\n{'Metric':<25} {'Pitch':<20} {'Yaw':<20} {'Ratio':<10}")
    print("-"*70)

    if not np.isnan(pitch_track['rms_error_deg']) and not np.isnan(yaw_track['rms_error_deg']):
        print(f"{'RMS error (deg)':<25} {pitch_track['rms_error_deg']:6.3f}          {yaw_track['rms_error_deg']:6.3f}          {pitch_track['rms_error_deg'] / yaw_track['rms_error_deg']:6.2f}×")
        print(f"{'RMS clean (deg)':<25} {pitch_track['rms_clean_deg']:6.3f}          {yaw_track['rms_clean_deg']:6.3f}          {pitch_track['rms_clean_deg'] / yaw_track['rms_clean_deg']:6.2f}×")
        print(f"{'RMS recoil (deg)':<25} {pitch_track['rms_recoil_deg']:6.3f}          {yaw_track['rms_recoil_deg']:6.3f}          {pitch_track['rms_recoil_deg'] / yaw_track['rms_recoil_deg']:6.2f}×")
        print(f"{'Bandwidth (Hz)':<25} {pitch_track['bandwidth_hz']:6.1f}          {yaw_track['bandwidth_hz']:6.1f}          {pitch_track['bandwidth_hz'] / yaw_track['bandwidth_hz']:6.2f}×")
    else:
        print("  No tracking data available")

    print(f"{'n (tracking sequences)':<25} {pitch_track['n_samples']:<20} {yaw_track['n_samples']:<20} {'N/A':<10}")

    # ========== GRAVITY ASYMMETRY ==========
    print("\n" + "="*70)
    print("GRAVITY ASYMMETRY (Pitch only)")
    print("="*70)
    print(f"\n{'Metric':<25} {'Up-slew':<20} {'Down-slew':<20} {'Δ (ms)':<10}")
    print("-"*70)

    if not np.isnan(gravity['up_dead_time_ms']) and not np.isnan(gravity['down_dead_time_ms']):
        dead_delta = gravity['down_dead_time_ms'] - gravity['up_dead_time_ms']
        rise_delta = gravity['down_rise_time_ms'] - gravity['up_rise_time_ms']

        print(f"{'Dead time (ms)':<25} {gravity['up_dead_time_ms']:6.1f}          {gravity['down_dead_time_ms']:6.1f}          {dead_delta:+6.1f}")
        print(f"{'Rise time (ms)':<25} {gravity['up_rise_time_ms']:6.1f}          {gravity['down_rise_time_ms']:6.1f}          {rise_delta:+6.1f}")
        print(f"{'n (steps)':<25} {gravity['n_up']:<20} {gravity['n_down']:<20} {'N/A':<10}")

        print(f"\nInterpretation: Down-slew dead time is {abs(dead_delta):.1f} ms {'longer' if dead_delta > 0 else 'shorter'} than up-slew.")
        print(f"                This is {'consistent' if abs(dead_delta) < 5 else 'inconsistent'} with gravity-free operation (expect Δ ≈ 0).")
    else:
        print(f"{'Dead time (ms)':<25} {gravity['up_dead_time_ms']:6.1f}          {'N/A':<20} {'N/A':<10}")
        print(f"{'Rise time (ms)':<25} {gravity['up_rise_time_ms']:6.1f}          {'N/A':<20} {'N/A':<10}")
        print(f"{'n (steps)':<25} {gravity['n_up']:<20} {gravity['n_down']:<20} {'N/A':<10}")
        print("\nInterpretation: No down-slew steps found (magnitude is unsigned in catalog)")

    # ========== TRACKING ASYMMETRY ==========
    print("\n" + "="*70)
    print("TRACKING ASYMMETRY (by velocity direction)")
    print("="*70)
    print(f"\n{'Axis':<15} {'Up RMS (deg)':<20} {'Down RMS (deg)':<20} {'Ratio':<10}")
    print("-"*70)

    if not np.isnan(pitch_track_asym['up_rms_deg']) and not np.isnan(pitch_track_asym['down_rms_deg']):
        print(f"{'Pitch':<15} {pitch_track_asym['up_rms_deg']:6.3f}          {pitch_track_asym['down_rms_deg']:6.3f}          {pitch_track_asym['up_rms_deg'] / pitch_track_asym['down_rms_deg']:6.2f}×")
    else:
        print(f"{'Pitch':<15} {'N/A':<20} {'N/A':<20} {'N/A':<10}")

    if not np.isnan(yaw_track_asym['up_rms_deg']) and not np.isnan(yaw_track_asym['down_rms_deg']):
        print(f"{'Yaw':<15} {yaw_track_asym['up_rms_deg']:6.3f}          {yaw_track_asym['down_rms_deg']:6.3f}          {yaw_track_asym['up_rms_deg'] / yaw_track_asym['down_rms_deg']:6.2f}×")
    else:
        print(f"{'Yaw':<15} {'N/A':<20} {'N/A':<20} {'N/A':<10}")

    # ========== MOI RATIO INTERPRETATION ==========
    print("\n" + "="*70)
    print("EFFECTIVE τ RATIO (from step response)")
    print("="*70)

    tau_ratio = pitch_step['tau_ms'][0] / yaw_step['tau_ms'][0]

    print(f"\nτ_pitch / τ_yaw = {tau_ratio:.2f}×")
    print(f"\nThis is an *effective* ratio reflecting:")
    print(f"  • Moment of inertia (MOI) ratio")
    print(f"  • Gear ratio differences")
    print(f"  • Motor torque/tuning differences")
    print(f"\nIt is NOT a clean MOI measurement.")
    print(f"\nFor reference:")
    print(f"  • τ = J / (K_motor × gear_ratio²)")
    print(f"  • Identical slopes (11.1 vs 11.4 °/s per degree) suggest similar velocity-loop tuning")
    print(f"  • Yaw reaches 1.41× higher velocities on equivalent commands")
    print(f"  • This {tau_ratio:.2f}× difference could arise from gearing, MOI, or both")

    print("\n" + "="*70)
    print("Comparison complete!")
    print("="*70)


if __name__ == "__main__":
    main()
