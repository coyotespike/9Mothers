"""
Firing Disturbance Analysis

Analyzes motor response to firing impulse - the cleanest experiment in the dataset:
- Recoil impulse is roughly known (shotgun geometry)
- Motor responds to mechanical disturbance, not commanded signal
- Ring-down reveals structural ωn and ζ independent of control-loop tuning

Key questions:
1. Does pitch ring-down ωn match pitch closed-loop ωn?
2. Is there yaw deflection during firing? (tests "primarily pitch" prior)
3. Can the system sustain burst-fire cadence (<300ms recovery)?

Partitions:
- Isolated shots (>5s gaps): clean ring-down for structural parameters
- Burst shots (<3s gaps): operationally critical, sustained-fire scenario

Usage:
    python scripts/phase9_disturbance.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from turret_analysis import load_recording, align_signals


def fit_damped_sinusoid(time_rel: np.ndarray, deflection: np.ndarray):
    """
    Fit damped sinusoid to ring-down to extract structural ωn and ζ.

    Model: y(t) = A × exp(-ζ×ωn×t) × cos(ωd×t + φ) + offset
    where ωd = ωn × sqrt(1 - ζ²)
    """
    if len(time_rel) < 10:
        return {
            'wn_rad_s': np.nan,
            'zeta': np.nan,
            'amplitude': np.nan,
            'r_squared': np.nan,
        }

    # Initial parameter estimation
    # Amplitude: max deflection
    A_init = np.max(np.abs(deflection))

    # Frequency: from FFT
    dt = np.median(np.diff(time_rel))
    freqs = np.fft.rfftfreq(len(deflection), dt)
    fft_mag = np.abs(np.fft.rfft(deflection - np.mean(deflection)))
    if len(fft_mag) > 1:
        peak_idx = np.argmax(fft_mag[1:]) + 1  # Skip DC
        wd_init = 2 * np.pi * freqs[peak_idx]
    else:
        wd_init = 2 * np.pi * 5  # Guess 5 Hz

    # Damping: estimate from envelope decay
    envelope = np.abs(signal.hilbert(deflection - np.mean(deflection)))
    if len(envelope) > 2:
        try:
            # Fit exponential decay to envelope
            popt, _ = curve_fit(
                lambda t, a, b: a * np.exp(b * t),
                time_rel,
                envelope,
                p0=[A_init, -5.0],
                maxfev=1000
            )
            decay_rate = -popt[1]
            zeta_init = decay_rate / np.sqrt(wd_init**2 + decay_rate**2)
        except:
            zeta_init = 0.1
    else:
        zeta_init = 0.1

    # Constrain to reasonable ranges
    zeta_init = np.clip(zeta_init, 0.01, 2.0)
    wn_init = wd_init / np.sqrt(1 - zeta_init**2) if zeta_init < 1 else wd_init

    # Fit damped sinusoid
    def damped_sinusoid(t, A, wn, zeta, phi, offset):
        wd = wn * np.sqrt(1 - zeta**2) if zeta < 1 else 0.1
        return A * np.exp(-zeta * wn * t) * np.cos(wd * t + phi) + offset

    try:
        popt, _ = curve_fit(
            damped_sinusoid,
            time_rel,
            deflection,
            p0=[A_init, wn_init, zeta_init, 0.0, np.mean(deflection)],
            bounds=(
                [0, 1, 0.001, -np.pi, -10],
                [100, 500, 2.0, np.pi, 10]
            ),
            maxfev=5000
        )

        A, wn, zeta, phi, offset = popt

        # Compute R²
        predicted = damped_sinusoid(time_rel, *popt)
        ss_res = np.sum((deflection - predicted)**2)
        ss_tot = np.sum((deflection - np.mean(deflection))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            'wn_rad_s': wn,
            'zeta': zeta,
            'amplitude': A,
            'r_squared': r_squared,
        }
    except:
        return {
            'wn_rad_s': np.nan,
            'zeta': np.nan,
            'amplitude': np.nan,
            'r_squared': np.nan,
        }


def compute_population_median_structural_fit(
    fire_times: list,
    aligned_df: pl.DataFrame,
    isolated_results: pl.DataFrame,
    axis: str,
):
    """
    Compute structural parameters from population-median deflection trajectory.

    Instead of fitting each shot individually and averaging (high uncertainty),
    we align all isolated shots, compute the median deflection at each time point,
    and fit a single damped sinusoid to the median trajectory.

    This gives one clean structural ωn and ζ per axis with much lower uncertainty.

    Args:
        fire_times: List of isolated fire event times
        aligned_df: Aligned signal DataFrame
        isolated_results: DataFrame of isolated shot results
        axis: "pitch" or "yaw"

    Returns:
        Dictionary with structural parameters from population fit
    """
    # Extract deflection trajectories for all isolated shots
    trajectories = []

    for fire_time in fire_times:
        # Extract post-fire window
        post_mask = (
            (aligned_df['time_s'] >= fire_time) &
            (aligned_df['time_s'] <= fire_time + 1.0)  # 1.0s window for ring-down
        )
        post_window = aligned_df.filter(post_mask)

        if len(post_window) < 20:
            continue

        # Pre-fire baseline
        pre_mask = (
            (aligned_df['time_s'] >= fire_time - 0.2) &
            (aligned_df['time_s'] < fire_time)
        )
        pre_window = aligned_df.filter(pre_mask)

        if len(pre_window) < 5:
            continue

        baseline_actual = np.mean(pre_window['actual'].to_numpy())

        # Compute deflection trajectory
        time_rel = post_window['time_s'].to_numpy() - fire_time
        deflection = post_window['actual'].to_numpy() - baseline_actual

        trajectories.append({
            'time_rel': time_rel,
            'deflection': deflection,
        })

    if len(trajectories) < 10:
        print(f"  ⚠ Too few trajectories ({len(trajectories)}) for population fit")
        return None

    # Create common time grid (0 to 1.0s, 500 Hz)
    time_grid = np.linspace(0, 1.0, 500)

    # Interpolate all trajectories onto common grid
    interpolated_deflections = []

    for traj in trajectories:
        # Only interpolate if trajectory covers enough time
        if traj['time_rel'][-1] < 0.5:
            continue

        # Interpolate onto grid
        interp_func = interp1d(
            traj['time_rel'],
            traj['deflection'],
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )

        deflection_interp = interp_func(time_grid)
        interpolated_deflections.append(deflection_interp)

    if len(interpolated_deflections) < 10:
        print(f"  ⚠ Too few valid interpolations ({len(interpolated_deflections)}) for population fit")
        return None

    # Compute median deflection at each time point
    deflection_array = np.array(interpolated_deflections)
    median_deflection = np.nanmedian(deflection_array, axis=0)

    # Filter out NaN values (from interpolation extrapolation)
    valid_mask = ~np.isnan(median_deflection)
    time_grid_valid = time_grid[valid_mask]
    median_deflection_valid = median_deflection[valid_mask]

    if len(time_grid_valid) < 50:
        print(f"  ⚠ Too few valid points ({len(time_grid_valid)}) for population fit")
        return None

    # Fit damped sinusoid to median trajectory
    fit_result = fit_damped_sinusoid(time_grid_valid, median_deflection_valid)

    if not np.isnan(fit_result['wn_rad_s']):
        print(f"\n  Population-median structural fit ({axis}):")
        print(f"    Used {len(interpolated_deflections)} isolated shots")
        print(f"    Structural ωn: {fit_result['wn_rad_s']:.1f} rad/s")
        print(f"    Structural ζ: {fit_result['zeta']:.3f}")
        print(f"    R²: {fit_result['r_squared']:.3f}")

    return fit_result


def analyze_fire_event(
    fire_time: float,
    aligned_df: pl.DataFrame,
    axis: str,
    pre_window_s: float = 0.2,
    post_window_s: float = 2.5,  # Widened from 1.0s to 2.5s to capture full recovery
    baseline_rms: float = 1.0,
):
    """
    Analyze single firing event.

    Args:
        fire_time: Time of firing event
        aligned_df: Aligned signal DataFrame
        axis: "pitch" or "yaw"
        pre_window_s: Pre-fire baseline window
        post_window_s: Post-fire analysis window
        baseline_rms: Pre-fire tracking RMS (for recovery threshold)

    Returns:
        Dictionary of disturbance metrics
    """
    # Extract pre-fire baseline
    pre_mask = (
        (aligned_df['time_s'] >= fire_time - pre_window_s) &
        (aligned_df['time_s'] < fire_time)
    )
    pre_window = aligned_df.filter(pre_mask)

    # Extract post-fire window
    post_mask = (
        (aligned_df['time_s'] >= fire_time) &
        (aligned_df['time_s'] <= fire_time + post_window_s)
    )
    post_window = aligned_df.filter(post_mask)

    if len(pre_window) < 5 or len(post_window) < 10:
        return None

    # Baseline (pre-fire)
    pre_commanded = pre_window['commanded'].to_numpy()
    pre_actual = pre_window['actual'].to_numpy()
    baseline_actual = np.mean(pre_actual)  # Pre-fire actual position
    baseline_error = pre_actual - pre_commanded
    baseline_error_rms = np.sqrt(np.mean(baseline_error**2))

    # Post-fire response
    post_time = post_window['time_s'].to_numpy()
    post_commanded = post_window['commanded'].to_numpy()
    post_actual = post_window['actual'].to_numpy()

    # Deflection = deviation from pre-fire actual position
    # This is pure mechanical disturbance, ignoring commanded motion during recoil
    deflection = post_actual - baseline_actual

    # Peak deflection (look in first 200ms where disturbance is strongest)
    early_mask = (post_time - fire_time) <= 0.2
    if np.sum(early_mask) > 0:
        early_deflection = deflection[early_mask]
        early_time = post_time[early_mask]

        peak_deflection = np.max(np.abs(early_deflection))
        peak_time_idx = np.argmax(np.abs(early_deflection))
        peak_time = early_time[peak_time_idx] - fire_time
    else:
        peak_deflection = np.max(np.abs(deflection))
        peak_time_idx = np.argmax(np.abs(deflection))
        peak_time = post_time[peak_time_idx] - fire_time if peak_time_idx < len(post_time) else np.nan

    # Ring-down fit (first 0.5s for cleaner fit)
    ringdown_mask = (post_time - fire_time) <= 0.5
    if np.sum(ringdown_mask) >= 10:
        ringdown_time = post_time[ringdown_mask] - fire_time
        ringdown_deflection = deflection[ringdown_mask]

        ringdown_fit = fit_damped_sinusoid(ringdown_time, ringdown_deflection)
    else:
        ringdown_fit = {
            'wn_rad_s': np.nan,
            'zeta': np.nan,
            'amplitude': np.nan,
            'r_squared': np.nan,
        }

    # Recovery time: time until disturbance response decays to small fraction of peak
    # Use 10% of peak deflection as recovery threshold (more sensitive than baseline RMS)
    post_error = post_actual - post_commanded

    recovery_threshold = max(0.1 * peak_deflection, 0.1)  # At least 0.1°
    recovered_idx = None

    # Start looking after peak (no point checking before disturbance peaks)
    start_idx = peak_time_idx + 1 if peak_time_idx < len(post_time) - 1 else 0

    for i in range(start_idx, len(post_error)):
        if i + 5 >= len(post_error):
            break
        # Check if next 50ms (assume ~100Hz = 5 samples) stays below threshold
        window_error = post_error[i:i+5]
        if np.all(np.abs(window_error) <= recovery_threshold):
            recovered_idx = i
            break

    if recovered_idx is not None:
        recovery_time = post_time[recovered_idx] - fire_time
    else:
        recovery_time = np.nan  # Did not recover within window

    return {
        'fire_time': fire_time,
        'peak_deflection_deg': peak_deflection,
        'peak_time_ms': peak_time * 1000 if not np.isnan(peak_time) else np.nan,
        'ringdown_wn_rad_s': ringdown_fit['wn_rad_s'],
        'ringdown_zeta': ringdown_fit['zeta'],
        'ringdown_amplitude_deg': ringdown_fit['amplitude'],
        'ringdown_r_squared': ringdown_fit['r_squared'],
        'recovery_time_ms': recovery_time * 1000 if not np.isnan(recovery_time) else np.nan,
        'baseline_error_rms_deg': baseline_error_rms,
    }


def analyze_axis_disturbance(axis: str, recording: dict, baseline_rms: float):
    """Analyze firing disturbance for one axis."""
    print(f"\n{'='*70}")
    print(f"Analyzing {axis.upper()} firing disturbance")
    print(f"{'='*70}")

    cmd_key = f"{axis}_cmd"
    actual_key = f"{axis}_actual"
    cmd_df = recording[cmd_key]
    actual_df = recording[actual_key]

    aligned_df, metadata = align_signals(cmd_df, actual_df)

    fire_events = recording.get("fire")
    if fire_events is None or len(fire_events) == 0:
        print("No fire events found!")
        return None

    fire_times = fire_events['time_s'].to_numpy()
    print(f"  Total fire events: {len(fire_times)}")

    # Classify shots: isolated vs burst
    time_since_last = np.diff(np.concatenate([[0], fire_times]))
    time_until_next = np.diff(np.concatenate([fire_times, [1e9]]))

    isolated_mask = (time_since_last > 5.0) & (time_until_next > 5.0)
    burst_mask = (time_since_last < 3.0) | (time_until_next < 3.0)

    print(f"  Isolated shots (>5s gaps): {np.sum(isolated_mask)}")
    print(f"  Burst shots (<3s gaps): {np.sum(burst_mask)}")

    # Analyze all fire events
    results = []
    for i, fire_time in enumerate(fire_times):
        result = analyze_fire_event(
            fire_time,
            aligned_df,
            axis,
            baseline_rms=baseline_rms
        )
        if result is not None:
            result['shot_type'] = 'isolated' if isolated_mask[i] else 'burst'
            results.append(result)

    if len(results) == 0:
        print("No valid disturbance analyses!")
        return None

    results_df = pl.DataFrame(results)

    # Partition by shot type
    isolated_results = results_df.filter(pl.col('shot_type') == 'isolated')
    burst_results = results_df.filter(pl.col('shot_type') == 'burst')

    # Recovery statistics - handle right-censored data (NaN = didn't recover within window)
    recovery_all = results_df['recovery_time_ms'].to_numpy()
    recovery_iso = isolated_results['recovery_time_ms'].to_numpy()
    recovery_burst = burst_results['recovery_time_ms'].to_numpy()

    # Count censored (NaN) vs recovered
    n_censored_all = np.sum(np.isnan(recovery_all))
    n_censored_iso = np.sum(np.isnan(recovery_iso))
    n_censored_burst = np.sum(np.isnan(recovery_burst))
    n_recovered_all = len(recovery_all) - n_censored_all
    n_recovered_iso = len(recovery_iso) - n_censored_iso

    # For recovered shots only, get statistics
    recovery_all_valid = recovery_all[~np.isnan(recovery_all)]
    recovery_iso_valid = recovery_iso[~np.isnan(recovery_iso)]
    recovery_burst_valid = recovery_burst[~np.isnan(recovery_burst)]

    print(f"\nRecovery statistics ({axis}):")
    print(f"  Total shots: {len(results_df)}")
    print(f"  Recovered within 2.5s: {n_recovered_all} ({n_recovered_all/len(results_df)*100:.1f}%)")
    print(f"  Right-censored (>2.5s): {n_censored_all} ({n_censored_all/len(results_df)*100:.1f}%)")
    print(f"  Isolated recovered: {n_recovered_iso}/{len(isolated_results)} ({n_recovered_iso/len(isolated_results)*100:.1f}%)")

    # Compute population-median structural fit for isolated shots
    isolated_fire_times = fire_times[isolated_mask]
    population_fit = compute_population_median_structural_fit(
        isolated_fire_times,
        aligned_df,
        isolated_results,
        axis
    )

    return {
        'axis': axis,
        'all_results': results_df,
        'isolated_results': isolated_results,
        'burst_results': burst_results,
        'recovery_all_valid': recovery_all_valid,
        'recovery_iso_valid': recovery_iso_valid,
        'recovery_burst_valid': recovery_burst_valid,
        'n_censored_all': n_censored_all,
        'n_censored_iso': n_censored_iso,
        'n_censored_burst': n_censored_burst,
        'population_fit': population_fit,
    }


def main():
    """Firing disturbance analysis."""
    data_path = Path("motor.rrd")

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    print("="*70)
    print("FIRING DISTURBANCE ANALYSIS")
    print("="*70)
    print("\nThe cleanest experiment in the dataset:")
    print("  - Recoil impulse is known (shotgun geometry)")
    print("  - Motor responds to mechanical disturbance")
    print("  - Ring-down reveals structural ωn and ζ")

    recording = load_recording(data_path, verbose=False)

    # Use Phase 8 baseline RMS values for recovery threshold
    # From  pitch clean RMS = 1.214°, yaw clean RMS = 0.890°
    pitch_baseline_rms = 1.214
    yaw_baseline_rms = 0.890

    # Analyze pitch
    pitch_results = analyze_axis_disturbance("pitch", recording, pitch_baseline_rms)

    # Analyze yaw
    yaw_results = analyze_axis_disturbance("yaw", recording, yaw_baseline_rms)

    # Report results
    print("\n" + "="*70)
    print("ISOLATED SHOTS ANALYSIS (Clean Ring-Down)")
    print("="*70)

    if pitch_results and len(pitch_results['isolated_results']) > 0:
        p_iso = pitch_results['isolated_results']

        print(f"\nPITCH (n={len(p_iso)} isolated shots):")
        print(f"  Peak deflection: {p_iso['peak_deflection_deg'].median():.3f} ± {p_iso['peak_deflection_deg'].std():.3f}°")
        print(f"  Peak time: {p_iso['peak_time_ms'].median():.1f} ms")

        # Ring-down parameters - show both per-shot and population fits
        valid_fits = p_iso.filter(~pl.col('ringdown_wn_rad_s').is_nan())
        if len(valid_fits) > 0:
            print(f"  Per-shot structural fits (median ± std):")
            print(f"    ωn: {valid_fits['ringdown_wn_rad_s'].median():.1f} ± {valid_fits['ringdown_wn_rad_s'].std():.1f} rad/s")
            print(f"    ζ: {valid_fits['ringdown_zeta'].median():.3f} ± {valid_fits['ringdown_zeta'].std():.3f}")

        # Population-median fit (single fit to median trajectory)
        if pitch_results['population_fit'] is not None:
            pop_fit = pitch_results['population_fit']
            print(f"  Population-median structural fit:")
            print(f"    ωn: {pop_fit['wn_rad_s']:.1f} rad/s")
            print(f"    ζ: {pop_fit['zeta']:.3f}")
            print(f"    R²: {pop_fit['r_squared']:.3f}")
            print(f"  (From  closed-loop ωn = 31.5 rad/s, ζ = 1.04)")

        if len(pitch_results['recovery_iso_valid']) > 0:
            recovery_median = np.median(pitch_results['recovery_iso_valid'])
            recovery_std = np.std(pitch_results['recovery_iso_valid'])
            censored_pct = pitch_results['n_censored_iso'] / len(p_iso) * 100
            print(f"  Recovery time: {recovery_median:.1f} ± {recovery_std:.1f} ms (n={len(pitch_results['recovery_iso_valid'])})")
            if censored_pct > 0:
                print(f"    {censored_pct:.0f}% of shots did not recover within 2.5s")
        else:
            print(f"  Recovery time: >2500 ms (all isolated shots exceeded window)")

    if yaw_results and len(yaw_results['isolated_results']) > 0:
        y_iso = yaw_results['isolated_results']

        print(f"\nYAW (n={len(y_iso)} isolated shots):")
        print(f"  Peak deflection: {y_iso['peak_deflection_deg'].median():.3f} ± {y_iso['peak_deflection_deg'].std():.3f}°")

        if y_iso['peak_deflection_deg'].median() > 0.1:
            print(f"  ⚠ Yaw shows measurable deflection despite 'primarily pitch' prior")
            print(f"  → Recoil impulse not perfectly axial through yaw pivot")
        else:
            print(f"  ✓ Yaw deflection minimal (consistent with prior)")

        print(f"  Peak time: {y_iso['peak_time_ms'].median():.1f} ms")

        valid_fits = y_iso.filter(~pl.col('ringdown_wn_rad_s').is_nan())
        if len(valid_fits) > 0:
            print(f"  Per-shot structural fits (median ± std):")
            print(f"    ωn: {valid_fits['ringdown_wn_rad_s'].median():.1f} ± {valid_fits['ringdown_wn_rad_s'].std():.1f} rad/s")
            print(f"    ζ: {valid_fits['ringdown_zeta'].median():.3f} ± {valid_fits['ringdown_zeta'].std():.3f}")

        # Population-median fit
        if yaw_results['population_fit'] is not None:
            pop_fit = yaw_results['population_fit']
            print(f"  Population-median structural fit:")
            print(f"    ωn: {pop_fit['wn_rad_s']:.1f} rad/s")
            print(f"    ζ: {pop_fit['zeta']:.3f}")
            print(f"    R²: {pop_fit['r_squared']:.3f}")
            print(f"  (From  closed-loop ωn = 49.3 rad/s, ζ = 0.87)")

        if len(yaw_results['recovery_iso_valid']) > 0:
            recovery_median = np.median(yaw_results['recovery_iso_valid'])
            recovery_std = np.std(yaw_results['recovery_iso_valid'])
            censored_pct = yaw_results['n_censored_iso'] / len(y_iso) * 100
            print(f"  Recovery time: {recovery_median:.1f} ± {recovery_std:.1f} ms (n={len(yaw_results['recovery_iso_valid'])})")
            if censored_pct > 0:
                print(f"    {censored_pct:.0f}% of shots did not recover within 2.5s")
        else:
            print(f"  Recovery time: >2500 ms (all isolated shots exceeded window)")

    # Burst shots analysis
    print("\n" + "="*70)
    print("BURST SHOTS ANALYSIS (Sustained-Fire Scenario)")
    print("="*70)

    if pitch_results and len(pitch_results['burst_results']) > 0:
        p_burst = pitch_results['burst_results']

        print(f"\nPITCH (n={len(p_burst)} burst shots):")
        print(f"  Peak deflection: {p_burst['peak_deflection_deg'].median():.3f}°")

        if len(pitch_results['recovery_burst_valid']) > 0:
            recovery_median = np.median(pitch_results['recovery_burst_valid'])
            censored_pct = pitch_results['n_censored_burst'] / len(p_burst) * 100
            print(f"  Recovery time: {recovery_median:.1f} ms (n={len(pitch_results['recovery_burst_valid'])})")
            if censored_pct > 0:
                print(f"    {censored_pct:.0f}% of shots did not recover within 2.5s")
        else:
            print(f"  Recovery time: >2500 ms (all burst shots exceeded window)")

    if yaw_results and len(yaw_results['burst_results']) > 0:
        y_burst = yaw_results['burst_results']

        print(f"\nYAW (n={len(y_burst)} burst shots):")
        print(f"  Peak deflection: {y_burst['peak_deflection_deg'].median():.3f}°")

        if len(yaw_results['recovery_burst_valid']) > 0:
            recovery_median = np.median(yaw_results['recovery_burst_valid'])
            censored_pct = yaw_results['n_censored_burst'] / len(y_burst) * 100
            print(f"  Recovery time: {recovery_median:.1f} ms (n={len(yaw_results['recovery_burst_valid'])})")
            if censored_pct > 0:
                print(f"    {censored_pct:.0f}% of shots did not recover within 2.5s")
        else:
            print(f"  Recovery time: >2500 ms (all burst shots exceeded window)")

    # Rapid-fire capability assessment
    print("\n" + "="*70)
    print("RAPID-FIRE CAPABILITY (300ms Budget)")
    print("="*70)

    if pitch_results and len(pitch_results['recovery_all_valid']) > 0:
        median_recovery = np.median(pitch_results['recovery_all_valid'])
        censored_pct = pitch_results['n_censored_all'] / len(pitch_results['all_results']) * 100

        print(f"\nPITCH:")
        print(f"  Median recovery time: {median_recovery:.1f} ms (n={len(pitch_results['recovery_all_valid'])})")
        if censored_pct > 0:
            print(f"  {censored_pct:.0f}% of shots did not recover within 2.5s")
        print(f"  Rapid-fire budget: 300 ms")

        if median_recovery < 300:
            margin = 300 - median_recovery
            print(f"  ✓ System CAN sustain burst cadence (margin: {margin:.1f} ms)")
        else:
            deficit = median_recovery - 300
            print(f"  ✗ System CANNOT sustain burst cadence (deficit: {deficit:.1f} ms)")
            if censored_pct > 10:
                print(f"  ⚠ {censored_pct:.0f}% of shots don't recover within observation window")
                print(f"     Actual deficit may be larger than {deficit:.1f} ms")

    if yaw_results and len(yaw_results['recovery_all_valid']) > 0:
        median_recovery = np.median(yaw_results['recovery_all_valid'])
        censored_pct = yaw_results['n_censored_all'] / len(yaw_results['all_results']) * 100

        print(f"\nYAW:")
        print(f"  Median recovery time: {median_recovery:.1f} ms (n={len(yaw_results['recovery_all_valid'])})")
        if censored_pct > 0:
            print(f"  {censored_pct:.0f}% of shots did not recover within 2.5s")
        print(f"  Rapid-fire budget: 300 ms")

        if median_recovery < 300:
            margin = 300 - median_recovery
            print(f"  ✓ System CAN sustain burst cadence (margin: {margin:.1f} ms)")
        else:
            deficit = median_recovery - 300
            print(f"  ✗ System CANNOT sustain burst cadence (deficit: {deficit:.1f} ms)")
            if censored_pct > 10:
                print(f"  ⚠ {censored_pct:.0f}% of shots don't recover within observation window")
                print(f"     Actual deficit may be larger than {deficit:.1f} ms")

    # Shot-to-shot variance check
    print("\n" + "="*70)
    print("SHOT-TO-SHOT VARIANCE")
    print("="*70)

    if pitch_results and len(pitch_results['isolated_results']) > 0:
        p_iso = pitch_results['isolated_results']
        cv = p_iso['peak_deflection_deg'].std() / p_iso['peak_deflection_deg'].median()

        print(f"\nPITCH isolated shots:")
        print(f"  Coefficient of variation: {cv:.2f} ({cv*100:.1f}%)")

        if cv > 0.3:
            print(f"  ⚠ High variance (>30%) - investigate ammunition, thermal, or mechanical wear")
        else:
            print(f"  ✓ Consistent shot-to-shot response")

    if yaw_results and len(yaw_results['isolated_results']) > 0:
        y_iso = yaw_results['isolated_results']
        cv = y_iso['peak_deflection_deg'].std() / y_iso['peak_deflection_deg'].median()

        print(f"\nYAW isolated shots:")
        print(f"  Coefficient of variation: {cv:.2f} ({cv*100:.1f}%)")

        if cv > 0.3:
            print(f"  ⚠ High variance (>30%) - investigate cause")
        else:
            print(f"  ✓ Consistent shot-to-shot response")

    # Prior expectation check
    print("\n" + "="*70)
    print("PRIOR EXPECTATION CHECK")
    print("="*70)

    if pitch_results and len(pitch_results['all_results']) > 0:
        observed = pitch_results['all_results']['peak_deflection_deg'].median()
        expected = 0.5  # From Phase 0

        print(f"\nPitch deflection:")
        print(f"  Prior expectation: ~{expected}°")
        print(f"  Observed: {observed:.3f}°")
        print(f"  Ratio: {observed / expected:.2f}×")

        if observed / expected > 2 or observed / expected < 0.5:
            print(f"  ⚠ Differs by >2× from prior - warrants investigation")
        else:
            print(f"  ✓ Within expected range")

    print("\n" + "="*70)
    print("Comparison complete!")
    print("="*70)


if __name__ == "__main__":
    main()
