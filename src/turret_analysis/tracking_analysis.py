"""
Tracking performance analysis.

Analyzes continuous tracking behavior during rapid commanded updates.
Primary operational metric - this is how the turret performs 90% of the time.

Metrics:
- Tracking lag: Time delay between commanded and actual (cross-correlation)
- RMS tracking error: Position error during continuous tracking
- Bandwidth: Update rate where tracking degrades
- Recoil impact: Performance degradation during recovery
"""

from typing import Dict, Tuple, Optional
import numpy as np
import polars as pl
from scipy import signal


def compute_tracking_lag(
    time_s: np.ndarray,
    commanded: np.ndarray,
    actual: np.ndarray,
    max_lag_s: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute tracking lag using cross-correlation.

    Tracking lag is the time delay that maximizes correlation between
    commanded and actual signals. This captures control loop delay plus
    motor response lag.

    Args:
        time_s: Time vector (seconds)
        commanded: Commanded position
        actual: Actual position
        max_lag_s: Maximum lag to search (seconds)

    Returns:
        Tuple of (lag_s, correlation_coefficient)
    """
    # Ensure uniform sampling for cross-correlation
    dt = np.median(np.diff(time_s))

    # Compute cross-correlation
    correlation = signal.correlate(actual - actual.mean(),
                                   commanded - commanded.mean(),
                                   mode='same')

    # Normalize
    correlation = correlation / (np.std(actual) * np.std(commanded) * len(actual))

    # Find peak within max_lag window
    center_idx = len(correlation) // 2
    max_lag_samples = int(max_lag_s / dt)

    search_start = max(0, center_idx - max_lag_samples)
    search_end = min(len(correlation), center_idx + max_lag_samples)

    search_window = correlation[search_start:search_end]
    peak_idx = np.argmax(np.abs(search_window))

    # Convert to lag time
    lag_samples = peak_idx + search_start - center_idx
    lag_s = lag_samples * dt

    correlation_coef = correlation[search_start + peak_idx]

    return lag_s, correlation_coef


def analyze_tracking_sequence(
    sequence_row: dict,
    aligned_df: pl.DataFrame,
) -> Dict[str, float]:
    """
    Analyze a single tracking sequence.

    Args:
        sequence_row: Row from tracking sequence catalog
        aligned_df: Aligned signal DataFrame

    Returns:
        Dictionary of tracking metrics:
        - lag_ms: Tracking lag (milliseconds)
        - correlation: Cross-correlation coefficient
        - rms_error: RMS tracking error (degrees)
        - mean_error: Mean tracking error (degrees)
        - max_error: Maximum absolute error (degrees)
        - mean_abs_velocity: Mean absolute commanded velocity (deg/s)
        - rms_velocity: RMS commanded velocity (deg/s)

        Note: duration_s and update_rate_hz already exist in sequence_row, not duplicated
    """
    start_time = sequence_row['start_time']
    end_time = sequence_row['end_time']

    # Extract sequence window
    mask = (aligned_df['time_s'] >= start_time) & (aligned_df['time_s'] <= end_time)
    window = aligned_df.filter(mask)

    if len(window) < 10:
        return {
            'lag_ms': np.nan,
            'correlation': np.nan,
            'rms_error': np.nan,
            'mean_error': np.nan,
            'max_error': np.nan,
            'mean_abs_velocity': np.nan,
            'rms_velocity': np.nan,
        }

    time_s = window['time_s'].to_numpy()
    commanded = window['commanded'].to_numpy()
    actual = window['actual'].to_numpy()
    error = window['error'].to_numpy()

    # Tracking lag
    lag_s, correlation = compute_tracking_lag(time_s, commanded, actual, max_lag_s=0.5)

    # Error statistics
    rms_error = np.sqrt(np.mean(error**2))
    mean_error = np.mean(error)
    max_error = np.max(np.abs(error))

    # Commanded velocity (for error-velocity diagnostic)
    dt = np.diff(time_s)
    cmd_velocity = np.diff(commanded) / dt
    mean_abs_velocity = np.mean(np.abs(cmd_velocity))
    rms_velocity = np.sqrt(np.mean(cmd_velocity**2))

    # Return only new metrics (don't duplicate columns already in sequences DataFrame)
    return {
        'lag_ms': lag_s * 1000,
        'correlation': correlation,
        'rms_error': rms_error,
        'mean_error': mean_error,
        'max_error': max_error,
        'mean_abs_velocity': mean_abs_velocity,
        'rms_velocity': rms_velocity,
    }


def analyze_all_tracking_sequences(
    sequences: pl.DataFrame,
    aligned_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Analyze all tracking sequences.

    Args:
        sequences: Tracking sequence catalog from extract_tracking_sequences()
        aligned_df: Aligned signal DataFrame

    Returns:
        DataFrame with sequence metrics added
    """
    metrics_list = []

    for row in sequences.iter_rows(named=True):
        metrics = analyze_tracking_sequence(row, aligned_df)
        metrics_list.append(metrics)

    # Add metrics as new columns
    metrics_df = pl.DataFrame(metrics_list)

    # Combine with original sequence data
    result = pl.concat([sequences, metrics_df], how='horizontal')

    return result


def compare_tracking_performance(
    sequences_with_metrics: pl.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Compare tracking performance: clean vs recoil-contaminated.

    Args:
        sequences_with_metrics: Output from analyze_all_tracking_sequences()

    Returns:
        Dictionary with 'clean' and 'recoil' subdictionaries containing:
        - lag_ms: Median tracking lag
        - rms_error: Median RMS error
        - update_rate_hz: Median update rate
        - n_sequences: Number of sequences
        - total_duration_s: Total time in this regime
    """
    # Filter valid sequences (non-NaN metrics)
    valid = sequences_with_metrics.filter(
        ~pl.col('lag_ms').is_nan() &
        ~pl.col('rms_error').is_nan()
    )

    # Split by recoil flag
    clean = valid.filter(~pl.col('during_recoil'))
    recoil = valid.filter(pl.col('during_recoil'))

    def compute_stats(df: pl.DataFrame) -> Dict[str, float]:
        if len(df) == 0:
            return {
                'lag_ms': np.nan,
                'rms_error': np.nan,
                'update_rate_hz': np.nan,
                'n_sequences': 0,
                'total_duration_s': 0.0,
            }

        return {
            'lag_ms': df['lag_ms'].median(),
            'lag_ms_std': df['lag_ms'].std(),
            'rms_error': df['rms_error'].median(),
            'rms_error_std': df['rms_error'].std(),
            'max_error': df['max_error'].median(),
            'update_rate_hz': df['mean_update_rate_hz'].median(),
            'n_sequences': len(df),
            'total_duration_s': df['duration_s'].sum(),
        }

    return {
        'clean': compute_stats(clean),
        'recoil': compute_stats(recoil),
        'all': compute_stats(valid),
    }


def estimate_bandwidth(
    sequences_with_metrics: pl.DataFrame,
    error_threshold_deg: float = 0.5,
) -> Tuple[float, pl.DataFrame]:
    """
    Estimate tracking bandwidth: update rate where error exceeds threshold.

    Bins sequences by update rate and computes median error per bin.
    Bandwidth is the rate where error crosses threshold.

    Args:
        sequences_with_metrics: Output from analyze_all_tracking_sequences()
        error_threshold_deg: Error threshold defining "tracking failure"

    Returns:
        Tuple of:
        - bandwidth_hz: Estimated bandwidth (Hz)
        - binned_data: DataFrame with columns ['rate_bin_hz', 'median_error', 'n_sequences']
    """
    # Filter valid sequences
    valid = sequences_with_metrics.filter(
        ~pl.col('rms_error').is_nan() &
        (pl.col('mean_update_rate_hz') > 0)
    )

    if len(valid) < 10:
        return np.nan, pl.DataFrame()

    # Create rate bins
    rates = valid['mean_update_rate_hz'].to_numpy()
    errors = valid['rms_error'].to_numpy()

    # Bin by update rate (10 bins)
    n_bins = min(10, len(valid) // 3)
    bins = np.linspace(rates.min(), rates.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_indices = np.digitize(rates, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute median error per bin
    median_errors = []
    counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            median_errors.append(np.median(errors[mask]))
            counts.append(mask.sum())
        else:
            median_errors.append(np.nan)
            counts.append(0)

    # Find crossing point
    bandwidth_hz = np.nan
    for i in range(len(median_errors)):
        if not np.isnan(median_errors[i]) and median_errors[i] > error_threshold_deg:
            # Linear interpolation to find exact crossing
            if i > 0 and not np.isnan(median_errors[i-1]):
                # Interpolate between bin i-1 and i
                rate_low = bin_centers[i-1]
                rate_high = bin_centers[i]
                error_low = median_errors[i-1]
                error_high = median_errors[i]

                if error_high > error_low:  # Increasing error
                    frac = (error_threshold_deg - error_low) / (error_high - error_low)
                    bandwidth_hz = rate_low + frac * (rate_high - rate_low)
            else:
                bandwidth_hz = bin_centers[i]
            break

    # Build result DataFrame
    binned_data = pl.DataFrame({
        'rate_bin_hz': bin_centers,
        'median_error': median_errors,
        'n_sequences': counts,
    })

    return bandwidth_hz, binned_data


def diagnose_error_source(
    sequences_with_metrics: pl.DataFrame,
) -> Dict[str, any]:
    """
    Diagnose whether tracking error is from velocity lag, steady-state bias, or noise.

    Strategy:
    - Velocity lag: error correlates with commanded velocity (faster → bigger error)
    - Steady-state bias: error independent of velocity (constant offset)
    - Noise: error random, uncorrelated with velocity

    Args:
        sequences_with_metrics: Output from analyze_all_tracking_sequences()
                                Must include 'rms_error' and 'mean_abs_velocity' columns

    Returns:
        Dictionary with:
        - correlation: Pearson correlation between RMS error and velocity
        - p_value: Statistical significance (< 0.05 means significant correlation)
        - velocity_slope: Linear fit slope (deg error per deg/s velocity)
        - velocity_intercept: Linear fit intercept (deg error at zero velocity)
        - interpretation: Text description of dominant error source
    """
    # Filter valid sequences
    valid = sequences_with_metrics.filter(
        ~pl.col('rms_error').is_nan() &
        ~pl.col('mean_abs_velocity').is_nan() &
        (pl.col('mean_abs_velocity') > 0)
    )

    if len(valid) < 10:
        return {
            'correlation': np.nan,
            'p_value': np.nan,
            'velocity_slope': np.nan,
            'velocity_intercept': np.nan,
            'interpretation': 'Insufficient data for diagnosis (need ≥10 valid sequences)',
        }

    errors = valid['rms_error'].to_numpy()
    velocities = valid['mean_abs_velocity'].to_numpy()

    # Pearson correlation
    from scipy import stats
    correlation, p_value = stats.pearsonr(errors, velocities)

    # Linear fit: error = slope * velocity + intercept
    slope, intercept = np.polyfit(velocities, errors, 1)

    # Interpret results
    if p_value < 0.05:
        if correlation > 0.3:
            interpretation = (
                f"VELOCITY LAG DOMINANT: Error significantly correlates with velocity (r={correlation:.3f}, p={p_value:.4f}).\n"
                f"  Slope: {slope:.4f} deg error per deg/s velocity\n"
                f"  Intercept: {intercept:.3f}° (baseline error at zero velocity)\n"
                f"  → Motor cannot keep up with commanded velocity. Error grows linearly with speed.\n"
                f"  → Fix: Increase control loop gains, feedforward compensation, or reduce command rate."
            )
        elif correlation < -0.3:
            interpretation = (
                f"NEGATIVE CORRELATION: Error decreases with velocity (r={correlation:.3f}, p={p_value:.4f}).\n"
                f"  This is unusual and may indicate:\n"
                f"  - Higher velocities occur during cleaner operational conditions\n"
                f"  - Velocity-dependent damping or friction compensation\n"
                f"  → Review operational context of high-velocity sequences."
            )
        else:
            interpretation = (
                f"WEAK CORRELATION: Error shows weak correlation with velocity (r={correlation:.3f}, p={p_value:.4f}).\n"
                f"  → Error likely dominated by steady-state bias or noise, not velocity lag."
            )
    else:
        interpretation = (
            f"STEADY-STATE BIAS OR NOISE DOMINANT: Error uncorrelated with velocity (r={correlation:.3f}, p={p_value:.3f}).\n"
            f"  Mean error: {np.mean(errors):.3f}° ± {np.std(errors):.3f}°\n"
            f"  → Error does NOT grow with velocity. Motor keeps up with commanded speed.\n"
            f"  → Likely causes:\n"
            f"    - Constant position offset (lack of integral action, friction, gravity compensation)\n"
            f"    - Encoder quantization noise\n"
            f"    - Mechanical compliance/backlash\n"
            f"  → Fix: Add integral term, improve calibration, or filter noise."
        )

    return {
        'correlation': correlation,
        'p_value': p_value,
        'velocity_slope': slope,
        'velocity_intercept': intercept,
        'n_sequences': len(valid),
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'mean_velocity': np.mean(velocities),
        'std_velocity': np.std(velocities),
        'interpretation': interpretation,
    }
