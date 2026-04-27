"""
Signal alignment and error computation.

Handles resampling commanded and actual signals onto a common time grid
and computing tracking error. Deals with different sample rates and gaps.
"""

from typing import Dict, Tuple, Optional

import numpy as np
import polars as pl


def align_signals(
    commanded: pl.DataFrame,
    actual: pl.DataFrame,
    method: str = "previous",
    max_gap_s: float = 0.1,
) -> Tuple[pl.DataFrame, Dict[str, any]]:
    """
    Align commanded and actual signals onto actual's time grid.

    Uses actual's time grid as the reference (denser, continuous sampling).
    Interpolates commanded signal onto actual's timestamps.

    Default interpolation is 'previous' (zero-order hold) because commanded signals
    are piecewise constant (discrete steps), not continuous ramps. Linear interpolation
    would create artificial ramps between steps, distorting dead-time measurements.

    Interpolation flag now checks for actual encoder dropouts (gaps > 3× median or >100ms),
    not commanded sparsity. Commanded is event-driven (only logs on change), so sparse
    commanded logging is normal behavior, not a data quality issue.

    Args:
        commanded: DataFrame with columns ['time_s', 'value']
        actual: DataFrame with columns ['time_s', 'value']
        method: Interpolation method ('previous', 'linear', 'next')
                'previous' (default): zero-order hold, preserves step behavior
                'linear': creates ramps between samples (wrong for step commands)
        max_gap_s: Not used for interpolation detection (kept for API compatibility)

    Returns:
        Tuple of:
            - aligned_df: DataFrame with columns:
                ['time_s', 'commanded', 'actual', 'error', 'interpolated']
            - metadata: Dict with alignment statistics

    Raises:
        ValueError: If time ranges don't overlap or signals are empty
    """
    if commanded.height == 0 or actual.height == 0:
        raise ValueError("Input signals cannot be empty")

    # Check time range overlap
    cmd_min, cmd_max = commanded["time_s"].min(), commanded["time_s"].max()
    act_min, act_max = actual["time_s"].min(), actual["time_s"].max()

    if cmd_max < act_min or act_max < cmd_min:
        raise ValueError(
            f"Time ranges don't overlap: cmd=[{cmd_min:.3f}, {cmd_max:.3f}], "
            f"actual=[{act_min:.3f}, {act_max:.3f}]"
        )

    # Use actual's time grid as reference
    time_grid = actual["time_s"].to_numpy()
    actual_values = actual["value"].to_numpy()
    cmd_times = commanded["time_s"].to_numpy()
    cmd_values = commanded["value"].to_numpy()

    # Interpolate commanded onto actual's grid
    if method == "linear":
        # Linear interpolation - appropriate for commanded steps
        cmd_interpolated = np.interp(time_grid, cmd_times, cmd_values)
    elif method == "previous":
        # Zero-order hold - takes previous commanded value
        cmd_interpolated = np.interp(
            time_grid, cmd_times, cmd_values,
            left=cmd_values[0], right=cmd_values[-1]
        )
        # Manually implement previous-value logic
        indices = np.searchsorted(cmd_times, time_grid, side='right') - 1
        indices = np.clip(indices, 0, len(cmd_values) - 1)
        cmd_interpolated = cmd_values[indices]
    elif method == "next":
        # Takes next commanded value (rarely used)
        indices = np.searchsorted(cmd_times, time_grid, side='left')
        indices = np.clip(indices, 0, len(cmd_values) - 1)
        cmd_interpolated = cmd_values[indices]
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    # Compute error
    error = actual_values - cmd_interpolated

    # Detect interpolated regions (actual encoder dropouts, not commanded sparsity)
    # Check for gaps in actual data, not commanded (commanded is event-driven)
    actual_gaps = np.diff(time_grid)
    median_gap = np.median(actual_gaps)

    interpolated_mask = np.zeros(len(time_grid), dtype=bool)

    # Mark samples after large gaps in actual encoder data (>3× median = dropout)
    for i in range(1, len(actual_gaps)):
        if actual_gaps[i-1] > median_gap * 3:
            interpolated_mask[i] = True

    # Also mark if gap exceeds absolute threshold (100ms = real dropout)
    for i in range(1, len(actual_gaps)):
        if actual_gaps[i-1] > 0.1:  # 100ms gap in actual encoder
            interpolated_mask[i] = True

    # Create aligned DataFrame
    aligned_df = pl.DataFrame({
        "time_s": time_grid,
        "commanded": cmd_interpolated,
        "actual": actual_values,
        "error": error,
        "interpolated": interpolated_mask,
    })

    # Compute metadata
    metadata = {
        "num_samples": len(time_grid),
        "time_range_s": (time_grid[0], time_grid[-1]),
        "duration_s": time_grid[-1] - time_grid[0],
        "sample_rate_hz": len(time_grid) / (time_grid[-1] - time_grid[0]),
        "num_interpolated": interpolated_mask.sum(),
        "interpolated_fraction": interpolated_mask.sum() / len(interpolated_mask),
        "error_rms": np.sqrt(np.mean(error**2)),
        "error_max": np.abs(error).max(),
        "method": method,
        "max_gap_s": max_gap_s,
    }

    return aligned_df, metadata


def compute_aligned_signals(
    data: Dict[str, pl.DataFrame],
    method: str = "previous",
) -> Dict[str, Tuple[pl.DataFrame, Dict]]:
    """
    Compute aligned signals for both pitch and yaw axes.

    Args:
        data: Dictionary from load_recording() with 'pitch_cmd', 'pitch_actual', etc.
        method: Interpolation method for commanded signal

    Returns:
        Dictionary with keys 'pitch' and 'yaw', values are (aligned_df, metadata) tuples
    """
    results = {}

    for axis in ["pitch", "yaw"]:
        cmd_key = f"{axis}_cmd"
        actual_key = f"{axis}_actual"

        if cmd_key not in data or actual_key not in data:
            raise ValueError(f"Missing data for {axis} axis")

        aligned_df, metadata = align_signals(
            commanded=data[cmd_key],
            actual=data[actual_key],
            method=method,
        )

        results[axis] = (aligned_df, metadata)

    return results


def validate_alignment(
    aligned_df: pl.DataFrame,
    max_error_expected: float = 5.0,
) -> Dict[str, bool]:
    """
    Validate aligned signal sanity.

    Args:
        aligned_df: Output from align_signals()
        max_error_expected: Maximum expected error in degrees (sanity check)

    Returns:
        Dict of validation checks (True = passed)
    """
    checks = {}

    # Check 1: Error should be small most of the time
    error = aligned_df["error"].to_numpy()
    large_error_rate = (np.abs(error) > max_error_expected).sum() / len(error)
    checks["error_mostly_small"] = large_error_rate < 0.05  # <5% of samples

    # Check 2: Error RMS should be reasonable (< 1 degree typically)
    error_rms = np.sqrt(np.mean(error**2))
    checks["error_rms_reasonable"] = error_rms < 2.0

    # Check 3: Commanded and actual should have similar ranges
    cmd_range = aligned_df["commanded"].max() - aligned_df["commanded"].min()
    actual_range = aligned_df["actual"].max() - aligned_df["actual"].min()
    range_diff = abs(cmd_range - actual_range)
    checks["ranges_similar"] = range_diff < 5.0  # Within 5 degrees

    # Check 4: No NaN or inf values
    checks["no_nan_commanded"] = not aligned_df["commanded"].is_nan().any()
    checks["no_nan_actual"] = not aligned_df["actual"].is_nan().any()
    checks["no_nan_error"] = not aligned_df["error"].is_nan().any()

    # Check 5: Time grid is monotonic and uniform
    time = aligned_df["time_s"].to_numpy()
    time_deltas = np.diff(time)
    checks["time_monotonic"] = np.all(time_deltas > 0)

    # Check for approximately uniform sampling (allow 20% variation)
    median_dt = np.median(time_deltas)
    max_dt_variation = np.max(np.abs(time_deltas - median_dt))
    checks["time_approximately_uniform"] = max_dt_variation < median_dt * 0.2

    return checks


def get_time_slices(
    aligned_df: pl.DataFrame,
    slice_duration_s: float = 10.0,
) -> list[Tuple[float, float]]:
    """
    Divide aligned signal into time slices for analysis.

    Useful for analyzing error statistics over different periods.

    Args:
        aligned_df: Aligned signal DataFrame
        slice_duration_s: Duration of each slice in seconds

    Returns:
        List of (start_time, end_time) tuples
    """
    time = aligned_df["time_s"]
    t_min = time.min()
    t_max = time.max()

    slices = []
    t = t_min
    while t < t_max:
        slices.append((t, min(t + slice_duration_s, t_max)))
        t += slice_duration_s

    return slices


def compute_error_statistics(
    aligned_df: pl.DataFrame,
    percentiles: list[float] = [50, 90, 95, 99],
    transient_window_s: float = 0.5,
) -> Dict[str, float]:
    """
    Compute comprehensive error statistics split into transient and steady-state regimes.

    Transient error: RMS during configurable window after each commanded step.
    Steady-state error: RMS during periods where command has been stable.

    This separation is critical because overall RMS conflates two very different error regimes:
    - Slewing to new command (large transient errors during step response)
    - Holding at commanded position (small steady-state tracking errors)

    Reporting only overall RMS understates hold performance and overstates transient performance.

    Args:
        aligned_df: Aligned signal DataFrame
        percentiles: Percentiles to compute
        transient_window_s: Duration after commanded step to classify as transient (seconds)

    Returns:
        Dictionary of error statistics with keys:
        - overall_*: Statistics across entire signal
        - transient_*: Statistics during transient windows (if detectable)
        - steady_state_*: Statistics outside transient windows (if detectable)
    """
    error = aligned_df["error"].to_numpy()
    time = aligned_df["time_s"].to_numpy()
    commanded = aligned_df["commanded"].to_numpy()

    # Overall statistics (baseline)
    stats = {
        "overall_mean": np.mean(error),
        "overall_std": np.std(error),
        "overall_rms": np.sqrt(np.mean(error**2)),
        "overall_min": np.min(error),
        "overall_max": np.max(error),
        "overall_abs_mean": np.mean(np.abs(error)),
        "overall_abs_max": np.max(np.abs(error)),
    }

    # Add percentiles
    for p in percentiles:
        stats[f"overall_p{p}"] = np.percentile(np.abs(error), p)

    # Detect commanded steps (change in commanded value)
    cmd_diff = np.abs(np.diff(commanded))
    median_diff = np.median(cmd_diff[cmd_diff > 0]) if np.any(cmd_diff > 0) else 0.01
    step_threshold = max(0.1, median_diff * 0.1)  # 10% of typical change or 0.1° minimum

    step_indices = np.where(cmd_diff > step_threshold)[0] + 1  # +1 because diff shifts index

    if len(step_indices) > 0:
        # Mark transient regions (within transient_window_s after each step)
        transient_mask = np.zeros(len(time), dtype=bool)
        for step_idx in step_indices:
            step_time = time[step_idx]
            transient_end = step_time + transient_window_s
            in_window = (time >= step_time) & (time < transient_end)
            transient_mask |= in_window

        # Separate error into transient and steady-state
        transient_error = error[transient_mask]
        steady_state_error = error[~transient_mask]

        # Transient statistics
        if len(transient_error) > 0:
            stats["transient_rms"] = np.sqrt(np.mean(transient_error**2))
            stats["transient_abs_mean"] = np.mean(np.abs(transient_error))
            stats["transient_abs_max"] = np.max(np.abs(transient_error))
            stats["transient_sample_count"] = len(transient_error)
            stats["transient_fraction"] = len(transient_error) / len(error)

        # Steady-state statistics
        if len(steady_state_error) > 0:
            stats["steady_state_rms"] = np.sqrt(np.mean(steady_state_error**2))
            stats["steady_state_abs_mean"] = np.mean(np.abs(steady_state_error))
            stats["steady_state_abs_max"] = np.max(np.abs(steady_state_error))
            stats["steady_state_sample_count"] = len(steady_state_error)
            stats["steady_state_fraction"] = len(steady_state_error) / len(error)
    else:
        # No steps detected - entire signal is effectively steady-state
        stats["transient_rms"] = 0.0
        stats["transient_sample_count"] = 0
        stats["transient_fraction"] = 0.0
        stats["steady_state_rms"] = stats["overall_rms"]
        stats["steady_state_abs_mean"] = stats["overall_abs_mean"]
        stats["steady_state_abs_max"] = stats["overall_abs_max"]
        stats["steady_state_sample_count"] = len(error)
        stats["steady_state_fraction"] = 1.0

    return stats
