"""
Step response analysis.

Analyzes discrete step responses to measure motor control characteristics.
Diagnostic metric - characterizes the motor physics under isolated conditions.

Metrics:
- Dead time (L): Delay before motor starts moving
- Rise time: Time to reach target (10% to 90%)
- Overshoot: Peak deviation beyond target
- Settling time: Time to stabilize within tolerance
- Second-order fit: Natural frequency ωn, damping ratio ζ
"""

from typing import Dict, Tuple, Optional
import numpy as np
import polars as pl
from scipy.optimize import curve_fit


def measure_dead_time(
    time_rel_s: np.ndarray,
    actual: np.ndarray,
    threshold_deg: float = 0.05,
) -> float:
    """
    Measure dead time: time until actual first moves.

    Dead time (L) is the delay between command and first visible motion.
    Captures wiring delay + controller processing + motor inductance lag.

    Args:
        time_rel_s: Time relative to step command (t=0 at command)
        actual: Actual position
        threshold_deg: Motion detection threshold (degrees)

    Returns:
        Dead time in seconds (or np.nan if no motion detected)
    """
    # Baseline: actual position before step (t < 0)
    pre_step_mask = time_rel_s < 0
    if pre_step_mask.sum() == 0:
        return np.nan

    baseline = np.mean(actual[pre_step_mask])

    # Find first point where |actual - baseline| > threshold
    post_step_mask = time_rel_s >= 0
    post_step_actual = actual[post_step_mask]
    post_step_time = time_rel_s[post_step_mask]

    motion = np.abs(post_step_actual - baseline) > threshold_deg

    if motion.sum() == 0:
        return np.nan  # No motion detected

    first_motion_idx = np.where(motion)[0][0]
    dead_time = post_step_time[first_motion_idx]

    return dead_time


def measure_rise_time(
    time_rel_s: np.ndarray,
    actual: np.ndarray,
    value_before: float,
    value_after: float,
) -> Tuple[float, float, float]:
    """
    Measure rise time: time from 10% to 90% of final value.

    Args:
        time_rel_s: Time relative to step command
        actual: Actual position
        value_before: Commanded position before step
        value_after: Commanded position after step

    Returns:
        Tuple of (rise_time_s, t_10pct, t_90pct)
    """
    step_size = value_after - value_before

    # Target levels
    level_10pct = value_before + 0.1 * step_size
    level_90pct = value_before + 0.9 * step_size

    # Find crossing times
    post_step_mask = time_rel_s >= 0
    post_step_actual = actual[post_step_mask]
    post_step_time = time_rel_s[post_step_mask]

    # Handle both up and down steps
    if step_size > 0:  # Upward step
        cross_10 = post_step_actual >= level_10pct
        cross_90 = post_step_actual >= level_90pct
    else:  # Downward step
        cross_10 = post_step_actual <= level_10pct
        cross_90 = post_step_actual <= level_90pct

    if cross_10.sum() == 0 or cross_90.sum() == 0:
        return np.nan, np.nan, np.nan

    t_10pct = post_step_time[np.where(cross_10)[0][0]]
    t_90pct = post_step_time[np.where(cross_90)[0][0]]

    rise_time = t_90pct - t_10pct

    return rise_time, t_10pct, t_90pct


def measure_overshoot(
    time_rel_s: np.ndarray,
    actual: np.ndarray,
    value_after: float,
    settling_window_s: float = 0.5,
) -> Tuple[float, float]:
    """
    Measure overshoot: maximum deviation beyond target.

    Args:
        time_rel_s: Time relative to step command
        actual: Actual position
        value_after: Commanded position after step (target)
        settling_window_s: Window to search for overshoot

    Returns:
        Tuple of (overshoot_deg, overshoot_pct)
    """
    # Search window: 0 to settling_window_s
    search_mask = (time_rel_s >= 0) & (time_rel_s <= settling_window_s)

    if search_mask.sum() == 0:
        return np.nan, np.nan

    actual_window = actual[search_mask]

    # Find peak deviation from target
    error = actual_window - value_after
    max_deviation = np.max(np.abs(error))

    # Peak should be in direction of motion
    # (overshoot means going past target, not backward)
    peak_idx = np.argmax(np.abs(error))
    overshoot_deg = error[peak_idx]

    # Overshoot percentage (relative to step size)
    # This requires knowing step magnitude, which we can infer
    # For now, just return absolute overshoot
    overshoot_pct = np.nan  # TODO: compute if step_size available

    return overshoot_deg, overshoot_pct


def measure_settling_time(
    time_rel_s: np.ndarray,
    actual: np.ndarray,
    value_after: float,
    tolerance_deg: float = 0.1,
    min_stable_duration_s: float = 0.1,
) -> float:
    """
    Measure settling time: time to stabilize within tolerance.

    Args:
        time_rel_s: Time relative to step command
        actual: Actual position
        value_after: Commanded position after step (target)
        tolerance_deg: Settling tolerance (degrees)
        min_stable_duration_s: Minimum time within tolerance to declare settled

    Returns:
        Settling time in seconds
    """
    post_step_mask = time_rel_s >= 0
    post_step_actual = actual[post_step_mask]
    post_step_time = time_rel_s[post_step_mask]

    if len(post_step_actual) == 0:
        return np.nan

    # Error from target
    error = np.abs(post_step_actual - value_after)

    # Find first time where error stays within tolerance
    within_tolerance = error <= tolerance_deg

    # Need consecutive samples within tolerance
    dt = np.median(np.diff(post_step_time)) if len(post_step_time) > 1 else 0.01
    min_stable_samples = int(min_stable_duration_s / dt)

    for i in range(len(within_tolerance) - min_stable_samples):
        if np.all(within_tolerance[i:i+min_stable_samples]):
            return post_step_time[i]

    return np.nan  # Never settled


def second_order_response(t, wn, zeta, K, tau, offset):
    """
    Second-order step response model.

    s(t) = K * (1 - exp(-zeta*wn*t) * (cos(wd*t) + (zeta*wn/wd)*sin(wd*t))) + offset

    where wd = wn * sqrt(1 - zeta^2) (damped natural frequency)

    Args:
        t: Time
        wn: Natural frequency (rad/s)
        zeta: Damping ratio
        K: Steady-state gain
        tau: Time delay
        offset: Initial value

    Returns:
        Response at time t
    """
    t_shifted = t - tau
    t_shifted = np.maximum(t_shifted, 0)  # No response before delay

    if zeta >= 1:  # Overdamped or critically damped
        response = K * (1 - np.exp(-zeta * wn * t_shifted) *
                       (1 + zeta * wn * t_shifted)) + offset
    else:  # Underdamped
        wd = wn * np.sqrt(1 - zeta**2)
        response = K * (1 - np.exp(-zeta * wn * t_shifted) *
                       (np.cos(wd * t_shifted) +
                        (zeta * wn / wd) * np.sin(wd * t_shifted))) + offset

    return response


def fit_second_order_model(
    time_rel_s: np.ndarray,
    actual: np.ndarray,
    value_before: float,
    value_after: float,
) -> Dict[str, float]:
    """
    Fit second-order model to step response.

    Args:
        time_rel_s: Time relative to step command
        actual: Actual position
        value_before: Initial value
        value_after: Final value

    Returns:
        Dictionary with fit parameters:
        - wn: Natural frequency (rad/s)
        - zeta: Damping ratio
        - tau: Time delay (s)
        - K: Gain
        - r_squared: Fit quality
    """
    # Use only post-step data
    mask = time_rel_s >= 0
    t = time_rel_s[mask]
    y = actual[mask]

    if len(t) < 10:
        return {
            'wn': np.nan,
            'zeta': np.nan,
            'tau': np.nan,
            'K': np.nan,
            'r_squared': np.nan,
        }

    # Initial parameter guess
    K_guess = value_after - value_before
    tau_guess = 0.02  # 20ms delay
    wn_guess = 10.0  # 10 rad/s ~= 1.6 Hz
    zeta_guess = 0.7  # Slightly underdamped

    p0 = [wn_guess, zeta_guess, K_guess, tau_guess, value_before]

    try:
        # Fit model
        popt, pcov = curve_fit(
            second_order_response,
            t, y,
            p0=p0,
            bounds=([0.1, 0, -100, 0, -100],
                   [100, 2, 100, 0.5, 100]),
            maxfev=5000
        )

        wn, zeta, K, tau, offset = popt

        # Compute R²
        y_pred = second_order_response(t, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            'wn': wn,
            'zeta': zeta,
            'tau': tau,
            'K': K,
            'r_squared': r_squared,
        }

    except Exception:
        return {
            'wn': np.nan,
            'zeta': np.nan,
            'tau': np.nan,
            'K': np.nan,
            'r_squared': np.nan,
        }


def analyze_step_response(
    step_row: dict,
    epoch_df: pl.DataFrame,
) -> Dict[str, float]:
    """
    Analyze a single step response.

    Args:
        step_row: Row from discrete step catalog
        epoch_df: Epoch window from get_step_epoch()

    Returns:
        Dictionary of step response metrics
    """
    time_rel = epoch_df['time_rel_s'].to_numpy()
    actual = epoch_df['actual'].to_numpy()

    value_before = step_row['value_before']
    value_after = step_row['value_after']

    # Measure metrics
    dead_time = measure_dead_time(time_rel, actual, threshold_deg=0.05)

    rise_time, t_10, t_90 = measure_rise_time(
        time_rel, actual, value_before, value_after
    )

    overshoot_deg, overshoot_pct = measure_overshoot(
        time_rel, actual, value_after, settling_window_s=0.5
    )

    settling_time = measure_settling_time(
        time_rel, actual, value_after,
        tolerance_deg=0.1, min_stable_duration_s=0.05
    )

    # Fit second-order model
    fit_params = fit_second_order_model(
        time_rel, actual, value_before, value_after
    )

    return {
        'dead_time_ms': dead_time * 1000 if not np.isnan(dead_time) else np.nan,
        'rise_time_ms': rise_time * 1000 if not np.isnan(rise_time) else np.nan,
        'overshoot_deg': overshoot_deg,
        'settling_time_ms': settling_time * 1000 if not np.isnan(settling_time) else np.nan,
        'wn_rad_s': fit_params['wn'],
        'zeta': fit_params['zeta'],
        'tau_ms': fit_params['tau'] * 1000,
        'fit_r_squared': fit_params['r_squared'],
    }


def analyze_all_steps(
    steps: pl.DataFrame,
    aligned_df: pl.DataFrame,
    filter_flags: bool = True,
) -> pl.DataFrame:
    """
    Analyze all discrete steps.

    Args:
        steps: Discrete step catalog from extract_discrete_steps()
        aligned_df: Aligned signal DataFrame
        filter_flags: If True, skip steps with quality flags

    Returns:
        DataFrame with step response metrics added
    """
    from turret_analysis.segmentation import get_step_epoch

    # Optionally filter to clean steps
    if filter_flags:
        clean_steps = steps.filter(
            (pl.col('isolated') == True) &
            (pl.col('edge_truncated') == False) &
            (pl.col('has_interp_gap') == False) &
            (pl.col('magnitude') >= 0.5)
        )
    else:
        clean_steps = steps

    metrics_list = []

    for row in clean_steps.iter_rows(named=True):
        epoch = get_step_epoch(row, aligned_df)
        metrics = analyze_step_response(row, epoch)
        metrics_list.append(metrics)

    if len(metrics_list) == 0:
        return pl.DataFrame()

    # Add metrics as new columns
    metrics_df = pl.DataFrame(metrics_list)

    # Combine with original step data
    result = pl.concat([clean_steps, metrics_df], how='horizontal')

    return result
