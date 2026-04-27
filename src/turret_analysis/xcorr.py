"""
Cross-correlation lag measurement - whole-trace sanity check.

Purpose:
- Compute single whole-trace cross-correlation lag per axis
- Sanity check: should fall between dead-time floor (5b) and tracking lag (5a)
- If outside range, indicates measurement error or data quality issue
"""

from typing import Dict, Tuple
import numpy as np
import polars as pl
from scipy import signal


def compute_whole_trace_lag(
    commanded: np.ndarray,
    actual: np.ndarray,
    dt: float,
    max_lag_s: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute cross-correlation lag over entire trace.

    This is a coarse, whole-trace measurement. For detailed analysis,
    use tracking sequences and discrete steps

    Args:
        commanded: Commanded position array
        actual: Actual position array
        dt: Sample period (seconds)
        max_lag_s: Maximum lag to search (seconds)

    Returns:
        Tuple of (lag_s, correlation_coefficient)
    """
    # Remove DC component
    cmd_centered = commanded - np.mean(commanded)
    actual_centered = actual - np.mean(actual)

    # Compute cross-correlation
    correlation = signal.correlate(actual_centered, cmd_centered, mode='same')

    # Normalize
    norm = np.std(actual) * np.std(commanded) * len(actual)
    if norm > 0:
        correlation = correlation / norm
    else:
        return np.nan, np.nan

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


def analyze_whole_trace_lag(
    aligned_df: pl.DataFrame,
) -> Dict[str, float]:
    """
    Compute whole-trace cross-correlation lag.

    Args:
        aligned_df: Aligned signal DataFrame with 'commanded', 'actual', 'time_s'

    Returns:
        Dictionary with:
        - lag_ms: Whole-trace lag (milliseconds)
        - correlation: Cross-correlation coefficient
        - n_samples: Number of samples used
    """
    time_s = aligned_df['time_s'].to_numpy()
    commanded = aligned_df['commanded'].to_numpy()
    actual = aligned_df['actual'].to_numpy()

    if len(time_s) < 10:
        return {
            'lag_ms': np.nan,
            'correlation': np.nan,
            'n_samples': 0,
        }

    # Compute sample period
    dt = np.median(np.diff(time_s))

    # Compute lag
    lag_s, correlation = compute_whole_trace_lag(commanded, actual, dt, max_lag_s=0.5)

    return {
        'lag_ms': lag_s * 1000,
        'correlation': correlation,
        'n_samples': len(time_s),
    }


def validate_regime_consistency(
    whole_trace_lag_ms: float,
    dead_time_ms: float,
    rise_time_ms: float,
    tracking_lag_ms: float,
    tracking_rms_error_deg: float,
) -> Dict[str, any]:
    """
    Check if three lag measurements are consistent with regime-dependent behavior.

    The three measurements are fundamentally different:
    - Dead time (5b): Transport + current loop response to clean step (~25-35ms)
    - Tracking lag (5a): Temporal xcorr during continuous motion (~0ms, spatial offset instead)
    - Whole-trace lag (6): Weighted average over all regimes (dominated by steps)

    Expected relationships:
    1. tracking_lag < whole_trace_lag < (dead_time + rise_time)
       - Tracking sequences have ~0 temporal lag (spatial offset instead)
       - Whole-trace averages steps (high lag) and tracking (low lag)
       - Step total lag = dead_time + rise_time

    2. tracking_lag ~0 AND tracking_error ~1° confirms spatial (not temporal) offset
       - If temporal lag is ~0 but spatial error exists, motor maintains constant offset
       - This is expected behavior during continuous tracking

    Args:
        whole_trace_lag_ms: Whole-trace cross-correlation lag
        dead_time_ms: Median dead time from Phase 5b (discrete steps)
        rise_time_ms: Median rise time from Phase 5b (10-90%)
        tracking_lag_ms: Median tracking lag from Phase 5a
        tracking_rms_error_deg: RMS tracking error from Phase 5a (degrees)

    Returns:
        Dictionary with validation results and interpretation
    """
    step_total_lag_ms = dead_time_ms + rise_time_ms

    # Check 1: Whole-trace should be between tracking and step total
    check1_valid = tracking_lag_ms <= whole_trace_lag_ms <= step_total_lag_ms

    # Check 2: Tracking lag should be near zero (spatial, not temporal offset)
    check2_valid = abs(tracking_lag_ms) < 10.0  # Within 10ms of zero

    # Check 3: Tracking error should be non-zero (spatial offset exists)
    check3_valid = tracking_rms_error_deg > 0.5  # At least 0.5° spatial error

    # Check 4: Whole-trace should be closer to tracking than step (tracking is ~90% of operation)
    # If whole-trace is dominated by steps, that's suspicious (steps are only ~10% of operation)
    check4_valid = whole_trace_lag_ms < (0.5 * step_total_lag_ms)

    all_valid = check1_valid and check2_valid and check3_valid

    # Interpretation
    if all_valid:
        interpretation = (
            f"Regime-dependent lag confirmed:\n"
            f"  • Step response lag: {step_total_lag_ms:.1f}ms ({dead_time_ms:.1f}ms dead + {rise_time_ms:.1f}ms rise)\n"
            f"  • Tracking temporal lag: ~0ms (motor maintains {tracking_rms_error_deg:.2f}° spatial offset)\n"
            f"  • Whole-trace average: {whole_trace_lag_ms:.1f}ms (weighted mix of regimes)"
        )
    else:
        issues = []
        if not check1_valid:
            issues.append(f"Whole-trace lag ({whole_trace_lag_ms:.1f}ms) outside expected range [{tracking_lag_ms:.1f}, {step_total_lag_ms:.1f}]ms")
        if not check2_valid:
            issues.append(f"Tracking lag ({tracking_lag_ms:.1f}ms) is not near zero - expected spatial offset, not temporal")
        if not check3_valid:
            issues.append(f"Tracking error ({tracking_rms_error_deg:.2f}°) is too small - spatial offset should be measurable")
        interpretation = "Inconsistencies detected:\n  • " + "\n  • ".join(issues)

    # Additional note on whole-trace dominance
    step_weight = whole_trace_lag_ms / step_total_lag_ms
    note = ""
    if step_weight > 0.7:
        note = (
            f"\nNote: Whole-trace lag ({whole_trace_lag_ms:.1f}ms) is {step_weight*100:.0f}% of step total lag.\n"
            f"This suggests steps dominate the whole-trace average, despite being ~10% of operation.\n"
            f"This could indicate: (a) steps have much larger commanded excursions, giving them higher\n"
            f"weight in cross-correlation, or (b) measurement artifact from regime mixing."
        )

    return {
        'valid': all_valid,
        'check1_ordering': check1_valid,
        'check2_tracking_near_zero': check2_valid,
        'check3_spatial_error_exists': check3_valid,
        'check4_tracking_dominance': check4_valid,
        'dead_time_ms': dead_time_ms,
        'rise_time_ms': rise_time_ms,
        'step_total_lag_ms': step_total_lag_ms,
        'tracking_lag_ms': tracking_lag_ms,
        'tracking_rms_error_deg': tracking_rms_error_deg,
        'whole_trace_lag_ms': whole_trace_lag_ms,
        'interpretation': interpretation + note,
    }
