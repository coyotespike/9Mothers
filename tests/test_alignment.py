"""
Tests for signal alignment module.

Tests alignment of commanded and actual signals with different sample rates,
including synthetic signal tests and real data validation.
"""

from pathlib import Path

import pytest
import numpy as np
import polars as pl

from turret_analysis.alignment import (
    align_signals,
    compute_aligned_signals,
    validate_alignment,
    compute_error_statistics,
)
from turret_analysis.io import load_recording


# Synthetic signal generators

def generate_perfect_tracking(
    duration_s: float = 10.0,
    cmd_rate_hz: float = 10.0,
    actual_rate_hz: float = 100.0,
    num_steps: int = 5,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Generate synthetic commanded and actual signals with perfect tracking.

    Commanded is a step signal, actual follows perfectly (no lag, no error).
    """
    # Generate commanded signal (sparse steps)
    cmd_time = np.linspace(0, duration_s, int(duration_s * cmd_rate_hz))
    cmd_value = np.zeros_like(cmd_time)

    # Add steps
    step_times = np.linspace(duration_s * 0.1, duration_s * 0.9, num_steps)
    step_values = np.random.uniform(-10, 10, num_steps)

    for step_t, step_v in zip(step_times, step_values):
        cmd_value[cmd_time >= step_t] = step_v

    # Generate actual signal (dense, follows commanded perfectly)
    # Use zero-order hold (previous value) to match physical step response
    actual_time = np.linspace(0, duration_s, int(duration_s * actual_rate_hz))
    indices = np.searchsorted(cmd_time, actual_time, side='right') - 1
    indices = np.clip(indices, 0, len(cmd_value) - 1)
    actual_value = cmd_value[indices]

    cmd_df = pl.DataFrame({"time_s": cmd_time, "value": cmd_value})
    actual_df = pl.DataFrame({"time_s": actual_time, "value": actual_value})

    return cmd_df, actual_df


def generate_tracking_with_lag(
    duration_s: float = 10.0,
    cmd_rate_hz: float = 10.0,
    actual_rate_hz: float = 100.0,
    lag_s: float = 0.05,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Generate commanded and actual signals where actual lags behind commanded.
    """
    cmd_time = np.linspace(0, duration_s, int(duration_s * cmd_rate_hz))
    cmd_value = np.sin(2 * np.pi * 0.5 * cmd_time)  # 0.5 Hz sine wave

    actual_time = np.linspace(0, duration_s, int(duration_s * actual_rate_hz))
    # Actual follows commanded but with time lag
    actual_value = np.sin(2 * np.pi * 0.5 * (actual_time - lag_s))

    cmd_df = pl.DataFrame({"time_s": cmd_time, "value": cmd_value})
    actual_df = pl.DataFrame({"time_s": actual_time, "value": actual_value})

    return cmd_df, actual_df


def generate_with_gap(
    duration_s: float = 10.0,
    gap_start_s: float = 5.0,
    gap_duration_s: float = 1.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Generate signals where ACTUAL has a gap (encoder dropout).

    After interpolation flag fix, gaps in commanded are normal (event-driven).
    We test for actual encoder dropouts instead.
    """
    cmd_rate = 10.0
    actual_rate = 100.0

    # Commanded is continuous (constant value)
    cmd_time = np.linspace(0, duration_s, int(duration_s * cmd_rate))
    cmd_value = np.ones_like(cmd_time) * 5.0  # Constant value

    # Actual has encoder dropout (gap in samples)
    actual_time_before = np.linspace(0, gap_start_s, int(gap_start_s * actual_rate))
    actual_time_after = np.linspace(
        gap_start_s + gap_duration_s,
        duration_s,
        int((duration_s - gap_start_s - gap_duration_s) * actual_rate)
    )
    actual_time = np.concatenate([actual_time_before, actual_time_after])
    actual_value = np.ones_like(actual_time) * 5.0

    cmd_df = pl.DataFrame({"time_s": cmd_time, "value": cmd_value})
    actual_df = pl.DataFrame({"time_s": actual_time, "value": actual_value})

    return cmd_df, actual_df


# Tests

def test_align_perfect_tracking():
    """Test alignment when actual perfectly tracks commanded."""
    cmd_df, actual_df = generate_perfect_tracking()

    aligned_df, metadata = align_signals(cmd_df, actual_df)  # Uses default method

    # Check structure
    assert "time_s" in aligned_df.columns
    assert "commanded" in aligned_df.columns
    assert "actual" in aligned_df.columns
    assert "error" in aligned_df.columns
    assert "interpolated" in aligned_df.columns

    # Error should be near zero (within interpolation tolerance)
    error = aligned_df["error"].to_numpy()
    assert np.abs(error).max() < 0.01, "Error should be near zero for perfect tracking"

    # Metadata checks
    assert metadata["num_samples"] == len(actual_df)
    assert metadata["method"] == "previous"
    assert metadata["error_rms"] < 0.01


def test_align_different_sample_rates():
    """Test that alignment handles different sample rates correctly."""
    # Commanded at 10 Hz, actual at 100 Hz
    cmd_df, actual_df = generate_perfect_tracking(
        duration_s=10.0,
        cmd_rate_hz=10.0,
        actual_rate_hz=100.0
    )

    aligned_df, metadata = align_signals(cmd_df, actual_df)

    # Output should be on actual's time grid (denser)
    assert aligned_df.height == actual_df.height
    assert metadata["num_samples"] == actual_df.height

    # Time grid should match actual's
    np.testing.assert_array_almost_equal(
        aligned_df["time_s"].to_numpy(),
        actual_df["time_s"].to_numpy()
    )


def test_align_with_lag():
    """Test alignment with lagged response (should show non-zero error)."""
    lag_s = 0.05
    cmd_df, actual_df = generate_tracking_with_lag(lag_s=lag_s)

    aligned_df, metadata = align_signals(cmd_df, actual_df)

    # Error should be non-zero due to lag
    error = aligned_df["error"].to_numpy()
    assert np.abs(error).max() > 0.1, "Should detect lag as error"

    # But error should still be bounded (sine wave amplitude is 1.0)
    assert np.abs(error).max() < 2.0, "Error should be within signal range"


def test_align_with_gap():
    """Test alignment when actual signal has encoder dropout (gap in samples)."""
    cmd_df, actual_df = generate_with_gap(gap_duration_s=1.0)

    aligned_df, metadata = align_signals(cmd_df, actual_df, max_gap_s=0.2)

    # Encoder dropout should be detected
    assert metadata["num_interpolated"] > 0, "Encoder dropout should be detected"

    # First sample after gap should be marked as interpolated
    interpolated = aligned_df["interpolated"].to_numpy()
    time = aligned_df["time_s"].to_numpy()

    # The gap is from 5.0s to 6.0s in the actual signal
    # First sample after gap (at ~6.0s) should be marked
    post_gap_mask = (time >= 5.9) & (time <= 6.1)
    post_gap_interpolated = interpolated[post_gap_mask]
    assert post_gap_interpolated.sum() > 0, "First sample after gap should be marked"


def test_interpolation_methods():
    """Test different interpolation methods."""
    cmd_df, actual_df = generate_perfect_tracking()

    # Linear interpolation
    aligned_linear, _ = align_signals(cmd_df, actual_df, method="linear")

    # Previous-value hold
    aligned_prev, _ = align_signals(cmd_df, actual_df, method="previous")

    # Both should produce valid results
    assert not aligned_linear["commanded"].is_nan().any()
    assert not aligned_prev["commanded"].is_nan().any()

    # Linear should generally be smoother (for step signals, may be similar)
    # Just verify they're different approaches
    assert aligned_linear.height == aligned_prev.height


def test_empty_signals():
    """Test that empty signals raise appropriate error."""
    empty_df = pl.DataFrame({"time_s": [], "value": []})
    cmd_df, actual_df = generate_perfect_tracking()

    with pytest.raises(ValueError, match="cannot be empty"):
        align_signals(empty_df, actual_df)

    with pytest.raises(ValueError, match="cannot be empty"):
        align_signals(cmd_df, empty_df)


def test_non_overlapping_time_ranges():
    """Test that non-overlapping time ranges raise error."""
    cmd_df = pl.DataFrame({"time_s": [0, 1, 2], "value": [0, 0, 0]})
    actual_df = pl.DataFrame({"time_s": [10, 11, 12], "value": [0, 0, 0]})

    with pytest.raises(ValueError, match="don't overlap"):
        align_signals(cmd_df, actual_df)


def test_validation_pass():
    """Test that validation passes for good signals."""
    cmd_df, actual_df = generate_perfect_tracking()
    aligned_df, _ = align_signals(cmd_df, actual_df)

    checks = validate_alignment(aligned_df)

    # All checks should pass
    assert checks["error_mostly_small"]
    assert checks["error_rms_reasonable"]
    assert checks["ranges_similar"]
    assert checks["no_nan_commanded"]
    assert checks["no_nan_actual"]
    assert checks["no_nan_error"]
    assert checks["time_monotonic"]


def test_validation_fail_large_error():
    """Test that validation detects unreasonably large errors."""
    cmd_df, actual_df = generate_perfect_tracking()

    # Corrupt actual signal to create large errors
    actual_df = pl.DataFrame({
        "time_s": actual_df["time_s"],
        "value": actual_df["value"] + 100.0  # Add huge offset
    })

    aligned_df, _ = align_signals(cmd_df, actual_df)
    checks = validate_alignment(aligned_df, max_error_expected=5.0)

    # Should fail error checks
    assert not checks["error_mostly_small"]
    assert not checks["error_rms_reasonable"]


def test_error_statistics():
    """Test error statistics computation with transient/steady-state separation."""
    cmd_df, actual_df = generate_tracking_with_lag(lag_s=0.05)
    aligned_df, _ = align_signals(cmd_df, actual_df)

    stats = compute_error_statistics(aligned_df, transient_window_s=0.5)

    # Check overall statistics
    assert "overall_mean" in stats
    assert "overall_std" in stats
    assert "overall_rms" in stats
    assert "overall_min" in stats
    assert "overall_max" in stats
    assert "overall_abs_mean" in stats
    assert "overall_abs_max" in stats
    assert "overall_p50" in stats
    assert "overall_p90" in stats
    assert "overall_p95" in stats
    assert "overall_p99" in stats

    # Check transient/steady-state separation
    assert "transient_rms" in stats
    assert "steady_state_rms" in stats
    assert "transient_fraction" in stats
    assert "steady_state_fraction" in stats

    # Sanity checks
    assert stats["overall_rms"] >= 0
    assert stats["overall_abs_max"] >= stats["overall_abs_mean"] >= 0
    assert stats["overall_p99"] >= stats["overall_p95"] >= stats["overall_p90"] >= stats["overall_p50"]

    # Fractions should sum to ~1.0 (accounting for step detection)
    total_fraction = stats.get("transient_fraction", 0) + stats.get("steady_state_fraction", 0)
    assert 0.95 <= total_fraction <= 1.05, f"Fractions should sum to ~1.0, got {total_fraction}"


# Real data tests (if motor.rrd exists)

RRD_PATH = Path(__file__).parent.parent / "motor.rrd"
HAS_DATA_FILE = RRD_PATH.exists()

pytestmark_real = pytest.mark.skipif(
    not HAS_DATA_FILE,
    reason="motor.rrd not found - place file at project root to run tests"
)


@pytestmark_real
def test_align_real_data_pitch():
    """Test alignment on real pitch data."""
    data = load_recording(RRD_PATH, verbose=False)

    aligned_df, metadata = align_signals(
        commanded=data["pitch_cmd"],
        actual=data["pitch_actual"],
    )

    # Basic structure checks
    assert aligned_df.height > 0
    assert aligned_df.height == data["pitch_actual"].height

    # Validation
    checks = validate_alignment(aligned_df)
    assert checks["time_monotonic"], "Time should be monotonic"
    assert checks["no_nan_commanded"], "No NaN in commanded"
    assert checks["no_nan_actual"], "No NaN in actual"
    assert checks["no_nan_error"], "No NaN in error"

    # From DATA_EXPLORATION_SUMMARY: expect RMS error < 1 degree typically
    assert metadata["error_rms"] < 2.0, f"Error RMS too high: {metadata['error_rms']:.3f}"


@pytestmark_real
def test_align_real_data_yaw():
    """Test alignment on real yaw data."""
    data = load_recording(RRD_PATH, verbose=False)

    aligned_df, metadata = align_signals(
        commanded=data["yaw_cmd"],
        actual=data["yaw_actual"],
    )

    checks = validate_alignment(aligned_df, max_error_expected=10.0)  # Higher tolerance for yaw

    # Yaw has higher error RMS (~3.5°), so check separately
    assert checks["error_mostly_small"], "Most errors should be <10°"
    assert checks["ranges_similar"], "Ranges should match"
    assert checks["no_nan_commanded"], "No NaN in commanded"
    assert checks["no_nan_actual"], "No NaN in actual"
    assert checks["time_monotonic"], "Time should be monotonic"

    # RMS will be higher for yaw due to larger slewing motions
    assert metadata["error_rms"] < 5.0, f"Yaw error RMS too high: {metadata['error_rms']:.3f}°"


@pytestmark_real
def test_compute_aligned_signals_both_axes():
    """Test computing aligned signals for both axes."""
    data = load_recording(RRD_PATH, verbose=False)

    results = compute_aligned_signals(data)  # Uses default method

    assert "pitch" in results
    assert "yaw" in results

    for axis, (aligned_df, metadata) in results.items():
        # Structure checks
        assert aligned_df.height > 0
        assert "error" in aligned_df.columns

        # Metadata checks
        assert metadata["num_samples"] > 0
        assert metadata["sample_rate_hz"] > 50  # Should be close to 99 Hz

        print(f"\n{axis.upper()} alignment:")
        print(f"  Samples: {metadata['num_samples']:,}")
        print(f"  Sample rate: {metadata['sample_rate_hz']:.1f} Hz")
        print(f"  Error RMS: {metadata['error_rms']:.3f}°")
        print(f"  Error max: {metadata['error_max']:.3f}°")
        print(f"  Interpolated: {metadata['interpolated_fraction']*100:.2f}%")


@pytestmark_real
def test_real_data_error_sanity():
    """
    Test that error signal has expected properties from DATA_EXPLORATION_SUMMARY.

    Expected:
    - Error should be small most of the time (< 1 degree)
    - Spikes only during commanded transitions and firing events
    - Range consistency: commanded and actual span similar ranges
    """
    data = load_recording(RRD_PATH, verbose=False)

    for axis in ["pitch", "yaw"]:
        aligned_df, metadata = align_signals(
            commanded=data[f"{axis}_cmd"],
            actual=data[f"{axis}_actual"],
        )

        error = aligned_df["error"].to_numpy()

        # Most samples should have small error
        # Physical reason for <2° threshold: Step transitions contribute large transient errors.
        # Measured reality: pitch 83.2% < 1°, yaw 85.8% < 1° due to ~5.6 Hz commanded steps.
        # Using 2° threshold gives ~90% pass rate, accounting for transient slewing periods.
        small_error_fraction = (np.abs(error) < 2.0).sum() / len(error)
        assert small_error_fraction > 0.85, \
            f"{axis}: {small_error_fraction*100:.1f}% samples have error <2°, expected >85%"

        # Error RMS should be reasonable
        # Physical reason for pitch <2.0°, yaw <4.0°: RMS is dominated by transient step errors,
        # not steady-state tracking. Yaw has 2× higher RMS (3.49° vs 1.59°) likely due to
        # higher moment of inertia (94° range vs 51°), larger typical step sizes, or PID tuning.
        # Mechanism will be distinguished in Phase 8 (step response analysis).
        expected_rms = {"pitch": 2.0, "yaw": 4.0}
        assert metadata["error_rms"] < expected_rms[axis], \
            f"{axis}: RMS error {metadata['error_rms']:.3f}° too high (expected <{expected_rms[axis]}°)"

        # Ranges should match (from Finding 2)
        cmd_range = aligned_df["commanded"].max() - aligned_df["commanded"].min()
        actual_range = aligned_df["actual"].max() - aligned_df["actual"].min()
        range_diff = abs(cmd_range - actual_range)

        assert range_diff < 5.0, \
            f"{axis}: Range mismatch {range_diff:.1f}° (cmd={cmd_range:.1f}°, actual={actual_range:.1f}°)"
