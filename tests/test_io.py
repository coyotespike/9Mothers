"""
Tests for data loading module.
"""

from pathlib import Path

import pytest
import polars as pl

from turret_analysis.io import load_recording, get_sample_rate, validate_monotonic_time


# Check if motor.rrd exists at project root
RRD_PATH = Path(__file__).parent.parent / "motor.rrd"
HAS_DATA_FILE = RRD_PATH.exists()

pytestmark = pytest.mark.skipif(
    not HAS_DATA_FILE,
    reason="motor.rrd not found - place file at project root to run tests"
)


def test_load_recording_basic():
    """Test that load_recording returns expected structure."""
    data = load_recording(RRD_PATH, verbose=False)

    # Check that we got a dictionary
    assert isinstance(data, dict)

    # Check for required signals
    required_signals = ["pitch_cmd", "pitch_actual", "yaw_cmd", "yaw_actual"]
    for signal in required_signals:
        assert signal in data, f"Missing required signal: {signal}"

    # Check that each is a Polars DataFrame
    for signal_name, df in data.items():
        assert isinstance(df, pl.DataFrame)
        assert df.height > 0, f"{signal_name} has no data"

        # Check columns
        assert "time_s" in df.columns
        assert "value" in df.columns
        assert len(df.columns) == 2


def test_column_types():
    """Test that columns have correct types."""
    data = load_recording(RRD_PATH, verbose=False)

    for signal_name, df in data.items():
        # Time should be float
        assert df["time_s"].dtype == pl.Float64, f"{signal_name} time_s is not Float64"

        # Value should be float
        assert df["value"].dtype == pl.Float64, f"{signal_name} value is not Float64"


def test_monotonic_timestamps():
    """Test that timestamps are monotonically increasing."""
    data = load_recording(RRD_PATH, verbose=False)

    for signal_name, df in data.items():
        assert validate_monotonic_time(df), f"{signal_name} timestamps not monotonic"


def test_signal_ranges():
    """Test that signal value ranges are reasonable."""
    data = load_recording(RRD_PATH, verbose=False)

    # Motor positions should be in reasonable degree ranges
    # Based on Phase 0 observation: yaw ~60°, pitch ~40° spans
    for signal_name in ["pitch_cmd", "pitch_actual"]:
        df = data[signal_name]
        val_min = df["value"].min()
        val_max = df["value"].max()

        # Pitch should be reasonable (not wildly out of range)
        assert val_min >= -90, f"{signal_name} min ({val_min}) too low"
        assert val_max <= 90, f"{signal_name} max ({val_max}) too high"

        # Should have some actual range
        assert (val_max - val_min) > 1.0, f"{signal_name} has insufficient range"

    for signal_name in ["yaw_cmd", "yaw_actual"]:
        df = data[signal_name]
        val_min = df["value"].min()
        val_max = df["value"].max()

        # Yaw should be reasonable
        assert val_min >= -180, f"{signal_name} min ({val_min}) too low"
        assert val_max <= 180, f"{signal_name} max ({val_max}) too high"

        # Should have some actual range
        assert (val_max - val_min) > 1.0, f"{signal_name} has insufficient range"


def test_commanded_vs_actual_consistency():
    """Test that commanded and actual signals have similar ranges."""
    data = load_recording(RRD_PATH, verbose=False)

    # Pitch commanded and actual should span similar ranges (within 10 deg tolerance)
    pitch_cmd_range = data["pitch_cmd"]["value"].max() - data["pitch_cmd"]["value"].min()
    pitch_actual_range = data["pitch_actual"]["value"].max() - data["pitch_actual"]["value"].min()

    assert abs(pitch_cmd_range - pitch_actual_range) < 10.0, \
        f"Pitch cmd ({pitch_cmd_range:.1f}°) and actual ({pitch_actual_range:.1f}°) ranges differ significantly"

    # Yaw commanded and actual should span similar ranges
    yaw_cmd_range = data["yaw_cmd"]["value"].max() - data["yaw_cmd"]["value"].min()
    yaw_actual_range = data["yaw_actual"]["value"].max() - data["yaw_actual"]["value"].min()

    assert abs(yaw_cmd_range - yaw_actual_range) < 10.0, \
        f"Yaw cmd ({yaw_cmd_range:.1f}°) and actual ({yaw_actual_range:.1f}°) ranges differ significantly"


def test_sample_rate_estimation():
    """Test that sample rate estimation works and is reasonable."""
    data = load_recording(RRD_PATH, verbose=False)

    # Actual signals should have higher sample rate than commanded
    pitch_actual_rate = get_sample_rate(data["pitch_actual"])
    pitch_cmd_rate = get_sample_rate(data["pitch_cmd"])

    # Sample rates should be positive
    assert pitch_actual_rate > 0
    assert pitch_cmd_rate > 0

    # Actual should be sampled faster than commanded
    # (From  actual ~100 Hz, commands are step-based so lower effective rate)
    assert pitch_actual_rate > pitch_cmd_rate

    # Reasonable ranges (actual should be in ballpark of 50-200 Hz based on Phase 0)
    assert 10 < pitch_actual_rate < 500, \
        f"Actual sample rate ({pitch_actual_rate:.1f} Hz) outside expected range"


def test_event_signals():
    """Test that event signals are present and reasonable."""
    data = load_recording(RRD_PATH, verbose=False)

    # Check event signals exist
    event_signals = ["fire", "muzzle", "impact"]
    for signal_name in event_signals:
        if signal_name in data:
            df = data[signal_name]

            # Should have some events
            assert df.height > 0, f"{signal_name} has no events"

            # Event values should all be 1.0
            assert (df["value"] == 1.0).all(), f"{signal_name} should have value=1.0 for all events"

            # Timestamps should be monotonic
            assert validate_monotonic_time(df), f"{signal_name} timestamps not monotonic"


def test_file_not_found():
    """Test that appropriate error is raised for missing file."""
    with pytest.raises(FileNotFoundError):
        load_recording("nonexistent_file.rrd", verbose=False)


def test_time_alignment():
    """Verify commanded and actual signals overlap in time."""
    data = load_recording(RRD_PATH, verbose=False)

    # Check both axes
    for axis in ["pitch", "yaw"]:
        cmd_min = data[f"{axis}_cmd"]["time_s"].min()
        cmd_max = data[f"{axis}_cmd"]["time_s"].max()
        actual_min = data[f"{axis}_actual"]["time_s"].min()
        actual_max = data[f"{axis}_actual"]["time_s"].max()

        # Actual logging should start before or at commanded start
        # (motor logging starts when powered on, before commands)
        assert actual_min <= cmd_min, \
            f"{axis} actual starts after commanded (actual={actual_min:.3f}, cmd={cmd_min:.3f})"

        # Actual logging should end after or at commanded end
        # (motor keeps logging after last command)
        assert actual_max >= cmd_max, \
            f"{axis} actual ends before commanded (actual={actual_max:.3f}, cmd={cmd_max:.3f})"


def test_no_duplicate_timestamps():
    """Ensure no excessive duplicate timestamps within signals."""
    data = load_recording(RRD_PATH, verbose=False)

    for signal_name, df in data.items():
        if signal_name in ["fire", "muzzle", "impact"]:
            # Events can have exact duplicate timestamps (simultaneous triggers)
            continue

        duplicates = df["time_s"].is_duplicated().sum()
        dup_rate = duplicates / df.height

        # Allow small number of duplicates (CAN retransmissions, logging quirks)
        # but should be < 1%
        assert dup_rate < 0.01, \
            f"{signal_name} has {dup_rate*100:.1f}% duplicate timestamps"


def test_no_missing_values():
    """Verify no NaN or null values in loaded data."""
    data = load_recording(RRD_PATH, verbose=False)

    for signal_name, df in data.items():
        # Check for null values
        assert df["time_s"].null_count() == 0, \
            f"{signal_name} has {df['time_s'].null_count()} null timestamps"
        assert df["value"].null_count() == 0, \
            f"{signal_name} has {df['value'].null_count()} null values"

        # Check for NaN (different from null in Polars)
        nan_count = df["value"].is_nan().sum()
        assert nan_count == 0, \
            f"{signal_name} has {nan_count} NaN values"


def test_commanded_step_sizes():
    """Verify commanded steps are within expected bounds."""
    data = load_recording(RRD_PATH, verbose=False)

    for axis in ["pitch", "yaw"]:
        cmd_df = data[f"{axis}_cmd"]
        step_sizes = cmd_df["value"].diff().abs().drop_nulls()

        # Most steps should be < 30 degrees (reasonable slew)
        large_steps = (step_sizes > 30.0).sum()
        large_step_rate = large_steps / len(step_sizes)
        assert large_step_rate < 0.05, \
            f"{axis} has {large_step_rate*100:.1f}% large steps (>30°), expected <5%"

        # Max step should be < 90 degrees (sanity check)
        max_step = step_sizes.max()
        assert max_step < 90.0, \
            f"{axis} has unreasonable step size: {max_step:.1f}°"


def test_firing_event_order():
    """Verify Fire → Muzzle → Impact temporal order for sample events."""
    data = load_recording(RRD_PATH, verbose=False)

    # Skip if missing event data
    if not all(k in data for k in ["fire", "muzzle", "impact"]):
        pytest.skip("Missing event data")

    fire_times = data["fire"]["time_s"].to_list()
    muzzle_times = data["muzzle"]["time_s"].to_list()
    impact_times = data["impact"]["time_s"].to_list()

    # Check first 10 fire events for proper sequencing
    valid_sequences = 0
    for fire_time in fire_times[:10]:
        # Find nearest muzzle after fire (within 100ms)
        muzzle_deltas = [m - fire_time for m in muzzle_times if 0 < m - fire_time < 0.1]
        if not muzzle_deltas:
            continue

        nearest_muzzle_time = fire_time + min(muzzle_deltas)

        # Find nearest impact after muzzle (within 200ms)
        impact_deltas = [i - nearest_muzzle_time for i in impact_times if 0 < i - nearest_muzzle_time < 0.2]
        if impact_deltas:
            valid_sequences += 1

    # At least half of checked events should have proper Fire→Muzzle→Impact sequence
    assert valid_sequences >= 5, \
        f"Only {valid_sequences}/10 events have valid Fire→Muzzle→Impact sequence"


def test_sample_rate_consistency():
    """Verify sample rate doesn't drift significantly over time."""
    data = load_recording(RRD_PATH, verbose=False)

    # Check pitch_actual (should be steady ~100 Hz)
    pitch_actual = data["pitch_actual"]

    # Split into 5 time windows
    t_min = pitch_actual["time_s"].min()
    t_max = pitch_actual["time_s"].max()
    duration = t_max - t_min

    rates = []
    for i in range(5):
        t_start = t_min + i * duration / 5
        t_end = t_min + (i + 1) * duration / 5

        window = pitch_actual.filter(
            (pl.col("time_s") >= t_start) & (pl.col("time_s") < t_end)
        )

        if window.height > 10:  # Need enough samples
            rates.append(get_sample_rate(window))

    # All windows should be within 10% of mean
    mean_rate = sum(rates) / len(rates)
    for i, rate in enumerate(rates):
        deviation = abs(rate - mean_rate) / mean_rate
        assert deviation < 0.10, \
            f"Sample rate in window {i} deviates {deviation*100:.1f}% from mean (rates: {rates})"
