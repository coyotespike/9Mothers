"""
Tests for step segmentation module.

Tests regime classification, discrete step extraction, and tracking sequence extraction
with synthetic signals designed to exercise edge cases and quality gates.
"""

from pathlib import Path

import pytest
import numpy as np
import polars as pl

from turret_analysis.segmentation import (
    classify_commanded_changes,
    extract_discrete_steps,
    extract_tracking_sequences,
    get_step_epoch,
    get_tracking_epoch,
)
from turret_analysis.alignment import align_signals
from turret_analysis.io import load_recording


# Synthetic signal generators

def generate_single_step(
    duration_s: float = 5.0,
    step_time_s: float = 2.5,
    step_magnitude: float = 10.0,
    sample_rate_hz: float = 10.0,
) -> pl.DataFrame:
    """Generate commanded signal with single perfect step."""
    times = np.arange(0, duration_s, 1.0 / sample_rate_hz)
    values = np.zeros_like(times)

    # Step at step_time_s
    values[times >= step_time_s] = step_magnitude

    return pl.DataFrame({"time_s": times, "value": values})


def generate_noisy_signal(
    duration_s: float = 5.0,
    noise_amplitude: float = 0.1,
    sample_rate_hz: float = 10.0,
) -> pl.DataFrame:
    """Generate commanded signal with sub-threshold noise."""
    times = np.arange(0, duration_s, 1.0 / sample_rate_hz)
    values = np.random.uniform(-noise_amplitude, noise_amplitude, len(times))

    return pl.DataFrame({"time_s": times, "value": values})


def generate_close_doublet(
    duration_s: float = 5.0,
    step1_time_s: float = 2.0,
    step2_time_s: float = 2.1,  # 100ms apart
    step_magnitude: float = 5.0,
    sample_rate_hz: float = 10.0,
) -> pl.DataFrame:
    """Generate two steps close together (tracking regime, not discrete)."""
    times = np.arange(0, duration_s, 1.0 / sample_rate_hz)
    values = np.zeros_like(times)

    # First step
    values[times >= step1_time_s] = step_magnitude

    # Second step
    values[times >= step2_time_s] = 2 * step_magnitude

    return pl.DataFrame({"time_s": times, "value": values})


def generate_separated_doublet(
    duration_s: float = 5.0,
    step1_time_s: float = 1.5,
    step2_time_s: float = 3.0,  # 1.5s apart
    step_magnitude: float = 5.0,
    sample_rate_hz: float = 10.0,
) -> pl.DataFrame:
    """Generate two well-separated steps (both discrete)."""
    times = np.arange(0, duration_s, 1.0 / sample_rate_hz)
    values = np.zeros_like(times)

    values[times >= step1_time_s] = step_magnitude
    values[times >= step2_time_s] = 2 * step_magnitude

    return pl.DataFrame({"time_s": times, "value": values})


def generate_tracking_sequence(
    duration_s: float = 5.0,
    start_time_s: float = 1.0,
    num_steps: int = 6,
    step_spacing_s: float = 0.05,
    step_magnitude: float = 1.0,
    sample_rate_hz: float = 10.0,
) -> pl.DataFrame:
    """Generate ramping sequence (many small steps - tracking, not step response)."""
    times = np.arange(0, duration_s, 1.0 / sample_rate_hz)
    values = np.zeros_like(times)

    for i in range(num_steps):
        step_time = start_time_s + i * step_spacing_s
        values[times >= step_time] = (i + 1) * step_magnitude

    return pl.DataFrame({"time_s": times, "value": values})


def create_aligned_df_synthetic(cmd_df: pl.DataFrame, actual_rate_hz: float = 100.0) -> pl.DataFrame:
    """Create synthetic aligned DataFrame from commanded signal."""
    # Create dense actual time grid
    t_min = cmd_df["time_s"].min()
    t_max = cmd_df["time_s"].max()
    duration = t_max - t_min

    actual_times = t_min + np.arange(0, duration, 1.0 / actual_rate_hz)

    # Interpolate commanded using zero-order hold
    cmd_times = cmd_df["time_s"].to_numpy()
    cmd_values = cmd_df["value"].to_numpy()

    indices = np.searchsorted(cmd_times, actual_times, side='right') - 1
    indices = np.clip(indices, 0, len(cmd_values) - 1)
    commanded = cmd_values[indices]

    # Actual follows commanded (perfect tracking for tests)
    actual = commanded.copy()

    # Error is zero
    error = np.zeros_like(actual)

    # No interpolated samples
    interpolated = np.zeros(len(actual), dtype=bool)

    return pl.DataFrame({
        "time_s": actual_times,
        "commanded": commanded,
        "actual": actual,
        "error": error,
        "interpolated": interpolated,
    })


# Tests

def test_single_step_detection():
    """Test 1: Single step is classified as 'step' regime."""
    cmd_df = generate_single_step(
        duration_s=5.0,
        step_time_s=2.5,
        step_magnitude=10.0,
    )

    classified = classify_commanded_changes(cmd_df, min_detection_deg=0.25)

    assert len(classified) == 1, "Should detect exactly one transition"
    assert classified['regime'][0] == 'step', "Should classify as discrete step"
    assert abs(classified['magnitude'][0] - 10.0) < 0.1, "Should detect 10° step"


def test_sub_threshold_noise():
    """Test 2: Sub-threshold noise is not detected."""
    cmd_df = generate_noisy_signal(noise_amplitude=0.1)

    classified = classify_commanded_changes(cmd_df, min_detection_deg=0.25)

    assert len(classified) == 0, "Should not detect any transitions (all below threshold)"


def test_close_doublet_tracking():
    """Test 3: Close doublet (100ms apart) classified as NOT discrete steps."""
    cmd_df = generate_close_doublet(step2_time_s=2.1)  # 100ms apart

    classified = classify_commanded_changes(cmd_df, min_detection_deg=0.25)

    assert len(classified) == 2, "Should detect 2 transitions"

    # Both should NOT be discrete steps (dt gap = 100ms, at boundary → ambiguous or tracking)
    step_count = (classified['regime'] == 'step').sum()
    assert step_count == 0, f"Neither should be discrete step, got {classified['regime'].to_list()}"


def test_separated_doublet_discrete():
    """Test 4: Well-separated steps (1.5s apart) are both discrete."""
    cmd_df = generate_separated_doublet()

    classified = classify_commanded_changes(cmd_df, min_detection_deg=0.25)

    assert len(classified) == 2, "Should detect 2 transitions"

    # Both should be discrete steps (dt_before >200ms AND dt_after >200ms)
    step_count = (classified['regime'] == 'step').sum()
    assert step_count == 2, "Both should be discrete steps"


def test_tracking_sequence_detection():
    """Test 5: Rapid sequence classified as NOT discrete steps (tracking or ambiguous)."""
    cmd_df = generate_tracking_sequence(num_steps=6, step_spacing_s=0.05)

    classified = classify_commanded_changes(cmd_df, min_detection_deg=0.25)

    # Due to 10Hz sampling and 50ms spacing, not all transitions land on sample points
    assert len(classified) >= 3, f"Should detect at least 3 transitions, got {len(classified)}"

    # None should be discrete steps (rapid updates → tracking or ambiguous regime)
    step_count = (classified['regime'] == 'step').sum()
    assert step_count == 0, "Rapid sequence should not contain discrete steps"


def test_fire_event_flags():
    """Test 6: Fire event proximity flags work correctly."""
    cmd_df = generate_single_step(step_time_s=2.0, step_magnitude=10.0)
    aligned_df = create_aligned_df_synthetic(cmd_df)
    fire_df = pl.DataFrame({"time_s": [2.5]})  # Fire 500ms AFTER step

    classified = classify_commanded_changes(cmd_df, min_detection_deg=0.25)
    steps = extract_discrete_steps(
        classified,
        aligned_df,
        axis="pitch",
        min_magnitude_deg=0.5,
        fire_events=fire_df,
    )

    assert len(steps) == 1, "Should extract step"
    step = steps.row(0, named=True)

    # Step BEFORE fire should be flagged as pre_fire, not during_recoil
    assert step["pre_fire"] == True, "Should be flagged as pre-fire"
    assert step["during_recoil"] == False, "Should not be flagged as recoil"
    assert step["isolated"] == False, "Should not be isolated (fire within 1s)"

    # Test 2: Step AFTER fire
    cmd_df2 = generate_single_step(duration_s=10.0, step_time_s=3.0, step_magnitude=10.0)
    aligned_df2 = create_aligned_df_synthetic(cmd_df2)
    fire_df2 = pl.DataFrame({"time_s": [2.5]})  # Fire 500ms BEFORE step

    classified2 = classify_commanded_changes(cmd_df2, min_detection_deg=0.25)
    steps2 = extract_discrete_steps(
        classified2,
        aligned_df2,
        axis="pitch",
        min_magnitude_deg=0.5,
        fire_events=fire_df2,
    )

    assert len(steps2) == 1, "Should extract step"
    step2 = steps2.row(0, named=True)

    # Step AFTER fire should be flagged as during_recoil
    assert step2["during_recoil"] == True, "Should be flagged as recoil"
    assert step2["pre_fire"] == False, "Should not be flagged as pre-fire"
    assert step2["isolated"] == False, "Should not be isolated"


def test_boundary_truncation():
    """Test 7: Steps near recording boundaries are flagged."""
    cmd_df = generate_single_step(duration_s=5.0, step_time_s=0.5, step_magnitude=10.0)
    aligned_df = create_aligned_df_synthetic(cmd_df)

    classified = classify_commanded_changes(cmd_df, min_detection_deg=0.25)
    steps = extract_discrete_steps(
        classified,
        aligned_df,
        axis="pitch",
        min_magnitude_deg=0.5,
        pre_window_ms=200.0,
        post_window_ms=1000.0,
    )

    assert len(steps) == 1, "Should extract step"
    step = steps.row(0, named=True)

    # Step at 0.5s with 200ms pre-window → starts at 0.3s (too close to start at 0.0s)
    assert step["edge_truncated"] == True, "Should be flagged as edge truncated"


def test_interpolation_gap_flag():
    """Test 8: Steps with encoder gaps are flagged."""
    cmd_df = generate_single_step(step_time_s=2.5, step_magnitude=10.0)
    aligned_df = create_aligned_df_synthetic(cmd_df)

    # Corrupt interpolated flag to simulate encoder dropout
    aligned_df = aligned_df.with_columns([
        pl.when(pl.col("time_s") > 2.4)
          .then(True)
          .otherwise(pl.col("interpolated"))
          .alias("interpolated")
    ])

    classified = classify_commanded_changes(cmd_df, min_detection_deg=0.25)
    steps = extract_discrete_steps(
        classified,
        aligned_df,
        axis="pitch",
        min_magnitude_deg=0.5,
    )

    assert len(steps) == 1, "Should extract step"
    step = steps.row(0, named=True)

    # Epoch has >50% interpolated samples
    assert step["has_interp_gap"] == True, "Should be flagged for interpolation gap"


def test_epoch_truncation():
    """Test 9: Steps with overlapping epochs are truncated and flagged."""
    cmd_df = generate_separated_doublet(
        step1_time_s=2.0,
        step2_time_s=2.8,  # 800ms apart - second step within first's post-window
    )
    aligned_df = create_aligned_df_synthetic(cmd_df)

    classified = classify_commanded_changes(cmd_df, min_detection_deg=0.25)
    steps = extract_discrete_steps(
        classified,
        aligned_df,
        axis="pitch",
        min_magnitude_deg=0.5,
        post_window_ms=1000.0,
    )

    # First step should be truncated by second step
    first_step = steps.filter(pl.col("step_id") == 0).row(0, named=True)

    assert first_step["truncated_by_next_step"] == True, "First step should be truncated"
    assert first_step["epoch_end_time"] == pytest.approx(2.8, abs=0.1), "Epoch should end at next step"


def test_magnitude_flag():
    """Test 10: Small steps are flagged as below_min_magnitude."""
    cmd_df = generate_single_step(step_magnitude=0.3)
    aligned_df = create_aligned_df_synthetic(cmd_df)

    classified = classify_commanded_changes(cmd_df, min_detection_deg=0.25)
    steps = extract_discrete_steps(
        classified,
        aligned_df,
        axis="pitch",
        min_magnitude_deg=0.5,  # Minimum for step response analysis
    )

    assert len(steps) == 1, "Should extract step (flagged, not rejected)"
    step = steps.row(0, named=True)

    assert step["below_min_magnitude"] == True, "Should be flagged as too small for analysis"


def test_get_step_epoch():
    """Test 11: Epoch extraction returns correct window."""
    cmd_df = generate_single_step(step_time_s=2.5, step_magnitude=10.0)
    aligned_df = create_aligned_df_synthetic(cmd_df)

    classified = classify_commanded_changes(cmd_df, min_detection_deg=0.25)
    steps = extract_discrete_steps(
        classified,
        aligned_df,
        axis="pitch",
        min_magnitude_deg=0.5,
        pre_window_ms=200.0,
        post_window_ms=1000.0,
    )

    step_row = steps.row(0, named=True)
    epoch = get_step_epoch(step_row, aligned_df)

    # Check epoch has relative time column
    assert "time_rel_s" in epoch.columns, "Should have relative time column"

    # Check window spans correct range
    t_rel = epoch["time_rel_s"].to_numpy()
    assert t_rel.min() < -0.15, "Should include pre-window"
    assert t_rel.max() > 0.9, "Should include post-window"


# Real data tests

RRD_PATH = Path(__file__).parent.parent / "motor.rrd"
HAS_DATA_FILE = RRD_PATH.exists()

pytestmark_real = pytest.mark.skipif(
    not HAS_DATA_FILE,
    reason="motor.rrd not found - place file at project root to run tests"
)


@pytestmark_real
def test_real_data_classification():
    """Test 12: Real data classification into regimes."""
    data = load_recording(RRD_PATH, verbose=False)

    for axis in ["pitch", "yaw"]:
        print(f"\n{axis.upper()} classification:")
        print("=" * 60)

        cmd_df = data[f"{axis}_cmd"]

        # Classify all commanded changes
        classified = classify_commanded_changes(
            cmd_df,
            step_gap_ms=200.0,
            tracking_gap_ms=100.0,
            min_detection_deg=0.01,  # Detect everything
        )

        print(f"  Total transitions detected: {len(classified):,}")

        # Regime breakdown
        regime_counts = classified.group_by("regime").agg(pl.len().alias("count"))
        for row in regime_counts.iter_rows(named=True):
            regime = row["regime"]
            count = row["count"]
            frac = count / len(classified) * 100
            print(f"    {regime}: {count:,} ({frac:.1f}%)")

        # Check expectations
        step_count = (classified['regime'] == 'step').sum()
        tracking_count = (classified['regime'] == 'tracking').sum()

        assert step_count > 500, f"{axis}: Expected >500 discrete steps, got {step_count}"
        assert tracking_count > 10000, f"{axis}: Expected >10k tracking updates, got {tracking_count}"


@pytestmark_real
def test_real_data_discrete_steps():
    """Test 13: Extract discrete steps from real data."""
    data = load_recording(RRD_PATH, verbose=False)

    for axis in ["pitch", "yaw"]:
        print(f"\n{axis.upper()} discrete steps:")
        print("=" * 60)

        # Align signals
        aligned_df, _ = align_signals(
            data[f"{axis}_cmd"],
            data[f"{axis}_actual"],
        )

        # Classify and extract
        classified = classify_commanded_changes(
            data[f"{axis}_cmd"],
            min_detection_deg=0.25,
        )

        steps = extract_discrete_steps(
            classified,
            aligned_df,
            axis=axis,
            min_magnitude_deg=0.5,
            fire_events=data.get("fire"),
        )

        print(f"  Total discrete steps: {len(steps):,}")

        # Flag breakdown
        isolated_count = steps['isolated'].sum()
        recoil_count = steps['during_recoil'].sum()
        pre_fire_count = steps['pre_fire'].sum()

        print(f"    Isolated: {isolated_count:,} ({isolated_count/len(steps)*100:.1f}%)")
        print(f"    During recoil: {recoil_count:,} ({recoil_count/len(steps)*100:.1f}%)")
        print(f"    Pre-fire: {pre_fire_count:,} ({pre_fire_count/len(steps)*100:.1f}%)")

        # Magnitude stats for isolated steps
        isolated_steps = steps.filter(pl.col('isolated'))
        if len(isolated_steps) > 0:
            mags = isolated_steps['magnitude'].to_numpy()
            print(f"\n  Isolated step magnitudes:")
            print(f"    Min:    {mags.min():.2f}°")
            print(f"    Median: {np.median(mags):.2f}°")
            print(f"    Max:    {mags.max():.2f}°")


@pytestmark_real
def test_real_data_tracking_sequences():
    """Test 14: Extract tracking sequences from real data."""
    data = load_recording(RRD_PATH, verbose=False)

    for axis in ["pitch", "yaw"]:
        print(f"\n{axis.upper()} tracking sequences:")
        print("=" * 60)

        # Classify
        classified = classify_commanded_changes(
            data[f"{axis}_cmd"],
            min_detection_deg=0.01,
        )

        # Extract tracking sequences
        sequences = extract_tracking_sequences(
            classified,
            min_length=3,
            fire_events=data.get("fire"),
        )

        print(f"  Total sequences: {len(sequences):,}")

        if len(sequences) > 0:
            durations = sequences['duration_s'].to_numpy()
            n_updates = sequences['n_updates'].to_numpy()
            rates = sequences['mean_update_rate_hz'].to_numpy()

            print(f"\n  Sequence characteristics:")
            print(f"    Duration (median): {np.median(durations)*1000:.0f}ms")
            print(f"    Updates per sequence (median): {np.median(n_updates):.0f}")
            print(f"    Update rate (median): {np.median(rates):.1f} Hz")

            # Recoil vs clean
            recoil_seqs = sequences.filter(pl.col('during_recoil'))
            clean_seqs = sequences.filter(~pl.col('during_recoil'))

            print(f"\n  During recoil: {len(recoil_seqs):,} ({len(recoil_seqs)/len(sequences)*100:.1f}%)")
            print(f"  Clean: {len(clean_seqs):,} ({len(clean_seqs)/len(sequences)*100:.1f}%)")
