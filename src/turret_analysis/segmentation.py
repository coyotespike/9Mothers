"""
Regime classification and epoch extraction.

Classifies commanded changes as discrete steps vs tracking updates,
extracts both step epochs and tracking sequences for dual analysis.

- Nothing is rejected at segmentation stage
- Everything is flagged (recoil, pre-fire, isolated, edge cases)
- Filtering happens at analysis time based on question being answered
"""

from typing import Optional

import numpy as np
import polars as pl


def classify_commanded_changes(
    cmd_df: pl.DataFrame,
    step_gap_ms: float = 200.0,
    tracking_gap_ms: float = 100.0,
    min_detection_deg: float = 0.01,
) -> pl.DataFrame:
    """
    Classify each commanded change as: step | tracking | ambiguous.

    Classification logic:
    - DISCRETE STEP: dt_before > step_gap_ms AND dt_after > step_gap_ms
      (commanded holds steady before and after → motor can settle)

    - TRACKING UPDATE: dt_before < tracking_gap_ms OR dt_after < tracking_gap_ms
      (rapid updates → motor always slewing, can't settle)

    - AMBIGUOUS: everything else
      (edge cases, flag for manual inspection)

    Args:
        cmd_df: Commanded signal DataFrame with columns ['time_s', 'value']
        step_gap_ms: Minimum gap to classify as discrete step (milliseconds)
        tracking_gap_ms: Maximum gap to classify as tracking update (milliseconds)
        min_detection_deg: Minimum change to detect (filters micro-adjustments)

    Returns:
        DataFrame with columns:
            ['transition_id', 'time_s', 'value_before', 'value_after', 'magnitude',
             'dt_before_ms', 'dt_after_ms', 'regime']
        where regime ∈ {'step', 'tracking', 'ambiguous'}
    """
    times = cmd_df["time_s"].to_numpy()
    values = cmd_df["value"].to_numpy()

    if len(values) < 2:
        return pl.DataFrame({
            'transition_id': [],
            'time_s': [],
            'value_before': [],
            'value_after': [],
            'magnitude': [],
            'dt_before_ms': [],
            'dt_after_ms': [],
            'regime': [],
        })

    # Detect all changes > min_detection_deg
    diffs = np.diff(values)
    abs_diffs = np.abs(diffs)

    transition_indices = np.where(abs_diffs > min_detection_deg)[0] + 1  # +1 because diff shifts

    if len(transition_indices) == 0:
        return pl.DataFrame({
            'transition_id': [],
            'time_s': [],
            'value_before': [],
            'value_after': [],
            'magnitude': [],
            'dt_before_ms': [],
            'dt_after_ms': [],
            'regime': [],
        })

    # Compute time gaps before and after each transition
    dt_before_ms = np.zeros(len(transition_indices))
    dt_after_ms = np.zeros(len(transition_indices))

    for i, idx in enumerate(transition_indices):
        # Time since previous transition
        if i > 0:
            dt_before_ms[i] = (times[idx] - times[transition_indices[i-1]]) * 1000
        else:
            dt_before_ms[i] = 9999.0  # First transition - assume stable before

        # Time until next transition
        if i < len(transition_indices) - 1:
            dt_after_ms[i] = (times[transition_indices[i+1]] - times[idx]) * 1000
        else:
            dt_after_ms[i] = 9999.0  # Last transition - assume stable after

    # Classify regime
    regimes = []
    for dt_before, dt_after in zip(dt_before_ms, dt_after_ms):
        if dt_before >= step_gap_ms and dt_after >= step_gap_ms:
            regime = 'step'
        elif dt_before < tracking_gap_ms or dt_after < tracking_gap_ms:
            regime = 'tracking'
        else:
            regime = 'ambiguous'
        regimes.append(regime)

    # Build output DataFrame
    return pl.DataFrame({
        'transition_id': np.arange(len(transition_indices)),
        'time_s': times[transition_indices],
        'value_before': values[transition_indices - 1],
        'value_after': values[transition_indices],
        'magnitude': abs_diffs[transition_indices - 1],
        'dt_before_ms': dt_before_ms,
        'dt_after_ms': dt_after_ms,
        'regime': regimes,
    })


def extract_discrete_steps(
    classified: pl.DataFrame,
    aligned_df: pl.DataFrame,
    axis: str,
    min_magnitude_deg: float = 0.5,
    fire_events: Optional[pl.DataFrame] = None,
    pre_window_ms: float = 200.0,
    post_window_ms: float = 1000.0,
) -> pl.DataFrame:
    """
    Extract discrete step catalog with quality flags (no rejection).

    All steps are kept. Quality flags allow filtering at analysis time.

    Args:
        classified: Output from classify_commanded_changes()
        aligned_df: Aligned signal DataFrame from Phase 3
        axis: "pitch" or "yaw"
        min_magnitude_deg: Minimum magnitude for step response analysis
        fire_events: Fire event timestamps (optional)
        pre_window_ms: Pre-step window duration
        post_window_ms: Post-step window duration

    Returns:
        DataFrame with step catalog:
            ['step_id', 'axis', 'time_s', 'value_before', 'value_after', 'magnitude',
             'direction', 'epoch_start_idx', 'epoch_end_idx', 'epoch_start_time',
             'epoch_end_time', 'epoch_duration_s', 'dt_before_ms', 'dt_after_ms',
             'during_recoil', 'pre_fire', 'isolated', 'edge_truncated',
             'has_interp_gap', 'below_min_magnitude', 'truncated_by_next_step']
    """
    # Filter to discrete steps only
    steps = classified.filter(pl.col('regime') == 'step')

    if len(steps) == 0:
        return pl.DataFrame()

    catalog_rows = []

    aligned_times = aligned_df["time_s"].to_numpy()
    t_min = aligned_times[0]
    t_max = aligned_times[-1]

    fire_times = fire_events["time_s"].to_numpy() if fire_events is not None else np.array([])

    pre_s = pre_window_ms / 1000.0
    post_s = post_window_ms / 1000.0

    for row in steps.iter_rows(named=True):
        step_time = row['time_s']
        magnitude = row['magnitude']
        value_before = row['value_before']
        value_after = row['value_after']

        # Compute epoch boundaries
        epoch_start_time = step_time - pre_s
        epoch_end_time = step_time + post_s

        # Find indices in aligned DataFrame
        epoch_start_idx = np.searchsorted(aligned_times, epoch_start_time, side='left')
        epoch_end_idx = np.searchsorted(aligned_times, epoch_end_time, side='right')

        # Quality flags (no rejection)
        below_min_magnitude = magnitude < min_magnitude_deg
        edge_truncated = (epoch_start_time < t_min + 1.0) or (epoch_end_time > t_max - 1.0)

        # Interpolation gap check
        has_interp_gap = False
        if not edge_truncated:
            epoch_slice = aligned_df[epoch_start_idx:epoch_end_idx]
            n_interpolated = epoch_slice["interpolated"].sum()
            epoch_length = len(epoch_slice)
            if epoch_length > 0 and (n_interpolated / epoch_length) > 0.5:
                has_interp_gap = True

        # Fire event proximity flags
        during_recoil = False
        pre_fire = False
        isolated = True

        if len(fire_times) > 0:
            time_since_fire = step_time - fire_times  # Positive = after fire

            # During recoil: 0-500ms after any fire (inclusive)
            if np.any((time_since_fire >= 0) & (time_since_fire <= 0.5)):
                during_recoil = True
                isolated = False

            # Pre-fire: 500ms before any fire (inclusive)
            if np.any((time_since_fire >= -0.5) & (time_since_fire < 0)):
                pre_fire = True
                isolated = False

            # Isolated: no fire within ±1s
            if np.any(np.abs(time_since_fire) < 1.0):
                isolated = False

        # Check for truncation by next step
        truncated_by_next_step = False
        actual_end_time = epoch_end_time

        # Find if any other step falls within this epoch
        other_steps = steps.filter(
            (pl.col('time_s') != step_time) &
            (pl.col('time_s') > step_time) &
            (pl.col('time_s') < epoch_end_time)
        )

        if len(other_steps) > 0:
            next_step_time = other_steps['time_s'].min()
            actual_end_time = next_step_time
            epoch_end_idx = np.searchsorted(aligned_times, actual_end_time, side='left')
            truncated_by_next_step = True

        # Direction
        direction = "up" if value_after > value_before else "down"

        catalog_rows.append({
            "step_id": row['transition_id'],
            "axis": axis,
            "time_s": step_time,
            "value_before": value_before,
            "value_after": value_after,
            "magnitude": magnitude,
            "direction": direction,
            "epoch_start_idx": epoch_start_idx,
            "epoch_end_idx": epoch_end_idx,
            "epoch_start_time": epoch_start_time,
            "epoch_end_time": actual_end_time,
            "epoch_duration_s": actual_end_time - epoch_start_time,
            "dt_before_ms": row['dt_before_ms'],
            "dt_after_ms": row['dt_after_ms'],
            "during_recoil": during_recoil,
            "pre_fire": pre_fire,
            "isolated": isolated,
            "edge_truncated": edge_truncated,
            "has_interp_gap": has_interp_gap,
            "below_min_magnitude": below_min_magnitude,
            "truncated_by_next_step": truncated_by_next_step,
        })

    return pl.DataFrame(catalog_rows)


def extract_tracking_sequences(
    classified: pl.DataFrame,
    min_length: int = 3,
    fire_events: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Group tracking updates into continuous sequences.

    A tracking sequence is a run of consecutive 'tracking' regime transitions.

    Args:
        classified: Output from classify_commanded_changes()
        min_length: Minimum number of transitions to form a sequence
        fire_events: Fire event timestamps (optional)

    Returns:
        DataFrame with tracking sequences:
            ['sequence_id', 'start_time', 'end_time', 'duration_s', 'n_updates',
             'total_magnitude', 'mean_update_rate_hz', 'during_recoil']
    """
    tracking = classified.filter(pl.col('regime') == 'tracking')

    if len(tracking) < min_length:
        return pl.DataFrame()

    sequences = []
    sequence_id = 0

    # Find runs of consecutive tracking updates
    times = tracking['time_s'].to_numpy()
    magnitudes = tracking['magnitude'].to_numpy()

    # Group by time gaps (new sequence if gap >500ms)
    time_gaps = np.diff(times)
    break_points = np.where(time_gaps > 0.5)[0] + 1
    break_points = np.concatenate([[0], break_points, [len(times)]])

    fire_times = fire_events["time_s"].to_numpy() if fire_events is not None else np.array([])

    for i in range(len(break_points) - 1):
        start_idx = break_points[i]
        end_idx = break_points[i + 1]

        if end_idx - start_idx < min_length:
            continue

        start_time = times[start_idx]
        end_time = times[end_idx - 1]
        duration = end_time - start_time
        n_updates = end_idx - start_idx
        total_magnitude = magnitudes[start_idx:end_idx].sum()
        mean_rate = n_updates / duration if duration > 0 else 0

        # Check if during recoil window
        during_recoil = False
        if len(fire_times) > 0:
            # Check if sequence overlaps with any 0-500ms post-fire window
            for fire_t in fire_times:
                if (start_time >= fire_t) and (start_time < fire_t + 0.5):
                    during_recoil = True
                    break
                if (end_time >= fire_t) and (end_time < fire_t + 0.5):
                    during_recoil = True
                    break

        sequences.append({
            'sequence_id': sequence_id,
            'start_time': start_time,
            'end_time': end_time,
            'duration_s': duration,
            'n_updates': n_updates,
            'total_magnitude': total_magnitude,
            'mean_update_rate_hz': mean_rate,
            'during_recoil': during_recoil,
        })

        sequence_id += 1

    return pl.DataFrame(sequences)


def get_step_epoch(
    step_row: dict,
    aligned_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Extract epoch window data for a discrete step.

    Args:
        step_row: Single row from step catalog (as dict)
        aligned_df: Aligned signal DataFrame

    Returns:
        DataFrame containing the windowed slice with time relative to step
    """
    start_idx = step_row["epoch_start_idx"]
    end_idx = step_row["epoch_end_idx"]
    t_command = step_row["time_s"]

    epoch = aligned_df[start_idx:end_idx].clone()

    # Add relative time column (time since command)
    epoch = epoch.with_columns([
        (pl.col("time_s") - t_command).alias("time_rel_s")
    ])

    return epoch


def get_tracking_epoch(
    sequence_row: dict,
    aligned_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Extract epoch window data for a tracking sequence.

    Args:
        sequence_row: Single row from tracking sequence catalog (as dict)
        aligned_df: Aligned signal DataFrame

    Returns:
        DataFrame containing the sequence window with relative time
    """
    start_time = sequence_row["start_time"]
    end_time = sequence_row["end_time"]

    # Extract window with small padding
    mask = (aligned_df["time_s"] >= start_time - 0.1) & (aligned_df["time_s"] <= end_time + 0.1)
    epoch = aligned_df.filter(mask).clone()

    # Add relative time column (time since sequence start)
    epoch = epoch.with_columns([
        (pl.col("time_s") - start_time).alias("time_rel_s")
    ])

    return epoch


