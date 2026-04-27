"""
Data loading module for Rerun recordings.

Handles loading motor.rrd and converting to Polars DataFrames with proper
timestamp handling (Rerun uses nanoseconds internally).
"""

from pathlib import Path
from typing import Dict

import polars as pl
import pyarrow as pa
import rerun as rr


def load_recording(
    rrd_path: str | Path, verbose: bool = True
) -> Dict[str, pl.DataFrame]:
    """
    Load motor telemetry from Rerun recording file.

    Args:
        rrd_path: Path to the .rrd file
        verbose: If True, print loading progress and summary statistics

    Returns:
        Dictionary with keys:
            - 'pitch_cmd': Commanded pitch position
            - 'pitch_actual': Actual pitch position
            - 'yaw_cmd': Commanded yaw position
            - 'yaw_actual': Actual yaw position
            - 'fire': Fire trigger events
            - 'muzzle': Muzzle trigger events
            - 'impact': Impact trigger events

        Each DataFrame has columns:
            - 'time_s': Timestamp in seconds (converted from nanoseconds)
            - 'value': Signal value (degrees for positions, event marker for triggers)

    Raises:
        FileNotFoundError: If rrd_path does not exist
        ValueError: If recording structure is unexpected
    """
    rrd_path = Path(rrd_path)
    if not rrd_path.exists():
        raise FileNotFoundError(f"Recording file not found: {rrd_path}")

    if verbose:
        print(f"Loading recording: {rrd_path}")
        print(f"File size: {rrd_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Load the recording using Rerun bindings
    recording = rr.bindings.load_recording(str(rrd_path))

    if verbose:
        print(f"Application ID: {recording.application_id()}")
        print(f"Recording ID: {recording.recording_id()}")

    # Entity path mapping
    path_mapping = {
        "/motors/position/pitch/target": "pitch_cmd",
        "/motors/position/pitch/current": "pitch_actual",
        "/motors/position/yaw/target": "yaw_cmd",
        "/motors/position/yaw/current": "yaw_actual",
        "/trigger/fire": "fire",
        "/trigger/muzzle": "muzzle",
        "/trigger/impact": "impact",
    }

    # Collect data for each signal
    result = {}

    for entity_path, signal_name in path_mapping.items():
        # Collect all chunks for this entity
        batches = []
        for chunk in recording.chunks():
            if (
                chunk.entity_path == entity_path
                and chunk.num_rows > 0
                and len(chunk.timeline_names) > 0
            ):
                batches.append(chunk.to_record_batch())

        if not batches:
            if verbose:
                print(f"Warning: No data found for {entity_path}")
            continue

        # Convert batches to tables and concatenate
        tables = [pa.Table.from_batches([batch]) for batch in batches]
        combined_table = pa.concat_tables(tables)

        # Drop the rerun.controls.RowId column (extension type not supported by Polars)
        cols_to_keep = [col for col in combined_table.column_names if not col.startswith("rerun.controls")]
        combined_table = combined_table.select(cols_to_keep)

        # Convert to Polars
        df = pl.from_arrow(combined_table)

        # Extract time and value columns
        # Time is in nanoseconds, convert to seconds
        # Value is in Scalars:scalars column (list of doubles, take first element)
        if "log_time" not in df.columns:
            if verbose:
                print(f"Warning: No log_time column for {entity_path}")
            continue

        # Check if this is a scalar signal or event
        if "Scalars:scalars" in df.columns:
            # Scalar signal - extract first element from list
            processed_df = df.select(
                [
                    (pl.col("log_time").cast(pl.Int64) / 1e9).alias("time_s"),
                    pl.col("Scalars:scalars").list.get(0).alias("value"),
                ]
            ).sort("time_s")
        else:
            # Event marker - just record timestamp, value is 1.0 for presence
            processed_df = df.select(
                [
                    (pl.col("log_time").cast(pl.Int64) / 1e9).alias("time_s"),
                    pl.lit(1.0).alias("value"),
                ]
            ).sort("time_s")

        result[signal_name] = processed_df

        if verbose:
            print(f"\n{signal_name} ({entity_path}):")
            print(f"  Samples: {processed_df.height}")
            time_min = processed_df["time_s"].min()
            time_max = processed_df["time_s"].max()
            print(f"  Time range: {time_min:.3f} - {time_max:.3f} s ({time_max - time_min:.1f} s duration)")
            if signal_name not in ["fire", "muzzle", "impact"]:
                val_min = processed_df["value"].min()
                val_max = processed_df["value"].max()
                print(f"  Value range: {val_min:.3f} - {val_max:.3f} deg")
            else:
                print(f"  Event count: {processed_df.height}")

    # Validate we got the expected signals
    expected_signals = ["pitch_cmd", "pitch_actual", "yaw_cmd", "yaw_actual"]
    missing = [s for s in expected_signals if s not in result]
    if missing:
        raise ValueError(f"Missing required signals: {missing}")

    if verbose:
        print(f"\nLoading complete. Extracted {len(result)} signals.")

    return result


def get_sample_rate(df: pl.DataFrame, percentile: float = 50.0) -> float:
    """
    Estimate sample rate from timestamp deltas.

    Args:
        df: DataFrame with 'time_s' column
        percentile: Percentile of time deltas to use (median=50.0)

    Returns:
        Estimated sample rate in Hz
    """
    if df.height < 2:
        return 0.0

    # Compute time deltas
    deltas = df["time_s"].diff().drop_nulls()

    if len(deltas) == 0:
        return 0.0

    # Use percentile to be robust to outliers
    median_delta = deltas.quantile(percentile / 100.0)

    if median_delta == 0:
        return 0.0

    return 1.0 / median_delta


def validate_monotonic_time(df: pl.DataFrame) -> bool:
    """
    Check if time column is monotonically increasing.

    Args:
        df: DataFrame with 'time_s' column

    Returns:
        True if timestamps are monotonically increasing
    """
    if df.height < 2:
        return True

    deltas = df["time_s"].diff().drop_nulls()
    return (deltas >= 0).all()
