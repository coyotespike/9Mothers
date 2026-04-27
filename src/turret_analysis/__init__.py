"""
Turret Motor Analysis Package

Analysis tools for motor telemetry from anti-drone turret systems.
Provides latency decomposition, firing disturbance characterization,
and mission-level capability assessment.
"""

__version__ = "0.1.0"

# Main exports
from turret_analysis.io import load_recording
from turret_analysis.alignment import align_signals, compute_aligned_signals, compute_error_statistics
from turret_analysis.segmentation import (
    classify_commanded_changes,
    extract_discrete_steps,
    extract_tracking_sequences,
    get_step_epoch,
    get_tracking_epoch,
)

__all__ = [
    # I/O
    "load_recording",
    # Alignment
    "align_signals",
    "compute_aligned_signals",
    "compute_error_statistics",
    # Segmentation
    "classify_commanded_changes",
    "extract_discrete_steps",
    "extract_tracking_sequences",
    "get_step_epoch",
    "get_tracking_epoch",
]
