"""
Tests for cross-correlation lag measurement.
"""

import numpy as np
import polars as pl
import pytest

from turret_analysis.xcorr import (
    compute_whole_trace_lag,
    analyze_whole_trace_lag,
    validate_regime_consistency,
)


def test_compute_whole_trace_lag_zero_lag():
    """Test that zero lag is detected when signals are identical."""
    t = np.linspace(0, 10, 1000)
    dt = np.median(np.diff(t))
    signal = np.sin(2 * np.pi * t)

    lag_s, corr = compute_whole_trace_lag(signal, signal, dt)

    # Should be near zero (within one sample period)
    assert abs(lag_s) < dt * 2
    # Correlation should be high for identical signals
    assert corr > 0.9


def test_compute_whole_trace_lag_known_delay():
    """Test that a known delay is correctly measured."""
    t = np.linspace(0, 10, 1000)
    dt = np.median(np.diff(t))
    signal = np.sin(2 * np.pi * t)

    # Create delayed signal
    delay_samples = 10
    delay_s = delay_samples * dt

    delayed_signal = np.roll(signal, delay_samples)

    lag_s, corr = compute_whole_trace_lag(signal, delayed_signal, dt)

    # Should detect the delay (within a few sample periods)
    assert abs(lag_s - delay_s) < dt * 3
    # Correlation should still be high
    assert corr > 0.8


def test_compute_whole_trace_lag_negative_lag():
    """Test that negative lag is detected when actual leads commanded."""
    t = np.linspace(0, 10, 1000)
    dt = np.median(np.diff(t))
    signal = np.sin(2 * np.pi * t)

    # Create advanced signal (actual leads command)
    advance_samples = 5
    advance_s = advance_samples * dt

    advanced_signal = np.roll(signal, -advance_samples)

    lag_s, corr = compute_whole_trace_lag(signal, advanced_signal, dt)

    # Should detect negative lag
    assert lag_s < 0
    assert abs(lag_s + advance_s) < dt * 3


def test_analyze_whole_trace_lag():
    """Test the high-level analysis function with aligned DataFrame."""
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * t)

    aligned_df = pl.DataFrame({
        "time_s": t,
        "commanded": signal,
        "actual": signal,
    })

    result = analyze_whole_trace_lag(aligned_df)

    assert 'lag_ms' in result
    assert 'correlation' in result
    assert 'n_samples' in result
    assert result['n_samples'] == 1000
    assert abs(result['lag_ms']) < 1.0  # Should be near zero for identical signals
    assert result['correlation'] > 0.9


def test_analyze_whole_trace_lag_empty():
    """Test handling of empty dataframes."""
    empty_df = pl.DataFrame({
        "time_s": [],
        "commanded": [],
        "actual": [],
    })

    result = analyze_whole_trace_lag(empty_df)

    assert np.isnan(result['lag_ms'])
    assert np.isnan(result['correlation'])
    assert result['n_samples'] == 0


def test_validate_regime_consistency_valid():
    """Test validation when measurements are mutually consistent.

    Based on actual report values:
    - Dead time: 35ms (pitch), 25ms (yaw)
    - Rise time: 120ms (pitch), 50ms (yaw)
    - Tracking lag: ~0ms (spatial offset, not temporal)
    - Tracking RMS: ~0.9-1.4°
    """
    # Pitch axis example
    result = validate_regime_consistency(
        whole_trace_lag_ms=50.0,  # Between 0 and (35+120=155)
        dead_time_ms=35.0,
        rise_time_ms=120.0,
        tracking_lag_ms=5.0,  # Near zero
        tracking_rms_error_deg=0.9,
    )

    assert result['valid'] == True
    assert result['check1_ordering'] == True
    assert result['check2_tracking_near_zero'] == True
    assert result['check3_spatial_error_exists'] == True
    assert result['step_total_lag_ms'] == 155.0


def test_validate_regime_consistency_whole_trace_too_fast():
    """Test detection when whole-trace lag is impossibly fast."""
    # Whole-trace lag faster than tracking lag (impossible)
    result = validate_regime_consistency(
        whole_trace_lag_ms=10.0,  # Less than tracking_lag
        dead_time_ms=35.0,
        rise_time_ms=120.0,
        tracking_lag_ms=45.0,
        tracking_rms_error_deg=0.9,
    )

    assert result['valid'] == False
    assert result['check1_ordering'] == False


def test_validate_regime_consistency_whole_trace_too_slow():
    """Test detection when whole-trace exceeds step total lag."""
    # Whole-trace lag slower than step response (impossible)
    result = validate_regime_consistency(
        whole_trace_lag_ms=200.0,  # Greater than step total (155ms)
        dead_time_ms=35.0,
        rise_time_ms=120.0,
        tracking_lag_ms=5.0,
        tracking_rms_error_deg=0.9,
    )

    assert result['valid'] == False
    assert result['check1_ordering'] == False


def test_validate_regime_consistency_tracking_not_spatial():
    """Test detection when tracking shows temporal lag instead of spatial offset."""
    # Tracking lag is high (temporal, not spatial)
    result = validate_regime_consistency(
        whole_trace_lag_ms=50.0,
        dead_time_ms=35.0,
        rise_time_ms=120.0,
        tracking_lag_ms=50.0,  # Should be near zero for spatial offset
        tracking_rms_error_deg=0.9,
    )

    assert result['valid'] == False
    assert result['check2_tracking_near_zero'] == False


def test_validate_regime_consistency_no_spatial_error():
    """Test detection when tracking has no spatial error (unexpected)."""
    # Tracking error is too small (no spatial offset)
    result = validate_regime_consistency(
        whole_trace_lag_ms=50.0,
        dead_time_ms=35.0,
        rise_time_ms=120.0,
        tracking_lag_ms=5.0,
        tracking_rms_error_deg=0.1,  # Too small, should be ~1°
    )

    assert result['valid'] == False
    assert result['check3_spatial_error_exists'] == False


def test_validate_regime_consistency_interpretation():
    """Test that interpretation message is generated correctly."""
    result = validate_regime_consistency(
        whole_trace_lag_ms=50.0,
        dead_time_ms=35.0,
        rise_time_ms=120.0,
        tracking_lag_ms=5.0,
        tracking_rms_error_deg=0.9,
    )

    assert 'interpretation' in result
    assert 'Regime-dependent' in result['interpretation']
    assert 'spatial offset' in result['interpretation'].lower()
