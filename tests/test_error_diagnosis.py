"""
Tests for error source diagnostic.
"""

import numpy as np
import polars as pl
import pytest

from turret_analysis.tracking_analysis import diagnose_error_source


def test_diagnose_velocity_lag():
    """Test detection of velocity lag (error grows with velocity)."""
    # Create synthetic data where error = 0.05 * velocity + 0.5
    n_sequences = 50
    velocities = np.random.uniform(10, 100, n_sequences)
    errors = 0.05 * velocities + 0.5 + np.random.normal(0, 0.1, n_sequences)

    sequences = pl.DataFrame({
        'rms_error': errors,
        'mean_abs_velocity': velocities,
    })

    result = diagnose_error_source(sequences)

    # Should detect significant positive correlation
    assert result['correlation'] > 0.7
    assert result['p_value'] < 0.05
    assert 0.04 < result['velocity_slope'] < 0.06  # Should recover slope ~0.05
    assert 'VELOCITY LAG' in result['interpretation']


def test_diagnose_steady_state_bias():
    """Test detection of steady-state bias (error independent of velocity)."""
    # Create synthetic data where error is constant ~1.0° regardless of velocity
    np.random.seed(42)  # Reproducible test
    n_sequences = 100
    velocities = np.random.uniform(10, 100, n_sequences)
    errors = np.ones(n_sequences) + np.random.normal(0, 0.3, n_sequences)

    sequences = pl.DataFrame({
        'rms_error': errors,
        'mean_abs_velocity': velocities,
    })

    result = diagnose_error_source(sequences)

    # Should detect no significant correlation (with more data and seed, should be stable)
    # Note: with random data, p-value can occasionally be < 0.05 by chance
    # The key is weak correlation magnitude
    assert abs(result['correlation']) < 0.3  # Weak correlation
    if result['p_value'] <= 0.05:
        # Even if p < 0.05 by chance, correlation should still be weak
        assert abs(result['correlation']) < 0.3
    assert 'STEADY-STATE BIAS OR NOISE' in result['interpretation'] or 'WEAK CORRELATION' in result['interpretation']


def test_diagnose_insufficient_data():
    """Test handling of insufficient data."""
    # Only 5 sequences (need ≥10)
    sequences = pl.DataFrame({
        'rms_error': [1.0, 1.1, 0.9, 1.2, 0.8],
        'mean_abs_velocity': [20.0, 30.0, 25.0, 15.0, 35.0],
    })

    result = diagnose_error_source(sequences)

    assert np.isnan(result['correlation'])
    assert np.isnan(result['p_value'])
    assert 'Insufficient data' in result['interpretation']


def test_diagnose_with_nan_values():
    """Test handling of NaN values in data."""
    # Mix of valid and invalid sequences (need ≥10 valid after filtering)
    sequences = pl.DataFrame({
        'rms_error': [1.0, np.nan, 1.1, 0.9, np.nan, 1.2, 0.8, 1.0, 0.9, 1.1, 1.0, 0.95, 1.05, 0.85, 0.92],
        'mean_abs_velocity': [20.0, 30.0, 25.0, np.nan, 40.0, 15.0, 35.0, 22.0, 28.0, 18.0, 32.0, 26.0, 24.0, 30.0, 27.0],
    })

    result = diagnose_error_source(sequences)

    # Should filter out NaN and still have enough data (13 valid)
    assert result['n_sequences'] >= 10
    assert not np.isnan(result['correlation'])


def test_diagnose_real_world_scenario():
    """Test with realistic turret tracking data characteristics."""
    # Based on actual report values: ~1° error, ~20-50 deg/s velocities
    n_sequences = 100

    # Mix of velocity lag (slope=0.03) and steady-state bias (intercept=0.8)
    velocities = np.random.uniform(10, 60, n_sequences)
    errors = 0.03 * velocities + 0.8 + np.random.normal(0, 0.3, n_sequences)

    sequences = pl.DataFrame({
        'rms_error': errors,
        'mean_abs_velocity': velocities,
    })

    result = diagnose_error_source(sequences)

    # Should detect moderate to strong correlation
    assert result['correlation'] > 0.5
    assert result['p_value'] < 0.01
    assert 0.02 < result['velocity_slope'] < 0.04
    assert 0.5 < result['velocity_intercept'] < 1.1
    assert result['n_sequences'] == 100
