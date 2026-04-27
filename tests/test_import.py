"""
Smoke test to verify package installation and basic imports.
"""

import turret_analysis


def test_package_has_version():
    """Verify the package has a version attribute."""
    assert hasattr(turret_analysis, "__version__")
    assert isinstance(turret_analysis.__version__, str)
    assert len(turret_analysis.__version__) > 0


def test_package_imports():
    """Verify basic package imports work."""
    import turret_analysis

    # Should not raise
    assert turret_analysis is not None
