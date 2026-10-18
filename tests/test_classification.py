from thermocore.analysis.classification import (
    AnomalySeverity,
    SeverityThresholds,
    classify_by_relative_rise,
)


def test_below_general_threshold_is_normal():
    assert classify_by_relative_rise(10.0) == AnomalySeverity.NORMAL


def test_at_general_threshold_is_general():
    assert classify_by_relative_rise(35.0) == AnomalySeverity.GENERAL


def test_at_major_threshold_is_major():
    assert classify_by_relative_rise(80.0) == AnomalySeverity.MAJOR


def test_at_emergency_threshold_is_emergency():
    assert classify_by_relative_rise(95.0) == AnomalySeverity.EMERGENCY


def test_custom_thresholds_are_respected():
    custom = SeverityThresholds(general_pct=10.0, major_pct=20.0, emergency_pct=30.0)
    assert classify_by_relative_rise(15.0, thresholds=custom) == AnomalySeverity.GENERAL
    assert classify_by_relative_rise(25.0, thresholds=custom) == AnomalySeverity.MAJOR


def test_invalid_threshold_order_is_rejected():
    import pytest
    from thermocore.analysis import SeverityThresholds
    with pytest.raises(ValueError):
        SeverityThresholds(general_pct=80.0, major_pct=35.0, emergency_pct=95.0)
    with pytest.raises(ValueError):
        SeverityThresholds(general_pct=-1.0)


def test_nan_eta_is_rejected():
    import pytest
    from thermocore.analysis import classify_by_relative_rise
    with pytest.raises(ValueError):
        classify_by_relative_rise(float("nan"))
