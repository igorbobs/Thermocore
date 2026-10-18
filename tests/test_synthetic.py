import numpy as np
import pytest

from thermocore.io.synthetic import make_flat_thermogram, make_hotspot_thermogram


def test_flat_thermogram_is_marked_synthetic():
    t = make_flat_thermogram(base_celsius=25.0)
    assert t.is_synthetic is True
    assert t.source_id == "synthetic-flat"


def test_flat_thermogram_has_uniform_temperature():
    t = make_flat_thermogram(height=10, width=10, base_celsius=30.0)
    assert np.allclose(t.celsius, 30.0)


def test_hotspot_thermogram_peak_is_at_center_by_default():
    t = make_hotspot_thermogram(
        height=100, width=100, base_celsius=25.0, hotspot_peak_celsius=90.0
    )
    center_temp = t.celsius[50, 50]
    assert center_temp == pytest.approx(90.0)


def test_hotspot_thermogram_background_is_close_to_base():
    t = make_hotspot_thermogram(
        height=100,
        width=100,
        base_celsius=25.0,
        hotspot_center=(10, 10),
        hotspot_radius=3,
        hotspot_peak_celsius=90.0,
    )
    # Canto oposto ao hotspot deve estar essencialmente na temperatura base.
    corner_temp = t.celsius[90, 90]
    assert abs(corner_temp - 25.0) < 0.5


def test_hotspot_generation_params_are_recorded_for_reproducibility():
    t = make_hotspot_thermogram(hotspot_peak_celsius=77.0, seed=123)
    assert t.generation_params["hotspot_peak_celsius"] == 77.0
    assert t.generation_params["seed"] == 123
