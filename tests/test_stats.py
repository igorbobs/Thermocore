import pytest

from thermocore.analysis.stats import (
    compute_stats,
    delta_t,
    relative_temperature_rise,
)
from thermocore.exceptions import EmptyRegionError
from thermocore.io.synthetic import make_flat_thermogram, make_hotspot_thermogram


def test_flat_image_has_zero_std_and_equal_max_min_mean():
    t = make_flat_thermogram(base_celsius=25.0)
    stats = compute_stats(t.celsius)
    assert stats.t_max == pytest.approx(25.0)
    assert stats.t_min == pytest.approx(25.0)
    assert stats.t_mean == pytest.approx(25.0)
    assert stats.t_std == pytest.approx(0.0, abs=1e-9)


def test_hotspot_image_max_is_close_to_peak_temperature():
    t = make_hotspot_thermogram(base_celsius=25.0, hotspot_peak_celsius=90.0)
    stats = compute_stats(t.celsius)
    assert stats.t_max == pytest.approx(90.0, abs=0.5)
    assert stats.t_min < stats.t_max


def test_compute_stats_on_empty_mask_raises_explicit_error():
    t = make_flat_thermogram(height=5, width=5)
    empty_mask = (t.celsius > 1e9)  # nenhum pixel satisfaz isso
    with pytest.raises(EmptyRegionError):
        compute_stats(t.celsius, mask=empty_mask)


def test_delta_t_between_hotspot_and_reference():
    hotspot = make_hotspot_thermogram(base_celsius=25.0, hotspot_peak_celsius=90.0)
    reference = make_flat_thermogram(base_celsius=26.0)

    hotspot_stats = compute_stats(hotspot.celsius)
    reference_stats = compute_stats(reference.celsius)

    dt = delta_t(hotspot_stats, reference_stats)
    assert dt == pytest.approx(90.0 - 26.0, abs=0.5)


def test_relative_temperature_rise_matches_manual_formula():
    hotspot = make_hotspot_thermogram(base_celsius=25.0, hotspot_peak_celsius=90.0)
    reference = make_flat_thermogram(base_celsius=26.2)

    hotspot_stats = compute_stats(hotspot.celsius)
    reference_stats = compute_stats(reference.celsius)

    ambient = 20.0
    eta = relative_temperature_rise(hotspot_stats, reference_stats, ambient)

    t1 = hotspot_stats.t_max
    t2 = reference_stats.t_max
    expected = (t1 - t2) / (t1 - ambient) * 100.0
    assert eta == pytest.approx(expected)


def test_relative_temperature_rise_raises_on_zero_division():
    hotspot = make_flat_thermogram(base_celsius=25.0)
    reference = make_flat_thermogram(base_celsius=20.0)
    hotspot_stats = compute_stats(hotspot.celsius)
    reference_stats = compute_stats(reference.celsius)

    with pytest.raises(ValueError):
        # ambiente igual à temperatura do "ponto quente" -> denominador 0
        relative_temperature_rise(hotspot_stats, reference_stats, ambient_celsius=25.0)
