import pytest

from thermocore.analysis import AnomalySeverity, analyze_thermogram
from thermocore.io.synthetic import make_scene_with_background


def _scene(**kw):
    d = dict(height=120, width=160, material_celsius=40.0, background_celsius=5.0,
             background_edge="top", background_fraction=0.3, hotspot_peak_celsius=90.0)
    d.update(kw)
    return make_scene_with_background(**d)


def test_analysis_finds_hotspot_inside_material_and_reference_near_material_baseline():
    scene = _scene()
    r = analyze_thermogram(scene.celsius, ambient_celsius=25.0)
    assert r.hotspot.value_celsius == pytest.approx(90.0, abs=0.5)
    assert r.reference.value_celsius == pytest.approx(40.0, abs=1.5)   # não o céu (5°C)
    assert r.delta_t_celsius == pytest.approx(50.0, abs=2.0)


def test_eta_matches_dlt664_formula_and_is_classified():
    r = analyze_thermogram(_scene().celsius, ambient_celsius=25.0)
    expected = (r.hotspot.value_celsius - r.reference.value_celsius) / (r.hotspot.value_celsius - 25.0) * 100
    assert r.eta_pct == pytest.approx(expected)
    assert r.severity in set(AnomalySeverity)


def test_ambient_from_file_is_used_and_labeled():
    r = analyze_thermogram(_scene().celsius, file_ambient_celsius=25.0)
    assert r.ambient_source == "arquivo"


def test_explicit_ambient_has_priority_over_file_value():
    r = analyze_thermogram(_scene().celsius, ambient_celsius=20.0, file_ambient_celsius=25.0)
    assert r.ambient_celsius == 20.0 and r.ambient_source == "informado pelo usuário"


def test_missing_ambient_raises_instead_of_assuming_a_value():
    with pytest.raises(ValueError):
        analyze_thermogram(_scene().celsius)


def test_hotspot_below_ambient_is_rejected():
    with pytest.raises(ValueError):
        analyze_thermogram(_scene().celsius, ambient_celsius=200.0)
