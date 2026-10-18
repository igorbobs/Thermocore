"""Salvaguardas da análise de inspeção (versão 0.2.0): ΔT mínimo, faixa de η, avisos."""

import numpy as np
import pytest

from thermocore.analysis import AnomalySeverity, analyze_thermogram
from thermocore.exceptions import (
    AmbientTemperatureUnavailableError,
    HotspotBelowAmbientError,
    ThermocoreError,
)
from thermocore.io.synthetic import make_scene_with_background


def _scene(**kw):
    d = dict(height=120, width=160, material_celsius=40.0, background_celsius=5.0,
             background_edge="top", background_fraction=0.3, hotspot_peak_celsius=90.0)
    d.update(kw)
    return make_scene_with_background(**d)


def test_missing_ambient_raises_domain_error_that_is_also_value_error():
    with pytest.raises(AmbientTemperatureUnavailableError) as exc:
        analyze_thermogram(_scene().celsius)
    assert isinstance(exc.value, ThermocoreError) and isinstance(exc.value, ValueError)


def test_hotspot_below_ambient_raises_domain_error():
    with pytest.raises(HotspotBelowAmbientError) as exc:
        analyze_thermogram(_scene().celsius, ambient_celsius=200.0)
    assert isinstance(exc.value, ThermocoreError) and isinstance(exc.value, ValueError)


def test_small_delta_t_suppresses_severity_and_warns():
    # material quase uniforme (pico ~1 °C acima) e ambiente próximo: η alto, mas ΔT pequeno
    scene = _scene(hotspot_peak_celsius=41.2)
    r = analyze_thermogram(scene.celsius, ambient_celsius=39.0, min_delta_t_celsius=2.0)
    assert r.delta_t_celsius < 2.0
    assert r.severity == AnomalySeverity.NORMAL
    assert r.severity_by_eta != AnomalySeverity.NORMAL      # η sozinho classificaria
    assert any("menor que o mínimo" in w for w in r.warnings)


def test_min_delta_t_can_be_disabled():
    scene = _scene(hotspot_peak_celsius=41.2)
    r = analyze_thermogram(scene.celsius, ambient_celsius=39.0, min_delta_t_celsius=0.0)
    assert r.severity == r.severity_by_eta


def test_eta_by_percentile_is_monotonic_and_matches_main_result():
    r = analyze_thermogram(_scene().celsius, ambient_celsius=25.0)
    etas = [e.eta_pct for e in r.eta_by_percentile]
    assert etas == sorted(etas, reverse=True)               # referência mais quente → η menor
    p10 = next(e for e in r.eta_by_percentile if e.percentile == 10.0)
    assert p10.eta_pct == pytest.approx(r.eta_pct)
    lo, hi = r.eta_range_pct
    assert lo == pytest.approx(min(etas)) and hi == pytest.approx(max(etas))


def test_unstable_class_is_flagged():
    # material com gradiente suave: a classe muda conforme o percentil de referência
    c = np.full((100, 100), 5.0)
    yy, xx = np.mgrid[0:100, 0:100]
    c[20:, :] = 30.0 + 40.0 * (yy[20:, :] - 20) / 80.0       # 30 → 70 °C
    r = analyze_thermogram(c, ambient_celsius=25.0)
    assert not r.severity_is_stable
    assert r.severity_range[0] != r.severity_range[1]
    assert any("depende da referência" in w for w in r.warnings)


def test_reference_below_ambient_and_eta_over_100_are_flagged():
    # material frio (20 °C) com ambiente a 34 °C e um ponto quente
    scene = _scene(material_celsius=20.0, hotspot_peak_celsius=50.0)
    r = analyze_thermogram(scene.celsius, ambient_celsius=34.0)
    assert r.reference.value_celsius < 34.0
    assert any("abaixo da temperatura ambiente" in w for w in r.warnings)
    assert r.eta_pct > 100.0 and any("excede 100%" in w for w in r.warnings)


def test_hot_pixels_outside_material_are_flagged():
    c = _scene().celsius.copy()
    c[2, 2] = 120.0                                         # pico isolado no "céu" (fora do material)
    r = analyze_thermogram(c, ambient_celsius=25.0)
    assert r.global_max_celsius == pytest.approx(120.0)
    assert r.hotspot.value_celsius < 120.0
    assert any("fora da região de material" in w for w in r.warnings)


def test_parameters_are_recorded_for_traceability():
    r = analyze_thermogram(_scene().celsius, ambient_celsius=25.0, erosion_iterations=2,
                           min_delta_t_celsius=3.0)
    assert r.parameters["erosion_iterations"] == 2
    assert r.parameters["min_delta_t_celsius"] == 3.0
    assert r.parameters["limiares_eta_pct"] == [35.0, 80.0, 95.0]


def test_optional_median_filter_removes_single_pixel_spike():
    c = _scene(hotspot_peak_celsius=45.0).celsius.copy()
    c[80, 80] = 150.0                                       # pixel quente isolado dentro do material
    raw = analyze_thermogram(c, ambient_celsius=25.0)
    filt = analyze_thermogram(c, ambient_celsius=25.0, denoise_median_size=3)
    assert raw.hotspot.value_celsius == pytest.approx(150.0)
    assert filt.hotspot.value_celsius < 60.0


def test_mta_exceeded_warns_but_does_not_change_class():
    sc = _scene()
    base = analyze_thermogram(sc.celsius, ambient_celsius=25.0)
    r = analyze_thermogram(sc.celsius, ambient_celsius=25.0, max_admissible_celsius=70.0)
    assert r.mta_exceeded is True
    assert any("máxima temperatura admissível" in w for w in r.warnings)
    assert r.severity == base.severity                      # só avisa
    assert r.parameters["max_admissible_celsius"] == 70.0


def test_mta_not_exceeded_or_not_informed():
    sc = _scene()
    ok = analyze_thermogram(sc.celsius, ambient_celsius=25.0, max_admissible_celsius=200.0)
    none = analyze_thermogram(sc.celsius, ambient_celsius=25.0)
    assert ok.mta_exceeded is False and not any("admissível" in w for w in ok.warnings)
    assert none.mta_exceeded is None
