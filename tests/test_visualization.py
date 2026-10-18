import numpy as np
import pytest
from PIL import Image

from thermocore.analysis.classification import classify_by_relative_rise
from thermocore.analysis.stats import compute_stats, locate_extreme, relative_temperature_rise
from thermocore.io.synthetic import make_hotspot_thermogram
from thermocore.visualization.annotate import save_annotated_image
from thermocore.visualization.render import render_thermal_image


def test_render_thermal_image_returns_rgb_uint8():
    t = make_hotspot_thermogram(height=50, width=60, hotspot_peak_celsius=80.0)
    rgb = render_thermal_image(t.celsius)
    assert rgb.shape == (50, 60, 3)
    assert rgb.dtype == np.uint8


def test_render_thermal_image_handles_uniform_image_without_error():
    flat = np.full((10, 10), 25.0)
    rgb = render_thermal_image(flat)
    assert rgb.shape == (10, 10, 3)


def test_locate_extreme_finds_hotspot_center():
    t = make_hotspot_thermogram(
        height=100, width=100, hotspot_center=(40, 60), hotspot_radius=5,
        hotspot_peak_celsius=90.0,
    )
    hottest = locate_extreme(t.celsius, mode="max")
    assert hottest.row == 40
    assert hottest.col == 60
    assert hottest.value_celsius == pytest.approx(90.0, abs=0.1)


def test_save_annotated_image_creates_new_file_without_touching_original(tmp_path):
    t = make_hotspot_thermogram(
        height=80, width=100, base_celsius=25.0, hotspot_peak_celsius=90.0,
    )
    hotspot = locate_extreme(t.celsius, mode="max")
    reference = locate_extreme(t.celsius, mode="min")

    hotspot_stats = compute_stats(t.celsius)
    reference_stats = compute_stats(t.celsius)
    eta = relative_temperature_rise(hotspot_stats, reference_stats, ambient_celsius=24.0)
    severity = classify_by_relative_rise(eta)

    output_path = tmp_path / "anotada" / "resultado.jpg"
    result_path = save_annotated_image(
        t.celsius,
        hotspot=hotspot,
        reference=reference,
        output_path=output_path,
        delta_t_celsius=hotspot.value_celsius - reference.value_celsius,
        eta_pct=eta,
        severity=severity,
        title="Teste sintético",
    )

    assert result_path == output_path
    assert output_path.exists()

    # O arquivo gerado precisa ser um JPG válido e legível.
    img = Image.open(output_path)
    img.verify()


def test_save_annotated_image_creates_parent_directories(tmp_path):
    t = make_hotspot_thermogram(height=40, width=40)
    hotspot = locate_extreme(t.celsius, mode="max")
    reference = locate_extreme(t.celsius, mode="min")

    output_path = tmp_path / "nivel1" / "nivel2" / "saida.jpg"
    assert not output_path.parent.exists()

    save_annotated_image(
        t.celsius,
        hotspot=hotspot,
        reference=reference,
        output_path=output_path,
        delta_t_celsius=0.0,
    )

    assert output_path.exists()


def test_labels_of_close_points_are_placed_on_opposite_sides():
    from thermocore.visualization.annotate import _separate_labels
    hot_off, ref_off = _separate_labels((100, 300), (110, 340), (12, 12), (12, 12))
    assert hot_off[0] > 0 and ref_off[0] < 0                      # lados opostos
    far_hot, far_ref = _separate_labels((100, 100), (400, 500), (12, 12), (12, 12))
    assert (far_hot, far_ref) == ((12, 12), (12, 12))             # distantes: sem alteração
