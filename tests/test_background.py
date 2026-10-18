import numpy as np
import pytest

from thermocore.analysis.stats import locate_extreme, locate_reference_point
from thermocore.exceptions import EmptyRegionError
from thermocore.io.synthetic import make_flat_thermogram, make_scene_with_background
from thermocore.segmentation.background import (
    estimate_background_mask,
    estimate_material_mask,
    otsu_threshold,
)


def _scene(**kw):
    defaults = dict(height=100, width=100, material_celsius=30.0, background_celsius=5.0,
                    background_edge="top", background_fraction=0.25, hotspot_peak_celsius=90.0)
    defaults.update(kw)
    return make_scene_with_background(**defaults)


def test_otsu_threshold_falls_between_the_two_classes():
    thr = otsu_threshold(_scene().celsius)
    assert 5.0 < thr < 30.0


def test_material_mask_excludes_cold_background_block():
    scene = _scene()
    material = estimate_material_mask(scene.celsius, erosion_iterations=0)
    assert material[:20, :].mean() < 0.05   # faixa de fundo (topo)
    assert material[60:, :].mean() > 0.95   # material


def test_background_mask_is_complement_of_material_mask():
    scene = _scene(background_edge="left")
    assert np.array_equal(estimate_background_mask(scene.celsius), ~estimate_material_mask(scene.celsius))


def test_erosion_shrinks_material_mask():
    scene = _scene()
    assert estimate_material_mask(scene.celsius, erosion_iterations=4).sum() < \
           estimate_material_mask(scene.celsius, erosion_iterations=0).sum()


def test_unrestricted_min_is_background_but_restricted_min_is_material():
    scene = _scene(background_edge="bottom")
    assert locate_extreme(scene.celsius, mode="min").value_celsius == pytest.approx(5.0, abs=0.5)
    material = estimate_material_mask(scene.celsius)
    assert locate_extreme(scene.celsius, mask=material, mode="min").value_celsius == pytest.approx(30.0, abs=1.0)


def test_max_is_unaffected_by_material_restriction():
    scene = _scene()
    material = estimate_material_mask(scene.celsius)
    assert locate_extreme(scene.celsius, mode="max").value_celsius == pytest.approx(
        locate_extreme(scene.celsius, mask=material, mode="max").value_celsius)


def test_uniform_image_gives_empty_material_mask():
    flat = make_flat_thermogram(height=20, width=20)
    assert estimate_material_mask(flat.celsius).sum() == 0


def test_invalid_parameters_are_rejected():
    flat = make_flat_thermogram(height=10, width=10)
    with pytest.raises(ValueError):
        estimate_material_mask(flat.celsius, erosion_iterations=-1)
    with pytest.raises(ValueError):
        estimate_material_mask(flat.celsius, min_region_fraction=1.5)


def test_reference_point_is_robust_percentile_not_raw_minimum():
    celsius = np.full((50, 50), 40.0)
    celsius[10, 10] = 31.0          # um pixel frio isolado (artefato)
    celsius[:, 25:] = 45.0          # metade mais quente
    mask = np.ones_like(celsius, dtype=bool)
    ref = locate_reference_point(celsius, mask, percentile=10.0)
    assert ref.value_celsius == pytest.approx(40.0)   # não 31.0


def test_reference_point_rejects_empty_mask_and_bad_percentile():
    celsius = np.full((10, 10), 30.0)
    with pytest.raises(EmptyRegionError):
        locate_reference_point(celsius, np.zeros_like(celsius, dtype=bool))
    with pytest.raises(ValueError):
        locate_reference_point(celsius, np.ones_like(celsius, dtype=bool), percentile=150)
