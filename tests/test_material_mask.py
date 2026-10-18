"""Comportamento da máscara de material na versão 0.2.0."""

import numpy as np
import pytest
from scipy import ndimage as ndi

from thermocore.io.synthetic import make_flat_thermogram
from thermocore.segmentation.background import (
    estimate_material_mask,
    otsu_separability,
    otsu_threshold,
)


def _reference_loop_mask(celsius, erosion_iterations=4, min_region_fraction=0.005, border_value=1):
    """Implementação por laço (a da v0.1.0), usada como oráculo."""
    warm = celsius > otsu_threshold(celsius)
    labeled, n = ndi.label(warm)
    cleaned = np.zeros_like(warm)
    for i in range(1, n + 1):
        region = labeled == i
        if region.sum() >= min_region_fraction * celsius.size:
            cleaned |= region
    if erosion_iterations > 0:
        cleaned = ndi.binary_erosion(cleaned, iterations=erosion_iterations, border_value=border_value)
    return cleaned


def test_vectorized_filter_matches_loop_on_noisy_image():
    rng = np.random.default_rng(3)
    c = rng.normal(30, 3, (120, 160))
    assert np.array_equal(estimate_material_mask(c), _reference_loop_mask(c))


def test_vectorized_filter_matches_loop_on_scene_with_blobs():
    rng = np.random.default_rng(4)
    c = np.full((100, 140), 10.0) + rng.normal(0, 0.2, (100, 140))
    c[20:60, 30:90] = 40.0
    c[70:74, 100:104] = 45.0            # componente pequeno (deve ser descartado)
    mask = estimate_material_mask(c)
    assert np.array_equal(mask, _reference_loop_mask(c))
    assert not mask[70:74, 100:104].any()


def test_equipment_filling_the_frame_keeps_its_border_by_default():
    c = np.full((60, 60), 40.0)
    c[:, :10] = 5.0                      # faixa fria à esquerda; resto é equipamento até a borda do quadro
    keep = estimate_material_mask(c, erosion_iterations=4)
    legacy = estimate_material_mask(c, erosion_iterations=4, preserve_image_border=False)
    assert keep[30, 59] and keep[0, 40] and keep[59, 40]            # borda do quadro preservada
    assert not legacy[30, 59] and not legacy[0, 40]                 # v0.1.0 apagava a faixa de 4 px
    assert keep.sum() > legacy.sum()
    assert not keep[30, 12]                                         # mas a borda com o fundo frio é erodida


def test_separability_is_high_for_bimodal_and_lower_for_unimodal():
    bimodal = np.concatenate([np.full(5000, 10.0), np.full(5000, 40.0)]).reshape(100, 100)
    bimodal = bimodal + np.random.default_rng(0).normal(0, 0.1, bimodal.shape)
    unimodal = np.random.default_rng(1).normal(30, 1, (100, 100))
    assert otsu_separability(bimodal) > 0.95
    assert 0.55 < otsu_separability(unimodal) < 0.72
    assert 0.0 <= otsu_separability(make_flat_thermogram(height=20, width=20).celsius) <= 1.0


def test_noisy_image_is_fast_enough():
    # a versão 0.1.0 levava ~3 s nesta imagem (≈20 mil componentes)
    import time
    c = np.random.default_rng(1).normal(30, 3, (480, 640))
    t = time.perf_counter()
    estimate_material_mask(c)
    assert time.perf_counter() - t < 1.0
