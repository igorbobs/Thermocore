import numpy as np
import pytest

from thermocore.validation import (
    dice_coefficient,
    evaluate_segmentation,
    intersection_over_union,
    pixel_accuracy,
    precision_recall,
)


def _masks():
    a = np.zeros((10, 10), dtype=bool); a[2:6, 2:6] = True    # 16 px
    b = np.zeros((10, 10), dtype=bool); b[4:8, 4:8] = True    # 16 px, overlap 2x2=4
    return a, b


def test_identical_masks_give_perfect_scores():
    a = np.zeros((10, 10), dtype=bool); a[2:6, 2:6] = True
    assert intersection_over_union(a, a) == 1.0
    assert dice_coefficient(a, a) == 1.0
    assert pixel_accuracy(a, a) == 1.0
    p, r = precision_recall(a, a)
    assert p == 1.0 and r == 1.0


def test_disjoint_masks_give_zero_overlap_scores():
    a = np.zeros((10, 10), dtype=bool); a[0:3, 0:3] = True
    b = np.zeros((10, 10), dtype=bool); b[7:10, 7:10] = True
    assert intersection_over_union(a, b) == 0.0
    assert dice_coefficient(a, b) == 0.0
    p, r = precision_recall(a, b)
    assert p == 0.0 and r == 0.0


def test_partial_overlap_matches_hand_computed_values():
    a, b = _masks()
    inter, union = 4, 16 + 16 - 4  # 28
    assert intersection_over_union(a, b) == pytest.approx(inter / union)
    assert dice_coefficient(a, b) == pytest.approx(2 * inter / (16 + 16))


def test_both_empty_masks_are_perfect_agreement_by_convention():
    a = np.zeros((5, 5), dtype=bool)
    assert intersection_over_union(a, a) == 1.0
    assert dice_coefficient(a, a) == 1.0


def test_precision_recall_asymmetry_when_predicted_is_superset():
    true = np.zeros((10, 10), dtype=bool); true[0:2, 0:2] = True     # 4 px
    pred = np.zeros((10, 10), dtype=bool); pred[0:4, 0:4] = True     # 16 px, contém os 4
    precision, recall = precision_recall(pred, true)
    assert recall == 1.0          # achou todo o material verdadeiro
    assert precision == pytest.approx(4 / 16)  # mas com muitos falsos positivos


def test_evaluate_segmentation_bundles_all_metrics():
    a, b = _masks()
    m = evaluate_segmentation(a, b)
    assert m.iou == intersection_over_union(a, b)
    assert m.dice == dice_coefficient(a, b)
    assert 0.0 <= m.pixel_accuracy <= 1.0


def test_shape_mismatch_raises_value_error():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError):
        intersection_over_union(a, b)
