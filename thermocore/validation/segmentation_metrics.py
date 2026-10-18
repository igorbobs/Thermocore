"""
Métricas quantitativas de segmentação binária.

Usadas para comparar a máscara de material estimada por
`thermocore.segmentation.background.estimate_material_mask` (Otsu +
limpeza + erosão) contra uma máscara de REFERÊNCIA anotada manualmente
por um humano (ver ferramenta `annotate_mask.html` do projeto e
`examples/validate_segmentation.py`).

Antes deste módulo, a qualidade do Otsu só tinha sido avaliada
visualmente (olhar o contorno desenhado na imagem anotada) — isso não
é validação, é impressão. As funções aqui são puramente aritméticas
sobre arrays booleanos, determinísticas, sem aprendizado de máquina, e dão um número comparável entre imagens e ao longo do
tempo.

Convenção adotada para casos degenerados (ambas as máscaras vazias):
concordância perfeita (1.0) por definição — duas máscaras vazias
"concordam" que não há material ali.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _check_shapes(pred: np.ndarray, true: np.ndarray) -> None:
    if pred.shape != true.shape:
        raise ValueError(f"Shapes diferentes: predita={pred.shape}, referência={true.shape}")


def intersection_over_union(pred: np.ndarray, true: np.ndarray) -> float:
    """IoU (índice de Jaccard) entre a máscara predita e a de referência."""
    _check_shapes(pred, true)
    pred, true = pred.astype(bool), true.astype(bool)
    union = np.logical_or(pred, true).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(pred, true).sum()
    return float(inter) / float(union)


def dice_coefficient(pred: np.ndarray, true: np.ndarray) -> float:
    """Coeficiente de Dice (F1 ao nível de pixel) entre as duas máscaras."""
    _check_shapes(pred, true)
    pred, true = pred.astype(bool), true.astype(bool)
    denom = pred.sum() + true.sum()
    if denom == 0:
        return 1.0
    inter = np.logical_and(pred, true).sum()
    return float(2 * inter) / float(denom)


def pixel_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Fração de pixels em que predita e referência concordam (material OU fundo)."""
    _check_shapes(pred, true)
    return float((pred.astype(bool) == true.astype(bool)).mean())


def precision_recall(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """
    Precisão (dos pixels que o Otsu marcou como material, quantos são
    material de verdade) e revocação (do material de verdade, quantos
    o Otsu encontrou).
    """
    _check_shapes(pred, true)
    pred, true = pred.astype(bool), true.astype(bool)
    tp = int(np.logical_and(pred, true).sum())
    fp = int(np.logical_and(pred, ~true).sum())
    fn = int(np.logical_and(~pred, true).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    return precision, recall


@dataclass(frozen=True)
class SegmentationMetrics:
    iou: float
    dice: float
    pixel_accuracy: float
    precision: float
    recall: float

    def as_dict(self) -> dict:
        return {
            "iou": self.iou, "dice": self.dice, "pixel_accuracy": self.pixel_accuracy,
            "precision": self.precision, "recall": self.recall,
        }


def evaluate_segmentation(pred: np.ndarray, true: np.ndarray) -> SegmentationMetrics:
    """Calcula todas as métricas de uma vez, para um par (predita, referência)."""
    precision, recall = precision_recall(pred, true)
    return SegmentationMetrics(
        iou=intersection_over_union(pred, true),
        dice=dice_coefficient(pred, true),
        pixel_accuracy=pixel_accuracy(pred, true),
        precision=precision,
        recall=recall,
    )


def load_mask_png(path, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Lê uma máscara PNG exportada pela ferramenta de anotação (branco =
    material, preto = fundo), redimensiona para `target_shape` (altura,
    largura) se necessário, e retorna um array booleano.
    """
    from PIL import Image

    im = Image.open(path).convert("L")
    if (im.height, im.width) != tuple(target_shape):
        im = im.resize((target_shape[1], target_shape[0]))
    return np.array(im) > 127
