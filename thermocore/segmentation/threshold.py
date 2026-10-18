"""
Segmentação de regiões de interesse (ROI) por limiarização.

Método inicial do projeto (começar simples antes de ML/DL).
Cada função aqui retorna uma máscara booleana explicável: dado o
limiar usado, qualquer pessoa pode conferir manualmente por que um
pixel foi ou não incluído na região — não há parâmetro aprendido.
Segmentação por aprendizado profundo (por exemplo, com Faster R-CNN ou
YOLO, discutidos na revisão de literatura do projeto) fica reservada
para uma fase posterior, condicionada à existência de um conjunto de
imagens anotadas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


def segment_above_threshold(celsius: np.ndarray, threshold_celsius: float) -> np.ndarray:
    """
    Máscara booleana dos pixels com temperatura estritamente maior que
    `threshold_celsius`. A forma mais simples possível de segmentação
    — serve de base de comparação para métodos mais elaborados.
    """
    return celsius > threshold_celsius


def segment_above_percentile(celsius: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """
    Máscara booleana dos pixels acima do percentil `percentile` da
    própria imagem. Mais robusto que um limiar absoluto fixo quando a
    temperatura de fundo varia entre imagens (ex.: fotos tiradas em
    dias diferentes, com temperatura ambiente diferente).
    """
    threshold = np.percentile(celsius, percentile)
    return celsius > threshold


@dataclass(frozen=True)
class ConnectedRegion:
    """Uma região conectada resultante da rotulagem de uma máscara."""

    label_id: int
    mask: np.ndarray
    pixel_count: int
    centroid: tuple[float, float]


def label_connected_regions(
    mask: np.ndarray, min_pixel_count: int = 4
) -> list[ConnectedRegion]:
    """
    Rotula componentes conexos da máscara booleana `mask` (por
    exemplo, a saída de `segment_above_threshold`) e descarta regiões
    com menos de `min_pixel_count` pixels, tratadas como ruído pontual
    em vez de um ponto quente real.

    Retorna uma lista vazia se nenhuma região sobreviver ao filtro de
    tamanho mínimo — o chamador decide o que fazer com uma lista
    vazia (por exemplo, reportar "nenhuma anomalia térmica detectada
    acima do limiar usado"), em vez deste módulo decidir por ele.
    """
    labeled, num_labels = ndi.label(mask)
    regions: list[ConnectedRegion] = []
    for label_id in range(1, num_labels + 1):
        region_mask = labeled == label_id
        pixel_count = int(np.sum(region_mask))
        if pixel_count < min_pixel_count:
            continue
        centroid = ndi.center_of_mass(region_mask)
        regions.append(
            ConnectedRegion(
                label_id=label_id,
                mask=region_mask,
                pixel_count=pixel_count,
                centroid=(float(centroid[0]), float(centroid[1])),
            )
        )
    return regions
