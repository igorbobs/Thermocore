"""
Estimativa da região de "material" (o equipamento fotografado) versus
"fundo" da cena (tipicamente céu, em inspeções ao ar livre).

Motivação: sem restrição, o "ponto mais frio" de uma foto ao ar livre
cai no céu, não no equipamento — inútil como referência de ΔT/η.

MÉTODO ADOTADO (determinístico, sem aprendizado de máquina):
limiar de Otsu sobre o histograma de temperatura da própria imagem,
seguido de (i) remoção de componentes conexos muito pequenos e (ii)
erosão morfológica (elemento estruturante em cruz, conectividade-4),
que descarta pixels de borda (onde o pixel mistura equipamento e fundo
— efeito de resolução espacial finita, ver limitações da IRT em Xia et
al., 2021). A borda do QUADRO não é tratada como fundo: ela é só o
limite do enquadramento, não uma transição entre superfícies. O limiar de Otsu maximiza a
variância entre as duas classes (frio/quente) e já é usado na
literatura de segmentação de imagens infravermelhas de equipamentos
elétricos (ver Tabela 5 de Xia et al., 2021).

MÉTODO AVALIADO E DESCARTADO (registrado por honestidade
metodológica): uma primeira versão marcava como fundo apenas
as regiões frias CONEXAS e GRANDES (percentil baixo + área mínima).
Na imagem real de teste (FLIR T640) isso falhou: o céu aparece
fragmentado por fios e estruturas, nenhuma região atingia a área
mínima, apenas ~14% dos pixels foram marcados como fundo e o "ponto
mais frio" continuou caindo no céu (8,8 °C).

LIMITAÇÃO CONHECIDA: Otsu assume uma separação razoavelmente bimodal
entre fundo e equipamento. Falha em cenas sem fundo frio distinto (ex.:
ambiente fechado, outro equipamento quente atrás do alvo) — nesses
casos o resultado deve ser revisado visualmente, e não substitui uma
ROI escolhida por um termografista.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def otsu_threshold(celsius: np.ndarray, bins: int = 256) -> float:
    """Limiar de Otsu (°C) sobre o histograma da matriz de temperatura."""
    hist, edges = np.histogram(celsius.ravel(), bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight0 = np.cumsum(hist)
    weight1 = weight0[-1] - weight0
    sum_all = np.sum(hist * centers)
    cum_sum = np.cumsum(hist * centers)
    mean0 = cum_sum / np.maximum(weight0, 1)
    mean1 = (sum_all - cum_sum) / np.maximum(weight1, 1)
    between_class_variance = weight0 * weight1 * (mean0 - mean1) ** 2
    return float(centers[int(np.argmax(between_class_variance))])


def otsu_separability(celsius: np.ndarray, bins: int = 256) -> float:
    """
    Razão entre a variância entre classes máxima (a do limiar de Otsu) e
    a variância total do histograma, em [0, 1]. É uma medida de quão
    bimodal é a cena: valores próximos de 1 indicam duas populações
    bem separadas; uma distribuição gaussiana (unimodal) dá cerca de
    0,64. É informativa — o Otsu SEMPRE devolve um limiar, mesmo em
    cenas sem fundo distinto — e fica registrada no resultado da análise.
    """
    hist, edges = np.histogram(celsius.ravel(), bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    weight0 = np.cumsum(hist) / total
    weight1 = 1.0 - weight0
    mean_all = np.sum(hist * centers) / total
    cum_mean = np.cumsum(hist * centers) / total
    mean0 = cum_mean / np.maximum(weight0, 1e-12)
    mean1 = (mean_all - cum_mean) / np.maximum(weight1, 1e-12)
    between = weight0 * weight1 * (mean0 - mean1) ** 2
    variance = np.sum(hist * (centers - mean_all) ** 2) / total
    if variance <= 0:
        return 0.0
    return float(np.max(between) / variance)


def estimate_material_mask(
    celsius: np.ndarray,
    erosion_iterations: int = 4,
    min_region_fraction: float = 0.005,
    preserve_image_border: bool = True,
) -> np.ndarray:
    """
    Máscara booleana (True = provável equipamento/material).

    Parâmetros
    ----------
    erosion_iterations: número de erosões morfológicas para afastar a
        máscara das bordas com o fundo (pixels de borda são mistos).
    min_region_fraction: componentes conexos menores que esta fração da
        imagem são descartados (ruído/pontos isolados).
    preserve_image_border: se True (padrão), a borda do quadro NÃO é
        tratada como fundo na erosão — um equipamento que preenche o
        enquadramento não perde uma faixa de `erosion_iterations`
        pixels junto à borda da imagem. Com False reproduz-se o
        comportamento da versão 0.1.0 (borda do quadro = fundo).

    Retorna uma máscara totalmente False se nada sobreviver (caso
    degenerado, ex.: imagem uniforme) — o chamador decide o que fazer
    (`locate_extreme` levanta `EmptyRegionError`).
    """
    if erosion_iterations < 0:
        raise ValueError("erosion_iterations deve ser >= 0.")
    if not (0.0 <= min_region_fraction < 1.0):
        raise ValueError("min_region_fraction deve estar em [0, 1).")

    if float(np.max(celsius)) == float(np.min(celsius)):
        return np.zeros(celsius.shape, dtype=bool)

    warm = celsius > otsu_threshold(celsius)

    # Filtro de área vetorizado: custo O(N), independente do nº de
    # componentes (a versão 0.1.0 percorria cada componente com uma
    # comparação sobre a imagem inteira, O(N x componentes)).
    labeled, num_labels = ndi.label(warm)
    sizes = np.bincount(labeled.ravel(), minlength=num_labels + 1)
    keep = sizes >= min_region_fraction * celsius.size
    keep[0] = False  # rótulo 0 = fundo do rotulamento
    cleaned = keep[labeled]

    if erosion_iterations > 0:
        cleaned = ndi.binary_erosion(
            cleaned,
            iterations=erosion_iterations,
            border_value=1 if preserve_image_border else 0,
        )
    return cleaned


def estimate_background_mask(celsius: np.ndarray, **kwargs) -> np.ndarray:
    """Complemento de `estimate_material_mask` (True = provável fundo)."""
    return ~estimate_material_mask(celsius, **kwargs)
