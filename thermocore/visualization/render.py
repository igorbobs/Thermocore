"""
Renderização de mapas de calor a partir da matriz de temperatura.

Este módulo é deliberadamente simples: converte a matriz de
temperatura em uma imagem RGB usando uma paleta de cores padrão do
matplotlib, sem nenhuma etapa de aprendizado de máquina ou decisão
automática — cada pixel de saída é uma função direta e determinística
do valor de temperatura correspondente, então o resultado é
totalmente explicável e reproduzível.
"""

from __future__ import annotations

import numpy as np


def render_thermal_image(celsius: np.ndarray, colormap: str = "inferno") -> np.ndarray:
    """
    Converte uma matriz de temperatura em Celsius para uma imagem RGB
    (array uint8, shape (altura, largura, 3)), normalizando linearmente
    entre o mínimo e o máximo da própria matriz e aplicando a paleta de
    cores `colormap` (qualquer nome válido de colormap do matplotlib;
    "inferno" e "plasma" se aproximam visualmente das paletas mais
    comuns em câmeras térmicas FLIR).

    Esta normalização é por imagem (min/max da própria matriz), não uma
    escala fixa entre imagens — ou seja, a mesma cor em duas imagens
    diferentes não necessariamente representa a mesma temperatura
    absoluta. Isso é adequado para inspeção visual de uma imagem por
    vez (o uso deste projeto), mas não deve ser usado para comparar
    cores entre imagens diferentes.
    """
    import matplotlib

    t_min = float(np.min(celsius))
    t_max = float(np.max(celsius))
    if t_max == t_min:
        normalized = np.full_like(celsius, 0.5, dtype=np.float64)
    else:
        normalized = (celsius - t_min) / (t_max - t_min)

    cmap = matplotlib.colormaps[colormap]
    rgba = cmap(normalized)  # shape (H, W, 4), valores em [0, 1]
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return rgb
