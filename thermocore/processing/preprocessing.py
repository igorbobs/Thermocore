"""
Pré-processamento de matrizes térmicas.

Etapa "simples" do pipeline: apenas filtragem de ruído e
normalização, sem qualquer segmentação ou decisão sobre o que é ou não
uma anomalia. Trabalha sempre sobre uma CÓPIA da matriz de entrada —
nunca modifica o array recebido in-place (extensão, dentro do próprio
código, da preservação dos dados originais).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, gaussian_filter


def denoise_median(celsius: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Filtro de mediana — bom para remover ruído "sal e pimenta" pontual
    sem borrar excessivamente as bordas de regiões quentes, conforme
    discutido na literatura de pré-processamento de imagens térmicas
    (ver Seção 4.1 de Xia et al., 2020). Retorna uma nova matriz;
    `celsius` não é alterado.
    """
    return median_filter(celsius, size=size)


def denoise_gaussian(celsius: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Filtro gaussiano — suaviza ruído de forma mais uniforme que a
    mediana, ao custo de borrar um pouco mais as bordas. Retorna uma
    nova matriz; `celsius` não é alterado.
    """
    return gaussian_filter(celsius, sigma=sigma)


def normalize_0_1(celsius: np.ndarray) -> np.ndarray:
    """
    Normaliza a matriz de temperatura para o intervalo [0, 1], útil
    apenas para fins de VISUALIZAÇÃO (mapas de calor, por exemplo) —
    nunca usar o resultado normalizado para cálculo de estatísticas
    térmicas reais, pois a informação de temperatura absoluta é
    perdida nessa transformação.
    """
    t_min = float(np.min(celsius))
    t_max = float(np.max(celsius))
    if t_max == t_min:
        # Imagem uniforme: evita divisão por zero. Retorna tudo em 0.5
        # (meio da escala), já que não há variação a normalizar.
        return np.full_like(celsius, 0.5, dtype=np.float64)
    return (celsius - t_min) / (t_max - t_min)
