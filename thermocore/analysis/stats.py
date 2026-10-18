"""
Indicadores térmicos estatísticos.

Todas as funções aqui operam sobre uma matriz de temperatura em Celsius
(ou um subconjunto dela, selecionado por uma máscara de região de
interesse) e retornam apenas números — nenhuma inferência, nenhuma
classificação. A classificação de anomalia fica em
`thermocore.analysis.classification`, propositalmente separada, para
que a etapa puramente descritiva (o que foi medido) nunca se misture
com a etapa interpretativa (o que isso significa).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from thermocore.exceptions import EmptyRegionError


@dataclass(frozen=True)
class ThermalStats:
    """Indicadores descritivos de uma região térmica, em Celsius."""

    t_max: float
    t_min: float
    t_mean: float
    t_median: float
    t_std: float
    p10: float
    p25: float
    p75: float
    p90: float
    pixel_count: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExtremePoint:
    """
    Posição e valor de um pixel extremo (mais quente ou mais frio) em
    uma matriz de temperatura. `row`/`col` são índices de matriz
    (linha, coluna), na mesma convenção usada pelo NumPy — não são
    coordenadas x/y de imagem invertidas.
    """

    row: int
    col: int
    value_celsius: float


def locate_extreme(
    celsius: np.ndarray, mask: np.ndarray | None = None, mode: str = "max"
) -> ExtremePoint:
    """
    Localiza a posição do pixel de temperatura mais alta (`mode="max"`)
    ou mais baixa (`mode="min"`) em `celsius`, opcionalmente restrito a
    `mask`.

    IMPORTANTE — limitação conhecida (documentada no README e no
    artigo): por padrão, "mais frio" aqui significa o pixel mais frio
    de toda a matriz (ou da máscara fornecida), não necessariamente uma
    região de referência escolhida por um termografista em uma fase
    saudável do equipamento. Em cenas ao ar livre isso costuma
    corresponder ao céu de fundo, não ao próprio equipamento — por
    isso, para um ΔT/η fisicamente significativo, o chamador deve
    fornecer uma `mask` que restrinja a busca a uma região relevante
    do equipamento, em vez de usar o quadro inteiro sem restrição.

    Levanta EmptyRegionError se a máscara não contiver nenhum pixel.
    """
    if mode not in ("max", "min"):
        raise ValueError(f"mode deve ser 'max' ou 'min', recebido: {mode!r}")

    if mask is not None:
        if mask.shape != celsius.shape:
            raise ValueError(
                f"Shape da máscara {mask.shape} não bate com o da "
                f"imagem térmica {celsius.shape}."
            )
        bool_mask = mask.astype(bool)
        if not np.any(bool_mask):
            raise EmptyRegionError(
                "A região de interesse selecionada não contém nenhum pixel."
            )
        working = np.where(bool_mask, celsius, np.nan)
    else:
        working = celsius

    flat_index = (
        int(np.nanargmax(working)) if mode == "max" else int(np.nanargmin(working))
    )
    row, col = np.unravel_index(flat_index, celsius.shape)
    return ExtremePoint(row=int(row), col=int(col), value_celsius=float(celsius[row, col]))


def locate_reference_point(
    celsius: np.ndarray,
    mask: np.ndarray,
    percentile: float = 10.0,
) -> ExtremePoint:
    """
    Ponto de REFERÊNCIA (comparação) dentro do material `mask`: o pixel
    cuja temperatura é a mais próxima do percentil `percentile` da
    temperatura do material (padrão: P10, "parte fria típica do
    equipamento").

    Por que não simplesmente o pixel mínimo do material? Porque, quando
    `mask` vem de um limiar (ver `estimate_material_mask`), o mínimo
    dentro da máscara é essencialmente o próprio limiar de corte — um
    artefato da segmentação, não uma temperatura física. Um percentil
    baixo é robusto a isso e a pixels ruidosos isolados. (Verificado
    empiricamente na imagem real de teste: mínimo do material = 30,3 °C
    = limiar de Otsu.) Uma referência ideal continua sendo uma ROI
    escolhida por um termografista em fase/componente saudável.

    Levanta EmptyRegionError se a máscara não contiver nenhum pixel.
    """
    if not (0.0 <= percentile <= 100.0):
        raise ValueError("percentile deve estar entre 0 e 100.")
    if mask.shape != celsius.shape:
        raise ValueError(
            f"Shape da máscara {mask.shape} não bate com o da imagem térmica {celsius.shape}."
        )
    bool_mask = mask.astype(bool)
    if not np.any(bool_mask):
        raise EmptyRegionError("A região de interesse selecionada não contém nenhum pixel.")

    target = float(np.percentile(celsius[bool_mask], percentile))
    distance = np.where(bool_mask, np.abs(celsius - target), np.inf)
    row, col = np.unravel_index(int(np.argmin(distance)), celsius.shape)
    return ExtremePoint(row=int(row), col=int(col), value_celsius=float(celsius[row, col]))


def compute_stats(celsius: np.ndarray, mask: np.ndarray | None = None) -> ThermalStats:
    """
    Calcula os indicadores térmicos descritivos de `celsius`,
    opcionalmente restritos aos pixels onde `mask` é verdadeiro
    (booleana, mesmo shape de `celsius`).

    Levanta EmptyRegionError se a região selecionada não contiver
    nenhum pixel — nunca retorna estatísticas vazias/zeradas
    silenciosamente.
    """
    if mask is not None:
        if mask.shape != celsius.shape:
            raise ValueError(
                f"Shape da máscara {mask.shape} não bate com o da "
                f"imagem térmica {celsius.shape}."
            )
        values = celsius[mask.astype(bool)]
    else:
        values = celsius.ravel()

    if values.size == 0:
        raise EmptyRegionError(
            "A região de interesse selecionada não contém nenhum pixel."
        )

    return ThermalStats(
        t_max=float(np.max(values)),
        t_min=float(np.min(values)),
        t_mean=float(np.mean(values)),
        t_median=float(np.median(values)),
        t_std=float(np.std(values)),
        p10=float(np.percentile(values, 10)),
        p25=float(np.percentile(values, 25)),
        p75=float(np.percentile(values, 75)),
        p90=float(np.percentile(values, 90)),
        pixel_count=int(values.size),
    )


def delta_t(hotspot_stats: ThermalStats, reference_stats: ThermalStats) -> float:
    """
    Diferença absoluta de temperatura (ΔT) entre uma ROI quente e uma
    ROI de referência homóloga, ambas escolhidas pelo usuário, usando a
    temperatura máxima de cada região (critério clássico de ΔT absoluto).

    NÃO é usada por `analyze_thermogram`, que trabalha com pontos
    (ponto quente e referência por percentil do material); serve para
    quem já dispõe de ROIs definidas por um termografista.
    """
    return hotspot_stats.t_max - reference_stats.t_max


def relative_temperature_rise(
    hotspot_stats: ThermalStats,
    reference_stats: ThermalStats,
    ambient_celsius: float,
) -> float:
    """
    Diferença relativa de temperatura (η) entre duas ROIs escolhidas pelo
    usuário, conforme a norma DL/T 664-2016 citada por Xia et al. (2021):

        η = (T1 - T2) / (T1 - T0) * 100%

    onde T1 é a temperatura do ponto quente, T2 é a temperatura do
    ponto de referência e T0 é a temperatura ambiente. Este é o
    critério de classificação de severidade adotado neste projeto,
    por ser menos sensível a variações de carga e clima do que um
    limiar de temperatura absoluto.

    Atenção: aqui T1 e T2 são as temperaturas MÁXIMAS de cada ROI
    (`ThermalStats.t_max`), enquanto `analyze_thermogram` usa o pixel
    mais quente do material (T1) e o pixel do material mais próximo de
    um percentil baixo (T2). As duas funções só coincidem quando as
    ROIs são definidas com esse mesmo critério.

    Levanta ValueError se (T1 - T0) for zero, para evitar divisão por
    zero silenciosa — nunca retorna um valor arbitrário nesse caso.
    """
    t1 = hotspot_stats.t_max
    t2 = reference_stats.t_max
    denominator = t1 - ambient_celsius
    if denominator == 0:
        raise ValueError(
            "Temperatura do ponto quente é igual à temperatura ambiente; "
            "η não pode ser calculado (divisão por zero)."
        )
    return (t1 - t2) / denominator * 100.0
