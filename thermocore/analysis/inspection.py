"""
Análise de inspeção de uma imagem térmica, de ponta a ponta, sobre o
MATERIAL (equipamento) e não sobre a cena inteira.

Fluxo (todo determinístico e rastreável):
  1. (opcional) filtro de mediana sobre uma CÓPIA da matriz;
  2. estima a máscara de material (Otsu + limpeza + erosão);
  3. ponto mais QUENTE = máximo de temperatura dentro do material;
  4. ponto de REFERÊNCIA = pixel do material mais próximo de um
     percentil baixo (padrão P10) da temperatura do material — robusto
     ao artefato "mínimo = limiar de corte" (ver `locate_reference_point`);
  5. ΔT = T_quente − T_referência;
  6. η = (T1 − T2)/(T1 − T0) × 100 (DL/T 664-2016), com T0 = temperatura
     atmosférica LIDA DO PRÓPRIO ARQUIVO quando disponível; se o
     arquivo não a traz e o chamador não informa `ambient_celsius`,
     levanta `AmbientTemperatureUnavailableError` — nunca assume um valor;
  7. severidade por limiares normativos, com três salvaguardas:
       * ΔT mínimo: se ΔT < `min_delta_t_celsius`, a classe é suprimida
         (normal), pois a diferença está dentro da incerteza de medição;
       * faixa de η: η e a classe são recalculados para vários
         percentis de referência (P1–P50) e a instabilidade de classe é
         sinalizada;
       * avisos: referência abaixo de T0, η > 100%, pixels mais quentes
         fora do material (possível reflexo/ruído ou equipamento não
         segmentado).

A saída deve ser lida como TRIAGEM PRELIMINAR, não como laudo, e a
"severidade" é um critério TÉRMICO: a ABNT NBR 15866 recomenda que a
prioridade de ação considere também segurança, perda de produção,
criticidade e tendência do equipamento (matriz de risco da organização), o que esta ferramenta não faz.

LIMITAÇÕES (repetidas no README/artigo): um ponto quente e uma
referência por imagem; sem distinção de reflexo especular; a
referência automática NÃO substitui uma ROI escolhida por termografista
(η é sensível a essa escolha — ver `eta_by_percentile`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from thermocore.analysis.classification import (
    AnomalySeverity,
    SeverityThresholds,
    classify_by_relative_rise,
)
from thermocore.analysis.stats import ExtremePoint, locate_extreme, locate_reference_point
from thermocore.exceptions import AmbientTemperatureUnavailableError, HotspotBelowAmbientError
from thermocore.processing.preprocessing import denoise_median
from thermocore.segmentation.background import (
    estimate_material_mask,
    otsu_separability,
    otsu_threshold,
)

#: Percentis do material usados na análise de sensibilidade da referência.
SENSITIVITY_PERCENTILES: tuple[float, ...] = (1.0, 5.0, 10.0, 25.0, 50.0)

_SEVERITY_ORDER = (
    AnomalySeverity.NORMAL,
    AnomalySeverity.GENERAL,
    AnomalySeverity.MAJOR,
    AnomalySeverity.EMERGENCY,
)


@dataclass(frozen=True)
class EtaPoint:
    """η e severidade para uma escolha de percentil de referência."""

    percentile: float
    reference_celsius: float
    delta_t_celsius: float
    eta_pct: float
    severity: AnomalySeverity


@dataclass(frozen=True)
class InspectionResult:
    hotspot: ExtremePoint
    reference: ExtremePoint
    delta_t_celsius: float
    eta_pct: float
    severity: AnomalySeverity  # classe final (após a salvaguarda de ΔT mínimo)
    ambient_celsius: float
    ambient_source: str  # "arquivo" ou "informado pelo usuário"
    reference_percentile: float
    material_pixel_fraction: float
    material_mask: np.ndarray
    # --- acrescentados na versão 0.2.0 ---------------------------------
    severity_by_eta: AnomalySeverity = AnomalySeverity.NORMAL  # classe só por η
    otsu_threshold_celsius: float = float("nan")
    otsu_separability: float = float("nan")
    global_max_celsius: float = float("nan")  # máximo da imagem inteira
    eta_by_percentile: tuple[EtaPoint, ...] = ()
    severity_range: tuple[AnomalySeverity, AnomalySeverity] = (
        AnomalySeverity.NORMAL,
        AnomalySeverity.NORMAL,
    )
    severity_is_stable: bool = True
    warnings: tuple[str, ...] = field(default_factory=tuple)
    parameters: dict = field(default_factory=dict)
    mta_exceeded: bool | None = None  # None = MTA não informada

    @property
    def eta_range_pct(self) -> tuple[float, float]:
        """(mínimo, máximo) de η entre os percentis de referência testados."""
        values = [p.eta_pct for p in self.eta_by_percentile]
        return (min(values), max(values)) if values else (self.eta_pct, self.eta_pct)


def _eta(hot: float, ref: float, t0: float) -> float:
    return (hot - ref) / (hot - t0) * 100.0


def analyze_thermogram(
    celsius: np.ndarray,
    ambient_celsius: float | None = None,
    file_ambient_celsius: float | None = None,
    reference_percentile: float = 10.0,
    thresholds: SeverityThresholds = SeverityThresholds(),
    min_delta_t_celsius: float = 2.0,
    erosion_iterations: int = 4,
    min_region_fraction: float = 0.005,
    denoise_median_size: int | None = None,
    outside_hot_warning_celsius: float = 2.0,
    max_admissible_celsius: float | None = None,
) -> InspectionResult:
    """
    Analisa `celsius` conforme o fluxo do docstring do módulo.

    `ambient_celsius`: T0 informado explicitamente (tem prioridade).
    `file_ambient_celsius`: T0 lido dos metadados do arquivo
        (`thermogram.measurement.atmospheric_temperature_c`). Atenção:
        é o valor CONFIGURADO na câmera no momento da captura, não uma
        medição independente do ambiente.
    `min_delta_t_celsius`: ΔT mínimo para que a classe de severidade seja
        mantida. É um parâmetro de engenharia (não normativo), da ordem
        da incerteza de medição de câmeras radiométricas; configure-o
        conforme o procedimento da sua organização.
    `denoise_median_size`: se informado (ímpar), aplica filtro de
        mediana a uma cópia da matriz antes da análise (padrão: sem
        filtro, para que os extremos sejam os valores radiométricos
        originais).
    `outside_hot_warning_celsius`: gera aviso se o máximo da imagem
        exceder o ponto quente do material em mais que este valor.

    `max_admissible_celsius`: máxima temperatura admissível (MTA, NBR 15866)
        do componente, se conhecida (fabricante, norma do componente ou
        procedimento interno). Se o ponto quente a exceder, é emitido um
        aviso e `mta_exceeded` fica True; a classe NÃO é alterada.

    Se T0 faltar, levanta `AmbientTemperatureUnavailableError`; se o
    ponto quente não exceder T0, levanta `HotspotBelowAmbientError`.
    """
    if ambient_celsius is not None:
        t0, source = float(ambient_celsius), "informado pelo usuário"
    elif file_ambient_celsius is not None:
        t0, source = float(file_ambient_celsius), "arquivo"
    else:
        raise AmbientTemperatureUnavailableError(
            "Temperatura ambiente (T0) indisponível: o arquivo não a traz e "
            "nenhuma foi informada. Não será assumido um valor."
        )

    data = denoise_median(celsius, size=denoise_median_size) if denoise_median_size else celsius

    threshold = otsu_threshold(data)
    separability = otsu_separability(data)
    material = estimate_material_mask(
        data, erosion_iterations=erosion_iterations, min_region_fraction=min_region_fraction
    )
    hotspot = locate_extreme(data, mask=material, mode="max")
    reference = locate_reference_point(data, material, percentile=reference_percentile)

    delta_t = hotspot.value_celsius - reference.value_celsius
    if hotspot.value_celsius - t0 <= 0:
        raise HotspotBelowAmbientError(
            f"Ponto quente ({hotspot.value_celsius:.1f} °C) não excede a temperatura "
            f"ambiente ({t0:.1f} °C); η não é definido nesse caso."
        )
    eta = _eta(hotspot.value_celsius, reference.value_celsius, t0)
    severity_by_eta = classify_by_relative_rise(eta, thresholds)

    # --- sensibilidade da referência (P1–P50) ------------------------------
    sens: list[EtaPoint] = []
    for p in sorted({*SENSITIVITY_PERCENTILES, float(reference_percentile)}):
        ref_p = locate_reference_point(data, material, percentile=p)
        dt_p = hotspot.value_celsius - ref_p.value_celsius
        eta_p = _eta(hotspot.value_celsius, ref_p.value_celsius, t0)
        sev_p = (
            AnomalySeverity.NORMAL
            if dt_p < min_delta_t_celsius
            else classify_by_relative_rise(eta_p, thresholds)
        )
        sens.append(EtaPoint(p, ref_p.value_celsius, dt_p, eta_p, sev_p))
    classes = [s.severity for s in sens]
    sev_lo = min(classes, key=_SEVERITY_ORDER.index)
    sev_hi = max(classes, key=_SEVERITY_ORDER.index)
    is_stable = sev_lo == sev_hi

    # --- salvaguardas e avisos -------------------------------------------------
    warnings: list[str] = []
    severity = severity_by_eta
    if delta_t < min_delta_t_celsius:
        severity = AnomalySeverity.NORMAL
        warnings.append(
            f"ΔT = {delta_t:.1f} °C é menor que o mínimo configurado "
            f"({min_delta_t_celsius:.1f} °C): classificação suprimida (normal), "
            f"embora η = {eta:.1f}% corresponderia a '{severity_by_eta.value}'."
        )
    if reference.value_celsius < t0:
        warnings.append(
            f"A referência ({reference.value_celsius:.1f} °C) está abaixo da temperatura "
            f"ambiente T0 ({t0:.1f} °C): η pode ultrapassar 100% e a classe é pouco confiável."
        )
    if eta > 100.0:
        warnings.append(f"η = {eta:.1f}% excede 100%: referência inadequada.")
    global_max = float(np.max(data))
    if global_max - hotspot.value_celsius > outside_hot_warning_celsius:
        warnings.append(
            f"Há pixels mais quentes fora da região de material (máximo da imagem = "
            f"{global_max:.1f} °C, ponto quente do material = {hotspot.value_celsius:.1f} °C): "
            "pode ser reflexo/ruído pontual ou parte do equipamento fora da máscara. "
            "Verifique a imagem visível."
        )
    mta_exceeded = None
    if max_admissible_celsius is not None:
        mta_exceeded = hotspot.value_celsius > max_admissible_celsius
        if mta_exceeded:
            warnings.append(
                f"O ponto quente ({hotspot.value_celsius:.1f} °C) excede a máxima temperatura "
                f"admissível informada ({max_admissible_celsius:.1f} °C)."
            )
    if not is_stable:
        warnings.append(
            f"A classe depende da referência escolhida: varia de '{sev_lo.value}' a "
            f"'{sev_hi.value}' entre os percentis P{sens[0].percentile:.0f}–"
            f"P{sens[-1].percentile:.0f} do material (η de "
            f"{min(s.eta_pct for s in sens):.0f}% a {max(s.eta_pct for s in sens):.0f}%). "
            "Confirme a referência com um termografista."
        )

    return InspectionResult(
        hotspot=hotspot,
        reference=reference,
        delta_t_celsius=delta_t,
        eta_pct=eta,
        severity=severity,
        ambient_celsius=t0,
        ambient_source=source,
        reference_percentile=reference_percentile,
        material_pixel_fraction=float(material.mean()),
        material_mask=material,
        severity_by_eta=severity_by_eta,
        otsu_threshold_celsius=threshold,
        otsu_separability=separability,
        global_max_celsius=global_max,
        eta_by_percentile=tuple(sens),
        severity_range=(sev_lo, sev_hi),
        severity_is_stable=is_stable,
        warnings=tuple(warnings),
        mta_exceeded=mta_exceeded,
        parameters={
            "min_delta_t_celsius": min_delta_t_celsius,
            "erosion_iterations": erosion_iterations,
            "min_region_fraction": min_region_fraction,
            "denoise_median_size": denoise_median_size,
            "outside_hot_warning_celsius": outside_hot_warning_celsius,
            "max_admissible_celsius": max_admissible_celsius,
            "limiares_eta_pct": [thresholds.general_pct, thresholds.major_pct, thresholds.emergency_pct],
        },
    )
