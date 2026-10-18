"""
Teste de integração: exercita o pipeline completo — pré-processamento
-> segmentação -> estatísticas -> ΔT/η -> classificação -> registro de
rastreabilidade — sobre uma imagem térmica SINTÉTICA de ponta a ponta.

Este teste é o mais próximo que o projeto tem, hoje, de uma "prova de
conceito" do fluxo descrito na metodologia do artigo. Ele NÃO prova
nada sobre o comportamento do sistema com dados reais de outras
câmeras ou condições além da única imagem real de FLIR T640 já
validada em tests/test_loader.py —
isso continua pendente até a chegada de um arquivo de exemplo real.
"""

from thermocore.analysis.classification import (
    AnomalySeverity,
    classify_by_relative_rise,
)
from thermocore.analysis.stats import compute_stats, relative_temperature_rise
from thermocore.io.synthetic import make_flat_thermogram, make_hotspot_thermogram
from thermocore.processing.preprocessing import denoise_gaussian
from thermocore.report.provenance import AnalysisRecord
from thermocore.segmentation.threshold import (
    label_connected_regions,
    segment_above_threshold,
)

AMBIENT_CELSIUS = 24.0
THRESHOLD_CELSIUS = 60.0


def test_full_pipeline_flags_major_or_emergency_for_strong_hotspot():
    # 1) "Aquisição" (aqui, sintética) de uma imagem com uma falha
    #    simulada de conexão elétrica (ponto quente forte) e de uma
    #    imagem de referência homogênea (fase saudável).
    hotspot_image = make_hotspot_thermogram(
        height=120,
        width=160,
        base_celsius=25.0,
        hotspot_center=(60, 80),
        hotspot_radius=6,
        hotspot_peak_celsius=95.0,
        noise_std_celsius=0.5,
        seed=7,
    )
    reference_image = make_flat_thermogram(height=120, width=160, base_celsius=25.5)

    # 2) Pré-processamento (redução de ruído).
    hotspot_celsius = denoise_gaussian(hotspot_image.celsius, sigma=1.0)

    # 3) Segmentação da região de interesse.
    mask = segment_above_threshold(hotspot_celsius, threshold_celsius=THRESHOLD_CELSIUS)
    regions = label_connected_regions(mask, min_pixel_count=4)
    assert len(regions) == 1, "esperava exatamente um ponto quente na imagem sintética"
    hotspot_region_mask = regions[0].mask

    # 4) Indicadores térmicos da região segmentada e da referência.
    hotspot_stats = compute_stats(hotspot_celsius, mask=hotspot_region_mask)
    reference_stats = compute_stats(reference_image.celsius)

    # 5) Diferença relativa de temperatura (η) e classificação.
    eta_pct = relative_temperature_rise(hotspot_stats, reference_stats, AMBIENT_CELSIUS)
    severity = classify_by_relative_rise(eta_pct)

    assert severity in (AnomalySeverity.MAJOR, AnomalySeverity.EMERGENCY)

    # 6) Registro de rastreabilidade — nunca esquecido, mesmo em teste.
    record = AnalysisRecord(
        source_path=hotspot_image.source_id,
        is_synthetic=True,
        method="gaussian_denoise+threshold_segmentation+eta_DLT664",
        parameters={
            "gaussian_sigma": 1.0,
            "threshold_celsius": THRESHOLD_CELSIUS,
            "min_pixel_count": 4,
            "ambient_celsius": AMBIENT_CELSIUS,
        },
        results={
            "t_max_celsius": hotspot_stats.t_max,
            "eta_pct": eta_pct,
            "severity": severity.value,
        },
    )
    assert record.is_synthetic is True
    assert record.results["severity"] in ("grave", "emergencial")


def test_full_pipeline_reports_normal_for_healthy_equipment():
    # Imagem "saudável": variação térmica pequena, sem ponto quente.
    healthy_image = make_hotspot_thermogram(
        base_celsius=25.0,
        hotspot_peak_celsius=27.0,  # variação de só 2°C — não é uma falha
        hotspot_radius=6,
    )
    reference_image = make_flat_thermogram(base_celsius=25.0)

    mask = segment_above_threshold(healthy_image.celsius, threshold_celsius=THRESHOLD_CELSIUS)
    regions = label_connected_regions(mask, min_pixel_count=4)

    # Nenhuma região deve ultrapassar o limiar de 60°C nesta imagem saudável.
    assert regions == []
