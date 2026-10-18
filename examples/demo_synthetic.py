"""
Demonstração do pipeline completo do thermocore, de ponta a ponta,
usando dados SINTÉTICOS (nunca um resultado experimental real — ver
README.md e a regra final do projeto).

Rodar com:

    python examples/demo_synthetic.py

a partir da raiz do repositório (com o ambiente virtual ativado e as
dependências instaladas). Não é necessário configurar PYTHONPATH
manualmente — este script adiciona a raiz do projeto ao caminho de
busca por conta própria (ver bloco logo abaixo), já que rodar um
arquivo diretamente com `python arquivo.py` só coloca a pasta do
próprio arquivo no `sys.path`, não a raiz do projeto onde fica o
pacote `thermocore`.
"""

import sys
from pathlib import Path

# Garante que a raiz do projeto (a pasta que contém o pacote
# `thermocore/`) está no caminho de busca de módulos, mesmo quando
# este arquivo é executado diretamente (`python examples/demo_synthetic.py`)
# em vez de via `python -m` ou com PYTHONPATH configurado à mão.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermocore.analysis.classification import classify_by_relative_rise
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


def main() -> None:
    print("=" * 60)
    print("thermocore — demonstração com dados SINTÉTICOS")
    print("(nenhum arquivo real da câmera foi usado nesta execução)")
    print("=" * 60)

    # 1) "Aquisição" sintética: uma imagem com ponto quente simulando
    #    uma falha de conexão, e uma imagem de referência homogênea.
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

    # 2) Pré-processamento.
    hotspot_celsius = denoise_gaussian(hotspot_image.celsius, sigma=1.0)

    # 3) Segmentação da região de interesse.
    mask = segment_above_threshold(hotspot_celsius, threshold_celsius=THRESHOLD_CELSIUS)
    regions = label_connected_regions(mask, min_pixel_count=4)

    if not regions:
        print("Nenhuma região acima do limiar foi encontrada.")
        return

    hotspot_region_mask = regions[0].mask

    # 4) Indicadores térmicos.
    hotspot_stats = compute_stats(hotspot_celsius, mask=hotspot_region_mask)
    reference_stats = compute_stats(reference_image.celsius)

    print(f"\nRegiões detectadas: {len(regions)}")
    print(f"Pixels na região do ponto quente: {hotspot_stats.pixel_count}")
    print(f"Tmáx (ponto quente): {hotspot_stats.t_max:.1f} °C")
    print(f"Tmín (ponto quente): {hotspot_stats.t_min:.1f} °C")
    print(f"Tmédia (ponto quente): {hotspot_stats.t_mean:.1f} °C")
    print(f"Desvio padrão: {hotspot_stats.t_std:.2f} °C")

    # 5) ΔT / η e classificação.
    eta_pct = relative_temperature_rise(hotspot_stats, reference_stats, AMBIENT_CELSIUS)
    severity = classify_by_relative_rise(eta_pct)

    print(f"\nη (diferença relativa de temperatura): {eta_pct:.1f}%")
    print(f"Classificação de severidade: {severity.value}")

    # 6) Registro de rastreabilidade.
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
    output_path = "analysis_record_demo.json"
    record.save_json(output_path)
    print(f"\nRegistro de rastreabilidade salvo em: {output_path}")
    print("\nLembrete: todos os números acima vêm de dados SINTÉTICOS.")


if __name__ == "__main__":
    main()
