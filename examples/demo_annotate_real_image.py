"""
Gera uma cópia ANOTADA da imagem real de teste (FLIR T640): ponto mais
quente e ponto de referência AMBOS DENTRO DO MATERIAL (equipamento),
com ΔT, η e severidade.

    python examples/demo_annotate_real_image.py

O T0 do cálculo de η é lido dos metadados do próprio arquivo (nunca
assumido). A referência é automática (percentil baixo do material) e
NÃO substitui uma ROI escolhida por termografista; η é sensível a essa
escolha (ver artigo, análise de sensibilidade). Um ponto quente e uma
referência por imagem; sem distinção de reflexo especular.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermocore.analysis import analyze_thermogram
from thermocore.io.loader import load_rjpeg
from thermocore.visualization.annotate import save_annotated_image


def main() -> None:
    input_path = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "IR_2497_t640_sample.jpg"
    if not input_path.exists():
        print(f"Fixture não encontrada em: {input_path}")
        return

    t = load_rjpeg(input_path)
    result = analyze_thermogram(
        t.celsius, file_ambient_celsius=t.measurement.atmospheric_temperature_c
    )

    print("=" * 60)
    print("thermocore — anotação de imagem REAL (FLIR T640)")
    print("=" * 60)
    print(f"Câmera: {t.camera_model} | material estimado: {result.material_pixel_fraction*100:.1f}% da imagem")
    print(f"T0 ({result.ambient_source}): {result.ambient_celsius:.1f} °C | emissividade: {t.measurement.emissivity}")
    print(f"Ponto quente:  ({result.hotspot.row}, {result.hotspot.col}) = {result.hotspot.value_celsius:.1f} °C")
    print(f"Referência (P{result.reference_percentile:.0f} do material): ({result.reference.row}, {result.reference.col}) = {result.reference.value_celsius:.1f} °C")
    print(f"ΔT = {result.delta_t_celsius:.1f} °C | η = {result.eta_pct:.1f}% | severidade: {result.severity.value}")

    output_path = Path(__file__).resolve().parent / "output_annotated_t640.jpg"
    save_annotated_image(
        t.celsius,
        hotspot=result.hotspot,
        reference=result.reference,
        output_path=output_path,
        delta_t_celsius=result.delta_t_celsius,
        eta_pct=result.eta_pct,
        severity=result.severity,
        title="FLIR T640 — ponto quente e referência no material",
        material_mask=result.material_mask,
    )
    print(f"\nImagem anotada salva em: {output_path}")


if __name__ == "__main__":
    main()
