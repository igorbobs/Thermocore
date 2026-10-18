"""
Analisa QUALQUER arquivo RJPEG radiométrico FLIR, gerando imagem
anotada + relatório DOCX (nunca sobrescreve o arquivo original).

    python examples/analyze_image.py caminho/para/imagem.jpg

Se o arquivo não trouxer temperatura ambiente (T0) nos metadados,
informe-a manualmente:

    python examples/analyze_image.py imagem.jpg --ambient 28.5

Outras opções:
    --percentile 10     percentil do material usado como referência
    --min-delta-t 2.0   ΔT mínimo (°C) para manter a classe de severidade
    --erosion 4         iterações de erosão da máscara de material
    --mta 85.0          máxima temperatura admissível do componente (°C),
                        se conhecida (NBR 15866); só gera aviso

SAÍDA: examples/output/<nome>_anotada.jpg e examples/output/<nome>_relatorio.docx

Ressalvas: a referência é automática (percentil do material estimado por
Otsu), não uma ROI escolhida por termografista; um ponto quente e uma
referência por imagem; sem distinção de reflexo especular. A saída é uma
triagem preliminar — leia os avisos impressos e os do relatório.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermocore.exceptions import AmbientTemperatureUnavailableError, ThermocoreError
from thermocore.pipeline import analyze_file, export_outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Análise termográfica de um RJPEG FLIR.")
    ap.add_argument("image", type=Path)
    ap.add_argument("--ambient", type=float, default=None, help="T0 em °C, se o arquivo não a trouxer")
    ap.add_argument("--percentile", type=float, default=10.0)
    ap.add_argument("--min-delta-t", type=float, default=2.0)
    ap.add_argument("--erosion", type=int, default=4)
    ap.add_argument("--mta", type=float, default=None, help="máxima temperatura admissível (°C)")
    a = ap.parse_args()

    try:
        fa = analyze_file(
            a.image, ambient_celsius=a.ambient, reference_percentile=a.percentile,
            min_delta_t_celsius=a.min_delta_t, erosion_iterations=a.erosion,
            max_admissible_celsius=a.mta,
        )
    except AmbientTemperatureUnavailableError as exc:
        print(f"Falha na análise de '{a.image.name}': {exc}")
        print("Dica: rode de novo com --ambient <valor em °C>.")
        sys.exit(1)
    except ThermocoreError as exc:
        print(f"Não foi possível processar '{a.image}': {exc}")
        sys.exit(1)

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)
    stem = a.image.stem
    image_out, report_out = export_outputs(
        fa, out_dir / f"{stem}_anotada.jpg", out_dir / f"{stem}_relatorio.docx",
        title=f"{fa.thermogram.camera_model or 'Câmera desconhecida'} — {stem}",
    )

    r = fa.result
    lo, hi = r.eta_range_pct
    print(f"Câmera: {fa.thermogram.camera_model} | material estimado: {r.material_pixel_fraction*100:.1f}%")
    print(f"T0 ({r.ambient_source}): {r.ambient_celsius:.1f} °C")
    print(f"Ponto quente: {r.hotspot.value_celsius:.1f} °C | Referência (P{r.reference_percentile:.0f}): {r.reference.value_celsius:.1f} °C")
    print(f"ΔT = {r.delta_t_celsius:.1f} °C | η = {r.eta_pct:.1f}% (faixa P1–P50: {lo:.0f}%–{hi:.0f}%) | Severidade: {r.severity.value}")
    for w in r.warnings:
        print(f"⚠ {w}")
    print(f"\nImagem anotada: {image_out}\nRelatório: {report_out}")


if __name__ == "__main__":
    main()
