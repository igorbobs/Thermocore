"""
Pipeline completo até o relatório DOCX sobre a imagem real de teste
(FLIR T640): carregamento → material → ponto quente/referência → ΔT/η →
severidade → imagem anotada → relatório (`examples/relatorio_t640.docx`).

    python examples/demo_full_report.py

Ver ressalvas em `demo_annotate_real_image.py` (referência automática
≠ ROI de termografista; um ponto quente por imagem; sem detecção de
reflexo). O relatório registra as ressalvas e os avisos da análise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermocore.pipeline import analyze_file, export_outputs


def main() -> None:
    examples_dir = Path(__file__).resolve().parent
    input_path = examples_dir.parent / "tests" / "fixtures" / "IR_2497_t640_sample.jpg"
    if not input_path.exists():
        print(f"Fixture não encontrada em: {input_path}")
        return

    # origem gravada como caminho relativo ao repositório (sem vazar caminhos da máquina)
    fa = analyze_file(input_path, source_label="tests/fixtures/IR_2497_t640_sample.jpg")
    image_path, report_path = export_outputs(
        fa, examples_dir / "output_annotated_t640.jpg", examples_dir / "relatorio_t640.docx",
        title="FLIR T640 — ponto quente e referência no material",
    )
    r = fa.result
    print(f"Imagem anotada: {image_path}")
    print(f"Relatório DOCX: {report_path}")
    print(f"ΔT = {r.delta_t_celsius:.1f} °C | η = {r.eta_pct:.1f}% | severidade: {r.severity.value}")
    for w in r.warnings:
        print(f"⚠ {w}")


if __name__ == "__main__":
    main()
