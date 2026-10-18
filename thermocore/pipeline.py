"""
Fluxo ponta a ponta sobre um arquivo RJPEG: carregamento → análise →
imagem anotada → registro de rastreabilidade → relatório DOCX.

Concentra em um só lugar o que antes era duplicado nos scripts de
exemplo, para que a CLI, os exemplos e os testes usem exatamente o
mesmo caminho de código (e, portanto, produzam os mesmos números).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from thermocore.analysis.inspection import InspectionResult, analyze_thermogram
from thermocore.io.loader import Thermogram, load_rjpeg
from thermocore.report.document import generate_inspection_report_docx
from thermocore.report.provenance import AnalysisRecord
from thermocore.visualization.annotate import save_annotated_image

METHOD_NAME = "otsu_material_mask+hotspot_max+reference_percentile+eta_DLT664"


@dataclass(frozen=True)
class FileAnalysis:
    thermogram: Thermogram
    result: InspectionResult
    record: AnalysisRecord


def build_record(
    thermogram: Thermogram,
    result: InspectionResult,
    source_label: str,
) -> AnalysisRecord:
    """
    Monta o `AnalysisRecord` de uma análise. `source_label` é o texto
    gravado como origem (use um caminho relativo ou o nome do arquivo,
    para não vazar caminhos absolutos da máquina do usuário).
    """
    m = thermogram.measurement
    sensitivity = [
        {
            "percentil": e.percentile,
            "t_referencia_c": e.reference_celsius,
            "delta_t_c": e.delta_t_celsius,
            "eta_pct": e.eta_pct,
            "severidade": e.severity.value,
        }
        for e in result.eta_by_percentile
    ]
    lo, hi = result.eta_range_pct
    return AnalysisRecord(
        source_path=source_label,
        is_synthetic=False,
        method=METHOD_NAME,
        parameters={
            "camera": thermogram.camera_model,
            "emissividade (arquivo)": m.emissivity,
            "distancia_objeto_m (arquivo)": m.object_distance_m,
            "temperatura_refletida_c (arquivo)": m.reflected_apparent_temperature_c,
            "umidade_relativa (arquivo)": m.relative_humidity,
            "T0_c": result.ambient_celsius,
            "T0_origem": result.ambient_source,
            "percentil_referencia": result.reference_percentile,
            "fracao_material": result.material_pixel_fraction,
            **result.parameters,
        },
        results={
            "hotspot_row": result.hotspot.row,
            "hotspot_col": result.hotspot.col,
            "t_quente_c": result.hotspot.value_celsius,
            "referencia_row": result.reference.row,
            "referencia_col": result.reference.col,
            "t_referencia_c": result.reference.value_celsius,
            "delta_t_c": result.delta_t_celsius,
            "eta_pct": result.eta_pct,
            "eta_faixa_pct": [lo, hi],
            "severity": result.severity.value,
            "severity_por_eta": result.severity_by_eta.value,
            "severity_faixa": [result.severity_range[0].value, result.severity_range[1].value],
            "severity_estavel": result.severity_is_stable,
            "limiar_otsu_c": result.otsu_threshold_celsius,
            "separabilidade_otsu": result.otsu_separability,
            "maximo_da_imagem_c": result.global_max_celsius,
            "sensibilidade_referencia": sensitivity,
            "avisos": list(result.warnings),
            **({"mta_excedida": result.mta_exceeded} if result.mta_exceeded is not None else {}),
        },
    )


def analyze_file(
    path: str | Path,
    ambient_celsius: float | None = None,
    source_label: str | None = None,
    **analysis_kwargs,
) -> FileAnalysis:
    """
    Carrega `path` (RJPEG radiométrico) e executa `analyze_thermogram`.
    `analysis_kwargs` são repassados à análise (percentil, limiares,
    ΔT mínimo, erosão, etc.). Levanta subclasses de `ThermocoreError`
    em qualquer falha prevista (arquivo inválido, T0 ausente, ...).
    """
    path = Path(path)
    t = load_rjpeg(path)
    r = analyze_thermogram(
        t.celsius,
        ambient_celsius=ambient_celsius,
        file_ambient_celsius=t.measurement.atmospheric_temperature_c,
        **analysis_kwargs,
    )
    return FileAnalysis(t, r, build_record(t, r, source_label or path.name))


def export_outputs(
    analysis: FileAnalysis,
    image_path: str | Path,
    report_path: str | Path,
    title: str | None = None,
) -> tuple[Path, Path]:
    """Gera a imagem anotada e o relatório DOCX (arquivos NOVOS)."""
    t, r = analysis.thermogram, analysis.result
    image = save_annotated_image(
        t.celsius,
        hotspot=r.hotspot,
        reference=r.reference,
        output_path=image_path,
        delta_t_celsius=r.delta_t_celsius,
        eta_pct=r.eta_pct,
        severity=r.severity,
        title=title,
        material_mask=r.material_mask,
        footnote="⚠ ver avisos no relatório" if r.warnings else None,
    )
    report = generate_inspection_report_docx(analysis.record, report_path, annotated_image_path=image)
    return image, report
