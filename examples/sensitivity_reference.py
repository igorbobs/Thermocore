"""
Análise de sensibilidade do η e da severidade ao critério de escolha do
ponto de referência (percentil do material), sobre a imagem real de
teste (FLIR T640). Também compara, na mesma imagem, os métodos de
estimativa de fundo/material avaliados no projeto. Salva
`examples/sensitivity_reference.json` (rastreável).

    python examples/sensitivity_reference.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import ndimage as ndi

from thermocore.analysis import classify_by_relative_rise, locate_extreme, locate_reference_point
from thermocore.io.loader import load_rjpeg
from thermocore.segmentation.background import estimate_material_mask, otsu_threshold

PERCENTILES = [1, 5, 10, 25, 50]


def legacy_percentile_connected_background(celsius, cold_percentile=20.0, min_region_fraction=0.03):
    """Método A (avaliado e DESCARTADO): regiões frias conexas e grandes = fundo."""
    cold = celsius <= np.percentile(celsius, cold_percentile)
    labeled, n = ndi.label(cold)
    bg = np.zeros_like(cold)
    for i in range(1, n + 1):
        region = labeled == i
        if region.sum() >= min_region_fraction * celsius.size:
            bg |= region
    return bg


def main() -> None:
    path = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "IR_2497_t640_sample.jpg"
    t = load_rjpeg(path)
    c = t.celsius
    t0 = t.measurement.atmospheric_temperature_c
    out = {"arquivo": path.name, "camera": t.camera_model, "T0_arquivo_c": t0}

    # Método 0: quadro inteiro (sem restrição)
    g_hot, g_cold = locate_extreme(c, mode="max"), locate_extreme(c, mode="min")
    out["metodo_quadro_inteiro"] = {"t_quente": g_hot.value_celsius, "t_frio": g_cold.value_celsius,
                                    "delta_t": g_hot.value_celsius - g_cold.value_celsius}

    # Método A: percentil + região conexa grande (descartado)
    bg_a = legacy_percentile_connected_background(c)
    a_cold = locate_extreme(c, mask=~bg_a, mode="min")
    out["metodo_A_percentil_conexo"] = {"fundo_pct": float(bg_a.mean() * 100),
                                        "t_frio_no_material": a_cold.value_celsius,
                                        "posicao": [a_cold.row, a_cold.col]}

    # Método B: Otsu + limpeza + erosão (adotado)
    thr = otsu_threshold(c)
    raw = c > thr
    mat = estimate_material_mask(c)
    b_min = locate_extreme(c, mask=mat, mode="min")
    out["metodo_B_otsu"] = {"limiar_otsu_c": thr, "material_sem_erosao_pct": float(raw.mean() * 100),
                            "material_final_pct": float(mat.mean() * 100),
                            "minimo_bruto_no_material_c": b_min.value_celsius}

    hot = locate_extreme(c, mask=mat, mode="max")
    rows = []
    for p in PERCENTILES:
        ref = locate_reference_point(c, mat, percentile=p)
        dt = hot.value_celsius - ref.value_celsius
        eta = dt / (hot.value_celsius - t0) * 100
        rows.append({"percentil": p, "t_ref_c": ref.value_celsius, "delta_t_c": dt,
                     "eta_pct": eta, "severidade": classify_by_relative_rise(eta).value})
    out["t_quente_material_c"] = hot.value_celsius
    out["sensibilidade_referencia"] = rows

    dest = Path(__file__).resolve().parent / "sensitivity_reference.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
