"""
Reproduz TODOS os números e figuras de resultados do artigo a partir do
código e das imagens reais em `tests/fixtures/`, em uma única execução.

    python examples/reproduce_article.py

Saídas:
    examples/article_results.json        números citados no texto e nas tabelas
    examples/output/figures/fig*.png     Figuras 1, 2, 3, 4 e 5 do artigo

Fonte única da verdade: se o código mudar, rode este script de novo e o
artigo deve ser atualizado com os valores do JSON (e não à mão).

Observações:
  * Os valores de legenda da câmera (LEGENDA_MAX_C) foram lidos
    visualmente nas imagens FLIR e são o único dado digitado à mão.
  * A Figura 5 usa `benchmarks/benchmark_results.json` (medido pelo autor
    na máquina dele); as demais figuras e números são determinísticos.
  * As métricas de segmentação (Tabela 7) dependem das máscaras
    anotadas à mão: use `examples/validate_all.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import ndimage as ndi

from thermocore.analysis import analyze_thermogram, locate_extreme
from thermocore.io.loader import load_rjpeg
from thermocore.segmentation.background import estimate_material_mask, otsu_threshold
from thermocore.visualization.annotate import save_annotated_image
from thermocore.visualization.render import render_thermal_image

FIXTURES = ROOT / "tests" / "fixtures"
OUT = Path(__file__).resolve().parent / "output" / "figures"
IMAGES = {
    "IR_2497": ("IR_2497_t640_sample.jpg", "Transformador"),
    "IR_2501": ("IR_2501_t640_lines.jpg", "Linha aérea/isoladores"),
    "IR_2503": ("IR_2503_t640_panel.jpg", "Painel/quadro"),
    "IR_2511": ("IR_2511_t640_switchgear.jpg", "Switchgear/disjuntor"),
}
LEGENDA_MAX_C = {"IR_2497": 59.6, "IR_2501": 48.7, "IR_2503": 36.8, "IR_2511": 46.6}
LEGENDA_MIN_C = {"IR_2497": 13.7, "IR_2501": -1.5, "IR_2503": 33.2, "IR_2511": 30.9}
EROSIONS = (0, 2, 4, 6, 8)


def legacy_method_a(celsius, cold_percentile=20.0, min_region_fraction=0.03):
    """Método A (avaliado e descartado): regiões frias conexas e grandes = fundo."""
    cold = celsius <= np.percentile(celsius, cold_percentile)
    labeled, n = ndi.label(cold)
    sizes = np.bincount(labeled.ravel(), minlength=n + 1)
    keep = sizes >= min_region_fraction * celsius.size
    keep[0] = False
    return keep[labeled]


def hot_blobs(celsius, level):
    labeled, n = ndi.label(celsius > level)
    sizes = sorted((int(s) for s in np.bincount(labeled.ravel())[1:]), reverse=True)
    return {"regioes": n, "maiores_px": sizes[:4], "total_px": int((celsius > level).sum()),
            "total_pct": float((celsius > level).mean() * 100)}


def point(p):
    return {"row": p.row, "col": p.col, "valor_c": p.value_celsius}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out: dict = {"versao_software": __import__("thermocore").__version__, "imagens": {}}
    loaded = {}

    for key, (fname, nome) in IMAGES.items():
        t = load_rjpeg(FIXTURES / fname)
        c, m = t.celsius, t.measurement
        loaded[key] = (t, c)
        r = analyze_thermogram(c, file_ambient_celsius=m.atmospheric_temperature_c)
        legacy = estimate_material_mask(c, preserve_image_border=False)
        eros = []
        for it in EROSIONS:
            rr = analyze_thermogram(c, file_ambient_celsius=m.atmospheric_temperature_c, erosion_iterations=it)
            eros.append({"iteracoes": it, "t_quente_c": rr.hotspot.value_celsius,
                         "eta_pct": rr.eta_pct, "severidade": rr.severity.value,
                         "material_pct": rr.material_pixel_fraction * 100})
        out["imagens"][key] = {
            "arquivo": fname, "ativo": nome, "camera": t.camera_model,
            "medicao": {"emissividade": m.emissivity, "distancia_m": m.object_distance_m,
                        "t_atmosferica_c": m.atmospheric_temperature_c,
                        "t_refletida_c": m.reflected_apparent_temperature_c,
                        "umidade": m.relative_humidity},
            "matriz": {"min_c": float(c.min()), "max_c": float(c.max()),
                       "legenda_min_c": LEGENDA_MIN_C[key], "legenda_max_c": LEGENDA_MAX_C[key],
                       "dif_max_vs_legenda_c": float(c.max() - LEGENDA_MAX_C[key]),
                       "pixels_abaixo_10c": int((c < 10).sum()), "pixels_abaixo_10c_pct": float((c < 10).mean() * 100)},
            "regioes_acima_da_legenda": hot_blobs(c, LEGENDA_MAX_C[key]),
            "analise": {
                "limiar_otsu_c": r.otsu_threshold_celsius, "separabilidade_otsu": r.otsu_separability,
                "material_pct": r.material_pixel_fraction * 100,
                "material_pct_borda_legada": float(legacy.mean() * 100),
                "t_quente": point(r.hotspot), "referencia_p10": point(r.reference),
                "maximo_da_imagem_c": r.global_max_celsius,
                "t0_c": r.ambient_celsius, "delta_t_c": r.delta_t_celsius, "eta_pct": r.eta_pct,
                "severidade": r.severity.value, "severidade_por_eta": r.severity_by_eta.value,
                "severidade_faixa": [r.severity_range[0].value, r.severity_range[1].value],
                "severidade_estavel": r.severity_is_stable,
                "eta_faixa_pct": list(r.eta_range_pct),
                "sensibilidade": [{"percentil": e.percentile, "t_ref_c": e.reference_celsius,
                                   "delta_t_c": e.delta_t_celsius, "eta_pct": e.eta_pct,
                                   "severidade": e.severity.value} for e in r.eta_by_percentile],
                "avisos": list(r.warnings),
            },
            "sensibilidade_erosao": eros,
        }

    # ---- imagem de validação (transformador): comparação de métodos -------------
    t, c = loaded["IR_2497"]
    t0 = t.measurement.atmospheric_temperature_c
    thr = otsu_threshold(c)
    gmin, gmax = locate_extreme(c, mode="min"), locate_extreme(c, mode="max")
    bg_a = legacy_method_a(c)
    a_min = locate_extreme(c, mask=~bg_a, mode="min")
    mat_b = estimate_material_mask(c)
    b_min = locate_extreme(c, mask=mat_b, mode="min")
    r = analyze_thermogram(c, file_ambient_celsius=t0)
    out["metodos_transformador"] = {
        "quadro_inteiro": {"t_quente_c": gmax.value_celsius, "t_frio_c": gmin.value_celsius,
                           "delta_t_c": gmax.value_celsius - gmin.value_celsius,
                           "eta_com_t0_20c_pct": (gmax.value_celsius - gmin.value_celsius) / (gmax.value_celsius - 20.0) * 100,
                           "eta_com_t0_arquivo_pct": (gmax.value_celsius - gmin.value_celsius) / (gmax.value_celsius - t0) * 100},
        "metodo_A": {"fundo_pct": float(bg_a.mean() * 100), "t_frio_no_material_c": a_min.value_celsius, **{"posicao": [a_min.row, a_min.col]}},
        "metodo_B": {"limiar_otsu_c": thr, "material_sem_erosao_pct": float((c > thr).mean() * 100),
                     "material_final_pct": float(mat_b.mean() * 100), "minimo_bruto_c": b_min.value_celsius,
                     "referencia_p10_c": r.reference.value_celsius},
    }

    (Path(__file__).resolve().parent / "article_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # =========================== FIGURAS ===========================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    # ---- Figura 1: fluxo ----------------------------------------------------------
    steps = ["Arquivo\nRJPEG (FLIR)", "Extração\nradiométrica\n(flyr)", "Validação e\nmatriz em °C",
             "Material\n(Otsu +\nerosão)", "Ponto quente\ne referência", "ΔT, η,\nseveridade,\nfaixa e avisos",
             "Imagem anotada\ne relatório", "Registro de\nrastreabilidade"]
    fig, ax = plt.subplots(figsize=(11.6, 2.4), dpi=150)
    ax.set_xlim(0, len(steps) * 1.25); ax.set_ylim(-0.9, 1.1); ax.axis("off")
    for i, s in enumerate(steps):
        x = i * 1.25
        ax.add_patch(FancyBboxPatch((x, 0), 1.0, 1.0, boxstyle="round,pad=0.02", fc="#e8f0fa", ec="#2f5f98", lw=1))
        ax.text(x + 0.5, 0.5, s, ha="center", va="center", fontsize=7)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.23, 0.5), xytext=(x + 1.02, 0.5), arrowprops=dict(arrowstyle="->", color="#2f5f98"))
    ax.add_patch(FancyBboxPatch((2.5, -0.75), 1.0, 0.55, boxstyle="round,pad=0.02", fc="white", ec="#888", lw=1, ls="--"))
    ax.text(3.0, -0.475, "Filtro de mediana\n(opcional)", ha="center", va="center", fontsize=6.5, color="#555")
    ax.annotate("", xy=(3.75, 0.0), xytext=(3.5, -0.3), arrowprops=dict(arrowstyle="->", color="#888", ls="--"))
    fig.savefig(OUT / "fig1_fluxo.png", bbox_inches="tight"); plt.close(fig)

    # ---- Figura 2: comparação de métodos -----------------------------------------
    rgb = render_thermal_image(c)
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 3.5), dpi=150)
    for a in axs:
        a.imshow(rgb); a.set_xticks([]); a.set_yticks([])
    axs[0].set_title("(a) Quadro inteiro\n(mínimo global no céu)", fontsize=8)
    axs[0].plot(gmin.col, gmin.row, "o", mfc="none", mec="cyan", ms=9, mew=2, clip_on=False)
    axs[0].text(gmin.col + 14, gmin.row + 24, f"{gmin.value_celsius:.1f} °C (céu)", color="cyan", fontsize=7, weight="bold")
    axs[1].set_title(f"(b) Método A: região fria conexa\n(fundo = {bg_a.mean()*100:.0f}%; mínimo ainda no céu)", fontsize=8)
    axs[1].contour(bg_a.astype(float), levels=[0.5], colors="lime", linewidths=0.9)
    axs[1].plot(a_min.col, a_min.row, "o", mfc="none", mec="cyan", ms=9, mew=2)
    axs[2].set_title(f"(c) Método B: Otsu + erosão\n(material = {mat_b.mean()*100:.0f}%)", fontsize=8)
    axs[2].contour(mat_b.astype(float), levels=[0.5], colors="lime", linewidths=0.9)
    axs[2].plot(r.hotspot.col, r.hotspot.row, "o", mfc="none", mec="red", ms=9, mew=2)
    axs[2].text(r.hotspot.col + 10, r.hotspot.row + 4, f"P1 {r.hotspot.value_celsius:.1f}", color="red", fontsize=7, weight="bold")
    axs[2].plot(r.reference.col, r.reference.row, "o", mfc="none", mec="cyan", ms=9, mew=2)
    axs[2].text(r.reference.col + 10, r.reference.row + 20, f"P2 {r.reference.value_celsius:.1f} °C", color="cyan", fontsize=7, weight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig2_metodos.png"); plt.close(fig)

    # ---- Figura 3: η x percentil -----------------------------------------------------
    sens = r.eta_by_percentile
    xs = [s.percentile for s in sens]; ys = [s.eta_pct for s in sens]
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=150)
    ax.axhspan(35, 80, color="#fff1cc", label="Geral (≥35%)")
    ax.axhspan(80, 95, color="#f8d6b8", label="Grave (≥80%)")
    ax.plot(xs, ys, "-o", color="#2f5f98")
    for x_, y_ in zip(xs, ys):
        ax.annotate(f"{y_:.0f}%", (x_, y_), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{x:.0f}" for x in xs])
    ax.set_ylim(40, 95); ax.set_xlabel("Percentil do material usado como referência (P)"); ax.set_ylabel("η (%)")
    ax.legend(loc="upper right", fontsize=7); fig.tight_layout(); fig.savefig(OUT / "fig3_sensibilidade.png"); plt.close(fig)

    # ---- Figura 4: saída final (transformador) -----------------------------------------
    save_annotated_image(c, r.hotspot, r.reference, OUT / "fig4_saida.jpg", r.delta_t_celsius, r.eta_pct,
                         r.severity, title="FLIR T640 — ponto quente e referência no material",
                         material_mask=r.material_mask)

    # ---- Figura 5: desempenho (medido pelo autor) --------------------------------------
    bench = ROOT / "benchmarks" / "benchmark_results.json"
    if bench.exists():
        d = json.loads(bench.read_text(encoding="utf-8"))
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), dpi=150)
        px = [row["pixel_count"] for row in d["preprocessing_by_resolution"]]
        for name, label, mk in [("preprocessing_by_resolution", "Pré-processamento", "o"),
                                ("segmentation_by_resolution", "Segmentação (percentil)", "s"),
                                ("full_pipeline_by_resolution", "Pipeline de referência (legado)", "^"),
                                ("current_analysis_by_resolution", "Análise atual (Otsu)", "D")]:
            if name in d:
                a1.plot(px, [row["time_per_call_seconds"] * 1000 for row in d[name]], marker=mk, label=label)
        a1.set_xscale("log"); a1.set_yscale("log"); a1.axvline(307200, color="gray", ls=":", lw=0.8)
        a1.set_xlabel("Pixels por imagem"); a1.set_ylabel("Tempo por imagem (ms)"); a1.set_title("(a) Escala com a resolução", fontsize=9)
        a1.legend(fontsize=7); a1.grid(alpha=0.3, which="both")
        bs = [row["batch_size"] for row in d["batch_scalability"]]; ips = [row["images_per_second"] for row in d["batch_scalability"]]
        a2.plot(range(len(bs)), ips, "-o", color="#2e7d32")
        for i_, v_ in enumerate(ips):
            a2.annotate(f"{v_:.1f}", (i_, v_), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        a2.set_xticks(range(len(bs))); a2.set_xticklabels(bs); a2.set_ylim(0, 45)
        a2.set_xlabel("Tamanho do lote (imagens 640×480)"); a2.set_ylabel("Vazão (imagens/s)")
        a2.set_title("(b) Vazão por tamanho de lote (pipeline legado, em memória)", fontsize=9); a2.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / "fig5_desempenho.png"); plt.close(fig)

    print(json.dumps({k: out[k] for k in ("versao_software", "metodos_transformador")}, indent=2, ensure_ascii=False))
    for key, v in out["imagens"].items():
        a = v["analise"]
        print(f"{key}: T1={a['t_quente']['valor_c']:.1f} T2={a['referencia_p10']['valor_c']:.1f} "
              f"ΔT={a['delta_t_c']:.1f} η={a['eta_pct']:.1f}% {a['severidade']} faixa={a['severidade_faixa']}")


if __name__ == "__main__":
    main()
