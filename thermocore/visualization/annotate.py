"""
Geração de uma cópia anotada da imagem térmica, marcando o ponto mais
quente e o ponto de referência (mais frio) usados no cálculo de ΔT/η,
junto com os valores e a classificação de severidade.

Este módulo NUNCA sobrescreve ou modifica a imagem original:
sempre escreve um arquivo NOVO no caminho indicado pelo chamador.

LIMITAÇÃO CONHECIDA (documentada também no README e no artigo do
projeto): esta primeira versão marca apenas UM ponto quente e UM ponto
de referência — não trata múltiplos pontos quentes no mesmo
equipamento nem tenta distinguir um ponto quente real de um reflexo
especular. Ambos os refinamentos estão planejados como próxima etapa
do projeto, deliberadamente adiados até a validação da base atual.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from thermocore.analysis.classification import AnomalySeverity
from thermocore.analysis.stats import ExtremePoint
from thermocore.visualization.render import render_thermal_image


def _label_offset(
    row: int, col: int, height: int, width: int, magnitude: int = 12
) -> tuple[int, int]:
    """
    Escolhe o deslocamento (em pontos) do rótulo de texto em relação ao
    marcador, de forma que o rótulo sempre aponte para o lado com mais
    espaço livre dentro da imagem — evita que o texto saia da figura
    ou colida com o título quando o ponto marcado está perto de uma
    borda (como aconteceu com um ponto de referência no topo de uma
    imagem real durante o desenvolvimento deste módulo).
    """
    dx = magnitude if col < width * 0.6 else -magnitude
    dy = -magnitude if row < height * 0.15 else magnitude
    return dx, dy


def _separate_labels(
    hot: tuple[int, int], ref: tuple[int, int], hot_off: tuple[int, int], ref_off: tuple[int, int],
    close_px: int = 90,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Se os dois pontos estão próximos (< `close_px` pixels), empurra o
    rótulo da referência para o lado oposto ao do ponto quente, para
    que os dois rótulos não se sobreponham.
    """
    dist = ((hot[0] - ref[0]) ** 2 + (hot[1] - ref[1]) ** 2) ** 0.5
    if dist >= close_px:
        return hot_off, ref_off
    return hot_off, (-abs(ref_off[0]) * 3 if hot_off[0] > 0 else abs(ref_off[0]) * 3, -hot_off[1] * 3)


def save_annotated_image(
    celsius: np.ndarray,
    hotspot: ExtremePoint,
    reference: ExtremePoint,
    output_path: str | Path,
    delta_t_celsius: float,
    eta_pct: float | None = None,
    severity: AnomalySeverity | None = None,
    colormap: str = "inferno",
    title: str | None = None,
    material_mask: np.ndarray | None = None,
    footnote: str | None = None,
) -> Path:
    """
    Renderiza `celsius` como mapa de calor, marca `hotspot` (ponto
    quente) e `reference` (ponto de referência/mais frio) com
    marcadores e rótulos de texto, e salva o resultado como um arquivo
    JPG novo em `output_path`.

    Parâmetros
    ----------
    celsius: matriz de temperatura completa (a mesma usada para achar
        `hotspot` e `reference`, via `thermocore.analysis.stats.locate_extreme`).
    hotspot, reference: posições e valores dos pontos a marcar.
    output_path: caminho do arquivo JPG a ser criado. O diretório pai é
        criado automaticamente se não existir. Se o arquivo já existir,
        é sobrescrito (mas nunca a imagem/arquivo de ORIGEM da análise).
    delta_t_celsius: ΔT já calculado entre hotspot e reference.
    eta_pct: η (%) já calculado, se disponível. Se None, não é exibido.
    severity: classificação de severidade do ponto quente, se
        disponível. Se None, não é exibido.
    colormap: paleta de cores usada em `render_thermal_image`.
    title: título opcional exibido no topo da figura.
    material_mask: máscara booleana opcional do material estimado; se
        fornecida, seu contorno é desenhado em verde para que o
        operador possa conferir visualmente o que foi tratado como
        equipamento.
    footnote: texto curto opcional exibido no rodapé da imagem (por
        exemplo, "⚠ ver avisos no relatório").

    Retorna o `Path` do arquivo criado.
    """
    import matplotlib

    matplotlib.use("Agg")  # backend sem interface gráfica — funciona em qualquer SO
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = render_thermal_image(celsius, colormap=colormap)
    height, width = celsius.shape[:2]

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=150)
    ax.imshow(rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10, pad=14)

    if material_mask is not None:
        ax.contour(material_mask.astype(float), levels=[0.5], colors="lime", linewidths=0.8)

    hot_off = _label_offset(hotspot.row, hotspot.col, height, width)
    ref_off = _label_offset(reference.row, reference.col, height, width)
    hot_off, ref_off = _separate_labels(
        (hotspot.row, hotspot.col), (reference.row, reference.col), hot_off, ref_off
    )

    # Ponto quente: marcador vermelho, rótulo acima.
    ax.plot(hotspot.col, hotspot.row, marker="o", markersize=10,
            markerfacecolor="none", markeredgecolor="red", markeredgewidth=2)
    hotspot_label = f"P1 (mais quente)\n{hotspot.value_celsius:.1f} °C"
    if severity is not None:
        hotspot_label += f"\nSeveridade: {severity.value}"
    ax.annotate(
        hotspot_label,
        xy=(hotspot.col, hotspot.row),
        xytext=hot_off,
        textcoords="offset points",
        ha="left" if hot_off[0] >= 0 else "right",
        color="red",
        fontsize=8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="red", alpha=0.6),
    )

    # Ponto de referência (mais frio): marcador ciano, rótulo abaixo.
    ax.plot(reference.col, reference.row, marker="o", markersize=10,
            markerfacecolor="none", markeredgecolor="cyan", markeredgewidth=2)
    reference_label = f"P2 (referência no material)\n{reference.value_celsius:.1f} °C"
    ax.annotate(
        reference_label,
        xy=(reference.col, reference.row),
        xytext=ref_off,
        textcoords="offset points",
        ha="left" if ref_off[0] >= 0 else "right",
        color="cyan",
        fontsize=8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="cyan", alpha=0.6),
    )

    # Caixa de resumo no canto, com ΔT e η.
    summary_lines = [f"ΔT = {delta_t_celsius:.1f} °C"]
    if eta_pct is not None:
        summary_lines.append(f"η = {eta_pct:.1f} %")
    ax.text(
        0.02, 0.02, "\n".join(summary_lines),
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="black", alpha=0.6),
    )

    if footnote:
        ax.text(
            0.98, 0.02, footnote, transform=ax.transAxes, fontsize=7, color="yellow",
            ha="right", va="bottom", bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6),
        )

    fig.tight_layout(pad=0.3)
    fig.savefig(output_path, format="jpeg", pil_kwargs={"quality": 92})
    plt.close(fig)

    return output_path
