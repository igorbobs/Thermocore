"""
Valida NUMERICAMENTE a máscara de material do Otsu contra uma máscara
de referência anotada à mão.

FLUXO:
  1. Abra a ferramenta de anotação (annotate_mask.html) no navegador,
     carregue a imagem térmica, contorne o equipamento clicando, feche
     o polígono e exporte a máscara PNG.
  2. Salve o PNG exportado (ele já vem com o nome
     "<nome_da_imagem>_mask_manual.png").
  3. Rode:

        python examples/validate_segmentation.py imagem.jpg mascara.png [--legacy-border]

     `--legacy-border` reproduz a erosão da versão 0.1.0 (borda do
     quadro tratada como fundo).

Salva `examples/output/<nome>_otsu_validation.json` (números) e
`..._otsu_validation.png` (figura: referência, predição do Otsu, e
mapa de concordância — verde=acerto, vermelho=falso positivo do Otsu,
amarelo=falso negativo do Otsu).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from thermocore.io.loader import load_rjpeg
from thermocore.segmentation.background import estimate_material_mask
from thermocore.validation import evaluate_segmentation, load_mask_png
from thermocore.visualization.render import render_thermal_image


def main() -> None:
    if len([a for a in sys.argv[1:] if not a.startswith('--')]) < 2:
        print(__doc__)
        sys.exit(1)

    legacy = "--legacy-border" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    image_path = Path(args[0])
    mask_path = Path(args[1])

    t = load_rjpeg(image_path)
    celsius = t.celsius
    pred = estimate_material_mask(celsius, preserve_image_border=not legacy)
    true = load_mask_png(mask_path, celsius.shape)

    metrics = evaluate_segmentation(pred, true)

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)
    stem = image_path.stem

    result = {
        "imagem": str(image_path),
        "mascara_referencia": str(mask_path),
        "camera": t.camera_model,
        **metrics.as_dict(),
        "pixels_material_predito": int(pred.sum()),
        "pixels_material_referencia": int(true.sum()),
        "fracao_material_predito": float(pred.mean()),
        "fracao_material_referencia": float(true.mean()),
    }
    json_path = out_dir / f"{stem}_otsu_validation.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = render_thermal_image(celsius)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.8), dpi=200)
    for a in axs:
        a.set_xticks([]); a.set_yticks([])
    axs[0].imshow(rgb)
    axs[0].contour(true.astype(float), levels=[0.5], colors="yellow", linewidths=1.4)
    axs[0].set_title("Referência manual", fontsize=9)
    axs[1].imshow(rgb)
    axs[1].contour(pred.astype(float), levels=[0.5], colors="lime", linewidths=1.4)
    axs[1].set_title("Predição (Otsu + erosão)", fontsize=9)

    overlap = np.zeros((*celsius.shape, 3), dtype=np.uint8)
    overlap[np.logical_and(pred, true)] = [40, 200, 40]
    overlap[np.logical_and(pred, ~true)] = [220, 50, 50]
    overlap[np.logical_and(~pred, true)] = [230, 200, 40]
    axs[2].imshow(overlap)
    axs[2].set_title(
        f"IoU={metrics.iou:.2f}  Dice={metrics.dice:.2f}\n"
        "verde=acerto  vermelho=falso+  amarelo=falso-", fontsize=8
    )
    fig.tight_layout()
    fig_path = out_dir / f"{stem}_otsu_validation.png"
    fig.savefig(fig_path)
    plt.close(fig)

    print(f"IoU={metrics.iou:.3f}  Dice={metrics.dice:.3f}  Acurácia={metrics.pixel_accuracy:.3f}  "
          f"Precisão={metrics.precision:.3f}  Revocação={metrics.recall:.3f}")
    print(f"\nJSON: {json_path}")
    print(f"Figura: {fig_path}")


if __name__ == "__main__":
    main()
