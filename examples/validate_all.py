"""
Valida NUMERICAMENTE a máscara de material nas 4 imagens reais contra as
máscaras anotadas à mão (Tabela 7 do artigo), nas duas configurações de
erosão: a atual (borda do quadro preservada) e a da versão 0.1.0
(`--legacy-border` reproduz exatamente os valores publicados antes).

Espera os PNGs exportados pela ferramenta de anotação em
`tests/fixtures/masks/<nome_da_imagem>_mask_manual.png`, por exemplo
`IR_2497_t640_sample_mask_manual.png`.

    python examples/validate_all.py

Imprime uma tabela em Markdown e salva `examples/validation_all.json`.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermocore.io.loader import load_rjpeg
from thermocore.segmentation.background import estimate_material_mask
from thermocore.validation import evaluate_segmentation, load_mask_png

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
NAMES = {
    "IR_2497_t640_sample": "Transformador (IR_2497)",
    "IR_2501_t640_lines": "Linha aérea (IR_2501)",
    "IR_2503_t640_panel": "Painel/quadro (IR_2503)",
    "IR_2511_t640_switchgear": "Switchgear (IR_2511)",
}
FIELDS = ("iou", "dice", "pixel_accuracy", "precision", "recall")


def run(preserve_border: bool) -> dict:
    rows = {}
    for stem, label in NAMES.items():
        mask_path = FIX / "masks" / f"{stem}_mask_manual.png"
        if not mask_path.exists():
            return {}
        c = load_rjpeg(FIX / f"{stem}.jpg").celsius
        pred = estimate_material_mask(c, preserve_image_border=preserve_border)
        m = evaluate_segmentation(pred, load_mask_png(mask_path, c.shape))
        rows[label] = m.as_dict()
    return rows


def table(rows: dict) -> str:
    out = ["| Imagem | IoU | Dice | Acurácia | Precisão | Revocação |", "|---|---|---|---|---|---|"]
    for label, m in rows.items():
        out.append(f"| {label} | " + " | ".join(f"{m[f]:.3f}" for f in FIELDS) + " |")
    means = {f: sum(m[f] for m in rows.values()) / len(rows) for f in FIELDS}
    out.append("| **Média** | " + " | ".join(f"**{means[f]:.3f}**" for f in FIELDS) + " |")
    return "\n".join(out)


def main() -> None:
    current, legacy = run(True), run(False)
    if not current:
        print("Máscaras não encontradas em tests/fixtures/masks/. Exporte-as com a ferramenta de "
              "anotação (nome: <imagem>_mask_manual.png) e rode de novo.")
        sys.exit(1)
    print("== Configuração atual (borda do quadro preservada) ==\n" + table(current))
    print("\n== Configuração da v0.1.0 (--legacy-border: borda do quadro = fundo) ==\n" + table(legacy))
    (Path(__file__).resolve().parent / "validation_all.json").write_text(
        json.dumps({"atual": current, "legado_v0_1_0": legacy}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
