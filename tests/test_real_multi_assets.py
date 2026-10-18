"""
Testes sobre um segundo lote de imagens reais (FLIR T640), cedidas
pelo mesmo termografista parceiro, cobrindo ativos diferentes: linha
aérea com isoladores (IR_2501), painel/quadro (IR_2503) e
disjuntor/switchgear (IR_2511). Ampliam a amostra de validação além da
imagem única usada até aqui (mais material real).

ACHADO REGISTRADO (não um bug — ver `thermocore/segmentation/background.py`
e README, limitação "sem distinção de reflexo especular"): em
IR_2501 e IR_2511 a temperatura máxima extraída da matriz fica bem ACIMA
do valor mostrado na legenda da câmera (66,1 °C vs. 48,7 °C em IR_2501;
53,3 °C vs. 46,6 °C em IR_2511). Em IR_2501 o excesso está em uma região
isolada de 234 px (~0,08% da imagem, fios finos) FORA do material
estimado; em IR_2511 a maior região acima da legenda tem 450 px (~0,15%)
e recai sobre os terminais do módulo. O padrão é compatível com reflexo
especular/ruído pontual, mas NÃO o confirma: aquecimento real de conexão
também produziria manchas pequenas, e a legenda da câmera reflete a faixa
de contraste de exibição, não necessariamente os extremos radiométricos
(ver a Seção 5.2 do artigo). Em IR_2503 (painel) a extração bate com a
legenda a 0,2 °C, como na imagem de validação. Por isso os testes abaixo
NÃO exigem igualdade com a legenda para IR_2501/IR_2511 — testam
plausibilidade e registram o comportamento.
"""

from pathlib import Path

import pytest

from thermocore.io.loader import load_rjpeg

REAL_DIR = Path(__file__).parent / "fixtures"
pytestmark = pytest.mark.skipif(
    not (REAL_DIR / "IR_2501_t640_lines.jpg").exists(), reason="Fixtures reais não encontradas."
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"

REAL_ASSETS = {
    "IR_2501_t640_lines.jpg": {"legend_max": 48.7, "legend_min": -1.5},
    "IR_2503_t640_panel.jpg": {"legend_max": 36.8, "legend_min": 33.2},
    "IR_2511_t640_switchgear.jpg": {"legend_max": 46.6, "legend_min": 30.9},
}


@pytest.mark.parametrize("filename", REAL_ASSETS.keys())
def test_real_asset_loads_and_is_flir_t640(filename):
    path = FIXTURES_DIR / filename
    if not path.exists():
        pytest.skip(f"Fixture {filename} não encontrada.")
    t = load_rjpeg(path)
    assert t.is_synthetic is False
    assert t.camera_model == "FLIR T640"
    assert t.resolution == (640, 480)


@pytest.mark.parametrize("filename", REAL_ASSETS.keys())
def test_real_asset_temperature_is_physically_plausible(filename):
    path = FIXTURES_DIR / filename
    if not path.exists():
        pytest.skip(f"Fixture {filename} não encontrada.")
    celsius = load_rjpeg(path).celsius
    assert celsius.min() < celsius.max()
    assert -50.0 < celsius.min() < 200.0
    assert -50.0 < celsius.max() < 500.0


def test_panel_image_matches_camera_legend_closely():
    """IR_2503: cena sem estrutura fina/reflexiva — extração deve bater
    com a legenda, replicando o padrão da imagem de validação original."""
    path = FIXTURES_DIR / "IR_2503_t640_panel.jpg"
    if not path.exists():
        pytest.skip("Fixture não encontrada.")
    celsius = load_rjpeg(path).celsius
    legend = REAL_ASSETS["IR_2503_t640_panel.jpg"]
    assert celsius.max() == pytest.approx(legend["legend_max"], abs=0.5)
    assert celsius.min() == pytest.approx(legend["legend_min"], abs=0.5)


def test_wire_scene_max_exceeds_camera_legend_in_small_isolated_region():
    """IR_2501: documenta o achado do módulo — máximo bruto bem acima
    da legenda, concentrado em região pequena (candidato a reflexo)."""
    from scipy import ndimage as ndi

    path = FIXTURES_DIR / "IR_2501_t640_lines.jpg"
    if not path.exists():
        pytest.skip("Fixture não encontrada.")
    celsius = load_rjpeg(path).celsius
    legend_max = REAL_ASSETS["IR_2501_t640_lines.jpg"]["legend_max"]

    assert celsius.max() > legend_max + 10.0  # divergência grande e real

    above = celsius > legend_max
    labeled, n = ndi.label(above)
    largest = max((int((labeled == i).sum()) for i in range(1, n + 1)), default=0)
    assert above.mean() < 0.01          # menos de 1% da imagem
    assert largest < 300                 # regiões pequenas, não o equipamento inteiro
