from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from thermocore.exceptions import NoRadiometricDataError, UnsupportedFileError
from thermocore.io.loader import load_rjpeg, validate_file


def test_validate_file_raises_for_missing_file(tmp_path):
    missing = tmp_path / "nao_existe.jpg"
    with pytest.raises(UnsupportedFileError):
        validate_file(missing)


def test_validate_file_raises_for_unsupported_extension(tmp_path):
    fake = tmp_path / "dados.csv"
    fake.write_text("nao e uma imagem")
    with pytest.raises(UnsupportedFileError):
        validate_file(fake)


def test_validate_file_accepts_jpg_extension(tmp_path):
    fake = tmp_path / "foto.jpg"
    fake.write_bytes(b"\xff\xd8\xff")  # cabecalho minimo de JPEG, conteudo invalido
    # validate_file so checa existencia + extensao, nao le o conteudo
    result = validate_file(fake)
    assert result == fake


def test_load_rjpeg_raises_no_radiometric_data_for_plain_jpg(tmp_path):
    """
    Um JPEG comum (sem qualquer bloco de metadados radiométricos da
    FLIR) deve ser rejeitado explicitamente, nunca processado como se
    tivesse temperatura válida.
    """
    plain_jpg_path = tmp_path / "foto_comum.jpg"
    # Gera uma imagem RGB comum de verdade (JPEG válido, sem radiometria).
    img = Image.fromarray(
        (np.random.default_rng(0).random((20, 20, 3)) * 255).astype(np.uint8)
    )
    img.save(plain_jpg_path, format="JPEG")

    with pytest.raises(NoRadiometricDataError):
        load_rjpeg(plain_jpg_path)


def test_load_rjpeg_raises_unsupported_file_for_missing_path(tmp_path):
    with pytest.raises(UnsupportedFileError):
        load_rjpeg(tmp_path / "inexistente.jpg")


# ---------------------------------------------------------------------------
# PRIMEIRA VALIDAÇÃO COM ARQUIVO REAL (não sintético)
# ---------------------------------------------------------------------------
# Fixture: RJPEG real de uma câmera FLIR T640, cedida por um termografista
# profissional para fins de teste deste projeto (não é uma foto tirada por
# Igor Pantoja, nem capturada especificamente para este TCC — é material de
# terceiro, usado aqui só para validar a extração radiométrica).
#
# Valores de referência abaixo (Tmáx ≈ 59.8°C) foram conferidos manualmente
# contra a legenda de temperatura desenhada pela própria câmera na imagem
# (59.6°C) — diferença de 0.2°C, dentro da margem normal de arredondamento.
# Este é, portanto, o primeiro resultado experimental real do projeto,
# não mais um dado sintético.

REAL_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "IR_2497_t640_sample.jpg"


@pytest.mark.skipif(
    not REAL_FIXTURE_PATH.exists(),
    reason="Fixture real (tests/fixtures/IR_2497_t640_sample.jpg) não encontrada.",
)
def test_load_rjpeg_reads_real_flir_t640_file():
    t = load_rjpeg(REAL_FIXTURE_PATH)

    assert t.is_synthetic is False
    assert t.camera_model == "FLIR T640"
    assert t.resolution == (640, 480)

    # Tmáx extraído deve bater, com pequena tolerância, com o valor mostrado
    # na legenda da própria câmera na imagem original (59.6°C).
    assert t.celsius.max() == pytest.approx(59.6, abs=1.0)


@pytest.mark.skipif(
    not REAL_FIXTURE_PATH.exists(),
    reason="Fixture real (tests/fixtures/IR_2497_t640_sample.jpg) não encontrada.",
)
def test_full_pipeline_on_real_image_finds_hotspot_on_equipment():
    """
    Roda o pipeline (pré-processamento + segmentação) sobre a imagem REAL
    e confirma que a região mais quente detectada é fisicamente plausível
    (dentro dos limites da imagem, com temperatura próxima do Tmáx global).
    Não valida o VALOR absoluto de ΔT/η aqui — isso exigiria uma ROI de
    referência escolhida por um termografista, fora do escopo deste teste
    automatizado.
    """
    from thermocore.processing.preprocessing import denoise_gaussian
    from thermocore.segmentation.threshold import (
        label_connected_regions,
        segment_above_percentile,
    )
    from thermocore.analysis.stats import compute_stats

    t = load_rjpeg(REAL_FIXTURE_PATH)
    celsius = denoise_gaussian(t.celsius, sigma=1.0)

    mask = segment_above_percentile(celsius, percentile=99.0)
    regions = label_connected_regions(mask, min_pixel_count=10)

    assert len(regions) >= 1
    hottest_region = max(regions, key=lambda r: compute_stats(celsius, mask=r.mask).t_max)
    hottest_stats = compute_stats(celsius, mask=hottest_region.mask)

    assert hottest_stats.t_max == pytest.approx(t.celsius.max(), abs=0.5)



@pytest.mark.skipif(not REAL_FIXTURE_PATH.exists(), reason="Fixture real não encontrada.")
def test_real_file_exposes_measurement_parameters_from_camera_metadata():
    m = load_rjpeg(REAL_FIXTURE_PATH).measurement
    assert m.emissivity == pytest.approx(0.83, abs=0.01)
    assert m.object_distance_m == pytest.approx(12.0)
    assert m.atmospheric_temperature_c == pytest.approx(34.0, abs=0.1)
    assert m.reflected_apparent_temperature_c == pytest.approx(32.0, abs=0.1)
    assert m.relative_humidity == pytest.approx(0.57, abs=0.01)


@pytest.mark.skipif(not REAL_FIXTURE_PATH.exists(), reason="Fixture real não encontrada.")
def test_real_image_analysis_uses_ambient_from_file_and_keeps_reference_in_material():
    from thermocore.analysis import analyze_thermogram

    t = load_rjpeg(REAL_FIXTURE_PATH)
    r = analyze_thermogram(t.celsius, file_ambient_celsius=t.measurement.atmospheric_temperature_c)
    assert r.ambient_source == "arquivo"
    assert r.hotspot.value_celsius == pytest.approx(59.8, abs=0.3)
    # A referência dentro do material NÃO pode ser o céu (~8 °C) nem o limiar bruto:
    assert r.reference.value_celsius > 30.0
    assert 0.0 < r.eta_pct < 100.0
