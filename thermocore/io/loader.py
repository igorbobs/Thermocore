"""
Carregamento e validação de arquivos termográficos reais.

Este módulo NUNCA modifica o arquivo original: apenas lê
seus bytes e retorna estruturas de dados derivadas em memória.

A extração radiométrica usa a biblioteca `flyr`, que reimplementa em
Python puro o analisador de metadados RJPEG da FLIR (ver a seção de
trabalhos relacionados / fundamentação do artigo). Essa biblioteca foi
validada por seus autores em câmeras FLIR C2/E4/E5/E6/E8 — a
compatibilidade com o RJPEG da FLIR T640 (câmera de referência deste
projeto) foi confirmada empiricamente com um arquivo real de teste
(ver `tests/test_loader.py` e `tests/fixtures/`). Qualquer falha de
leitura em outros arquivos/câmeras é reportada como uma exceção
específica, nunca como um resultado silenciosamente incorreto.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from thermocore.exceptions import (
    CorruptedRadiometricDataError,
    MissingDependencyError,
    NoRadiometricDataError,
    UnsupportedFileError,
)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg"}


@dataclass(frozen=True)
class MeasurementParameters:
    """
    Parâmetros de medição gravados pela câmera no próprio arquivo
    RJPEG (todos opcionais: ficam None se o arquivo não os trouxer —
    nunca são preenchidos com valores inventados).
    Temperaturas em °C, distância em metros, umidade em fração 0-1.

    ATENÇÃO: estes valores são os parâmetros CONFIGURADOS na câmera (ou
    digitados pelo operador) no momento da captura, usados na compensação
    radiométrica; não são medições independentes do ambiente. Em um mesmo
    lote de imagens eles costumam ser idênticos, mesmo para alvos a
    distâncias diferentes.
    """

    emissivity: float | None = None
    object_distance_m: float | None = None
    atmospheric_temperature_c: float | None = None
    reflected_apparent_temperature_c: float | None = None
    relative_humidity: float | None = None


def _extract_measurement_parameters(metadata: dict | None) -> MeasurementParameters:
    """Converte o dicionário de metadados do flyr (temperaturas em K)
    em `MeasurementParameters` (temperaturas em °C)."""
    if not metadata:
        return MeasurementParameters()

    def _get(key):
        value = metadata.get(key)
        return None if value is None else float(value)

    def _kelvin_to_celsius(value):
        return None if value is None else value - 273.15

    return MeasurementParameters(
        emissivity=_get("emissivity"),
        object_distance_m=_get("object_distance"),
        atmospheric_temperature_c=_kelvin_to_celsius(_get("atmospheric_temperature")),
        reflected_apparent_temperature_c=_kelvin_to_celsius(
            _get("reflected_apparent_temperature")
        ),
        relative_humidity=_get("relative_humidity"),
    )


@dataclass
class Thermogram:
    """
    Estrutura de dados única para uma imagem térmica já carregada,
    seja ela real (via `load_rjpeg`) ou sintética (via
    `thermocore.io.synthetic`). O restante do pipeline (pré-
    processamento, segmentação, estatísticas) opera sobre esta
    estrutura e não precisa saber a origem dos dados — mas a origem
    fica sempre registrada em `source_path` / `is_synthetic`, para
    rastreabilidade.
    """

    kelvin: np.ndarray
    source_path: str
    is_synthetic: bool
    camera_model: str | None
    resolution: tuple[int, int]
    loaded_at: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat()
    )
    raw_metadata: dict = field(default_factory=dict)
    measurement: MeasurementParameters = field(default_factory=MeasurementParameters)

    @property
    def celsius(self) -> np.ndarray:
        return self.kelvin - 273.15


def validate_file(path: str | Path) -> Path:
    """
    Verificações mínimas ANTES de tentar interpretar o conteúdo do
    arquivo (não presumir o formato).

    Levanta UnsupportedFileError se:
      - o arquivo não existir;
      - a extensão não estiver entre as suportadas por este projeto
        (apenas .jpg/.jpeg RJPEG da FLIR, por ora — outros formatos
        como .seq, .csq ou exportações .csv NÃO são presumidos como
        compatíveis só porque "parecem" térmicos).
    """
    p = Path(path)
    if not p.exists():
        raise UnsupportedFileError(f"Arquivo não encontrado: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(
            f"Extensão '{p.suffix}' não suportada por este projeto ainda. "
            f"Suportadas: {sorted(SUPPORTED_EXTENSIONS)}. "
            "Se este for um formato FLIR diferente (.seq, .csq, etc.), "
            "ele precisa de um novo parser dedicado — não presuma "
            "compatibilidade."
        )
    return p


def load_rjpeg(path: str | Path) -> Thermogram:
    """
    Carrega um arquivo RJPEG real da câmera FLIR e extrai sua matriz
    de temperatura em Kelvin usando `flyr`.

    Levanta:
      - UnsupportedFileError: arquivo ausente ou extensão não suportada.
      - NoRadiometricDataError: o JPEG é válido, mas não contém dados
        radiométricos embutidos (é só uma foto colorida comum).
      - CorruptedRadiometricDataError: dados radiométricos foram
        encontrados, mas a extração falhou ou produziu uma matriz
        fisicamente implausível (variação de temperatura zero em toda
        a imagem, por exemplo — sinal comum de erro de parsing).

    Esta função NÃO lê o arquivo mais de uma vez em modo de escrita e
    NÃO grava nada no arquivo original.
    """
    p = validate_file(path)

    try:
        import flyr
    except ImportError as exc:  # pragma: no cover - ambiente sem a dependência
        raise MissingDependencyError(
            "A biblioteca 'flyr' não está instalada. Instale com "
            "'pip install flyr' para poder carregar arquivos RJPEG reais."
        ) from exc

    try:
        thermogram_flyr = flyr.unpack(str(p))
    except Exception as exc:
        # flyr levanta diferentes exceções internas a depender do que
        # deu errado; sem inspecionar caso a caso (o que exigiria
        # confiar em detalhes de implementação de uma lib externa),
        # tratamos qualquer falha de parsing como ausência de
        # radiometria confiável — nunca inventamos uma temperatura.
        raise NoRadiometricDataError(
            f"Não foi possível extrair dados radiométricos de '{p.name}'. "
            "O arquivo pode não ser um RJPEG radiométrico da FLIR, ou "
            "pode ter um formato de metadados não suportado pela "
            f"biblioteca flyr. Erro original: {exc!r}"
        ) from exc

    kelvin = thermogram_flyr.kelvin

    if kelvin is None or kelvin.size == 0:
        raise NoRadiometricDataError(
            f"'{p.name}' foi lido, mas não contém uma matriz de "
            "temperatura embutida (sem radiometria)."
        )

    # Checagem de plausibilidade física mínima — não valida a exatidão
    # da temperatura (isso exige comparação com referência externa),
    # só detecta falhas grosseiras de parsing.
    kelvin_arr = np.asarray(kelvin, dtype=np.float64)
    if np.allclose(kelvin_arr, kelvin_arr.flat[0]):
        raise CorruptedRadiometricDataError(
            f"'{p.name}': a matriz extraída é completamente uniforme "
            "(todos os pixels com o mesmo valor), o que é fisicamente "
            "implausível para uma cena real e sugere falha de parsing. "
            "Resultado descartado em vez de reportado como válido."
        )
    if kelvin_arr.min() < 173.15 or kelvin_arr.max() > 1273.15:  # -100°C a 1000°C
        raise CorruptedRadiometricDataError(
            f"'{p.name}': faixa de temperatura extraída "
            f"({kelvin_arr.min() - 273.15:.1f}°C a "
            f"{kelvin_arr.max() - 273.15:.1f}°C) está fora de uma faixa "
            "fisicamente plausível para inspeção de equipamentos "
            "elétricos. Resultado descartado."
        )

    height, width = kelvin_arr.shape[:2]

    camera_model = None
    camera_metadata = getattr(thermogram_flyr, "camera_metadata", None)
    if camera_metadata is not None:
        camera_model = getattr(camera_metadata, "model", None)

    return Thermogram(
        kelvin=kelvin_arr,
        source_path=str(p),
        is_synthetic=False,
        camera_model=camera_model,
        resolution=(width, height),
        raw_metadata={"extractor": "flyr"},
        measurement=_extract_measurement_parameters(
            getattr(thermogram_flyr, "metadata", None)
        ),
    )

