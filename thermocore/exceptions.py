"""
Exceções do domínio.

Uma exceção específica (em vez de um valor sentinela como None ou NaN
silencioso) é usada sempre que o sistema não pode garantir a
confiabilidade de um resultado térmico: o chamador é forçado a lidar
com a ausência de dados confiáveis, em vez de receber um número
plausível, porém inventado, e nenhum resultado estimado é apresentado
como experimental.
"""


class ThermocoreError(Exception):
    """Classe-base para todas as exceções do pacote."""


class UnsupportedFileError(ThermocoreError):
    """
    Levantada quando o arquivo de entrada não pôde ser reconhecido como
    um formato suportado, ou quando sua extensão/estrutura não bate com
    o que foi declarado.
    """


class NoRadiometricDataError(ThermocoreError):
    """
    Levantada quando o arquivo é uma imagem térmica válida, mas NÃO
    contém dados radiométricos embutidos (ou seja, é apenas uma imagem
    colorida/visual, sem matriz de temperatura por pixel).

    O sistema nunca segue adiante calculando estatísticas térmicas a
    partir de uma imagem sem radiometria.
    """


class CorruptedRadiometricDataError(ThermocoreError):
    """
    Levantada quando dados radiométricos foram encontrados, mas a
    extração falhou ou produziu valores fisicamente implausíveis
    (por exemplo, todos os pixels com a mesma temperatura, ou uma
    faixa de temperatura absurda), impedindo o cálculo confiável.
    """


class MissingDependencyError(ThermocoreError):
    """
    Levantada quando uma dependência externa necessária (por exemplo,
    a biblioteca `flyr`) não está instalada no ambiente.
    """


class EmptyRegionError(ThermocoreError):
    """
    Levantada quando uma região de interesse (ROI) solicitada para
    análise não contém nenhum pixel (por exemplo, máscara vazia).
    """


class AmbientTemperatureUnavailableError(ThermocoreError, ValueError):
    """
    Levantada quando a temperatura de referência T0 necessária ao cálculo
    de η não está nos metadados do arquivo e não foi informada pelo
    chamador. O sistema não assume nenhum valor. Herda também de
    `ValueError` para manter compatibilidade com código da versão 0.1.0.
    """


class HotspotBelowAmbientError(ThermocoreError, ValueError):
    """
    Levantada quando o ponto mais quente do material não excede T0, caso
    em que η = (T1 − T2)/(T1 − T0) não está definido. Herda também de
    `ValueError` por compatibilidade com a versão 0.1.0.
    """
