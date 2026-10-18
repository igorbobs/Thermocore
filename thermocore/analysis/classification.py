"""
Classificação preliminar de severidade de anomalia térmica.

Implementação determinística, baseada inteiramente em limiares
normativos (norma DL/T 664-2016, citada por Xia et al., 2021, Tabela
3), e não em um modelo treinado, preferindo
um método simples e explicável a uma caixa-preta. Uso de aprendizado de máquina fica reservado para uma
fase posterior do projeto, quando houver dados reais suficientes para
validá-lo contra este critério de referência.

Os limiares abaixo são os "gerais" da norma citada; por equipamento a
norma prevê limiares específicos (ver Tabela 3 do artigo de Xia et
al., 2021) — este módulo aceita limiares customizados por chamada
justamente para não fixar um único conjunto de limiares como universal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AnomalySeverity(str, Enum):
    NORMAL = "normal"
    GENERAL = "geral"          # falha leve
    MAJOR = "grave"            # falha grave
    EMERGENCY = "emergencial"  # falha emergencial


@dataclass(frozen=True)
class SeverityThresholds:
    """
    Limiares de η (%) para cada nível de severidade. Os valores padrão
    correspondem à linha "Outros equipamentos de corrente" da Tabela 3
    de Xia et al. (2021), citando a norma DL/T 664-2016 — devem ser
    substituídos pelos limiares específicos do equipamento sob
    inspeção sempre que a norma os definir separadamente.
    """

    general_pct: float = 35.0
    major_pct: float = 80.0
    emergency_pct: float = 95.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.general_pct < self.major_pct < self.emergency_pct):
            raise ValueError(
                "Limiares inválidos: é preciso 0 <= geral < grave < emergencial "
                f"(recebido {self.general_pct}, {self.major_pct}, {self.emergency_pct})."
            )


def classify_by_relative_rise(
    eta_pct: float,
    thresholds: SeverityThresholds = SeverityThresholds(),
) -> AnomalySeverity:
    """
    Classifica a severidade a partir da diferença relativa de
    temperatura η (em %), conforme os limiares fornecidos.

    Esta função é puramente determinística: dado o mesmo η e os
    mesmos limiares, sempre retorna o mesmo resultado, e a razão da
    classificação pode ser lida diretamente do código — não há
    parâmetro aprendido nem caixa-preta envolvida.
    """
    if eta_pct != eta_pct:  # NaN
        raise ValueError("η é NaN: a severidade não pode ser classificada.")
    if eta_pct >= thresholds.emergency_pct:
        return AnomalySeverity.EMERGENCY
    if eta_pct >= thresholds.major_pct:
        return AnomalySeverity.MAJOR
    if eta_pct >= thresholds.general_pct:
        return AnomalySeverity.GENERAL
    return AnomalySeverity.NORMAL
