"""
Registro de rastreabilidade de uma análise.

Cada vez que uma imagem é processada de ponta a ponta, um
`AnalysisRecord` deve ser criado e salvo junto do resultado. Ele é a
peça que permite, meses depois, responder com precisão: "de qual
arquivo veio este número, com quais parâmetros, usando qual método, em
que versão do software?" — pergunta que qualquer metodologia de TCC
precisa conseguir responder.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path


def _read_software_version() -> str:
    """
    Lê a versão do pacote sob demanda (só quando um `AnalysisRecord` é
    de fato criado), não no momento em que este módulo é carregado.
    Isso evita qualquer dependência da ordem em que os módulos do
    pacote são importados — o problema real por trás de um bug de
    importação circular já corrigido neste projeto.
    """
    from thermocore._version import __version__

    return __version__


@dataclass
class AnalysisRecord:
    """
    Registro rastreável de uma análise térmica completa.

    Campos
    ------
    source_path: caminho do arquivo de origem (ou o `source_id`
        sintético, se `is_synthetic=True`) — NUNCA vazio.
    is_synthetic: sinaliza explicitamente se a imagem de origem é
        sintética, para que este registro nunca seja confundido com um
        resultado experimental real (regra final do projeto).
    analyzed_at: timestamp UTC ISO-8601 de quando a análise foi
        executada (não confundir com a data de captura da imagem,
        que vem dos metadados EXIF quando disponíveis).
    method: nome curto do método usado (ex.: "threshold+eta_DLT664").
    parameters: todos os parâmetros que afetam o resultado (limiares,
        tamanho de filtro, etc.) — o suficiente para reproduzir a
        análise exatamente.
    results: os números/valores produzidos pela análise.
    software_version: versão do pacote thermocore usada.
    """

    source_path: str
    is_synthetic: bool
    method: str
    parameters: dict
    results: dict
    analyzed_at: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat()
    )
    software_version: str = field(default_factory=_read_software_version)

    def as_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "is_synthetic": self.is_synthetic,
            "analyzed_at": self.analyzed_at,
            "method": self.method,
            "parameters": self.parameters,
            "results": self.results,
            "software_version": self.software_version,
        }

    def save_json(self, path: str | Path) -> None:
        """
        Salva o registro como JSON legível por humanos e por
        ferramentas. Este é um ARQUIVO NOVO derivado da análise — a
        imagem de origem nunca é tocada.
        """
        Path(path).write_text(
            json.dumps(self.as_dict(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
