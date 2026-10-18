"""
thermocore
==========

Ferramenta computacional em Python para processamento e análise de
imagens termográficas radiométricas (RJPEG) aplicadas à inspeção de
ativos elétricos.

Projeto de TCC — Igor Pantoja (Engenharia de Computação, UFPA Campus
Castanhal).

Diretrizes de engenharia do projeto:
  - não presumir o formato do arquivo de entrada;
  - começar simples (carregar -> extrair -> estatísticas -> segmentar ->
    analisar -> relatório) antes de qualquer ML/DL;
  - nunca modificar os arquivos originais;
  - todo resultado é rastreável (origem, data, parâmetros, método,
    versão do software);
  - nunca inventar uma temperatura que não possa ser calculada com
    confiabilidade;
  - preferir métodos simples e explicáveis a modelos de caixa-preta,
    enquanto um método simples produzir resultado equivalente.

Consulte o README.md do repositório para a descrição completa do
pipeline e das limitações atuais.
"""

from thermocore._version import __version__  # noqa: E402,F401
