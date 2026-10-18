# thermocore

Ferramenta em Python para processamento e análise de imagens termográficas
radiométricas (RJPEG FLIR) aplicadas à inspeção de ativos elétricos: lê o
arquivo, converte em matriz de temperatura, estima a região do equipamento
(Otsu), localiza o ponto mais quente e um ponto de referência dentro do
material, calcula ΔT e η (DL/T 664-2016), classifica a severidade e gera
imagem anotada, relatório DOCX e registro de rastreabilidade.

Projeto de TCC — **Igor Pantoja** (Engenharia de Computação, UFPA, Campus
Castanhal). Versão atual: **0.2.0** (ver `CHANGELOG.md`).

> **A saída é uma triagem preliminar, não um laudo.** A referência (T2) é
> automática e η depende dela; a classe pode mudar conforme o percentil usado
> (ver "Limitações"). A "severidade" é um **critério térmico**, não uma matriz
> de risco: a ABNT NBR 15866 recomenda considerar também segurança, perda de
> produção, criticidade e tendência. Confirme sempre com um termografista.

## Estado do projeto

- **Câmera de referência: FLIR T640**, com quatro imagens reais de falhas em
  subestações cedidas por um termografista profissional
  (`tests/fixtures/`). O pipeline aceita qualquer RJPEG radiométrico FLIR
  legível pela biblioteca `flyr`, mas só a T640 foi testada.
- A temperatura máxima extraída da imagem de validação (59,8 °C) concorda
  com a legenda da câmera (59,6 °C) dentro de 0,2 °C. Nas outras imagens a
  concordância depende da cena (ver `examples/article_results.json`).
- Os parâmetros de medição lidos do arquivo (emissividade, distância, T0,
  temperatura refletida, umidade) são os **configurados na câmera**, não
  medições independentes; nas quatro imagens do acervo eles são idênticos.
- Dados sintéticos (`thermocore.io.synthetic`) são sempre marcados
  `is_synthetic=True` e nunca são resultado experimental.
- Sem aprendizado de máquina, por decisão de projeto: primeiro uma base
  determinística validada.
- Licença: **MIT** (ver `LICENSE`). Reprodutibilidade verificada em Windows 11 e Linux: a suíte de 107 testes, que fixa os valores das quatro imagens reais, passa nas duas plataformas, e `examples/reproduce_article.py` gera os mesmos números.

## Instalação

### Windows — PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Se aparecer "a execução de scripts foi desabilitada neste sistema", libere só
a sessão atual: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

### Windows — cmd.exe

```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Se `python` não for reconhecido no Windows, use `py`. Para reproduzir
exatamente o ambiente testado, use `requirements-lock.txt`.
Para sair do ambiente virtual: `deactivate`.

## Uso rápido

```
python examples/analyze_image.py caminho/para/imagem.jpg
python examples/analyze_image.py imagem.jpg --ambient 28.5     # se o arquivo não trouxer T0
python examples/analyze_image.py imagem.jpg --mta 85           # MTA do componente (NBR 15866); só gera aviso
```

Gera `examples/output/<nome>_anotada.jpg` e `<nome>_relatorio.docx` (arquivos
novos; o original nunca é tocado) e imprime ΔT, η, a **faixa de η** para
referências P1–P50 e os **avisos** da análise. Opções: `--percentile`,
`--min-delta-t`, `--erosion`, `--mta`.

Em código:

```python
from thermocore.pipeline import analyze_file, export_outputs

fa = analyze_file("imagem.jpg")          # levanta ThermocoreError em falhas previstas
r = fa.result
print(r.hotspot, r.reference, r.delta_t_celsius, r.eta_pct, r.severity)
print(r.eta_range_pct, r.severity_range, r.warnings)
export_outputs(fa, "anotada.jpg", "relatorio.docx")
```

### O que a análise faz (`thermocore.analysis.analyze_thermogram`)

1. Máscara de material: limiar de Otsu + remoção de componentes < 0,5% da
   imagem + erosão de 4 iterações (elemento em cruz). A borda do **quadro**
   não é tratada como fundo.
2. Ponto quente = máximo dentro do material; referência = pixel do material
   mais próximo do percentil P10 do material (não o mínimo, que sob limiar
   coincide com o próprio limiar).
3. ΔT = T1 − T2; η = (T1 − T2)/(T1 − T0)·100, com T0 do arquivo (ou
   `--ambient`). Sem T0 → `AmbientTemperatureUnavailableError`; nunca assume
   valor.
4. Severidade por limiares do DL/T 664-2016 (categoria "outros equipamentos
   de corrente": 35 / 80 / 95%), com salvaguardas: **ΔT mínimo** (padrão
   2,0 °C, parâmetro de engenharia não normativo — ajuste ao seu
   procedimento), **faixa de η e de classe** para P1–P50, e **avisos**
   (referência abaixo de T0, η > 100%, pixels mais quentes fora do
   material, classe instável). Se a **máxima temperatura admissível (MTA)**
   do componente for informada (`max_admissible_celsius`), há aviso quando o
   ponto quente a excede; a classe não muda.

## Reproduzindo os números e figuras do artigo

```
python examples/reproduce_article.py     # gera examples/article_results.json e examples/output/figures/
python examples/sensitivity_reference.py # comparação de métodos e sensibilidade (imagem de validação)
python examples/validate_all.py          # Tabela 7 (precisa das máscaras manuais; ver abaixo)
python -m pytest                         # inclui valores de regressão das 4 imagens reais
```

`tests/test_pipeline.py` fixa os valores esperados das quatro imagens reais;
rodar o `pytest` no Windows e no Linux verifica a reprodutibilidade entre
plataformas.

## Validação numérica da segmentação

Compara a máscara do Otsu com um contorno anotado à mão (IoU, Dice,
acurácia, precisão, revocação — `thermocore/validation/`).

1. Anote o equipamento com a ferramenta de anotação em navegador
   (`annotate_mask.html`) e exporte a máscara PNG como
   `<nome_da_imagem>_mask_manual.png` em `tests/fixtures/masks/`.
2. `python examples/validate_all.py` (as 4 imagens, nas configurações atual e
   da v0.1.0) ou `python examples/validate_segmentation.py imagem.jpg mascara.png`.

**Pendência de reprodutibilidade:** `tests/fixtures/masks/` está vazia neste
repositório e a ferramenta de anotação não está versionada (existia só como
artefato web). Para que a Tabela 7 do artigo seja reprodutível, inclua aqui as
quatro máscaras PNG e o arquivo `annotate_mask.html`.

## Desempenho

```
python benchmarks/run_benchmarks.py
```

Mede extração do RJPEG, tempo por etapa por resolução, vazão em lote, a
análise atual (`analyze_thermogram`) por resolução e sobre as quatro imagens
reais, o ponta a ponta com leitura de arquivo e o pior caso da máscara
(imagem ruidosa). Salva `benchmarks/benchmark_results.json`.
Os números valem só para a máquina em que rodou. `benchmark_results.json` traz a medição do autor com a v0.2.0 (Windows 11, 2 CPUs lógicas, 3,9 GB de RAM: ~354 ms por imagem ponta a ponta); os arquivos `reference_results_igor_windows11*.json` são medições anteriores (v0.1.0, outra configuração) e não devem ser misturados com a atual.

## Testes

```
python -m pytest -v
```

Use `python -m pytest` (não só `pytest`). A suíte cobre dados sintéticos,
carregamento e validação, estatísticas, segmentação, classificação, análise
de inspeção e suas salvaguardas, métricas de segmentação, visualização,
relatório, rastreabilidade, o pipeline ponta a ponta e regressão com as
imagens reais (testes que dependem das fixtures são pulados se elas faltarem).

Problemas comuns: `ModuleNotFoundError: thermocore` → rode na pasta raiz (a
que contém `pytest.ini`); comportamento estranho após sobrescrever arquivos →
apague `__pycache__` e `.pytest_cache`.

## Estrutura

```
thermocore/
├── exceptions.py       # erros de domínio (nunca mascarar dado não confiável)
├── pipeline.py         # arquivo → análise → registro → imagem/relatório
├── io/                 # loader.py (RJPEG via flyr), synthetic.py
├── processing/         # filtros de mediana/gaussiano (opcionais)
├── segmentation/       # threshold.py; background.py (Otsu, máscara de material)
├── analysis/           # stats.py, inspection.py, classification.py
├── visualization/      # render.py, annotate.py
├── validation/         # segmentation_metrics.py
└── report/             # provenance.py (AnalysisRecord), document.py (DOCX)
tests/  examples/  benchmarks/
```

`analysis/stats.py` também mantém `compute_stats`, `delta_t` e
`relative_temperature_rise`, para ROIs escolhidas pelo usuário; elas **não**
são usadas por `analyze_thermogram` (que trabalha com pontos).

## Limitações conhecidas

1. Validação real: **quatro imagens**, um acervo de terceiro, uma câmera.
2. A referência é automática (P10 do material), não uma ROI de termografista.
   Com o ponto quente fixo, η varia bastante com o percentil e a classe pode
   mudar (na imagem de validação: 57%–89%, de "geral" a "grave"); a análise
   informa a faixa e avisa quando a classe é instável.
3. Um ponto quente e uma referência por imagem; sem distinção entre reflexo
   especular e anomalia real. A análise avisa quando há pixels mais quentes
   fora do material, mas isso não prova reflexo.
4. O Otsu sempre devolve um limiar (a separabilidade fica registrada) e tende
   a subsegmentar partes do equipamento mais frias que o núcleo; o ponto
   quente é pouco afetado, mas a área do equipamento não é confiável.
5. O resultado depende do número de erosões (ver `sensibilidade_erosao` em
   `examples/article_results.json`): picos isolados são removidos de forma
   incidental pela erosão, não por um critério físico.
6. Limiares: DL/T 664-2016 (norma chinesa, via Xia et al., 2021), categoria
   geral; use `SeverityThresholds` para limiares por equipamento. A ABNT NBR
   15866:2010 orienta a metodologia de avaliação térmica e **não prescreve
   limiares numéricos**; os limiares aqui são uma escolha de triagem. O ΔT
   mínimo também é parâmetro de engenharia (a norma não o define).
7. Relatório apenas em DOCX (para PDF, exporte pelo Word/LibreOffice).
8. T0 e demais parâmetros são os configurados na câmera.
9. Não há correção de carga nem de vento (a NBR 15866 a recomenda para sistemas
   críticos e remete à NBR 15572) e a emissividade é única por imagem; metais de
   baixa emissividade, comuns em conexões, dificultam a avaliação quantitativa.
10. A referência automática (P10 do material) não é um "elemento similar
   adjacente" no sentido da NBR 15866; só uma ROI de termografista é.

## Próximos passos

- Comparar a saída com o laudo do termografista nas imagens do acervo.
- Medir a segmentação pelo uso real (o ponto quente cai dentro do equipamento
  anotado?) e ampliar anotações/anotadores.
- Múltiplos pontos quentes e distinção de reflexos (multi-imagem/visível).
- Limiares/MTA por tipo de componente, correção de carga e vento, emissividade
  por material e matriz de risco com criticidade (NBR 15866).
- Versionar máscaras e ferramenta de anotação.
- Otimizar `analyze_thermogram` (histograma e referências são recalculados; ~2× de ganho esperado).
