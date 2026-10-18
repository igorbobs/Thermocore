# Changelog

## 0.2.0

Correções
- `analyze_thermogram` agora levanta `AmbientTemperatureUnavailableError` e
  `HotspotBelowAmbientError` (subclasses de `ThermocoreError` e de
  `ValueError`); antes, a CLI não capturava o `ValueError` e a dica de
  `--ambient` nunca aparecia.
- Erosão da máscara de material preserva a borda do quadro
  (`preserve_image_border=True`); a v0.1.0 tratava-a como fundo e removia
  uma faixa de 4 px de equipamento que preenchia o enquadramento.
  `preserve_image_border=False` reproduz a v0.1.0.
- Filtro de área de componentes vetorizado: ~280× mais rápido em imagem
  ruidosa (≈20 mil componentes), com resultado idêntico.
- O relatório DOCX não afirma mais que a referência é "o pixel mais frio de
  toda a imagem".
- Rótulos de P1/P2 não se sobrepõem quando os pontos estão próximos.

Novidades
- Salvaguardas na análise: ΔT mínimo configurável, faixa de η e de classe
  (P1–P50), avisos (referência < T0, η > 100%, pixels mais quentes fora do
  material, classe instável), separabilidade do Otsu, parâmetros registrados.
- `thermocore.pipeline` (`analyze_file`, `build_record`, `export_outputs`)
  unifica CLI, exemplos e testes; a origem gravada é relativa/nome de
  arquivo (sem caminhos absolutos).
- `SeverityThresholds` valida a ordem dos limiares; η NaN é rejeitado.
- Filtro de mediana opcional (`denoise_median_size`) antes da análise.
- Máxima temperatura admissível opcional (`max_admissible_celsius`, `--mta`): só
  emite aviso (NBR 15866); a classe não muda.
- Relatório e README deixam claro que a "severidade" é critério térmico, não
  matriz de risco, e listam a ausência de correção de carga/vento.
- `examples/reproduce_article.py` e `examples/validate_all.py`.
- Benchmarks: análise atual sobre imagens reais e pior caso ruidoso.
- 31 testes novos (107 no total), incluindo regressão com as 4 imagens reais.

Documentação
- README reescrito (sem seções duplicadas e sem afirmações desatualizadas);
  referências internas a "Regra N" removidas do código.
