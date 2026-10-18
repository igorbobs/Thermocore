"""
Bateria de avaliação de desempenho do thermocore (tempo de processamento, uso de memória, quantidade de
imagens processadas e escalabilidade.

Só faz sentido rodar isto DEPOIS da validação funcional (testes +
resultado real já confirmados) — não antes, conforme a regra final do
projeto ("primeiro criar o método, depois executar o teste, depois
registrar o resultado, só então discutir o desempenho").

Rodar com:

    python benchmarks/run_benchmarks.py

Gera um arquivo `benchmarks/benchmark_results.json` com os números
brutos (rastreável: inclui data, versão do software e informações da
máquina onde rodou) e imprime um resumo legível no
terminal.

LIMITAÇÕES METODOLÓGICAS (documentadas explicitamente):
- Medição de memória usa `psutil` (RSS do processo — memória
  residente total do processo Python, não só da estrutura de dados
  em si). É uma medida de GRANULARIDADE GROSSA: sofre influência do
  coletor de lixo do Python, de alocações internas do NumPy/SciPy/
  Matplotlib que nem sempre são liberadas de volta ao sistema
  operacional imediatamente, e do estado geral do processo. Os
  números aqui servem para observar TENDÊNCIA de crescimento com o
  tamanho/quantidade de imagens, não como contagem exata de bytes
  usados por uma função.
- O teste de extração real (`load_rjpeg`) usa UM único arquivo de
  teste (fixture da FLIR T640) — mede a velocidade de parsing desse
  arquivo específico, não uma média representativa de arquivos FLIR
  em geral.
- Os testes de escalabilidade usam dados SINTÉTICOS (resolução e
  quantidade de imagens controladas artificialmente) — não capturam
  efeitos de I/O de disco que apareceriam ao processar um lote real
  de arquivos gravados fisicamente.
- Os números absolutos (segundos, MB) dependem inteiramente do
  hardware onde o benchmark é executado — os resultados salvos
  incluem informações da máquina exatamente para que isso fique
  registrado e comparável no futuro.
"""

import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import psutil

from thermocore._version import __version__ as THERMOCORE_VERSION
from thermocore.analysis.classification import classify_by_relative_rise
from thermocore.analysis import analyze_thermogram
from thermocore.analysis.stats import compute_stats, relative_temperature_rise
from thermocore.io.loader import load_rjpeg
from thermocore.io.synthetic import make_hotspot_thermogram
from thermocore.processing.preprocessing import denoise_gaussian
from thermocore.segmentation.threshold import (
    label_connected_regions,
    segment_above_percentile,
)

RESOLUTIONS = [
    (160, 120),
    (320, 240),
    (640, 480),
    (1280, 960),
    (1920, 1440),
]
BATCH_SIZES = [1, 5, 10, 25, 50, 100]
REAL_FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests" / "fixtures" / "IR_2497_t640_sample.jpg"
)


def _process_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _time_and_memory(fn, repeats: int) -> dict:
    """Roda `fn` (sem argumentos) `repeats` vezes e mede tempo total e
    variação de memória residente do processo antes/depois do bloco
    inteiro de repetições (não por chamada individual, para reduzir
    ruído — ver limitações no topo deste arquivo)."""
    rss_before = _process_rss_mb()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    t1 = time.perf_counter()
    rss_after = _process_rss_mb()

    total_time = t1 - t0
    return {
        "repeats": repeats,
        "total_time_seconds": total_time,
        "time_per_call_seconds": total_time / repeats,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": rss_after - rss_before,
    }


def benchmark_real_extraction() -> dict:
    if not REAL_FIXTURE_PATH.exists():
        return {"skipped": True, "reason": "fixture real não encontrada"}
    return _time_and_memory(lambda: load_rjpeg(REAL_FIXTURE_PATH), repeats=20)


def benchmark_preprocessing_by_resolution() -> list:
    results = []
    for width, height in RESOLUTIONS:
        t = make_hotspot_thermogram(height=height, width=width)
        celsius = t.celsius
        result = _time_and_memory(lambda c=celsius: denoise_gaussian(c, sigma=1.0), repeats=10)
        result["resolution"] = f"{width}x{height}"
        result["pixel_count"] = width * height
        results.append(result)
    return results


def benchmark_segmentation_by_resolution() -> list:
    results = []
    for width, height in RESOLUTIONS:
        t = make_hotspot_thermogram(height=height, width=width, hotspot_peak_celsius=90.0)
        celsius = t.celsius

        def _run(c=celsius):
            mask = segment_above_percentile(c, percentile=99.0)
            label_connected_regions(mask, min_pixel_count=4)

        result = _time_and_memory(_run, repeats=10)
        result["resolution"] = f"{width}x{height}"
        result["pixel_count"] = width * height
        results.append(result)
    return results


def benchmark_full_pipeline_by_resolution() -> list:
    results = []
    for width, height in RESOLUTIONS:
        t = make_hotspot_thermogram(height=height, width=width, hotspot_peak_celsius=90.0)
        reference = make_hotspot_thermogram(height=height, width=width, hotspot_peak_celsius=26.0)
        celsius = t.celsius
        reference_celsius = reference.celsius

        def _run(c=celsius, r=reference_celsius):
            denoised = denoise_gaussian(c, sigma=1.0)
            mask = segment_above_percentile(denoised, percentile=99.0)
            regions = label_connected_regions(mask, min_pixel_count=4)
            if regions:
                hotspot_stats = compute_stats(denoised, mask=regions[0].mask)
            else:
                hotspot_stats = compute_stats(denoised)
            reference_stats = compute_stats(r)
            eta = relative_temperature_rise(hotspot_stats, reference_stats, ambient_celsius=24.0)
            classify_by_relative_rise(eta)

        result = _time_and_memory(_run, repeats=10)
        result["resolution"] = f"{width}x{height}"
        result["pixel_count"] = width * height
        results.append(result)
    return results


def benchmark_current_analysis_by_resolution() -> list:
    """Pipeline ATUAL de análise (`analyze_thermogram`): máscara de
    material por Otsu + ponto quente + referência robusta + η +
    severidade, sobre cenas sintéticas com fundo frio."""
    from thermocore.io.synthetic import make_scene_with_background

    results = []
    for width, height in RESOLUTIONS:
        celsius = make_scene_with_background(
            height=height, width=width, material_celsius=40.0, background_celsius=5.0,
            hotspot_peak_celsius=90.0,
        ).celsius

        def _run(c=celsius):
            analyze_thermogram(c, ambient_celsius=25.0)

        result = _time_and_memory(_run, repeats=10)
        result["resolution"] = f"{width}x{height}"
        result["pixel_count"] = width * height
        results.append(result)
    return results


def benchmark_end_to_end_real() -> dict:
    """Ponta a ponta sobre o arquivo REAL: leitura do RJPEG (flyr) +
    `analyze_thermogram` com T0 lido do próprio arquivo. Mede o custo
    real por imagem quando se parte de arquivos em disco."""
    if not REAL_FIXTURE_PATH.exists():
        return {"skipped": True, "reason": "fixture real não encontrada"}

    def _run():
        t = load_rjpeg(REAL_FIXTURE_PATH)
        analyze_thermogram(t.celsius, file_ambient_celsius=t.measurement.atmospheric_temperature_c)

    result = _time_and_memory(_run, repeats=20)
    result["images_per_second"] = 1.0 / result["time_per_call_seconds"]
    return result


def benchmark_current_analysis_real_images() -> list:
    """`analyze_thermogram` (só a análise, sem ler o arquivo) sobre as 4
    imagens REAIS do acervo — cenas com fios finos têm muito mais
    componentes conexos que as cenas sintéticas lisas."""
    results = []
    fixtures = REAL_FIXTURE_PATH.parent
    for name in sorted(fixtures.glob("IR_*_t640_*.jpg")):
        t = load_rjpeg(name)
        c, t0 = t.celsius, t.measurement.atmospheric_temperature_c

        def _run(c=c, t0=t0):
            analyze_thermogram(c, file_ambient_celsius=t0)

        r = _time_and_memory(_run, repeats=10)
        r["image"] = name.name
        results.append(r)
    return results


def benchmark_material_mask_noisy() -> dict:
    """Pior caso da estimativa de material: imagem 640x480 de ruído
    gaussiano (~20 mil componentes conexos). A versão 0.1.0 levava
    ~3 s por chamada nesta imagem; a 0.2.0 vetoriza o filtro de área."""
    from thermocore.segmentation.background import estimate_material_mask

    noisy = np.random.default_rng(1).normal(30.0, 3.0, (480, 640))
    return _time_and_memory(lambda: estimate_material_mask(noisy), repeats=5)


def benchmark_batch_scalability() -> list:
    """Mede o throughput (imagens/segundo) processando lotes de N
    imagens sintéticas de resolução fixa (640x480, igual à câmera
    real usada no projeto)."""
    results = []
    for batch_size in BATCH_SIZES:
        images = [
            make_hotspot_thermogram(height=480, width=640).celsius
            for _ in range(batch_size)
        ]

        def _run(imgs=images):
            for celsius in imgs:
                denoised = denoise_gaussian(celsius, sigma=1.0)
                mask = segment_above_percentile(denoised, percentile=99.0)
                label_connected_regions(mask, min_pixel_count=4)
                compute_stats(denoised)

        result = _time_and_memory(_run, repeats=3)
        result["batch_size"] = batch_size
        result["images_per_second"] = batch_size / (
            result["total_time_seconds"] / result["repeats"]
        )
        results.append(result)
    return results


def _environment_metadata() -> dict:
    return {
        "thermocore_version": THERMOCORE_VERSION,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor() or "desconhecido",
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_memory_mb": psutil.virtual_memory().total / (1024 * 1024),
    }


def _print_summary(results: dict) -> None:
    env = results["environment"]
    print("=" * 70)
    print("thermocore — bateria de desempenho")
    print("=" * 70)
    print(f"Versão: {env['thermocore_version']}  |  Python: {env['python_version'].split()[0]}")
    print(f"Plataforma: {env['platform']}")
    print(f"CPUs lógicas: {env['cpu_count_logical']}  |  RAM total: {env['total_memory_mb']:.0f} MB")
    print()

    real = results["real_extraction"]
    if real.get("skipped"):
        print(f"Extração real (load_rjpeg): PULADO ({real['reason']})")
    else:
        print(
            f"Extração real (load_rjpeg), {real['repeats']}x: "
            f"{real['time_per_call_seconds']*1000:.1f} ms/chamada "
            f"| ΔRSS: {real['rss_delta_mb']:+.1f} MB"
        )
    print()

    print("Pré-processamento (denoise_gaussian) por resolução:")
    for r in results["preprocessing_by_resolution"]:
        print(
            f"  {r['resolution']:>12} ({r['pixel_count']:>9,} px): "
            f"{r['time_per_call_seconds']*1000:8.2f} ms/img"
        )
    print()

    print("Segmentação (limiar + rotulagem) por resolução:")
    for r in results["segmentation_by_resolution"]:
        print(
            f"  {r['resolution']:>12} ({r['pixel_count']:>9,} px): "
            f"{r['time_per_call_seconds']*1000:8.2f} ms/img"
        )
    print()

    print("Pipeline completo por resolução:")
    for r in results["full_pipeline_by_resolution"]:
        print(
            f"  {r['resolution']:>12} ({r['pixel_count']:>9,} px): "
            f"{r['time_per_call_seconds']*1000:8.2f} ms/img"
        )
    print()

    print("Escalabilidade por tamanho de lote (640x480, imagens/segundo):")
    for r in results["batch_scalability"]:
        print(
            f"  N={r['batch_size']:>4}: {r['images_per_second']:7.1f} img/s "
            f"| ΔRSS: {r['rss_delta_mb']:+.1f} MB"
        )
    print()
    print("Análise ATUAL (material por Otsu + η + severidade) por resolução:")
    for r in results["current_analysis_by_resolution"]:
        print(
            f"  {r['resolution']:>12} ({r['pixel_count']:>9,} px): "
            f"{r['time_per_call_seconds']*1000:8.2f} ms/img"
        )
    print()
    e2e = results["end_to_end_real"]
    if e2e.get("skipped"):
        print(f"Ponta a ponta com arquivo real: PULADO ({e2e['reason']})")
    else:
        print(
            f"Ponta a ponta (leitura RJPEG + análise), arquivo real, {e2e['repeats']}x: "
            f"{e2e['time_per_call_seconds']*1000:.1f} ms/img ({e2e['images_per_second']:.1f} img/s)"
        )
    print()
    print("Análise atual sobre as imagens REAIS (sem leitura de arquivo):")
    for r in results["current_analysis_real_images"]:
        print(f"  {r['image']:<32} {r['time_per_call_seconds']*1000:8.2f} ms/img")
    noisy = results["material_mask_noisy"]
    print(f"Máscara de material, imagem ruidosa 640x480: {noisy['time_per_call_seconds']*1000:.1f} ms")
    print()
    print("Resultados completos salvos em benchmarks/benchmark_results.json")


def main() -> None:
    results = {
        "environment": _environment_metadata(),
        "real_extraction": benchmark_real_extraction(),
        "preprocessing_by_resolution": benchmark_preprocessing_by_resolution(),
        "segmentation_by_resolution": benchmark_segmentation_by_resolution(),
        "full_pipeline_by_resolution": benchmark_full_pipeline_by_resolution(),
        "batch_scalability": benchmark_batch_scalability(),
        "current_analysis_by_resolution": benchmark_current_analysis_by_resolution(),
        "end_to_end_real": benchmark_end_to_end_real(),
        "current_analysis_real_images": benchmark_current_analysis_real_images(),
        "material_mask_noisy": benchmark_material_mask_noisy(),
    }

    output_path = Path(__file__).resolve().parent / "benchmark_results.json"
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _print_summary(results)


if __name__ == "__main__":
    main()
