"""
Geração de matrizes térmicas sintéticas para uso em testes automatizados.

Enquanto não houver um conjunto amplo de imagens reais da câmera
FLIR T640 (câmera de referência do projeto — ver README.md), os
testes do pipeline usam dados gerados aqui. Toda
estrutura de dados produzida por este módulo carrega o campo
`is_synthetic=True` e a origem "synthetic-generator", exatamente para
que nenhum resultado calculado a partir dela possa, por engano, ser
reportado como um resultado experimental (diretriz do projeto: dado
sintético nunca é resultado experimental).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SyntheticThermogram:
    """
    Representa uma matriz térmica sintética, com o mesmo formato de
    dados que o restante do pipeline espera de uma leitura real
    (ver `thermocore.io.loader.Thermogram`), mas explicitamente
    marcada como não real.
    """

    kelvin: np.ndarray
    source_id: str
    is_synthetic: bool = True
    generation_params: dict = field(default_factory=dict)

    @property
    def celsius(self) -> np.ndarray:
        return self.kelvin - 273.15


def make_flat_thermogram(
    height: int = 120,
    width: int = 160,
    base_celsius: float = 25.0,
    source_id: str = "synthetic-flat",
) -> SyntheticThermogram:
    """
    Gera uma matriz térmica sintética totalmente uniforme (sem pontos
    quentes). Útil para testar se as estatísticas básicas (Tmáx, Tmín,
    Tmédia, desvio padrão) se comportam corretamente no caso trivial:
    Tmáx == Tmín == Tmédia, desvio padrão == 0.
    """
    kelvin = np.full((height, width), base_celsius + 273.15, dtype=np.float64)
    return SyntheticThermogram(
        kelvin=kelvin,
        source_id=source_id,
        generation_params={
            "height": height,
            "width": width,
            "base_celsius": base_celsius,
            "kind": "flat",
        },
    )


def make_scene_with_background(
    height: int = 120,
    width: int = 160,
    material_celsius: float = 30.0,
    background_celsius: float = 10.0,
    background_edge: str = "top",
    background_fraction: float = 0.25,
    hotspot_center: tuple[int, int] | None = None,
    hotspot_radius: int = 8,
    hotspot_peak_celsius: float = 90.0,
    source_id: str = "synthetic-scene-with-background",
) -> SyntheticThermogram:
    """
    Gera uma cena sintética com três componentes, para testar a
    heurística de `thermocore.segmentation.background`:
      1. um "material" (equipamento) de fundo, em `material_celsius`,
         preenchendo toda a imagem;
      2. um bloco retangular de "fundo de cena" (céu), em
         `background_celsius`, ocupando `background_fraction` da
         imagem a partir da borda `background_edge`
         ("top"/"bottom"/"left"/"right");
      3. um ponto quente gaussiano (o mesmo perfil de
         `make_hotspot_thermogram`), posicionado por padrão fora do
         bloco de fundo (no centro da região de material).

    Como em `make_hotspot_thermogram`, todos os parâmetros de geração
    ficam registrados em `generation_params` para reprodutibilidade.
    """
    if background_edge not in ("top", "bottom", "left", "right"):
        raise ValueError('background_edge deve ser "top", "bottom", "left" ou "right".')
    if not (0.0 < background_fraction < 1.0):
        raise ValueError("background_fraction deve estar entre 0 e 1.")

    celsius = np.full((height, width), material_celsius, dtype=np.float64)

    if background_edge in ("top", "bottom"):
        n_rows = max(1, int(round(height * background_fraction)))
        if background_edge == "top":
            celsius[:n_rows, :] = background_celsius
        else:
            celsius[height - n_rows :, :] = background_celsius
    else:
        n_cols = max(1, int(round(width * background_fraction)))
        if background_edge == "left":
            celsius[:, :n_cols] = background_celsius
        else:
            celsius[:, width - n_cols :] = background_celsius

    if hotspot_center is None:
        # Centro da região de material, tentando evitar o bloco de fundo.
        hotspot_center = (height // 2, width // 2)

    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = hotspot_center
    dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
    gaussian = np.exp(-dist_sq / (2.0 * hotspot_radius**2))
    is_background = np.isclose(celsius, background_celsius)
    # O ponto quente só se aplica sobre o material, não sobre o fundo.
    celsius = np.where(
        is_background, celsius, celsius + gaussian * (hotspot_peak_celsius - material_celsius)
    )

    kelvin = celsius + 273.15
    return SyntheticThermogram(
        kelvin=kelvin,
        source_id=source_id,
        generation_params={
            "height": height,
            "width": width,
            "material_celsius": material_celsius,
            "background_celsius": background_celsius,
            "background_edge": background_edge,
            "background_fraction": background_fraction,
            "hotspot_center": hotspot_center,
            "hotspot_radius": hotspot_radius,
            "hotspot_peak_celsius": hotspot_peak_celsius,
            "kind": "scene_with_background",
        },
    )


def make_hotspot_thermogram(
    height: int = 120,
    width: int = 160,
    base_celsius: float = 25.0,
    hotspot_center: tuple[int, int] | None = None,
    hotspot_radius: int = 8,
    hotspot_peak_celsius: float = 85.0,
    noise_std_celsius: float = 0.0,
    seed: int | None = 42,
    source_id: str = "synthetic-hotspot",
) -> SyntheticThermogram:
    """
    Gera uma matriz térmica sintética com um único ponto quente
    circular de perfil gaussiano sobre um fundo uniforme, simulando
    (de forma simplificada e sem qualquer pretensão de realismo físico)
    uma falha de conexão elétrica.

    Parâmetros
    ----------
    hotspot_center: posição (linha, coluna) do centro do ponto quente.
        Se None, usa o centro da imagem.
    hotspot_radius: "raio" característico (desvio padrão) do perfil
        gaussiano do ponto quente, em pixels.
    hotspot_peak_celsius: temperatura no centro do ponto quente.
    noise_std_celsius: desvio padrão de ruído gaussiano aditivo,
        aplicado a toda a imagem (0 = sem ruído).
    seed: semente do gerador aleatório, para reprodutibilidade do
        ruído. Ignorado se noise_std_celsius == 0.

    Retorna um SyntheticThermogram cujo `generation_params` registra
    todos os parâmetros acima, para que o teste que o consumir seja
    plenamente reprodutível e rastreável.
    """
    if hotspot_center is None:
        hotspot_center = (height // 2, width // 2)

    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = hotspot_center
    dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
    gaussian = np.exp(-dist_sq / (2.0 * hotspot_radius**2))

    celsius = base_celsius + gaussian * (hotspot_peak_celsius - base_celsius)

    if noise_std_celsius > 0:
        rng = np.random.default_rng(seed)
        celsius = celsius + rng.normal(0.0, noise_std_celsius, size=celsius.shape)

    kelvin = celsius + 273.15

    return SyntheticThermogram(
        kelvin=kelvin,
        source_id=source_id,
        generation_params={
            "height": height,
            "width": width,
            "base_celsius": base_celsius,
            "hotspot_center": hotspot_center,
            "hotspot_radius": hotspot_radius,
            "hotspot_peak_celsius": hotspot_peak_celsius,
            "noise_std_celsius": noise_std_celsius,
            "seed": seed,
            "kind": "hotspot",
        },
    )
