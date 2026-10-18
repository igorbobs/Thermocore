from thermocore.io.synthetic import make_hotspot_thermogram
from thermocore.segmentation.threshold import (
    label_connected_regions,
    segment_above_percentile,
    segment_above_threshold,
)


def test_segment_above_threshold_isolates_hotspot():
    t = make_hotspot_thermogram(
        height=100, width=100, base_celsius=25.0, hotspot_peak_celsius=90.0
    )
    mask = segment_above_threshold(t.celsius, threshold_celsius=60.0)
    # O centro (ponto quente) deve estar na máscara; um canto distante não.
    assert mask[50, 50] == True  # noqa: E712
    assert mask[0, 0] == False  # noqa: E712


def test_segment_above_percentile_returns_nonempty_mask_for_hotspot():
    t = make_hotspot_thermogram(base_celsius=25.0, hotspot_peak_celsius=90.0)
    mask = segment_above_percentile(t.celsius, percentile=99.0)
    assert mask.sum() > 0


def test_label_connected_regions_finds_single_hotspot():
    t = make_hotspot_thermogram(
        height=100,
        width=100,
        base_celsius=25.0,
        hotspot_center=(50, 50),
        hotspot_radius=5,
        hotspot_peak_celsius=90.0,
    )
    mask = segment_above_threshold(t.celsius, threshold_celsius=60.0)
    regions = label_connected_regions(mask, min_pixel_count=4)

    assert len(regions) == 1
    region = regions[0]
    # O centroide deve estar próximo do centro real do hotspot.
    assert abs(region.centroid[0] - 50) < 3
    assert abs(region.centroid[1] - 50) < 3


def test_label_connected_regions_filters_out_small_noise():
    t = make_hotspot_thermogram(base_celsius=25.0, hotspot_peak_celsius=90.0)
    mask = segment_above_threshold(t.celsius, threshold_celsius=60.0)
    # min_pixel_count absurdamente alto deve eliminar até o hotspot real.
    regions = label_connected_regions(mask, min_pixel_count=10_000)
    assert regions == []
