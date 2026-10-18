import json

from thermocore import __version__ as THERMOCORE_VERSION
from thermocore.report.provenance import AnalysisRecord


def test_analysis_record_captures_software_version():
    record = AnalysisRecord(
        source_path="synthetic-hotspot",
        is_synthetic=True,
        method="threshold+eta_DLT664",
        parameters={"threshold_celsius": 60.0},
        results={"eta_pct": 42.0},
    )
    assert record.software_version == THERMOCORE_VERSION


def test_analysis_record_marks_synthetic_data_explicitly():
    record = AnalysisRecord(
        source_path="synthetic-hotspot",
        is_synthetic=True,
        method="threshold+eta_DLT664",
        parameters={},
        results={},
    )
    assert record.as_dict()["is_synthetic"] is True


def test_save_json_round_trips_all_fields(tmp_path):
    record = AnalysisRecord(
        source_path="/caminho/para/imagem.jpg",
        is_synthetic=False,
        method="threshold+eta_DLT664",
        parameters={"threshold_celsius": 60.0, "min_pixel_count": 4},
        results={"eta_pct": 87.3, "severity": "grave"},
    )
    out_path = tmp_path / "analysis_record.json"
    record.save_json(out_path)

    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["source_path"] == "/caminho/para/imagem.jpg"
    assert loaded["is_synthetic"] is False
    assert loaded["results"]["eta_pct"] == 87.3
    assert loaded["software_version"] == THERMOCORE_VERSION
    assert "analyzed_at" in loaded
