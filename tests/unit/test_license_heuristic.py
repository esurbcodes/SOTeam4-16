# tests/unit/test_license_heuristic.py
from pathlib import Path

#update metric import route depending on your file structure. should be phase1 though.
from src.metrics.license import metric

def test_heurstic_mit(tmp_path: Path):
    p = tmp_path / "LICENSE"
    p.write_text("MIT License\nCopyright (c) ...", encoding="utf-8")
    score, lat = metric({"local_dir": str(tmp_path)})
    assert score == 1.0
    assert isinstance(lat, int)

