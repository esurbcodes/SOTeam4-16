# tests/unit/test_treescore.py
import pytest
from unittest.mock import patch, MagicMock

import src.metrics.treescore as treescore

def fake_metric_factory(return_score):
    def metric(resource):
        return float(return_score), 1
    return metric

@patch("src.metrics.treescore._download_config_json_via_hf")
@patch("src.metrics.treescore._load_other_metrics")
def test_no_config_returns_self_score(mock_load_metrics, mock_download):
    # No config.json -> should compute parent's own net score via other metrics
    mock_download.return_value = None

    # Mock metric functions that return fixed scores
    mock_load_metrics.return_value = {
        "a": lambda resource: (0.9, 1),
        "b": lambda resource: (0.7, 1),
    }

    res = {"name": "owner/model", "url": "https://huggingface.co/owner/model"}
    score, lat = treescore.metric(res)

    # Expected parent's net score = average(0.9, 0.7) = 0.8
    assert pytest.approx(score, rel=1e-3) == 0.8
    assert isinstance(lat, int) and lat >= 0


@patch("src.metrics.treescore._download_config_json_via_hf")
@patch("src.metrics.treescore._compute_parent_net_score")
def test_parents_in_config_use_parent_scores(mock_compute_parent, mock_download):
    # Suppose config.json lists two parents
    mock_download.return_value = {"parent_model": ["p1", "p2"]}  # or "parents": [...]
    # parent net-scores:
    mock_compute_parent.side_effect = lambda p, cache: {"p1": 0.5, "p2": 1.0}.get(p, 0.0)
    res = {"name": "child/model", "url": "https://huggingface.co/child/model"}
    score, lat = treescore.metric(res)
    assert pytest.approx(score, rel=1e-3) == 0.75  # average(0.5, 1.0)
    assert isinstance(lat, int)

@patch("src.metrics.treescore._download_config_json_via_hf")
@patch("src.metrics.treescore._compute_parent_net_score")
def test_cycle_detection(mock_compute_parent, mock_download):
    # config shows parent which references back to child; compute_parent returns 0 for cycle
    # We simulate by making compute_parent return 0 for 'child' to show fallback
    mock_download.return_value = {"parents": ["child/model"]}
    mock_compute_parent.return_value = 0.0
    res = {"name": "child/model"}
    score, lat = treescore.metric(res)
    assert score == 0.0
