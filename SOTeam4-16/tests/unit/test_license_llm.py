# tests/unit/test_license_score_llm.py
import json
from unittest.mock import patch, MagicMock
from src.metrics.license import metric

FAKE_RESPONSE_BODY = {
    "id": "llama3.1:example",
    "choices": [
        {"message": {"content": '{"license_spdx":"MIT","category":"permissive","compatibility_score":0.98,"compatibility_with_commercial_use":true,"explanation":"MIT is permissive."}'}}
    ]
}

class FakeResp:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body or {}
    def json(self):
        return self._body
    @property
    def text(self):
        return json.dumps(self._body)

@patch("phase1.src.metrics.license.requests.post")
def test_llm_integration(mock_post, tmp_path):
    # prepare a license file
    p = tmp_path / "LICENSE"
    p.write_text("MIT License", encoding="utf-8")
    mock_post.return_value = FakeResp(status_code=200, body=FAKE_RESPONSE_BODY)
    # ensure env var set for test
    import os
    os.environ["PURDUE_GENAI_API_KEY"] = "fake-key"
    score, lat = metric({"local_dir": str(tmp_path)})
    assert 0.95 <= score <= 1.0

