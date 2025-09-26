# Import the metric function we want to test
from src.metrics.performance_claims import metric
# Import the specific error the API might raise
from huggingface_hub.utils import HfHubHTTPError

# This is a "fake" or "mock" class that pretends to be the result from the real API.
# We use it to control the test conditions.
class FakeModelInfo:
    def __init__(self, downloads):
        self.downloads = downloads

# This is the first test case. Pytest knows it's a test because it starts with "test_".
# The 'mocker' argument is a special tool from the pytest-mock library.
def test_performance_claims_high_downloads(mocker):
    """Test that a model with many downloads gets a high score of 1.0."""

    # We tell the mocker to replace the real 'model_info' function.
    # Instead of making an internet call, it will now just return our FakeModelInfo object.
    mocker.patch('src.metrics.performance_claims.model_info', return_value=FakeModelInfo(5_000_000))

    # This is the same input our metric function expects
    resource = {"name": "google/gemma-2b"}
    # We call the metric function, which is now using our faked API response.
    score, latency = metric(resource)

    # We check if the result is what we expect.
    assert score == 1.0
    assert latency >= 0

# This is the second test case, for when the API call fails.
def test_performance_claims_api_error(mocker):
    """Test that the metric returns 0.0 if the model is not found on the Hub."""

    # This time, we tell the mocker to raise an error when 'model_info' is called.
    # This simulates what happens when a model isn't found (a 404 error).
    mocker.patch('src.metrics.performance_claims.model_info', side_effect=HfHubHTTPError("Not Found"))

    resource = {"name": "not/a-real-model"}
    score, latency = metric(resource)

    # We check that the score is 0.0 when the API call fails, as it should.
    assert score == 0.0